"""exp_he19_step_loop - H-E19's write-back read: arithmetic solved one VM-verified step at a time, the state
written back as $S slots (cubbyllm/reasoning/step_loop.py), the whole program never in one pass.

Reads the arithmetic records of a slot-form jsonl (the raw `question` and `gold`; the loop annotates the question
itself, words and the arithmetic world's constants), runs `StepLoop` around the emitter, and scores the answer
against the gold with the harvest's rule (`gold_matches`). Reported: correct, refused (by reason), ran past the
cap, by the reference's step count, steps taken against the reference's, and the emissions per problem.

    --export/--adapter/--tokenizer   the 450M adapter on grilly2 (standin.emitter.Cubby450mEmitter)
    --gguf                           the stand-in (llama.cpp; its prompts are raw, so this is a format check only)
    --replay-gold STEP_FILE          no model: the gold step rows replayed through the loop and the REAL VM
                                     (the round trip of wrap_step and the state slots through cubelang)
    --teacher-forced STEP_FILE       the per-step read in the step format: every gold step row's prompt (the gold
                                     state written back), one emission, scored by the VALUE the step computes and
                                     the stop -- the same read as exp_he19_next_step.py's, so the two compare

    python validation/exp_he19_step_loop.py --data standin/data/out/pf_heldout_eval_w_slots.jsonl --tag gold_held \\
        --replay-gold standin/data/out/pf_heldout_eval_w_step.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cubbyllm.reasoning.arith_world import ArithmeticWorld  # noqa: E402
from cubbyllm.reasoning.step_loop import StepLoop, blocks_of  # noqa: E402


def load(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == "arithmetic" and r.get("split", "val") == "val" and "#" not in r["id"]:
            out.append(r)
    return out


class GoldReplay:
    """The gold step rows of one record, in order; what the loop would see from a perfect emitter."""

    def __init__(self, rows_by_record: dict):
        self.rows, self.rid, self.i = rows_by_record, None, 0
        self.name = "gold-replay"

    def start(self, rid: str):
        self.rid, self.i = rid, 0

    def emit(self, prompt: str, max_new_tokens: int = 64, **kw) -> str:
        rows = self.rows.get(self.rid, [])
        out = rows[self.i]["program"] if self.i < len(rows) else ""
        self.i += 1
        return out


def teacher_forced(a) -> None:
    """Each gold step row: emit once on its prompt, score the value the emitted block computes over the row's
    slot values (N, K and the written-back S) against the row's gold, or the stop against `return $S<K>;`."""
    from fractions import Fraction
    from cubbyllm.reasoning.slots import num_value
    from cubbyllm.reasoning.step_loop import parse_emission, simulate
    rows = [json.loads(l) for l in open(a.teacher_forced, encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "arithmetic" and "#" in r["id"] and r.get("split", "val") == "val"]
    if a.limit:
        keep = sorted({r["id"].split("#")[0] for r in rows})[:a.limit]
        rows = [r for r in rows if r["id"].split("#")[0] in keep]
    if a.gguf:
        from standin.emitter import LlamaCppEmitter
        emitter = LlamaCppEmitter(a.gguf)
    else:
        from standin.emitter import Cubby450mEmitter
        emitter = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    out_path = os.path.join(HERE, "logs", f"exp_he19_step_loop_{a.tag}.json")
    by = defaultdict(Counter)
    res, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        k = r["stats"]["k"]
        is_stop = r["program"].lstrip().startswith("return")
        text = emitter.emit(r["prompt"], max_new_tokens=a.max_new)
        kind, what = parse_emission(text)
        if is_stop:
            ok = kind == "stop" and what == r["program"].split()[1].rstrip(";")
            key = "stop"
        else:
            values = {}
            for sp in r["spans"]:
                v = num_value(sp.get("value") if sp.get("value") is not None else sp.get("text"))
                if v is not None:
                    values[sp["id"]] = Fraction(str(v))
            got = simulate(what.ops, values) if kind == "step" else None
            ok = got is not None and abs(float(got) - float(r["gold"])) < 1e-6
            key = "step0" if k == 0 else "step1+"
            by[f"k={k}"]["n"] += 1
            by[f"k={k}"]["ok"] += ok
            by["ran_into_stop" if kind == "stop" else "_"]["n"] += 0
        by[key]["n"] += 1
        by[key]["ok"] += ok
        if not is_stop and kind == "stop":
            by["stopped_early"]["n"] += 1
        res.append({"id": r["id"], "k": k, "stop": is_stop, "kind": kind, "ok": ok, "gold": r["gold"], "text": text[:300]})
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            line = "  ".join(f"{k2}={c['ok'] / c['n']:.3f}/{c['n']}" for k2, c in sorted(by.items()) if k2 in ("step0", "step1+", "stop") and c["n"])
            print(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)  {line}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"emitter": getattr(emitter, "name", "?"), "data": a.teacher_forced,
                           "summary": {k2: {"n": c["n"], "ok": c["ok"] / c["n"] if c["n"] else 0} for k2, c in by.items()},
                           "rows": res}, f, indent=1)
    print("\nteacher-forced, step format:")
    for k2 in ["step0", "step1+", "stop"] + sorted((x for x in by if x.startswith("k=")), key=lambda x: int(x[2:])):
        c = by.get(k2)
        if c and c["n"]:
            print(f"  {k2:8s} n={c['n']:4d} right={c['ok'] / c['n']:.3f}")
    print(f"  stopped early (a stop where a step was due): {by['stopped_early']['n']}  -> {out_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="", help="a *_slots.jsonl (whole records: question, gold, subtype)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--export", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--replay-gold", default="", help="a *_step.jsonl: replay its rows instead of a model")
    ap.add_argument("--teacher-forced", default="", help="a *_step.jsonl: score each gold step row's next emission")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=96)   # a step ends at </s>; the cap only guards a run-on
    a = ap.parse_args(argv)
    from build_emitter_sft import gold_matches
    if a.teacher_forced:
        return teacher_forced(a)
    recs = load(a.data)
    if a.limit:
        recs = recs[:a.limit]
    if a.replay_gold:
        rows = defaultdict(list)
        for line in open(a.replay_gold, encoding="utf-8"):
            r = json.loads(line)
            if r.get("task") == "arithmetic" and "#" in r["id"]:
                rows[r["id"].split("#")[0]].append(r)
        emitter = GoldReplay(rows)
    elif a.gguf:
        from standin.emitter import LlamaCppEmitter
        emitter = LlamaCppEmitter(a.gguf)
    else:
        from standin.emitter import Cubby450mEmitter
        emitter = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    loop = StepLoop(emitter, world=ArithmeticWorld(), max_steps=a.max_steps, max_new_tokens=a.max_new)
    out_path = os.path.join(HERE, "logs", f"exp_he19_step_loop_{a.tag}.json")
    st, by_k, results, t0 = Counter(), defaultdict(Counter), [], time.perf_counter()
    for i, r in enumerate(recs):
        if isinstance(emitter, GoldReplay):
            emitter.start(r["id"])
        K = len(blocks_of(r.get("program") or r.get("reference") or ""))
        res = loop.solve(r["question"])
        ok = res["answer"] is not None and bool(gold_matches(res["answer"], r.get("gold")))
        st["n"] += 1
        st["correct"] += ok
        st["answered"] += res["answer"] is not None
        st["wrong_answer"] += res["answer"] is not None and not ok
        if res["refused"]:
            st["refused:" + re.sub(r"\$S\d+|\d+", "#", res["refused"])[:40]] += 1
        st["same_steps"] += len(res["steps"]) == K
        by_k[K]["n"] += 1
        by_k[K]["correct"] += ok
        st["emissions"] += len(res["emissions"])
        results.append({"id": r["id"], "K": K, "gold": r.get("gold"), "answer": res["answer"], "correct": ok,
                        "refused": res["refused"], "steps": res["steps"], "emissions": res["emissions"][:6]})
        if (i + 1) % 20 == 0 or i + 1 == len(recs):
            print(f"  {i + 1}/{len(recs)} ({time.perf_counter() - t0:.0f}s) correct={st['correct'] / st['n']:.3f} "
                  f"answered={st['answered'] / st['n']:.3f} same_steps={st['same_steps'] / st['n']:.3f}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"emitter": getattr(emitter, "name", "?"), "data": a.data, "n": st["n"],
                           "summary": {k: (v / st["n"] if k not in ("n",) else v) for k, v in st.items()},
                           "by_steps": {k: {"n": c["n"], "correct": c["correct"] / c["n"]} for k, c in sorted(by_k.items())},
                           "results": results}, f, indent=1)
    n = st["n"] or 1
    print(f"\n{a.tag}: n={n} correct={st['correct'] / n:.3f} answered={st['answered'] / n:.3f} wrong_answer={st['wrong_answer'] / n:.3f} "
          f"same_steps={st['same_steps'] / n:.3f} emissions/problem={st['emissions'] / n:.2f}")
    print("  by steps: " + "  ".join(f"{k}:{c['correct'] / c['n']:.2f}/{c['n']}" for k, c in sorted(by_k.items())))
    print("  refusals: " + ", ".join(f"{k[8:]}={v}" for k, v in sorted(st.items()) if k.startswith("refused:")))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
