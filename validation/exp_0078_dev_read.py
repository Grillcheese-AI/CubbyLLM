"""exp_0078_dev_read - order 0078's dev read (Cowork's ruling, 2026-11-20): one arm's adapter on GSM8K dev (the 329
val arithmetic problems of `emitter_sft_v12e_w_slots.jsonl`), step by step in the VM (H-E19's `StepLoop`).

Per question it writes one row (jsonl, resumable: a question already in the file is skipped):
- **greedy:** one greedy chain (`StepLoop.solve`, H-E19's loop): answer, refusal, steps. Reported, not deciding.
- **tries:** K sampled chains (temperature 0.7, top-k 40), each run step by step in the VM exactly as the greedy chain
  is (the same slot check, the same VM, the same stop), each try's answer the VM value it stopped on (None when it was
  refused). The chains run in lockstep: at each step the live chains that share a prompt are sampled in ONE batched
  decode (`emit_samples`), seeded by (question, step, group). k = 8 is the first 8 of the 16.

The speak rule's m and k are chosen afterwards from these rows (`--score`, `cubbyllm.reasoning.speak_vote.choose_m`),
with no second pass. Test is never read here.

    python validation/exp_0078_dev_read.py --arm s0 --export <base450m_cont/grilly> --adapter <adapter> \\
        --tokenizer <tokenizer.json> --data <emitter_sft_v12e_w_slots.jsonl> [--limit 20]
    python validation/exp_0078_dev_read.py --score validation/logs/gsm8k_dev_read_s0_<date>.jsonl [--drop <ids.json>]
    python validation/exp_0078_dev_read.py --score <rows> --rules --data <v12e_w_slots> [--gsm8k-train <train.jsonl>]
    python validation/exp_0078_dev_read.py --arm novm --ckpt <base .pt> --tokenizer <..> --data <..> --gsm8k-train <..>

`--rules` (0078's addendum, part 1): the answer-kind rules checked on GSM8K train's golds, then the score without and
with them (`exp_0078_dev_score.rules_read`). `--arm novm` (part 4): the base with no adapter, 4-shot, no VM
(`exp_0078_novm.py`); its rows are scored by `--score` as any arm's.

Wiring: STANDALONE (an experiment; order 0078).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(ROOT, "validation", "sft")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cubbyllm.reasoning.arith_world import ArithmeticWorld  # noqa: E402
from cubbyllm.reasoning.slots import Span, canon, extract, num_value  # noqa: E402
from cubbyllm.reasoning.step_loop import (PROMPT_HEAD, PROMPT_TAIL, StepLoop, derivation, parse_emission,  # noqa: E402
                                          render_step, wrap_step)

__wiring__ = "STANDALONE"

K, TEMPERATURE, TOP_K = 16, 0.7, 40


class Chain:
    """One try: its own slot table, stepped by the emission it is handed (StepLoop.solve's body, one step a call)."""

    def __init__(self, question: str, world, vm, record: bool = False):
        self.table = extract(question, constants=world.constants(question) if world else None)
        self.vm, self.k, self.steps = vm, 0, []
        self.answer, self.refused, self.done = None, None, False
        self.record, self.log, self.stop = record, [], None   # order 0085: what each step did, kept only to write it

    def prompt(self) -> str:
        return PROMPT_HEAD + self.table.annotate().strip() + PROMPT_TAIL

    def step(self, raw: str, max_steps: int) -> None:
        kind, what = parse_emission(raw)
        if kind == "stop":
            s = self.table.get(what)
            self.answer, self.refused = (s.filled, None) if s is not None else (None, f"stop on unknown slot {what}")
            self.done, self.stop = True, what
            return
        if kind != "step":
            self.refused, self.done = "malformed step", True
            return
        block_text = render_step(self.k, what.ops)
        verdict = self.table.check(block_text)
        if not verdict.ok:
            self.refused, self.done = verdict.reason, True
            if self.record:
                self.log.append({"k": self.k, "program": block_text, "refused": verdict.reason})
            return
        ran = wrap_step(self.table.fill(block_text))
        ok, result = self.vm(ran)
        v = num_value(result) if ok else None
        if self.record:
            from cubbyllm.records.arith_trace import sha256_text
            self.log.append({"k": self.k, "program": block_text, "ran_sha256": sha256_text(ran),
                             "outcome": "return" if v is not None else "error",
                             **({"value": canon(v)} if v is not None else {})})
        if v is None:
            self.refused, self.done = "vm: the step did not run", True
            return
        self.k += 1
        self.table.spans.append(Span(f"$S{self.k}", derivation(what.ops).replace("$", ""), -1, -1, "S", canon(v)))
        self.steps.append(canon(v))
        if self.k > max_steps:
            self.refused, self.done = f"no stop within {max_steps} steps", True


def tries(emitter, loop: StepLoop, question: str, qi: int, k: int = K, keep: "list | None" = None) -> list[dict]:
    """``keep``: a list the chains are appended to, each recording its steps (order 0085's writer)."""
    chains = [Chain(question, loop.world, loop.vm, record=keep is not None) for _ in range(k)]
    if keep is not None:
        keep.extend(chains)
    for step in range(loop.max_steps + 2):
        live = [c for c in chains if not c.done]
        if not live:
            break
        if hasattr(emitter, "emit_rows"):                 # one call for every live try (dev_read_torch: by length)
            raws = emitter.emit_rows([c.prompt() for c in live], max_new_tokens=loop.max_new_tokens,
                                     temperature=TEMPERATURE, top_k=TOP_K, seed=qi * 100 + step)
            for c, raw in zip(live, raws):
                c.step(raw, loop.max_steps)
            continue
        groups: dict = {}
        for c in live:
            groups.setdefault(c.prompt(), []).append(c)
        for gi, (prompt, cs) in enumerate(groups.items()):
            raws = emitter.emit_samples(prompt, len(cs), max_new_tokens=loop.max_new_tokens, temperature=TEMPERATURE,
                                        top_k=TOP_K, seed=(qi * 100 + step) * 100 + gi)
            for c, raw in zip(cs, raws):
                c.step(raw, loop.max_steps)
    return [{"answer": c.answer, "refused": c.refused, "steps": c.steps} for c in chains]


def _ids(spec: str) -> list:
    if os.path.isfile(spec):
        text = open(spec, encoding="utf-8").read()
        if spec.endswith(".json"):
            d = json.loads(text)
            return list(d["ids"] if isinstance(d, dict) else d)
        return text.split()
    return spec.split(",")


def load(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == "arithmetic" and r.get("split", "val") == "val" and "#" not in r["id"]:
            out.append(r)
    return out


def run(a) -> None:
    recs = list(enumerate(load(a.data)))[a.offset:]
    if a.limit:
        recs = recs[:a.limit]
    if a.ids:                                              # the dev order's index kept: the same seeds as the read
        ids = set(_ids(a.ids))
        recs = [(i, r) for i, r in recs if r["id"] in ids]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(ln)["id"] for ln in open(a.out, encoding="utf-8")}
    if a.ckpt:
        from dev_read_torch import TorchStepEmitter        # the batched decode (validation/dev_read_torch.py)
        emitter = TorchStepEmitter(a.ckpt, a.adapter, a.tokenizer)
    else:
        from cubbyllm.emit.seats_450m import Cubby450mEmitter
        emitter = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    loop = StepLoop(emitter, world=ArithmeticWorld(), max_steps=a.max_steps, max_new_tokens=a.max_new)
    writer = None
    if a.records:                                          # order 0085: a record per question, the read frozen
        from arith_records import Writer
        from cubbyllm.reasoning.answer_kind import RULES
        if not (a.speak_k and a.speak_m):
            raise SystemExit("--records needs the frozen --speak-k and --speak-m")
        writer = Writer(a.records, a.adapter, {"k": a.speak_k, "m": a.speak_m, "temperature": TEMPERATURE,
                                               "top-k": TOP_K, "rules": list(RULES) if a.rules else [],
                                               "tries decoded": a.k, "max steps": a.max_steps, "max new": a.max_new},
                        RULES if a.rules else (), name=a.run_name or a.arm, calibration=a.calibration,
                        calibration_at=a.calibration_at)
    t0, n = time.perf_counter(), 0
    try:
        with open(a.out, "a", encoding="utf-8") as f:
            for i, r in recs:
                if r["id"] in done:
                    continue
                q0 = time.perf_counter()
                g = loop.solve(r["question"])
                q1 = time.perf_counter()
                keep = [] if writer else None
                ts = tries(emitter, loop, r["question"], i, a.k, keep)
                q2 = time.perf_counter()
                sec = {"greedy": round(q1 - q0, 2), "tries": round(q2 - q1, 2)}
                if writer:
                    sec["writer"] = writer.write(r["id"], i, r["question"], keep)
                f.write(json.dumps({"id": r["id"], "i": i, "gold": r["gold"], "arm": a.arm, "adapter": emitter.name,
                                    "greedy": {"answer": g["answer"], "refused": g["refused"],
                                               "steps": [s["value"] for s in g["steps"]]},
                                    "tries": ts,
                                    "decode": getattr(emitter, "decode_kind", "rows" if hasattr(emitter, "emit_rows") else "groups"),
                                    "seconds": sec}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                el = time.perf_counter() - t0
                print(f"  {n}/{len(recs)} {r['id']} greedy {q1 - q0:.1f}s tries {q2 - q1:.1f}s | mean {el / n:.1f}s a question",
                      flush=True)
    finally:
        if writer:
            writer.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="")
    ap.add_argument("--export", default="", help="the grilly2 export (home card)")
    ap.add_argument("--ckpt", default="", help="the torch base checkpoint instead (Colab, CUDA): TorchStepEmitter")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--score", default="", help="a rows file: choose m and k on it (exp_0078_dev_score)")
    ap.add_argument("--drop", default="", help="with --score: a json list of dev ids to leave out (S0's footnote)")
    ap.add_argument("--all-dev", action="store_true", help="with --score: choose on all rows (before the split)")
    ap.add_argument("--rules", action="store_true", help="with --score and --data: the answer-kind rules (part 1)")
    ap.add_argument("--gsm8k-train", default="", help="GSM8K train with ids (the rules' gold check; novm's shots)")
    ap.add_argument("--shots-seed", type=int, default=0, help="--arm novm: the seed drawing the 4 shots")
    ap.add_argument("--records", default="", help="order 0085: write a record per question here (off the repo)")
    ap.add_argument("--speak-k", type=int, default=0, help="with --records: the frozen k (the first k tries vote)")
    ap.add_argument("--speak-m", type=int, default=0, help="with --records: the frozen m")
    ap.add_argument("--ids", default="", help="read only these dev ids (a json list, a file of ids, or a comma list)")
    ap.add_argument("--calibration", default="", help="with --records: the dev rows m was chosen on (hashed)")
    ap.add_argument("--calibration-at", default="", help="with --calibration: when m was chosen (xsd:dateTime)")
    ap.add_argument("--run-name", default="", help="with --records: the run's name (its record ids derive from it)")
    a = ap.parse_args(argv)
    if not a.gsm8k_train:
        from cubbyllm import local_paths
        a.gsm8k_train = os.path.join(local_paths.value("CUBBY_DATASETS") or ".", "gsm8k", "train.jsonl")
    if a.score:
        from exp_0078_dev_score import HELD, rules_read, score   # k, m on dev without 80:200; 80:200 read once at them
        held = None if a.all_dev else HELD
        if a.rules:
            qs = {r["id"]: r["question"] for r in load(a.data)}
            print(json.dumps(rules_read(a.score, a.gsm8k_train, qs, a.drop, held), indent=1))
        else:
            print(json.dumps(score(a.score, a.drop, held), indent=1))
        return
    a.out = a.out or os.path.join(HERE, "logs", f"gsm8k_dev_read_{a.arm}_{time.strftime('%Y-%m-%d')}.jsonl")
    if a.arm == "novm":
        from exp_0078_novm import run as run_novm
        recs = load(a.data)[a.offset:]
        run_novm(a, recs[:a.limit] if a.limit else recs)
        return
    run(a)


if __name__ == "__main__":
    main()
