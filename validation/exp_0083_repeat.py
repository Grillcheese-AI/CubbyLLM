"""exp_0083_repeat -- order 0083's bench: questions asked, then asked again, through the served stack, one process an arm.

The stack is the server's (`cubby_serve.cli.boot` with ``--ask --web off --history "" --no-board``: the wiki world, the
online Wikidata source the server hands its fetch, lexicon on, the everyday table on; the 450M programs seat on grilly2).
Arms, by flag: A (all off); M (``CB_ASK_MEMORY=1``, ``CB_ASK_MEMORY_WRITE=1``, the memory mounted empty); H
(``CB_HEAD_SHORTLIST=1``, route 5's index from ``--head-route``); MH (both). Each arm boots its own loop, so nothing
one arm learned (lever 4's facts, the episodes) serves another. Run M before A: the source's cache, warmed by the
first arm, then favours A, not memory.

Passes: gen3-1, gen3-2 -- the gen-3 held split, seed 7, 600 (r17's rows, as 0081 reads them), asked twice in the same
words and order. With ``--other N`` (in the arms ``--other-arms`` names): other-1, other-2 -- N one-hop pairs of the hdc
rephrased set (r18's file, seed 7), every pair's template wording, then every rephrasing (reported, no gate).

Per row (the scratch file ``validation/logs/_exp_0083_<arm>.rows.jsonl``, with the question): the answer against the
record's gold (r11's match: correct / near / WRONG), verified, the refusal reason, how (a remembered plan or not), the
episode, the memory step's time and tries, the time, and each seat call's slotted program, decode steps and ms and the
head that ran (``Cubby450mTrunk.steps``). ``--report`` reads every arm's rows into the report (no questions in it).

  <grilly2's python> validation/exp_0083_repeat.py --arm M --cubby450m <export> --emitter-adapter <dir> --head-route <npz>
  <grilly2's python> validation/exp_0083_repeat.py --report      -> validation/logs/exp_0083_repeat_<date>.json

Wired: STANDALONE (order 0083's bench).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

ARMS = {"A": {}, "M": {"CB_ASK_MEMORY": "1", "CB_ASK_MEMORY_WRITE": "1"}, "H": {"CB_HEAD_SHORTLIST": "1"},
        "MH": {"CB_ASK_MEMORY": "1", "CB_ASK_MEMORY_WRITE": "1", "CB_HEAD_SHORTLIST": "1"}}
FLAGS = ("CB_ASK_MEMORY", "CB_ASK_MEMORY_WRITE", "CB_HEAD_SHORTLIST", "CB_HEAD_ROUTE_INDEX")
WARM_UP = "What is the capital of France?"            # one seat call before the passes: the load is not in a question's time


def rows_file(arm: str) -> pathlib.Path:
    return LOGS / f"_exp_0083_{arm}.rows.jsonl"


def gen3_rows(n: int = 600, seed: int = 7) -> list[dict]:
    """r17's rows: the gen-3 held split's plan records, ``Random(seed).sample``."""
    from cubbyllm.local_paths import data_dir
    held = [json.loads(l) for l in open(data_dir("gen3_free_text.jsonl"), encoding="utf-8")]
    held = [r for r in held if r["task"] == "plan" and r["split"] == "held"]
    return [{"q": r["prompt"], "gold": r["gold"], "wording": r.get("wording")}
            for r in random.Random(seed).sample(held, min(n, len(held)))]


def other_rows(n: int, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """N one-hop pairs of the hdc rephrased set: (the template wordings, the rephrasings), pair by pair."""
    import pyarrow.parquet as pq
    from cubbyllm.local_paths import expand
    path = pathlib.Path(expand("$CUBBY_DATASETS/hdc")) / "ood_validation_rephrased.pq"   # r18's file
    table = [r for r in pq.read_table(path).to_pylist() if int(r["n_hop"]) == 1]
    pick = random.Random(seed).sample(table, min(n, len(table)))
    return ([{"q": r["question_prompt_original"], "gold": str(r["answer"]), "wording": "template"} for r in pick],
            [{"q": r["question_prompt"], "gold": str(r["answer"]), "wording": "rephrased"} for r in pick])


def timed(trunk, calls: list) -> None:
    """Each trunk decode -> calls: its role, ms, steps and the head that ran them (the trunk's own counts)."""
    gen = trunk.generate

    def generate(role, rows, max_new_tokens, **kw):
        before = collections.Counter(trunk.steps)
        t = time.perf_counter()
        out = gen(role, rows, max_new_tokens, **kw)
        heads = {k: v - before.get(k, 0) for k, v in trunk.steps.items() if v != before.get(k, 0)}
        calls.append({"role": role, "ms": round((time.perf_counter() - t) * 1000, 1), "steps": sum(heads.values()),
                      "heads": heads})
        return out
    trunk.generate = generate


def run_arm(a) -> None:
    for k in FLAGS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[a.arm])
    if a.head_route and "CB_HEAD_SHORTLIST" in ARMS[a.arm]:
        os.environ["CB_HEAD_ROUTE_INDEX"] = a.head_route
    from cubby_serve import cli
    from cubbyllm.emit.seats_450m import short_name
    from cubbyllm.ops.grilly_seats import device_name
    from cubbyllm.reasoning.planner import normalize
    from exp_r11_search_learn import match
    sap = cli.parser()
    argv = ["--cubby450m", a.cubby450m, "--emitter-adapter", a.emitter_adapter, "--ask", "--web", "off",
            "--history", "", "--no-board"]               # a bench's asks stay out of the sleep cycle's input
    argv += ["--tokenizer", a.tokenizer] if a.tokenizer else []
    loop = cli.boot(sap.parse_args(argv), sap).ask_loop
    seat = loop.emitter.adapters["programs"] if hasattr(loop.emitter, "adapters") else loop.emitter
    trunk = seat.inner.trunk
    calls: list = []
    timed(trunk, calls)
    try:
        seat.emit(WARM_UP, max_new_tokens=loop.max_new)   # loads the base (and attaches the route) outside the passes
    except Exception as e:                                # noqa: BLE001 -- a refused warm-up program still loaded the base
        print(f"warm-up: {type(e).__name__}: {e}", flush=True)
    calls.clear()
    head = {"arm": a.arm, "flags": ARMS[a.arm], "export": short_name(a.cubby450m), "adapter": short_name(a.emitter_adapter),
            "route": trunk.route is not None, "card": device_name(), "relations": len(loop.known),
            "date": datetime.date.today().isoformat()}
    print(json.dumps(head), flush=True)
    passes = [("gen3-1", gen3_rows(a.n, a.seed)), ("gen3-2", gen3_rows(a.n, a.seed))]
    if a.other and a.arm in a.other_arms.split(","):
        template, rephrased = other_rows(a.other, a.seed)
        passes += [("other-1", template), ("other-2", rephrased)]
    out = rows_file(a.arm)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"head": head}, ensure_ascii=False) + "\n")
        for name, rows in passes:
            t_pass = time.perf_counter()
            for i, r in enumerate(rows):
                calls.clear()
                seat.last = {}
                t = time.perf_counter()
                rec = loop.ask(r["q"])
                s = time.perf_counter() - t
                mem = rec.get("memory") or {}
                row = {"pass": name, "i": i, "q": r["q"], "gold": r["gold"], "wording": r["wording"],
                       "verified": bool(rec.get("verified")), "answer": rec.get("answer"), "reason": rec.get("reason"),
                       "match": match(rec["answer"], r["gold"], normalize) if rec.get("answer") else None,
                       "how": rec.get("how"), "episode": rec.get("episode"), "restated": bool(rec.get("restated")),
                       "plan": rec.get("plan"), "seed": rec.get("seed"), "s": round(s, 4),
                       "memory": {"order": mem.get("order"), "s": mem.get("s"),
                                  "tried": [(t_["kind"], t_["reason"]) for t_ in mem.get("tried", [])]} if mem else None,
                       "program": (seat.last or {}).get("raw"), "calls": list(calls)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"{a.arm} {name}: {len(rows)} in {time.perf_counter() - t_pass:.0f} s", flush=True)
    print("->", out, flush=True)


def pct(xs: list, q: float):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else None


def tally(rows: list[dict]) -> dict:
    refused = collections.Counter((r["reason"] or "?").split(":", 1)[0] for r in rows if not r["verified"])
    prog = [c for r in rows for c in r["calls"] if c["role"] == "programs" and c["steps"]]
    heads = collections.Counter()
    for c in prog:
        heads.update(c["heads"])
    return {"n": len(rows), "answered": sum(bool(r["answer"]) for r in rows), "verified": sum(r["verified"] for r in rows),
            "correct": sum(r["match"] == "correct" for r in rows), "near": sum(r["match"] == "near" for r in rows),
            "WRONG": sum(r["match"] == "WRONG" for r in rows),
            "WRONG_remembered": sum(r["match"] == "WRONG" and r["how"] == "a remembered plan" for r in rows),
            "remembered": sum(r["how"] == "a remembered plan" for r in rows),
            "restated": sum(r["restated"] for r in rows), "seat_calls": len(prog),
            "refused": dict(refused.most_common()), "s_median": pct([r["s"] for r in rows], 0.5),
            "s_p90": pct([r["s"] for r in rows], 0.9),
            "memory_miss_s_median": pct([r["memory"]["s"] for r in rows if r["memory"] and r["memory"]["s"] is not None
                                         and r["how"] != "a remembered plan"], 0.5),
            "steps": dict(heads), "step_ms_median": pct([c["ms"] / c["steps"] for c in prog], 0.5)}


def report(out: str | None) -> None:
    arms: dict = {}
    for arm in ARMS:
        p = rows_file(arm)
        if p.exists():
            lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            arms[arm] = {"head": lines[0]["head"], "rows": lines[1:]}
    rep = {"order": "0083", "what": "questions asked, then asked again, through the served stack, one loop an arm",
           "arms": {}, "gates": {}}
    for arm, d in arms.items():
        by = collections.defaultdict(list)
        for r in d["rows"]:
            by[r["pass"]].append(r)
        rep["arms"][arm] = {"head": d["head"], "passes": {k: tally(v) for k, v in by.items()}}
        if "gen3-1" in by and "gen3-2" in by:                 # the second asking against the first, on its certified rows
            first = {r["i"]: r for r in by["gen3-1"]}
            cert = [r for r in by["gen3-2"] if first[r["i"]]["verified"]]
            same = [r for r in cert if r["verified"] and r["answer"] == first[r["i"]]["answer"]]
            rep["arms"][arm]["repeat"] = {"certified_first": len(cert), "same_answer": len(same),
                                          "remembered": sum(r["how"] == "a remembered plan" for r in cert),
                                          "s_median_on_certified": pct([r["s"] for r in cert], 0.5),
                                          "lost": [r["i"] for r in cert if r not in same]}
        if "other-1" in by and "other-2" in by:
            first = {r["i"]: r for r in by["other-1"]}
            got = [r for r in by["other-2"] if first[r["i"]]["verified"]]
            rep["arms"][arm]["other_words"] = {"template_certified": len(got),
                                               "rephrased_verified": sum(r["verified"] for r in got),
                                               "rephrased_remembered": sum(r["how"] == "a remembered plan" for r in got),
                                               "rephrased_correct": sum(r["match"] == "correct" for r in got)}
    g = rep["gates"]
    for m, base in (("M", "A"), ("MH", "H")):
        if m in rep["arms"] and base in rep["arms"] and "repeat" in rep["arms"][m]:
            mr, mp, bp = rep["arms"][m]["repeat"], rep["arms"][m]["passes"], rep["arms"][base]["passes"]
            g[f"memory {m} vs {base}"] = {
                "second asking gives the first's certified answer on all": mr["same_answer"] == mr["certified_first"],
                "second asking median <= 0.2 s": (mr["s_median_on_certified"] or 0) <= 0.2,
                "first asking median <= base + 0.05 s": mp["gen3-1"]["s_median"] <= bp["gen3-1"]["s_median"] + 0.05,
                "0 wrong on every pass": all(p["WRONG"] == 0 for p in mp.values()),
                "kill: a wrong answer from a remembered plan": any(p["WRONG_remembered"] for p in mp.values())}
            base_wrong = {(r["pass"], r["i"]) for r in arms[base]["rows"] if r["match"] == "WRONG"}
            g[f"memory {m} vs {base}"]["WRONG not on the base arm's row"] = [   # reported beside the line, not instead
                [r["pass"], r["i"], r["how"]] for r in arms[m]["rows"]
                if r["match"] == "WRONG" and (r["pass"], r["i"]) not in base_wrong]
    for x, base in (("H", "A"), ("MH", "M")):                 # route 5 against none: what each arm says on the same rows
        if x in arms and base in arms:
            said = {(r["pass"], r["i"]): (r["verified"], r["answer"]) for r in arms[base]["rows"]}
            g[f"route {x} vs {base}: answers differ on"] = [[r["pass"], r["i"]] for r in arms[x]["rows"]
                                                            if said.get((r["pass"], r["i"]), (r["verified"], r["answer"]))
                                                            != (r["verified"], r["answer"])]
    if "A" in arms and "H" in arms:                          # the programs seat's greedy programs, route against none
        a_prog = {(r["pass"], r["i"]): r["program"] for r in arms["A"]["rows"] if r["program"] is not None}
        h_prog = {(r["pass"], r["i"]): r["program"] for r in arms["H"]["rows"] if r["program"] is not None}
        both = [k for k in a_prog if k in h_prog and k[0] == "gen3-1"]
        same = sum(a_prog[k] == h_prog[k] for k in both)
        g["route programs identical"] = {"prompts": len(both), "identical": same,
                                         "share": round(same / len(both), 4) if both else None,
                                         ">= 99%": bool(both) and same / len(both) >= 0.99,
                                         "differ": [k[1] for k in both if a_prog[k] != h_prog[k]][:40]}
    out = out or str(LOGS / f"exp_0083_repeat_{datetime.date.today().isoformat()}.json")
    pathlib.Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["gates"], indent=1), "\n->", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--report", action="store_true", help="read every arm's rows file into the report")
    ap.add_argument("--cubby450m")
    ap.add_argument("--emitter-adapter")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--head-route", default=None, help="route 5's index (0068's index_router_succ.npz) for arms H, MH")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--other", type=int, default=0, help="N hdc rephrased one-hop pairs for the other-words pass")
    ap.add_argument("--other-arms", default="M")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.report:
        return report(a.out)
    if not (a.arm and a.cubby450m and a.emitter_adapter):
        ap.error("--arm, --cubby450m and --emitter-adapter (or --report)")
    run_arm(a)


if __name__ == "__main__":
    main()
