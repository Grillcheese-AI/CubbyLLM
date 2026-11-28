"""exp_0087_cover_names -- order 0087's read: the served stack, arm A as 0083 ran it, with ``CB_COVER_NAMES`` off and on.

The stack is the server's (``cubby_serve.cli.boot`` with ``--ask --web off --history "" --no-board``, as
``exp_0083_repeat`` boots it: the wiki world, the online Wikidata source, lexicon and the everyday table on, the 450M
programs seat on grilly2), every other flag off. One process an arm (``--arm off`` / ``--arm on``); run ``on`` first,
so the source's cache, warmed by the first arm, favours the arm without the rule.

Passes, each asked once: gen3 -- the gen-3 held split, seed 7, 600 (r17's rows); other -- the hdc rephrased one-hop
set, 150 (seed 7; 0083's other-2, where the "Ghana" defect was found), reported apart; live8 -- H-E15's eight live
questions; self -- "who are you?" and "what are you?".

Per row (``validation/logs/_exp_0087_<arm>.rows.jsonl``, with the question): verified, the answer against gold (r11's
match), the reason and the names check's detail, the plan and seed, the time and the time spent in ``cover_names``.
``--report`` reads both arms into the gates the order pre-registered (no questions in it). The self line as Cowork
read it (2026-11-28): "who are you?" and "what are you?" are never answered from a store -- refused with
``seed_is_pronoun`` or answered by the host's self shape -- and the report says which took each.

  <grilly2's python> validation/exp_0087_cover_names.py --arm on --cubby450m <export> --emitter-adapter <dir>
  <grilly2's python> validation/exp_0087_cover_names.py --report       -> validation/logs/exp_0087_cover_names_<date>.json

Wired: STANDALONE (order 0087's bench).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]
LOGS = ROOT / "validation" / "logs"
from exp_0083_repeat import FLAGS as FLAGS_0083, WARM_UP, gen3_rows, other_rows, pct, timed  # noqa: E402

FLAGS = tuple(FLAGS_0083) + ("CB_COVER_NAMES",)
LIVE8 = ["Who is the mother of Justin Trudeau?", "Which country is Oak Park located in?", "Who is the father of Bill Haslam?",
         "who is marie curie?", "Where was Marie Curie born?", "Where was Ernest Hemingway born?",
         "What is the capital of the country of citizenship of Marie Curie?", "who is the prime minister of canada?"]
SELF = ["who are you?", "what are you?"]
REASONS = ("seed_is_relation", "name_left_over", "seed_is_pronoun")


def rows_file(arm: str) -> pathlib.Path:
    return LOGS / f"_exp_0087_{arm}.rows.jsonl"


def clocked(spent: list) -> None:
    """Every call into cover_names' two entry points adds its seconds to spent[0] (the time the rule adds)."""
    from cubbyllm.reasoning import cover_names as C
    for name in ("refusal", "slots"):
        fn = getattr(C, name)

        def wrap(*a, _fn=fn, **k):
            t = time.perf_counter()
            try:
                return _fn(*a, **k)
            finally:
                spent[0] += time.perf_counter() - t
        setattr(C, name, wrap)


def run_arm(a) -> None:
    for k in FLAGS:
        os.environ.pop(k, None)
    if a.arm == "on":
        os.environ["CB_COVER_NAMES"] = "1"
    from cubby_serve import cli
    from cubbyllm.emit.seats_450m import short_name
    from cubbyllm.ops.grilly_seats import device_name
    from cubbyllm.reasoning import cover_names as C
    from cubbyllm.reasoning.planner import normalize
    from exp_r11_search_learn import match
    sap = cli.parser()
    argv = ["--cubby450m", a.cubby450m, "--emitter-adapter", a.emitter_adapter, "--ask", "--web", "off",
            "--history", "", "--no-board"]
    argv += ["--tokenizer", a.tokenizer] if a.tokenizer else []
    loop = cli.boot(sap.parse_args(argv), sap).ask_loop
    seat = loop.emitter.adapters["programs"] if hasattr(loop.emitter, "adapters") else loop.emitter
    calls: list = []
    timed(seat.inner.trunk, calls)
    try:
        seat.emit(WARM_UP, max_new_tokens=loop.max_new)
    except Exception as e:                                # noqa: BLE001 -- a refused warm-up program still loaded the base
        print(f"warm-up: {type(e).__name__}: {e}", flush=True)
    spent = [0.0]
    clocked(spent)
    gen3 = gen3_rows(a.n, a.seed)
    _template, rephrased = other_rows(a.other, a.seed)
    pron = sum(any(C.is_pronoun(t) for t, _s, _e in seat.index.spans(r["q"])) for r in gen3)   # the index's own spans
    head = {"arm": a.arm, "flags": {"CB_COVER_NAMES": os.environ.get("CB_COVER_NAMES")}, "export": short_name(a.cubby450m),
            "adapter": short_name(a.emitter_adapter), "card": device_name(), "relations": len(loop.known),
            "names": len(seat.index), "gen3_with_a_pronoun_span": pron, "date": datetime.date.today().isoformat()}
    print(json.dumps(head), flush=True)
    passes = [("gen3", gen3), ("other", rephrased), ("live8", [{"q": q, "gold": None, "wording": None} for q in LIVE8]),
              ("self", [{"q": q, "gold": None, "wording": None} for q in SELF])]
    out = rows_file(a.arm)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"head": head}, ensure_ascii=False) + "\n")
        for name, rows in passes:
            t_pass = time.perf_counter()
            for i, r in enumerate(rows):
                calls.clear()
                seat.last = {}
                spent[0] = 0.0
                t = time.perf_counter()
                rec = loop.ask(r["q"])
                s = time.perf_counter() - t
                refused = rec.get("refused") if isinstance(rec.get("refused"), dict) else {}
                row = {"pass": name, "i": i, "q": r["q"], "gold": r["gold"], "wording": r["wording"],
                       "verified": bool(rec.get("verified")), "answer": rec.get("answer"), "reason": rec.get("reason"),
                       "names": refused.get("names"),
                       "match": match(rec["answer"], r["gold"], normalize) if rec.get("answer") and r["gold"] else None,
                       "restated": bool(rec.get("restated")), "plan": rec.get("plan"), "seed": rec.get("seed"),
                       "s": round(s, 4), "names_s": round(spent[0], 6), "program": (seat.last or {}).get("raw"),
                       "seat_ms": round(sum(c["ms"] for c in calls), 1)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"{a.arm} {name}: {len(rows)} in {time.perf_counter() - t_pass:.0f} s", flush=True)
    print("->", out, flush=True)


def load(arm: str) -> tuple[dict, list[dict]]:
    lines = [json.loads(l) for l in rows_file(arm).open(encoding="utf-8")]
    return lines[0]["head"], lines[1:]


def tally(rows: list[dict]) -> dict:
    refused = collections.Counter((r["reason"] or "?").split(":", 1)[0] for r in rows if not r["answer"])
    return {"n": len(rows), "spoken": sum(bool(r["answer"]) for r in rows), "verified": sum(r["verified"] for r in rows),
            "correct": sum(r["match"] == "correct" for r in rows), "near": sum(r["match"] == "near" for r in rows),
            "WRONG": sum(r["match"] == "WRONG" for r in rows),
            "by_the_rule": {k: refused.get(k, 0) for k in REASONS}, "refused": dict(refused.most_common()),
            "s_median": pct([r["s"] for r in rows], 0.5), "names_s_median": pct([r["names_s"] for r in rows], 0.5),
            "names_s_max": max((r["names_s"] for r in rows), default=None)}


def report(out: str | None) -> None:
    (h_off, off), (h_on, on) = load("off"), load("on")
    by = {arm: {p: [r for r in rows if r["pass"] == p] for p in ("gen3", "other", "live8", "self")}
          for arm, rows in (("off", off), ("on", on))}
    passes = {p: {"off": tally(by["off"][p]), "on": tally(by["on"][p])} for p in ("gen3", "other", "live8", "self")}
    g_off = {r["i"]: r for r in by["off"]["gen3"]}
    wrong_on = [r for r in by["on"]["gen3"] if r["match"] == "WRONG"]
    added = [r["i"] for r in wrong_on if not (g_off[r["i"]]["match"] == "WRONG"
                                             and (g_off[r["i"]]["answer"] or "").lower() == (r["answer"] or "").lower())]
    down = passes["gen3"]["off"]["correct"] - passes["gen3"]["on"]["correct"]
    lost = [{"i": r["i"], "reason": r["reason"], "names": r["names"], "plan": r["plan"], "seed": r["seed"]}
            for r in by["on"]["gen3"] if g_off[r["i"]]["match"] == "correct" and r["match"] != "correct"]
    wrong_off = [r["i"] for r in by["off"]["gen3"] if r["match"] == "WRONG"]
    on_at = {r["i"]: r for r in by["on"]["gen3"]}
    ghana = {arm: sum((r["answer"] or "").strip().lower() == "ghana" for r in by[arm]["other"]) for arm in ("off", "on")}
    rel_seed_spoken = [r["i"] for r in by["on"]["other"] if r["answer"] and normalize_seed(r["seed"]) == "citizenship"]
    live = [(a["q"], a["verified"], a["answer"], b["verified"], b["answer"])
            for a, b in zip(by["off"]["live8"], by["on"]["live8"])]
    selfs = {r["q"]: r["reason"] for r in by["on"]["self"]}
    gates = {"other: Ghana on 0 rows": ghana["on"] == 0, "other: a relation-word seed spoken": rel_seed_spoken,
             "self: never answered from a store": all(v in ("seed_is_pronoun", "self") for v in selfs.values()),
             "gen3: no WRONG added": not added, "gen3: correct at most 6 down": down <= 6,
             "live8 unchanged": all(v1 == v2 and (s1 or "") == (s2 or "") for _q, v1, s1, v2, s2 in live),
             "kill: a WRONG added or more than 12 down": bool(added) or down > 12}
    rep = {"order": "0087", "what": "CB_COVER_NAMES off and on, arm A as 0083 ran it", "heads": {"off": h_off, "on": h_on},
           "passes": passes, "gates": gates, "gen3_correct_down": down, "gen3_wrong_added": added,
           "gen3_right_answers_refused": lost,
           "gen3_wrong_off_and_on": [{"i": i, "on": on_at[i]["reason"] if not on_at[i]["answer"] else on_at[i]["match"]}
                                     for i in wrong_off],
           "other_ghana": ghana, "self_on": selfs,
           "live8": [{"off": [v1, s1], "on": [v2, s2]} for _q, v1, s1, v2, s2 in live]}
    path = pathlib.Path(out or LOGS / f"exp_0087_cover_names_{datetime.date.today().isoformat()}.json")
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=1), "\n->", path)


def normalize_seed(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["off", "on"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cubby450m")
    ap.add_argument("--emitter-adapter")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--other", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.report:
        return report(a.out)
    if not (a.arm and a.cubby450m and a.emitter_adapter):
        ap.error("--arm, --cubby450m and --emitter-adapter (or --report)")
    run_arm(a)


if __name__ == "__main__":
    main()
