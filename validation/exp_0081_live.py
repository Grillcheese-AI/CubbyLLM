"""exp_0081_live -- order 0081: H-E15's eight live questions through the served loop (``--ask --web off``), the 450M
programs seat under each ``CB_EMIT_TRIE`` arm (A ``off``, B ``rel``, C ``rel+slot``), on the same questions.

The brain is booted the way the server boots it (`cubby_serve.cli.boot`: the wiki world, the online Wikidata source
the server hands its fetch, lexicon on, the web off). H-E15's log did not keep the questions' exact words; they are
restated here from its entry (the Oak Park wording is the one `learn.py` quotes from that day). Each question runs
twice per arm: with today's everyday wording table (``CB_EVERYDAY`` unset: the host restates "Where was X born?"
itself, and the seat is not asked) and with it off (``CB_EVERYDAY=0``, as on 2026-10-17, so the seat plans every one).

  <grilly2's python> validation/exp_0081_live.py --cubby450m <export> --emitter-adapter <dir> [--talk-adapter <dir>]
      -> validation/logs/exp_0081_live_<date>.json

Wired: STANDALONE (order 0081's bench).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

QUESTIONS = [
    "Who is the mother of Justin Trudeau?",
    "Which country is Oak Park located in?",
    "Who is the father of Bill Haslam?",
    "who is marie curie?",
    "Where was Marie Curie born?",
    "Where was Ernest Hemingway born?",
    "What is the capital of the country of citizenship of Marie Curie?",
    "who is the prime minister of canada?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubby450m", required=True)
    ap.add_argument("--emitter-adapter", required=True)
    ap.add_argument("--talk-adapter", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--arms", default="off,rel,rel+slot")
    ap.add_argument("--everyday", default="on,off", help="the everyday table: on (today's default), off (2026-10-17's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from cubby_serve import cli
    from cubbyllm.emit.seats_450m import short_name
    from cubbyllm.ops.grilly_seats import device_name
    sap = cli.parser()
    argv = ["--cubby450m", a.cubby450m, "--emitter-adapter", a.emitter_adapter, "--ask", "--web", "off"]
    argv += ["--talk-adapter", a.talk_adapter] if a.talk_adapter else []
    argv += ["--tokenizer", a.tokenizer] if a.tokenizer else []
    brain = cli.boot(sap.parse_args(argv), sap)
    loop = brain.ask_loop
    seat = loop.emitter.adapters["programs"] if hasattr(loop.emitter, "adapters") else loop.emitter
    report = {"order": "0081", "what": "H-E15's eight live questions, the 450M seat under each trie arm",
              "export": short_name(a.cubby450m), "adapter": short_name(a.emitter_adapter),
              "talk": short_name(a.talk_adapter) if a.talk_adapter else None, "relations": len(loop.known),
              "card": device_name(),
              "questions": QUESTIONS, "runs": []}
    for everyday in a.everyday.split(","):
        loop._everyday = None
        if everyday == "on":
            from cubbyllm.reasoning.everyday import EverydayForms
            loop._everyday = EverydayForms()
        for q in QUESTIONS:
            for arm in a.arms.split(","):
                os.environ["CB_EMIT_TRIE"] = arm
                seat.last = {}
                t0 = time.perf_counter()
                rec = loop.ask(q)
                last = seat.last or {}
                row = {"everyday": everyday, "arm": arm, "question": q, "verified": rec.get("verified"),
                       "answer": rec.get("answer"), "reason": rec.get("reason"), "plan": rec.get("plan"),
                       "seed": rec.get("seed"), "restated": rec.get("restated"), "learned": rec.get("learned"),
                       "aliased": rec.get("aliased"), "raw": last.get("raw"), "trie": last.get("trie"),
                       "s": round(time.perf_counter() - t0, 2)}
                report["runs"].append(row)
                print(f"[{everyday}|{arm}] {q} -> {row['answer']!r} verified={row['verified']} reason={row['reason']} "
                      f"plan={row['plan']} seed={row['seed']} ({row['s']}s)", flush=True)
    os.environ["CB_EMIT_TRIE"] = "off"
    out = a.out or str(ROOT / "validation" / "logs" / f"exp_0081_live_{datetime.date.today().isoformat()}.json")
    pathlib.Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
