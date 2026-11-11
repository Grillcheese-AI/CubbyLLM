"""Order 0054, the field watch's step 1 (approved by Nick, 2026-11-11): a right answer whose certificate does not
carry it. From "Trace Integrity for LLM Data Agents" and owl-equivariance (a reasoner checks that the cited axioms
alone entail the answer).

Every spoken answer of arm C is asked again, by the same host with the same flags and the same seat (the run's own
programs, replayed by prompt), over a world that holds ONLY the statements its certificate names (the record's
`trace`), with a live source that holds nothing (an empty offline cache). The same answer must come out.

Outcomes per spoken answer:
- ``same``: spoken again, the same answer;
- ``other``: spoken again, another answer -- the certificate does not say what the answer rests on;
- ``refused``: not spoken again (its reason kept) -- the answer needed something the certificate does not carry;
- ``no_certificate``: spoken with an empty trace (nothing to replay).
**Line:** every spoken answer ``same``. **Kill:** one ``other`` or ``refused`` is a hole in the certificate, read
before any other number of the order.

With ``--names`` (Cowork's shape of 2026-11-11): the record's name statement (`reasoning.name_statement`, kept by
run_c) is honoured: its step's ground is checked (``name_statement.ground``: re-derived / stated only / not the same,
reported apart), and the world also holds the trace's facts said of the asked name, so the walk can start from it.
"Stated only" is reported beside "same", never inside it; "not the same" is the kill.

    python validation/system_eval/replay_cert.py --run <run dir> --bench webq [--file c_webq.jsonl]
        --programs <programs.jsonl> --out validation/logs/cert_replay_<date>.json

Wired: STANDALONE (a measurement; never imported by cubbyllm/).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"


def outcome(rec: dict, again: dict) -> str:
    if not rec.get("trace"):
        return "no_certificate"
    if not again.get("verified"):
        return "refused"
    return "same" if str(again.get("answer")) == str(rec.get("answer")) else "other"


class Replayer:
    """The run's host, one loop per answer, its world the answer's certificate."""

    def __init__(self, programs: str) -> None:
        from cubbyllm.worlds import wikikg as wk
        from cubbyllm.reasoning.slots import SlotEmitter
        from cubbyllm.sources.wikidata import WikidataSource
        from cubby_serve.build import slot_index
        from system_eval.seats import Gather
        wk.ensure_data(("triplets",))
        self.seat = Gather.load(programs)
        self.em = SlotEmitter(self.seat, index=None)
        self.em.index = slot_index(wk.wiki_world())      # the names the host knows, as in the run: the same prompts
        self._tmp = tempfile.TemporaryDirectory()
        self.source = WikidataSource(cache_dir=self._tmp.name, offline=True)    # holds nothing

    def ask(self, question: str, trace: list[str], name: dict | None = None) -> dict:
        from cubbyllm.reasoning.ask import AskLoop
        from cubbyllm.worlds.base import FactStore
        facts = list(trace)
        if name and (name.get("item") or {}).get("label") and name.get("asked"):
            label, asked = name["item"]["label"], name["asked"]                  # the name statement, honoured: the
            facts += [f.replace(f" of {label}", f" of {asked}") for f in trace if f.endswith(f" of {label}")]   # asked name
        world = FactStore(facts, name="certificate")
        loop = AskLoop(self.em, world=world, source=self.source, lexicon=True, history_path=None)
        try:
            return loop.ask(question)
        except Exception as e:                           # noqa: BLE001 -- a crash is a refusal, its reason kept
            return {"answer": None, "verified": False, "reason": f"error: {str(e)[:160]}"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's folder")
    ap.add_argument("--bench", action="append", required=True)
    ap.add_argument("--file", action="append", default=None, help="the records file per --bench (default c_<bench>.jsonl)")
    ap.add_argument("--programs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", action="store_true", help="honour each record's name statement (run_c's `name`)")
    a = ap.parse_args(argv)
    os.environ["CB_RECORD_UNCERTIFIED"] = "1"
    files = a.file or [f"c_{b}.jsonl" for b in a.bench]
    t0 = time.perf_counter()
    rp = Replayer(a.programs)
    print(f"host loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    out = {"what": "order 0054, field-watch step 1: each spoken answer asked again over its certificate alone",
           "run": os.path.basename(os.path.normpath(a.run)),
           "flags": {k: v for k, v in os.environ.items() if k.startswith("CB_")}, "benches": {}, "not_same": []}
    for bench, fn in zip(a.bench, files):
        recs = [json.loads(x) for x in open(os.path.join(a.run, fn), encoding="utf-8") if x.strip()]
        spoken = [r for r in recs if r.get("spoken")]
        counts = collections.Counter()
        for r in spoken:
            st = r.get("name") if a.names else None
            again = rp.ask(r["question"], r.get("trace") or [], st)
            o = outcome(r, again)
            counts[o] += 1
            if a.names:
                from cubbyllm.reasoning.name_statement import ground
                g = ground(st, r.get("trace") or []) if st else "no statement"
                counts["name: " + g] += 1
                if g == "not the same":
                    out["not_same"].append({"bench": bench, "id": r["id"], "question": r["question"], "outcome": "name not the same",
                                            "name": st})
            if o != "same":
                out["not_same"].append({"bench": bench, "id": r["id"], "question": r["question"], "outcome": o,
                                        "answer": r.get("answer"), "verdict": r.get("verdict"), "trace": r.get("trace"),
                                        "again": {"answer": again.get("answer"), "reason": again.get("reason"),
                                                  "trace": again.get("trace")}})
        out["benches"][bench] = {"spoken": len(spoken), **dict(counts)}
        print(bench, out["benches"][bench], flush=True)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for n in out["not_same"]:
        print("  NOT SAME", n["bench"], n["outcome"], n["question"], "|", n["answer"], "->", n["again"]["answer"], n["again"]["reason"])


if __name__ == "__main__":
    main()
