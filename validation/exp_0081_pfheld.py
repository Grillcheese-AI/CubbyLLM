"""exp_0081_pfheld -- order 0081's third read set, reported with no gate: H-E18's unseen-writer eval
(``pf_heldout_eval_w_slots.jsonl``: its plan and chain records, in the slot form the served adapter reads), generated
per ``CB_EMIT_TRIE`` arm (A ``off``, B ``rel``, C ``rel+slot``) by the 450M programs seat as served (greedy, no
repetition guards, the full head), each record's own slot table checking and filling the program as
`train_emitter_torch.generate_val` does, and scored by `exp_he18_ab.score` (every program run in the real VM).
The relations: the served loop's store (the wiki world). Arm A here is the floor order 0082's adapter is read against.

  <grilly2's python> validation/exp_0081_pfheld.py --cubby450m <export> --emitter-adapter <dir> --data <the eval jsonl>
      -> validation/logs/exp_0081_pfheld_<date>.json (resumable: --resume)

Wired: STANDALONE (order 0081's bench).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubby450m", required=True)
    ap.add_argument("--emitter-adapter", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True, help="pf_heldout_eval_w_slots.jsonl")
    ap.add_argument("--tasks", default="plan,chain")
    ap.add_argument("--arms", default="off,rel,rel+slot")
    ap.add_argument("--max-new", type=int, default=640, help="the trainer's --gen-max-new")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    from cubbyllm.emit import plan_trie
    from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk, short_name
    from cubbyllm.ops.grilly_seats import device_name
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.worlds import wikikg as wk
    from emitter_data import table_of
    from exp_he18_ab import score
    tasks, arms = a.tasks.split(","), a.arms.split(",")
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    recs = [r for r in recs if r["task"] in tasks]
    wk.ensure_data(("triplets",))
    known = StoreRelations(wk.wiki_world().index._seen)
    trunk = Cubby450mTrunk(a.cubby450m, a.tokenizer or os.path.join(a.cubby450m, "tokenizer.json"),
                           {"programs": a.emitter_adapter})
    seat = Cubby450mEmitter(trunk)
    progress = LOGS / "_exp_0081_pfheld.rows.jsonl"
    rows = [json.loads(l) for l in open(progress, encoding="utf-8")] if a.resume and progress.exists() else []
    if not a.resume:
        progress.write_text("", encoding="utf-8")
    done = {(r["id"], r["arm"]) for r in rows}
    print(f"{len(recs)} records ({', '.join(tasks)}), {len(known):,} relations, arms {arms}; {len(done)} done", flush=True)
    t0 = time.time()
    for i, r in enumerate(recs, 1):
        table = table_of(r["spans"], r["question"])
        for arm in arms:
            if (r["id"], arm) in done:
                continue
            os.environ["CB_EMIT_TRIE"] = arm
            trie = plan_trie.for_seat(seat, known, [s.id for s in table.spans if s.kind in "ER"])
            t = time.perf_counter()
            gen = seat.emit(r["prompt"], max_new_tokens=a.max_new, **({"allowed": trie.allowed} if trie else {}))
            ms = round((time.perf_counter() - t) * 1000, 1)
            verdict = table.check(gen)
            row = {"id": r["id"], "arm": arm, "task": r["task"], "subtype": r.get("subtype", ""),
                   "reference": r["reference"], "gold": r.get("gold"), "generated": table.fill(gen),
                   "generated_slotted": gen, "slots_ok": verdict.ok, "slot_reason": verdict.reason, "ms": ms,
                   "trie": trie.stats() if trie else None}
            rows.append(row)
            with open(progress, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"  {i}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    os.environ["CB_EMIT_TRIE"] = "off"
    report = {"order": "0081", "what": "H-E18's unseen-writer eval, per trie arm (no gate; arm A = 0082's floor)",
              "data": pathlib.Path(a.data).name, "export": short_name(a.cubby450m),
              "adapter": short_name(a.emitter_adapter), "relations": len(known), "max_new": a.max_new,
              "card": device_name(), "arms": {}}
    for arm in arms:
        out = [x for x in rows if x["arm"] == arm]
        ms = [x["ms"] for x in out]
        report["arms"][arm] = {"score": score(out), "ms_median": statistics.median(ms) if ms else None,
                               "ms_p95": sorted(ms)[int(0.95 * len(ms))] if ms else None,
                               "entered": {k: sum((x["trie"] or {}).get("entered", {}).get(k, 0) for x in out)
                                           for k in ("HOP", "SEED")}}
        print(arm, json.dumps(report["arms"][arm]["score"].get("_all")), flush=True)
    report["rows"] = rows
    path = LOGS / f"exp_0081_pfheld_{datetime.date.today().isoformat()}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
