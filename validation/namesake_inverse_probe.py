"""namesake_inverse_probe -- order 0036, item 1: labels the hdc world shares between two or more items as OBJECTS, read
the inverse way ("what has X as its <relation>?").

Two readings per label:
- **the object index** (``HdcWorld._by_obj``): how many items' rows it returns for the label. Before 0036 it returned
  every namesake's ("paris": 655 rows from 14 items); now one item's, or none when several share the label;
- **the ask loop** (wiki + hdc, offline, 0014's grammar and lexicon): "What has X as its R?" for a relation X is the
  object of. An answer mixes items when it comes from the shared label read through more than one item.

The line: 0 mixed answers in 50.

    python validation/namesake_inverse_probe.py --src <hdc folder> [--n 50] [--seed 0]
-> validation/logs/namesake_inverse_probe_<date>.json

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the hdc folder (0014's grammar)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.question_forms import TemplateEmitter
    from cubbyllm.worlds import wikikg as wk
    from cubbyllm.worlds.hdc_store import HdcWorld, Stores
    from exp_hdc_bench import Offline, set_forms
    hdc = HdcWorld()
    keys = sorted(k for (k,) in hdc.db.execute("SELECT okey FROM facts GROUP BY okey HAVING COUNT(DISTINCT oq) >= 2")
                  if len(k) >= 3 and not k.isdigit())
    labels = random.Random(a.seed).sample(keys, min(a.n, len(keys)))
    forms = set_forms(a.src, hdc)
    loop = AskLoop(TemplateEmitter(forms), world=Stores(wk.wiki_world(), hdc), source=Offline(), lexicon=True)
    loop.resolvers.append(forms)
    rows, tally = [], collections.Counter()
    for key in labels:
        items = {q for (q,) in hdc.db.execute("SELECT DISTINCT oq FROM facts WHERE okey = ?", (key,))}
        label, rel = hdc.db.execute("SELECT obj, rel FROM facts WHERE okey = ? ORDER BY rel LIMIT 1", (key,)).fetchone()
        read = hdc._by_obj.get(key)
        q = f"What has {label} as its {rel}?"
        rec = loop.ask(q)
        spoken = bool(rec.get("verified"))
        mixed = spoken and len(items) > 1 and not rec.get("namesake")
        outcome = "mixed" if mixed else "answered" if spoken else rec.get("reason") or "refused"
        tally[outcome] += 1
        rows.append({"label": label, "items": len(items), "index_rows_now": len(read), "question": q, "outcome": outcome,
                     "answer": rec.get("answer")})
    out = {"what": "order 0036 item 1: object labels the hdc world shares between items, read the inverse way",
           "labels": len(rows), "mixed": tally["mixed"], "outcomes": dict(tally),
           "index_reads_a_shared_label": sum(r["index_rows_now"] > 0 for r in rows), "args": {"n": a.n, "seed": a.seed},
           "rows": rows}
    path = os.path.join(HERE, "logs", f"namesake_inverse_probe_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), "->", path)


if __name__ == "__main__":
    main()
