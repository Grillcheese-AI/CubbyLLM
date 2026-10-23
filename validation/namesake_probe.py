"""namesake_probe -- order 0033's probe: labels the hdc world shares between two or more items, asked "where is X?" and
"what is X?" through the ask loop (the wiki world, then the hdc world; no model, no network). An answer mixes two items
when no one item holds every line it says. The line: 0 mixed answers in 50 labels; a label it cannot settle is asked
back (`ambiguous_entity`, the items named), never merged.

Each line spoken is traced to the items that hold it (the hdc rows with that label, relation and value). Lines only the
wiki world holds carry no item and are counted apart.

    CUBBY_HDC_WORLD=<hdc_world.sqlite> python validation/namesake_probe.py [--n 50] [--seed 0] [--tag _before]
-> validation/logs/namesake_probe_<date><tag>.json

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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.reasoning.planner import normalize  # noqa: E402

__wiring__ = "STANDALONE"

WHERE_WHAT = ("country", "located in the administrative territorial entity", "continent", "located in or next to body of "
              "water", "instance of", "part of", "capital of", "headquarters location", "located on terrain feature")


GLOBE = (("Where is the Globe Theatre?", None), ("Where is the Globe Theatre?", "Q5571005"),
         ("Where is the Globe Theatre in Boston?", None), ("Where is the Globe Theatre in London?", None))
HOPS = (("capital", "France"), ("currency", "Japan"), ("capital", "Canada"), ("capital", "Italy"), ("capital", "Georgia"),
        ("population", "Toronto"), ("country", "Globe Theatre"), ("located in the administrative territorial entity", "Wantage"),
        ("official language", "Brazil"), ("country", "Eiffel Tower"), ("instance of", "Terry White"))


class NoPlan:
    """The emitter: none (a profile question is read off the question itself)."""
    name = "none"

    def emit(self, prompt, max_new_tokens=300, **kw):
        return ""


class Offline:
    name = "offline"

    def facts(self, entity, **kw):
        return []


def shared_labels(db, n: int, seed: int) -> list[str]:
    """Labels whose facts come from two or more items, at least two of them with a where / what fact."""
    marks = ",".join("?" * len(WHERE_WHAT))
    rows = db.execute(f"SELECT skey, COUNT(DISTINCT sq) FROM facts WHERE rel IN ({marks}) GROUP BY skey "
                      "HAVING COUNT(DISTINCT sq) >= 2", WHERE_WHAT).fetchall()
    keys = sorted(k for k, _n in rows if len(k) >= 4 and not k.isdigit())
    return random.Random(seed).sample(keys, min(n, len(keys)))


def items_of(db, label_key: str, rel: str, obj: str) -> set[str]:
    return {q for (q,) in db.execute("SELECT sq FROM facts WHERE skey = ? AND rel = ? AND obj = ?",
                                     (label_key, rel, obj))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("CUBBY_HDC_WORLD"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.worlds import wikikg as wk
    from cubbyllm.worlds.hdc_store import HdcWorld, Stores
    t0 = time.time()
    hdc = HdcWorld(a.db)
    labels = shared_labels(hdc.db, a.n, a.seed)
    loop = AskLoop(NoPlan(), world=Stores(wk.wiki_world(), hdc), source=Offline(), lexicon=False)
    rows, tally = [], collections.Counter()
    for key in labels:
        name = hdc.db.execute("SELECT subj FROM facts WHERE skey = ? LIMIT 1", (key,)).fetchone()[0]
        n_items = hdc.db.execute("SELECT COUNT(DISTINCT sq) FROM facts WHERE skey = ?", (key,)).fetchone()[0]
        for ask in ("Where is", "What is"):
            q = f"{ask} {name}?"
            rec = loop.ask(q)
            lines = rec.get("profile") or []
            per = [items_of(hdc.db, key, normalize(l["relation"]), l["value"]) or items_of(hdc.db, key, l["relation"], l["value"])
                   for l in lines]
            held = [s for s in per if s]
            mixed = bool(held) and not set.intersection(*held)
            kind = ("mixed" if mixed else "answered" if lines else rec.get("reason") or "refused")
            tally[kind] += 1
            rows.append({"label": name, "items": n_items, "question": q, "outcome": kind, "reason": rec.get("reason"),
                         "answer": rec.get("answer"), "lines_by_item": [sorted(s) for s in per],
                         "wiki_only_lines": sum(1 for s in per if not s),
                         "candidates": len(rec.get("candidates") or []) or None})
    # the order's own case, a choice coming back, and a qualifier that decides and one that does not
    fixed = []
    for q, item in GLOBE:
        rec = loop.ask(q, item=item)
        fixed.append({"question": q, "item": item, "reason": rec.get("reason"), "answer": rec.get("answer"),
                      "namesake": rec.get("namesake"), "choices": (rec.get("clarify") or {}).get("choices")})
    # a walk's hop on real shared names: the relation decides, or the items are withheld (hdc_namesakes.hop_pick)
    from cubbyllm.reasoning.planner import QuestionPlan
    hops = []
    for rel, name in HOPS:
        hdc.clear_withheld()
        t = time.perf_counter()
        got = hdc.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)
        hops.append({"hop": f"{rel} of {name}", "items": len(hdc.namesakes(name)), "read": [x.obj for _f, x in got][:3],
                     "withheld": hdc.withheld(name), "ms": round(1000 * (time.perf_counter() - t), 1)})
    out = {"what": "order 0033: labels the hdc world shares between items, asked where / what is X", "labels": len(labels),
           "questions": len(rows), "outcomes": dict(tally), "mixed": tally["mixed"], "wall_s": round(time.time() - t0, 1),
           "args": {"n": a.n, "seed": a.seed}, "globe_theatre": fixed, "hops": hops, "rows": rows}
    path = os.path.join(HERE, "logs", f"namesake_probe_{a.date}{a.tag}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False))
    print("->", path)


if __name__ == "__main__":
    main()
