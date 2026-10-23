"""audit_namesake_tier -- order 0036, item 4: 50 of the relation tier's decisions, drawn by seed, for Cowork to read by
hand. The tier (``hdc_namesakes.hop_pick``) lets a walk's hop read one item of a shared name when that item alone
holds the hop's relation, its kind is known, no other item shares a kind with it, and no other item's kind carries the
relation anywhere in the world (the kinds table).

Each decision is listed with:
- the label and the relation;
- every item the label names, with its kinds and a few of its own values;
- the one chosen, and the value the hop reads.

The reader judges: is that the item a person asking "what is the <relation> of <label>?" means? The line: at least 49
of 50 right; one wrong and the tier goes strict.

Each decision is also made with the scan the table replaced (item 3), and any difference is listed.

    python validation/audit_namesake_tier.py [--n 50] [--seed 0]
-> validation/logs/audit_namesake_tier_<date>.json (and .md: the list to read)

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.planner import QuestionPlan, normalize  # noqa: E402
from cubbyllm.worlds.hdc_namesakes import kinds  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld  # noqa: E402

__wiring__ = "STANDALONE"


def decide(hdc: HdcWorld, label: str, rel: str):
    hdc.clear_withheld()
    got = hdc.hop(QuestionPlan(relations=[None], tail=f"{rel} of {label}", n_hop=1), 0, None)
    return got


def chosen_item(by: dict, got) -> str | None:
    facts = {f for f, _t in got}
    return next((q for q, rows in by.items() if facts and facts <= {f for f, _t in rows}), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    table, scan = HdcWorld(), HdcWorld()
    scan._kinds = None
    labels = sorted(k for (k,) in table.db.execute("SELECT skey FROM facts GROUP BY skey HAVING COUNT(DISTINCT sq) >= 2")
                    if len(k) >= 3 and not k.isdigit())
    random.Random(a.seed).shuffle(labels)
    rows, tried, differ = [], 0, []
    for key in labels:
        if len(rows) >= a.n:
            break
        by = table.namesakes(key)
        if len(by) < 2:
            continue
        tried += 1
        held = {}
        for q, its in by.items():
            for _f, t in its:
                held.setdefault(t.rel, set()).add(q)
        for rel in sorted(r for r, qs in held.items() if len(qs) == 1 and normalize(r) != "instance of"):
            got = decide(table, key, rel)
            if not got:
                continue
            q = chosen_item(by, got)
            s_got = decide(scan, key, rel)
            if {f for f, _t in s_got} != {f for f, _t in got}:
                differ.append({"label": key, "relation": rel})
            label = got[0][1].subj
            rows.append({"label": label, "relation": rel, "question": f"What is the {rel} of {label}?",
                         "chosen": q, "reads": sorted({t.obj for _f, t in got}),
                         "items": [{"item": i, "kinds": sorted(kinds(its)), "values": [f"{t.rel}: {t.obj}" for _f, t in its][:4]}
                                   for i, its in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]],
                         "items_n": len(by)})
            break
    out = {"what": "order 0036 item 4: the relation tier's decisions on shared hdc labels, drawn by seed, for a reading "
                   "by hand (right = the chosen item is the one the question means)", "date": a.date, "seed": a.seed,
           "decisions": len(rows), "labels_tried": tried, "scan_differs": differ, "rows": rows}
    path = os.path.join(HERE, "logs", f"audit_namesake_tier_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = [f"# The relation tier's {len(rows)} decisions (order 0036 item 4; seed {a.seed}; {tried} shared labels tried)", "",
          "Right = the chosen item is the one a person asking the question means. Mark each line right or wrong.", ""]
    for i, r in enumerate(rows, 1):
        md.append(f"{i}. **{r['question']}** reads {', '.join(r['reads'])} (from {r['chosen']}, {r['items_n']} items)")
        for it in r["items"]:
            mark = "**chosen**" if it["item"] == r["chosen"] else "other"
            md.append(f"   - {mark} {it['item']} ({', '.join(it['kinds']) or 'no kind'}): {'; '.join(it['values'])}")
    md.append("")
    md.append(f"The scan gives a different decision on {len(differ)} of {len(rows)}: {differ or 'none'}.")
    open(path[:-5] + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps({k: out[k] for k in ("decisions", "labels_tried", "scan_differs")}), "->", path)


if __name__ == "__main__":
    main()
