"""H-E19: what KIND of mistake is a wrong next step? Re-reads the baseline 400-row menu ranking (exp_he19_rank) and
classifies every step row whose top-1 is wrong against its gold step:
  op        same two operands, wrong operation
  operands  same operation, different operands
  both      operation and operands differ
and, separately: does the gold step reuse an earlier VM result ($S), does the wrong pick, and does it pick an
older $S than gold. Also top-1 by menu size within each depth, to separate depth from menu size. CPU, no model.

    python validation/exp_he19_errtype.py --rank validation/logs/exp_he19_rank_held400.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEP = re.compile(r"assign s\d+ = (\$[NS]\d+);\s*(add|sub|mul|div) s\d+, (\$[NS]\d+);")


def parse(text: str):
    m = STEP.search(text or "")
    return (m.group(2), m.group(1), m.group(3)) if m else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default=os.path.join(HERE, "logs", "exp_he19_rank_held400.json"))
    ap.add_argument("--data", default=os.path.join(ROOT, "standin", "data", "out", "pf_heldout_eval_w_step.jsonl"))
    ap.add_argument("--tag", default="held400")
    a = ap.parse_args(argv)
    gold = {}
    for line in open(a.data, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r
    rows = [r for r in json.load(open(a.rank, encoding="utf-8"))["rows"] if not r["is_stop"]]
    kind = defaultdict(Counter)
    by_size = defaultdict(lambda: [0, 0])
    lines = []
    for r in rows:
        d = r["k"] if r["k"] < 3 else 3
        g = parse(gold[r["id"]]["program"])
        size = r["groups"]
        b = "<=60" if size <= 60 else "61-120" if size <= 120 else ">120"
        by_size[(d, b)][0] += 1
        by_size[(d, b)][1] += r["rank"] == 0
        if r["rank"] == 0 or g is None:
            continue
        p = parse(r["top"])
        if p is None:
            kind[d]["unparsed"] += 1
            continue
        same_ops = sorted(p[1:]) == sorted(g[1:])
        kind[d]["op" if same_ops and p[0] != g[0] else "operands" if p[0] == g[0] else "both"] += 1
        gs = [x for x in g[1:] if x.startswith("$S")]
        ps = [x for x in p[1:] if x.startswith("$S")]
        if gs and not ps:
            kind[d]["gold reuses $S, pick does not"] += 1
        if ps and not gs:
            kind[d]["pick reuses $S, gold does not"] += 1
        if gs and ps and max(int(x[2:]) for x in ps) < max(int(x[2:]) for x in gs):
            kind[d]["pick uses an OLDER $S"] += 1
    lines.append("wrong top-1 step picks by type (baseline adapter, 400-row ranking):")
    keys = ["op", "operands", "both", "gold reuses $S, pick does not", "pick reuses $S, gold does not",
            "pick uses an OLDER $S", "unparsed"]
    for d in sorted(kind):
        n = sum(kind[d][k] for k in ("op", "operands", "both", "unparsed"))
        lines.append(f"  depth {d if d < 3 else '3+'}: {n} wrong | " +
                     " | ".join(f"{k} {kind[d][k]} ({kind[d][k] / max(1, n):.2f})" for k in keys if kind[d][k]))
    lines.append("\ntop-1 by menu size (value groups) within depth:")
    for (d, b), (n, ok) in sorted(by_size.items()):
        lines.append(f"  depth {d if d < 3 else '3+'}  menu {b:<7} n={n:<4} top-1 {ok / max(1, n):.3f}")
    print("\n".join(lines))
    out = os.path.join(HERE, "logs", f"exp_he19_errtype_{a.tag}")
    open(out + ".log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"kind": {str(k): dict(v) for k, v in kind.items()},
               "by_size": {f"{k[0]}|{k[1]}": v for k, v in by_size.items()}}, open(out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
