"""Did the speaker attribution survive the pipeline?

Owner, annotating the gold set: "SELF is ME most of the time", "PEER = ME also",
"ALL ME AGAIN". Most of those read as confirmation, but "PEER = ME also" reads as a
CORRECTION, and if speaker attribution is wrong anywhere the corpus is compromised -
an affect corpus that mixes up who said what is worse than no corpus, because every
label inherits the error silently.

So: walk back from the scrubbed rows to the original export and check that each
[SELF]/[PEER] tag matches the author the export actually recorded. Prints initials
only, never full names, so the check can be reported without echoing anyone's name.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib

CORPUS = pathlib.Path(r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out\teams_corpus.jsonl")
KEY = pathlib.Path(r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out\goldset_key.json")
OUT = pathlib.Path(r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out\speaker_check.txt")


def initials(name: str) -> str:
    return "".join(p[0] for p in (name or "?").split()[:2]).upper() or "?"


def main() -> int:
    export = glob.glob(r"C:\Users\grill\Desktop\TeamsExport*.json")[0]
    blob = json.load(open(export, encoding="utf-8"))
    raw = [m for m in blob["messages"]
           if not m.get("system") and not m.get("deleted") and (m.get("text") or "").strip()]
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    key = json.load(open(KEY, encoding="utf-8"))

    # map (timestamp, text) -> original author, so the check does not rely on index
    by_ts = {}
    for m in raw:
        by_ts.setdefault(m.get("timestamp"), []).append(m)

    lines = []
    authors = {}
    mismatch = 0
    for r in rows:
        cand = by_ts.get(r.get("when")) or []
        if not cand:
            continue
        a = cand[0].get("author") or "?"
        authors.setdefault(a, {"[SELF]": 0, "[PEER]": 0, "[PEER2]": 0, "other": 0})
        tag = r.get("speaker")
        authors[a][tag if tag in authors[a] else "other"] += 1

    lines.append("Original author -> tag it received (initials only)")
    lines.append("")
    for a, counts in sorted(authors.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(counts.values())
        spread = ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        lines.append(f"  {initials(a)}  n={tot:<6} -> {spread}")
        if sum(1 for v in counts.values() if v) > 1:
            mismatch += 1
    lines.append("")
    lines.append("Each original author should map to EXACTLY ONE tag. "
                 f"Authors with a split mapping: {mismatch}")

    # the items the owner flagged
    lines.append("")
    lines.append("Context around the flagged gold-set items, with true initials:")
    for n in (4, 5, 8, 12):
        item = next((k for k in key if k["n"] == n), None)
        if not item:
            continue
        i = item["i"]
        lines.append("")
        lines.append(f"--- item {n:02d} ---")
        for r in rows[max(0, i - 3):i + 1]:
            cand = by_ts.get(r.get("when")) or []
            who = initials(cand[0].get("author")) if cand else "??"
            mark = "  >" if r is rows[i] else "   "
            lines.append(f"{mark} tag={r['speaker']:<8} true={who}  {r['text'][:80]}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
