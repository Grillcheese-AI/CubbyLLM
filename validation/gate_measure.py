"""What the harvest correctness gate (VM step 0, 2026-09-29) removes from the v3cf
harvest: verified chains split by correct / wrong / ungradable, and covered /
uncovered against the prompt the SFT builder would show. Read-only.

  python validation/gate_measure.py [harvest.jsonl]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "standin", "data"))
import build_emitter_sft as b  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else b.HARVEST
c = Counter()
uncovered_examples = []
by_hop = Counter()
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    c["records"] += 1
    if not (r.get("verified") and r.get("program_source")):
        c["not_verified"] += 1
        continue
    c["verified"] += 1
    correct = b.chain_correct(r)
    c["correct" if correct else ("wrong" if correct is False else "ungradable")] += 1
    if correct is not True:
        continue
    prompt = r["question"]
    facts = [t.get("fact") for t in (r.get("trace") or []) if t.get("fact")]
    if facts:
        prompt = prompt + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
    unc = b.uncovered_literals(prompt, r["program_source"])
    if unc:
        c["uncovered"] += 1
        if len(uncovered_examples) < 5:
            uncovered_examples.append((r["question"][:80], unc[:3]))
    else:
        c["kept"] += 1
        by_hop[r.get("n_hop")] += 1
print(os.path.relpath(path, ROOT))
for k in ("records", "not_verified", "verified", "correct", "wrong", "ungradable", "uncovered", "kept"):
    print(f"  {k:<13} {c[k]}")
print(f"  kept by n_hop: {dict(sorted(by_hop.items(), key=lambda kv: str(kv[0])))}")
for q, unc in uncovered_examples:
    print(f"  uncovered: {q!r} -> {unc}")
