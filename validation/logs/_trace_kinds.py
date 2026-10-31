"""A profile_base chrome trace re-read from the device kernels only (the 2026-10-20 check of the
double count): device time by kind, per step, with profile_base's own kinds."""
import json
import sys
from collections import defaultdict

sys.path.insert(0, sys.argv[2])
from profile_base import kind_of  # noqa: E402

ev = json.load(open(sys.argv[1], encoding="utf-8"))["traceEvents"]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
dev = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cats = defaultdict(float)
for e in ev:
    if e.get("ph") == "X":
        cats[e.get("cat")] += e.get("dur", 0)
by, names = defaultdict(float), defaultdict(float)
for e in dev:
    by[kind_of(e["name"])] += e["dur"]
    names[e["name"]] += e["dur"]
tot = sum(by.values())
span = (max(e["ts"] + e["dur"] for e in dev) - min(e["ts"] for e in dev)) / 1e3 / steps
print(f"device kernels {tot / 1e3 / steps:.1f} ms a step over a {span:.1f} ms span ({100 * tot / 1e3 / steps / span:.0f}% busy)")
for k, t in sorted(by.items(), key=lambda kv: -kv[1]):
    print(f"  {k:24s} {100 * t / tot:5.1f}%  {t / 1e3 / steps:7.1f} ms")
print("event categories (ms a step):", {k: round(v / 1e3 / steps, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])})
print("top kernels:")
for n, t in sorted(names.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {100 * t / tot:5.1f}%  {t / 1e3 / steps:7.2f} ms  {kind_of(n)[:10]:10s} {n[:100]}")
