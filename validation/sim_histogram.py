"""WO-0.4 -- read the serve-time similarity log and raise the two alarms.

`cubbyllm/reasoning/simlog.py` writes one JSONL line per SPOKEN answer, with
the similarity of every accepted `recover` binding. This reads that file and
answers the two questions the threshold alone cannot:

  1. **How much of what we say rests on a weak binding?** The fraction of
     spoken bindings below 0.5. Rising across rounds is the alarm.

  2. **Is the mass piling up just above the threshold?** The margin
     distribution -- how far the weakest accepted binding sat above the tau it
     had to clear. Decisions riding the threshold are the confabulation
     signature: the VSA cleanup returns the nearest symbol rather than the
     right one, and tau is the only thing between that and a spoken answer.
     This matters because `recover` on an unbound role never returns Null --
     it returns the globally-nearest symbol at ~0.03 (cubelang DRIFT.md B11a).

It also prints the separation between accepted bindings and the CONTROL role,
which is supposed to stay below tau. A shrinking separation is the same alarm
arriving earlier: it means the frame is getting noisy enough that the control
and the real answer are becoming hard to tell apart.

    CUBBY_SIMLOG=validation/logs/similarity.jsonl  <run the thing that serves>
    python validation/sim_histogram.py validation/logs/similarity.jsonl

Weekly is the intended cadence; there is nothing expensive here.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import sys

# Bin edges chosen around the taus actually in use, not round numbers:
# tau_vm is 1.0 at hop 1, 0.4736328125 at hop 2, 0.22021484375 at hop 3. The
# 0.22-0.30 band is the one WO-0.4 names explicitly as the danger zone.
BINS = [0.0, 0.05, 0.10, 0.22, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
WEAK = 0.5


def _bin(x: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= x < BINS[i + 1]:
            return i
    return len(BINS) - 2


def _bar(n: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return ""
    return "#" * max(0, round(width * n / total))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="the JSONL written by CUBBY_SIMLOG")
    ap.add_argument("--min-separation", type=float, default=4.0,
                    help="alarm when the weakest accepted binding is less than "
                         "this multiple of the strongest control-role "
                         "similarity (default 4.0). THE alarm -- see the note "
                         "the report prints about why the absolute level and "
                         "the margin above tau are not.")
    ap.add_argument("--compare",
                    help="a previous window's JSONL. WO-0.4's first alarm is "
                         "about the weak-binding fraction RISING, which needs "
                         "two windows to read.")
    ap.add_argument("--weak-limit", type=float, default=0.10,
                    help="with --compare, alarm when the fraction of accepted "
                         "bindings below 0.5 rises by more than this "
                         "(default 0.10)")
    ap.add_argument("--margin-limit", type=float, default=0.25,
                    help="retained for reporting only; see --min-separation.")
    a = ap.parse_args(argv)

    path = pathlib.Path(a.log)
    if not path.is_file():
        print(f"sim_histogram: no such log: {path}", file=sys.stderr)
        print("Set CUBBY_SIMLOG to this path and serve some answers first.",
              file=sys.stderr)
        return 2

    sims: list[float] = []
    margins: list[float] = []
    ctrls: list[float] = []
    by_hop: dict[int, list[float]] = collections.defaultdict(list)
    n_rows = 0

    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            for s in row.get("sims") or []:
                if isinstance(s, (int, float)):
                    sims.append(float(s))
                    by_hop[row.get("n_hop") or 0].append(float(s))
            m = row.get("margin")
            if isinstance(m, (int, float)):
                margins.append(float(m))
            c = row.get("ctrl_sim")
            if isinstance(c, (int, float)):
                ctrls.append(float(c))

    if not sims:
        print(f"{path}: {n_rows} rows, no accepted-binding similarities recorded.")
        return 0

    print(f"{path}")
    print(f"  spoken answers      : {n_rows}")
    print(f"  accepted bindings   : {len(sims)}")
    print()

    counts = collections.Counter(_bin(s) for s in sims)
    print("  accepted-binding similarity")
    for i in range(len(BINS) - 1):
        n = counts.get(i, 0)
        lo, hi = BINS[i], BINS[i + 1]
        mark = "  <- tau_vm hop3 band" if (lo, hi) == (0.22, 0.30) else ""
        print(f"    [{lo:.2f},{hi:.2f})  {n:6d}  {_bar(n, len(sims))}{mark}")
    print()

    weak = sum(1 for s in sims if s < WEAK)
    weak_frac = weak / len(sims)
    print(f"  below {WEAK}          : {weak} ({weak_frac:.1%})   "
          f"(absolute, NOT an alarm on its own -- see below)")

    if margins:
        tight = sum(1 for m in margins if m < 0.05)
        tight_frac = tight / len(margins)
        margins_sorted = sorted(margins)
        med = margins_sorted[len(margins_sorted) // 2]
        print(f"  weakest-hop margin  : median {med:+.4f}, "
              f"{tight} of {len(margins)} ({tight_frac:.1%}) under 0.05   "
              f"(expected to be small -- see below)")
    else:
        tight_frac = 0.0

    # THE alarm statistic. See the note under `SEPARATION` below for why the
    # two numbers above are descriptive rather than diagnostic.
    sep_ratio = None
    if ctrls:
        ctrl_sorted = sorted(ctrls)
        ctrl_med = ctrl_sorted[len(ctrl_sorted) // 2]
        ctrl_max = ctrl_sorted[-1]
        acc_min = min(sims)
        print(f"  control role sim    : median {ctrl_med:.4f}, "
              f"max {ctrl_max:.4f} (must stay below tau_vm)")
        sep_ratio = acc_min / ctrl_max if ctrl_max > 0 else float("inf")
        print(f"  SEPARATION          : weakest accepted {acc_min:.4f} vs "
              f"strongest control {ctrl_max:.4f} = {sep_ratio:.1f}x")

    if len(by_hop) > 1:
        print("\n  by hop count")
        for n_hop in sorted(by_hop):
            v = sorted(by_hop[n_hop])
            print(f"    {n_hop}-hop: {len(v):5d} bindings, median {v[len(v) // 2]:.4f}, "
                  f"min {v[0]:.4f}")

    print(f"""
  SEPARATION is the alarm statistic, not the two numbers above it.
  tau_vm is DERIVED from the expected cosine of a k-element bundle
  (1.0 at 1 hop, 0.4736328125 at 2, 0.22021484375 at 3), so a correct
  binding lands just above its tau BY CONSTRUCTION. "Margin under 0.05"
  and "below 0.5" therefore fire at ~100% on a perfectly healthy run --
  the first version of this tool did exactly that and called it a
  confabulation signature, which was wrong. What actually distinguishes a
  real binding from cleanup noise is the gap between the accepted
  bindings and the CONTROL role, which is bound to nothing and measures
  the floor directly.""")

    alarms = []
    if sep_ratio is not None and sep_ratio < a.min_separation:
        alarms.append(
            f"accepted/control separation is only {sep_ratio:.1f}x "
            f"(limit {a.min_separation:.0f}x). The weakest thing being spoken is "
            f"getting close to a role bound to nothing. THIS is the "
            f"confabulation signature: `recover` never returns Null, so once "
            f"the gap closes, tau is picking between noise and noise.")
    # Trend, not level, is what WO-0.4 asks for on the weak fraction. With a
    # single window there is no trend to read, so it is reported and not
    # alarmed on; feed two windows to compare.
    if a.compare:
        prev = pathlib.Path(a.compare)
        if prev.is_file():
            prev_sims = []
            with io.open(prev, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for s in r.get("sims") or []:
                        if isinstance(s, (int, float)):
                            prev_sims.append(float(s))
            if prev_sims:
                prev_weak = sum(1 for s in prev_sims if s < WEAK) / len(prev_sims)
                delta = weak_frac - prev_weak
                print(f"\n  vs {prev.name}: below-{WEAK} fraction "
                      f"{prev_weak:.1%} -> {weak_frac:.1%} ({delta:+.1%})")
                if delta > a.weak_limit:
                    alarms.append(
                        f"the fraction of accepted bindings below {WEAK} rose "
                        f"{delta:+.1%} since the previous window "
                        f"(limit {a.weak_limit:+.0%}). WO-0.4's first alarm is "
                        f"about this RISING, not about its level.")
        else:
            print(f"\n  (no previous window at {prev}; nothing to compare)")

    if alarms:
        print("\nALARM")
        for msg in alarms:
            print(f"  - {msg}")
        return 1

    print("\nOK: accepted bindings are cleanly separated from the control floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
