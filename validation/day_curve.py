"""The within-day fatigue curve, on the right clock.

Owner: *"my writing cadence slows down, I reply less often, sentences are getting
shorter and shorter especially past 3pm"*.

Four claims, all aggregate rather than per-message, which is why this one can work
where the others did not. Per-message inference failed on every channel tried; a
trajectory over 9,738 messages does not need any single message to be readable.

  CADENCE     gap between consecutive messages should lengthen
  FREQUENCY   messages per active hour should fall
  LENGTH      words per message should fall
  THRESHOLD   the break should sit near 15:00 LOCAL

The earlier hour-of-day check missed this because it binned raw UTC and called the
result inconclusive. Quebec runs UTC-5 in winter and UTC-4 in summer, so "past 3pm"
was sitting at 19:00-20:00 in that table - exactly the bins dismissed as thin. Using
the wrong clock is not a small error here; it moved the whole claim out of the frame.

Two ways of asking, because they fail differently:
  POOLED      every message by local hour. Simple, but a few very long days dominate.
  PER-DAY     each day normalised against its own morning, then averaged. Controls for
              day-to-day variation, which is what pooling cannot do - a fortnight of
              crises would otherwise look like a daily rhythm.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
OUT = ROOT / "docs" / "teams_day_curve.md"

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Toronto")

    def local(dt):
        return dt.astimezone(TZ)
    TZNOTE = "America/Toronto via zoneinfo"
except Exception:                       # Windows without tzdata - approximate DST
    def local(dt):
        off = 4 if 3 <= dt.month <= 10 else 5
        return dt - timedelta(hours=off)
    TZNOTE = "fixed UTC-4 (Mar-Oct) / UTC-5, zoneinfo unavailable"


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    for r in rows:
        try:
            dt = datetime.fromisoformat((r.get("when") or "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            lt = local(dt)
            r["hour"], r["date"] = lt.hour, lt.date().isoformat()
        except Exception:
            r["hour"] = r["date"] = None

    me = [r for r in rows if r["speaker"] == "[SELF]" and r["hour"] is not None]
    peer = [r for r in rows if r["speaker"] == "[PEER]" and r["hour"] is not None]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# The within-day curve\n\n")
        f.write(f"{len(me)} `[SELF]` messages with a timestamp. Local time: {TZNOTE}.\n\n")
        f.write("Testing four claims made before the data was looked at: cadence slows, "
                "replies thin out, sentences shorten, and the break sits near 15:00 local.\n\n")
        f.write("The earlier hour-of-day check binned raw UTC and found nothing. Quebec is "
                "UTC-4/-5, so 15:00 local was landing in the 19:00-20:00 bins that were "
                "dismissed as too thin to read.\n\n")

        f.write("## Pooled, by local hour\n\n")
        f.write("| hour | msgs | median words | mean words | median gap (s) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for h in range(24):
            sub = [r for r in me if r["hour"] == h]
            if len(sub) < 15:
                continue
            w = [r.get("words", 0) for r in sub]
            g = [r["gap_s"] for r in sub if isinstance(r.get("gap_s"), (int, float))]
            f.write(f"| {h:02d} | {len(sub)} | {st.median(w):.0f} | {st.mean(w):.1f} | "
                    f"{(st.median(g) if g else float('nan')):.0f} |\n")

        f.write("\n## Before vs after 15:00 local\n\n")
        early = [r for r in me if r["hour"] is not None and 8 <= r["hour"] < 15]
        late = [r for r in me if r["hour"] is not None and 15 <= r["hour"] < 23]
        for label, sub in (("08:00-15:00", early), ("15:00-23:00", late)):
            w = [r.get("words", 0) for r in sub]
            g = [r["gap_s"] for r in sub if isinstance(r.get("gap_s"), (int, float))]
            short = sum(1 for r in sub if r.get("words", 0) <= 5) / max(len(sub), 1)
            f.write(f"- **{label}** - n={len(sub)}, median words "
                    f"**{st.median(w):.0f}**, mean **{st.mean(w):.1f}**, median gap "
                    f"**{(st.median(g) if g else float('nan')):.0f}s**, "
                    f"1-5 word messages **{short:.1%}**\n")

        f.write("\n## Per-day, normalised against each day's own morning\n\n")
        f.write("Pooling lets a handful of very long days set the shape. This asks the "
                "question inside each day and then averages, so a fortnight of crises "
                "cannot masquerade as a daily rhythm.\n\n")
        byday = defaultdict(list)
        for r in me:
            byday[r["date"]].append(r)
        ratios, gaps, counts = [], [], []
        for day, msgs in byday.items():
            am = [r.get("words", 0) for r in msgs if 8 <= r["hour"] < 13]
            pm = [r.get("words", 0) for r in msgs if 15 <= r["hour"] < 20]
            if len(am) >= 5 and len(pm) >= 5:
                ratios.append(st.mean(pm) / max(st.mean(am), 1e-9))
                counts.append(len(pm) / max(len(am), 1e-9))
                ga = [r["gap_s"] for r in msgs if 8 <= r["hour"] < 13
                      and isinstance(r.get("gap_s"), (int, float))]
                gp = [r["gap_s"] for r in msgs if 15 <= r["hour"] < 20
                      and isinstance(r.get("gap_s"), (int, float))]
                if ga and gp:
                    gaps.append(st.median(gp) / max(st.median(ga), 1e-9))
        if ratios:
            below = sum(1 for x in ratios if x < 1)
            f.write(f"{len(ratios)} days had 5+ messages in both windows.\n\n")
            f.write(f"- **words**: afternoon / morning = median **{st.median(ratios):.3f}** "
                    f"- shorter on **{below}/{len(ratios)}** days ({below / len(ratios):.0%})\n")
            if gaps:
                gb = sum(1 for x in gaps if x > 1)
                f.write(f"- **gap**: afternoon / morning = median **{st.median(gaps):.3f}** "
                        f"- slower on **{gb}/{len(gaps)}** days ({gb / len(gaps):.0%})\n")
            cb = sum(1 for x in counts if x < 1)
            f.write(f"- **volume**: afternoon / morning = median **{st.median(counts):.3f}** "
                    f"- fewer on **{cb}/{len(counts)}** days ({cb / len(counts):.0%})\n")
            f.write("\nA ratio of 1.000 means no within-day change at all. Below 1 for words "
                    "and volume, above 1 for gap, is the predicted shape.\n")

        f.write("\n## Control: the other writer\n\n")
        f.write("If this is a fatigue curve rather than a property of the workday itself, "
                "the two writers should not have to share it.\n\n")
        for label, sub in (("[SELF]", me), ("[PEER]", peer)):
            e = [r.get("words", 0) for r in sub if 8 <= r["hour"] < 15]
            l = [r.get("words", 0) for r in sub if 15 <= r["hour"] < 23]
            if len(e) >= 30 and len(l) >= 30:
                f.write(f"- **{label}** - median words {st.median(e):.0f} before 15:00, "
                        f"{st.median(l):.0f} after ({len(e)} / {len(l)} msgs)\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
