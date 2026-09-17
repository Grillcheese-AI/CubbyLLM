"""Does a late report night predict a depleted next day?

Owner: *"when hour 21 is my longest its usually to make a report because I either did
overtime or planning to work all night then the next day I will probably start my day
tired as hell lol"*.

Every check so far has been within a message or within a day. This one crosses the
sleep boundary, which makes it the only test here that bears on the architectural
premise directly: hormones INTEGRATE AND DECAY, they do not reset at midnight. If a
long 21:00 report predicts a thinner next day, that is carryover showing up in
behaviour - state that outlives the episode that produced it.

A per-turn model has nowhere to put that. It sees tomorrow's first message with no
memory of last night, and reads terseness as calm.

The marker: a long message sent at 20:00 local or later. The owner says those are
reports written after overtime or before an all-nighter, so they mark the night rather
than describing it. What the next working day should look like, from the day-curve
result (fatigue here shows as withdrawal, not compression):

  VOLUME    fewer messages - the variable that actually moved
  CADENCE   longer gaps
  START     a later first message, if the night ran long
  LENGTH    little or no change - words per message did not track fatigue within a day
            either, and predicting it here would contradict the day-curve finding

Weekends and gaps are skipped by walking to the next day that HAS activity, not the
next calendar date. Comparison is against days following an ordinary evening.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
OUT = ROOT / "docs" / "teams_carryover.md"

LATE_HOUR = 20
LONG_WORDS = 15


def local(dt):
    off = 4 if 3 <= dt.month <= 10 else 5      # Quebec, no tzdata on this machine
    return dt - timedelta(hours=off)


def perm_p(a, b, n=20000, seed=7):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng = list(a) + list(b), 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    me = []
    for r in rows:
        try:
            dt = datetime.fromisoformat((r.get("when") or "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        lt = local(dt)
        r["hour"], r["date"] = lt.hour, lt.date()
        if r["speaker"] == "[SELF]":
            me.append(r)

    # SESSIONS, NOT CALENDAR DATES. The first run keyed on local date and produced the
    # opposite of the prediction - 105 messages the "next day" against 40, starting at
    # 07:00 against 10:00. Then the per-night listing gave it away: those next days
    # started at 00:00 and 02:00. That is not an early start, it is the same work
    # session continuing past midnight. The owner said "planning to work all night";
    # a calendar boundary cuts the all-nighter in half and calls the second half the
    # morning after.
    #
    # So a session breaks on a real gap in activity, not at midnight. The recovery day
    # is the session AFTER the one containing the late report.
    me.sort(key=lambda r: r["when"])
    SESSION_BREAK = 6 * 3600
    sessions, cur, prev = [], [], None
    for r in me:
        t = datetime.fromisoformat(r["when"].replace("Z", "+00:00"))
        if prev is not None and (t - prev).total_seconds() > SESSION_BREAK and cur:
            sessions.append(cur)
            cur = []
        cur.append(r)
        prev = t
    if cur:
        sessions.append(cur)
    byday = {i: s for i, s in enumerate(sessions)}
    days = sorted(byday)

    def stats(d):
        msgs = byday[d]
        g = [r["gap_s"] for r in msgs if isinstance(r.get("gap_s"), (int, float))]
        return {
            "n": len(msgs),
            "words": st.median([r.get("words", 0) for r in msgs]),
            "gap": st.median(g) if g else float("nan"),
            "start": min(r["hour"] for r in msgs),
            "short": sum(1 for r in msgs if r.get("words", 0) <= 5) / len(msgs),
        }

    late_nights, normal = [], []
    for i, d in enumerate(days[:-1]):
        nxt = days[i + 1]
        last = datetime.fromisoformat(byday[d][-1]["when"].replace("Z", "+00:00"))
        first = datetime.fromisoformat(byday[nxt][0]["when"].replace("Z", "+00:00"))
        if (first - last).total_seconds() > 4 * 86400:   # a week off is not "next"
            continue
        msgs = byday[d]
        is_late = any(r["hour"] >= LATE_HOUR and r.get("words", 0) >= LONG_WORDS
                      for r in msgs)
        (late_nights if is_late else normal).append((d, nxt))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Carryover: does a late report predict a thin next day?\n\n")
        f.write(f"{len(me)} `[SELF]` messages across **{len(days)} active days**. A day "
                f"counts as a report night if it contains a message at {LATE_HOUR}:00 local "
                f"or later of {LONG_WORDS}+ words.\n\n")
        f.write(f"**{len(late_nights)} report nights**, **{len(normal)} ordinary evenings**, "
                f"each paired with its next active day.\n\n")
        f.write("This is the only test here that crosses a night, and so the only one that "
                "bears on the premise the whole architecture rests on: state integrates and "
                "decays rather than resetting. A per-turn model reads tomorrow's terseness "
                "as calm because it has nowhere to keep last night.\n\n")

        if len(late_nights) < 3:
            f.write(f"Only {len(late_nights)} report nights in the corpus - too few to test. "
                    f"That is the finding: the claim is about a pattern that this export does "
                    f"not contain enough of, and no amount of analysis manufactures the "
                    f"missing days.\n")
            print("wrote", OUT, "- too few late nights", len(late_nights))
            return 0

        f.write("## The morning after\n\n")
        f.write("| metric | after a report night | after an ordinary evening | p |\n")
        f.write("|---|---:|---:|---:|\n")
        A = [stats(n) for _, n in late_nights]
        B = [stats(n) for _, n in normal]
        for key, label, fmt in (("n", "messages that day", "{:.0f}"),
                                ("gap", "median gap (s)", "{:.0f}"),
                                ("start", "first message (local hour)", "{:.1f}"),
                                ("words", "median words", "{:.1f}"),
                                ("short", "share 1-5 words", "{:.1%}")):
            a = [x[key] for x in A if x[key] == x[key]]
            b = [x[key] for x in B if x[key] == x[key]]
            if not a or not b:
                continue
            f.write(f"| {label} | {fmt.format(st.mean(a))} | {fmt.format(st.mean(b))} | "
                    f"{perm_p(a, b):.3f} |\n")

        f.write("\n## The report nights themselves\n\n")
        for d, nxt in late_nights[:12]:
            longest = max((r for r in byday[d] if r["hour"] >= LATE_HOUR),
                          key=lambda r: r.get("words", 0), default=None)
            s = stats(nxt)
            f.write(f"\n- **{d}** {longest['hour']:02d}:00, {longest.get('words')} words: "
                    f"_{longest['text'][:110]}_\n"
                    f"  next active day {nxt}: {s['n']} messages, "
                    f"started {s['start']:02d}:00, median gap {s['gap']:.0f}s\n")

        f.write("\n## Reading this\n\n")
        f.write("Volume and cadence are where fatigue showed up within a day - length did "
                "not move. So a real carryover effect should look the same: fewer messages "
                "and longer gaps, with word count roughly flat. Word count dropping instead "
                "would contradict the day-curve result and is more likely noise at this n.\n")

    print("wrote", OUT, f"({len(late_nights)} late nights, {len(normal)} normal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
