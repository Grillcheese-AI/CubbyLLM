"""Does intensity climb with burst DEPTH, or was bucketing at >=2 hiding it?

Owner, unprompted and before seeing this: *"knowing myself when I do bursts I am very
intense for like 10 messages in a row"*. That is a prediction made about behaviour, not
about the data, which makes it the best kind of check available here - it can fail.

The first pass compared `burst >= 2` against `burst == 0` and got +0.029 (p=0.036). If
intensity climbs with depth, that bucket is exactly wrong: it averages the second
message of a two-message run together with the tenth of a ten-message run, and a
gradient flattened into two buckets looks like a weak effect.

Two different things get measured here, and they are not the same:

  DEPTH        how far into a run this message is (0, 1, 2, ... ). Tests "he winds up".
  RUN LENGTH   how long the whole run turned out to be, attached to every message in
               it. Tests "some runs are hot from the first word" - a message that is
               3rd of 10 is a different animal from 3rd of 3, and depth alone cannot
               tell them apart.

Also re-checks the typographic score against depth. It separated on burst in the
OPPOSITE direction (-0.047, p=0.0018): deeper in a run, less marking. If that holds
across depth it is not noise, it means emoji are social punctuation, not arousal -
nobody stops mid-rant to add a smiley.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "standin" / "data" / "out" / "teams_labelled.jsonl"
OUT = ROOT / "docs" / "teams_burst_depth.md"

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


def perm_p(a, b, n=20000, seed=7):
    if not a or not b:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng = list(a) + list(b), 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    a, b = rank(xs), rank(ys)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def main() -> int:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

    # Attach run length: walk forward, a run breaks when burst resets to 0.
    runs, cur = [], []
    for r in rows:
        if r.get("burst", 0) == 0 and cur:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    for run in runs:
        for r in run:
            r["run_len"] = len(run)

    live = [r for r in rows if r.get("primary") != "neutral"
            and isinstance(r.get("intensity"), (int, float))]

    def bucket(r):
        d = r.get("burst", 0)
        return "0" if d == 0 else "1" if d == 1 else "2-4" if d <= 4 else "5-9" if d <= 9 else "10+"

    ORDER = ["0", "1", "2-4", "5-9", "10+"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Burst depth vs intensity\n\n")
        f.write(f"{len(rows)} labelled rows, {len(live)} non-neutral, {len(runs)} runs. "
                f"No spend - re-reads the pilot's own output.\n\n")
        f.write("Testing a prediction the owner made before seeing the data: bursts run "
                "*\"very intense for like 10 messages in a row\"*. The first pass bucketed "
                "`burst>=2` against `burst==0`, which averages the 2nd message of a "
                "two-message run with the 10th of a ten-message run - if intensity climbs "
                "with depth, that bucket is the wrong shape.\n\n")

        f.write("## How deep do runs actually go?\n\n")
        lens = [len(r) for r in runs]
        f.write(f"- runs: **{len(runs)}**, median length **{st.median(lens):.0f}**, "
                f"longest **{max(lens)}**\n")
        for lo, hi, label in ((1, 1, "single message"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 999, "10+")):
            c = sum(1 for L in lens if lo <= L <= hi)
            f.write(f"- runs of {label}: **{c}** ({c / len(runs):.1%})\n")

        f.write("\n## 1. Intensity by burst depth\n\n")
        f.write("| depth | n | mean intensity | median |\n|---|---:|---:|---:|\n")
        for b in ORDER:
            vals = [r["intensity"] for r in live if bucket(r) == b]
            if vals:
                f.write(f"| {b} | {len(vals)} | {st.mean(vals):.3f} | {st.median(vals):.3f} |\n")
        deep = [r["intensity"] for r in live if r.get("burst", 0) >= 5]
        shallow = [r["intensity"] for r in live if r.get("burst", 0) == 0]
        if len(deep) >= 10:
            f.write(f"\ndepth>=5 vs depth==0: **{st.mean(deep) - st.mean(shallow):+.3f}**, "
                    f"p = **{perm_p(deep, shallow):.4f}**\n")
        else:
            f.write(f"\nOnly {len(deep)} non-neutral messages at depth>=5 - too few to test. "
                    f"That is itself the finding: 40 windows of a 9,738-message corpus do not "
                    f"contain enough deep bursts, and the prediction needs the full run to check.\n")
        rho = spearman([r["intensity"] for r in live], [float(r.get("burst", 0)) for r in live])
        f.write(f"\nSpearman rho (intensity vs depth) = **{rho:+.3f}**\n")

        f.write("\n## 2. Intensity by RUN LENGTH\n\n")
        f.write("A message 3rd of 10 is not a message 3rd of 3. This asks whether long "
                "runs are hot throughout, rather than whether people wind up.\n\n")
        f.write("| run length | n | mean intensity |\n|---|---:|---:|\n")
        for lo, hi, label in ((1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 999, "10+")):
            vals = [r["intensity"] for r in live if lo <= r.get("run_len", 1) <= hi]
            if vals:
                f.write(f"| {label} | {len(vals)} | {st.mean(vals):.3f} |\n")
        a = [r["intensity"] for r in live if r.get("run_len", 1) >= 5]
        b = [r["intensity"] for r in live if r.get("run_len", 1) == 1]
        if len(a) >= 10 and len(b) >= 10:
            f.write(f"\nrun>=5 vs run==1: **{st.mean(a) - st.mean(b):+.3f}**, "
                    f"p = **{perm_p(a, b):.4f}**\n")
        rho2 = spearman([r["intensity"] for r in live], [float(r.get("run_len", 1)) for r in live])
        f.write(f"\nSpearman rho (intensity vs run length) = **{rho2:+.3f}**\n")

        f.write("\n## 3. Emoji by depth - social marker or arousal marker?\n\n")
        f.write("| depth | n | % with emoji |\n|---|---:|---:|\n")
        for b in ORDER:
            sub = [r for r in rows if bucket(r) == b]
            if sub:
                e = sum(1 for r in sub if EMOJI.search(r["text"]))
                f.write(f"| {b} | {len(sub)} | {e / len(sub):.1%} |\n")
        f.write("\nIf emoji thin out as runs deepen, they are punctuation for the social "
                "beats of a conversation - greetings, thanks, softening - and not a measure "
                "of how worked up anyone is. Nobody stops mid-rant to add a smiley.\n")

        f.write("\n## 4. The deepest runs, as text\n\n")
        for run in sorted(runs, key=lambda r: -len(r))[:3]:
            f.write(f"\n**run of {len(run)}** ({run[0]['speaker']})\n\n")
            for r in run[:12]:
                f.write(f"- `b{r.get('burst')}` `{r.get('primary')}/{r.get('intensity')}` "
                        f"{r['text'][:95]}\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
