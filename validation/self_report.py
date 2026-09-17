"""Self-report: the channel that was sitting there the whole time.

Five inferred channels were tested against this corpus. Emoji turned out to be the
social register (falls to zero as bursts deepen). Exclamation marks barely exist
(1.3%). Reply timing was flat (p=0.83). Burst depth held, weakly (+0.117). Compression
is pinned at the ceiling for this writer - he bottomed out years ago and has nothing
left to shorten.

And then the owner, mid-session: *"like in this moment im dead tired lol that gives you
a cue :)"*.

He SAID it. No inference, no model, no ambiguity. The declarative channel was never
considered, and it is the only one that is exact.

Why this matters beyond the joke: a self-report is FREE GROUND TRUTH. Every other
channel has been checked against a proxy (the clock, the burst counter) or against a
model's guess. A message where someone states their own state is an annotation written
by the only person who can see it, at the moment it was true. Find those, and the
window around each one becomes a labelled sample - which is what the whole hand-label
exercise was trying to manufacture.

The catch, and it is real: self-reports are rare and they are not random. People say
"chu brule" when it is worth mentioning, which biases toward the extremes and toward
whatever is socially sayable to that particular person. So this is an anchor to
calibrate the other channels against, not a corpus to train on directly.

Note the `lol` on "dead tired lol" - softening the admission while making it. That is
the discharge pattern this corpus has been showing all along, now visible in a
self-report about being depleted.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
OUT = ROOT / "docs" / "teams_self_report.md"

# Grouped by the state being reported, in the register this corpus actually uses.
# First person only - "t'as l'air fatigue" is a read of somebody else, not a report.
STATES = {
    "depleted": r"\b(chu|jsuis|je suis|chui)\s+(brul[ée]|mort|crev[ée]|fatigu[ée]|"
                r"[ée]puis[ée]|vid[ée]|au bout|[aà] boute|dead|done)\b"
                r"|\bj'?en peux p(?:u|lus)\b|\bjen peux pu\b|\bjme meurs\b"
                r"|\bjdormirai\b|\bpas dormi\b|\bjai pas dormi\b",
    "fed_up":   r"\b(chu|jsuis|je suis|chui)\s+(tann[ée]|[ée]c(?:o|oe|œ)ur[ée]|"
                r"fatigu[ée] de|sick)\b|\bjen ai (?:mon voyage|assez|marre)\b"
                r"|\bjte dis\b.{0,20}\btann[ée]\b",
    "stressed": r"\bjcapote\b|\bje capote\b|\bca me stresse\b|\bjstresse\b"
                r"|\b(chu|jsuis|je suis)\s+(stress[ée]|nerveu|anxieu)\w*\b"
                r"|\bjai peur\b|\bca minqui[eè]te\b|\bjminqui[eè]te\b",
    "pleased":  r"\b(chu|jsuis|je suis)\s+(content|fier|heureu)\w*\b|\bjtripe\b"
                r"|\bje tripe\b|\bjadore\b|\bjai h[aâ]te\b|\bjsuis dedans\b",
    "confused": r"\bje comprends? (?:pas|rien)\b|\bjcomprends pas\b|\bchu m[eê]l[ée]\b"
                r"|\bje sais pas trop\b|\bjsais pas\b.{0,12}\?",
    "busy":     r"\b(chu|jsuis|je suis)\s+(occup[ée]|dans|dessus|en train)\b"
                r"|\bjai pas le temps\b|\bpas le temps\b|\bjsuis encore dessus\b",
}


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    for i, r in enumerate(rows):
        r["_i"] = i
        r["states"] = [s for s, pat in STATES.items() if re.search(pat, r["text"], re.I)]

    hits = [r for r in rows if r["states"]]
    by_state = Counter(s for r in hits for s in r["states"])
    by_speaker = Counter(r["speaker"] for r in hits)

    SOFT = re.compile(r"\b(lol+|haha+|mdr)\b|[\U0001F300-\U0001FAFF]", re.I)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Self-reports: free ground truth\n\n")
        f.write(f"**{len(hits)} of {len(rows)} messages ({len(hits) / len(rows):.1%}) state "
                f"the writer's own condition outright.** No model, no annotation - the only "
                f"person who can see the state wrote it down at the moment it was true.\n\n")
        f.write("Every other channel tested here was inferred and checked against a proxy. "
                "This one is declared. It is rare and it is biased - people say it when it is "
                "worth saying - so it is an anchor to calibrate the inferred channels "
                "against, not a corpus to train on.\n\n")

        f.write("## What gets reported\n\n| state | n | by speaker |\n|---|---:|---|\n")
        for s, n in by_state.most_common():
            who = Counter(r["speaker"] for r in hits if s in r["states"])
            f.write(f"| {s} | {n} | {', '.join(f'{k} {v}' for k, v in who.most_common())} |\n")
        f.write(f"\nOverall by speaker: "
                f"{', '.join(f'{k} {v}' for k, v in by_speaker.most_common())}\n\n")

        soft = sum(1 for r in hits if SOFT.search(r["text"]))
        f.write(f"## Softened while said\n\n**{soft} of {len(hits)} ({soft / max(len(hits), 1):.0%})** "
                f"carry a `lol`/`haha`/emoji in the same message. Against a corpus baseline of "
                f"6.4% for `lol` and 8.9% for emoji.\n\n")
        f.write("The owner did it again in chat while pointing this channel out - *\"im dead "
                "tired lol\"*. Admitting the state and defusing it in the same breath is the "
                "discharge pattern, and it shows up most where the admission costs something.\n\n")

        f.write("## How the inferred channels look around a self-report\n\n")
        f.write("If the failed channels carry anything at all, they should move in the "
                "messages AROUND a declared state - the five before and after.\n\n")
        dep = [r for r in hits if "depleted" in r["states"] or "fed_up" in r["states"]]
        near = set()
        for r in dep:
            near.update(range(max(0, r["_i"] - 5), min(len(rows), r["_i"] + 6)))
        a = [rows[i] for i in near if rows[i]["speaker"] == "[SELF]"]
        b = [r for r in rows if r["speaker"] == "[SELF]" and r["_i"] not in near]
        if len(a) >= 20:
            f.write(f"| | near a depleted/fed-up report (n={len(a)}) | elsewhere (n={len(b)}) |\n")
            f.write("|---|---:|---:|\n")
            f.write(f"| median words | {st.median([r.get('words', 0) for r in a]):.0f} | "
                    f"{st.median([r.get('words', 0) for r in b]):.0f} |\n")
            for label, key in (("mean burst", lambda r: r.get("burst", 0)),):
                f.write(f"| {label} | {st.mean([key(r) for r in a]):.2f} | "
                        f"{st.mean([key(r) for r in b]):.2f} |\n")
            ga = [r["gap_s"] for r in a if isinstance(r.get("gap_s"), (int, float))]
            gb = [r["gap_s"] for r in b if isinstance(r.get("gap_s"), (int, float))]
            if ga and gb:
                f.write(f"| median gap (s) | {st.median(ga):.0f} | {st.median(gb):.0f} |\n")

        f.write("\n## The reports themselves\n\n")
        for s in by_state:
            f.write(f"\n**{s}**\n\n")
            for r in [x for x in hits if s in x["states"]][:8]:
                f.write(f"- `{r['speaker']}` {r['text'][:130]}\n")

    print(f"wrote {OUT}  ({len(hits)} self-reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
