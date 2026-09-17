"""emotions.jsonl as the emotion family's source, in place of GoEmotions.

Wired: STANDALONE (data-building only).

WHY SWAP. The emotion family has been the worst-scoring task in every stand-in read -
v5 0.450, v6 0.625, v7 0.675, v9t 0.600 - while chat, content, affect and every game
family sit at 0.95-1.0. Two reasons, and this corpus removes both.

  THE MAPPING. GoEmotions has 27 labels + neutral and none of them are Plutchik petals,
  so every row had to be MAPPED IN: petal x tier, 8 x 3 = 24 distinguishable points for
  28 labels. By construction it cannot separate them - {disappointment, embarrassment,
  remorse, sadness} share one point, so do {excitement, love, pride}. An entire panel
  question was spent on how to fix that.

  THE TRANSLATION. The French half was machine-translated GoEmotions. Facts survive
  translation; affect does not - intensity ladders do not align, and `ecoeure` inverts
  polarity in Quebec usage.

`emotions.jsonl` (owner's own, 1,830 rows) is already in the target taxonomy:

    {"text": "...", "intent": "share_news",
     "plutchik": {"primary": "joy", "intensity": 0.9, "secondary": "optimism"},
     "tone": "ecstatic"}

Petal given, not derived. Intensity CONTINUOUS (0.2-1.0, 14 distinct values), so there
is nothing to quantise and the 24-for-28 collision does not exist. `secondary` is the
dyad partner the margin mechanism already produces. All eight petals present including
anticipation (140 rows).

WHAT THIS DOES. Loads, normalises, and VALIDATES against the corrected cube: every
(primary, secondary) pair must be cube-ADJACENT, because a dyad midpoint is only a
representable point when its two corners differ in exactly one axis - between corners
differing in two or three, the midpoint sits equidistant from four or more and the
classifier reads it as something else entirely. Two panel models raised that
independently and it is a fact about the geometry, not an opinion.

Rows that fail are REPORTED, not silently dropped: a corpus that quietly discards what
it cannot represent is how you end up believing a mapping is cleaner than it is.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402
from goemotions_cube import PETAL_CORNER, NO_CORNER  # noqa: E402

SRC = pathlib.Path(r"I:\grillcheese_training_data\pre\emotions.jsonl")
OUT = ROOT / "standin" / "data" / "out" / "emotions_clean.jsonl"
REPORT = ROOT / "docs" / "emotions_corpus.md"

PETALS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")

# Plutchik dyad names that appear in `secondary` where a PETAL was expected. The field
# is doing two jobs in the source - sometimes a second petal ("joy", "anger"), sometimes
# the compound's own name ("remorse", "optimism", "awe") - so the compound names are
# resolved back to the petal pair they stand for. Plutchik's primary dyads, closed set.
DYAD_TO_PAIR = {
    "love": ("joy", "trust"), "submission": ("trust", "fear"),
    "awe": ("fear", "surprise"), "disapproval": ("surprise", "sadness"),
    "remorse": ("sadness", "disgust"), "contempt": ("disgust", "anger"),
    "aggressiveness": ("anger", "anticipation"), "optimism": ("anticipation", "joy"),
    "guilt": ("joy", "fear"), "curiosity": ("trust", "surprise"),
    "despair": ("fear", "sadness"), "envy": ("sadness", "anger"),
    "cynicism": ("disgust", "anticipation"), "pride": ("anger", "joy"),
    "hope": ("anticipation", "fear"), "delight": ("joy", "surprise"),
    "sentimentality": ("trust", "sadness"), "shame": ("fear", "disgust"),
    "outrage": ("surprise", "anger"), "pessimism": ("sadness", "anticipation"),
    "morbidness": ("disgust", "joy"), "dominance": ("anger", "trust"),
    "anxiety": ("anticipation", "fear"), "gratitude": ("joy", "trust"),
    "excitement": ("anticipation", "joy"), "grief": ("sadness", "sadness"),
    "annoyance": ("anger", "anger"), "regret": ("sadness", "disgust"),
    "admiration": ("trust", "trust"), "disappointment": ("surprise", "sadness"),
}


def adjacent(a: str, b: str) -> bool:
    ca, cb = PETAL_CORNER.get(a), PETAL_CORNER.get(b)
    if a == NO_CORNER or b == NO_CORNER or not ca or not cb:
        return None                               # not checkable — trust has no corner
    va, vb = Neurochemistry._CORNERS[ca], Neurochemistry._CORNERS[cb]
    return sum(x != y for x, y in zip(va, vb)) == 1


def resolve_secondary(sec, primary):
    """-> (petal, how). The field mixes petals and dyad names; both resolve to a petal."""
    if not sec:
        return None, "none"
    if isinstance(sec, list):
        sec = sec[0] if sec else None
        if not sec:
            return None, "none"
    s = str(sec).strip().lower()
    if s in PETALS:
        return s, "petal"
    pair = DYAD_TO_PAIR.get(s)
    if pair:
        other = pair[1] if pair[0] == primary else pair[0]
        return (None, "dyad-is-primary") if other == primary else (other, f"dyad:{s}")
    return None, f"unknown:{s}"


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"source not found: {SRC}")
    raw = [json.loads(l) for l in open(SRC, encoding="utf-8")]

    kept, rejected = [], []
    prim, how, tiers, adj = Counter(), Counter(), Counter(), Counter()
    for i, r in enumerate(raw):
        p = (r.get("plutchik") or {})
        primary = p.get("primary")
        if isinstance(primary, list):
            rejected.append((i, "multi-primary", primary))
            continue
        primary = str(primary or "").strip().lower()
        if primary not in PETALS:
            rejected.append((i, "primary not a petal", primary))
            continue
        inten = p.get("intensity")
        if not isinstance(inten, (int, float)) or not 0 < inten <= 1:
            rejected.append((i, "bad intensity", inten))
            continue
        sec, method = resolve_secondary(p.get("secondary"), primary)
        how[method] += 1
        if sec and sec not in PETALS:
            sec = None
        a = adjacent(primary, sec) if sec else None
        if a is False:
            adj["non-adjacent — dropped to primary only"] += 1
            sec = None
        elif a is True:
            adj["adjacent — kept as a dyad"] += 1
        elif sec:
            adj["not checkable (trust has no corner)"] += 1
        prim[primary] += 1
        tiers[round(float(inten), 1)] += 1
        kept.append({"text": r.get("text", ""), "intent": r.get("intent"),
                     "tone": r.get("tone"), "primary": primary,
                     "intensity": round(float(inten), 3), "secondary": sec,
                     "corner": PETAL_CORNER.get(primary),
                     "placed": primary != NO_CORNER})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# emotions.jsonl as the emotion family's source\n\n")
        f.write(f"{len(raw)} rows in, **{len(kept)} kept**, {len(rejected)} rejected.\n\n")
        f.write("Replaces GoEmotions, which had to be MAPPED into the taxonomy — 8 petals "
                "x 3 tiers = 24 points for 28 labels, so it could not separate them by "
                "construction — and whose French half was a machine translation. This "
                "corpus arrives in the target taxonomy with continuous intensity, so "
                "there is nothing to quantise and the collision does not exist.\n\n")

        f.write("## Petals\n\n| petal | n | share | corner |\n|---|---:|---:|---|\n")
        tot = sum(prim.values()) or 1
        for p, n in prim.most_common():
            c = PETAL_CORNER.get(p) or "**none (oxytocin)**"
            f.write(f"| {p} | {n} | {n/tot:.1%} | {c} |\n")

        f.write("\n## Intensity is continuous\n\n")
        f.write(", ".join(f"{k}: {v}" for k, v in sorted(tiers.items())) + "\n\n")
        f.write(f"{len(tiers)} distinct values, not three tiers. That single fact is what "
                f"dissolves the 24-points-for-28-labels problem rather than solving it.\n\n")

        f.write("## The `secondary` field was doing two jobs\n\n")
        f.write("Sometimes a second petal (`joy`, `anger`), sometimes the compound's own "
                "name (`remorse`, `optimism`, `awe`). Both resolve to a petal via "
                "Plutchik's closed dyad table.\n\n")
        for k, v in how.most_common(14):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Dyad adjacency check\n\n")
        f.write("A dyad midpoint is only a representable point when its two corners differ "
                "in exactly ONE axis. Between corners differing in two or three it sits "
                "equidistant from four or more and the classifier reads it as something "
                "else. Non-adjacent pairs keep the primary and drop the secondary rather "
                "than being thrown away.\n\n")
        for k, v in adj.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Rejected\n\n")
        rc = Counter(r[1] for r in rejected)
        for k, v in rc.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\nReported rather than silently dropped — a corpus that quietly discards "
                "what it cannot represent is how you end up believing a mapping is cleaner "
                "than it is.\n")

    print(f"wrote {OUT}  ({len(kept)} rows)")
    print(f"wrote {REPORT}")
    for p, n in prim.most_common():
        print(f"  {p:<13} {n:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
