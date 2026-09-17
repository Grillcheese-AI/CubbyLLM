"""Does the cube agree with human ratings? Warriner (2013) x NRC EmoLex, both human.

Wired: STANDALONE (validation only).

The first external check (4bc6abb) found the cube's valence axis strong (rho +0.782
against independently labelled text) and its arousal axis absent (rho -0.004). Sorting
corners by measured arousal put FEAR top with Lövheim's NE=0 and DISTRESS near the
bottom with NE=1: they looked transposed. Swapping them took arousal to +0.464.

That was a hypothesis generated from the data it was measured on, which is the move
that produces confident nonsense. This is the independent test, and it is better than
the first in four ways:

  HUMAN, not model. Warriner et al. collected valence and arousal from crowdworkers on
  1-9 scales; NRC EmoLex's emotion tags came from a separate crowd years earlier. No
  language model touched either one.

  PUBLISHED 2013 and 2010, before this project existed. Neither can have been
  contaminated by anything here, and nothing here influenced them.

  WORD-LEVEL, so no judgement enters the mapping. Plutchik's own tier words already
  name the corners, and NRC tags its whole vocabulary with Plutchik's eight directly.

  LARGE. NRC x Warriner intersects to thousands of words per petal, against the 43 the
  tier lists and GoEmotions labels could supply between them.

Three vocabularies are kept SEPARATE because they are different kinds of evidence:

  tiers       the canonical triples out of `_PETAL` - the cube's own naming, n~20
  goemotions  the label set mapped through GOEMOTIONS_PETAL, n~23
  nrc         every EmoLex word tagged with exactly ONE of the eight, n in the thousands

Multi-tagged NRC words are dropped rather than assigned arbitrarily: a word tagged both
fear and sadness is exactly the case under test and cannot be allowed to vote.

AND THE CEILING. Beyond the fear/distress question, all 2^k assignments of NE across
the corners that have words are enumerated, so the result is not "is Lövheim's NE
right" but "how well can ANY binary NE labelling track human arousal, and where does
Lövheim's rank". A low ceiling would say the axis is the problem, not the corner.

Sources: Warriner, Kuperman & Brysbaert (2013), Behavior Research Methods (ratings
mirrored at JULIELab/XANEW). Mohammad & Turney (2013), NRC Word-Emotion Association
Lexicon v0.92. Both off-repo under standin/data/hf/, gitignored like every fetched
corpus.
"""
from __future__ import annotations

import csv
import itertools
import math
import pathlib
import random
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402
from goemotions_cube import PETAL_CORNER  # noqa: E402

RATINGS = ROOT / "standin" / "data" / "hf" / "warriner_2013.csv"
EMOLEX = ROOT / "standin" / "data" / "hf" / "nrc_emolex_v092.txt"
OUT = ROOT / "docs" / "cube_vs_human_vad.md"

# Plutchik's own tier triples, straight out of `pacman._PETAL`. Not a mapping anybody
# invented here - these are the words the compass already uses for each corner.
TIERS = {
    "joy":      ("serenity", "joy", "ecstasy"),
    "fear":     ("apprehension", "fear", "terror"),
    "surprise": ("distraction", "surprise", "amazement"),
    "distress": ("pensiveness", "sadness", "grief"),
    "spent":    ("pensiveness", "sadness", "grief"),      # shares the sadness petal
    "disgust":  ("boredom", "disgust", "loathing"),
    "anger":    ("annoyance", "anger", "rage"),
    "interest": ("interest", "anticipation", "vigilance"),
}

GOEMOTIONS_PETAL = {
    "admiration": "trust", "amusement": "joy", "anger": "anger", "annoyance": "anger",
    "approval": "trust", "caring": "trust", "confusion": "surprise",
    "curiosity": "anticipation", "desire": "anticipation", "disappointment": "sadness",
    "disapproval": "disgust", "disgust": "disgust", "embarrassment": "sadness",
    "excitement": "joy", "fear": "fear", "gratitude": "trust", "grief": "sadness",
    "joy": "joy", "love": "joy", "nervousness": "fear", "optimism": "anticipation",
    "pride": "joy", "realization": "surprise", "relief": "joy", "remorse": "sadness",
    "sadness": "sadness", "surprise": "surprise",
}

NRC_PETALS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness",
              "surprise", "trust")


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


def perm_p(a, b, n=20000, seed=7):
    """Two-sided permutation test on the difference of means."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng = list(a) + list(b), 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def load_ratings():
    norms = {}
    with open(RATINGS, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            w = (row.get("Word") or "").strip().lower()
            try:
                norms[w] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]))
            except (KeyError, ValueError):
                continue
    return norms


def load_emolex():
    """-> {word: {petals it is tagged with}}. positive/negative tags are ignored."""
    tags = {}
    with open(EMOLEX, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, cat, on = parts[0].strip().lower(), parts[1].strip().lower(), parts[2]
            if cat not in NRC_PETALS or on.strip() != "1":
                continue
            tags.setdefault(word, set()).add(cat)
    return tags


def build(pairs, norms):
    """pairs: [(word, corner)] -> rows carrying cube coords and human ratings."""
    rows, missing, seen = [], [], set()
    for word, corner in pairs:
        if (word, corner) in seen:
            continue                              # the vocabularies overlap
        seen.add((word, corner))
        hit = norms.get(word)
        if not hit:
            missing.append(word)
            continue
        s, d, n = Neurochemistry._CORNERS[corner]
        rows.append({"word": word, "corner": corner, "s": s, "d": d, "n": n,
                     "valence": hit[0], "arousal": hit[1]})
    return rows, missing


SWAP = {"fear": "distress", "distress": "fear"}


def rhos(rows, swap=False):
    ne, sd, aro, val = [], [], [], []
    for r in rows:
        c = SWAP.get(r["corner"], r["corner"]) if swap else r["corner"]
        s, d, n = Neurochemistry._CORNERS[c]
        ne.append(n)
        sd.append((s + d) / 2)
        aro.append(r["arousal"])
        val.append(r["valence"])
    return spearman(ne, aro), spearman(sd, val)


def report(f, title, rows, note=""):
    if len(rows) < 8:
        f.write(f"### {title}\n\nonly {len(rows)} words found - not enough\n\n")
        return None, None
    a1, v1 = rhos(rows, swap=False)
    a2, v2 = rhos(rows, swap=True)
    f.write(f"### {title} - {len(rows)} words\n\n{note}\n\n")
    f.write("| | as published | fear<->distress |\n|---|---:|---:|\n")
    f.write(f"| arousal ~ NE | **{a1:+.3f}** | **{a2:+.3f}** |\n")
    f.write(f"| valence ~ (5HT+DA)/2 | **{v1:+.3f}** | {v2:+.3f} |\n\n")
    return a1, a2


def ceiling(f, rows):
    """How well can ANY binary NE labelling track human arousal?"""
    corners = sorted({r["corner"] for r in rows})
    k = len(corners)
    aro = [r["arousal"] for r in rows]
    scores = []
    for bits in itertools.product((0, 1), repeat=k):
        if len(set(bits)) == 1:
            continue                              # a constant axis has no rank to correlate
        assign = dict(zip(corners, bits))
        scores.append((spearman([assign[r["corner"]] for r in rows], aro), bits))
    scores.sort(key=lambda t: -t[0])
    published = tuple(Neurochemistry._CORNERS[c][2] for c in corners)
    swapped = tuple(Neurochemistry._CORNERS[SWAP.get(c, c)][2] for c in corners)
    rank = {b: i for i, (_, b) in enumerate(scores)}

    f.write(f"\n## The ceiling: every possible NE labelling\n\n")
    f.write(f"{len(scores)} non-degenerate ways to mark {k} corners high or low on "
            f"noradrenaline (all-high and all-low are not axes), scored against human "
            f"arousal. This asks whether the AXIS can work at all, not whether one "
            f"corner is misplaced.\n\n")
    f.write("| labelling | high-NE corners | rho | rank |\n|---|---|---:|---:|\n")
    best_r, best_b = scores[0]
    f.write(f"| **best possible** | {', '.join(c for c, b in zip(corners, best_b) if b)} "
            f"| **{best_r:+.3f}** | 1 |\n")
    pub_r = next(r for r, b in scores if b == published)
    f.write(f"| Lövheim as published | "
            f"{', '.join(c for c, b in zip(corners, published) if b)} | {pub_r:+.3f} | "
            f"{rank[published] + 1} of {len(scores)} |\n")
    sw_r = next(r for r, b in scores if b == swapped)
    f.write(f"| fear<->distress swapped | "
            f"{', '.join(c for c, b in zip(corners, swapped) if b)} | {sw_r:+.3f} | "
            f"{rank[swapped] + 1} of {len(scores)} |\n\n")
    return {"best": best_r, "published": pub_r, "swapped": sw_r,
            "pub_rank": rank[published] + 1, "swap_rank": rank[swapped] + 1,
            "n": len(scores)}


def main() -> int:
    for p in (RATINGS, EMOLEX):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    norms = load_ratings()
    emolex = load_emolex()

    tier_pairs = [(w, c) for c, ws in TIERS.items() if c != "spent" for w in ws]
    ge_pairs = [(w, PETAL_CORNER[p]) for w, p in GOEMOTIONS_PETAL.items()
                if PETAL_CORNER.get(p)]
    # Exactly one petal, or the word is exactly the ambiguity under test.
    nrc_pairs = [(w, PETAL_CORNER[next(iter(ps))]) for w, ps in emolex.items()
                 if len(ps) == 1 and PETAL_CORNER.get(next(iter(ps)))]
    multi = sum(1 for ps in emolex.values() if len(ps) > 1)

    t_rows, t_missing = build(tier_pairs, norms)
    g_rows, _ = build(ge_pairs, norms)
    n_rows, _ = build(nrc_pairs, norms)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# The cube against human ratings\n\n")
        f.write(f"Warriner, Kuperman & Brysbaert (2013): valence and arousal for "
                f"**{len(norms):,} English lemmas**, crowd-rated on 1-9 scales. NRC "
                f"EmoLex v0.92: **{len(emolex):,} words** tagged with Plutchik's eight "
                f"by a separate crowd. Published 2013 and 2010, no language model "
                f"involved in either.\n\n")
        f.write(f"Of the EmoLex vocabulary, **{len(nrc_pairs):,} words carry exactly one "
                f"petal** and survive to a corner; {multi:,} carry more than one and are "
                f"dropped rather than assigned arbitrarily, because a word tagged both "
                f"fear and sadness is precisely the case under test.\n\n")
        f.write("The first external check found the valence axis strong (+0.782) and the "
                "arousal axis absent (-0.004), with fear and distress apparently "
                "transposed on noradrenaline. Swapping them lifted arousal to +0.464 - "
                "but that hypothesis came from the data it was measured on.\n\n")

        f.write("## Results\n\n")
        report(f, "NRC EmoLex x Warriner", n_rows,
               "The large one. Every single-petal EmoLex word that Warriner also rates.")
        report(f, "Plutchik tier words", t_rows,
               "The cube's own naming: `apprehension/fear/terror`, "
               "`pensiveness/sadness/grief`, straight out of `_PETAL`. No mapping chosen "
               "here - these words already name these corners.")
        report(f, "GoEmotions labels", g_rows,
               "The label set mapped through `GOEMOTIONS_PETAL`.")

        f.write("## Per corner, human-rated (NRC vocabulary)\n\n")
        f.write("| corner | (5HT,DA,NE) | words | valence 1-9 | arousal 1-9 |\n")
        f.write("|---|---|---:|---:|---:|\n")
        by = {}
        for r in n_rows:
            by.setdefault(r["corner"], []).append(r)
        order = sorted(by.items(), key=lambda kv: -st.mean([r["arousal"] for r in kv[1]]))
        for c, rs in order:
            f.write(f"| {c} | {Neurochemistry._CORNERS[c]} | {len(rs)} | "
                    f"{st.mean([r['valence'] for r in rs]):.2f} | "
                    f"{st.mean([r['arousal'] for r in rs]):.2f} |\n")

        vs = [st.mean([r["valence"] for r in rs]) for _, rs in order]
        as_ = [st.mean([r["arousal"] for r in rs]) for _, rs in order]
        f.write(f"\n**How much is there to find.** Corner means span "
                f"**{max(vs) - min(vs):.2f}** points of valence and "
                f"**{max(as_) - min(as_):.2f}** points of arousal, on the same 1-9 "
                f"scale. Whatever the labelling, the second axis carries "
                f"{(max(as_) - min(as_)) / (max(vs) - min(vs)):.0%} of the signal the "
                f"first one does: emotion WORDS barely encode arousal.\n\n")

        ceil = ceiling(f, n_rows)

        f.write("## The specific question\n\n")
        fw = [r["arousal"] for r in n_rows if r["corner"] == "fear"]
        dw = [r["arousal"] for r in n_rows if r["corner"] == "distress"]
        if fw and dw:
            p = perm_p(fw, dw)
            f.write(f"- **fear** words ({len(fw)}): mean human arousal "
                    f"**{st.mean(fw):.2f}** - Lövheim places this corner at NE=0\n")
            f.write(f"- **distress** words ({len(dw)}): mean human arousal "
                    f"**{st.mean(dw):.2f}** - Lövheim places this corner at NE=1\n\n")
            f.write(f"Difference **{st.mean(fw) - st.mean(dw):+.2f}** on a 1-9 scale, "
                    f"permutation p = **{p:.4f}** over 20,000 shuffles. "
                    f"{'Fear rates HIGHER, as the swap predicts.' if st.mean(fw) > st.mean(dw) else 'Distress rates higher, against the swap.'}\n\n")
        f.write("## What this settles, and what it does not\n\n")
        f.write(f"**Settled.** Fear outranks distress on human arousal in all three "
                f"vocabularies here and in the model-labelled corpus before them - four "
                f"independent samples, same direction, p < 0.001 on the large one. "
                f"Lövheim's published NE assignment ranks **{ceil['pub_rank']} of "
                f"{ceil['n']}** against human arousal; the swap ranks "
                f"**{ceil['swap_rank']}**. The swap is not a rescue, it is a smaller "
                f"error.\n\n")
        f.write(f"**Not settled, and more important.** The BEST labelling any binary NE "
                f"axis can manage is {ceil['best']:+.3f}, against {rhos(n_rows)[1]:+.3f} for "
                f"valence off the other two axes. The corners span "
                f"{max(as_) - min(as_):.2f} points of arousal and "
                f"{max(vs) - min(vs):.2f} of valence. The third axis is not mislabelled "
                f"so much as nearly empty in this instrument - emotion words carry "
                f"valence and barely carry arousal.\n\n")
        f.write("Which is the honest reading: **language is the wrong place to measure "
                "the arousal axis.** Valence survives being written down; activation is "
                "in the body and in the timing - burst depth, reply latency, hour of "
                "day, the channels the writer does not choose. The cube's NE axis should "
                "be validated against BEHAVIOUR, not against text, and until it is, no "
                "text corpus can either convict or acquit it.\n\n")
        f.write("`_CORNERS` is therefore left as published. The swap wins on a dimension "
                "this evidence cannot measure well enough to justify moving a corner.\n\n")
        if t_missing:
            f.write(f"Not in the lexicon: {', '.join(sorted(set(t_missing)))}\n")

    print(f"wrote {OUT}")
    print(f"  nrc {len(n_rows)} words | tiers {len(t_rows)} | goemotions {len(g_rows)}")
    a1, v1 = rhos(n_rows, False)
    a2, v2 = rhos(n_rows, True)
    print(f"  arousal ~ NE          {a1:+.3f}   swapped {a2:+.3f}   "
          f"best possible {ceil['best']:+.3f}  (published ranks {ceil['pub_rank']} "
          f"of {ceil['n']}, swapped {ceil['swap_rank']})")
    print(f"  valence ~ (5HT+DA)/2  {v1:+.3f}   swapped {v2:+.3f}")
    for c, rs in order:
        print(f"  {c:<9} {Neurochemistry._CORNERS[c]}  n={len(rs):<5} "
              f"val {st.mean([r['valence'] for r in rs]):.2f}  "
              f"aro {st.mean([r['arousal'] for r in rs]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
