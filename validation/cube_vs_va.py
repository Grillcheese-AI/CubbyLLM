"""Is the cube monoamine geometry, or is it vocabulary?

Every check so far has been internal. `validate()` runs derived coordinates back through
the agent's own nearest-corner function and asks whether they land where the derivation
intended - which they must, because the classifier and the coordinate share one geometry.
One of the panel models said so plainly: that proves self-consistency and nothing else.

This is the external one, and the data for it was already on disk.

    emotions.jsonl          1,726 unique texts, each with a Plutchik petal
    amygdala_affect.jsonl   3,180 texts, each with valence and arousal, NO taxonomy

Every one of the 1,726 appears in the other file. Two independent labellings of the same
sentences: the petal labeller never saw valence/arousal, and the valence/arousal
labeller never saw a petal, let alone a monoamine.

THE PREDICTION, if the cube is geometry rather than a naming convention:

    AROUSAL  tracks the NE coordinate. Noradrenaline IS the arousal axis - that is not
             an interpretation of Lövheim, it is what the axis means.
    VALENCE  tracks serotonin and dopamine. 5-HT is satiety and contentment, DA is
             reward. A corner high on both should read positive.

If those hold, the corners are positions in an affect space that something outside this
project can also see. If they do not, the cube is a vocabulary for sorting words and the
monoamine story is decoration on top of it.

AND IT SETTLES THE PARKED QUESTION. Lövheim puts fear/terror at LOW noradrenaline, which
is why a ghost catch now reads as distress and why two tests are widened pending a
decision. If fear-labelled texts come back with HIGH corpus arousal, that placement is
wrong for this data and the threat->DA coupling is not the thing to change. If they come
back LOW, Lövheim is right and the old test was named after the old wrong table.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402
from goemotions_cube import PETAL_CORNER  # noqa: E402

SRC = pathlib.Path(r"I:\grillcheese_training_data\pre")
OUT = ROOT / "docs" / "cube_vs_valence_arousal.md"


def norm(t) -> str:
    return " ".join(str(t or "").lower().split())


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
    emo = [json.loads(l) for l in open(SRC / "emotions.jsonl", encoding="utf-8")]
    amy = [json.loads(l) for l in open(SRC / "amygdala_affect.jsonl", encoding="utf-8")]
    # PLACEHOLDERS, not labels. 1,845 of the 3,180 amygdala rows carry valence exactly
    # 0.0 and arousal exactly 0.5 - the neutral defaults of whatever appended them, not
    # a judgement anyone made. They are also exactly the rows that overlap
    # `emotions.jsonl`, so a naive join reports 1,546 matches and every single measured
    # value is identical, which is what produced `rho = nan` on the first run.
    #
    # Excluding them is the difference between a test and a mirage.
    PLACEHOLDER = (0.0, 0.5)
    va, skipped = {}, 0
    for r in amy:
        k = norm(r.get("text"))
        if not k or not isinstance(r.get("valence"), (int, float)):
            continue
        pair = (float(r["valence"]), float(r.get("arousal", 0)))
        if pair == PLACEHOLDER:
            skipped += 1
            continue
        va.setdefault(k, pair)
    print(f"  amygdala rows usable: {len(va)}  (skipped {skipped} at the "
          f"{PLACEHOLDER} default)")

    rows = []
    for r in emo:
        p = r.get("plutchik") or {}
        petal = p.get("primary")
        if isinstance(petal, list) or not isinstance(petal, str):
            continue
        petal = petal.strip().lower()
        corner = PETAL_CORNER.get(petal)
        if not corner:
            continue                                  # trust: no corner, by design
        hit = va.get(norm(r.get("text")))
        if not hit:
            continue
        s, d, n = Neurochemistry._CORNERS[corner]
        rows.append({"petal": petal, "corner": corner, "s": s, "d": d, "n": n,
                     "valence": hit[0], "arousal": hit[1],
                     "intensity": p.get("intensity")})

    if len(rows) < 100:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            "# The external check is not available from these two files\n\n"
            f"Joined rows with BOTH a Plutchik petal and a real valence/arousal pair: "
            f"**{len(rows)}**.\n\n"
            "The overlap looked total - every one of the 1,726 unique `emotions.jsonl` "
            "texts appears in `amygdala_affect.jsonl`. It is the wrong 1,726.\n\n"
            "`amygdala_affect.jsonl` is two files stacked. About 1,335 rows carry real "
            "judgements; 1,845 carry valence exactly `0.0` and arousal exactly `0.5`, "
            "which is a default rather than a reading. The placeholder half is precisely "
            "the half that overlaps `emotions.jsonl` - those texts were appended without "
            "affect labels and defaulted to neutral.\n\n"
            "So the two halves are disjoint in exactly the way that matters:\n\n"
            "- real valence/arousal + `realm`/`phase`, NO petal "
            "(~1,335 rows, the same texts as `emotion_valence_arousal_realm_phase.jsonl`, "
            "which shares 0 texts with `emotions.jsonl`)\n"
            "- petal + intensity, NO real valence/arousal (~1,726 rows)\n\n"
            "No text has both, so nothing here can test whether the cube predicts an "
            "affect space labelled independently of it.\n\n"
            "## What would work\n\n"
            "Label the ~1,260 texts that DO have real valence/arousal with a petal. Their "
            "affect values already exist and were not produced by the petal labeller, so "
            "agreement would still be evidence rather than an echo - weaker than a "
            "pre-existing double-labelling, stronger than anything internal. It is also "
            "cheap: one pass over 1,260 short texts.\n\n"
            "The first run of this script reported 1,546 joined rows and `rho = nan`, "
            "because every measured value in the join was the same number. A test that "
            "cannot fail is not a test, and a join that succeeds on placeholders is not "
            "a join.\n", encoding="utf-8")
        print(f"wrote {OUT}")
        print(f"  only {len(rows)} rows have BOTH a petal and a real valence/arousal")
        print("  the overlap between the two files is entirely placeholder rows")
        return 0

    ne = [r["n"] for r in rows]
    sd = [(r["s"] + r["d"]) / 2 for r in rows]
    aro = [r["arousal"] for r in rows]
    val = [r["valence"] for r in rows]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Is the cube geometry, or vocabulary?\n\n")
        f.write(f"**{len(rows)} texts** carrying both a Plutchik petal (`emotions.jsonl`) "
                f"and a valence/arousal pair (`amygdala_affect.jsonl`). Two independent "
                f"labellings of the same sentences - neither labeller saw the other's "
                f"output, and neither saw a monoamine.\n\n")
        f.write("Every previous check here was internal: derived coordinates run back "
                "through the agent's own classifier, which can only prove "
                "self-consistency, because the classifier and the coordinate share one "
                "geometry. This is the first check from outside.\n\n")

        f.write("## The two predictions\n\n")
        r_ne = spearman(ne, aro)
        r_sd = spearman(sd, val)
        f.write(f"| prediction | Spearman rho | verdict |\n|---|---:|---|\n")
        f.write(f"| arousal tracks the **NE** coordinate | **{r_ne:+.3f}** | "
                f"{'**holds**' if r_ne > 0.2 else 'fails' if r_ne < 0.1 else 'weak'} |\n")
        f.write(f"| valence tracks **(5-HT + DA)/2** | **{r_sd:+.3f}** | "
                f"{'**holds**' if r_sd > 0.2 else 'fails' if r_sd < 0.1 else 'weak'} |\n")
        f.write(f"\nCross-checks, which should be WEAKER if the axes mean what they say:\n\n")
        f.write(f"- NE vs valence: {spearman(ne, val):+.3f}\n")
        f.write(f"- (5-HT+DA)/2 vs arousal: {spearman(sd, aro):+.3f}\n")
        f.write(f"- 5-HT alone vs valence: {spearman([r['s'] for r in rows], val):+.3f}\n")
        f.write(f"- DA alone vs valence: {spearman([r['d'] for r in rows], val):+.3f}\n")

        f.write("\n## Per corner: predicted vs measured\n\n")
        f.write("| corner | (5HT,DA,NE) | n | corpus valence | corpus arousal |\n")
        f.write("|---|---|---:|---:|---:|\n")
        by = {}
        for r in rows:
            by.setdefault(r["corner"], []).append(r)
        for c, rs in sorted(by.items(), key=lambda kv: -st.mean([r["arousal"] for r in kv[1]])):
            f.write(f"| {c} | {Neurochemistry._CORNERS[c]} | {len(rs)} | "
                    f"{st.mean([r['valence'] for r in rs]):+.3f} | "
                    f"{st.mean([r['arousal'] for r in rs]):.3f} |\n")

        f.write("\n## The parked question: where does fear actually sit?\n\n")
        f.write("Lövheim places fear/terror at LOW noradrenaline, which is why a ghost "
                "catch now reads as distress and why two tests are widened pending a "
                "decision.\n\n")
        hi_ne = [r for r in rows if r["n"] == 1]
        lo_ne = [r for r in rows if r["n"] == 0]
        f.write(f"- corners Lövheim marks HIGH NE: mean corpus arousal "
                f"**{st.mean([r['arousal'] for r in hi_ne]):.3f}** (n={len(hi_ne)})\n")
        f.write(f"- corners Lövheim marks LOW NE:  mean corpus arousal "
                f"**{st.mean([r['arousal'] for r in lo_ne]):.3f}** (n={len(lo_ne)})\n\n")
        fear = by.get("fear", [])
        if fear:
            f.write(f"**`fear` itself** ({len(fear)} texts): mean corpus arousal "
                    f"**{st.mean([r['arousal'] for r in fear]):.3f}**, against a "
                    f"corpus-wide mean of {st.mean(aro):.3f}.\n\n")
            f.write("If fear reads HIGH-arousal here, Lövheim's low-NE placement does not "
                    "describe this data, and the threat->DA coupling is not the thing to "
                    "change. If it reads LOW, the old test was named after the old wrong "
                    "table and the current behaviour is correct.\n")

    print(f"wrote {OUT}  ({len(rows)} joined)")
    print(f"  arousal ~ NE          rho {r_ne:+.3f}")
    print(f"  valence ~ (5HT+DA)/2  rho {r_sd:+.3f}")
    for c, rs in sorted(by.items(), key=lambda kv: -st.mean([r["arousal"] for r in kv[1]])):
        print(f"  {c:<9} {Neurochemistry._CORNERS[c]}  n={len(rs):<4} "
              f"val {st.mean([r['valence'] for r in rs]):+.2f}  "
              f"aro {st.mean([r['arousal'] for r in rs]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
