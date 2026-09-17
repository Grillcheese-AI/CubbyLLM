"""Typographic intensity: is the marking in the text a better scale than the model's?

Owner's hypothesis: intensity goes with exclamation marks and emoji. That is worth
testing precisely because it is MEASURABLE - if it holds, intensity stops being a
number a model asserts and becomes a number the writer actually produced.

One asymmetry to be honest about up front. `gap_s` and `burst` were held-out: the
labeller never saw them, so agreement with them was evidence. Exclamation marks and
emoji are IN THE TEXT, so the labeller did see them. That makes this a different test,
and it has three possible outcomes, all informative:

  labels track typography strongly   the model is reading the marking, which means
                                     intensity is a proxy for punctuation - and we can
                                     compute punctuation for free, with no model at all.
  labels ignore typography           the model is ignoring the most visible intensity
                                     signal in the text. Strong evidence the labels are
                                     decorative, on top of the flat-median result.
  typography beats the labels on the HELD-OUT channel (burst)
                                     the decisive one. If a formula computed from the
                                     text separates on burst better than the model's
                                     number did (+0.029, p=0.036), the formula is the
                                     better scale and the model is only needed for the
                                     petal.

Markers counted, and why each:
  !        the direct one. Runs weigh more than the total: "!!!" is not three "!".
  emoji    the register carries real affect here - 8.9% of messages have one.
  ?? ?!    doubled or mixed terminals mark disbelief and pressure.
  CAPS     shouting. Noisy in THIS corpus because SKUs and acronyms (GDJ, SEO, QTY,
           BARBIE) are capitalised, so it is scored separately and reported apart
           rather than being quietly folded into one number.
  streeetch  elongation - "yessss", "loll", "nooon" - a deliberate, unambiguous marker.
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
OUT = ROOT / "docs" / "teams_typographic_intensity.md"

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿←-⇿️⬀-⯿]")
BANG_RUN = re.compile(r"!+")
QMARK_RUN = re.compile(r"\?{2,}|\?!|!\?")
ELONG = re.compile(r"([a-zA-ZàâçéèêëîïôûùüÿñæœÀÂÇÉÈÊËÎÏÔÛÙÜŸÑÆŒ])\1{2,}")
CAPWORD = re.compile(r"\b[A-ZÀ-Þ]{3,}\b")


def typo(text: str) -> dict:
    """Per-message marking counts, plus a combined score.

    The score is deliberately simple and hand-set rather than fitted: fitting weights
    to the model's own labels would make the comparison circular - it would measure how
    well typography predicts the labels, when the question is whether typography is a
    BETTER scale than the labels."""
    bangs = BANG_RUN.findall(text)
    n_bang = sum(len(b) for b in bangs)
    runs = sum(1 for b in bangs if len(b) >= 2)
    emo = len(EMOJI.findall(text))
    qm = len(QMARK_RUN.findall(text))
    el = len(ELONG.findall(text))
    caps = len(CAPWORD.findall(text))
    # saturating, so one message with fifteen emoji does not dominate the corpus
    raw = 1.0 * min(n_bang, 4) + 1.0 * min(runs, 2) + 1.2 * min(emo, 3) \
        + 0.8 * min(qm, 2) + 1.0 * min(el, 2)
    return {"bang": n_bang, "bang_runs": runs, "emoji": emo, "qmark": qm,
            "elong": el, "caps": caps,
            "score": 1 - math.exp(-raw / 3.0)}          # 0..1, saturating


def spearman(xs: list[float], ys: list[float]) -> float:
    """Pearson on ranks. No dependency, handles ties by average rank."""
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


def perm_p(a: list[float], b: list[float], n: int = 20000, seed: int = 7) -> float:
    if not a or not b:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng = a + b, 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def main() -> int:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    for r in rows:
        r["typo"] = typo(r["text"])
    live = [r for r in rows if r.get("primary") != "neutral"
            and isinstance(r.get("intensity"), (int, float))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Typographic intensity vs the model's intensity\n\n")
        f.write(f"{len(rows)} labelled rows, {len(live)} non-neutral. No spend - computed "
                f"from text already on disk.\n\n")

        marked = [r for r in rows if r["typo"]["score"] > 0.01]
        f.write(f"**{len(marked)} of {len(rows)} messages carry any marking at all "
                f"({len(marked) / len(rows):.1%}).**\n\n")
        f.write("| marker | messages | rate |\n|---|---:|---:|\n")
        for k, label in (("bang", "any `!`"), ("bang_runs", "`!!` or longer"),
                         ("emoji", "emoji"), ("qmark", "`??` / `?!`"),
                         ("elong", "streeetched word"), ("caps", "ALL-CAPS word")):
            c = sum(1 for r in rows if r["typo"][k])
            f.write(f"| {label} | {c} | {c / len(rows):.1%} |\n")

        f.write("\n## 1. Do the model's labels track the marking?\n\n")
        f.write("The labeller saw this text, so agreement here is NOT independent "
                "evidence - it only says whether the model is reading the most obvious "
                "signal available to it.\n\n")
        rho = spearman([r["intensity"] for r in live], [r["typo"]["score"] for r in live])
        f.write(f"Spearman rho (labelled intensity vs typographic score) = **{rho:+.3f}**\n\n")
        for k in ("bang", "emoji", "caps"):
            rk = spearman([r["intensity"] for r in live], [float(r["typo"][k]) for r in live])
            f.write(f"- vs `{k}` alone: **{rk:+.3f}**\n")

        f.write("\n## 2. The decisive test: which scale separates on BURST?\n\n")
        f.write("`burst` is the one channel the labeller never saw that separated at all "
                "(model intensity: +0.029, p=0.036). Whichever scale separates it better "
                "is the better measure of arousal.\n\n")
        deep = [r for r in live if r.get("burst", 0) >= 2]
        first = [r for r in live if r.get("burst", 0) == 0]
        for name, key in (("model intensity", lambda r: r["intensity"]),
                          ("typographic score", lambda r: r["typo"]["score"])):
            a, b = [key(r) for r in deep], [key(r) for r in first]
            if len(a) < 10 or len(b) < 10:
                continue
            d = st.mean(a) - st.mean(b)
            p = perm_p(a, b)
            f.write(f"**{name}** - burst>=2 mean {st.mean(a):.3f} (n={len(a)}) vs "
                    f"burst==0 mean {st.mean(b):.3f} (n={len(b)}), "
                    f"diff **{d:+.3f}**, p = **{p:.4f}**\n\n")

        f.write("## 3. Spread\n\n")
        f.write("The pilot's giveaway was a median of 0.350 in every single group - a "
                "model anchoring on the middle. A usable scale has to actually spread.\n\n")
        for name, vals in (("model intensity", [r["intensity"] for r in live]),
                           ("typographic score", [r["typo"]["score"] for r in live])):
            f.write(f"- **{name}**: median {st.median(vals):.3f}, "
                    f"sd {st.pstdev(vals):.3f}, "
                    f"range {min(vals):.2f}-{max(vals):.2f}, "
                    f"distinct values {len(set(round(v, 2) for v in vals))}\n")

        f.write("\n## 4. The most marked messages\n\n")
        for r in sorted(rows, key=lambda r: -r["typo"]["score"])[:15]:
            t = r["typo"]
            f.write(f"- `{t['score']:.2f}` ({r.get('primary')}/{r.get('intensity')}) "
                    f"{r['speaker']}: {r['text'][:110]}\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
