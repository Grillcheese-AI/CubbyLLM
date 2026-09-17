"""Does the labelled intensity track arousal, or is it decoration?

The pilot's headline check came back flat: mean intensity 0.342 for messages sent
within 30s of the previous one vs 0.344 for messages sent later. A difference of 0.002
across ~500 messages. The labeller never saw the clock, so if intensity meant anything
about arousal that comparison should have separated.

Two explanations, needing opposite fixes:

  (a) the intensity number encodes nothing - the exact failure this whole stack exists
      to avoid, and a reason to ground intensity differently before spending on a full
      run;
  (b) TIMING IS NOT AROUSAL IN WRITTEN CHAT. Fast replies in this corpus are mostly
      "ok", "yep", "oui" - genuinely low-intensity, and numerous enough to swamp the
      signal. The pilot excluded `neutral` but did not exclude short acknowledgements
      that happened to get a non-neutral label.

So this re-runs the comparison on the rows already labelled (no new spend, nothing on
the network) under several stratifications, with a PERMUTATION TEST rather than a bare
difference of means - a gap of 0.002 and a gap of 0.02 both look like "not much" until
you know what the null distribution looks like.

It also tries the other plausible arousal proxies the corpus carries and the labeller
never saw: position within a burst, and burst length. If none of them separate, (a) is
the answer and the intensity axis needs a different grounding.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "standin" / "data" / "out" / "teams_labelled.jsonl"
OUT = ROOT / "docs" / "teams_intensity_check.md"

ACK = {"ok", "oui", "yep", "yes", "non", "dac", "ouais", "parfait", "super", "merci",
       "bon", "ouaip", "correct", "exact", "cool", "nice", "ah", "oki", "yep!", "ok!"}


def perm_p(a: list[float], b: list[float], n: int = 20000, seed: int = 7) -> float:
    """Two-sided permutation test on the difference of means. No dependencies, no
    normality assumption, and it answers the only question that matters: how often
    does chance alone produce a gap this big?"""
    if not a or not b:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool = a + b
    k, rng = 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def report(f, title: str, a: list[float], b: list[float], la: str, lb: str) -> None:
    if len(a) < 10 or len(b) < 10:
        f.write(f"### {title}\n\nnot enough rows ({len(a)} / {len(b)})\n\n")
        return
    d = st.mean(a) - st.mean(b)
    p = perm_p(a, b)
    verdict = "**separates**" if p < 0.05 else "flat"
    f.write(f"### {title}\n\n")
    f.write(f"| group | n | mean | median |\n|---|---:|---:|---:|\n")
    f.write(f"| {la} | {len(a)} | {st.mean(a):.3f} | {st.median(a):.3f} |\n")
    f.write(f"| {lb} | {len(b)} | {st.mean(b):.3f} | {st.median(b):.3f} |\n\n")
    f.write(f"difference **{d:+.3f}**, permutation p = **{p:.4f}** -> {verdict}\n\n")


def main() -> int:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    live = [r for r in rows if r.get("primary") != "neutral"
            and isinstance(r.get("intensity"), (int, float))]

    def gap_split(sub, cut=30):
        a = [r["intensity"] for r in sub
             if isinstance(r.get("gap_s"), (int, float)) and r["gap_s"] <= cut]
        b = [r["intensity"] for r in sub
             if isinstance(r.get("gap_s"), (int, float)) and r["gap_s"] > cut]
        return a, b

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Does labelled intensity track arousal?\n\n")
        f.write(f"{len(rows)} labelled rows, {len(live)} non-neutral. No new spend - this "
                f"re-reads the pilot's own output. Permutation test, 20,000 shuffles, "
                f"two-sided.\n\n")
        f.write("The labeller never saw `gap_s` or `burst`. Every comparison below is "
                "therefore a check against a channel it could not have copied.\n\n")

        f.write("## 1. The original check, reproduced\n\n")
        a, b = gap_split(live)
        report(f, "all non-neutral messages", a, b, "within 30s", "after 30s")

        f.write("## 2. Short acknowledgements removed\n\n")
        f.write("The hypothesis for the flat result: fast replies here are mostly `ok` / "
                "`oui` / `yep`, genuinely low-intensity and numerous enough to swamp the "
                "signal.\n\n")
        for w in (3, 5, 10):
            sub = [r for r in live if r.get("words", 0) > w]
            a, b = gap_split(sub)
            report(f, f"messages longer than {w} words", a, b, "within 30s", "after 30s")

        sub = [r for r in live if r["text"].strip().strip("!.?").lower() not in ACK]
        a, b = gap_split(sub)
        report(f, "bare acknowledgements removed by wordlist", a, b, "within 30s", "after 30s")

        f.write("## 3. Other arousal proxies the labeller never saw\n\n")
        a = [r["intensity"] for r in live if r.get("burst", 0) >= 2]
        b = [r["intensity"] for r in live if r.get("burst", 0) == 0]
        report(f, "deep in a burst vs first of a run", a, b,
               "burst >= 2", "burst == 0")

        a = [r["intensity"] for r in live
             if isinstance(r.get("gap_s"), (int, float)) and r["gap_s"] <= 10]
        b = [r["intensity"] for r in live
             if isinstance(r.get("gap_s"), (int, float)) and r["gap_s"] > 300]
        report(f, "very fast (<=10s) vs long pause (>5 min)", a, b, "<=10s", ">300s")

        a = [r["intensity"] for r in live if r.get("reactions")]
        b = [r["intensity"] for r in live if not r.get("reactions")]
        report(f, "drew a reaction vs did not", a, b, "reacted to", "no reaction")

        f.write("## 4. What the petals do, not just the intensity\n\n")
        f.write("Intensity might be decoration while the PETAL still tracks something. "
                "Share of high-arousal petals (anger, fear, surprise, anticipation) by "
                "timing:\n\n")
        HIGH = {"anger", "fear", "surprise", "anticipation"}
        for label, sel in (("within 30s", lambda r: r.get("gap_s", 1e9) <= 30),
                           ("after 30s", lambda r: r.get("gap_s", -1) > 30)):
            sub = [r for r in live if isinstance(r.get("gap_s"), (int, float)) and sel(r)]
            n = len(sub) or 1
            hi = sum(1 for r in sub if r["primary"] in HIGH)
            f.write(f"- {label}: **{hi / n:.1%}** high-arousal ({hi}/{n})\n")

        f.write("\n## Reading this\n\n")
        f.write("If section 2 separates and section 1 does not, timing works as a check "
                "and short acknowledgements were the noise - the full run can proceed.\n\n")
        f.write("If everything is flat, the intensity number is not carrying arousal. That "
                "is a real finding and it costs one pilot rather than a full run: intensity "
                "would then need grounding in something measured rather than asserted, and "
                "the corpus already holds candidates the labeller cannot see.\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
