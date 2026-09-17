"""Are 1-5 word replies 'neutral', or are they a state the labeller is missing?

Owner, again unprompted and before seeing any of this: *"when I answer a 1-5 word
message, im either sick of it, busy or tired"*.

That contradicts an assumption baked into the pipeline. The pilot treated short replies
as low-signal and excluded `neutral` from the arousal checks on the grounds that "ok"
and "oui" carry nothing. If the owner is right, they carry a LOT - they are the
depleted end of the range, and the labeller is reading a real state as the absence of
one. An agent trained on that reads terseness as calm, which is the opposite of true.

This matters beyond the corpus. Lövheim's all-low corner (0,0,0) - no serotonin, no
dopamine, no noradrenaline - is what two of the panel models independently argued
should be named DEPLETION or collapse rather than shame, precisely because it is the
reachable, behaviourally important state after sustained unrewarded effort. If short
replies are the corpus's depletion signal, that corner has training data, and it is
being thrown away as "neutral".

TWO HELD-OUT CHANNELS, neither of which the labeller saw:

  HOUR OF DAY   "tired" has a shape. If short replies cluster late and early rather
                than spreading evenly, that is evidence they carry fatigue rather than
                simple brevity.
  GAP BEFORE    "busy" has a shape too: a long silence, then a curt answer.

Both are in the export and neither was in the prompt, so agreement is evidence rather
than the model echoing something back.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
from collections import Counter
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "standin" / "data" / "out" / "teams_labelled.jsonl"
OUT = ROOT / "docs" / "teams_short_replies.md"


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


def hour(r):
    try:
        return datetime.fromisoformat((r.get("when") or "").replace("Z", "+00:00")).hour
    except Exception:
        return None


def main() -> int:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    self_rows = [r for r in rows if r.get("speaker") == "[SELF]"]
    short = [r for r in self_rows if 1 <= r.get("words", 0) <= 5]
    long_ = [r for r in self_rows if r.get("words", 0) > 5]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Short replies: neutral, or depleted?\n\n")
        f.write(f"{len(self_rows)} `[SELF]` messages labelled - **{len(short)}** of 1-5 "
                f"words, {len(long_)} longer. No spend.\n\n")
        f.write("Testing a claim the owner made before seeing the data: a 1-5 word reply "
                "means *sick of it, busy, or tired*. The pipeline had been treating those "
                "as low-signal.\n\n")

        f.write("## 1. What does the labeller call them?\n\n")
        f.write("| petal | short (1-5w) | longer |\n|---|---:|---:|\n")
        cs, cl = Counter(r["primary"] for r in short), Counter(r["primary"] for r in long_)
        for p in sorted(set(cs) | set(cl), key=lambda p: -cs.get(p, 0)):
            a = cs.get(p, 0) / max(len(short), 1)
            b = cl.get(p, 0) / max(len(long_), 1)
            f.write(f"| {p} | {cs.get(p, 0)} ({a:.1%}) | {cl.get(p, 0)} ({b:.1%}) |\n")
        neg = {"sadness", "anger", "disgust", "fear"}
        ns = sum(cs.get(p, 0) for p in neg) / max(len(short), 1)
        nl = sum(cl.get(p, 0) for p in neg) / max(len(long_), 1)
        f.write(f"\n- negative petals: short **{ns:.1%}** vs longer **{nl:.1%}**\n")
        f.write(f"- neutral: short **{cs.get('neutral', 0) / max(len(short), 1):.1%}** "
                f"vs longer **{cl.get('neutral', 0) / max(len(long_), 1):.1%}**\n\n")
        f.write("If short replies come back overwhelmingly `neutral` while the person who "
                "wrote them says they mean *sick of it*, the labeller is mapping a real "
                "state onto the absence of one - and `neutral` is where the depletion "
                "corner's training data went.\n\n")

        f.write("## 2. Held-out: hour of day\n\n")
        f.write("Never in the prompt. Tiredness has a shape; simple brevity does not.\n\n")
        f.write("| hour (UTC) | short | longer | short share |\n|---|---:|---:|---:|\n")
        hs = Counter(h for h in (hour(r) for r in short) if h is not None)
        hl = Counter(h for h in (hour(r) for r in long_) if h is not None)
        for h in sorted(set(hs) | set(hl)):
            tot = hs[h] + hl[h]
            f.write(f"| {h:02d} | {hs[h]} | {hl[h]} | {hs[h] / tot:.0%} |\n")

        f.write("\n## 3. Held-out: silence before the reply\n\n")
        f.write("*Busy* should look like a long gap followed by a curt answer.\n\n")
        gs = [r["gap_s"] for r in short if isinstance(r.get("gap_s"), (int, float))]
        gl = [r["gap_s"] for r in long_ if isinstance(r.get("gap_s"), (int, float))]
        if gs and gl:
            f.write(f"- before a short reply: median gap **{st.median(gs):.0f}s** (n={len(gs)})\n")
            f.write(f"- before a longer one:  median gap **{st.median(gl):.0f}s** (n={len(gl)})\n")
            capped_s = [min(g, 3600) for g in gs]
            capped_l = [min(g, 3600) for g in gl]
            f.write(f"- means capped at 1h: **{st.mean(capped_s):.0f}s** vs "
                    f"**{st.mean(capped_l):.0f}s**, p = **{perm_p(capped_s, capped_l):.4f}**\n")

        f.write("\n## 4. The short replies themselves\n\n")
        f.write("Read these against the labels. The question is not whether the label is "
                "defensible in isolation - it is whether it is what the writer meant.\n\n")
        for r in short[:40]:
            g = r.get("gap_s")
            f.write(f"- `{r['primary']}/{r['intensity']}` +{g}s b{r.get('burst')}: {r['text'][:70]}\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
