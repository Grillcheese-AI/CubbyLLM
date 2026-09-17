"""Pull a stratified sample of the owner's own messages for HAND labelling.

Fifty labels from the person who wrote the messages are worth more than a thousand
from a model, for one reason: the model can only read the text, and the owner can
remember the day. "oui" is not recoverable from its characters. It is recoverable from
whoever typed it.

This exists because two of the owner's own claims survived checking and the third
broke the pipeline's assumption:

  - bursts run hot ("very intense for like 10 messages in a row") - CONFIRMED,
    depth>=5 vs depth==0 is +0.117 at p=0.0006, and the first bucketing hid it.
  - emoji are the warmth channel, not arousal - emoji rate falls monotonically to
    zero as burst depth rises.
  - 1-5 word replies mean "sick of it, busy or tired" - and the labeller called 44%
    of them `neutral` and read `oui`/`oki`/`okidoo` as TRUST. If the owner is right,
    the label is not merely imprecise, it has the wrong sign, and the depletion corner
    (Lövheim all-low) is where that training data should have gone.

BLIND ON PURPOSE. The model's guess is not shown. Anchoring a human annotator to a
machine label is how you measure agreement with a mistake and call it validation. The
guesses are kept in a separate key file and scored afterwards.

Strata, and what each one is for:
  SHORT    1-5 words - the class in dispute, the whole reason this sample exists.
  LONG     6+ words, a control: if the label set behaves differently here, the short
           finding is about length rather than about state.
  BURST    deep in a run - checks the arousal result against the person who was in it.

Three preceding messages come with each item, because the owner said it himself: "ok"
after five messages about a crash is not "ok" at the start of a day.
"""
from __future__ import annotations

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
LABELLED = ROOT / "standin" / "data" / "out" / "teams_labelled.jsonl"
OUT = ROOT / "standin" / "data" / "out" / "goldset_todo.txt"
KEY = ROOT / "standin" / "data" / "out" / "goldset_key.json"

LEGEND = """\
==========================================================================
  HAND LABELS - 50 of your own messages
==========================================================================

Put one letter after LABEL: on each item. Add a few words after NOTE: if
you want; leave it blank if you don't.

    T   tanne / sick of it
    B   busy - nothing wrong, no time
    F   fatigue / tired, running out
    N   neutral - genuinely just agreeing, no charge either way
    +   positive - content, pleased, proud
    !   wound up - intense, on a roll, fired up (good OR bad)
    ?   confused, thrown, surprised
    -   down, discouraged, defeated

If two fit, put both (e.g. "T!"). If none fit, write what does.

The three lines above each item are what came just before it, for context.
`+Ns` is the seconds since the previous message; `b2` means it was the 3rd
message you sent in a row without waiting.

No model guesses are shown - that is deliberate. Seeing one would pull your
answer toward it, and then the agreement number would be measuring nothing.
==========================================================================

"""


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    for i, r in enumerate(rows):
        r["_i"] = i
    guesses = {}
    if LABELLED.exists():
        for l in open(LABELLED, encoding="utf-8"):
            o = json.loads(l)
            guesses[(o.get("when"), o.get("text"))] = (o.get("primary"), o.get("intensity"))

    me = [r for r in rows if r.get("speaker") == "[SELF]"]
    short = [r for r in me if 1 <= r.get("words", 0) <= 5]
    long_ = [r for r in me if r.get("words", 0) >= 6]
    burst = [r for r in me if r.get("burst", 0) >= 3]

    rng = random.Random(11)
    pick = (rng.sample(short, min(28, len(short)))
            + rng.sample(long_, min(12, len(long_)))
            + rng.sample(burst, min(10, len(burst))))
    seen, items = set(), []
    for r in pick:
        if r["_i"] in seen:
            continue
        seen.add(r["_i"])
        items.append(r)
    rng.shuffle(items)          # strata interleaved, so the task doesn't telegraph itself

    key = []
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(LEGEND)
        for n, r in enumerate(items, 1):
            ctx = rows[max(0, r["_i"] - 3):r["_i"]]
            f.write(f"\n--- {n:02d} " + "-" * 58 + "\n")
            for c in ctx:
                f.write(f"    {c['speaker']}: {c['text'][:100]}\n")
            g = r.get("gap_s")
            f.write(f"  > YOU (+{g}s b{r.get('burst', 0)}, {r.get('words')} words): "
                    f"{r['text'][:200]}\n")
            if r.get("reactions"):
                f.write(f"    [{', '.join(r['reactions'])} from them]\n")
            f.write("\n  LABEL: \n  NOTE:  \n")
            key.append({"n": n, "i": r["_i"], "when": r.get("when"),
                        "words": r.get("words"), "burst": r.get("burst", 0),
                        "gap_s": g, "text": r["text"],
                        "stratum": ("short" if r.get("words", 0) <= 5
                                    else "burst" if r.get("burst", 0) >= 3 else "long"),
                        "model_guess": guesses.get((r.get("when"), r["text"]))})

    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    n_g = sum(1 for k in key if k["model_guess"])
    print(f"wrote {OUT}  ({len(items)} items, {n_g} with a model guess to score against)")
    print(f"wrote {KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
