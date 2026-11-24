"""build_emotion_read_0079 - order 0079's emotion-read rows for the talk seat (Cowork's addendum, 2026-11-24; Nick: "yes").

The trunk does not read the user's emotion (`exp_0079_appraiser_read`: 193 of 1,714 right, the asked form on 9 of
2,914). These rows teach the talk seat the read `perception.ModelAppraiser` asks for, in the exact prompt the seat is
shown at serving (`talk.talk_prompt` over `perception.EMOTION_PROMPT`, the text's spaces collapsed, as `read` sends it;
the seat has no system turn) and the answer `parse_emotion` reads as it stands: the emotion word, an em dash (U+2014),
the petal.

- **emotions.jsonl** (Nick's own; `emotions_corpus.py`'s cleaned file): the petal is the row's primary; the word is the
  petal's tier by intensity (`mood.TIERS`: up to 0.4 the mild word, up to 0.8 the middle one, above it the strong one).
  A fifth of its distinct texts (seed 79) is held out for the test and never trained.
- **GoEmotions** (Apache-2.0; English, the simplified config): train rows, the first rater label as the word, its
  petal by `mood.LABEL_PETAL` (neutral -> calm). Its test split is held out for the test.
- **The mix:** emotion rows at most 15% of the talk seat's rows (talk_v3a trained on 254,231: at most 38,134).
  emotions.jsonl's train rows are repeated `--repeat` times (they are the first gate's source and only ~1,450), the
  rest is GoEmotions train drawn uniformly (seed 79), its neutral rows capped at one petal's share (a ninth of the
  draw) so calm is not the read's default. The pilot's chat rows (`--pilot`, off the repo) are written
  beside them as family `chat` so the voice is kept; a twentieth of them is held for the trainer's held loss.
- **Not used:** `mental_health_svc.jsonl` (real people's posts; licence unchecked) and machine-translated French
  GoEmotions (affect does not survive translation): there is no native French source, so a French sample (the
  multilingual file's `fr` rows) goes to the held file only, flagged.

Writes, off the repo: `--out` (the trainer's rows: id, family, split train|held, prompt, answer) with its
`.manifest.json` (counts, sources, licences), and `--held` (the test: source, lang, text, petal). Counts also to
`validation/logs/emotion_read_0079.manifest.json`.

    python validation/sft/build_emotion_read_0079.py --pilot <chat_0079_pilot_rows.jsonl>

Wiring: STANDALONE (data-building only).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.affect.mood import LABEL_PETAL, TIERS  # noqa: E402
from cubbyllm.affect.perception import EMOTION_PROMPT, EMOTION_PROMPT_FR, parse_emotion  # noqa: E402
from cubbyllm.emit.talk import talk_prompt  # noqa: E402

__wiring__ = "STANDALONE"

DASH = "—"
OUT_DIR = os.path.join(ROOT, "standin", "data", "out")
HF = os.path.join(ROOT, "standin", "data", "hf")
TALK_V3A_ROWS = 254_231
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
              "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
              "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
              "surprise", "neutral"]


def prompt_of(text: str, lang: str = "en") -> str:
    """What the talk seat is shown for `ModelAppraiser.read(text, lang)`."""
    return talk_prompt((EMOTION_PROMPT_FR if lang == "fr" else EMOTION_PROMPT).format(t=" ".join(text.split())), "talk")


def answer_of(word: str, petal: str) -> str:
    a = f"{word} {DASH} {petal}"
    assert parse_emotion(a) == (word, petal), a
    return a


def tier_word(petal: str, intensity: float) -> str:
    return TIERS[petal][0 if intensity <= 0.4 else 1 if intensity <= 0.8 else 2]


def emotions_rows(path: str, seed: int) -> tuple[list, list]:
    seen, rows = set(), []
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        t = " ".join(str(r["text"]).split())
        if r.get("primary") in TIERS and t not in seen:
            seen.add(t)
            rows.append((t, r["primary"], float(r["intensity"])))
    rng = random.Random(seed)
    held = set(rng.sample(range(len(rows)), len(rows) // 5))
    return [r for i, r in enumerate(rows) if i not in held], [r for i, r in enumerate(rows) if i in held]


def goemotions(path: str) -> list[tuple[str, str]]:
    import pandas as pd
    d = pd.read_parquet(path)
    out = []
    for text, labels in zip(d["text"], d["labels"]):
        labels = json.loads(labels) if isinstance(labels, str) else list(labels)
        if len(labels):
            out.append((" ".join(str(text).split()), GOEMOTIONS[int(labels[0])]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emotions", default=os.path.join(OUT_DIR, "emotions_clean.jsonl"))
    ap.add_argument("--goemotions-train", default=os.path.join(HF, "goemotions_train.parquet"))
    ap.add_argument("--goemotions-test", default=os.path.join(HF, "goemotions_test.parquet"))
    ap.add_argument("--goemotions-multi", default=os.path.join(HF, "goemotions_multi_train.parquet"))
    ap.add_argument("--pilot", required=True, help="the pilot's chat rows (off the repo)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "talk_emotion_0079.jsonl"))
    ap.add_argument("--held", default=os.path.join(OUT_DIR, "emotion_read_0079_held.jsonl"))
    ap.add_argument("--share", type=float, default=0.15)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--fr", type=int, default=300)
    ap.add_argument("--seed", type=int, default=79)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    cap = int(a.share * TALK_V3A_ROWS)
    e_train, e_held = emotions_rows(a.emotions, a.seed)
    g_train = goemotions(a.goemotions_train)
    g_need = max(0, cap - a.repeat * len(e_train))
    neutral = [r for r in g_train if r[1] == "neutral"]           # a third of GoEmotions: capped at one petal's share,
    others = [r for r in g_train if r[1] != "neutral"]            # so calm is not the read's default (emotions.jsonl
    n_neu = min(len(neutral), g_need // 9)                        # has no calm at all)
    g_take = rng.sample(others, min(g_need - n_neu, len(others))) + rng.sample(neutral, n_neu)
    c, rows = Counter(), []
    for k in range(a.repeat):
        for i, (t, p, x) in enumerate(e_train):
            rows.append({"id": f"emo-{i}-{k}", "family": "emotion_read", "split": "train", "prompt": prompt_of(t),
                         "answer": answer_of(tier_word(p, x), p), "source": "emotions.jsonl"})
    for i, (t, lab) in enumerate(g_take):
        rows.append({"id": f"goe-{i}", "family": "emotion_read", "split": "train", "prompt": prompt_of(t),
                     "answer": answer_of(lab, LABEL_PETAL[lab]), "source": "goemotions"})
    for i, (t, p, x) in enumerate(e_held[:60]):                 # a few for the trainer's held loss (never trained)
        rows.append({"id": f"emo-held-{i}", "family": "emotion_read", "split": "held", "prompt": prompt_of(t),
                     "answer": answer_of(tier_word(p, x), p), "source": "emotions.jsonl"})
    pilot = [json.loads(ln) for ln in open(a.pilot, encoding="utf-8")]
    held_ids = set(rng.sample(range(len(pilot)), len(pilot) // 20))
    for i, r in enumerate(pilot):
        rows.append({"id": f"chat-{r['id']}", "family": "chat", "split": "held" if i in held_ids else "train",
                     "prompt": r["prompt"], "answer": r["target"].strip(), "source": "chat_0079_pilot"})
    rng.shuffle(rows)
    for r in rows:
        c[f"{r['family']}:{r['split']}:{r.get('source')}"] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    held = [{"source": "emotions.jsonl", "lang": "en", "text": t, "petal": p} for t, p, _ in e_held]
    held += [{"source": "goemotions_test", "lang": "en", "text": t, "petal": LABEL_PETAL[lab]}
             for t, lab in goemotions(a.goemotions_test)]
    if a.fr and os.path.exists(a.goemotions_multi):
        import pandas as pd
        d = pd.read_parquet(a.goemotions_multi)
        d = d[d["language"] == "fr"]
        fr = [(" ".join(str(t).split()), GOEMOTIONS[int(json.loads(ls)[0])]) for t, ls in zip(d["text"], d["labels"])
              if json.loads(ls)]
        held += [{"source": "goemotions_fr_machine_translated", "lang": "fr", "text": t, "petal": LABEL_PETAL[lab]}
                 for t, lab in random.Random(a.seed).sample(fr, min(a.fr, len(fr)))]
    with open(a.held, "w", encoding="utf-8") as f:
        for r in held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"what": "order 0079: the talk seat's emotion-read rows and the read's held-out test (counts only; the rows "
                   "and texts are off the repo)",
           "rows": dict(sorted(c.items())), "emotion_rows_train": sum(v for k, v in c.items() if k.startswith(
               "emotion_read:train")), "cap (15% of talk_v3a's 254,231 rows)": cap,
           "emotions.jsonl": {"distinct texts": len(e_train) + len(e_held), "train": len(e_train),
                              "held": len(e_held), "repeat": a.repeat},
           "goemotions": {"train available": len(g_train), "train drawn": len(g_take), "neutral drawn": n_neu},
           "held test": dict(Counter(r["source"] for r in held)),
           "answer petals (train)": dict(Counter(r["answer"].split(DASH)[-1].strip() for r in rows
                                                 if r["family"] == "emotion_read" and r["split"] == "train")),
           "row_licences": {"emotions.jsonl": "Nick's own", "goemotions": "Apache-2.0",
                            "chat_0079_pilot": "drafted by Claude agents for 0079 (Cowork's pilot)"},
           "seed": a.seed, "not used": ["mental_health_svc.jsonl", "machine-translated French GoEmotions (test only)"]}
    for p in (a.out + ".manifest.json", os.path.join(ROOT, "validation", "logs", "emotion_read_0079.manifest.json")):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("rows", "emotion_rows_train", "held test")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
