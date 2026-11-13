"""Order 0067, the reader's sixth round (Nick, 2026-11-13: "both"): rows from long, real, human-written prompts.

Round 5's IFBench read: on a whole prompt (a task with its constraints inside) the reader says "none" or invents the
nearest count; every training row before was short. The prompts here are people's own: OpenAssistant's OASST1 first
turns (Apache-2.0; English and French; not deleted, reviewed, not synthetic) and Databricks' Dolly-15k instructions
(CC BY-SA 3.0; English; the instruction screened, its passage, when it has one, mostly put before or after it). Prompts
under 200 characters are thinned so long ones weigh more (IFBench's median is 293). Off-repo:
``$CUBBY_DATASETS/oasst1/2023-04-12_oasst_prompts.messages.jsonl.gz``, ``$CUBBY_DATASETS/dolly15k/databricks-dolly-15k.jsonl``.
Only the prompts are used; no answer from either set.

Every prompt kept carries no constraint on the reply (the screens of round 4: `exp_if_follow.rules_of`, `alpaca_rows`'
constraint words and counts, and a French counterpart), so its rule is known by construction. Each prompt is used once:
- ``real_none``: the prompt alone -> ``none``;
- ``real_rule``: a measurable rule from the training wordings put inside the prompt (after it, before it, or after its
  first sentence) -> that rule; sometimes with a tone or structure phrase too (the answer stays the rule);
- ``real_asked``: a phrase no measurement covers (round 5's tone phrases, `struct_phrases.STRUCT`) put inside the
  prompt -> ``asked``.
Held out, never trained: ``real_none_test``, ``real_rule_test`` (unseen prompts; the rule rows keep IFEval's form for
its checker), ``real_asked_test`` (unseen prompts with the held-out quarter of the structure phrases and round 5's
held-out tone phrases).

`rows(seed, measurable)` -> rows in `build_rule_rows`' shape; `measurable`: [(text, plan, lang, ifeval forms)].
"""
from __future__ import annotations

import gzip
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation"), str(Path(__file__).resolve().parent)]
import alpaca_rows as AL  # noqa: E402
import asked_rows as AR  # noqa: E402
import exp_if_follow as IF  # noqa: E402
import struct_phrases as SP  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

FR_SIGNALS = re.compile(r"\b(?:mots?|phrases?|paragraphes?|lettres?|majuscules?|minuscules?|json|liste|puces?|anglais"
                        r"|français|espagnol|allemand|langue|tradu\w*|citation|guillemets|titre|section|markdown|virgule"
                        r"|exactement|au moins|au plus|moins de|maximum|minimum|seulement|uniquement|limite|commence\w*"
                        r"|termine\w*|format|tableau|résum\w*|court|brève?|bref|long(?:ue)?)\b", re.I)
SPLIT_SHARE = {"none": 0.35, "rule": 0.45, "asked": 0.20}
TEST_N = {"en": 300, "fr": 60}
SHORT_MAX = {"en": 6000, "fr": 100000}                   # prompts under 200 characters kept at most


def _screen(text: str, lang: str) -> bool:
    if not 20 <= len(text) <= 1200 or IF.rules_of(text) or AL.LENGTHY.search(text):
        return False
    return not (AL.SIGNALS.search(text) if lang == "en" else FR_SIGNALS.search(text) or AL.SIGNALS.search(text))


def pool() -> dict:
    """{lang: [(source id, prompt)]}: the screened prompts of both sets."""
    root = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = {"en": [], "fr": []}
    oa = root / "oasst1" / "2023-04-12_oasst_prompts.messages.jsonl.gz"
    if oa.exists():
        for line in gzip.open(oa, "rt", encoding="utf-8"):
            m = json.loads(line)
            if m["lang"] in out and not m.get("deleted") and m.get("review_result") and not m.get("synthetic") \
                    and _screen(m["text"].strip(), m["lang"]):
                out[m["lang"]].append(("oasst1:" + m["message_id"], m["text"].strip()))
    dl = root / "dolly15k" / "databricks-dolly-15k.jsonl"
    if dl.exists():
        rng = random.Random(11)
        for i, line in enumerate(open(dl, encoding="utf-8")):
            d = json.loads(line)
            text = d["instruction"].strip()
            if not _screen(text, "en"):                  # the instruction is screened; its passage is reference text,
                continue                                 # not a constraint (numbers in it are not counts to keep)
            ctx = re.sub(r"\[\d+\]", "", d.get("context") or "").strip()
            if ctx and rng.random() < 0.8 and len(text) + len(ctx) < 1150:
                text = rng.choice([f"{text}\n\n{ctx}", f"{ctx}\n\n{text}"])
            out["en"].append((f"dolly:{i}", text))
    rng = random.Random(12)
    for lang, ps in out.items():                         # long prompts kept whole; short ones thinned (IFBench: median
        short = [p for p in ps if len(p[1]) < 200]       # 293 characters)
        out[lang] = [p for p in ps if len(p[1]) >= 200] + rng.sample(short, min(len(short), SHORT_MAX[lang]))
    return out


def embed(prompt: str, piece: str, rng: random.Random) -> str:
    """`piece` put after the prompt, before it, or after its first sentence."""
    where = rng.random()
    first = re.match(r"(.+?[.!?])\s+(\S.*)", prompt, re.S)
    if where < 0.2 and first:
        return f"{first.group(1)} {piece} {first.group(2)}"
    sep = rng.choice([" ", "\n", "\n\n"])
    return f"{piece}{sep}{prompt}" if where < 0.4 else f"{prompt}{sep}{piece}"


def _phrases(seed: int) -> tuple[dict, dict]:
    """({lang: training phrases}, {lang: held-out phrases}): round 5's tone phrases (its held-out keys stay held out)
    and the structure phrases (a seeded quarter of their keys held out)."""
    tone_held = AR.held_keys(seed + 3)
    keys = [k for k, _, _ in SP.STRUCT]
    struct_held = set(random.Random(seed + 6).sample(keys, len(keys) // 4))
    train, held = {"en": [], "fr": []}, {"en": [], "fr": []}
    for src, gone in ((AR.PHRASES, tone_held), (SP.STRUCT, struct_held)):
        for key, en, fr in src:
            for lang, ws in (("en", en), ("fr", fr)):
                (held if key in gone else train)[lang] += [(key, w) for w in ws]
    return train, held


def rows(seed: int = 6, measurable=()) -> list[dict]:
    rng = random.Random(seed)
    train_ph, held_ph = _phrases(seed)
    out = []
    for lang, prompts in pool().items():
        rng.shuffle(prompts)
        rules = [m for m in measurable if m[2] == lang]
        tests, rest = prompts[:3 * TEST_N[lang]], prompts[3 * TEST_N[lang]:]
        n = TEST_N[lang]
        for i, (sid, p) in enumerate(tests):            # held out: unseen prompts
            use = ("none", "rule", "asked")[i // n]
            out.append(_row(use, f"real_{use}_test", lang, sid, p, rules, held_ph[lang], rng))
        cut1 = int(len(rest) * SPLIT_SHARE["none"])
        cut2 = cut1 + int(len(rest) * SPLIT_SHARE["rule"])
        for i, (sid, p) in enumerate(rest):
            use = "none" if i < cut1 else "rule" if i < cut2 else "asked"
            out.append(_row(use, "dev" if rng.random() < 0.1 else "train", lang, sid, p, rules, train_ph[lang], rng))
    return out


def _row(use, split, lang, sid, prompt, rules, phrases, rng) -> dict:
    base = {"kind": f"real_{use}", "lang": lang, "template": None, "source_id": sid, "split": split}
    if use == "none" or (use == "rule" and not rules):
        return dict(base, kind="real_none", instruction=prompt, plan=None, asked=False, ifeval=[])
    if use == "asked":
        key, w = rng.choice(phrases)
        return dict(base, kind=f"real_asked_{key}", instruction=embed(prompt, w, rng), plan=None, asked=True, ifeval=[])
    text, plan, _, forms = rng.choice(rules)
    if rng.random() < 0.15 and phrases:                  # a phrase beside the rule: the answer is the rule alone
        text = f"{text} {rng.choice(phrases)[1]}" if rng.random() < 0.5 else f"{rng.choice(phrases)[1]} {text}"
    return dict(base, instruction=embed(prompt, text, rng), plan=plan, asked=False, ifeval=forms)


__all__ = ["embed", "pool", "rows"]
