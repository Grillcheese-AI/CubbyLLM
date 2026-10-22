"""standin/data/chat_sft_v6.py -- what live code still reads from the v5/v6 chat builder (order 0030).

The builder `build_chat_sft.py` was rewritten for one instruction-following source (b647ebc, 2026-09-24); the v6
builder it replaced (b647ebc^, 1,596 lines: chat, content, era, history, affect, emotion families) is in the history.
These are its pieces that other code still imports, kept word for word:
- `MULTI_PROMPT`: the two-exchange prompt the chat cortex renders (`standin/chat.py`);
- `GOEMOTIONS`, `GOEMOTIONS_PETAL`, `GOEMOTIONS_FR`: the 28 labels and their Plutchik petals (`goemotions_cube.py`,
  `notebooks/_build_feel.py`);
- `history_ok`, `affect_ok` (with `proper_nouns`, `parse_affect`): the emitter eval's checks (`eval_emitter_vm.py`).
`build_chat_sft` re-exports them, so their importers are unchanged.

Wired: STANDALONE (stand-in tooling; re-exported by build_chat_sft.py; nothing in cubbyllm/ imports it).
"""
from __future__ import annotations

import re

from validation.sft.identity import is_model_guard, voice_ok

__wiring__ = "STANDALONE"

MULTI_PROMPT = {"en": "Conversation so far:\nUser: {u1}\nCubby: {a1}\nUser: {u2}",
                "fr": "Conversation jusqu'ici :\nUtilisateur : {u1}\nCubby : {a1}\nUtilisateur : {u2}"}


GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]


GOEMOTIONS_PETAL = {"admiration": "trust", "amusement": "joy", "anger": "anger", "annoyance": "anger",
                    "approval": "trust", "caring": "trust", "confusion": "surprise", "curiosity": "anticipation",
                    "desire": "anticipation", "disappointment": "sadness", "disapproval": "disgust", "disgust": "disgust",
                    "embarrassment": "sadness", "excitement": "joy", "fear": "fear", "gratitude": "trust",
                    "grief": "sadness", "joy": "joy", "love": "joy", "nervousness": "fear", "optimism": "anticipation",
                    "pride": "joy", "realization": "surprise", "relief": "joy", "remorse": "sadness",
                    "sadness": "sadness", "surprise": "surprise", "neutral": "calm"}


GOEMOTIONS_FR = dict(zip(GOEMOTIONS, [
    "admiration", "amusement", "colère", "agacement", "approbation", "sollicitude", "confusion", "curiosité",
    "désir", "déception", "désapprobation", "dégoût", "embarras", "excitation", "peur", "gratitude", "chagrin",
    "joie", "amour", "nervosité", "optimisme", "fierté", "prise de conscience", "soulagement", "remords",
    "tristesse", "surprise", "neutre"]))


_NOUN_STOP = {"the", "this", "that", "these", "those", "during", "after", "before", "when", "while", "however",
              "although", "despite", "also", "with", "from", "into", "over", "under", "about", "their", "they",
              "there", "then", "than", "some", "many", "most", "more", "such", "which", "what", "where", "other",
              "first", "later", "early", "both", "each", "between", "around", "following", "established", "regarding",
              "certainly", "indeed", "well", "yes", "absolutely", "great", "good", "here", "his", "her", "its",
              "several", "another", "though", "because", "since", "until", "through", "without", "within"}


def proper_nouns(text: str, cap: int = 12) -> list[str]:
    """Capitalised words (4+ letters, not sentence-opening stopwords) and
    years — what an answer must mention to count as being about the same
    thing (the eval's containment check)."""
    out: list[str] = []
    for m in re.finditer(r"\b([A-Z][A-Za-z'\-]{3,}|\d{3,4}(?: BCE)?)\b", text):
        w = m.group(1)
        if w.lower() in _NOUN_STOP or w in out:
            continue
        out.append(w)
        if len(out) >= cap:
            break
    return out


def history_ok(rec: dict, gen: str, facts: dict) -> bool:
    """The eval's read of a history answer: Cubby's rules hold, and the
    answer is about the right thing — `when`: the gold year is named;
    `dating`: the first year named is within 5 years of the gold; the rest:
    one of the record's proper nouns / years appears."""
    if not gen or not voice_ok(gen, facts) or is_model_guard(gen):
        return False
    sub = rec.get("subtype")
    gold = str(rec.get("gold") or "")
    if sub == "dating":
        m = re.search(r"\b(\d{4})\b", gen)
        return bool(m) and gold[:4].isdigit() and abs(int(m.group(1)) - int(gold[:4])) <= 5
    if sub == "when":
        return re.sub(r"\s*BCE?$", "", gold) in gen if gold else False
    low = gen.lower()
    keys = rec.get("gold_any") or proper_nouns(str(rec.get("program") or ""))   # replay files carry the reference, not gold_any
    return any(str(g).lower() in low for g in keys)


def parse_affect(gen: str) -> tuple[float, float] | None:
    m = re.search(r"valence\s*[:=]?\s*([+-]?\d*\.?\d+)", gen or "", re.I)
    n = re.search(r"arousal\s*[:=]?\s*([+-]?\d*\.?\d+)", gen or "", re.I)
    if m and n:
        return float(m.group(1)), float(n.group(1))
    nums = re.findall(r"[+-]?\d*\.?\d+", gen or "")
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def affect_ok(rec: dict, gen: str, tol: float = 0.35) -> bool:
    """Both numbers within `tol` of the rated ones."""
    p = parse_affect(gen)
    if p is None:
        return False
    ga = rec.get("gold_any") or parse_affect(str(rec.get("gold") or rec.get("program") or ""))   # replay files carry the reference
    v, a = (list(ga) if ga else [None, None])[:2]
    return v is not None and abs(p[0] - v) <= tol and abs(p[1] - a) <= tol
