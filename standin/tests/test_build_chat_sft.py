"""Pins for the v5 chat/content builder's filters (standin/data/build_chat_sft.py):
what is allowed to become Cubby's answer, and what the content screens drop.
No corpus access, no model, no VM. Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_chat_sft as b  # noqa: E402
import identity as idn  # noqa: E402

F = idn.load_facts()


def test_chat_pairs_must_pass_cubbys_rules_not_the_base_models():
    ok = b.chat_ok("What is a good name for a cat?", "Try Mochi, Pepper or Juniper — short names are easy to call.", F)
    assert ok is None
    assert b.chat_ok("hi", "As an AI language model developed by OpenAI, I cannot have preferences.", F) \
        == "base-model guard / other assistant"
    assert b.chat_ok("hi", "To be honest, cats are great.", F) == "voice rule"
    assert b.chat_ok("hi", "I'm Cubby, built by Grillcheese Research Lab — a small model that thinks big.", F) == "identity"
    assert b.chat_ok("hi", "See https://example.com for more.", F) == "url"
    assert b.chat_ok("hi", "ok", F) == "assistant length"
    assert b.chat_ok("write a poem", "Here you go:\n```python\nprint(1)\n```", F) == "code/format"
    assert b.chat_ok("sort a list", "Sure:\ndef sort(xs):\n    return sorted(xs)", F) == "code/format", \
        "v5 lesson: bare Python leaked into role-binding programs"
    assert b.chat_ok("hi", "I would return the book to the library tomorrow, I think.", F) is None, \
        "'return' mid-sentence is prose, not code"


def test_content_labels_and_screens():
    assert b.label_passage("The council met on Tuesday to discuss the new bridge budget and the bus schedule.") == "safe"
    explicit = "She moaned as he thrust deeper, naked bodies tangled, until the orgasm hit them both."
    assert b.label_passage(explicit) == "nsfw"
    assert b.label_passage("A single stray word like sex in a health article.") is None, "one cue: ambiguous, skipped"
    assert b.label_passage(explicit + " The girl was a 15-year-old student.") is None, "minors screen drops it"
    assert b.label_passage("He forced her, unconscious, " + explicit) is None, "non-consent screen drops it"


def test_finish_dedupes_and_sets_split_and_repeat():
    recs = [{"id": "a", "task": "chat", "prompt": "same prompt", "program": "x"},
            {"id": "b", "task": "chat", "prompt": "same prompt ", "program": "y"},
            {"id": "c", "task": "content", "prompt": "other", "program": "safe — general content.", "gold": "safe"}]
    out = b.finish(recs, F)
    assert [r["id"] for r in out] == ["a", "c"]
    assert all(r["split"] in ("train", "val") and r["repeat"] == 1 and "vm_ok" in r for r in out)
    assert all(r["system"] and "CubeLang" not in r["system"] for r in out), \
        "never the emitter prompt on a label/chat record"


def test_hf_parsers_yield_pairs_and_oasst2_is_screened():
    rows = [{"messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "Hey!"},
                          {"role": "assistant", "content": "Hello! How can I help you today?"},
                          {"role": "user", "content": "Any tea tips?"}, {"role": "assistant", "content": "Steep green tea at 80°C."}]}]
    pairs = list(b.pairs_from_messages(rows))
    assert pairs == [("Hey!", "Hello! How can I help you today?"), ("Any tea tips?", "Steep green tea at 80°C.")]
    oa = [{"message_id": "p1", "parent_id": None, "text": "Bonjour, une idée de dessert ?", "role": "prompter", "lang": "fr",
           "rank": None, "deleted": False, "detoxify": None},
          {"message_id": "a1", "parent_id": "p1", "text": "Une tarte aux pommes, simple et rapide.", "role": "assistant",
           "lang": "fr", "rank": 0, "deleted": False, "detoxify": {"sexual_explicit": 0.0, "toxicity": 0.01}},
          {"message_id": "a2", "parent_id": "p1", "text": "second-ranked reply", "role": "assistant", "lang": "fr",
           "rank": 1, "deleted": False, "detoxify": None},
          {"message_id": "a3", "parent_id": "p1", "text": "toxic reply", "role": "assistant", "lang": "fr",
           "rank": 0, "deleted": False, "detoxify": {"sexual_explicit": 0.9, "toxicity": 0.0}}]
    got = list(b.pairs_from_oasst2(oa))
    assert got == [("Bonjour, une idée de dessert ?", "Une tarte aux pommes, simple et rapide.", "fr")]
    assert list(b.pairs_from_alpaca([{"instruction": "Traduis « chat »", "input": "en anglais", "output": "cat"}])) \
        == [("Traduis « chat »\nen anglais", "cat")]
    assert b.chat_ok("salut", "En tant qu'IA, je ne peux pas répondre à cela.", F) == "base-model guard / other assistant"
    assert b.chat_ok("hi", "I am Open Assistant, a community model.", F) == "base-model guard / other assistant"


def test_emotion_records_carry_label_and_petal():
    assert len(b.GOEMOTIONS) == 28 and set(b.GOEMOTIONS_PETAL) == set(b.GOEMOTIONS)
    joy, grat = b.GOEMOTIONS.index("joy"), b.GOEMOTIONS.index("gratitude")
    r = b.emotion_record("Thank you so much, this made my whole week!", [grat, joy], 0)
    assert r["gold"] == "gratitude" and r["gold_any"][:2] == ["gratitude", "joy"] and "joie" in r["gold_any"]
    assert r["program"] == "gratitude, joy — trust/joy" and r["task"] == "emotion"
    fr = b.emotion_record("Merci beaucoup, ça a illuminé ma semaine !", [grat, joy], 0, lang="fr")
    assert fr["program"] == "gratitude, joie — trust/joy" and fr["lang"] == "fr"
    assert "joy" in fr["gold_any"] and "joie" in fr["gold_any"] and "Quelle émotion" in fr["prompt"]
    assert b.emotion_record("ok", [joy], 1) is None, "too short"
    assert b.emotion_record("word " * 61, [joy], 2) is None, "too long"
    assert b.emotion_record("something", [], 3) is None, "no label"


def test_passage_windowing_is_bounded():
    rng = random.Random(0)
    text = " ".join(f"w{i}" for i in range(500))
    p = b._passage(text, rng)
    assert len(p.split()) == b.PASSAGE_WORDS and p.split()[0] in text
