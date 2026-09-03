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


def test_history_records_are_dated_and_screened():
    rng = random.Random(0)
    ev = {"title": "Unification of Ancient Egypt", "text": "King Menes united Upper and Lower Egypt into one state.",
          "year_start": -3100, "year_end": None, "actors": ["Menes"]}
    recs = b.event_records(ev, rng, F, 0, first_of_year=True)
    by = {r["subtype"]: r for r in recs}
    assert set(by) == {"about", "when", "year"} and all(r["task"] == "history" and r["lang"] == "en" for r in recs)
    assert by["when"]["program"] == "The Unification of Ancient Egypt was around 3100 BCE." and by["when"]["gold"] == "3100 BCE"
    assert by["about"]["program"].endswith("That was around 3100 BCE.") and "Menes" in by["about"]["gold_any"]
    assert "the Unification of Ancient Egypt" in by["about"]["prompt"], "a common-noun title takes 'the'"
    assert b.year_phrase(1914, 1918) == "from 1914 to 1918" and b.year_phrase(1066) == "in 1066"
    assert b.title_phrase("Magna Carta") == "Magna Carta" and b.title_phrase("The Black Death") == "The Black Death"
    # the shared screens: voice, guard, length
    assert b.history_record("year", 1, "What happened in 1900?", "To be honest, nothing much happened in 1900 at all.", "1900", ["1900"], "x", F) is None
    assert b.history_record("year", 2, "What happened in 1900?", "As an AI language model I cannot recall the year 1900.", "1900", ["1900"], "x", F) is None
    assert b.history_record("year", 3, "What happened in 1900?", "Too short.", "1900", ["1900"], "x", F) is None
    assert b.history_record("year", 4, "What happened in 1900?", "The Paris Exposition opened in 1900 and drew millions of visitors.", "1900", ["1900"], "x", F)


def test_dialogue_pairs_stand_alone_and_nyt_records_pair_recall_with_dating():
    rows = [{"speaker": "student", "topic": "Dante Alighieri", "message": "Who was Dante Alighieri?"},
            {"speaker": "expert", "topic": "Dante Alighieri", "message": "Dante was a Florentine poet, the author of the Divine Comedy."},
            {"speaker": "student", "topic": "Dante Alighieri", "message": "So what was his broader legacy?"},
            {"speaker": "expert", "topic": "Dante Alighieri", "message": "His impact on the Italian language was immense."},
            {"speaker": "student", "topic": "Other Person", "message": "unpaired"}]
    pairs = list(b.dialogue_pairs(rows))
    assert [q for _, q, _ in pairs] == ["Who was Dante Alighieri?", "Regarding Dante Alighieri: So what was his broader legacy?"]
    item = {"headline": {"main": "HOOVER PREDICTS COMING YEAR WILL BE BEST; Secretary Finds Large Assets"},
            "abstract": "LEAD: Sec Hoover says annual survey of Commerce Dept indicates prosperity for the coming year.",
            "pub_date": "1925-01-01T05:00:00+0000"}
    recs = {r["subtype"]: r for r in b.nyt_records(item, random.Random(0), F, 0)}
    assert set(recs) == {"news", "dating"}
    assert recs["news"]["prompt"].endswith("January 1, 1925?") or "January 1, 1925" in recs["news"]["prompt"]
    assert recs["news"]["program"].startswith("Hoover Predicts Coming Year Will Be Best: Sec Hoover says"), "ALL-CAPS headline title-cased, kicker dropped, LEAD: stripped"
    assert recs["dating"]["program"] == "January 1, 1925." and recs["dating"]["gold"] == "1925"
    assert b.clean_headline("Paid Notice: Deaths") is None and b.clean_headline("Word") is None


def test_history_ok_scores_when_dating_and_containment():
    when = {"subtype": "when", "gold": "3100 BCE", "gold_any": ["3100 BCE"]}
    assert b.history_ok(when, "It was around 3100 BCE.", F) and b.history_ok(when, "About 3100 BC, I believe.", F)
    assert not b.history_ok(when, "Around 2000 BCE.", F)
    dating = {"subtype": "dating", "gold": "1925", "gold_any": ["1925"]}
    assert b.history_ok(dating, "That reads like 1927, the late twenties.", F) and not b.history_ok(dating, "Probably 1961.", F)
    assert not b.history_ok(dating, "I cannot tell.", F)
    about = {"subtype": "about", "gold": "1066", "gold_any": ["Hastings", "William", "1066"]}
    assert b.history_ok(about, "William won at Hastings.", F) and not b.history_ok(about, "A battle somewhere, long ago.", F)
    assert not b.history_ok(about, "To be honest, William won at Hastings.", F), "voice rules still hold"
    assert b.proper_nouns("During the siege, General Grant took Vicksburg in 1863.") == ["General", "Grant", "Vicksburg", "1863"]


def test_local_chat_parsers_and_the_non_latin_screen():
    assert b.pair_from_arena("user: Count to 3\nassistant: 1, 2, 3.\nuser: thanks\nassistant: welcome") == ("Count to 3", "1, 2, 3.")
    assert b.pair_from_arena("assistant: no user turn") is None
    assert b.pair_from_nemotron("Instruction: how often?\n\nAnswer: Weekly, at least.") == ("how often?", "Weekly, at least.")
    assert b.wikiqa_pair({"question": "HOW ARE GLACIER CAVES FORMED", "answer": "Within the ice of a glacier."}) \
        == ("How are glacier caves formed?", "Within the ice of a glacier.")
    p, a = b.grammar_pair({"incorrect_example": "The dog barked, it wanted out.", "correct_example": "The dog barked; it wanted out.",
                           "explanation": "A comma splice joins two clauses with only a comma."})
    assert "The dog barked, it wanted out." in p and a.endswith('Better: "The dog barked; it wanted out."')
    assert b.ei_pair({"context_input": "{'conversation_history': [], 'query': \"It still fails!\"}",
                      "emotion_adapted_response": "{'content': \"That sounds frustrating. Let's get this sorted.\"}"}) \
        == ("It still fails!", "That sounds frustrating. Let's get this sorted.")
    assert b.chat_ok("what is OpenCL?", "OpenCL is a 并行编程接口 for parallel code on any platform.", F) == "non-latin"
    long = "word " * 120
    assert b.chat_ok("hi", long, F) == "assistant length" and b.chat_ok("hi", long, F, max_words=150) is None
    assert b.chat_ok("tangent of a sum?", "Use the formula \[\tan(A+B) = \frac{a+b}{1-ab}\] and expand it.", F) == "code/format", "LaTeX is a format leak"
    assert b.chat_ok("hi", "The cost is $$ high, but the view is worth it, they say.", F) == "code/format"


def test_plutchik_affect_and_quote_records():
    r = b.plutchik_record({"text": "I'm thrilled about the launch, all our work paid off!", "plutchik": {"primary": "joy", "secondary": "optimism"}}, 0)
    assert r["task"] == "emotion" and r["program"] == "joy, optimism — joy/anticipation" and r["gold"] == "joy" and "joie" in r["gold_any"]
    assert b.plutchik_record({"text": "x", "plutchik": {"primary": ["joy"]}}, 1) is None and \
        b.plutchik_record({"text": "a calm and quiet evening at home", "plutchik": {"primary": "stress"}}, 2) is None
    a = b.affect_record("A sense of calm washes over me.", 0.7, 0.2, 0, "realm_phase")
    assert a["task"] == "affect" and a["program"] == "valence +0.7, arousal 0.2" and a["gold_any"] == [0.7, 0.2]
    assert b.affect_record("see [FILE_PATH_6] please", 0.1, 0.2, 1, "convos") is None, "placeholder rows are dropped"
    assert b.affect_ok(a, "valence +0.5, arousal 0.4") and not b.affect_ok(a, "valence -0.5, arousal 0.2")
    assert b.affect_ok(a, "I'd say 0.6 and 0.3.") and not b.affect_ok(a, "calm and warm")
    replay = {"subtype": "realm_phase", "gold": None, "program": "valence +0.7, arousal 0.2"}   # the notebook's file: reference only
    assert b.affect_ok(replay, "valence +0.5, arousal 0.3") and not b.affect_ok(replay, "valence -0.9, arousal 0.3")
    hist = {"subtype": "dialogue", "gold": None, "program": "Dante was a Florentine poet, the author of the Divine Comedy."}
    assert b.history_ok(hist, "He was Dante, the poet of Florence.", F) and not b.history_ok(hist, "A poet, I believe.", F)
    rng = random.Random(0)
    recs = {x["subtype"]: x for x in b.quote_records({"quote": "As soon as you trust yourself, you will know how to live.",
                                                       "author": "Johann Wolfgang von Goethe", "category": "['trust']"}, rng, F, 0)}
    assert set(recs) == {"quote_about", "quote_who"}
    assert recs["quote_about"]["program"] == "“As soon as you trust yourself, you will know how to live.” — Johann Wolfgang von Goethe"
    assert "trust" in recs["quote_about"]["prompt"] and recs["quote_who"]["program"] == "Johann Wolfgang von Goethe." \
        and recs["quote_who"]["gold"] == "Johann Wolfgang von Goethe"
    assert b.history_ok(recs["quote_who"], "That is Goethe — Johann Wolfgang von Goethe.", F)
    assert b.quote_records({"quote": "Short one.", "author": "Unknown", "category": "[]"}, rng, F, 1) == []


def test_era_records_take_clean_book_windows():
    rng = random.Random(0)
    body = ("The Athenian assembly met on the Pnyx and voted on war and peace. " * 60)
    text = "ISBN 978-0-00 copyright 1998 all rights reserved. " * 5 + body + "Index: Athens, Sparta. " * 20
    p = b.era_passage(text, rng)
    assert p and len(p.split()) == b.PASSAGE_WORDS and "ISBN" not in p
    assert b.era_passage("too short", rng) is None
    r = b.era_record("AncientClassical", "Ancient Greece", p, 0, F)
    assert r["task"] == "history" and r["subtype"] == "era" and r["program"] == "The ancient and classical world — Ancient Greece."
    assert b.history_ok(r, "This is about the ancient world, Greece I think.", F) and not b.history_ok(r, "Medieval Europe.", F)
    m = b.era_record("MiddleAges", "Miscellaneous", p, 1, F)
    assert m["program"] == "The Middle Ages." and b.history_ok(m, "The Middle Ages, a monastery.", F)


def test_science_pairs_keep_the_answers_opening_sentences():
    r = {"input": "context: tag/antimatter/ question: Does antimatter have negative mass?",
         "label": " Antimatter does not have negative mass. In our universe, there is no such thing as negative mass. " + "Extra sentence here. " * 40}
    q, a = b.science_pair(r, cap=18)                   # 17 words fit; the 3-word filler would make 20
    assert q == "Does antimatter have negative mass?" and a == "Antimatter does not have negative mass. In our universe, there is no such thing as negative mass."
    assert b.science_pair({"input": "context: You are an AI assistant. question: hi", "label": "x"}) is None, "only the tag/ science rows"
    assert b.first_sentences("One. Two. Three.", cap=2) == "One. Two."


def test_movie_scene_records_read_the_dialogue_register():
    r = {"movie_title": "X", "scene_number": 1, "original_scene_text": "INT. ROOM - NIGHT ...",
         "full_dialogue_context": "I don't have a few years. Wish to hell I did, though.",
         "main_base_emotion": "Sadness", "plutchik_score": -0.7, "plutchik_label": "Regret",
         "emotion_timeline": ["Resignation (Michael)"], "annotated_dialogue": [{"line": "I don't have a few years.", "keyword": "years"}]}
    rec = b.movie_scene_record(r, 0)
    assert rec["task"] == "emotion" and rec["program"] == "regret, sadness — sadness" and rec["gold"] == "sadness"
    assert "regret" in rec["gold_any"] and "tristesse" in rec["gold_any"] and "Wish to hell" in rec["prompt"]
    assert "INT. ROOM" not in rec["prompt"], "the scene direction never reaches the prompt"
    only_lines = dict(r, full_dialogue_context="")
    assert "I don't have a few years." in b.movie_scene_record(only_lines, 1)["prompt"], "falls back to the annotated lines"
    assert b.movie_scene_record(dict(r, full_dialogue_context="", annotated_dialogue=[]), 2) is None, "no dialogue, no record"
    assert b.movie_scene_record(dict(r, main_base_emotion="Stress"), 3) is None, "only the eight Plutchik bases"


def test_fresh_identity_records_carry_the_world_intent_in_the_replay_schema():
    recs = b.fresh_identity_records(repeat=4, facts=F)
    assert recs and "world" in {r["subtype"] for r in recs}, "v7: the cubbyverse intent enters training here"
    r = recs[0]
    assert set(r) >= {"task", "subtype", "source", "prompt", "program", "gold", "system", "state", "lang", "id", "split", "repeat", "vm_ok"}
    assert r["task"] == "identity" and r["id"].startswith("identity-") and r["system"] and "CubeLang" not in r["system"]
    assert {r["repeat"] for r in recs if r["split"] == "train"} == {4} and all(r["repeat"] == 1 for r in recs if r["split"] == "val")
    assert len({r["prompt"] for r in recs}) >= 250                 # ~282 questions x 2 answers each
