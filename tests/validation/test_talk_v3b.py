"""talk_v3b (order 0010, Cowork's addendum of 2026-11-20): the gate every target passes, and the lines written by rule."""
from __future__ import annotations

import pytest

from cubbyllm.speech import chat_checks, cubby_record
from validation.sft import talk_voice_v3b as v3b


@pytest.fixture(scope="module")
def gate():
    return v3b.Gate()


def row(family, prompt, answer, wording=()):
    return {"id": "t", "family": family, "split": "train", "prompt": prompt, "answer": answer, "wording": list(wording)}


def test_emoji_kept_only_when_the_user_used_one(gate):
    out, why = gate.check(row("voice_social", "Question: thanks!\nAnswer:", "Anytime! 😊"))
    assert out["answer"] == "Anytime!" and why == "ok"
    out, _ = gate.check(row("voice_emoji", "Question: thanks! 😊\nAnswer:", "Anytime! 😊"))
    assert out["answer"] == "Anytime! 😊"


def test_the_host_checks_drop_what_the_host_would_refuse(gate):
    assert gate.check(row("chat", "Question: hi\nAnswer:", "Hello! Hello! Hello! Hello!"))[1] == "g: loop"
    assert gate.check(row("chat", "Question: hi\nAnswer:", "I can send your emails for you."))[1] == "h: ability"
    assert gate.check(row("voice_social", "Question: who?\nAnswer:", "I'm a person, just like you."))[1] == "d: self claim"
    assert gate.check(row("voice_social", "Question: hi\nAnswer:", "Hi! Daniel says hello."))[1].startswith("d: ungrounded")


def test_a_joke_that_says_a_thing_twice_is_dropped(gate):
    r = row("voice_social", "Question: hi\nAnswer:", "Hi! Guess what? I can't guess.", ["joke"])
    assert gate.check(r)[1] == "joke repeats a word"


def test_the_same_entity_three_times_is_dropped(gate):
    p = "Facts:\n- Alpha Band — member: Ann\n- Alpha Band — member: Bo\nQuestion: who is in Alpha Band?\nAnswer:"
    a = "Alpha Band has two members. In Alpha Band: Ann and Bo. That's Alpha Band."
    assert gate.check(row("voice_list", p, a))[1] == "the same entity three times"


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_the_personal_lines_pass_the_host(lang):
    for line in v3b.PERSONAL[lang]:
        assert not chat_checks.loops(line) and not chat_checks.abilities(line), line
        assert not cubby_record.self_claims(line), line
        assert cubby_record.chat_check(line, "do you have pets?")[0], line


def test_languages_answered_from_the_record():
    rec = cubby_record.load()
    assert v3b._langs(rec, "en", None) == "I speak English and French today. Russian and Spanish are next."
    assert v3b._langs(rec, "en", "Spanish").startswith("Not yet!")
    for lang, asked in (("en", "French"), ("fr", "Spanish"), ("fr", None)):
        assert cubby_record.chat_check(v3b._langs(rec, lang, asked), "")[0]


def test_interjections_and_the_running_joke_are_no_longer_refused():
    assert cubby_record.chat_check("Yes! I can show you the steps I took to reach an answer.", "")[0]
    assert cubby_record.chat_check("Nope! I'm Cubby, built by GrillCheese Research Lab (GRL).", "are you Claude?")[0]
    assert cubby_record.chat_check("Oui ! Je peux te montrer les étapes qui m'ont mené à une réponse.", "")[0]
    assert cubby_record.chat_check("In a past life in the 1700s, I sold cheese.", "")[0]
    assert not cubby_record.chat_check("In 1850 I sold cheese.", "")[0]            # a number from nowhere still is


def test_natural_chat_is_no_longer_refused_and_real_claims_still_are():
    ok = cubby_record.chat_check
    assert ok("Oh no, that sounds really worrying. If it keeps going past a day, a vet visit is a good idea.", "my cat")[0]
    assert ok("Wow, that's a great question! Haha, I love that.", "guess what")[0]
    assert not ok("Lou Reed is a painter.", "tell me about Lou Reed", ["Lou Reed — occupation: musician"])[0]
    assert not ok("Oh, Daniel told me about it.", "hi")[0]                   # a name from nowhere still is


def test_user_turn_and_strip_emoji():
    assert v3b.user_turn("Conversation:\nUser: hi\nAssistant: Hey!\nQuestion: thanks 🙏\nAnswer:") == "thanks 🙏"
    assert v3b.strip_emoji("Bye for now 👋 !") == "Bye for now!"
