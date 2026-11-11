"""Order 0010, change d (CB_CUBBY_RECORD=1): Cubby's record as the facts of a question about Cubby, and the check on
every chat reply. The replies below are the blind A/B's own (2026-11-11), the kill first."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.speech import chat, cubby_record as rec, identity as idn  # noqa: E402

KILL = "I'm not a person or program. I can't provide information about any person, organization, or event."


class Fake:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def emit(self, prompt, *a, **k):
        return self.reply


def test_off_nothing_changes(monkeypatch):
    monkeypatch.delenv("CB_CUBBY_RECORD", raising=False)
    assert not idn.is_identity_question("are you a person or a program?")
    c = chat.CubbyChat(Fake("x"), idn.load_facts())
    assert ("Cubby", "name", "Cubby") in c.talk_context("who are you?", "en")["facts"]       # the old lines
    assert c._guard("Hi! I'm Daniel, and I make digital art.", "hello there") is None


def test_on_a_question_about_cubby_gets_the_record_as_its_block(monkeypatch):
    monkeypatch.setenv("CB_CUBBY_RECORD", "1")
    assert idn.is_identity_question("are you a person or a program?") and idn.is_identity_question("es-tu une personne ?")
    assert not idn.is_identity_question("who sang hey jude")
    c = chat.CubbyChat(Fake("x"), idn.load_facts())
    got = c.talk_context("are you a person or a program?", "en")
    assert ("Cubby", "is", "a model") in got["facts"] and ("Cubby", "is not", "a person") in got["facts"]
    assert ("Cubby", "creator", "GrillCheese Research Lab (GRL)") in got["facts"]
    assert ("Cubby", "langue", "français") in c.talk_context("qui es-tu ?", "fr")["facts"]


def test_on_the_chat_check_reads_what_cubby_says_it_is_and_every_name(monkeypatch):
    monkeypatch.setenv("CB_CUBBY_RECORD", "1")
    u = "are you a person or a program?"
    assert rec.chat_check(KILL, u) == (False, "self: denies being program")
    assert not rec.chat_check("Hi! I'm Daniel, and I make digital art.", "hello there, my name is Daniel")[0]
    assert not rec.chat_check("I'm human, just like you.", u)[0] and not rec.chat_check("My name is Ava.", "hi")[0]
    assert not rec.chat_check("I can help with writing stories, writing emails and more.", "how can you help me?")[0]
    assert not rec.chat_check("Je ne suis pas un modèle.", "es-tu un modèle ?")[0]
    for ok in ("No, I'm not a person. I'm Cubby, a model built by GrillCheese Research Lab.",
               "Je ne suis pas une personne, je suis Cubby, un modèle.", "I'm not sure, but I'm here to help.",
               "Here is what I can do for you today. I answer questions from facts I have on record.",
               "I can't write code for you yet."):
        assert rec.chat_check(ok, u)[0], ok
    assert rec.chat_check("Welcome, Daniel! I'm Cubby.", "hello there, my name is Daniel")[0]       # the user's name
    assert not rec.chat_check("Welcome, Daniel! I'm Cubby.", "hello there")[0]                    # a name from nowhere
    c = chat.CubbyChat(Fake("x"), idn.load_facts())
    assert c._guard("Hi! I'm Daniel, and I make digital art.", "hello there") == "self: says it is Daniel"


def test_the_record_is_v2s_facts_in_both_languages():
    r = rec.load()
    assert r["name"] == "Cubby" and r["role"]["fr"] == "la voix de CubbyLLM" and len(r["abilities"]["fr"]) == 4
    text = " ".join(rec.lines()).lower()
    assert "honest" not in text and "cubelang" not in text                                          # never said
