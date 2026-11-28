"""Order 0079, item 5 (CB_TALK_CHECKED=1): served replies through the sheet's checks. Four replies in one batched
decode; the first that passes is spoken; when none does, the first that passed the host's guards is repaired (repeats
cut, emoji dropped) and spoken only if it then passes; markdown emphasis out of every talk reply; "how are you?" goes
as a chat turn, not with the record's abilities. Off: the turn is as it was."""
from __future__ import annotations

import pytest

from cubbyllm.emit import talk
from cubbyllm.speech import chat as chat_mod
from cubbyllm.speech import identity as idn
from cubbyllm.speech import sheet_checks as sc

FACTS = idn.load_facts()
DK = idn.T(FACTS, "dont_know_line", "en")


class Batched:
    name = "batched-fake"

    def __init__(self, replies):
        self.replies, self.asked, self.last = list(replies), [], {}

    def emit(self, prompt, max_new_tokens=160, system=None, **kw):
        self.asked.append(1)
        return self.replies[0]

    def emit_samples(self, prompt, n, max_new_tokens=160, system=None, **kw):
        self.asked.append(n)
        return (self.replies * n)[:n]


@pytest.fixture
def flags(monkeypatch):
    for k in ("CB_TALK_CHECKED", "CB_CUBBY_RECORD", "CB_TALK_ONE_EMOJI", "CB_CHAT_NO_LOOP", "CB_CHAT_ABILITIES"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_off_the_turn_is_as_it_was(flags):
    em = Batched(["Here you go \U0001F31E"])
    offered, rejected = chat_mod.CubbyChat(em, FACTS).candidates("thanks")
    assert em.asked == [1] and offered[0] == "Here you go \U0001F31E"


def test_on_the_first_reply_that_passes_is_spoken(flags):
    flags.setenv("CB_TALK_CHECKED", "1")
    em = Batched(["I'm in, sweetie.", "Sure \U0001F31E thing \U0001F31E", "Fine, thanks for asking.", "Fine, thanks for asking."])
    offered, rejected = chat_mod.CubbyChat(em, FACTS).candidates("thanks for the help")
    assert em.asked == [sc.DRAWS] and offered[0] == "Fine, thanks for asking."
    assert "I'm in, sweetie." in rejected and "Sure \U0001F31E thing \U0001F31E" in rejected


def test_on_none_passes_the_first_is_repaired(flags):
    flags.setenv("CB_TALK_CHECKED", "1")
    em = Batched(["A little moon, a little moon. A little moon, a little moon. \U0001F31E"])
    offered, _ = chat_mod.CubbyChat(em, FACTS).candidates("write me a tiny poem")
    assert offered[0] == "A little moon."
    em = Batched(["I'm in, sweetie."])                           # a pet name cannot be repaired: the safe line
    offered, _ = chat_mod.CubbyChat(em, FACTS).candidates("lookup marie curie")
    assert offered == [DK]


def test_the_checks_themselves():
    calm = {}
    assert sc.served("Thanks, dear!", "thanks dear", [], calm) is None              # the user used one first
    assert sc.served("Merci, ma belle !", "salut", [], calm).startswith("an endearment")
    assert sc.served("Ta belle-soeur a raison.", "ma belle-soeur dit que", [], calm) is None
    assert sc.served("Oh dear, that is a lot.", "long day", [], calm) is None       # an interjection
    assert sc.served("I'm so happy for you!", "I got the job", [], {"joy": 2}).startswith("feeling not backed")
    assert sc.served("I'm so happy for you!", "I got the job", [], {"joy": 5}) is None
    assert sc.served("Here:\n1. one\n2. two", "any ideas?", [], calm) == "a list nobody asked for"
    assert sc.cut_repeats("I'll write about it, if you'd like. I'll write about it, if you'd like.") == \
        "I'll write about it, if you'd like."


def test_markdown_out_of_every_talk_reply(flags):
    assert talk.talk_answer("**Hi** there, *friend*.") == "**Hi** there, *friend*."
    flags.setenv("CB_TALK_CHECKED", "1")
    assert talk.talk_answer("**Hi** there, *friend*. 5 * 3 is 15.") == "Hi there, friend. 5 * 3 is 15."


def test_how_are_you_is_a_chat_turn_not_the_record(flags):
    flags.setenv("CB_CUBBY_RECORD", "1")
    c = chat_mod.CubbyChat(Batched(["ok"]), FACTS)
    assert "facts" in c.talk_context("how are you?", "en")                # today: the record, abilities and all
    flags.setenv("CB_TALK_CHECKED", "1")
    assert "facts" not in c.talk_context("how are you?", "en") and "turns" in c.talk_context("how are you?", "en")
    assert "facts" in c.talk_context("what can you do?", "en")            # what it can do: still the record
