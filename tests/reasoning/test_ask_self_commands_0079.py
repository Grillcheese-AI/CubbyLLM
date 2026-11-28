"""Order 0079, items 6 and 7. CB_ASK_SELF=1: "who is cubby?" is answered from Cubby's record, never a store (the
board offered Peter Pan's Cubby). CB_ASK_COMMANDS=1: "lookup marie curie then tell me the discoveries" is a lookup of
her discoveries, not chat. Off: both as they were."""
from __future__ import annotations

import pytest

from cubbyllm.brain_route import BrainRoute
from cubbyllm.reasoning import ask_commands, ask_self
from test_ask import FakeEmitter, make_loop  # noqa: E402
from test_search_learn import DictSource  # noqa: E402

CMD = "lookup marie curie then tell me the discoveries"


class Talker(FakeEmitter):
    """A talk seat for the record path: says `line` when asked with the record's block."""

    def __init__(self, line):
        super().__init__({})
        self.line, self.contexts = line, []

    def emit(self, prompt, max_new_tokens=300, **kw):
        ctx = kw.get("context") or {}
        if ctx.get("role") == "talk":
            self.contexts.append(ctx)
            return self.line
        return super().emit(prompt, max_new_tokens, **kw)


@pytest.fixture
def flags(monkeypatch):
    for k in ("CB_ASK_SELF", "CB_ASK_COMMANDS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_who_is_cubby_from_the_record(flags):
    loop = make_loop({}, DictSource({}))
    loop.emitter = Talker("I'm Cubby, the voice of CubbyLLM.")
    assert loop.ask("who is cubby?").get("world") != "record"            # off: the loop's own path, as before
    flags.setenv("CB_ASK_SELF", "1")
    rec = loop.ask("who is cubby?")
    assert rec["world"] == "record" and rec["reply"] == "I'm Cubby, the voice of CubbyLLM." and not rec["verified"]
    assert loop.emitter.contexts[-1]["facts"][0][0] == "Cubby"           # the record's block, handed to the talk seat
    loop.emitter = Talker("I'm the lost boy from Neverland, Peter Pan's friend.")
    rec = loop.ask("who is cubby?")                                       # a reply the record does not back
    assert rec["reply"] == ask_self.record_line("en") and rec["seat_refused"]
    assert ask_self.about("are you a robot?") and ask_self.about("what can you do?") and not ask_self.about("who is Lou Reed?")


def test_questions_to_cubby_about_its_life_are_not_the_worlds(flags):
    asks = ["Et toi, tu lis quoi ?", "Vous avez déjà fait du bénévolat, vous ?", "where have you traveled?",
            "Do you use google often?", "I just got off work, you?"]
    assert all(BrainRoute().needs_facts(q)[0] for q in asks[:4])            # off: sent to the world, as on 2026-11-28
    flags.setenv("CB_ASK_SELF", "1")
    assert not any(BrainRoute().needs_facts(q)[0] for q in asks)
    assert all(ask_self.about(q) for q in asks)
    assert BrainRoute().needs_facts("what's the capital of peru?")[0] and BrainRoute().needs_facts("who is Lou Reed?")[0]


def test_a_command_is_a_lookup(flags):
    assert BrainRoute().needs_facts(CMD)[0] is False                      # off: chat
    flags.setenv("CB_ASK_COMMANDS", "1")
    assert BrainRoute().needs_facts(CMD) == (True, "a request to look something up")
    assert BrainRoute().needs_facts("tell me about your day")[0] is False
    assert ask_commands.restate(CMD) == "What are the discoveries of marie curie?"
    assert ask_commands.restate("find the capital of peru") == "What is the capital of peru?"
    assert ask_commands.restate("tell me about it") == "tell me about it"
    # the first batch's router misses (Cowork, 2026-11-28)
    assert BrainRoute().needs_facts("hey do you know who Howard K Smith is? my teacher mentioned him")[0] is True
    assert ask_commands.restate("hey do you know who Howard K Smith is? my teacher") == "Who is Howard K Smith?"
    assert ask_commands.restate("do you know when Lou Reed was born?") == "When was Lou Reed born?"
    assert ask_commands.restate("Tu sais qui a fondé Ikea ?") == "Qui a fondé Ikea ?"
    assert BrainRoute().needs_facts("do you know who made you?")[0] is False
    assert BrainRoute().needs_facts("you know what, I love jazz")[0] is False
    rec = make_loop({}, DictSource({})).ask(CMD)
    assert rec["question"] == "What are the discoveries of marie curie?" and rec["asked_as"] == CMD
    assert not rec["verified"]                                            # stored the other way round: refused today
