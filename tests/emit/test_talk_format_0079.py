"""Order 0079's perspective addendum, items 1 and 3: the second talk prompt shape (CB_TALK_FORMAT=2) and the
one-emoji guard (CB_TALK_ONE_EMOJI=1). Off, the prompt and the answer are today's; on, one label per speaker."""
from __future__ import annotations

import pytest

from cubbyllm.emit import talk, talk_format as tf

TURNS = [("user", "hi there what is your mood today?"), ("assistant", "Pretty relaxed, thanks for asking."),
         ("user", "Thanks and congrat on the reply")]
FACTS = [("Lou Reed", "occupation", "musician"), "Lou Reed was born in 1942."]
BOARD = ("I'm feeling quite relaxed and ready to chat without taking a break! \U0001F31E I always enjoy our "
         "conversations. \U0001F31E What's on your mind today? \U0001F31E")


@pytest.fixture
def clean(monkeypatch):
    for k in ("CB_TALK_FORMAT", "CB_TALK_ONE_EMOJI", "CB_TALK_LINES"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_off_is_todays_shape(clean):
    assert talk.talk_prompt("hello") == "Question: hello\nAnswer:"
    assert talk.talk_prompt("x", {"turns": TURNS}) == (
        "Conversation:\nUser: hi there what is your mood today?\nAssistant: Pretty relaxed, thanks for asking.\n"
        "Question: Thanks and congrat on the reply\nAnswer:")
    assert talk.talk_prompt("x", {"facts": FACTS, "question": "who?", "explicit": True}) == (
        "Explicit: allowed\nFacts:\n- Lou Reed — occupation: musician\n- Lou Reed was born in 1942.\n"
        "Question: who?\nAnswer:")
    assert talk.talk_answer(BOARD) == BOARD                      # the guard is off: every emoji kept
    assert talk.talk_answer("Sure!\nCubby: more") == "Sure!"
    clean.setenv("CB_TALK_LINES", "1")
    assert talk.talk_answer("Sure!\nCubby: more") == "Sure!\nCubby: more"


def test_format_2_one_label_per_speaker(clean):
    clean.setenv("CB_TALK_FORMAT", "2")
    assert talk.talk_prompt("hello") == "User: hello\nCubby:"   # a first turn: nothing above it
    assert talk.talk_prompt("x", {"turns": TURNS}) == (
        "User: hi there what is your mood today?\nCubby: Pretty relaxed, thanks for asking.\n"
        "User: Thanks and congrat on the reply\nCubby:")
    assert talk.talk_prompt("x", {"facts": FACTS, "question": "who?", "explicit": True}) == (
        "Explicit: allowed\nFacts:\n- Lou Reed — occupation: musician\n- Lou Reed was born in 1942.\n"
        "User: who?\nCubby:")
    row = {"mood": {"joy": 2, "trust": 4, "fear": 0, "surprise": 0, "sadness": 0, "disgust": 0, "anger": 0,
                    "anticipation": 1}, "shade": [], "seems": "gratitude", "facts": FACTS[:1], "turns": TURNS}
    assert tf.render("", row) == (
        "Mood: joy +2, trust +4, fear 0, surprise 0, sadness 0, disgust 0, anger 0, anticipation +1\nShade: none\n"
        "They seem: gratitude\nFacts:\n- Lou Reed — occupation: musician\nUser: hi there what is your mood today?\n"
        "Cubby: Pretty relaxed, thanks for asking.\nUser: Thanks and congrat on the reply\nCubby:")
    clean.setenv("CB_TALK_LINES", "1")
    assert talk.talk_answer("Sure!\nCubby: more\nUser: and") == "Sure!"   # \nCubby: is a stop


def test_unlock_is_talks():
    assert tf.UNLOCK == talk.TALK_UNLOCK


def test_one_emoji(clean):
    assert tf.one_emoji(BOARD) == ("I'm feeling quite relaxed and ready to chat without taking a break! \U0001F31E "
                                   "I always enjoy our conversations. What's on your mind today?")
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    assert tf.emojis(f"a {family} b \U0001F1EB\U0001F1F7 c ❤️") == [family, "\U0001F1EB\U0001F1F7", "❤️"]
    assert tf.one_emoji(f"Nice {family} and \U0001F44D\U0001F3FD!") == f"Nice {family} and!"
    assert tf.one_emoji("No emoji here.") == "No emoji here."
    assert tf.one_emoji("Just one \U0001F642 here.") == "Just one \U0001F642 here."
    clean.setenv("CB_TALK_ONE_EMOJI", "1")
    # held to the sheet: the board's turn had no emoji, so the reply keeps none; a turn with one allows the first
    assert talk.talk_answer(BOARD, "hi there what is your mood today?") == (
        "I'm feeling quite relaxed and ready to chat without taking a break! I always enjoy our conversations. "
        "What's on your mind today?")
    assert talk.talk_answer(BOARD, "hi \U0001F642") == tf.one_emoji(BOARD)
    assert talk.talk_answer(BOARD) == tf.one_emoji(BOARD)               # a caller with no turn to give: the first
    assert tf.guard("\U0001F31E Morning!", "morning") == "Morning!"
    assert tf.user_turn("p", {"turns": TURNS}) == TURNS[-1][1] and tf.user_turn("p", {"question": "q"}) == "q"
    assert tf.user_turn("p") == "p"


@pytest.mark.parametrize("ctx", [
    {"question": "hello"},
    {"facts": FACTS, "question": "who was Lou Reed?"},
    {"turns": TURNS},
    {"turns": TURNS, "explicit": True},
    {"turns": [("user", "a"), ("assistant", "a list:\n1. one\n2. two"), ("user", "and\nthen?")]},
])
def test_todays_rows_convert_to_the_second_shape(clean, ctx):
    v1 = talk.talk_prompt("hello", ctx)
    parsed = tf.parse_v1(v1)
    assert parsed is not None
    clean.setenv("CB_TALK_FORMAT", "2")
    assert tf.render("", parsed) == talk.talk_prompt("hello", ctx)


def test_a_pilot_row_converts():
    v1 = ("Mood: joy 0, trust +1, fear 0, surprise 0, sadness 0, disgust 0, anger 0, anticipation +1\nShade: none\n"
          "They seem: curiosity\nFacts:\n- X — genre: jazz\nConversation:\nUser: hey\nAssistant: Hi! What's up?\n"
          "Question: tell me about X\nAnswer:")
    assert tf.render("", tf.parse_v1(v1)) == (
        "Mood: joy 0, trust +1, fear 0, surprise 0, sadness 0, disgust 0, anger 0, anticipation +1\nShade: none\n"
        "They seem: curiosity\nFacts:\n- X — genre: jazz\nUser: hey\nCubby: Hi! What's up?\n"
        "User: tell me about X\nCubby:")
    assert tf.render("", tf.parse_v1("Premise: contradicted\nFacts:\n- a — b: c\nQuestion: q?\nAnswer:")) == (
        "Premise: contradicted\nFacts:\n- a — b: c\nUser: q?\nCubby:")


@pytest.mark.parametrize("v1", [
    "Question: Translate this.\nQuestion: what?\nAnswer:",                    # a label inside the question
    "Conversation:\nUser: a\nCubby: b\nQuestion: c\nAnswer:",                 # a turn's line that reads as a label
    "Facts:\nnot a fact line\nQuestion: q\nAnswer:",                          # a fact block that does not open on "- "
    "Question: no answer label",
])
def test_unsure_rows_are_left_out(v1):
    assert tf.parse_v1(v1) is None
