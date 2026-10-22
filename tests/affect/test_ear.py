"""The ear (standin/ear.py): good or bad by contrast, negation included, learning from what follows."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "standin", "tests")), ".."))
from ear import Ear, aimed, features, from_pairs  # noqa: E402

PAIRS = [{"good": "Good job, Cubby!", "bad": "Bad job, Cubby."},
         {"good": "You did great.", "bad": "You did terribly."},
         {"good": "Bravo, c'est super!", "bad": "Bravo pour rien, c'est nul."},
         {"good": "That's not bad at all.", "bad": "That's bad."},
         {"good": "It's safe now.", "bad": "It's dangerous now."},
         {"good": "Merci, t'es fin.", "bad": "Merci pour rien, t'es pas fin."}]


def test_a_word_after_a_negation_is_its_negated_self():
    fs = dict(features("that's not bad"))
    assert "~bad" in fs and "bad" not in fs and "not bad" in fs
    assert "~mal" in dict(features("c'est pas mal"))


def test_taught_pairs_are_heard_on_their_side():
    ear = from_pairs(PAIRS)
    for p in PAIRS:
        assert ear.valence(p["good"]) > ear.valence(p["bad"])
    assert ear.valence("Good job, Cubby!") > 0 > ear.valence("Bad job, Cubby.")


def test_negation_turns_the_sign():
    ear = from_pairs(PAIRS)
    assert ear.valence("not bad") > ear.valence("bad")


def test_what_follows_the_words_teaches_them():
    ear = from_pairs(PAIRS)
    before = ear.valence("careful, lava")
    for _ in range(4):
        ear.learn("careful, lava", -1.0)                  # each time, a burn came after it
    assert ear.valence("careful, lava") < before and ear.lived == 4


def test_it_says_which_words_decided():
    ear = from_pairs(PAIRS)
    top = ear.why("You did terribly.", 2)
    assert top[0][1] < 0 and any("terribly" in f for f, _ in top)


def test_said_to_him_or_about_someone_else():
    assert aimed("Good job, Cubby!") and aimed("t'es bon") and aimed("you did it")
    assert not aimed("the villagers are happy")


def test_kept_and_reloaded_it_hears_the_same(tmp_path):
    ear = from_pairs(PAIRS)
    ear.learn("well done", 1.0)
    back = Ear.load(ear.save(tmp_path / "ear.npz"))
    assert abs(back.valence("Good job, Cubby!") - ear.valence("Good job, Cubby!")) < 1e-6
    assert back.lived == 1


def test_a_lexicon_carries_a_contrast_to_words_never_taught():
    from ear_lexicon import Lexicon
    lex = Lexicon.find()
    if lex is None:
        pytest.skip("no static embedding table in the Hugging Face cache")
    ear = from_pairs([{"good": "You are great.", "bad": "You are awful."},
                      {"good": "What a wonderful day.", "bad": "What a horrible day."}], lexicon=lex)
    assert ear.valence("You are brilliant.") > ear.valence("You are dreadful.")
