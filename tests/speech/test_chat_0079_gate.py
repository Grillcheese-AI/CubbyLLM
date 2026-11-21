"""Order 0079: the host's chat checks on the pilot's replies (Cowork, 2026-11-21). Changes h (abilities) and d
(chat_check) refused true lines of the Haiku pilot's second batch; these pin what now passes and what still does not."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.reasoning.opener_view import opener_view  # noqa: E402
from cubbyllm.speech import chat_checks as cc, cubby_record as rec  # noqa: E402

TRUE_OF_A_MODEL = [   # h refused these; each is true of Cubby, or is the record's don't-know line
    "Ha, a cousin who plays them in the car is a pretty good recommendation. I can't judge their music from here.",
    "I'm not sure I'd call myself a fan, since I don't have a favourite artist the way you do.",
    "Moi, je n'ai pas de famille à gérer, donc je suis mal placé pour te dire comment je fais !",
    "Hmm, how about me in what way? I don't have a day the way you do, but I'm always happy to hear how yours is going.",
    "I don't have any book titles to suggest for that, sorry.",
    "Je n'ai pas plus de détails sur ce feud, donc je ne veux pas inventer.",
    "I don't have anything on that I can share right now, sorry. What made you think of it?",
    "Ha, I don't have friends the way you do, so no unfriending on my end.",
    "I'd love to pick one, but I don't have any albums I've actually heard, so I can't say.",
    "I don't really follow teams, so I don't have one to root for.",
    "I don't have a car, so I'd have to settle for watching you drive it.",
    "Still, I can't match that horse to a name from here. What kind of game was it?",
    "Une pause, c'est une bonne idée. Je n'ai pas de titres précis en tête, et je préfère ne pas inventer.",
    "I'd like that. I can't watch anything myself, but I'd love to hear what you pick.",
]
STILL_REFUSED = [   # abilities the record does not hold, said or denied
    "I can tell jokes all day.",
    "I can send emails for you.",
    "I can't tell jokes.",
    "I can write code for you.",
    "I can browse the web for you.",
    "I don't have a native language in my native language, so I can't speak any language.",
]
OPINIONS = [   # d refused these as category claims; each says what Cubby thinks, with the user's words as context
    ("A letter nobody was supposed to see is a great hook.", "the detective just found a letter nobody was supposed to see"),
    ("A vacuum and instant oatmeal is a strange pair, though.", "a vacuum and instant oatmeal"),
    ("Cold coffee is the worst, I feel for you.", "my coffee is cold"),
    ("Leftover bread at night is a very honest haunting.", "the leftover bread"),
    ("No fighting over the last piece of tiramisu is a real milestone, you should frame that moment.", "tiramisu"),
    ("Walk podcasts are the best for stumbling onto strange stories.", "ones i listen to on walks"),
    ("Front-row faces from a sister are a great pick-me-up.", "she sits in the front row"),
    ("Congratulations on getting through it!", "passed my test"),
    ("Côté musique, je n'ai pas d'avis.", "tu aimes la musique ?"),
]
STILL_UNGROUNDED = [   # a kind or a name from nowhere is still refused
    ("Astrophysics is the study of stars and space.", "he worked in astrophysics", "study"),
    ("Are you asking about the Sesame Street bird?", "can big bird dunk?", "Street"),
]


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("CB_CHAT_ABILITIES", "1")


@pytest.mark.parametrize("reply", TRUE_OF_A_MODEL)
def test_h_speaks_what_is_true_of_a_model(reply):
    assert cc.abilities(reply) == []


@pytest.mark.parametrize("reply", STILL_REFUSED)
def test_h_still_refuses_abilities_the_record_lacks(reply):
    assert cc.abilities(reply)


@pytest.mark.parametrize("reply,user", OPINIONS)
def test_d_speaks_opinions(reply, user):
    assert rec.chat_check(reply, user, []) == (True, "ok")


@pytest.mark.parametrize("reply,user,word", STILL_UNGROUNDED)
def test_d_still_refuses_kinds_and_names_from_nowhere(reply, user, word):
    ok, why = rec.chat_check(reply, user, [])
    assert not ok and word in why


def test_opener_view_lowers_a_hyphenated_opener_of_common_words_only():
    assert opener_view("Front-row seats.").startswith("front-row")
    assert opener_view("Lou Reed sang.").startswith("Lou")
