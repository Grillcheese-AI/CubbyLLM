"""Order 0010, changes g (CB_CHAT_NO_LOOP) and h (CB_CHAT_ABILITIES), and the French opener fix: the blind A/B's
third run's replies (talk_ab_2026-11-11c) as fixtures."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.speech import chat_checks as cc, cubby_record as rec  # noqa: E402

LOOPS = {   # g's refusals: the control's French replies say one sentence over and over; the voice repeats a line
    "Bonjour ! 😊\nC’est un bonjour, je peux t’aider avec toi.\nBonjour, je peux t’aider avec toi.\nBonjour, je peux t’aider avec toi.",
    "Vous avez des questions ou besoin d'aide ?\nDonner à un moment, un moment, un moment, un moment",
    "Could you please tell me:\n\n1. What you're asking?\n2. What you need help with?\n3. What you're curious about?\n4. What you're curious about?",
}
ABILITY_NO = {   # h's refusals: an ability the record does not hold
    "Bon ! Je peux maintenant enregistrer et envoyer mes messages. 😊",
    "I can't tell jokes or jokes that involve explicit or inappropriate content.",
    "I'm sorry, but I can't make that request. If you'd like, I can help you brainstorm creative ways.",
    "I'm here to help, but I don't have access to personal information or personal experiences.",
    "I don't have a native language in my native language, so I can't speak any language.",
    "Je ne peux pas raconter une blague, mais je peux vous aider à me tenir à l'esprit.",
}
SPOKEN = {   # what must stay spoken: the record's lines, a plain offer, the don't-know lines
    "I can answer questions from facts I have on record, and I check the answer before I say it.",
    "Here's what I can do for you:\n- I answer questions from facts I have on record.\n- I can show you the steps I took to reach an answer.",
    "Voici ce que je peux faire pour toi :\n- Je peux te montrer les étapes qui m'ont mené à une réponse.",
    "Voici ce que je sais faire : Quand plusieurs choses portent le même nom, je demande avant de répondre.",
    "Of course! I can help you with that. Please share the question.",
    "I am sorry, my training is not finished, I do not have that information yet.",
    "Je suis désolé, mon entraînement n'est pas terminé, je n'ai pas encore cette information.",
    "I can't write code for you yet.",
    "The facts don't say.",
    "Hello there! What shall we look up today?",
    "Good morning! 🌞 I hope your day is off to a wonderful start. I hope you're having a wonderful day too! 🌞",
    # true lines of the voice set's held chat rows that h refused as first written (the kill line, read 2026-11-11)
    "Ha — no. I'm Cubby, a model. I can talk about feelings, but I don't have them; I'm a tool GrillCheese Research Lab (GRL) built to help you.",
    "Je ne partage pas de lieux. Je peux te dire qui m'a créé : GrillCheese Research Lab (GRL).",
    "Voilà un aperçu de ce que je peux faire pour toi:\n- **Dire quand je ne sais pas :** Si je n'ai aucun fait vérifié, je te le dis.",
    "Take a look at what I can help with:\n- **Answer from checked facts:** I answer questions from facts I have on record.",
    "Alors voilà, je peux t'aider avec tout ça. Ce que je peux t'offrir : des réponses vérifiées.",
    "Coucou ! Qu'est-ce que je peux faire pour toi aujourd'hui ?",
    "Welcome aboard! I'm Cubby. I stick to what I can confirm, and I'm ready when you are.",
}


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setenv("CB_CHAT_NO_LOOP", "1")
    monkeypatch.setenv("CB_CHAT_ABILITIES", "1")


def test_g_a_reply_that_loops_is_not_spoken(on):
    for t in LOOPS:
        assert (cc.check(t) or "").startswith("loop"), t
    assert cc.check("Hi there!", hit_limit=True) == "loop: stopped at the token limit"
    assert cc.loops("Ha ha ha, good one.") is not None                                # one word three times in a row
    assert cc.loops("I hope it helps. I hope you enjoy it.") is None                  # a phrase, not a sentence, again


def test_h_what_cubby_says_it_can_do_comes_from_its_record(on):
    for t in ABILITY_NO:
        assert (cc.check(t) or "").startswith("ability:"), t
    for t in SPOKEN:
        assert cc.check(t) is None, t


def test_flags_off_nothing_changes(monkeypatch):
    monkeypatch.delenv("CB_CHAT_NO_LOOP", raising=False)
    monkeypatch.delenv("CB_CHAT_ABILITIES", raising=False)
    for t in LOOPS | ABILITY_NO:
        assert cc.check(t, hit_limit=True) is None


def test_with_h_the_forbidden_list_gives_way_to_the_record(monkeypatch):
    monkeypatch.setenv("CB_CUBBY_RECORD", "1")
    t = "I can help with writing stories, writing emails and more."
    monkeypatch.delenv("CB_CHAT_ABILITIES", raising=False)
    assert rec.self_claims(t)                                                         # d's list
    monkeypatch.setenv("CB_CHAT_ABILITIES", "1")
    assert not rec.self_claims(t) and cc.abilities(t)                                 # h's record, in its place


def test_a_french_verb_joined_to_its_pronoun_is_no_name(monkeypatch):
    monkeypatch.setenv("CB_CUBBY_RECORD", "1")
    for t in ("Bonjour ! Installe-toi, on va mettre les faits au clair.", "Appelle-moi Cubby.", "Allez-y, pose ta question."):
        assert rec.chat_check(t, "bonjour Cubby")[0], t
    for t in ("Bonjour ! Jean-Pierre est là.", "Bonjour ! Saint-Denis est une ville."):
        assert not rec.chat_check(t, "bonjour Cubby")[0], t                           # names stay names
