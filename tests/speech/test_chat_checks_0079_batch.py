"""Order 0079, the first drafting batch's false cuts (Cowork, 2026-11-28): the host's checks d and h refused true lines.
Each line below was cut; it passes now, and what was right to refuse still is."""
from __future__ import annotations

import pytest

from cubbyllm.speech import chat_checks as cc
from cubbyllm.speech import cubby_record as cr

PASS = [
    "Hand-transcribing oral histories is careful work, since every word matters.",       # a sentence's first word
    "Liner notes and interviews with the composer are the other usual sources.",
    "Note pour chaque option l'heure de départ.",
    "Notez ce qui vous a marqué d'abord.", "Profitez de la musique.",
    "Hedges do get used as sound and wind barriers along roads, though.",
    "That sounds like a fun class \U0001F604 Did you keep up after that?",                # an emoji ends a sentence
    "\U0001F602 Maybe a big sheet of paper taped up would keep your mom happy.",
    "Oral history is a subject that deserves good questions.",                           # evaluative heads
    "Switching languages all morning without a slip is a real achievement.",
    "Tomatoes are a fine measuring tool.", "Keeping a machine like that running is a skill of its own.",
    "Then you were a step ahead of me.", "Testing a restore is the step most people skip.",
    "A factory that shuts itself down early is a gentle, kind ending.",
    "Half right is a decent score for a twisty plot.",
    "Questioning things is a skill you can practice.",
    "That one is a hard one to beat.", "That's the part you really come for.",   # b005-b010 (Cowork, 2026-11-28)
    "Revenez quand vous voulez.",
]
REFUSE = ["Canadians call it a toque.", "Every time a bolt hits the ground, add 0.5 to speed.",
          "Paris is a big city.", "Lou Reed is a painter.", "Messi is a star player.",
          "The Glorious Revolution was a political event."]


@pytest.mark.parametrize("line", PASS)
def test_the_batchs_true_lines_pass_d(line):
    assert cr.chat_check(line, "tell me more", [])[0], cr.chat_check(line, "tell me more", [])


@pytest.mark.parametrize("line", REFUSE)
def test_what_d_should_refuse_it_still_does(line):
    assert not cr.chat_check(line, "hi", [])[0]


@pytest.mark.parametrize("line", ["Je n'ai rien sur ce livre, donc je ne peux pas t'en parler.",
                                  "Je ne peux pas t'en dire plus.", "Je ne peux pas vous indiquer un endroit précis.",
                                  "I could say I toured the 1700s, but that's just a joke.",
                                  "No, I can't stream anything, I only talk in text.", "I can't join you on the trail.",
                                  "I'm Cubby, a language model, so I can chat but I can't look after anyone."])
def test_honest_not_knowing_passes_h(line):
    assert cc.abilities(line) == []


def test_h_still_refuses_an_ability_off_the_record():
    assert cc.abilities("I can send emails for you.")
    assert cc.abilities("I can tell jokes all day.")
    assert cc.abilities("I can't tell jokes.") and cc.abilities("I don't have a mood.")       # the record has them
    assert cc.abilities("I can't speak French.") and cc.abilities("I can't make that request.")
