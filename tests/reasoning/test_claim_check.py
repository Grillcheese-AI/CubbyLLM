"""standin/ask.py `claim_check` -- a reply may say more of the block than the one value asked for, and still never
say what the block does not (H-E10).

Pinned: a fuller answer built from the block passes ("It began in 1914 in Sarajevo, when Gavrilo Princip ..."); a
value tied to the wrong event, a cause turned round, a two-step chain said as one step, and an end year said as a
beginning are refused; "it" stands for the event asked about; the returned value must still be said.
Run: python -m pytest standin/tests/test_claim_check.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ask import claim_check, talk_reply  # noqa: E402

A, J, W, T = ("Assassination of Archduke Franz Ferdinand", "July Crisis", "World War I", "Treaty of Versailles")
LINES = [
    [A, "date", "1914-06-28"], [A, "location", "Sarajevo"], [A, "participant", "Gavrilo Princip"],
    [A, "caused", J], [J, "year", "1914"], [J, "caused", W],
    [W, "start year", "1914"], [W, "end year", "1918"], [W, "location", "Europe"],
    [T, "response to", W], [T, "date", "1919-06-28"], [T, "location", "Versailles"],
]


def ok(answer, returned, entity):
    return claim_check(answer, LINES, returned, entity)[0]


def test_a_fuller_answer_from_the_block_passes():
    assert ok("It happened on June 28, 1914 in Sarajevo, and Gavrilo Princip took part.", ["1914-06-28"], A)
    assert ok("The assassination of Archduke Franz Ferdinand led to it. That was in 1914.", [A], J)
    assert ok("It led to World War I, which began in 1914 and ended in 1918.", [W], J)
    assert ok("World War I led to the Treaty of Versailles, signed at Versailles on June 28, 1919.", [T], W)
    assert ok("The Treaty of Versailles came in response to World War I.", [T], W)


def test_what_the_block_does_not_say_is_refused():
    assert not ok("It happened in 1914 at Versailles.", ["1914"], J)                 # Versailles is the treaty's
    assert not ok("The July Crisis led to the assassination of Archduke Franz Ferdinand.", [A], J)   # turned round
    assert not ok("The assassination of Archduke Franz Ferdinand led to World War I.", [J], A)       # two steps as one
    assert not ok("World War I began in 1918.", ["1914"], W)                            # an end year as a beginning
    assert not ok("It was caused by World War I.", [A], J)                             # backward cue, wrong way
    assert not ok("Gavrilo Princip took part.", ["1914"], J)                           # the returned value unsaid


def test_the_host_uses_it_only_when_asked_and_keeps_the_guard():
    facts = [f"{e} {r}: {v}" for e, r, v in LINES]
    fuller = "It led to World War I, which began in 1914 and ended in 1918."
    others = [A, "1914", "1918", "Europe", T]
    assert talk_reply([W], fuller, facts, J, others)[0] is None                        # the value check: too much
    assert talk_reply([W], fuller, facts, J, others, lines=LINES, check="claims") == (fuller, "spoken")
    reply, why = talk_reply([W], "It led to World War I and the Russian Revolution.", facts, J, others,
                            lines=LINES, check="claims")
    assert reply is None and why.startswith("ungrounded")                             # the name guard still runs
