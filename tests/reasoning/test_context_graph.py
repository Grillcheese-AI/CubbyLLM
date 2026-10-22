"""standin/context_graph.py -- one conversation's memory, joined to the history graph and to the host's reasons.

Pinned: a follow-up that names no event ("when did it happen?", "what led to the battle?", "and then what?")
resolves to the event the conversation is about; a new event takes the focus; the reply to "which one did you
mean?" binds to one of the offered events; a follow-up with nothing in focus is not guessed; the trace says how
each answer was reached; a saved session is not readable without the host key.
Run: python -m pytest standin/tests/test_context_graph.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from context_graph import ContextGraph  # noqa: E402
from history_graph import HistoryGraph, parse_time  # noqa: E402
from history_lookup import HistoryLookup  # noqa: E402
from test_history_lookup import graph  # noqa: E402


def fresh():
    g, ids = graph()
    return ContextGraph(HistoryLookup(g)), ids


def test_a_pronoun_or_a_description_follows_the_conversation():
    cg, ids = fresh()
    a = cg.ask("What led to the July Crisis?")
    assert a.status == "clear" and a.via == "question" and a.returned == ["Assassination of Archduke Franz Ferdinand"]
    b = cg.ask("When did it happen?")
    assert b.status == "clear" and b.via == "context" and b.event == ids["July Crisis"] and b.returned == ["1914"]
    c = cg.ask("Who took part in the Battle of Vienna?")
    assert c.via == "question" and c.event == ids["Battle of Vienna"]
    d = cg.ask("What led to the battle?")
    assert d.via == "context" and d.event == ids["Battle of Vienna"] and d.returned == ["Ottoman advance on Vienna"]


def test_a_new_event_takes_the_focus_and_a_bare_follow_up_reads_it():
    cg, ids = fresh()
    cg.ask("What led to the July Crisis?")
    cg.ask("When was the Treaty of Versailles?")
    assert cg.ask("Where did it take place?").event == ids["Treaty of Versailles"]
    cg.ask("What led to the July Crisis?")
    e = cg.ask("And then what?")
    assert e.kind == "effect" and e.event == ids["July Crisis"] and e.returned == ["World War I"]


def test_the_reply_to_which_one_binds_and_nothing_in_focus_is_not_guessed():
    cg, ids = fresh()
    a = cg.ask("Who fought in the Battle of Newbury?")
    assert a.status == "ask" and cg.pending is a
    b = cg.ask("the one in 1650")
    assert b.via == "clarified" and b.event == ids["Battle of Newbury 1644"] and b.returned == ["Parliament"]
    assert cg.pending is None
    cg2, _ = fresh()
    c = cg2.ask("Why did it happen?")
    assert c.status == "none" and c.event is None
    cg2.ask("What led to the July Crisis?")               # a new topic the host cannot find: "it" is not the old one
    assert cg2.ask("Who fought at the Battle of Zzyzx?").status == "none"
    assert cg2.ask("When did it happen?").status == "none"
    assert cg2.ask("Who took part in Zzyzx?").status == "none"


def test_a_year_that_fits_two_readings_of_one_event_picks_and_that_asks_again():
    g, ids = graph()
    again = g.add_event("Battle at Newbury", parse_time("1650"), ["Newbury"], ["Earl of Essex"], "", ("b", 98))
    cg = ContextGraph(HistoryLookup(g))
    assert cg.ask("Who fought in the Battle of Newbury?").status == "ask"
    b = cg.ask("Where was that?")                         # the question stands unanswered: "that" is still unclear
    assert b.status == "ask" and b.via == "context" and cg.pending is b
    c = cg.ask("the one in 1650")                         # two readings hold 1650, and they are one battle
    assert c.via == "clarified" and set(c.members) == {ids["Battle of Newbury 1644"], again} and c.kind == "where"


def test_the_pick_is_among_the_events_the_host_offered():
    g = HistoryGraph()
    near = g.add_event("Domestication of cats", parse_time("3000 BC"), ["Egypt"], [], "", ("b", 1))
    far = g.add_event("Domestication of Cats", parse_time("8000 BC"), ["Near East"], [], "", ("b", 2))
    horse = g.add_event("Domestication of Horse", parse_time("8000 BC"), ["Steppe"], [], "", ("b", 3))
    for i, place in enumerate(["York", "Ascot", "Epsom", "Aintree", "Goodwood", "Newmarket", "Chester", "Ripon"]):
        g.add_event(f"Horse fair at {place}", parse_time(str(1700 + i)), [place], [], "", ("f", i))
    lk = HistoryLookup(g)
    hit = lk.ask("Where did the domestication of cats take place?")
    assert hit.status == "ask" and horse in {c["event"] for c in hit.candidates}
    assert {c["event"] for c in hit.offered} == {near, far}          # the horse is a candidate, not an option
    cg = ContextGraph(lk)
    cg.ask("Where did the domestication of cats take place?")
    b = cg.ask("The one in 8000 BC.")
    assert b.via == "clarified" and b.event == far and b.returned == ["Near East"]


def test_the_trace_says_how_and_a_saved_session_is_sealed(tmp_path):
    cg, ids = fresh()
    cg.ask("What led to the July Crisis?")
    a = cg.ask("When did it happen?")
    cg.record(a, "The July Crisis took place in 1914.", "The July Crisis took place in 1914.", "spoken")
    steps = [s["step"] for s in cg.why()]
    assert steps == ["kind", "context", "served", "draft", "verdict"]
    assert cg.decided(ids["July Crisis"], "when") == [a]
    key = b"k" * 32
    cg.save(tmp_path / "s.cbv", key)
    raw = (tmp_path / "s.cbv").read_bytes()
    assert b"July Crisis" not in raw and b"When did it happen" not in raw
    back = ContextGraph.load(tmp_path / "s.cbv", cg.lk, key)
    assert [t.text for t in back.turns] == [t.text for t in cg.turns] and back.asks[1].event == ids["July Crisis"]
