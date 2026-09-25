"""standin/history_lookup.py -- a question in plain words -> the event, the kind, the block the host serves.

Pinned: the kind reads off the question (cause, effect, when, where, who, what would change); an event is found
by its name word for word and by its rarer words when reworded; a year in the question picks between two events
of one name; two close events that would answer differently make the host ask, not guess; the block and the
returned values follow the kind, and a kind the graph does not hold returns nothing (the host's "don't say").
Run: python -m pytest standin/tests/test_history_lookup.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from history_graph import HistoryGraph, parse_time  # noqa: E402
from history_lookup import HistoryLookup, parse_kind, serve, words  # noqa: E402


def graph():
    g = HistoryGraph()
    ids = {}
    for name, when, where, who in [
        ("Assassination of Archduke Franz Ferdinand", "June 28, 1914", ["Sarajevo"], ["Gavrilo Princip"]),
        ("July Crisis", "1914", [], []),
        ("World War I", "1914-1918", ["Europe"], ["Germany", "France"]),
        ("Treaty of Versailles", "June 28, 1919", ["Versailles"], ["Germany", "Allied Powers"]),
        ("Battle of Newbury", "1643", ["Newbury"], ["Royalists"]),
        ("Siege of Vienna", "1529", ["Vienna"], ["Ottoman Empire"]),
        ("Battle of Vienna", "1683", ["Vienna"], ["Holy League", "Ottoman Empire"]),
        ("Ottoman advance on Vienna", "1683", ["Vienna"], ["Ottoman Empire"]),
    ]:
        ids[name] = g.add_event(name, parse_time(when), where, who, "", ("b", len(ids)))
    ids["Battle of Newbury 1644"] = g.add_event("Battle of Newbury", parse_time("1650"), ["Newbury"], ["Parliament"], "", ("b", 99))
    g.add_link(ids["Assassination of Archduke Franz Ferdinand"], "caused", ids["July Crisis"])
    g.add_link(ids["July Crisis"], "caused", ids["World War I"])
    g.add_link(ids["Treaty of Versailles"], "response to", ids["World War I"])
    g.add_link(ids["Ottoman advance on Vienna"], "caused", ids["Battle of Vienna"])
    return g, ids


def test_the_kind_reads_off_the_question():
    assert parse_kind("What led to the July Crisis?")[0] == "cause"
    assert parse_kind("Why did the Turks besiege Vienna in 1683?")[0] == "cause"
    assert parse_kind("What did the July Crisis lead to?")[0] == "effect"
    assert parse_kind("What were the consequences of the Great War?")[0] == "effect"
    assert parse_kind("When did the Treaty of Versailles happen?")[0] == "when"
    assert parse_kind("What year was the treaty signed at Versailles?")[0] == "when"
    assert parse_kind("Where did the Battle of Newbury take place?")[0] == "where"
    assert parse_kind("Who took part in the Battle of Vienna?")[0] == "who"
    assert parse_kind("What would change if the July Crisis had not happened?")[0] == "downstream"
    assert parse_kind("How would history be different if the archduke had never been shot?")[0] == "downstream"
    assert parse_kind("Tell me a story.")[0] is None


def test_found_by_name_and_by_its_rarer_words():
    g, ids = graph()
    lk = HistoryLookup(g)
    hit = lk.ask("What led to the July Crisis?")
    assert hit.status == "clear" and hit.event == ids["July Crisis"]
    assert hit.returned == ["Assassination of Archduke Franz Ferdinand"]
    assert ["Assassination of Archduke Franz Ferdinand", "caused", "July Crisis"] in hit.lines
    hit = lk.ask("When was the archduke Franz Ferdinand assassinated?")
    assert hit.status == "clear" and hit.event == ids["Assassination of Archduke Franz Ferdinand"]
    assert hit.returned == ["1914-06-28"]


def test_a_year_picks_between_events_and_close_ones_that_differ_make_the_host_ask():
    g, ids = graph()
    lk = HistoryLookup(g)
    assert lk.ask("Who fought in the Battle of Newbury in 1650?").event == ids["Battle of Newbury 1644"]
    assert lk.ask("Who fought in the Battle of Newbury in 1643?").event == ids["Battle of Newbury"]
    hit = lk.ask("Who fought in the Battle of Newbury?")         # two battles, different sides: ask
    assert hit.status == "ask" and hit.event is None and len(hit.candidates) >= 2


def test_a_year_inside_the_name_is_not_read_against_the_date():
    g = HistoryGraph()
    e = g.add_event("July Stock Trading Volume Largest Since 1933", parse_time("July 31, 1938"), [], [], "", ("n", 1))
    hit = HistoryLookup(g).ask("When did July Stock Trading Volume Largest Since 1933 happen?")
    assert hit.status == "clear" and hit.event == e


def test_a_description_no_name_fits_is_read_by_its_facts():
    g = HistoryGraph()
    kinsale = g.add_event("Battle of Kinsale", parse_time("1601"), ["Kinsale", "near Cork"],
                          ["Spain", "English", "Hugh O'Neill"], "", ("b", 1))
    g.add_event("Death of Hugh O'Neill", parse_time("1616"), ["Rome"], ["Hugh O'Neill"], "", ("b", 2))
    g.add_event("Siege of Cork", parse_time("1690"), ["Cork"], ["Marlborough"], "", ("b", 3))
    b = g.add_event("Nine Years' War", parse_time("1593-1603"), ["Ireland"], ["Hugh O'Neill", "English"], "", ("b", 4))
    g.add_link(kinsale, "part of", b)
    lk = HistoryLookup(g)
    hit = lk.ask("If the 1601 clash near Cork involving Spain, the English, and Hugh O'Neill had never occurred, "
                 "how might Irish history have unfolded differently?")
    assert hit.kind == "downstream" and hit.status == "clear" and hit.via == "facts" and hit.event == kinsale
    # a description two events fit as well is not guessed
    hit = lk.ask("Who fought beside the English and Spain in Ireland?")
    assert hit.status == "none" and {c["event"] for c in lk.find_facts("English Spain Ireland")[:2]} == {kinsale, b}


def test_the_kind_reads_off_rewordings():
    assert parse_kind("What led Henry II to make a pilgrimage to Canterbury in 1171?")[0] == "cause"
    assert parse_kind("How might Athens have developed differently if Hippias had never disarmed the militia?") == (
        "downstream", "Hippias had never disarmed the militia")
    assert parse_kind("How might the outcome of the campaign have changed if it never happened?")[1].startswith("How")
    assert parse_kind("How would the Thracian revolt against Baldwin have altered Balkan history?")[0] == "downstream"
    assert parse_kind("Which countries took part in the Congress of Vienna?")[0] == "who"
    assert parse_kind("Which specific areas did the Greeks colonize?")[0] == "where"
    assert words("Persecution of the Maronites") == words("persecute Maronites")
    assert words("Carnutes Rebellion") == words("Carnutes rebelled") and words("Khwārazm") == words("Khwarazm")
    assert words("Writing of 2 Maccabees") != words("Writing of 4 Maccabees") and words("in 56 BC") == []


def test_two_readings_of_one_event_are_served_as_one():
    g = HistoryGraph()
    a = g.add_event("Sack of Rome by Gauls", parse_time("390 BC"), ["Rome"], ["Gauls"], "", ("b1", 1))
    b = g.add_event("Sack of Rome by Brennus' Gauls", parse_time("390 BC"), ["Rome"], ["Brennus"], "", ("b2", 7))
    c = g.add_event("Battle of the Allia", parse_time("390 BC"), ["Allia"], [], "", ("b2", 7))
    first = g.add_event("First Battle of Bull Run", parse_time("1861"), [], [], "", ("b3", 1))
    second = g.add_event("Second Battle of Bull Run", parse_time("1862"), [], [], "", ("b3", 2))
    g.add_link(c, "caused", b)
    lk = HistoryLookup(g)
    assert lk.same_event(a, b) and not lk.same_event(a, c) and not lk.same_event(first, second)
    hit = lk.ask("What led to the sack of Rome by the Gauls?")
    assert hit.status == "clear" and set(hit.members) == {a, b} and hit.returned == ["Battle of the Allia"]


def test_a_kind_the_graph_does_not_hold_returns_nothing():
    g, ids = graph()
    lines, returned, others = serve(g, ids["Assassination of Archduke Franz Ferdinand"], "cause")
    assert returned == [] and lines                       # the event's own lines, no cause: the host says "don't say"
    lines, returned, others = serve(g, ids["World War I"], "effect")
    assert returned == ["Treaty of Versailles"] and ["Treaty of Versailles", "response to", "World War I"] in lines
    assert "World War I" not in others and "Germany" in others
    lines, returned, _ = serve(g, ids["Assassination of Archduke Franz Ferdinand"], "downstream")
    assert returned == ["July Crisis", "World War I", "Treaty of Versailles"]
