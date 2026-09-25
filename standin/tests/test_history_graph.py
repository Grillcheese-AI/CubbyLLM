"""standin/history_graph.py -- events in time, linked by cause and precursor.

Pinned: dates as history books write them read to years (BC negative) and days; one event read from two books is
one node when the dates agree, and two when the same name is years apart; a link whose dates contradict it is
refused; downstream(x) is everything a change to x reaches, in time order; chain(a, b) is the causal path; the
graph saves and loads back identical.
Run: python -m pytest standin/tests/test_history_graph.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from history_graph import HistoryGraph, parse_time  # noqa: E402


def y(text):
    t = parse_time(text)
    return (t["y0"], t["y1"], t["day"]) if t else None


def test_dates_as_books_write_them():
    assert y("April 12, 1861") == (1861, 1861, "1861-04-12")
    assert y("12 avril 1861") == (1861, 1861, "1861-04-12")
    assert y("1066") == (1066, 1066, None)
    assert y("44 BC") == (-44, -44, None)
    assert y("c. 500 B.C.") == (-500, -500, None)
    assert y("1861-65") == (1861, 1865, None)
    assert y("490-479 BC") == (-490, -479, None)
    assert y("the 1790s") == (1790, 1799, None)
    assert y("18th century") == (1701, 1800, None)
    assert y("5th century BC") == (-500, -401, None)
    assert y("in the spring") is None
    assert y("1813-02-29") == (1813, 1813, None) and y("June 31, 1934") == (1934, 1934, None)   # no such day


def test_one_event_from_two_books_and_two_events_with_one_name():
    g = HistoryGraph()
    a = g.add_event("Battle of Gettysburg", parse_time("July 1, 1863"), ["Gettysburg"], ["George Meade"], "battle", ("b1", 3))
    b = g.add_event("the Battle of Gettysburg", parse_time("1863"), ["Pennsylvania"], ["Robert E. Lee"], "battle", ("b2", 9))
    assert a == b and g.events[a].when["day"] == "1863-07-01"
    assert g.events[a].who == ["George Meade", "Robert E. Lee"] and len(g.events[a].sources) == 2
    n1 = g.add_event("Battle of Newbury", parse_time("1643"))
    n2 = g.add_event("Battle of Newbury", parse_time("1644"))
    n3 = g.add_event("Battle of Newbury", parse_time("1650"))
    assert n1 == n2 and n3 != n1                                # a year of slack; six years apart is another battle


def test_a_link_whose_dates_contradict_it_is_refused():
    g = HistoryGraph()
    sumter = g.add_event("Battle of Fort Sumter", parse_time("April 12, 1861"))
    war = g.add_event("American Civil War", parse_time("1861-1865"))
    gettysburg = g.add_event("Battle of Gettysburg", parse_time("1863"))
    assert g.add_link(sumter, "caused", war) is None
    assert g.add_link(gettysburg, "part of", war) is None
    assert g.add_link(gettysburg, "caused", sumter) == "time_contradicts"
    assert g.add_link(war, "part of", gettysburg) == "time_contradicts"
    assert g.add_link(war, "caused", war) == "self"


def test_downstream_and_chain(tmp_path):
    g = HistoryGraph()
    ids = {n: g.add_event(n, parse_time(d)) for n, d in [
        ("Assassination of Archduke Franz Ferdinand", "June 28, 1914"), ("July Crisis", "1914"),
        ("World War I", "1914-1918"), ("Battle of the Somme", "1916"), ("Treaty of Versailles", "1919"),
        ("Congress of Vienna", "1815")]}
    g.add_link(ids["Assassination of Archduke Franz Ferdinand"], "caused", ids["July Crisis"])
    g.add_link(ids["July Crisis"], "caused", ids["World War I"])
    g.add_link(ids["Battle of the Somme"], "part of", ids["World War I"])
    g.add_link(ids["Treaty of Versailles"], "response to", ids["World War I"])
    down = [g.events[i].name for i in g.downstream(ids["Assassination of Archduke Franz Ferdinand"])]
    assert down == ["July Crisis", "World War I", "Battle of the Somme", "Treaty of Versailles"]
    assert "Congress of Vienna" not in down
    path = g.chain(ids["Assassination of Archduke Franz Ferdinand"], ids["Treaty of Versailles"])
    assert [g.events[i].name for i in path] == ["Assassination of Archduke Franz Ferdinand", "July Crisis",
                                                "World War I", "Treaty of Versailles"]
    assert g.chain(ids["Congress of Vienna"], ids["World War I"]) is None
    g.add_fact("Woodrow Wilson", "position held", "President of the United States", ("b", 1))
    g.save(tmp_path / "g.jsonl")
    h = HistoryGraph.load(tmp_path / "g.jsonl")
    assert h.links == g.links and h.facts == g.facts and set(h.events) == set(g.events)
    assert h.downstream(ids["July Crisis"]) == g.downstream(ids["July Crisis"])


def test_every_event_gets_a_date_marked_by_how_it_was_found():
    """A stated date beats a context year; an undated event takes its book's year there, else its whole's span."""
    g = HistoryGraph()
    ctx = dict(parse_time("1864"), basis="context", approx=True)
    a = g.add_event("Battle of Resaca", ctx, source=("foote", 12))
    assert g.add_event("Battle of Resaca", parse_time("May 14, 1864"), source=("mcpherson", 3)) == a
    assert g.events[a].when["day"] == "1864-05-14" and g.events[a].when.get("basis", "stated") == "stated"
    campaign = g.add_event("Atlanta Campaign", parse_time("1864"), source=("foote", 12))
    skirmish = g.add_event("Skirmish at Rocky Face Ridge", None, source=("foote", 13))
    raid = g.add_event("Wheeler's Raid on Dalton", None, source=("other", 40))
    g.add_link(raid, "part of", campaign)
    stats = g.infer_dates({("foote", 13): 1864})
    assert g.events[skirmish].when["y0"] == 1864 and g.events[skirmish].when["basis"] == "book"
    assert g.events[raid].when["basis"] == "part of" and g.events[raid].when["y0"] == 1864
    assert stats["book"] == 1 and stats["part of"] == 1 and stats["undated"] == 0
    # a context year merges with a stated date a few years off; a stated date does not
    b = g.add_event("Siege of Petersburg", dict(parse_time("1865"), basis="context", approx=True))
    assert g.add_event("Siege of Petersburg", parse_time("1862")) == b
    c = g.add_event("Battle of Newbury", parse_time("1643"))
    assert g.add_event("Battle of Newbury", parse_time("1646")) != c


def test_links_are_checked_again_once_dates_are_final():
    """'ended' runs from the end to what it ends; a link to an undated event is checked once that event has a date."""
    g = HistoryGraph()
    treaty = g.add_event("Treaty of Fort Laramie", parse_time("1868"))
    war = g.add_event("Red Cloud's War", parse_time("1865-1867"))
    assert g.add_link(war, "ended", treaty) == "time_contradicts"
    assert g.add_link(treaty, "ended", war) is None
    petersburg = g.add_event("Battle of Petersburg", parse_time("March 29, 1865"))
    richmond = g.add_event("Fall of Richmond", parse_time("April 3, 1865"))
    assert g.add_link(petersburg, "ended", richmond) == "time_contradicts"
    russo = g.add_event("Russo-Turkish War", parse_time("1877"))
    stefano = g.add_event("Treaty of San Stefano", parse_time("1878"))
    assert g.add_link(russo, "ended", stefano) == "time_contradicts"   # no year of slack for 'ended'
    mayence = g.add_event("Capture of Mayence", None, source=("b", 1))
    campo = g.add_event("Treaty of Campo Formio", parse_time("October 17, 1797"))
    assert g.add_link(campo, "caused", mayence) is None           # its end undated: nothing to check yet
    g.infer_dates({("b", 1): 1792})
    assert g.prune_links() == 1
    assert g.out(campo) == [] and g.into(mayence) == [] and g.into(war) == [(treaty, "ended")]
    assert g.refused["time_contradicts_dated"] == 1
