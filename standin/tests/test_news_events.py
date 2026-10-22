"""Order 0024's events by day on synthetic items: names are capitals that do not start a sentence; a title-cased
headline gives no names; one item through two feeds is one item; statements are not grouped; two outlets' items about
one event group, across languages by their names; a single shared name does not chain three stories; the day window
holds.
Run: python -m pytest standin/tests/test_news_events.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_events as ne  # noqa: E402


def item(key, title, day="2026-10-22", feed="a", summary="", link=None):
    return {"key": key, "feed": feed, "title": title, "summary": summary, "published": day,
            "first_seen": day + "T12:00:00+00:00", "link": link or f"https://{feed}.example/{key}"}


def test_features():
    names, words = ne.features("Earthquake strikes Bogotá as Colombia counts its dead")
    assert names == {"bogota", "colombia"} and "earthquake" in words
    names, words = ne.features("Humanitarian Needs Grow as Nepal Recovers from Catastrophic Floods", title_case=True)
    assert names == set() and {"nepal", "flood"} <= words                   # a title-case feed: no names from headlines
    house = [item(str(i), h, feed="ips") for i, h in enumerate(["Humanitarian Needs Grow as Nepal Recovers from Floods",
             "Winter Approaches as Afghanistan Faces Record Hunger", "Kazakhstan Spends Years Bringing Faiths Together",
             "Smartphone Bans Rob Female Journalists of Working Tools", "Ebola Spread Raises Fresh Urgency for Funding"])]
    one = [item("x", "Recomptage dans Vachon : le Parti Québécois remporte Vachon, Saint-Hubert", feed="fr")]
    assert ne.title_case_feeds(house + one) == {"ips"}
    assert ne.features("Floods", "Rescuers reached Pokhara on Monday.")[0] == {"pokhara", "monday"}


def test_one_item_through_two_feeds_is_one():
    rows = [item("1", "Ebola case confirmed in Lagos, Nigeria", feed="un", link="https://un.example/x"),
            item("2", "Ebola case confirmed in Lagos, Nigeria", feed="topic", link="https://un.example/x")]
    (g,) = ne.group(rows)
    assert len(g) == 1 and g[0]["feeds"] == ["un", "topic"]


def test_statements_are_not_grouped():
    rows = [item("1", "Minister Dabrusin met Provost in Ottawa", feed="gc"),
            item("2", "Minister Dabrusin met Provost in Ottawa today", feed="news")]
    groups = ne.group(rows, kinds={"gc": "statement"})
    assert [len(g) for g in groups] == [1] and groups[0][0]["feed"] == "news"


def test_two_outlets_one_event_across_languages():
    rows = [item("1", "Recount in Vachon: the Parti Québécois wins Vachon after Saint-Hubert recount", feed="en"),
            item("2", "Recomptage dans Vachon : le Parti Québécois remporte Vachon, Saint-Hubert", feed="fr"),
            item("3", "Storm closes schools in Halifax", feed="en")]
    groups = ne.group(rows)
    assert sorted(len(g) for g in groups) == [1, 2]
    assert {r["key"] for r in groups[0]} == {"1", "2"}


def test_one_shared_name_does_not_chain():
    # A-B share two names and B-C share two others; A and C share nothing: average linkage keeps C out.
    rows = [item("a", "Carney meets Erdogan in Ankara on trade"),
            item("b", "Carney and Erdogan in Ankara sign Washington trade note"),
            item("c", "Washington accuses Ottawa of interference, Washington says")]
    rows[2]["summary"] = "Officials in Washington and Ottawa traded notes; Carney was asked."
    sizes = sorted(len(g) for g in ne.group(rows))
    assert sizes in ([1, 2], [1, 1, 1]) and 3 not in sizes


def test_day_window():
    rows = [item("1", "Plague case suspected in Moscow, Russia", day="2026-10-10"),
            item("2", "Plague case suspected in Moscow, Russia", day="2026-10-20", link="https://a.example/z")]
    assert [len(g) for g in ne.group(rows)] == [1, 1]
