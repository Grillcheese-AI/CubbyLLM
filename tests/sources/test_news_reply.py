"""Order 0024's replies on a synthetic archive: the four question shapes (EN, FR) are read with their windows; a reply
is dated, says what reports named, names no outlet and carries no 8-word run of an item; a statement is not read as
news; nothing in the window is a refusal, in the question's language.
Run: python -m pytest standin/tests/test_news_reply.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_reply as nr  # noqa: E402
from grounding import longest_run  # noqa: E402

TODAY = dt.date(2026, 10, 22)
FEEDS = [{"id": "dailyplanet-news", "url": "https://www.dailyplanet.example/rss", "kind": "news"},
         {"id": "gov-releases", "url": "https://gov.example/rss", "kind": "statement"}]


def test_questions_and_windows():
    assert nr.parse("What happened with Ada Lovelace on 21 October?")[:2] == ("en", "Ada Lovelace")
    assert nr.parse("Qu'a-t-on rapporté sur Gaza cette semaine ?")[:2] == ("fr", "Gaza")
    assert nr.parse("Who is the president of France?") is None
    assert nr.window("on 21 October", TODAY) == (dt.date(2026, 10, 21),) * 2
    assert nr.window("le 1er octobre", TODAY) == (dt.date(2026, 10, 1),) * 2
    assert nr.window("this week", TODAY) == (dt.date(2026, 10, 16), TODAY)


def test_a_dated_reply_in_its_own_words(tmp_path):
    rows = [{"key": "a", "feed": "dailyplanet-news", "kind": "news", "published": "2026-10-21", "first_seen": "2026-10-21T09:00:00",
             "link": "https://www.dailyplanet.example/a",
             "title": "Ada Lovelace opens the new Babbage engine works in Leeds, the Daily Planet has learned",
             "summary": "Ada Lovelace and Charles Babbage showed the engine to visitors in Leeds."},
            {"key": "b", "feed": "gov-releases", "kind": "statement", "published": "2026-10-21",
             "first_seen": "2026-10-21T09:00:00", "link": "https://gov.example/b",
             "title": "Minister welcomes Ada Lovelace to Leeds", "summary": ""}]
    (tmp_path / "2026-10-21.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    r = nr.answer("What happened with Ada Lovelace on 21 October?", tmp_path, TODAY, FEEDS)
    assert r["answered"] and r["tier"] == "reported" and r["reply"].startswith("On 21 October, reports about Ada Lovelace")
    assert "Leeds" in r["reply"] and "Planet" not in r["reply"] and "dailyplanet" not in r["reply"].lower()
    assert r["items"] == ["a"] and longest_run(r["reply"], rows[0]["title"] + " " + rows[0]["summary"]) < 8
    none = nr.answer("Que s'est-il passé avec Ada Lovelace le 20 octobre ?", tmp_path, TODAY, FEEDS)
    assert not none["answered"] and none["reply"] == "Je n'ai aucun article sur Ada Lovelace le 20 octobre."
