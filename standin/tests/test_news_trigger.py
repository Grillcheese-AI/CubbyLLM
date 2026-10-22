"""Order 0024's change trigger on synthetic items: a change word beside a person the world holds marks that person,
in English and French; a name the world does not hold, or no change word, marks nothing; an analysis is not read; a
GDELT event's URL words count, the URL never fetched; the marks file keeps each mark once; the ask loop's "as of"
reply carries the later report, and without marks (or off) it is the reply as before; a user never sees the marks.
Run: python -m pytest standin/tests/test_news_trigger.py
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_trigger as nt  # noqa: E402

KNOWN = {"ada lovelace", "acme holdings", "jean tremblay"}
known = lambda n: n.lower() in KNOWN                                              # noqa: E731


def item(key, title, feed="news", day="2026-10-21", summary=""):
    return {"key": key, "feed": feed, "title": title, "summary": summary, "published": day,
            "first_seen": day + "T10:00:00+00:00"}


def test_cues_and_names():
    assert nt.cues("Ada Lovelace resigns as chair") == ["resigned"]
    assert nt.cues("Jean Tremblay nommé à la tête de la société") == ["appointed"]
    assert nt.cues("Storm closes schools") == []
    assert nt.names("Board says Ada Lovelace resigns; Acme Holdings shares fall") == ["Board", "Ada Lovelace", "Acme Holdings"][1:]


def test_marks_from_items_and_gdelt():
    rows = [item("1", "Ada Lovelace resigns as chair of Acme Holdings"),
            item("2", "Jean Tremblay nommé ministre"),
            item("3", "Grace Hopper resigns"),                                       # not a name the world holds
            item("4", "Ada Lovelace visits a school"),                               # no change word
            item("5", "Ada Lovelace appointed to the board", feed="think")]          # an analysis feed: not read
    marks = nt.from_items(rows, known, kinds={"news": "news", "think": "analysis"})
    assert sorted((m["entity"], m["cue"]) for m in marks) == [("Acme Holdings", "resigned"), ("Ada Lovelace", "resigned"),
                                                               ("Jean Tremblay", "appointed")]
    ev = [{"id": "9", "day": "20261021", "a1_name": "ACME HOLDINGS", "a2_name": "",
           "source_url": "https://x.example/business/acme-holdings-acquired-by-rival-2026"},
          {"id": "8", "day": "20261021", "a1_name": "ACME HOLDINGS", "source_url": "https://x.example/acme-wins-award"}]
    (m,) = nt.from_gdelt(ev, known)
    assert (m["entity"], m["cue"], m["day"], m["source"]) == ("Acme Holdings", "acquired", "2026-10-21", "gdelt:9")


def test_scan_keeps_each_mark_once_and_changes_reads_them(tmp_path):
    import json
    (tmp_path / "2026-10-21.jsonl").write_text(json.dumps(item("1", "Ada Lovelace resigns as chair")) + "\n", encoding="utf-8")
    assert len(nt.scan(tmp_path, known)) == 1 and nt.scan(tmp_path, known) == []
    ch = nt.Changes(tmp_path)
    assert [m["day"] for m in ch.since("ada lovelace", 2019)] == ["2026-10-21"] and ch.since("Ada Lovelace", 2026) == []


def test_the_as_of_reply_carries_the_later_report():
    import ask
    loop = ask.AskLoop.__new__(ask.AskLoop)
    h = SimpleNamespace(triple=SimpleNamespace(subj="Ada Lovelace"))
    loop.news_changes = None
    assert loop._changed(h, "2019", {}) == ""                                       # off: the reply as before
    loop.news_changes = SimpleNamespace(since=lambda e, y: [{"day": "2026-10-21", "source": "item:1"}] if e == "Ada Lovelace" else [])
    rec = {}
    assert loop._changed(h, "2019", rec) == " Reports on 2026-10-21 say this may have changed." and rec["news_changes"]
    assert loop._changed(SimpleNamespace(triple=SimpleNamespace(subj="Someone Else")), "2019", {}) == ""
    from audience import for_user
    assert "news_changes" not in for_user({"reply": "x", "news_changes": rec["news_changes"]})
