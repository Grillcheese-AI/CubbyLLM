"""Order 0027's review: the public scrub matches whole words against what is actually internal; Mood shows today's
chemistry; a training run shows the age of its last synced line and is stale after an hour; the background refresh
answers requests without rebuilding.
Run: python -m pytest standin/tests/test_board_review.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

import board_api  # noqa: E402
import board_data as bd  # noqa: E402


def real_like(chem=None):
    """A brain with the serve stack's generic world keys mounted (the probe's case)."""
    worlds = {k: SimpleNamespace(texts=[]) for k in ("facts", "wiki", "self", "reality:history", "data")}
    return SimpleNamespace(board=True, worlds=worlds, ask_loop=SimpleNamespace(history=[]), cortices=set(),
                           chat=SimpleNamespace(chem=chem), plugins=None)


def test_the_scrub_matches_whole_internal_words(monkeypatch):
    import news_archive
    monkeypatch.setattr(news_archive, "feeds", lambda path=None: [{"id": "globalwire", "url": "https://www.globalwire.example/rss"}])
    monkeypatch.delenv("CUBBY_WORLDS", raising=False)
    names = board_api.internal_names(real_like())
    assert "wikidata5m" in names and "globalwire" in names and not {"facts", "wiki", "self", "data"} & names
    keep = ["I don't have facts about that", "3.4M facts held", "Cubby talks about itself", "trained on wikis",
            "Weather warnings in effect", "a globalwireless router"]
    gone = ["trained on wikidata5m", "from Wikidata5M today", "read on globalwire.example", "globalwire: 3 new"]
    out = board_api.scrub({"keep": keep, "gone": gone}, names)
    assert out["keep"] == keep and out["gone"] == ["[internal]"] * len(gone)


def test_dataset_names_from_the_registry(tmp_path, monkeypatch):
    import sqlite3
    con = sqlite3.connect(tmp_path / "registry.sqlite")
    con.execute("CREATE TABLE datasets(name TEXT PRIMARY KEY, tree TEXT, kind TEXT, status TEXT, reason TEXT, report TEXT, at TEXT)")
    con.executemany("INSERT INTO datasets(name) VALUES (?)", [("Grand Archive Collection",), ("misc",)])
    con.commit()
    con.close()
    monkeypatch.setenv("CUBBY_WORLDS", str(tmp_path))
    names = board_api.internal_names(real_like())
    assert "grand archive collection" in names and "misc" not in names


def test_mood_shows_todays_chemistry():
    from neurochemistry_probe import chem
    c = chem()
    m = bd.mood(real_like(c), "user")
    st = c.to_dict()
    assert m["built"] and m["bpm"] == round(40 + 55 * st["arousal"])
    assert m["cube"]["corner"] == c.corner_position()["corner"] and len(m["cube"]["pos"]) == 3
    assert set(m["levels"]) <= set(bd.LEVELS) and m["arcs"]["built"] is False and "0020" in m["arcs"]["message"]
    assert bd.mood(SimpleNamespace(), "user")["built"] is False


def test_a_run_shows_its_age_and_goes_stale(tmp_path, monkeypatch):
    for name, age_s in (("fresh_run", 9 * 60), ("old_run", 3 * 3600)):
        d = tmp_path / name
        d.mkdir()
        m = d / f"{name}_metrics.jsonl"
        m.write_text('{"step": 10, "loss": 2.5}\n{"step": 20, "loss": 2.25}\n', encoding="utf-8")
        t = time.time() - age_s
        os.utime(m, (t, t))
    monkeypatch.setenv("CUBBY_RUNS", str(tmp_path))
    runs = {r["run"]: r for r in bd.training()}
    assert runs["fresh_run"]["stale"] is False and runs["fresh_run"]["as_of"].endswith("9 min ago")
    assert runs["old_run"]["stale"] is True and runs["old_run"]["as_of"].endswith("3 h ago")
    assert runs["fresh_run"]["step"] == 20 and runs["fresh_run"]["loss"] == 2.25


def test_requests_read_the_background_build(monkeypatch):
    b = real_like()
    board_api._cache.clear()
    stop = threading.Event()
    board_api.start_refresh(b, every=0.05, stop=stop)
    try:
        for _ in range(200):
            if {"home:user", "system:user", "names"} <= board_api._refreshing:
                break
            time.sleep(0.02)
        assert {"home:user", "system:user", "names"} <= board_api._refreshing

        def boom(*_a, **_k):
            raise AssertionError("a request rebuilt Home")
        monkeypatch.setattr(bd, "home", boom)
        code, out = board_api.handle(b, "/board/home", "user")
        assert code == 200 and out["sections"]
    finally:
        stop.set()
        time.sleep(0.1)
        board_api._refreshing.clear()
        board_api._cache.clear()


def test_the_cube_centre_names_no_corner():
    class Rest:
        def to_dict(self):
            return {"arousal": 0.2}

        def corner_position(self):
            return {"corner": "spent", "intensity": 0.0, "runner_up": "distress", "pos": (0.5, 0.5, 0.5)}
    c = bd.mood(real_like(Rest()), "user")["cube"]
    assert c["at_rest"] and c["corner"] is None and c["runner_up"] is None


def test_the_poller_tile_reads_the_servers_archive(tmp_path, monkeypatch):
    import json
    import news_archive
    monkeypatch.delenv("CUBBY_NEWS_ARCHIVE", raising=False)
    monkeypatch.setattr(news_archive, "feeds", lambda path=None: [{"id": "wire-a", "url": "https://a.example/rss", "status": "active"}])
    (tmp_path / "_polls.jsonl").write_text(json.dumps({"feed": "wire-a", "at": "2026-10-22T09:00:00+00:00", "items": 4}) + "\n",
                                           encoding="utf-8")
    p = bd.system(SimpleNamespace(news_archive=str(tmp_path)), "user")["poller"]
    assert p["up"] == 1 and p["last_poll_ago"]
