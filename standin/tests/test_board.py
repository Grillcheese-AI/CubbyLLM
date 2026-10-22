"""Order 0027's control board, over a running server with a fake brain (no model):
- every /board/* response as the public audience leaves out a source the fixture marks internal; the audit audience
  gets it;
- --board off: every /board path answers 404; on: the server's other answers are unchanged;
- a section whose order is not done answers its empty state, with no numbers, on its own path and on Home;
- the page and its scripts name no host: it calls only the server it came from;
- the static and asset routes serve only their own files.
Run: python -m pytest standin/tests/test_board.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

SECRET, TOKEN = "secretwire", "tok-0027"
GRAPH = {"nodes": [{"id": 1, "kind": "question", "label": "q"}, {"id": 2, "kind": "plan", "label": "p"},
                   {"id": 3, "kind": "fetch", "label": "f", "source": SECRET},
                   {"id": 4, "kind": "hop", "label": "h", "provenance": SECRET},
                   {"id": 5, "kind": "fact", "label": "x", "subj": "Ada Lovelace", "rel": "born in", "obj": "London"},
                   {"id": 6, "kind": "vm", "label": "v", "verified": True}, {"id": 7, "kind": "answer", "label": "a", "verdict": "ok"}],
         "edges": []}
REC = {"id": "r1", "ts": "2026-10-22T10:00:00", "question": "Where was Ada Lovelace born?", "answer": "London",
       "verified": True, "reply": "London.", "wall_s": 0.12, "graph": GRAPH, "trace": ["London is the born in of Ada Lovelace"]}
PATHS = ["/board/home", "/board/system", "/board/journey/r1", "/board/watch", "/board/worlds", "/board/forecast",
         "/board/mood", "/board/agents"]


class Chem:
    def to_dict(self):
        return {"valence": 0.1}


def brain(board: bool):
    loop = SimpleNamespace(history=[dict(REC)], _forms=None, calls={"asked": 1}, world=SimpleNamespace(), source=None, news=None)
    return SimpleNamespace(board=board, ask_loop=loop, worlds={}, cortices=set(), emitter=SimpleNamespace(name="fake"),
                           chat=SimpleNamespace(chem=Chem()), events=[], harness=None)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import news_archive
    import serve_api
    feeds = [{"id": SECRET, "url": f"https://www.{SECRET}.example/rss", "publisher_hosts": [f"www.{SECRET}.example"],
              "status": "active", "licence": "publisher", "kind": "news", "every_h": 1},
             {"id": "open-releases", "url": "https://gov.example/rss", "status": "active", "licence": "open-government-canada"}]
    monkeypatch.setattr(news_archive, "feeds", lambda path=None: feeds)
    (tmp_path / "_polls.jsonl").write_text(json.dumps({"feed": SECRET, "at": "2026-10-22T09:00:00+00:00", "new": 3, "items": 9}) + "\n",
                                           encoding="utf-8")
    monkeypatch.setenv("CUBBY_NEWS_ARCHIVE", str(tmp_path))
    monkeypatch.setenv("CUBBY_AUDIT_TOKEN", TOKEN)
    monkeypatch.delenv("CUBBY_RUNS", raising=False)
    import board_api
    board_api._cache.clear()
    started = []

    def start(board: bool):
        httpd = serve_api.ThreadingHTTPServer(("127.0.0.1", 0), serve_api.make_handler(brain(board)))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        board_api._cache.clear()
        return f"http://127.0.0.1:{httpd.server_address[1]}"
    yield start
    for h in started:
        h.shutdown()


def fetch(base, path, audit=False):
    req = urllib.request.Request(base + path, headers={"X-Audit-Token": TOKEN} if audit else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def test_public_responses_never_name_an_internal_source(server):
    base = server(True)
    for p in PATHS:
        code, body = fetch(base, p)
        assert code == 200, p
        assert SECRET not in body.lower(), f"{p} leaks the internal source to the public audience"
        assert json.loads(body)["audience"] == "public"
    audit = "".join(fetch(base, p, audit=True)[1] for p in PATHS)
    assert SECRET in audit                                     # the audit view names it (the feeds, the journey's sources)


def test_off_is_404_and_on_changes_nothing_else(server):
    off, on = server(False), server(True)
    for p in ["/board", "/board/", "/board/home", "/board/static/board.css", "/board/assets/chat_still.webp"] + PATHS:
        assert fetch(off, p)[0] == 404, p
    assert fetch(on, "/board")[0] == 200 and "Cubby" in fetch(on, "/board")[1]
    for p in ["/health", "/state", "/worlds", "/nothing-here"]:
        assert fetch(off, p) == fetch(on, p), p


def test_sections_whose_order_is_not_done_show_no_numbers(server):
    base = server(True)
    home = json.loads(fetch(base, "/board/home")[1])["sections"]
    waiting = {s["id"]: s for s in home if not s["built"]}
    assert set(waiting) == {"watch", "worlds", "forecast", "mood", "agents"}
    for sid, s in waiting.items():
        assert "cards" not in s and s["message"].startswith("not built yet")
        body = json.loads(fetch(base, f"/board/{sid}")[1])
        assert body["built"] is False and re.sub(r"order \d{4}", "", body["message"]) == "not built yet ()" or "later" in body["message"]
        assert not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in body.values())
    built = {s["id"] for s in home if s["built"]}
    assert built == {"chat", "system"}


def test_the_journey_reads_the_record():
    import board_data as bd
    j = bd.journey(REC, {1: 100.0, 2: 100.01, 5: 100.05, 6: 100.06, 7: 100.07}, None, "user")
    assert [s["step"] for s in j["steps"]] == ["sense", "route", "recall", "check", "say"]
    assert j["steps"][2]["facts"] == ["Ada Lovelace · born in · London"] and "sources" not in j["steps"][2]
    assert j["steps"][1]["ms"] == 10 and j["said"] == "London."
    assert bd.journey(REC, {}, None, "audit")["steps"][2]["sources"] == [SECRET]


def test_the_page_calls_no_other_host_and_files_stay_in_their_folder(server):
    page = ROOT / "dashboard" / "board"
    for f in page.iterdir():
        text = f.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")   # a namespace, never fetched
        assert not re.search(r"https?://|//fonts\.|@import", text), f"{f.name} names a host"
    base = server(True)
    for p in ["/board/static/..%2Fserve_api.py", "/board/static/app.py", "/board/assets/cubby_sections.blend",
              "/board/assets/..%2F..%2FREADME.md"]:
        assert fetch(base, p)[0] == 404, p
    assert fetch(base, "/board/static/app.js")[0] == 200 and fetch(base, "/board/assets/system_still.webp")[0] == 200
