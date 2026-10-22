"""Order 0028 in the server: plugins off, nothing the server answers changes; an internal plugin's name is absent from
every public board response and from a public /ask record, and present for the audit audience; Approve / Pin /
Disable answer the audit audience only; a question wakes its plugin and the record says so (audit view).
Run: python -m pytest standin/tests/test_plugins_serve.py
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

TOKEN, SECRET = "tok-0028", "hushfeed"
PATHS = ["/board/plugins", "/board/plugins/route?q=is+there+a+hushfeed+secret+story", f"/board/plugins/doc?name={SECRET}",
         "/board/home", "/board/system"]


def brain():
    hist = []

    def ask(text, **_kw):
        rec = {"id": f"r{len(hist)}", "question": text, "answer": None, "verified": False, "reason": "no_fact"}
        hist.append(rec)
        return rec
    loop = SimpleNamespace(history=hist, _forms=None, calls={"asked": 0}, world=SimpleNamespace(), source=None, news=None, ask=ask)
    return SimpleNamespace(board=True, ask_loop=loop, worlds={}, cortices=set(), emitter=SimpleNamespace(name="fake"),
                           chat=SimpleNamespace(chem=SimpleNamespace(to_dict=lambda: {"v": 1})), events=[], harness=None)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import board_api
    import plugins_guard as guard
    import serve_api
    from plugins_index import discover
    from plugins_serve import mount
    monkeypatch.setenv("CUBBY_AUDIT_TOKEN", TOKEN)
    monkeypatch.setenv("CUBBY_PLUGINS_LOCK", str(tmp_path / "lock.json"))
    monkeypatch.setenv("CUBBY_PLUGINS_DATA", str(tmp_path / "data"))
    dirs = tmp_path / "plugins"
    shutil.copytree(ROOT / "plugins" / "units", dirs / "units")
    d = dirs / SECRET
    d.mkdir()
    (d / "PLUGIN.md").write_text(f"---\nname: {SECRET}\ndescription: the hushfeed secret story source\nexamples: [is there a "
                                 f"hushfeed secret story]\nkinds: [source]\naudience: internal\nentry: {{source: \"s.py:S\"}}\n---\n",
                                 encoding="utf-8")
    (d / "s.py").write_text("class S:\n    def __init__(self, ctx):\n        pass\n    def covers(self, q):\n        return 0.9 if "
                            "'secret' in q else 0.0\n    def answer(self, q):\n        return ['the hushfeed secret is 42']\n",
                            encoding="utf-8")
    for p in discover([dirs]).plugins.values():
        guard.approve(p)
    started = []

    def start(with_plugins: bool):
        b = brain()
        if with_plugins:
            mount(b, str(dirs), log=lambda *_: None)
        httpd = serve_api.ThreadingHTTPServer(("127.0.0.1", 0), serve_api.make_handler(b))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        board_api._cache.clear()
        return f"http://127.0.0.1:{httpd.server_address[1]}", b
    yield start
    for h in started:
        h.shutdown()


def call(base, path, audit=False, data=None):
    req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"X-Audit-Token": TOKEN} if audit else {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def test_off_changes_nothing(server):
    (off, _), (on, _) = server(False), server(True)
    for p in ["/health", "/state", "/worlds", "/board/home", "/board/system", "/nothing"]:
        assert call(off, p) == call(on, p), p
    assert json.loads(call(off, "/board/plugins")[1])["built"] is False


def test_an_internal_plugin_is_absent_from_public_responses(server):
    base, b = server(True)
    for p in PATHS:
        code, body = call(base, p)
        assert code == 200 and SECRET not in body.lower(), p
    code, body = call(base, "/ask", data={"text": "is there a secret story tonight?"})   # the user's own words are echoed
    assert code == 200 and SECRET not in body.lower() and "42" not in body and "plugin" not in body
    assert b.ask_loop.history[-1]["plugins"] == [f"plugin:{SECRET}"]            # the record keeps it (audit view)
    assert b.ask_loop.history[-1]["plugin_facts"]["facts"] == ["the hushfeed secret is 42"]
    audit = "".join(call(base, p, audit=True)[1] for p in PATHS)
    assert SECRET in audit


def test_actions_are_for_the_audit_audience(server):
    base, b = server(True)
    assert call(base, "/board/plugins/units/pin", data={})[0] == 403
    code, body = call(base, "/board/plugins/units/pin", audit=True, data={})
    assert code == 200 and json.loads(body)["status"] == "pinned" and "units" in b.plugins.loaded
    assert json.loads(call(base, "/board/plugins/units/disable", audit=True, data={})[1])["status"] == "disabled"
    assert "units" not in b.plugins.loaded
