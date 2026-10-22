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
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "tests"):
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


def test_plugin_facts_go_through_the_gate(tmp_path, monkeypatch):
    """A contradicting plugin fact is refused, ledgered and not stored; a new one is admitted with plugin:<name>; a
    fiction plugin's facts never reach the reality store."""
    import plugins_guard as guard
    from ledger import Ledger
    from plugins_index import discover
    from plugins_serve import mount, on_question
    from worlds import FactStore
    import test_plugins as tp
    monkeypatch.setenv("CUBBY_PLUGINS_LOCK", str(tmp_path / "lock.json"))
    monkeypatch.setenv("CUBBY_PLUGINS_DATA", str(tmp_path / "data"))
    d = tmp_path / "p" / "byron-facts"
    d.mkdir(parents=True)
    (d / "PLUGIN.md").write_text("---\nname: byron-facts\ndescription: facts about the poet Byron and his family\nexamples: "
                                 "[who was the mother of Byron]\nkinds: [source]\nentry: {source: \"s.py:S\"}\n---\n", encoding="utf-8")
    (d / "s.py").write_text("class S:\n    def __init__(self, ctx):\n        pass\n    def covers(self, q):\n        return 0.9 if "
                            "'Byron' in q else 0.0\n    def answer(self, q):\n        return ['Lady Byron is the mother of Byron', "
                            "'London is the birthplace of Byron']\n", encoding="utf-8")
    for p in discover([tmp_path / "p"]).plugins.values():
        guard.approve(p)
    store = FactStore(["Catherine Gordon is the mother of Byron"])
    store.provenance = {}
    b = brain()
    b.ask_loop.world, b.ledger = store, Ledger(tmp_path / "ledger.sqlite", key=b"k" * 32)
    mount(b, str(tmp_path / "p"), log=lambda *_: None)
    on_question(b, "who was the mother of Byron?")
    gate = {g["fact"]: g["status"] for g in b.plugins.last_admitted["gate"]}
    assert gate == {"Lady Byron is the mother of Byron": "contradiction", "London is the birthplace of Byron": "accepted"}
    assert "Lady Byron is the mother of Byron" not in store and "London is the birthplace of Byron" in store
    assert store.provenance["London is the birthplace of Byron"] == "plugin:byron-facts"
    rows = b.ledger.rows("plugin_fact")
    assert len(rows) == 1 and rows[0]["ok"] in (0, False)
    # a fiction plugin: its facts answer inside their world and never enter the reality store
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "1513.txt").write_text(tp.PLAY, encoding="utf-8")
    (pack / "1513_metadata.json").write_text(json.dumps({"title": "The Test Lovers"}), encoding="utf-8")
    monkeypatch.setenv("CUBBY_TEST_PLAYS", str(pack))
    tp.plugin(tmp_path / "f", "plays", kinds="world", extra='world: {route: fiction, data_env: CUBBY_TEST_PLAYS, glob: "[0-9]*.txt"}\n'
              'examples: [In The Test Lovers who is Juliet mother]\n')
    for p in discover([tmp_path / "f"]).plugins.values():
        guard.approve(p)
    b2 = brain()
    b2.ask_loop.world = FactStore(["Catherine Gordon is the mother of Byron"])
    mount(b2, str(tmp_path / "f"), log=lambda *_: None)
    on_question(b2, "In The Test Lovers, who is Juliet's mother?")
    last = b2.plugins.last_admitted
    assert last["fiction"] and any("Lady Capulet" in f for f in last["facts"]) and last["gate"] == []
    assert len(b2.ask_loop.world) == 1
