"""Sources stay internal (standin/audience.py, order 0016): the two audiences, the user's view of a record, the
leak guard at the mouth, and the HTTP surface.

Pinned: audit only for the configured token (none configured: never); a user's record carries no provenance and
no internal working, a profile no per-line sources; a source named in a clause that only names it is cut, named
anywhere else the reply is refused, a URL never said, the answer itself never a leak; a run of 8 words from a
stored passage is refused; a profile whose answer names a source refuses; over HTTP a user's /ask has no
sources, the audit token's has all of it, /audit/<id> returns the sealed chain to the audit token only, the
panel and the event streams refuse a user, /health and /worlds name no source or dataset, and a token in a
query string is masked in the log.
Run: python -m pytest standin/tests/test_audience.py -q
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

import audience as A  # noqa: E402
from books import BookIndex  # noqa: E402

TOKEN = "t0k3n-for-tests"


@pytest.fixture
def token(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBBY_AUDIT_TOKEN", TOKEN)
    monkeypatch.setattr(A, "TOKEN_FILE", tmp_path / "audit_token")
    return TOKEN


def test_audit_only_for_the_configured_token_and_never_without_one(monkeypatch, tmp_path):
    monkeypatch.delenv("CUBBY_AUDIT_TOKEN", raising=False)
    monkeypatch.setattr(A, "TOKEN_FILE", tmp_path / "audit_token")
    assert A.audience(None) == "user" and A.audience("") == "user" and A.audience("anything") == "user"
    (tmp_path / "audit_token").write_text("from-a-file\n", encoding="utf-8")
    assert A.audience("from-a-file") == "audit" and A.audience("from-a-fil") == "user"
    monkeypatch.setenv("CUBBY_AUDIT_TOKEN", TOKEN)
    assert A.audience(TOKEN) == "audit" and A.audience("from-a-file") == "user"
    h = {"X-Audit-Token": "a"}
    assert A.token_of(h) == "a" and A.token_of({"Authorization": "Bearer b"}) == "b"
    assert A.token_of({}, "/loop/stream?token=c&x=1") == "c" and A.token_of({}, "/health") is None
    assert A.redact('"GET /panel?token=c&x=1 HTTP/1.1" 200') == '"GET /panel?token=***&x=1 HTTP/1.1" 200'


def test_a_users_record_has_no_provenance_and_no_internal_working():
    rec = {"answer": "Romulus", "verified": True, "learned": [{"fact": "f", "source": "book:bob jones"}],
           "evidence": {"f": [{"author": "Bob Jones"}]}, "graph": {"nodes": []}, "task": {"walk_sources": ["wiki"]},
           "trace": ["Romulus is the founded by of Rome"],
           "profile": [{"relation": "founder", "value": "Romulus", "provenance": "wikidata", "fact": "x"}]}
    u = A.for_user(rec)
    assert set(u) == {"answer", "verified", "trace", "profile"} and u["profile"] == [{"relation": "founder", "value": "Romulus", "fact": "x"}]
    assert rec["learned"] and rec["profile"][0]["provenance"] == "wikidata"                 # the record itself keeps all


def test_the_guard_cuts_a_source_clause_refuses_a_source_said_otherwise_and_never_the_answer(tmp_path):
    g = A.LeakGuard()
    rec = {"answer": "Paris", "learned": [{"source": "book:bob jones+web:example.org"}]}
    assert g.check("Paris is the capital (source: Wikipedia).", rec) == ("Paris is the capital.", ["wikipedia"])
    assert g.check("Paris is the capital, according to bob jones, of France.", rec)[0] == "Paris is the capital of France."
    said, hits = g.check("Wikipedia says it is Paris.", rec)
    assert said is None and hits == ["wikipedia"]
    assert g.check("See https://example.org/paris for Paris.", rec)[0] is None
    assert g.check("It is Wikipedia.", {"answer": "Wikipedia"}) == ("It is Wikipedia.", [])     # the answer is not a leak
    assert g.check("Paris.", rec) == ("Paris.", [])
    idx = BookIndex(tmp_path / "b.sqlite")
    idx.add([(0, "the walls of the city were raised in a single summer by the king's men")], "h", "x", {})
    q = A.LeakGuard(index=idx)
    said, hits = q.check("The walls of the city were raised in a single summer.", {"answer": "x"})
    assert said is None and hits[0].startswith("quote: ")
    assert q.check("The city walls went up in one summer.", {"answer": "x"}) == ("The city walls went up in one summer.", [])


def test_a_profile_answers_item_that_names_a_source_goes_and_with_nothing_left_it_refuses():
    rec = {"answer": "Rome — founder: Romulus; source: wikidata (3)", "verified": True, "seed": "rome",
           "profile": [{"relation": "founder", "value": "Romulus"}]}
    A.mouth(rec, A.LeakGuard())                                                # the item that names it goes whole
    assert rec["answer"] == "Rome — founder: Romulus" and rec["verified"] and rec["guard"]["answer"] == ["wikidata"]
    rec = {"answer": "Rome — source: wikidata (3)", "verified": True, "seed": "rome", "profile": []}
    A.mouth(rec, A.LeakGuard())                                                # nothing left: refused
    assert rec["answer"] is None and rec["verified"] is False and rec["reason"] == "leak_guard"
    ok = {"answer": "Rome — founder: Romulus", "verified": True, "seed": "rome", "profile": [{"relation": "founder", "value": "Romulus"}]}
    assert A.mouth(ok, A.LeakGuard())["answer"] == "Rome — founder: Romulus" and "guard" not in ok


class _Chem:
    def to_dict(self):
        return {"state": {}}


class _Chat:
    chem = _Chem()


def _stub_brain(loop):
    return type("Brain", (), {"ask_loop": loop, "emitter": type("E", (), {"name": "stub"})(), "chat": _Chat(),
                              "worlds": {"wiki": loop.world}, "cortices": {}, "events": collections.deque()})()


def _call(port, path, body=None, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers=dict({"Content-Type": "application/json"}, **({"X-Audit-Token": token} if token else {})))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_over_http_a_user_sees_answers_and_the_audit_token_sees_the_chain(token, tmp_path, capsys):
    import test_books as tb
    from serve_api import serve_http
    loop = tb._loop(tmp_path, tb.DictSource({}))
    httpd = serve_http(_stub_brain(loop), "127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        code, rec = _call(port, "/ask", {"text": "Who founded Rome?"})
        assert code == 200 and rec["reply"] == "One account says Romulus." and rec["tier"] == "attributed"
        assert not set(rec) & {"learned", "evidence", "graph", "hash", "guard"}
        assert "bob" not in json.dumps(rec).lower() and "the city" not in json.dumps(rec).lower()
        code, full = _call(port, "/ask", {"text": "Who founded Rome?"}, token=token)
        assert code == 200 and full["evidence"] and full["learned"] is not None and full["hash"]
        assert _call(port, f"/audit/{full['id']}")[0] == 403
        code, chain = _call(port, f"/audit/{full['id']}", token=token)
        assert code == 200 and chain["sealed"] and chain["tier"] == "attributed"
        assert chain["facts"][0]["source"].startswith("book:bob jones") and chain["facts"][0]["passages"][0]["author"] == "Bob Jones"
        assert _call(port, "/audit/no-such-id", token=token)[0] == 404
        for path in ("/panel", "/loop/events", "/events", "/brain", "/map"):
            assert _call(port, path)[0] == 403, path
        code, page = _call(port, f"/panel?token={token}")
        assert code == 200 and b"X-Audit-Token" in page                          # the shim: its requests carry the token
        code, health = _call(port, "/health")
        assert code == 200 and health["ask_source"] is None and health["news"] is None
        assert _call(port, "/health", token=token)[1]["ask_source"] == "test"
        assert _call(port, "/worlds")[1]["worlds"] == {"all": len(loop.world.texts)}
    finally:
        httpd.shutdown()
    assert token not in capsys.readouterr().out                                      # the access log masks it
