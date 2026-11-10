"""The dashboards behind cubby-auth (cubby_serve/auth.py, auth_gate.py), over a running server with a fake brain and a
fake cubby-auth on loopback that speaks its protocol (discovery, token, introspection, lock, revocation):
- off (no --auth-issuer): no /auth route, the audit token opens the audit view as before;
- on: every dashboard path needs a sign-in (a page goes to the sign-in, a call gets 401), the token opens nothing;
  other paths are unchanged;
- the sign-in: PKCE S256, state bound to the browser, the answer's iss checked, the ID token's nonce checked;
- the audience comes from the account: board:audit -> audit unless the page asks as the public; without it, public;
- writes need the page's own Origin and board:write; a locked sign-in ends the session; API keys go through the
  same check; cubby-auth unreachable -> 503, never open.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import urllib.error
import urllib.request
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

TOKEN, SECRET = "tok-auth", "s3cret"
OWNER = {"sub": "u_owner", "name": "Nick", "user_type": "owner", "roles": ["operator"],
         "permissions": ["board:read", "board:audit", "board:write"]}
VIEWER = {"sub": "u_view", "name": "Ann", "user_type": "admin", "roles": ["viewer"], "permissions": ["board:read"]}


class FakeAuth:
    """cubby-auth's side, enough to be the other end of the protocol."""

    def __init__(self):
        self.codes, self.tokens, self.calls, self.locked = {}, {}, [], set()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/.well-known/openid-configuration":
                    i = fake.issuer
                    self._json(200, {"issuer": i, "authorization_endpoint": i + "/authorize", "token_endpoint": i + "/token",
                                     "introspection_endpoint": i + "/introspect", "revocation_endpoint": i + "/revoke",
                                     "end_session_endpoint": i + "/logout"})
                else:
                    self._json(404, {})

            def do_POST(self):
                form = {k: v[0] for k, v in parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()).items()}
                basic = base64.b64decode(self.headers.get("Authorization", "Basic ")[6:]).decode()
                fake.calls.append((self.path, form))
                if basic != f"board:{SECRET}":
                    return self._json(401, {"error": "invalid_client"})
                if self.path == "/token":
                    c = fake.codes.pop(form.get("code"), None)
                    verifier = base64.urlsafe_b64encode(sha256(form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
                    if c is None or c["challenge"] != verifier:
                        return self._json(400, {"error": "invalid_grant"})
                    claims = {"iss": fake.issuer, "aud": "board", "nonce": c["nonce"], "exp": 4_000_000_000, "sub": c["who"]["sub"]}
                    idt = "e30." + base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode() + ".sig"
                    at = f"at_{len(fake.tokens)}"
                    fake.tokens[at] = c["who"]
                    return self._json(200, {"access_token": at, "id_token": idt, "token_type": "Bearer"})
                if self.path == "/introspect":
                    who = fake.tokens.get(form["token"])
                    if who is None or form["token"] in fake.locked:
                        return self._json(200, {"active": False, "reason": "locked" if who else "unknown"})
                    need = form.get("require", "").split()
                    missing = [p for p in need if p not in who["permissions"]]
                    kind = "api_key" if form["token"].startswith("cbk_") else "access_token"
                    return self._json(200, {"active": True, "token_type": kind, **who, "allowed": not missing, "missing": missing})
                if self.path == "/lock":
                    fake.locked.add(form["token"])
                    return self._json(200, {"locked": True})
                if self.path == "/revoke":
                    fake.tokens.pop(form["token"], None)
                    return self._json(200, {})
                self._json(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.issuer = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def authorize(self, url: str, who: dict) -> str:
        """The browser at cubby-auth's sign-in: a passkey later, the redirect back with a code."""
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        assert (q["code_challenge_method"], q["response_type"], q["client_id"]) == ("S256", "code", "board")
        code = f"code_{len(self.codes)}"
        self.codes[code] = {"challenge": q["code_challenge"], "nonce": q["nonce"], "who": who}
        return f"{q['redirect_uri']}?" + urlencode({"code": code, "state": q["state"], "iss": self.issuer})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


OPEN = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))


def call(url, headers=None, data=None):
    req = urllib.request.Request(url, headers=headers or {}, data=data, method="POST" if data is not None else "GET")
    try:
        with OPEN.open(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def cookie_of(headers) -> str:
    return headers.get("Set-Cookie", "").split(";", 1)[0]


@pytest.fixture()
def setup(monkeypatch):
    from cubby_serve import api
    from cubby_serve.auth import CubbyAuth
    from cubby_serve.board import api as board_api
    monkeypatch.setenv("CUBBY_AUDIT_TOKEN", TOKEN)
    board_api._cache.clear()
    fake, servers = FakeAuth(), []

    def start(on=True, issuer=None):
        loop = SimpleNamespace(history=[], _forms=None, calls={"asked": 0}, world=SimpleNamespace(), source=None, news=None)
        brain = SimpleNamespace(board=True, ask_loop=loop, worlds={}, cortices=set(), emitter=SimpleNamespace(name="fake"),
                                chat=SimpleNamespace(chem=SimpleNamespace(to_dict=lambda: {})), events=[], harness=None, auth=None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), api.make_handler(brain))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        if on:
            brain.auth = CubbyAuth(issuer=issuer or fake.issuer, client_id="board", client_secret=SECRET,
                                   redirect_uri=base + "/auth/callback", timeout=3)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return base
    yield fake, start
    for s in servers + [fake.httpd]:
        s.shutdown()


def sign_in(fake, base, who=OWNER, nxt="/board"):
    code, h, _ = call(base + "/auth/login?next=" + nxt)
    assert code == 302
    browser = cookie_of(h)
    code, h2, _ = call(fake.authorize(h["Location"], who), {"Cookie": browser})
    assert (code, h2["Location"]) == (302, nxt)
    return cookie_of(h2)


def test_off_changes_nothing(setup):
    _fake, start = setup
    base = start(on=False)
    assert call(base + "/auth/me")[0] == 404
    code, _h, body = call(base + "/board/mood", {"X-Audit-Token": TOKEN})
    assert code == 200 and json.loads(body)["audience"] == "audit"


def test_on_every_dashboard_needs_the_sign_in_and_the_token_opens_nothing(setup):
    _fake, start = setup
    base = start()
    code, h, _ = call(base + "/board", {"Accept": "text/html"})
    assert (code, h["Location"]) == (302, "/auth/login?next=%2Fboard")
    for path in ("/board/mood", "/panel", "/loop/events?since=0", "/audit/x", "/events"):
        code, _h, body = call(base + path, {"X-Audit-Token": TOKEN})
        assert code == 401 and json.loads(body)["error"] == "sign in", path
    assert call(base + "/health")[0] != 401                     # the front end's paths are not the dashboards'


def test_the_sign_in_is_bound_and_checked(setup):
    fake, start = setup
    base = start()
    def start_one(nxt="/board"):
        code, h, _ = call(base + "/auth/login?next=" + nxt)
        return fake.authorize(h["Location"], OWNER), cookie_of(h)
    back, browser = start_one()
    assert call(back)[0] == 400                                 # no browser cookie: started elsewhere
    assert call(back, {"Cookie": browser})[0] == 400            # and that attempt burned its state
    back, browser = start_one()
    assert call(back.replace("iss=", "iss=x"), {"Cookie": browser})[0] == 400
    back, browser = start_one()
    fake.codes[list(fake.codes)[-1]]["nonce"] = "other"       # an ID token minted for another sign-in
    assert call(back, {"Cookie": browser})[0] == 403
    back, h = start_one("//evil.example")
    h = {"Set-Cookie": h}
    code, h2, _ = call(back, {"Cookie": cookie_of(h)})
    assert (code, h2["Location"]) == (302, "/")                 # an open redirect is cut to this server's root
    assert call(back, {"Cookie": cookie_of(h)})[0] == 400       # a state is used once
    assert "Max-Age=600" not in h2.get("Set-Cookie", "")


def test_the_audience_comes_from_the_account(setup):
    fake, start = setup
    base = start()
    owner, viewer = sign_in(fake, base), sign_in(fake, base, VIEWER)
    get = lambda c, extra=None: json.loads(call(base + "/board/mood", {"Cookie": c, **(extra or {})})[2])["audience"]
    assert get(owner) == "audit"
    assert get(owner, {"X-Cubby-Audience": "user"}) == "public"
    assert get(viewer, {"X-Cubby-Audience": "audit"}) == "public"
    assert get(viewer, {"X-Audit-Token": TOKEN}) == "public"   # the shared token adds nothing on a dashboard path
    me = json.loads(call(base + "/auth/me", {"Cookie": owner})[2])
    assert (me["name"], me["audit"], me["kind"]) == ("Nick", True, "session")
    touch = [f.get("touch") for p, f in fake.calls if p == "/introspect"]
    assert "false" in touch                                     # a call with no input behind it does not keep it awake


def test_writes_need_the_origin_and_board_write(setup):
    fake, start = setup
    base = start()
    owner, viewer = sign_in(fake, base), sign_in(fake, base, VIEWER)
    assert call(base + "/board/plugins/x/approve", {"Cookie": owner}, b"{}")[0] == 403       # no Origin
    code, _h, body = call(base + "/board/plugins/x/approve", {"Cookie": viewer, "Origin": base}, b"{}")
    assert code == 403 and json.loads(body)["missing"] == ["board:write"]
    assert call(base + "/board/plugins/x/approve", {"Cookie": owner, "Origin": base}, b"{}")[0] not in (401, 403)


def test_lock_ends_it_and_api_keys_use_the_same_check(setup):
    fake, start = setup
    base = start()
    owner = sign_in(fake, base)
    assert call(base + "/auth/lock", {"Cookie": owner}, b"{}")[0] == 403                    # cross-site
    code, _h, body = call(base + "/auth/lock", {"Cookie": owner, "Origin": base}, b"{}")
    assert code == 200 and json.loads(body)["locked"] is True
    assert call(base + "/board/mood", {"Cookie": owner})[0] == 401
    other = sign_in(fake, base)
    fake.locked.add(next(t for t, w in fake.tokens.items() if w is OWNER and t not in fake.locked))
    code, h, _ = call(base + "/board/mood", {"Cookie": other})
    assert code == 401 and "Max-Age=0" in h.get("Set-Cookie", "")                              # locked elsewhere
    fake.tokens["cbk_1"] = VIEWER
    code, _h, body = call(base + "/board/mood", {"Authorization": "Bearer cbk_1"})
    assert code == 200 and json.loads(body)["audience"] == "public"
    assert call(base + "/board/mood", {"Authorization": "Bearer cbk_bad"})[0] == 401


def test_unreachable_is_closed(setup):
    _fake, start = setup
    base = start(issuer="http://127.0.0.1:9")
    assert call(base + "/board/mood")[0] == 401                 # no session: the sign-in, nothing asked yet
    assert call(base + "/auth/login")[0] == 503
    assert call(base + "/board/mood", {"Authorization": "Bearer cbk_1"})[0] == 503
    assert call(base + "/health")[0] != 503
