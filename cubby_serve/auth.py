"""cubby_serve/auth.py -- cubby-auth (the one sign-in for every Cubby app) for this server's dashboards: a port of
cubby-auth's Python client (`clients/python/cubby_auth_client.py`, ASGI) to the stdlib server here (http.server).

The same protocol and the same checks: OpenID Connect's code flow with PKCE (S256) and a `state` bound to the
browser by a cookie, the answer's `iss` checked (RFC 9207), the ID token's `iss`, `aud`, `nonce` and `exp` checked
(it comes straight from the token endpoint, so not its signature: OpenID Connect Core 3.1.3.7, rule 6), then every
request introspected (RFC 7662), so a lock, a sign-out everywhere, a disabled account or a removed role applies to
the very next request. Programs send `Authorization: Bearer cbk_...` (an API key from cubby-auth's account page),
checked the same way. Writes made with the session cookie must come from this server's own pages (`Origin`).
Server-to-server calls go to the back channel (`http://127.0.0.1:8791`): this Python's TLS cannot do the
post-quantum key exchange cubby-auth's public listener requires.

Left out from the ASGI client: the per-session data key (this server keeps nothing per person to seal with it)
and the ContextVar (the handler carries the identity as `h.cubby`). Sessions live in memory: a restart signs
everyone out of the dashboards (not out of cubby-auth).

    GET  /auth/login?next=/board   to cubby-auth's sign-in
    GET  /auth/callback            back from it (the redirect URI's path)
    POST /auth/lock                "lock now": the cubby-auth sign-in locks, every app's session ends
    POST /auth/logout              ends this server's session; answers with cubby-auth's sign-out page

Wired: WIRED (`cubby_serve/auth_gate.py`, when the server is given `--auth-issuer`).
"""
from __future__ import annotations

import base64
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from cubby_serve.auth_parts import CubbyAuthError, Identity, Reply, _b64url_json, _query, cookies, local_path

__wiring__ = "WIRED"

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
ACTIVITY_HEADER = "X-Cubby-Activity"     # set by the page on calls that follow the person's own input
LOGIN_TTL, MAX_PENDING, MAX_SESSIONS = 600, 10_000, 10_000


@dataclass
class _Session:
    access_token: str
    identity: Identity
    created: float


class CubbyAuth:
    def __init__(self, *, issuer: str, client_id: str, client_secret: str, redirect_uri: str,
                 backchannel: str | None = None, cookie: str = "cubby_serve", timeout: float = 10.0):
        self.issuer = issuer.rstrip("/")
        self.client_id, self._secret = client_id, client_secret
        self.redirect_uri = redirect_uri
        u = urlsplit(redirect_uri)
        self.origin, self.callback_path, self.secure = f"{u.scheme}://{u.netloc}", u.path, u.scheme == "https"
        self.cookie = ("__Host-" + cookie) if self.secure else cookie
        self.login_cookie = self.cookie + "_login"
        self.back = (backchannel or issuer).rstrip("/")
        self.timeout = timeout
        self._open = build_opener(ProxyHandler({}))      # never through a proxy: the back channel is loopback
        self._conf: dict | None = None
        self._pending: dict[str, dict] = {}
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ---- talking to cubby-auth (server to server) ----

    def _http(self, url: str, data: dict | None = None) -> tuple[int, dict]:
        req = Request(url, data=urlencode(data).encode("ascii") if data is not None else None)
        if data is not None:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            basic = base64.b64encode(f"{self.client_id}:{self._secret}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {basic}")
        try:
            with self._open.open(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}
        except (URLError, OSError, ValueError) as e:
            raise CubbyAuthError(f"cannot reach cubby-auth at {self.back}: {e}") from e

    def discover(self) -> dict:
        if self._conf is None:
            code, conf = self._http(f"{self.back}/.well-known/openid-configuration")
            if code != 200:
                raise CubbyAuthError(f"cubby-auth at {self.back} answered {code} for its configuration")
            if conf.get("issuer", "").rstrip("/") != self.issuer:
                raise CubbyAuthError(f"cubby-auth at {self.back} says its issuer is {conf.get('issuer')!r}, not {self.issuer!r}")
            for k in ("token_endpoint", "introspection_endpoint", "revocation_endpoint"):
                conf[k] = conf[k].replace(self.issuer, self.back, 1)
            conf["lock_endpoint"] = f"{self.back}/lock"
            self._conf = conf
        return self._conf

    def _post(self, endpoint: str, data: dict) -> tuple[int, dict]:
        return self._http(self.discover()[endpoint], data)

    def introspect(self, token: str, require=(), touch: bool = True) -> dict:
        data = {"token": token}
        if require:
            data["require"] = " ".join(require)
        if not touch:
            data["touch"] = "false"
        code, got = self._post("introspection_endpoint", data)
        if code == 401:
            raise CubbyAuthError("cubby-auth refused this server's client secret")
        if code != 200:
            raise CubbyAuthError(f"cubby-auth answered {code} to an introspection")
        return got

    # ---- the sign-in ----

    def _set(self, name: str, value: str, max_age: int | None = None) -> tuple[str, str]:
        c = f"{name}={value}; Path=/; HttpOnly; SameSite=Lax" + ("; Secure" if self.secure else "")
        return ("Set-Cookie", c + (f"; Max-Age={max_age}" if max_age is not None else ""))

    def login(self, h) -> Reply:
        conf, now, q = self.discover(), time.time(), _query(h)
        browser = cookies(h).get(self.login_cookie, "")
        if len(browser) != 43:                           # one id per browser while it has sign-ins in flight
            browser = secrets.token_urlsafe(32)
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        with self._lock:
            for k in [k for k, v in self._pending.items() if now - v["at"] > LOGIN_TTL]:
                del self._pending[k]
            while len(self._pending) >= MAX_PENDING:
                del self._pending[next(iter(self._pending))]
            self._pending[state] = {"browser": browser, "nonce": nonce, "verifier": verifier, "at": now,
                                    "next": local_path(q.get("next"))}
        challenge = base64.urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        args = {"client_id": self.client_id, "redirect_uri": self.redirect_uri, "response_type": "code", "scope": "openid",
                "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256"}
        if q.get("prompt") in ("login", "select_account"):
            args["prompt"] = q["prompt"]
        # the cookie binds the sign-in to this browser: a code someone else got cannot be planted here
        return Reply.redirect(f"{conf['authorization_endpoint']}?{urlencode(args)}", [self._set(self.login_cookie, browser, LOGIN_TTL)])

    def callback(self, h) -> Reply:
        q = _query(h)
        with self._lock:
            pending = self._pending.pop(q.get("state", ""), None)
        if pending is None or time.time() - pending["at"] > LOGIN_TTL:
            return Reply.text(400, "This sign-in is unknown here, or it took too long. Start again.")
        if not hmac.compare_digest(cookies(h).get(self.login_cookie, ""), pending["browser"]):
            return Reply.text(400, "This sign-in was started in another browser. Start again here.")
        if q.get("iss", "").rstrip("/") != self.issuer:
            return Reply.text(400, "The sign-in came back from an unexpected server.")
        if "error" in q:
            return Reply.text(403, f"The sign-in did not finish: {q.get('error_description') or q.get('error')}")
        code, tok = self._post("token_endpoint", {"grant_type": "authorization_code", "code": q.get("code", ""),
                                                   "redirect_uri": self.redirect_uri, "code_verifier": pending["verifier"]})
        if code != 200 or "id_token" not in tok or "access_token" not in tok:
            return Reply.text(403, "cubby-auth did not accept the sign-in. Start again.")
        try:
            claims = _b64url_json(tok["id_token"].split(".")[1])
        except (IndexError, ValueError):
            return Reply.text(403, "The ID token did not match this sign-in.")
        aud = claims.get("aud")
        if (claims.get("iss", "").rstrip("/") != self.issuer
                or (aud != self.client_id and not (isinstance(aud, list) and self.client_id in aud))
                or not hmac.compare_digest(str(claims.get("nonce", "")), pending["nonce"])
                or claims.get("exp", 0) < time.time()):
            return Reply.text(403, "The ID token did not match this sign-in.")
        i = self.introspect(tok["access_token"])
        if not i.get("active"):
            return Reply.text(403, "cubby-auth says this account cannot use this server.")
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions.pop(cookies(h).get(self.cookie, ""), None)   # this browser's previous session, replaced
            while len(self._sessions) >= MAX_SESSIONS:
                del self._sessions[next(iter(self._sessions))]
            self._sessions[sid] = _Session(tok["access_token"], self.identity(i, "session"), time.time())
        return Reply.redirect(pending["next"], [self._set(self.cookie, sid), self._set(self.login_cookie, "", 0)])

    @staticmethod
    def identity(i: dict, kind: str) -> Identity:
        return Identity(sub=i["sub"], name=i.get("name", ""), user_type=i.get("user_type", ""), roles=list(i.get("roles", [])),
                        permissions=list(i.get("permissions", [])), kind=kind, key_id=i.get("key_id"),
                        tenant=i.get("tenant"), tenant_name=i.get("tenant_name"), tenant_role=i.get("tenant_role"))

    # ---- each request ----

    def check(self, h, need=()) -> Identity | Reply:
        """The identity for this request, or the answer that refuses it."""
        method = h.command
        authz = h.headers.get("Authorization") or ""
        if authz.lower().startswith("bearer "):
            i = self.introspect(authz[7:].strip(), need)
            if not i.get("active"):
                return Reply.json(401, {"error": "invalid API key", "reason": i.get("reason")},
                                  [("WWW-Authenticate", 'Bearer error="invalid_token"')])
            if not i.get("allowed", True):
                return Reply.json(403, {"error": "missing permission", "missing": i.get("missing", [])})
            return self.identity(i, "api_key")
        sid = cookies(h).get(self.cookie, "")
        with self._lock:
            s = self._sessions.get(sid)
        if s is None:
            return self.sign_in(h)
        if method in WRITE_METHODS and h.headers.get("Origin") != self.origin:
            return Reply.json(403, {"error": "cross-site request refused"})
        # activity: a write, a page load, or a call right after the person's own input; a page polling on its
        # own does not keep a sign-in awake
        touch = method in WRITE_METHODS or bool(h.headers.get(ACTIVITY_HEADER)) or "text/html" in (h.headers.get("Accept") or "")
        i = self.introspect(s.access_token, need, touch)
        if not i.get("active"):
            with self._lock:
                self._sessions.pop(sid, None)
            return self.sign_in(h, clear=True)
        fresh = self.identity(i, "session")
        s.identity.roles, s.identity.permissions = fresh.roles, fresh.permissions
        s.identity.name, s.identity.user_type = fresh.name or s.identity.name, fresh.user_type or s.identity.user_type
        s.identity.tenant_name, s.identity.tenant_role = fresh.tenant_name, fresh.tenant_role
        if not i.get("allowed", True):
            return Reply.json(403, {"error": "missing permission", "missing": i.get("missing", [])})
        return s.identity

    def sign_in(self, h, clear: bool = False) -> Reply:
        login = "/auth/login?" + urlencode({"next": local_path(h.path)})
        extra = [self._set(self.cookie, "", 0)] if clear else []
        if h.command == "GET" and "text/html" in (h.headers.get("Accept") or ""):
            return Reply.redirect(login, extra)
        return Reply.json(401, {"error": "sign in", "login": login}, extra)

    # ---- lock and sign-out, from the page ----

    def _end(self, h) -> _Session | None:
        with self._lock:
            return self._sessions.pop(cookies(h).get(self.cookie, ""), None)

    def lock(self, h) -> Reply:
        if h.headers.get("Origin") != self.origin:
            return Reply.json(403, {"error": "cross-site request refused"})
        s, locked = self._end(h), False
        if s is not None:
            code, got = self._post("lock_endpoint", {"token": s.access_token})
            locked = code == 200 and bool(got.get("locked"))
        return Reply.json(200, {"locked": locked, "login": "/auth/login"}, [self._set(self.cookie, "", 0)])

    def logout(self, h) -> Reply:
        if h.headers.get("Origin") != self.origin:
            return Reply.json(403, {"error": "cross-site request refused"})
        s = self._end(h)
        if s is not None:
            self._post("revocation_endpoint", {"token": s.access_token, "token_type_hint": "access_token"})
        return Reply.json(200, {"signed_out": True, "end_session": self.discover().get("end_session_endpoint")},
                          [self._set(self.cookie, "", 0)])
