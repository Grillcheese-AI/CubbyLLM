"""cubby_serve/auth_gate.py -- the dashboards behind cubby-auth: which paths need a sign-in, which permission each
needs, and which audience (0016) a signed-in request gets. Off unless the server is given `--auth-issuer` (and
`CUBBY_AUTH_SECRET`); off, nothing here runs and every path answers as before (the audit token decides).

On, every dashboard path needs a cubby-auth sign-in (a person's session, or a program's `cbk_` API key):

    /board...  /panel  /brain  /map  /loop/...  /events...  /audit/...     board:read
    POST /board/...                                                        board:write (approve, pin)

and the audience comes from the account, not from the shared audit token: `audit` when it holds `board:audit`,
unless the page asks as the public (`X-Cubby-Audience: user`, the board's Public switch); `user` otherwise. The
token still works on the other paths (/turn, /ask, /state ...), which are the front end's, not the dashboards'.

cubby-auth's policy gets a client for this server, for example (`cubby-auth check-policy`, then `client-secret
--client board`, the secret into `CUBBY_AUTH_SECRET`):

    "board": {"name": "Cubby board", "redirect_uris": ["http://127.0.0.1:8765/auth/callback"],
              "allowed_types": ["owner", "admin", "service"],
              "roles": {"viewer": ["board:read"], "auditor": ["board:read", "board:audit"],
                        "operator": ["board:read", "board:audit", "board:write"]},
              "default_roles": {"owner": ["operator"], "admin": ["auditor"]}}

Wired: WIRED (`cubby_serve/api.py`'s handler; `cubby_serve/cli.py` builds it from `--auth-issuer`).
"""
from __future__ import annotations

import os

from cubby_serve.auth import CubbyAuth
from cubby_serve.auth_parts import CubbyAuthError, Reply

__wiring__ = "WIRED"

GUARDED = ("/board", "/panel", "/brain", "/map", "/loop", "/events", "/audit")   # as the routes match: by prefix
RULES = [("POST", "/board/", ("board:write",))]          # the first match wins; no match: board:read
READ, AUDIT = "board:read", "board:audit"


def guarded(path: str) -> bool:
    return path.startswith(GUARDED)                 # wider than the routes rather than narrower: closed, never open


def required(method: str, path: str) -> tuple:
    p = path.split("?", 1)[0]
    return next((perms for m, prefix, perms in RULES if m in ("*", method) and p.startswith(prefix)), (READ,))


def send(h, r: Reply) -> None:
    h.send_response(r.status)
    h.send_header("Content-Type", r.ctype)
    h.send_header("Cache-Control", "no-store")
    for k, v in r.headers:
        h.send_header(k, v)
    h.send_header("Content-Length", str(len(r.body)))
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(r.body)


def answer(h, auth: CubbyAuth) -> bool:
    """True when the request was answered here (a sign-in route, a refusal, /auth/me); else `h.cubby` is set (a
    guarded path, signed in) or left None (a path that needs no sign-in) and the server goes on."""
    h.cubby = None
    path, method = h.path.split("?", 1)[0], h.command
    routes = {("GET", "/auth/login"): auth.login, ("GET", auth.callback_path): auth.callback,
              ("POST", "/auth/lock"): auth.lock, ("POST", "/auth/logout"): auth.logout}
    try:
        fn = routes.get((method, path))
        if fn is not None:
            send(h, fn(h))
            return True
        if path != "/auth/me" and not guarded(path):
            return False
        got = auth.check(h, required(method, path))
    except CubbyAuthError as e:                      # cubby-auth down or misconfigured: closed, never open
        print(f"  auth: {e}", flush=True)
        send(h, Reply.json(503, {"error": "the sign-in service cannot be reached"}))
        return True
    if isinstance(got, Reply):
        send(h, got)
        return True
    h.cubby = got
    if path == "/auth/me":
        send(h, Reply.json(200, {**got.public(), "audit": got.can(AUDIT)}))
        return True
    return False


def audience(h) -> str | None:
    """The audience of a signed-in request ('audit' or 'user'); None when the request is not a signed-in one."""
    ident = getattr(h, "cubby", None)
    if ident is None:
        return None
    asks_public = (h.headers.get("X-Cubby-Audience") or "").lower() in ("user", "public") or "audience=user" in h.path
    return "audit" if ident.can(AUDIT) and not asks_public else "user"


def from_args(args) -> CubbyAuth | None:
    """The sign-in the arguments ask for, or None (off). The secret comes from the environment only."""
    if not getattr(args, "auth_issuer", None):
        return None
    secret = os.environ.get("CUBBY_AUTH_SECRET")
    if not secret:
        raise SystemExit("--auth-issuer needs CUBBY_AUTH_SECRET (from `cubby-auth client-secret --client "
                         f"{args.auth_client}`) in the environment")
    redirect = args.auth_redirect or f"http://{args.host}:{args.port}/auth/callback"
    return CubbyAuth(issuer=args.auth_issuer, client_id=args.auth_client, client_secret=secret, redirect_uri=redirect,
                     backchannel=args.auth_backchannel)
