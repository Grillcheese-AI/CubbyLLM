"""cubby_serve/auth_parts.py -- the pieces `auth.py` (cubby-auth for this server's dashboards) is built from: the
identity a request carries, an answer the handler sends as it is, and the request helpers (cookies, the query, a
local path for `next`).

Wired: WIRED (`cubby_serve/auth.py`, `cubby_serve/auth_gate.py`).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs

__wiring__ = "WIRED"


class CubbyAuthError(RuntimeError):
    pass


@dataclass
class Identity:
    """Who is making this request, as cubby-auth said a moment ago."""
    sub: str
    name: str
    user_type: str
    roles: list
    permissions: list
    kind: str                         # "session" (a person in a browser) or "api_key"
    key_id: str | None = None
    tenant: str | None = None
    tenant_name: str | None = None
    tenant_role: str | None = None

    def can(self, *perms: str) -> bool:
        return all(p in self.permissions for p in perms)

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("sub", "name", "user_type", "roles", "permissions", "kind", "key_id",
                                              "tenant", "tenant_name", "tenant_role")}


@dataclass
class Reply:
    """An answer the handler sends as it is."""
    status: int
    body: bytes = b""
    ctype: str = "application/json; charset=utf-8"
    headers: list = field(default_factory=list)

    @classmethod
    def json(cls, status: int, obj: dict, headers=()) -> "Reply":
        return cls(status, json.dumps(obj).encode("utf-8"), headers=list(headers))

    @classmethod
    def redirect(cls, url: str, headers=()) -> "Reply":
        return cls(302, b"", "text/plain; charset=utf-8", [("Location", url), *headers])

    @classmethod
    def text(cls, status: int, text: str) -> "Reply":
        return cls(status, text.encode("utf-8"), "text/plain; charset=utf-8")


def local_path(p: str | None) -> str:
    """Only a path on this server: never another site (an open redirect) or a scheme-relative URL."""
    if not p or not p.startswith("/") or p.startswith("//") or "\\" in p or any(ord(c) < 0x20 for c in p):
        return "/"
    return p


def cookies(h) -> dict:
    out = {}
    for part in (h.headers.get("Cookie") or "").split(";"):
        k, sep, v = part.strip().partition("=")
        if sep and k and k not in out:
            out[k] = v
    return out


def _query(h) -> dict:
    return {k: v[0] for k, v in parse_qs(h.path.split("?", 1)[1] if "?" in h.path else "").items()}


def _b64url_json(part: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
