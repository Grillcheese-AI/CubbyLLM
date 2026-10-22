"""standin/audience.py -- who a reply is for (order 0016, Nick: "the WHERE it learned it should be for internal
purposes only"; "if we have an audit it's available but not to regular users").

Two audiences. `user`, the default, gets answers and never their sources: no source, author, title, site,
domain, URL or dataset named in the reply, no run of 8 or more words from any stored passage, no provenance
fields in the JSON. `audit`, unlocked by the token in `CUBBY_AUDIT_TOKEN` (or the git-ignored file
`standin/data/out/audit_token`; neither set = audit locked), gets everything. Nick's rule, so `user` is not
behind a flag: it is what a user sees.

- `audience(token)`: the audience a request's token earns (compared in constant time);
- `for_user(rec)`: a record as a user may see it -- the provenance fields dropped, the per-line sources of a
  profile dropped, a self-profile's sources counted, not named;
- `LeakGuard`: the last check at the mouth. The names behind the answer (its facts' provenance, the books'
  authors and titles) and the store's source names, plus URLs and domains; a hit is cut out where it sits in a
  clause that only names the source ("(source: X)", "according to X,"), otherwise the reply is refused; and
  every 8-word run is looked up in the passage index (`books.BookIndex`): a quote is refused.

Wired: WIRED (`serve_api` reads the audience of every request; `ask.AskLoop.ask` passes the reply through the
guard).
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import pathlib
import re
import urllib.parse

__wiring__ = "WIRED"

TOKEN_FILE = pathlib.Path(__file__).resolve().parents[2] / "standin" / "data" / "out" / "audit_token"
# the store's own source names: every world and fetch source a fact can come from, and the datasets behind them
SOURCE_NAMES = {"wikidata", "wikipedia", "dbpedia", "wikikg", "wikidata5m", "hypernet-scaling-law-data", "brave",
                "searxng", "encyclopedia americana", "openrouter", "project gutenberg", "historical_facts"}
PROVENANCE_FIELDS = ("learned", "evidence", "provenance", "sources", "passages", "citations", "news_changes")
# internal working the user's JSON never carries: the thought graph (it records each hop's provenance), the store's
# snapshot, the reasoning task (its facts' walk sources), the drafts not spoken, the route (a world's name)
INTERNAL_FIELDS = ("graph", "store_snapshot", "task", "rejected", "offered", "route", "plugin", "guard", "hash",
                   "plugins", "plugin_facts", "plugins_declined")
AUDIT_PATHS = ("/panel", "/brain", "/map", "/loop/events", "/loop/stream", "/events", "/audit/")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|fr|uk|de|info)\b", re.I)
_CLAUSE = r"(?:\s*\((?:source|sources|via|from)\s*:?\s*{n}\)|,?\s*according to {n},?|\s*\(?\bsource\s*:\s*{n}\)?)"


def configured_token() -> str | None:
    tok = os.environ.get("CUBBY_AUDIT_TOKEN")
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return tok or None


def audience(token: str | None) -> str:
    """'audit' for the configured token, 'user' for anything else (no token configured: always 'user')."""
    want = configured_token()
    return "audit" if want and token and hmac.compare_digest(str(token), want) else "user"


def token_of(headers, path: str = "") -> str | None:
    """A request's audit token: `X-Audit-Token`, `Authorization: Bearer <token>`, or `?token=` (an EventSource
    cannot send a header)."""
    tok = headers.get("X-Audit-Token") if headers is not None else None
    auth = (headers.get("Authorization") or "") if headers is not None else ""
    if not tok and auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
    if not tok and "?" in (path or ""):
        tok = (urllib.parse.parse_qs(path.split("?", 1)[1]).get("token") or [None])[0]
    return tok or None


def redact(line: str) -> str:
    """A request line with its token masked (the access log prints paths)."""
    return re.sub(r"(token=)[^&\s\"]+", r"\1***", line)


def audit_only(path: str) -> bool:
    route = path.split("?", 1)[0].rstrip("/") or "/"
    return any(route == p.rstrip("/") or route.startswith(p if p.endswith("/") else p + "/") for p in AUDIT_PATHS)


# injected into an audit page served with a valid token: its own requests carry the token it was opened with
AUDIT_SHIM = """<script>(function(){var t=new URLSearchParams(location.search).get('token');if(!t)return;
var f=window.fetch;window.fetch=function(u,o){o=o||{};o.headers=Object.assign({},o.headers||{},{'X-Audit-Token':t});
return f(u,o);};var E=window.EventSource;window.EventSource=function(u,c){u+=(String(u).indexOf('?')<0?'?':'&')+
'token='+encodeURIComponent(t);return new E(u,c);};})();</script>"""


def record_hash(rec: dict) -> str:
    """The record's own hash (SHA-256 over its canonical JSON, the hash itself left out): the audit chain's seal."""
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


def _strip_profile(prof):
    if isinstance(prof, list):
        return [_strip_profile(x) for x in prof]
    if isinstance(prof, dict):
        return {k: _strip_profile(v) for k, v in prof.items() if k not in PROVENANCE_FIELDS + ("source",)}
    return prof


def for_user(rec: dict) -> dict:
    """The record a user may see: the provenance fields and the internal working gone, the profile's per-line
    sources gone."""
    out = {k: copy.deepcopy(v) for k, v in rec.items() if k not in PROVENANCE_FIELDS + INTERNAL_FIELDS}
    if "profile" in out:
        out["profile"] = _strip_profile(out["profile"])
    return out


def source_names(rec: dict, books=None) -> set[str]:
    """The names a reply must not say: the store's sources, the provenance behind this record (sites, books by
    their authors), and the authors and titles of the passages its facts cite."""
    names = set(SOURCE_NAMES)
    for p in rec.get("learned") or []:
        for part in re.split(r"\+", str(p.get("source") or "")):
            part = re.sub(r"\s*\((?:latent|attributed)\)$", "", part).strip()
            part = re.sub(r"^(?:book|web|news|site):", "", part)
            names.update(x for x in re.split(r"&", part) if x)
    for ws in (rec.get("evidence") or {}).values():
        for w in ws:
            names.update(x for x in (w.get("author"), w.get("title")) if x)
    return {n.strip().lower() for n in names if n and len(n.strip()) >= 3}


class LeakGuard:
    """`check(text, rec) -> (text or None, hits)`: a reply with no source in it; None when one cannot be cut."""

    def __init__(self, index=None, n: int = 8) -> None:
        self.index, self.n = index, int(n)

    def quotes(self, text: str) -> list[str]:
        """Every run of `n` words in `text` that a stored passage holds verbatim (the passage index's FTS)."""
        if self.index is None:
            return []
        words = re.findall(r"[\w']+", text)
        out = []
        for i in range(len(words) - self.n + 1):
            q = '"' + " ".join(words[i:i + self.n]) + '"'
            if self.index.db.execute("SELECT 1 FROM fts WHERE fts MATCH ? LIMIT 1", (q,)).fetchone():
                out.append(" ".join(words[i:i + self.n]))
        return out

    def check(self, text: str | None, rec: dict) -> tuple[str | None, list[str]]:
        """The answer and the entity asked about are never a leak ("Who founded Wikipedia?")."""
        if not text:
            return text, []
        own = {str(rec.get("seed") or "").strip().lower()}
        if "profile" not in rec:                                    # a chain's answer is a value; a profile's is the reply
            own.add(str(rec.get("answer") or "").strip().lower())
        for line in rec.get("profile") or []:
            if isinstance(line, dict):
                v = line.get("value")
                own.update(str(x).strip().lower() for x in (v if isinstance(v, list) else [v]) if x)
        own.update(f.split(" is the ", 1)[0].strip().lower() for f in rec.get("trace") or [] if " is the " in str(f))
        own -= {""}
        hits = []
        for name in sorted(source_names(rec) - own, key=len, reverse=True):
            rx = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.I)
            if rx.search(text):
                hits.append(name)
                text = re.sub(_CLAUSE.format(n=re.escape(name)), "", text, flags=re.I)
        urls = [m.group(0) for m in _URL.finditer(text)]
        for u in urls:                                              # cut only where a clause just names it
            text = re.sub(_CLAUSE.format(n=re.escape(u)), "", text, flags=re.I)
        hits += urls
        left = [h for h in hits if re.search(rf"(?<![\w]){re.escape(h)}(?![\w])", text, re.I)]
        quoted = self.quotes(text)
        if left or quoted:
            return None, hits + [f"quote: {q}" for q in quoted]
        return (" ".join(text.split()) if hits else text), hits


SPOKEN_FIELDS = ("reply", "prose", "statement", "needs")


def mouth(rec: dict, guard: "LeakGuard") -> dict:
    """The last check before a record is spoken: every text a user reads, through the guard. A text it cannot
    cut is dropped (a profile's or an introspection's answer, which IS its reply: the record refuses,
    reason leak_guard); what it found stays on the record for the audit view (`guard`)."""
    found = {}
    fields = SPOKEN_FIELDS + (("answer",) if "profile" in rec else ())
    for f in fields:
        if rec.get(f):
            text = str(rec[f])
            if f == "answer" and " — " in text:               # 'entity — rel: values; rel: values': an item that names
                head, _, body = text.partition(" — ")          # a source goes whole, the others stay
                items = body.split("; ")
                kept = [i for i in items if not guard.check(i, rec)[1]]
                if len(kept) < len(items):
                    found[f] = [h for i in items for h in guard.check(i, rec)[1]]
                    rec[f] = f"{head} — {'; '.join(kept)}" if kept else None
                continue
            said, hits = guard.check(text, rec)
            if hits:
                found[f] = hits
                rec[f] = said
    if found:
        rec["guard"] = found
        if "answer" in found and rec.get("answer") is None:
            rec.update(verified=False, reason="leak_guard")
    return rec


def audit_chain(loop, record_id: str) -> dict | None:
    """Everything behind one kept record (the audit view, `GET /audit/<id>`): the plan, the facts with their
    sources, dates and passages, the gate's verdicts, the VM's, the tier, the thought graph, and the record's
    hash with whether it still matches."""
    rec = next((r for r in reversed(getattr(loop, "history", [])) if r.get("id") == record_id), None)
    if rec is None:
        return None
    prov = getattr(loop.world, "provenance", None) or {}
    times = getattr(loop.world, "times", None) or {}
    key = lambda f: " ".join(str(f).split())                                  # noqa: E731
    facts = [{"fact": f, "source": prov.get(key(f)), "time": times.get(key(f)),
              "passages": (rec.get("evidence") or {}).get(f, [])} for f in rec.get("trace") or []]
    return {"id": record_id, "question": rec.get("question"), "plan": rec.get("plan"), "seed": rec.get("seed"),
            "facts": facts, "gate": rec.get("learned"), "vm": {"verified": rec.get("verified"), "reason": rec.get("reason"),
            "refused": rec.get("refused")}, "tier": rec.get("tier") or ("flat" if rec.get("verified") else None),
            "guard": rec.get("guard"), "graph": rec.get("graph"), "hash": rec.get("hash"),
            "sealed": bool(rec.get("hash")) and rec.get("hash") == record_hash(rec), "record": rec}
