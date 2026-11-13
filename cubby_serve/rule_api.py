"""cubby_serve/rule_api.py -- roles in the API (order 0067, step 1). Off unless ``CB_API_ROLES=1``; off, `POST /turn`
reads ``{"text"}`` as before and nothing here runs.

On, `POST /turn` also takes messages with a role, in the shape other models' clients send, and a session:

    {"messages": [{"role": "system", "level": "user", "content": "Keep replies under 50 words."},
                  {"role": "user", "content": "..."}], "session": "<id from an earlier reply>"?, "lang": "en"?}

- **system** carries its level: ``global`` (the operator: the permission ``rules:global``), ``company`` (an admin of
  the caller's company: cubby-auth's ``tenant_role`` "admin"), ``user`` (any signed-in caller, for their own session;
  the default). The caller's role is checked before a prompt is set; without cubby-auth (``--auth-issuer``) nobody can
  set one here (closed, never open). A prompt is compiled once (the same text from the same setter is not read
  again); a new one at the same place withdraws the old one's rules from that turn. Its rule book, in plain words, is
  in the reply, to whoever set it (`RuleBook.show_book`); a company may ask ``"confirm_before_apply": true``.
- **user**: the last one is the turn's text. **assistant** and **tool** are accepted and set nothing (a tool's result
  is text that is read, below the user); the chat keeps its own history.
- **session**: made by the server on the first call and returned (``"session"``); a session is its creator's (another
  caller is refused); an id the server did not make is unknown. Without one, a turn reads the global rules only.
  A session's conversation is its own: its history (the last exchange in the context, the replies ranked against),
  the words its user has used (the appraisal's novelty) and its last choice are swapped into the one chat for its
  turn and out after, so no turn sees another session's words. The hormones stay the one Cubby's. Rules count the
  server's turns (one clock for every book), not a session's.

    POST /rules/correct  {"level", "rule", "text", "session"?}  -> the corrected reading said back (its setter only)
    POST /rules/confirm  {"rules": [...]}                       -> a company's held rules confirmed (its admin)

A turn's rule rows show a user the rules above their own level by how each acted and its outcome only (not the
operator's or the company's sentence, nor who set it); the audit view shows all.

The reader: ``brain.rule_reader(sentence) -> {"how", "plan"}`` (the plan reader on the programs seat). Nothing in the
server's boot sets it yet (its adapter waits for the view reader's read): with none, every sentence is kept as "asked"
and said so in the rule book; nothing is checked that nobody read.

Wired: WIRED (`cubby_serve/post.py`'s /turn and /rules/ routes, behind ``CB_API_ROLES=1``).
"""
from __future__ import annotations

import os
import secrets
import threading
from collections import OrderedDict

from cubbyllm.speech.rule_book import RuleBook
from cubbyllm.speech.rule_stack import RuleStack, compile_prompt

__wiring__ = "WIRED"

ROLES = ("system", "user", "assistant", "tool")
SET_LEVELS = ("global", "company", "user")
MAX_SESSIONS = 10_000


def on() -> bool:
    return os.environ.get("CB_API_ROLES", "0") == "1"


class Refused(Exception):
    def __init__(self, code: int, why: str) -> None:
        super().__init__(why)
        self.code, self.why = code, why


CONVERSATION = ("history", "_seen_vocab", "last_choice", "last_rejection")    # the chat's per-conversation state


def fresh() -> dict:
    return {"history": [], "_seen_vocab": set(), "last_choice": {}, "last_rejection": None}


class Session:
    def __init__(self, owner, tenant) -> None:
        self.owner, self.tenant = owner, tenant
        self.conversation = RuleBook(prefix="conversation-rule")
        self.chat = fresh()                                     # this session's own conversation (CONVERSATION)


class RuleStore:
    def __init__(self) -> None:
        self.lock, self.turn_lock = threading.Lock(), threading.Lock()
        self.books: dict = {}                      # (level, place) -> RuleBook, kept across prompts (rules never deleted)
        self.prompts: dict = {}                    # (level, place) -> the last compile {"text", "set_by", "result"}
        self.sessions: OrderedDict = OrderedDict()
        self.clock = 0                             # the server's turns under the flag: every book's turn

    def book(self, key: tuple) -> RuleBook:
        if key not in self.books:
            self.books[key] = RuleBook(prefix=f"{key[0]}-rule{'' if key[0] == 'global' else '-' + str(key[1])}-")
        return self.books[key]

    def session(self, sid: str | None, ident) -> tuple[str, Session]:
        with self.lock:
            if sid:
                s = self.sessions.get(sid)
                if s is None:
                    raise Refused(404, "unknown session")
                if s.owner is not None and (ident is None or ident.sub != s.owner):
                    raise Refused(403, "this session is someone else's")
                return sid, s
            sid = secrets.token_urlsafe(18)
            self.sessions[sid] = Session(ident.sub if ident else None, ident.tenant if ident else None)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
            return sid, self.sessions[sid]

    def stack(self, sid: str | None, s: Session | None) -> RuleStack:
        books = [self.books.get(("global", None))]
        if s is not None:
            books += [self.books.get(("company", s.tenant)) if s.tenant else None, self.books.get(("user", sid))]
        return RuleStack(books, s.conversation if s is not None else RuleBook(prefix="conversation-rule"), self.clock)


def store_of(brain) -> RuleStore:
    st = getattr(brain, "_rule_store", None)
    if st is None:
        st = brain._rule_store = RuleStore()
    return st


def clock(brain) -> int:
    return store_of(brain).clock


def caller(h, brain):
    """The signed-in identity of this request, or None (no cubby-auth, or not signed in)."""
    got = getattr(h, "cubby", None)
    if got is not None:
        return got
    auth = getattr(brain, "auth", None)
    if auth is None:
        return None
    from cubby_serve.auth_parts import CubbyAuthError, Identity
    try:
        got = auth.check(h, ())
    except CubbyAuthError:
        raise Refused(503, "the sign-in service cannot be reached")
    return got if isinstance(got, Identity) else None


def place(level: str, ident, sid: str) -> tuple:
    """Where a prompt of `level` set by `ident` goes, or Refused: the role is checked before anything is set."""
    if ident is None:
        raise Refused(401, "setting a system prompt needs a sign-in")
    if level == "global":
        if "rules:global" not in (ident.permissions or []):
            raise Refused(403, "only the operator sets the global prompt")
        return ("global", None)
    if level == "company":
        if not ident.tenant or ident.tenant_role != "admin":
            raise Refused(403, "only a company's admin sets its prompt")
        return ("company", ident.tenant)
    if level == "user":
        return ("user", sid)
    raise Refused(400, f"unknown level {level!r}: {SET_LEVELS}")


def set_prompt(brain, store: RuleStore, key: tuple, text: str, ident, lang: str, confirm: bool) -> dict:
    with store.lock:
        last = store.prompts.get(key)
        if last is not None and last["text"] == text and last["set_by"] == ident.sub:
            return dict(last["result"], level=key[0], compiled_now=False)         # compiled once
        book, at = store.book(key), clock(brain)
        for r in book.rules:                                                    # the new prompt replaces the old
            if r.withdrawn_at is None:
                book.withdraw(r.id, at)
        if key[0] == "company" and confirm:
            book.confirm_levels.add("company")
        res = compile_prompt(text, key[0], ident.sub, at, getattr(brain, "rule_reader", None), book, lang)
        store.prompts[key] = {"text": text, "set_by": ident.sub, "result": res}
        return dict(res, level=key[0], compiled_now=True)


def parse(req: dict) -> tuple[list[dict], str]:
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not all(isinstance(m, dict) for m in msgs):
        raise Refused(400, "messages: a list of {role, content}")
    for m in msgs:
        if m.get("role") not in ROLES or not isinstance(m.get("content", ""), str):
            raise Refused(400, f"a message's role is one of {ROLES} and its content a string")
    users = [m["content"].strip() for m in msgs if m["role"] == "user"]
    return [m for m in msgs if m["role"] == "system"], (users[-1] if users else "")


def shown_rows(rows: list, audience: str) -> list:
    """A user sees the rules above their own level by how each acted and its outcome only."""
    if audience == "audit":
        return rows
    return [r if r.get("level") in ("user", "conversation") else
            {k: r[k] for k in ("rule", "level", "how", "steered", "outcome") if k in r} for r in rows]


def turn(h, brain, req: dict, run) -> tuple[int, dict]:
    """`POST /turn` with the flag on: (code, payload). ``run(text) -> rec`` is the server's turn, unchanged."""
    store, lang = store_of(brain), (req.get("lang") if req.get("lang") in ("en", "fr") else "en")
    try:
        if "messages" not in req:                                               # the old shape: the global rules only
            text = str(req.get("text", "")).strip()
            if not text:
                return 400, {"error": "empty text"}
            return 200, _run(brain, store, None, text, run, h._who())
        systems, text = parse(req)
        ident = caller(h, brain)                       # a signed-in caller's session is theirs from its first turn
        sid, s = store.session(req.get("session"), ident)
        compiled = []
        for m in systems:
            key = place(str(m.get("level") or "user"), ident, sid)
            compiled.append(set_prompt(brain, store, key, m["content"], ident, lang, bool(m.get("confirm_before_apply"))))
        if not text:
            return 200, {"session": sid, "compiled": compiled}
        rec = _run(brain, store, (sid, s), text, run, h._who())
        return 200, dict(rec, session=sid, compiled=compiled)
    except Refused as e:
        return e.code, {"error": e.why}


def _run(brain, store: RuleStore, session, text: str, run, audience: str) -> dict:
    """The turn on the one chat, with this session's rules and conversation swapped in (none: the chat's own)."""
    chat, (sid, s) = getattr(brain, "chat", None), session or (None, None)
    with store.turn_lock:
        stack = store.stack(sid, s)
        saved = {k: getattr(chat, k, None) for k in ("rules",) + CONVERSATION} if chat is not None else {}
        if chat is not None:
            chat.rules = stack
            for k, v in (s.chat.items() if s is not None else ()):
                setattr(chat, k, v)
        try:
            rec = run(text)
        finally:
            if chat is not None:
                if s is not None:
                    s.chat = {k: getattr(chat, k, None) for k in CONVERSATION}
                for k, v in saved.items():
                    setattr(chat, k, v)
            store.clock += 1
    if rec.get("rules"):
        rec["rules"] = shown_rows(rec["rules"], audience)
    return rec


def rules_post(h, brain, path: str, req: dict) -> tuple[int, dict]:
    """`POST /rules/correct` and `/rules/confirm`."""
    store, lang = store_of(brain), (req.get("lang") if req.get("lang") in ("en", "fr") else "en")
    try:
        ident = caller(h, brain)
        if path == "/rules/confirm":
            key = place("company", ident, "")
            with store.lock:
                book = store.book(key)
                ids = [i for i in req.get("rules") or [] if any(r.id == i for r in book.rules)]
                book.confirm(ids, clock(brain))
            return 200, {"confirmed": ids}
        if path != "/rules/correct":
            return 404, {"error": "unknown path"}
        sid = req.get("session") or ""
        if req.get("level") == "user":
            sid, _ = store.session(sid or None, ident) if sid else ("", None)
            if not sid:
                raise Refused(400, "a user-level correction names its session")
        key = place(str(req.get("level") or ""), ident, sid)
        read = getattr(brain, "rule_reader", None)
        got = read(str(req.get("text") or "")) if read is not None else {"how": "asked"}
        if got.get("how") != "written" or not got.get("plan"):
            return 422, {"error": "that reads as no rule I can check", "how": got.get("how")}
        with store.lock:                          # between turns: the clock is the next turn, which the new rule reads
            book = store.book(key)
            r = next((x for x in book.rules if x.id == req.get("rule")), None)
            if r is None or r.set_by != ident.sub:
                raise Refused(404 if r is None else 403, "no such rule" if r is None else "only its setter corrects a rule")
            try:
                new, said = book.correct(r.id, got["plan"], ident.sub, clock(brain) - 1, str(req.get("text")), lang)
            except ValueError as e:
                return 422, {"error": str(e)[:200]}
        return 200, {"rule": new.id, "replaces": r.id, "said": said, "from_turn": new.set_at}
    except Refused as e:
        return e.code, {"error": e.why}


__all__ = ["RuleStore", "on", "rules_post", "turn"]
