"""cubby_serve/api_sessions.py -- what an API session keeps of its own on the one chat (order 0067, step 1, behind
``CB_API_ROLES=1``): its conversation, and the mood it shares with its company or user.

- **The conversation** (`CONVERSATION`): the chat's history (the last exchange in the seat's context, the replies a
  candidate is ranked against), the words its user has used (the appraisal's novelty), the last choice and rejection.
  Each session has its own: no turn sees another session's words (Cowork, 2026-11-13: "a leak between people").
- **The mood** (`MOOD`): the neurochemistry (the hormone ODE and its emotion), the user's state as read (`Afferent`),
  the hormones the turn reads, the last appraisal. Nick, 2026-11-13: "per company / user": a session of a company's
  member shares its company's mood; a signed-in caller without a company has their own; an anonymous session has
  its own. One company's or user's conversations never move another's mood. A turn without a session reads the
  chat's own mood and conversation, as before.

`swapped(chat, session, moods)` puts a session's conversation and mood into the chat for its turn and takes them
out after (the caller holds the turn lock).

Wired: WIRED (`cubby_serve/rule_api.py`'s turns).
"""
from __future__ import annotations

from contextlib import contextmanager

from cubbyllm.speech.rule_book import RuleBook

__wiring__ = "WIRED"

CONVERSATION = ("history", "_seen_vocab", "last_choice", "last_rejection")
MOOD = ("chem", "afferent", "state", "signals")


def fresh() -> dict:
    return {"history": [], "_seen_vocab": set(), "last_choice": {}, "last_rejection": None}


class Session:
    def __init__(self, owner, tenant) -> None:
        self.owner, self.tenant = owner, tenant
        self.conversation = RuleBook(prefix="conversation-rule")
        self.chat = fresh()                                     # this session's own conversation (CONVERSATION)


def mood_key(sid: str, s: Session) -> tuple:
    """Whose mood a session's turns move: its company's, else its user's, else its own."""
    if s.tenant:
        return ("company", s.tenant)
    return ("user", s.owner) if s.owner else ("session", sid)


def new_mood(chat) -> dict:
    """A mood as a new chat starts with it: the chat's own kinds, fresh."""
    return {"chem": type(chat.chem)(), "afferent": type(chat.afferent)(), "state": None, "signals": None}


@contextmanager
def swapped(chat, sid: str | None, s: Session | None, moods: dict):
    if chat is None or s is None:
        yield
        return
    keys = CONVERSATION + (MOOD if hasattr(chat, "chem") else ())
    saved = {k: getattr(chat, k, None) for k in keys}
    mk = mood_key(sid, s)
    mood = moods.get(mk) or (new_mood(chat) if hasattr(chat, "chem") else {})
    for k, v in list(s.chat.items()) + list(mood.items()):
        setattr(chat, k, v)
    if hasattr(chat, "chem") and chat.state is None:
        chat.state = chat._hormones()                          # the hormones of a fresh ODE
    try:
        yield
    finally:
        s.chat = {k: getattr(chat, k, None) for k in CONVERSATION}
        if hasattr(chat, "chem"):
            moods[mk] = {k: getattr(chat, k, None) for k in MOOD}
        for k, v in saved.items():
            setattr(chat, k, v)


__all__ = ["CONVERSATION", "MOOD", "Session", "fresh", "mood_key", "swapped"]
