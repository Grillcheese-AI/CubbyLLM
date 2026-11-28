"""cubbyllm/reasoning/ask_commands.py -- a request written as a command goes to the lookup (order 0079, item 7:
``CB_ASK_COMMANDS=1``, off by default; read by `brain_route.BrainRoute.needs_facts` and by `AskLoop.ask`).

"lookup marie curie then tell me the discoveries" went to chat on the board (2026-11-27) and got "I'm in, sweetie."
With the flag on, a turn that asks Cubby to look up, find, search or tell it about a subject needs facts (the router),
and the ask loop is asked about that subject (`restate`): "What are the discoveries of marie curie?", "What is the
capital of peru?". Marie Curie's discoveries are stored the other way round (radium's discoverer is her), so the loop
refuses that one today, which is the honest answer until 0081's reverse lookup (Nick's call). A subject that is the
user or Cubby ("tell me about your day", "tell me about yourself") or a pronoun ("tell me about it") is not a command.

Wired: WIRED (`brain_route.BrainRoute.needs_facts`, `AskLoop.ask`).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

_CMD = re.compile(r"^\W*(?:(?:can|could|would|will) you\s+|please\s+|pls\s+)?(?:look\s*up|find(?: out about)?|"
                  r"search(?: for| up)?|google|tell me about|recherche|cherche|trouve|parle[- ]moi d[eu']\s*)\s*"
                  r"(?P<rest>.+)$", re.I)
_NOT = re.compile(r"^(?:you|your|yours|yourself|me|my|myself|us|our|it|this|that|them|him|her|toi|moi|ton|ta|tes|"
                  r"nous|ça|cela)\b", re.I)
_THEN = re.compile(r"\s*,?\s*\b(?:and then|then|and|puis|et puis|ensuite|et)\s+(?=(?:tell|give|show|list|what|dis|donne)\b)",
                   re.I)
_ASKED = re.compile(r"^(?:tell me|give me|show me|list|what (?:are|is|were|was)|dis[- ]moi|donne[- ]moi)\s+(?:about\s+)?"
                    r"(?:the |her |his |its |their |les |ses |leurs? |la |le )?(?P<what>.+?)\W*$", re.I)


def on() -> bool:
    return os.environ.get("CB_ASK_COMMANDS") == "1"


def _parts(text: str) -> tuple[str, str | None] | None:
    m = _CMD.match((text or "").strip())
    if not m:
        return None
    rest = m.group("rest").strip().rstrip("?.!… ")
    head, *tail = _THEN.split(rest, maxsplit=1)
    subject = head.strip(" ,")
    if not subject or _NOT.match(subject):
        return None
    asked = _ASKED.match(tail[0].strip()) if tail else None
    return subject, (asked.group("what").strip() if asked else None)


def is_command(text: str) -> bool:
    return _parts(text) is not None


def restate(text: str) -> str:
    """The question the loop is asked for a command, or the text as it is."""
    got = _parts(text)
    if got is None:
        return text
    subject, what = got
    fr = bool(re.match(r"^\W*(?:recherche|cherche|trouve|parle)", text.strip(), re.I))
    if what:
        return f"Quels sont les {what} de {subject} ?" if fr else f"What are the {what} of {subject}?"
    return f"C'est quoi {subject} ?" if fr else f"What is {subject}?"
