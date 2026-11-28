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
# a question asked inside "do you know ...": "hey do you know who Howard K Smith is? my teacher mentioned him" went
# "about Cubby himself" and never reached the loop (0079's batch, 2026-11-28); "what do you know about X" too
_EMBED = re.compile(r"^\W*(?:hey|so|ok|okay|and|say|dis|alors)?[\s,]*(?:do you know|can you tell me|could you tell me|"
                    r"do you happen to know|tu sais|sais[- ]tu|savez[- ]vous|est[- ]ce que tu sais)\s+(?P<q>(?:who|what|when|"
                    r"where|which|how many|how much|how old|qui|quand|o[ùu]|quel\w*|combien)\b(?!,)[^?]*)", re.I)
_ABOUT = re.compile(r"^\W*(?:hey|so)?[\s,]*what do you know about\s+(?P<s>[^?]+)", re.I)
_TAIL_VERB = re.compile(r"^(?P<wh>\w+(?:\s+(?:many|much|old)\s+\w+)?)\s+(?P<rest>.+?)\s+(?P<v>is|was|are|were|est|était)"
                        r"(?:\s+(?P<tail>\w+(?:\s+\w+)?))?$", re.I)   # "when X was born" -> "When was X born?"


def _embedded(text: str) -> str | None:
    """The question inside "do you know who X is", put back in question order ("Who is X?"), unless it is about
    Cubby ("do you know who made you?")."""
    m = _ABOUT.match(text or "")
    if m and not _NOT.match(m.group("s").strip()):
        return f"What is {m.group('s').strip().rstrip('.!')}?"
    m = _EMBED.match(text or "")
    if not m:
        return None
    q = " ".join(m.group("q").split()).rstrip(" .!")
    if re.search(r"\b(?:you|your|yourself|toi|tu|vous|te)\b", q, re.I):
        return None
    v = _TAIL_VERB.match(q)
    q = f"{v.group('wh')} {v.group('v')} {v.group('rest')}" + (f" {v.group('tail')}" if v.group("tail") else "") if v else q
    return q[:1].upper() + q[1:] + (" ?" if re.match(r"^(?:qui|quand|o[ùu]|quel|combien)", q, re.I) else "?")


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
    return _parts(text) is not None or _embedded(text) is not None


def restate(text: str) -> str:
    """The question the loop is asked for a command, or the text as it is."""
    got = _parts(text)
    if got is None:
        return _embedded(text) or text
    subject, what = got
    fr = bool(re.match(r"^\W*(?:recherche|cherche|trouve|parle)", text.strip(), re.I))
    if what:
        return f"Quels sont les {what} de {subject} ?" if fr else f"What are the {what} of {subject}?"
    return f"C'est quoi {subject} ?" if fr else f"What is {subject}?"
