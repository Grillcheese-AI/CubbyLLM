"""cubbyllm/reasoning/ask_self.py -- a question about Cubby is answered from its record, never from a store (order 0079,
item 6: ``CB_ASK_SELF=1``, off by default; read by `AskLoop._ask` before anything else).

"who is cubby?" went to the ask loop on the board (2026-11-27) and offered three Wikidata items named Cubby, Peter Pan's
lost boy among them. With the flag on, a question whose subject is Cubby, by name or as "you" (`about`:
`cubby_record.is_about_cubby` or `identity.asks_about_cubby`), is answered by the talk seat from Cubby's record (the
facts block `cubby_record.block`, as a chat turn about Cubby is) and checked by `cubby_record.chat_check` against the
record; a reply that fails it gives way to a line made from the record itself. The record says `world: "record"`; no
source or store is asked, and `verified` stays False (nothing was looked up).

Wired: WIRED (`AskLoop._ask`).
"""
from __future__ import annotations

import os
import re
import time

__wiring__ = "WIRED"

# a question to Cubby about its own life or habits, its subject "you" ("where have you traveled?", "Do you use google
# often?", "Et toi, tu lis quoi ?", "Vous avez déjà fait du bénévolat, vous ?"; 0079's batch, 2026-11-28): the router
# sent these to the world as "a question about the world". A "do you know who X is" is a lookup (`ask_commands`).
_YOU = re.compile(r"\b(?:do|did|does|have|has|are|were|can|could|would|will|should|had)\s+you\b|"
                  r"^\W*(?:and\s+|so\s+|what\s+about\s+|how\s+about\s+)?(?:you|yourself)\b|"
                  r"^\W*(?:\w+[,!]\s*)?you(?:'re|’re|\s+are)\b|\b(?:where|what|when|how|why|which)\s+(?:have|do|did|are|were)\s+you\b|"
                  r",\s*(?:and\s+|what\s+about\s+)?(?:you|yourself)\s*\?|\b(?:et\s+)?toi\b|\b(?:tu|vous)\s+(?:\w+\s+){0,2}?[\w'’-]+|\best[- ]ce\s+que\s+(?:tu|vous)\b", re.I)
_TO_WORLD = re.compile(r"\b(?:do you know|can you tell me|could you tell me|tell me|tu sais|sais[- ]tu|savez[- ]vous|"
                       r"tu connais|connais[- ]tu|vous connaissez)\b", re.I)


LINE = {"en": "I'm {n}, {role}: {kind} built by {creator}.", "fr": "Je suis {n}, {role} : {kind} créé par {creator}."}


def on() -> bool:
    return os.environ.get("CB_ASK_SELF") == "1"


def about_you(text: str) -> bool:
    """A question whose subject is "you" (Cubby), not a "do you know who X is" (a lookup)."""
    t = (text or "").strip()
    return "?" in t and bool(_YOU.search(t)) and not _TO_WORLD.search(t)


def about(question: str) -> bool:
    from ..speech.cubby_record import is_about_cubby
    from ..speech.identity import asks_about_cubby
    return bool(is_about_cubby(question) or asks_about_cubby(question) or about_you(question))


def record_line(lang: str) -> str:
    """Who Cubby is, in the record's own words."""
    from ..speech import cubby_record as cr
    r = cr.load()
    return LINE[lang].format(n=r["name"], role=r["role"][lang], kind=r["kind"][lang], creator=r["creator"])


def answer(loop, question: str, t0: float) -> dict:
    """The loop's record for a question about Cubby: the talk seat's reply from the record, or the record's line."""
    from ..emit.base import clean_reply
    from ..speech import cubby_record as cr
    from ..speech.identity import guess_lang
    lang = "fr" if guess_lang(question) == "fr" else "en"
    facts = cr.block(lang)
    lines = [f"{e} — {r}: {v}" for e, r, v in facts]
    reply, why = None, None
    try:
        reply = clean_reply(loop.emitter.emit(question, max_new_tokens=160,
                                              context={"role": "talk", "facts": facts, "question": question}))
        ok, why = cr.chat_check(reply, question, lines) if reply else (False, "empty")
        reply = reply if ok else None
    except Exception as e:                               # noqa: BLE001 -- the seat failed: the record's own line
        why = f"emit_error: {str(e)[:120]}"
    spoken = reply or record_line(lang)
    return {"question": question, "answer": spoken, "reply": spoken, "verified": False, "kind": "self",
            "reason": "about Cubby: answered from its record", "world": "record", "record": lines,
            "seat_refused": why if reply is None else None, "plan": None, "seed": None, "learned": [],
            "entities": [], "aliased": [], "snapped": None, "trace": [], "wall_s": round(time.perf_counter() - t0, 3)}
