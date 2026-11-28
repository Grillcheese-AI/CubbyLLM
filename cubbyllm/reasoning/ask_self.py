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
import time

__wiring__ = "WIRED"

LINE = {"en": "I'm {n}, {role}: {kind} built by {creator}.", "fr": "Je suis {n}, {role} : {kind} créé par {creator}."}


def on() -> bool:
    return os.environ.get("CB_ASK_SELF") == "1"


def about(question: str) -> bool:
    from ..speech.cubby_record import is_about_cubby
    from ..speech.identity import asks_about_cubby
    return bool(is_about_cubby(question) or asks_about_cubby(question))


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
