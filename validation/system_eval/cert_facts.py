"""Order 0076, step 2: what a certificate needs that arm C's record did not keep, recorded as the run goes
(``CB_CERT_FACTS=1``; off, run_c's records are as before). A record only: no answer reads it.

For each question:
- ``facts``: each fact of the trace, its triple as the worlds hold it, the world that handed it, the items it was
  read from and, from a statement store (`wd_store_statements`), every statement that said it (item, property,
  statement id, rank, the item's revision and modified time, its start / end / point-in-time qualifiers);
- ``programs``: the SHA-256 of each program source the VM ran for the question, in order;
- ``checks``: the host's checks that ran on the answer and their outcome (`ask_memory.ran`);
- ``at``: the time the record was made (UTC), the certificate's generated time;
- ``as_of``: the year the question asks about (`ask_words.strip_as_of`), or None when it names none: then the
  question was evaluated at the store's own date (the dump's, in the run's meta).

Wired: STANDALONE (`run_c.one` calls it with the flag on).
"""
from __future__ import annotations

import hashlib
import os
import time

__wiring__ = "STANDALONE"


def on() -> bool:
    return os.environ.get("CB_CERT_FACTS") == "1"


def note_program(timing: dict, source) -> None:
    """Called by `run_c.timed` for each VM run: the program's hash, kept for the question running."""
    if on() and isinstance(source, str):
        timing.setdefault("programs", []).append(hashlib.sha256(source.encode("utf-8")).hexdigest())


def _worlds(loop) -> list:
    w = getattr(loop, "world", None)
    return [x for x in [w, *getattr(w, "others", [])] if x is not None]


def facts(loop, rec: dict) -> list:
    from cubbyllm.reasoning.planner import parse_fact
    worlds, rels = _worlds(loop), {}
    for w in worlds:
        try:
            rels.update(w.relations() if hasattr(w, "relations") else {})
        except Exception:                                # noqa: BLE001 -- a record never breaks the run
            pass
    out = []
    for f in rec.get("trace") or []:
        t = parse_fact(f, known=rels or None)
        row = {"fact": f, "triple": [t.subj, t.rel, t.obj] if t is not None else None, "worlds": []}
        for w in worlds:
            items = w.fact_items(f) if hasattr(w, "fact_items") else []
            sts = w.statements_of(t.subj, t.rel, t.obj) if t is not None and hasattr(w, "statements_of") else []
            if items or sts:
                row["worlds"].append({"world": getattr(w, "name", type(w).__name__), "items": items,
                                      "statements": [s for s in sts if not items or s["item"] in items]})
        out.append(row)
    return out


def record(loop, rec: dict, question: str, programs: list) -> dict:
    from cubbyllm.reasoning.ask_words import strip_as_of
    try:
        return {"facts": facts(loop, rec), "programs": list(programs), "as_of": strip_as_of(question)[1],
                "checks": rec.get("checks_ran") or [], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except Exception as e:                               # noqa: BLE001 -- a record never breaks the run
        return {"error": str(e)[:160], "programs": list(programs)}


__all__ = ["facts", "note_program", "on", "record"]
