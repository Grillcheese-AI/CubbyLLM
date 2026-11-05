"""cubbyllm/reasoning/uncertified.py -- the value a walk reached, certified or not, kept on the ask's record for a
measurement (order 0054: the stack with the VM, against the same proposals spoken without it). Off unless
``CB_RECORD_UNCERTIFIED=1``; off, the record is unchanged. Nothing reads the field to answer: it is never said.

  verified                the answer
  answer_type_mismatch    the certified chain's value (``refused["value"]``)
  ambiguous_hop           the first object the ambiguous hop held (``refused["objects"][0]``)
  vm_verify_failed        the last chain's object, when that chain reached every hop of the plan
  a check after the walk  (tense, contested, the time rule) the answer it withheld: the walk verified, so this is
                          the walk's answer
  otherwise               None (no plan, nothing retrieved, a chain that stopped short): nothing to speak

Wired: WIRED (``cubbyllm/reasoning/ask.py``, behind the flag).
"""
from __future__ import annotations

import os

__wiring__ = "WIRED"


def on() -> bool:
    """``CB_RECORD_UNCERTIFIED=1``: the ask's record keeps ``uncertified`` (off by default)."""
    return os.environ.get("CB_RECORD_UNCERTIFIED") == "1"


def value(result, n_hop: int) -> str | None:
    """The value ``result`` (a ``CoTResult``) reached, whether or not it was certified; None if it reached none."""
    if result.verified:
        return result.answer
    r = result.refused if isinstance(result.refused, dict) else {}
    if r.get("value"):
        return str(r["value"])
    if r.get("answer"):
        return str(r["answer"])
    objs = r.get("objects") or []
    if objs:
        return str(objs[0])
    if result.reason == "vm_verify_failed":
        hops = [h for h in (result.trace or []) if getattr(h, "triple", None) is not None]
        if hops and len(hops) >= n_hop:
            return str(hops[-1].triple.obj)
    return None


__all__ = ["on", "value"]
