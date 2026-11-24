"""exp_0081_seat -- order 0081's 450M arm for the benches: the programs seat built the way `cubby_serve.build` builds it
(``SlotEmitter(Cubby450mEmitter(trunk))`` over the worlds' names, plus a question's own capitalised runs), and one
emission under one ``CB_EMIT_TRIE`` arm (A ``off``, B ``rel``, C ``rel+slot``), timed.

Run under grilly2's own Python (the stack's), from this repo's root: the bench scripts import it
(`validation/exp_r17_gen3_heldout.py --cubby450m`, `validation/exp_0081_live.py`).

Wired: STANDALONE (order 0081's benches).
"""
from __future__ import annotations

import os
import time

__wiring__ = "STANDALONE"

ARMS = ("off", "rel", "rel+slot")


def names(*worlds) -> set:
    """Every subject and object the worlds' triple indexes hold (`cubby_serve.build.slot_index`'s names)."""
    out: set = set()
    for w in worlds:
        idx = getattr(w, "index", None)
        for attr in ("_by_subj", "_by_obj"):
            out.update(getattr(idx, attr, {}) or {})
    return out


def build_seat(export: str, adapter: str, tokenizer: "str | None", worlds, known):
    from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk, short_name
    from cubbyllm.reasoning.slots import NameIndex, SlotEmitter
    trunk = Cubby450mTrunk(export, tokenizer or os.path.join(export, "tokenizer.json"), {"programs": adapter})
    seat = SlotEmitter(Cubby450mEmitter(trunk), index=NameIndex(names(*worlds)))
    seat.relations = known
    from cubbyllm.ops.grilly_seats import device_name
    seat.described = {"export": short_name(export), "adapter": short_name(adapter), "names": len(seat.index),
                      "card": device_name()}
    return seat


def emit(seat, question: str, arm: str, max_new: int, known) -> dict:
    """One emission under ``arm``: the program (filled), the plan `emitted_plan` reads, which of its relations the
    store holds, the refusal (a slot verdict), the trie's record and the time."""
    from cubbyllm.emit.base import strip_fences
    from cubbyllm.reasoning.emitted import emitted_plan
    from cubbyllm.reasoning.planner import normalize
    from cubbyllm.reasoning.slots import SlotRefused
    os.environ["CB_EMIT_TRIE"] = arm
    seat.last = {}
    t0 = time.perf_counter()
    out = {"arm": arm, "program": None, "plan": None, "seed": None, "refused": None}
    try:
        out["program"] = seat.emit(question, max_new_tokens=max_new)
    except SlotRefused as e:
        out["refused"] = "unknown_slot" if str(e).startswith("unknown slot") else "copied_name"
        out["why"] = str(e)[:160]
    except Exception as e:                               # noqa: BLE001 -- the proposer failed
        out["refused"] = "emit_error"; out["why"] = str(e)[:160]
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["raw"] = (seat.last or {}).get("raw")
    out["trie"] = (seat.last or {}).get("trie")
    ep = emitted_plan(strip_fences(out["program"]), normalize) if out["program"] else None
    if ep and ep[0]:
        out["plan"], out["seed"] = ep[0], ep[1]
        out["unheld"] = [r for r in ep[0] if r not in known]
    os.environ["CB_EMIT_TRIE"] = "off"
    return out


def options_described(seat, known) -> dict:
    """How many of the store's relations the trie holds, and the ones it left out (their tokens did not split
    cleanly after the role)."""
    from cubbyllm.emit import plan_trie
    o = plan_trie.relation_options(seat.inner.trunk.tokenizer, known)
    return {"relations": len(known), "held_in_trie": o.n, "skipped": len(o.skipped), "skipped_first": o.skipped[:20]}


__all__ = ["ARMS", "build_seat", "emit", "names", "options_described"]
