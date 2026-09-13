"""Every step of the loop as an event (2026-09-13, Nick: "all process should be listened to at every
step so we can visually check what's happening in a three.js control panel with links between query
and such").

The host already computes each step as a value -- the question, the plan, the disposer's verdict,
each hop's lookup and fact, the sources asked and what the gate said, the VM call, the answer or
the refusal. `emit(kind, parent, **fields)` records that value as an event with an id and its
parent's id, so a listener can draw the run as a graph: a node per event, an edge to its parent.
No sink registered = nothing happens (a dict built and dropped); the loop's behaviour is untouched
either way. `JsonlSink` appends one JSON line per event; a live listener (the stand-in server's SSE
endpoint, the three.js panel) registers a callable.

Kinds and their links (parent -> child):
  question -> plan -> verdict | alias | hop -> fact | vm -> answer
  question -> fetch -> gate (one per fact the source handed over)
  question -> latent (a verified chain held back: the LFM tier)
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from typing import Callable

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

_lock = threading.Lock()
_sinks: list[Callable[[dict], None]] = []
_ids = itertools.count(1)


def add_sink(fn: Callable[[dict], None]) -> Callable[[dict], None]:
    with _lock:
        _sinks.append(fn)
    return fn


def remove_sink(fn: Callable[[dict], None]) -> None:
    with _lock:
        if fn in _sinks:
            _sinks.remove(fn)


def listening() -> bool:
    return bool(_sinks)


def emit(kind: str, parent: int | None = None, **fields) -> int:
    """Record one step. Returns the event's id (the parent for what follows it). Cheap when no
    one listens: the id is still handed out so links stay consistent within a run."""
    eid = next(_ids)
    if not _sinks:
        return eid
    ev = {"id": eid, "t": round(time.time(), 3), "kind": kind, "parent": parent}
    ev.update(fields)
    with _lock:
        sinks = list(_sinks)
    for s in sinks:
        try:
            s(ev)
        except Exception:                       # noqa: BLE001 -- a listener never breaks the loop
            pass
    return eid


def emit_walk(parent: int | None, res, provenance: dict | None = None, key=None) -> int:
    """A walk's result as events: the walk (verdict, answer, refusal), a hop per trace entry with its
    fact (and the fact's provenance when the store keeps one), the VM call when a program was built,
    and the latent hold when one applies. Returns the walk event's id."""
    if not _sinks:
        return next(_ids)
    fk = key or (lambda s: " ".join(s.split()))
    refused = res.refused if isinstance(getattr(res, "refused", None), dict) else None
    wid = emit("walk", parent, verified=res.verified, reason=res.reason, answer=res.answer, refused=refused)
    for i, h in enumerate(res.trace or []):
        prov = (provenance or {}).get(fk(h.fact)) if h.fact else None
        fid = emit("hop", wid, hop=i, query=h.query, fact=h.fact, how=h.source, similarity=h.similarity, provenance=prov)
        if h.fact:
            t = h.triple
            emit("fact", fid, text=h.fact, subj=t.subj if t else None, rel=t.rel if t else None, obj=t.obj if t else None)
    if getattr(res, "source", None):
        emit("vm", wid, verified=bool(res.verified and res.reason is None), program_lines=res.source.count("\n") + 1,
             repairs=getattr(res, "repairs_used", 0))
    if res.reason == "latent_only":
        emit("latent", wid, answer=(refused or {}).get("answer"), facts=(refused or {}).get("latent_facts"))
    return wid


def emit_answer(root: int | None, res, **extra) -> int:
    refused = res.refused if isinstance(getattr(res, "refused", None), dict) else None
    return emit("answer", root, verified=res.verified, answer=res.answer, reason=res.reason, refused=refused, **extra)


class JsonlSink:
    """One JSON line per event, appended; `close()` when the run ends."""

    def __init__(self, path) -> None:
        self.path = str(path)
        self._f = open(self.path, "a", encoding="utf-8")
        self.n = 0

    def __call__(self, ev: dict) -> None:
        self._f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        self.n += 1
        if self.n % 50 == 0:
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush(); self._f.close()
        except Exception:                       # noqa: BLE001
            pass


class MemorySink(list):
    """Events kept in a list (tests, the panel's replay)."""

    def __call__(self, ev: dict) -> None:
        self.append(ev)
