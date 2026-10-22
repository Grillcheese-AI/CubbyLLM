"""serve_events -- the server's JSON shaping and the loop's events, listened to (split out of serve_api.py,
order 0030): a ring of the last events (`/loop/events`, the board's journey timings) and a queue per live
client (`/loop/stream`).

Wired: WIRED (cubby_serve.api and cubby_serve.get).
"""
from __future__ import annotations

__wiring__ = "WIRED"


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


# ---- the loop's events, listened to (2026-09-13): a ring of the last events and a queue per live client ----
import collections as _collections
import queue as _queue
import threading as _threading

_LOOP_RING: "_collections.deque[dict]" = _collections.deque(maxlen=20000)
_LOOP_CLIENTS: list = []
_LOOP_LOCK = _threading.Lock()
_LOOP_WIRED = False


def _loop_sink(ev: dict) -> None:
    with _LOOP_LOCK:
        _LOOP_RING.append(ev)
        clients = list(_LOOP_CLIENTS)
    for q in clients:
        try:
            q.put_nowait(ev)
        except _queue.Full:
            pass


def ring_times() -> dict:
    """Each loop event's time by its id, from the ring (the board's journey timings, order 0027)."""
    with _LOOP_LOCK:
        return {e["id"]: e.get("t") for e in _LOOP_RING if "id" in e}


def wire_loop_events() -> None:
    """Register the server as a listener of `cubbyllm.reasoning.events` (idempotent)."""
    global _LOOP_WIRED
    if _LOOP_WIRED:
        return
    from cubbyllm.reasoning import events as loop_events
    loop_events.add_sink(_loop_sink)
    _LOOP_WIRED = True
