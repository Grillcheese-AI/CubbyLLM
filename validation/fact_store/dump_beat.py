"""validation/fact_store/dump_beat.py -- order 0055: a line every minute from a dump pass, so a long pass that is
working never looks stuck, and one that is stuck says where.

``live`` is a dict the pass updates as it goes: ``phase`` (what it is doing), and any of ``part`` / ``parts`` (the
compressed part being read), ``at`` (decompressed bytes read so far) and ``fed_all`` (every part has been handed to
the decompressor: the pass is at the dump's end). ``beat`` prints them every ``every`` seconds until stopped.

Wired: STANDALONE (imported by dump_count.py).
"""
from __future__ import annotations

import threading
import time

__wiring__ = "STANDALONE"


def line(live: dict, minutes: float) -> str:
    bits = [live.get("phase", "working")]
    if "part" in live:
        bits.append(f"reading part {live['part'] + 1} of {live['parts']}")
    if "at" in live:
        bits.append(f"{live['at'] / 1e9:.1f} GB decompressed")
    if live.get("fed_all"):
        bits.append("every part read")
    return f"  [{time.strftime('%H:%M')}, {minutes:.0f} min] " + "; ".join(bits)


def beat(live: dict, every: float = 60.0) -> threading.Event:
    """Starts the beat; set the returned event to stop it."""
    stop, t0 = threading.Event(), time.time()

    def run() -> None:
        while not stop.wait(every):
            print(line(live, (time.time() - t0) / 60), flush=True)
    threading.Thread(target=run, daemon=True).start()
    return stop
