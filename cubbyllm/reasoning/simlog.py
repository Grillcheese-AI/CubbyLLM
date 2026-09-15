"""WO-0.4 -- the similarity histogram at serve time.

Wired: WIRED (pipeline.py calls `record` at the one verified=True site).

Every spoken answer rests on one or more `recover()` bindings that cleared
`tau_vm`. The similarity of each is already computed and already sits on
`HopTrace.similarity` -- it is simply thrown away after the comparison. This
module keeps it, as one JSONL line per spoken answer, so the *distribution*
can be read rather than only the pass/fail.

Two things the distribution shows that the threshold cannot:

1. **The fraction of spoken bindings below 0.5 rising over rounds.** Bindings
   that clear tau_vm but sit far below a confident match are the ones that
   will flip when the world changes slightly.
2. **Mass concentrated just above a tau boundary** (0.22-0.30 at hop 3, where
   tau_vm is 0.22021484375). Decisions riding the threshold are the
   confabulation signature: the cleanup is returning the nearest symbol rather
   than the right one, and the threshold is the only thing standing between
   that and a spoken answer.

Context for why this matters more than it looks: `recover` on an unbound role
in an already-bound frame returns the globally-nearest symbol at ~0.03 -- it
never returns Null (cubelang docs/DRIFT.md B11a). Two models reviewing the
design called a never-Null `recover` incompatible with the kill line outright.
The threshold is therefore load-bearing, and a load-bearing threshold with an
unwatched distribution is exactly the thing to instrument.

Off by default and free when off: set `CUBBY_SIMLOG` to a path to turn it on.

    CUBBY_SIMLOG=validation/logs/similarity.jsonl python -m ...
    python validation/sim_histogram.py validation/logs/similarity.jsonl

Never raises. A logging sink that can break serving is worse than no sink, so
every failure here is swallowed -- the caller is in the middle of answering.
"""
from __future__ import annotations

import json
import os
import time

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

_ENV = "CUBBY_SIMLOG"


def enabled() -> bool:
    return bool(os.environ.get(_ENV))


def record(question: str, answer: str, tau_vm: float, trace, ctrl_sim: float | None = None) -> None:
    """Append one line for a spoken answer. No-op unless CUBBY_SIMLOG is set.

    `trace` is the list of `HopTrace`; every entry's similarity is >= tau_vm at
    the call site (the pipeline asserts it), so these are exactly the ACCEPTED
    bindings -- which is the population WO-0.4 asks about. `ctrl_sim` is the
    control role's similarity, which must stay BELOW tau_vm; logging it beside
    the accepted ones makes the separation between the two visible, and a
    shrinking separation is the same alarm arriving early.
    """
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        sims = [h.similarity for h in trace]
        row = {
            "t": round(time.time(), 3),
            "question": question,
            "answer": answer,
            "tau_vm": tau_vm,
            "n_hop": len(trace),
            "sims": sims,
            "min_sim": min((s for s in sims if s is not None), default=None),
            "ctrl_sim": ctrl_sim,
            # The margin is the number to watch: how far the WEAKEST accepted
            # binding sat above the threshold it had to clear. Mass piling up
            # near zero here is the "riding the threshold" alarm.
            "margin": (min((s for s in sims if s is not None), default=None) - tau_vm)
                      if any(s is not None for s in sims) else None,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - a telemetry sink must never break serving
        pass
