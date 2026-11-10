"""cubby_serve/wd_world.py -- builds the hot Wikidata store's world for the ask loop (order 0055, step 3), behind
``CB_WD_STORE=1`` (`cubby_serve/cli.py`); off, nothing here runs.

The world (`cubbyllm/worlds/wd_store_world.py`) gets a function, not a process: this module starts its own
`wdstore serve` pipe (`board.store_client.StoreClient`, apart from the board's, so a walk never waits on a board
request) and hands the world that pipe's `ask`. The dump's date and the statement count come from the store's
manifest. Where the store and the binary are: `CUBBY_WD_STORE`, `CUBBY_WDSTORE_EXE` (the environment or
`paths.local.json`; never a committed file). With ``CB_WD_KINDS=1`` the world also reads the store's kinds table
(`<store dir>.kinds.sqlite`, or `CUBBY_WD_KINDS_TABLE`) and its cue table with the fictional classes
(`<store dir>.cues.sqlite`, or `CUBBY_WD_CUES_TABLE`), so order 0037's rule decides its shared names; off until that
rule's audit on the store is read (``validation/audit_wd_kinds.py``).

Wired: WIRED (`cubby_serve/cli.py`, ``CB_WD_STORE=1``).
"""
from __future__ import annotations

import json
import os
import time

from cubbyllm.local_paths import value

__wiring__ = "WIRED"


def build(log=print):
    """The store's world, or SystemExit when the flag is on and the store cannot be read (never a silent off)."""
    from cubby_serve.board.store_client import StoreClient
    from cubbyllm.worlds.wd_store_world import WdStoreWorld
    store, exe = value("CUBBY_WD_STORE"), value("CUBBY_WDSTORE_EXE")
    if not store or not exe or not os.path.isdir(store) or not os.path.isfile(exe):
        raise SystemExit("CB_WD_STORE=1 needs CUBBY_WD_STORE (the store's folder) and CUBBY_WDSTORE_EXE (wdstore)")
    with open(os.path.join(store, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    c = man.get("counts", {})
    t0 = time.perf_counter()
    on = os.environ.get("CB_WD_KINDS") == "1"
    kinds = (value("CUBBY_WD_KINDS_TABLE") or os.path.normpath(store) + ".kinds.sqlite") if on else None
    cues = (value("CUBBY_WD_CUES_TABLE") or os.path.normpath(store) + ".cues.sqlite") if on else None
    world = WdStoreWorld(StoreClient(exe, store).ask, man.get("dump_date") or "", c.get("statements_hot", 0) + c.get("statements_cold", 0),
                         kinds=kinds, cues=cues)
    if on and (world._kinds is None or world._cues is None):
        raise SystemExit(f"CB_WD_KINDS=1: the kinds table ({kinds}) and the cue table ({cues}) must both be built from this store")
    log(f"  wikidata store: {len(world.relations())} relations, snapshot {str(world.state)[:12]}, dump {man.get('dump_date')}, "
        f"kinds table {'read' if world._kinds is not None else 'off'}, ready in {time.perf_counter() - t0:.1f} s "
        f"(ahead of the Wikidata5M world)")
    return world
