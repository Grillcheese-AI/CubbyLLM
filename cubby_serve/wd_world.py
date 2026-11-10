"""cubby_serve/wd_world.py -- builds the hot Wikidata store's world for the ask loop (order 0055, step 3), behind
``CB_WD_STORE=1`` (`cubby_serve/cli.py`); off, nothing here runs.

The world (`cubbyllm/worlds/wd_store_world.py`) gets a function, not a process: this module starts its own
`wdstore serve` pipe (`board.store_client.StoreClient`, apart from the board's, so a walk never waits on a board
request) and hands the world that pipe's `ask`. The dump's date and the statement count come from the store's
manifest. Where the store and the binary are: `CUBBY_WD_STORE`, `CUBBY_WDSTORE_EXE` (the environment or
`paths.local.json`; never a committed file).

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
    world = WdStoreWorld(StoreClient(exe, store).ask, man.get("dump_date") or "", c.get("statements_hot", 0) + c.get("statements_cold", 0))
    log(f"  wikidata store: {len(world.relations())} relations, snapshot {str(world.state)[:12]}, dump {man.get('dump_date')}, "
        f"ready in {time.perf_counter() - t0:.1f} s (ahead of the Wikidata5M world)")
    return world
