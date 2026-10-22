"""cubbyllm/worlds/science.py -- the science worlds on a serving brain (``serve_api --science``; order 0031 split it out of
``standin/data/import_science.py``, which builds them from MegaScience rows and stays a builder).

Wired: WIRED (the server's ``--science``).
"""
from __future__ import annotations

import os

__wiring__ = "WIRED"


def mount_science(brain, folder: str, attested_only: bool = True) -> dict[str, int]:
    """Mount every saved science world on a serving brain as `brain.worlds[name]`: the ATTESTED facts only
    (a latent fact is held from speech), retrieved with the brain's own encoder. -> {name: facts mounted}."""
    from capsules import CapsuleStore
    from worlds import FactStore
    enc = getattr(brain.worlds.get("facts"), "enc", None)
    out = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(os.path.join(path, "capsules.meta.json")):
            continue
        store = CapsuleStore.load(path, gpu=False)
        texts = [t for t in store.texts if not (attested_only and store.meta[t].get("latent") is not None)]
        if texts:
            brain.worlds[name] = FactStore(texts, enc=enc, name=name)
            out[name] = len(texts)
    return out
