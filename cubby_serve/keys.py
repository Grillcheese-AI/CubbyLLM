"""cubby_serve/keys.py -- where the serve stack keeps its host key: ``standin/data/out/host.key`` (a local secret,
git-ignored, generated once; ``CB_HOST_KEY`` overrides it). The vault in the package names no place of its own (order
0031, the review's answer 4); everything that encrypts or signs with it in the serve stack tells it this one first.

Wired: WIRED (the server at boot; the ledger, the harness wiring and the stand-in's two vault users).
"""
from __future__ import annotations

import pathlib

__wiring__ = "WIRED"

HOST_KEY_FILE = pathlib.Path(__file__).resolve().parents[1] / "standin" / "data" / "out" / "host.key"


def use_host_key() -> None:
    """Tell the vault where the host key is (once is enough; again changes nothing)."""
    from cubbyllm.speech import vault
    if vault.KEY_FILE is None:
        vault.use_key_file(HOST_KEY_FILE)
