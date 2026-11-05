"""cubbyllm.selfimprove -- the self-improvement loop's smallest working form (order 0049, H-A15).

Gaps noticed, adapters made on demand, mistakes kept as "never this". The trunk stays frozen; only adapters learn; the
VM checks every answer. Off the serving path, and off unless ``CB_SELFIMPROVE=1``:
  signature   a request's skill: relations, language, program shape, asked from the value's side
  ledger      each request's signature and outcome (the VM's verdict; right or wrong against a gold where one exists)
  gaps        a signature that fails often enough AND that a short probe adapter learns (learning progress, H-S5)
  mistakes    per problem, the wrong attempts kept with the VM's reason until a right one; then (wrong, right) pairs
  router      a request to the adapter whose gap it matches, else the base; every decision logged; the gate (H-A7)
              and the lifecycle (loaded / latent)
The practice runs, the adapters' training and the stream are the harness's (``validation/selfimprove/``).

Wired: STANDALONE (order 0049's harness; nothing in serving imports it).
"""
from __future__ import annotations

import os

__wiring__ = "STANDALONE"


def on() -> bool:
    """``CB_SELFIMPROVE=1``: the loop runs (its harness). Off by default."""
    return os.environ.get("CB_SELFIMPROVE") == "1"
