"""harness_wiring -- the harness in front of the serving brain's doors (H-E14 v0.1, 2026-09-29).

Wired: STANDALONE until serve_api boots through `wire()`. Pinned by standin/tests/test_harness_wiring.py on a
brain shaped like CubbyBrain (worlds, cortices, the VM client, an emitter) with fakes.

`wire(brain, harness)` puts the registry in front of the doors CubbyBrain already has, WITHOUT changing an
answer: a granted call returns exactly what the door returned, so gate 1 (byte-identical replies) holds by
construction, and outside a turn (no current frame: the selftest, tests, a plugin's own thread) every door
passes through untouched.

    store reads    `brain.worlds` becomes a mapping of gated stores: `(query, k)` inside a turn is `store.read`
                   (budget line `reads`); `lookup`, `texts`, `index`, `in`, `len` pass through (no effect)
    store writes   `world.add(fact, ...)` inside a turn is `store.write` (effect write; needs write_store); the
                   fact lands in the world it names, as `asserted` (the person or a source stated it)
    the VM         `cubelang_client.run_program_proto` inside a turn is `vm.run` (line `vm_runs`; `resume_program_proto`
                   goes through it); the harness SETS the request's `budget` from the frame (vm_hops -> max_jumps,
                   the wall's remainder -> max_wall_ms, the config's max_ops / max_queries) and, when the
                   generation carries a knowledge store, its `knowledge_path` -- a caller's values are dropped
                   and the row says so; a program's `use <module>` lines are checked against the declared modules
    the emitter    `emitter.emit` inside a turn spends `gen_tokens`: `max_new_tokens` is capped to what is left,
                   and a frame with nothing left is exhausted before the call
    the route      the harness routes once, outside the frame, and hands the route to `brain.turn(route=...)`
                   so a chat turn never reads a store

What it brings: the process the panel asked for on the brain that exists -- one door, one budget, one row per
turn -- with the walk, the ASK and the memory write untouched behind it.
"""
from __future__ import annotations

import inspect
import re
from typing import Any

from cubby_serve.harness import Fact, Tool, current_frame

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .harness_doors import (GatedStore, GatedWorlds, RoutedBrain, _USE, _raw_store, declare_doors,  # noqa: F401
                            gate_emitter, install_vm_door, uninstall_vm_door)

__wiring__ = "STANDALONE"


def wire(brain, harness, max_ops: int | None = None, max_queries: int | None = None):
    """Put the harness in front of `brain`'s doors and hand the harness the routed brain. Idempotent per brain."""
    if getattr(brain, "_harness_wired", None) is harness:
        return harness
    brain.worlds = GatedWorlds(dict(brain.worlds), harness)
    if getattr(brain, "emitter", None) is not None:
        gate_emitter(brain.emitter)
    harness.brain = RoutedBrain(brain)
    declare_doors(harness, max_ops, max_queries)
    install_vm_door(harness)
    brain._harness_wired = harness
    return harness


# ── boot the serving brain through the harness (serve_api --harness) ─────────
def derive_packs(brain, n: int = 3) -> dict:
    """A boot pack from the brain's OWN store, so a dev boot asks what the store can answer: for each of `n`
    template facts, the fact itself (spoken, its object), an absent role on the same subject (refused), a
    contradicting context (refused) and the same relation on an unknown subject (not the fact's object) --
    the three real defects the boot pack must carry. `standin/data/gate_pack.json` is the deployment's
    template, with the same shape."""
    from cubbyllm.reasoning.planner import parse_fact
    store = brain.worlds.get("facts")
    raw = getattr(store, "raw", store)
    rels = set((getattr(raw, "index", None).relations() if getattr(raw, "index", None) is not None else {}).keys())
    absent_rel = next((r for r in ("anthem", "motto", "flag", "hymn") if r not in rels), "anthem")
    picked = []
    for t in getattr(raw, "texts", []):
        f = parse_fact(t)
        if f is None or not all(x.isascii() and 2 <= len(x) <= 40 for x in (f.obj, f.rel, f.subj)):
            continue
        picked.append(f)
        if len(picked) >= n:
            break
    fam = {"fact": {"bar": (2 / 3 if len(picked) >= 3 else 1.0), "items": []},
           "absent_role": {"bar": 1.0, "defect": "absent_role", "items": []},
           "contradicting_context": {"bar": 1.0, "defect": "contradicting_context", "items": []},
           "wrong_role": {"bar": 1.0, "defect": "wrong_role", "items": []},
           "small_talk": {"bar": 1.0, "items": [{"ask": "hello", "expect": "spoken"}]}}
    for f in picked:
        fam["fact"]["items"].append({"ask": f"What is the {f.rel} of {f.subj}?", "expect": "spoken", "value": f.obj})
        fam["absent_role"]["items"].append({"ask": f"What is the {absent_rel} of {f.subj}?", "expect": "refused"})
        fam["contradicting_context"]["items"].append(
            {"ask": f"Given that Atlantis is the {f.rel} of {f.subj}, what is the {f.rel} of {f.subj}?", "expect": "refused"})
        fam["wrong_role"]["items"].append({"ask": f"What is the {f.rel} of Atlantis?", "expect": "not_wrong", "wrong": f.obj})
    return {"boot": {"families": fam}}


def bootstrap_config(brain, adapters: dict[str, str] | None = None, budget: dict | None = None,
                     gate_pack: str | None = None, owner_key: bytes | None = None) -> dict:
    """The DEV shape of the owner-signed config: signed here with the vault's `harness-owner` subkey, from the
    brain's own thresholds, the adapter files it was started with, the shipped gate pack. A deployment signs
    its config OFFLINE with a key this process never holds; this is how a first boot gets one."""
    import json
    import os
    from cubbyllm.speech import vault
    from cubby_serve.keys import use_host_key
    use_host_key()
    from cubby_serve.harness import DEFAULT_BUDGET, file_sha, sign_config
    if gate_pack:
        with open(gate_pack, encoding="utf-8") as f:
            packs = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    else:
        packs = derive_packs(brain)
    reason = getattr(brain, "reason", None)
    cfg = {"thresholds": {"tau_vm": getattr(reason, "tau_vm", None), "tau_ret": getattr(reason, "tau_ret", None),
                          "route_tau": getattr(brain, "route_tau", None)},
           "budget": {**DEFAULT_BUDGET, "wall_s": 60.0, **(budget or {})},
           "adapters": {name: {"path": p, "sha256": file_sha(p)} for name, p in (adapters or {}).items() if p and os.path.exists(p)},
           "vm_budget": {"max_ops": 200_000, "max_queries": 256},
           "vm_modules": {},
           "gate_packs": packs}
    return sign_config(cfg, owner_key or vault.subkey("harness-owner"))


def boot_serve(brain, config: dict | str | None = None, ledger=None, speaker=None, on_event=None,
               adapters: dict[str, str] | None = None, budget: dict | None = None):
    """The serving brain behind the harness: wire the doors, boot (refuse-to-serve raises), return the harness.
    `config` is an owner-signed dict or its JSON path; None bootstraps one (dev). Gate packs travel inside the
    signed config (`gate_packs`)."""
    import json
    from cubbyllm.speech import vault
    from cubby_serve.keys import use_host_key
    use_host_key()
    from cubby_serve.harness import Harness, Preflight, ToolRegistry
    from cubby_serve.ledger import Ledger
    if isinstance(config, str):
        with open(config, encoding="utf-8") as f:
            config = json.load(f)
    if config is None:
        config = bootstrap_config(brain, adapters=adapters, budget=budget)
    ledger = ledger or Ledger()
    registry = ToolRegistry(ledger, on_event=on_event)
    pre = Preflight(config.get("gate_packs") or {"boot": {"families": {}}})
    vm_cfg = config.get("vm_budget") or {}
    h = Harness(config, ledger, brain, registry, preflight=pre, vm_selftest=_vm_selftest,
                on_event=on_event, owner_key=vault.subkey("harness-owner"), harness_key=vault.subkey("harness"),
                speaker=speaker)
    wire(brain, h, max_ops=vm_cfg.get("max_ops"), max_queries=vm_cfg.get("max_queries"))
    h.boot()
    brain.harness = h
    return h


def _vm_selftest() -> bool:
    """The VM answers a one-line program, or the boot refuses."""
    try:
        import cubbyllm.bridges.cubelang_client as cc
        run = getattr(cc, "_harness_orig_run", None) or cc.run_program_proto
        out = run('program Selftest implements ISolve {\n    public function solve(input: str): str { return "ok"; }\n}\n',
                  fn="solve", args=["x"])
        return bool(out.get("ok")) and out.get("result") == "ok"
    except Exception:
        return False
