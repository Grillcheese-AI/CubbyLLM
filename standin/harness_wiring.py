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

from harness import Fact, Tool, current_frame

__wiring__ = "STANDALONE"

_USE = re.compile(r"^\s*use\s+([A-Za-z_][\w.]*)", re.M)


class GatedStore:
    """A FactStore behind the registry. Everything but a query and an add passes through."""

    def __init__(self, name: str, store, harness):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_harness", harness)

    @property
    def raw(self):
        return self._store

    def __call__(self, query: str, k: int):
        f = current_frame()
        if f is None:
            return self._store(query, k)
        return self._harness.registry.call(f, "store.read", world=self._name, query=query, k=k)

    def add(self, text: str, *args, **kwargs):
        f = current_frame()
        if f is None:
            return self._store.add(text, *args, **kwargs)
        fact = Fact(" ".join(str(text).split()), self._name, status="asserted", provenance=kwargs.get("source"))
        return self._harness.registry.call(f, "store.write", fact=fact, extra={"args": list(args), "kwargs": dict(kwargs)})

    def __contains__(self, text) -> bool:
        return text in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self):
        return iter(self._store)

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._store, name, value)              # `w.oracle = ...` (mount_oracle) reaches the store

    def __repr__(self) -> str:
        return f"GatedStore({self._name!r}, {self._store!r})"


class GatedWorlds(dict):
    """`brain.worlds` with every value gated; a world mounted later is gated on the way in."""

    def __init__(self, worlds: dict, harness):
        super().__init__()
        self._harness = harness
        for name, w in worlds.items():
            self[name] = w

    def __setitem__(self, name: str, w) -> None:
        if not isinstance(w, GatedStore):
            w = GatedStore(name, w, self._harness)
        super().__setitem__(name, w)


def _raw_store(harness, name: str):
    w = harness.brain.worlds.get(name)
    if w is None:
        raise KeyError(f"no world {name}")
    return w.raw if isinstance(w, GatedStore) else w


def declare_doors(harness, max_ops: int | None = None, max_queries: int | None = None) -> None:
    """The tools the doors call: store.read, store.write, vm.run (+ promote), each ledgered and budgeted."""
    reg = harness.registry
    import cubbyllm.bridges.cubelang_client as cc
    orig_run = getattr(cc, "_harness_orig_run", None) or cc.run_program_proto

    def read(world: str, query: str, k: int):
        return _raw_store(harness, world)(query, k)

    def write(fact: Fact, extra: dict | None = None):
        extra = extra or {}
        return _raw_store(harness, fact.world).add(fact.text, *extra.get("args", []), **extra.get("kwargs", {}))

    def vm_run(uses=None, **kw):
        return orig_run(**kw)

    def bind(frame, args: dict) -> dict:
        bound = {"budget": frame.budget.vm(max_ops, max_queries)}
        kp = harness.stores.get("knowledge")
        if kp:
            bound["knowledge_path"] = kp
        return bound

    reg.declare(Tool("store.read", "read", frozenset({"read_store"}), fn=read, spends="reads", doc="(query, k) on a world"))
    reg.declare(Tool("store.write", "write", frozenset({"write_store"}), fn=write, doc="world.add(fact)"))
    reg.declare(Tool("vm.run", "read", frozenset({"vm"}), fn=vm_run, spends="vm_runs", bind=bind,
                     doc="cubelang run-proto; budget and knowledge set by the harness"))
    reg.promote_tool(lambda fact: _raw_store(harness, fact.world).add(fact.text, source=fact.provenance))
    reg.declare_vm_modules(harness.config.get("vm_modules") or {})


def install_vm_door(harness) -> None:
    """`cubelang_client.run_program_proto` goes through the registry inside a turn (resume flows through it)."""
    import cubbyllm.bridges.cubelang_client as cc
    if getattr(cc, "_harness_door", None) is not None:
        cc._harness_door = harness                       # a second harness (tests) takes the door over
        return
    orig = cc.run_program_proto
    sig = inspect.signature(orig)

    def door(*args, **kwargs):
        h = cc._harness_door
        f = current_frame()
        if h is None or f is None:
            return orig(*args, **kwargs)
        kw = dict(sig.bind_partial(*args, **kwargs).arguments)
        src = kw.get("program_source", "")
        uses = _USE.findall(src) if h.registry.vm_modules else []
        return h.registry.call(f, "vm.run", uses=uses, **kw)

    cc._harness_orig_run = orig
    cc._harness_door = harness
    cc.run_program_proto = door


def uninstall_vm_door() -> None:
    import cubbyllm.bridges.cubelang_client as cc
    if getattr(cc, "_harness_orig_run", None) is not None:
        cc.run_program_proto = cc._harness_orig_run
        cc._harness_orig_run = None
        cc._harness_door = None


def gate_emitter(emitter) -> None:
    """`emitter.emit` spends gen_tokens inside a turn; capped to what the frame has left."""
    if getattr(emitter, "_harness_gated", False):
        return
    orig = emitter.emit

    def emit(prompt, *args, **kwargs):
        f = current_frame()
        if f is None:
            return orig(prompt, *args, **kwargs)
        left = f.budget.remaining("gen_tokens")
        if left <= 0:
            f.budget.spend("gen_tokens")                 # raises Exhausted("gen_tokens") before the model runs
        want = int(kwargs.get("max_new_tokens", 450))
        kwargs["max_new_tokens"] = min(want, left)
        out = orig(prompt, *args, **kwargs)
        text = out if isinstance(out, str) else str(out)
        f.budget.spend("gen_tokens", n=max(1, min(kwargs["max_new_tokens"], len(text) // 4)))
        return out

    emitter.emit = emit
    emitter._harness_gated = True


class RoutedBrain:
    """The brain as the harness drives it: SENSED and routed once, outside the frame, in the brain's own order
    (appraisal -> hormones -> route), the route handed to the turn so nothing is re-read inside the frame."""

    def __init__(self, brain):
        self._brain = brain

    def route(self, text: str) -> dict:
        b = self._brain
        if hasattr(b, "sense"):
            b.sense(text)
        return b.route(text)

    def turn(self, user_text: str, feedback: str | None = None, frame=None) -> dict:
        b = self._brain
        params = inspect.signature(b.turn).parameters
        kw: dict[str, Any] = {}
        if "route" in params and frame is not None and getattr(frame, "route", None):
            kw["route"] = frame.route
            if "sensed" in params:
                kw["sensed"] = True
        if "frame" in params:
            kw["frame"] = frame
        return b.turn(user_text, feedback, **kw)

    def __getattr__(self, name: str):
        return getattr(self._brain, name)


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
    import vault
    from harness import DEFAULT_BUDGET, file_sha, sign_config
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
    import vault
    from harness import Harness, Preflight, ToolRegistry
    from ledger import Ledger
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
