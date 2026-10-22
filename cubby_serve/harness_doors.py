"""cubby_serve/harness_doors.py -- the harness's doors: the store and the worlds behind the gate, the doors declared,
the VM's door installed and removed, the emitter gated, the brain routed through it. Split out of harness_wiring.py
(order 0031), verbatim; harness_wiring.py re-exports every name.

Wired: WIRED (the harness wiring).
"""
from __future__ import annotations

import inspect
import re
from typing import Any
from cubby_serve.harness import Fact, Tool, current_frame

__wiring__ = "WIRED"


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
