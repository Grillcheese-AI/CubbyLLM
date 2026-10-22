"""harness -- one process that owns boot, state, the turn and the night (v0.1, 2026-09-29).

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this). Spec: docs/research/2026-09-28-harness-v0.md and
the round-3 amendments (docs/research/2026-09-28-panel-round3-synthesis.md, "The harness: v0 -> v0.1");
pre-registered as H-E14. Pinned by standin/tests/test_harness.py with fakes: no model, no VM, no GPU. Wired to the
serving brain by standin/harness_wiring.py.

What it adds: nothing. What it makes one system: the parts that exist.

    boot     the OWNER-signed config (thresholds, gate packs, adapters, the live emitter) -> the current store
             GENERATION (a manifest this process signed after the gate pack passed on it) -> VM self-test ->
             adapters' hashes -> the live emitter's size cap -> the boot pack -> serve, or REFUSE, with a ledger
             row either way; orphan intents from a killed process are resolved, never re-spoken. The preflight
             never moves a threshold.
    turn     a TaskFrame with a Budget and a TYPED postcondition declared at creation; every tool call through
             ONE registry (deny by default, effect classes at call time, budgets spent, every call ledgered; the
             harness sets a program's knowledge path and jump budget from the frame, never the emitter); the
             brain's own turn does the work; the row carries everything that bent the turn (hormones,
             generation, context delta, model and adapter hashes, decode seed); SPEAKING is an effect -- an
             intent row before the bytes leave, a spoken row after, at most once.
    night    writes generation g+1 beside g, never touching g; the gate pack runs against g+1; a signed manifest
             is swapped atomically; turns in flight keep the generation they pinned. No lock: the day serves.
    replay   AUDIT replay (the recorded model outputs and the deterministic host: must reproduce the reply hash)
             apart from REGENERATION replay (re-decoding on the engine: a property of the engine, measured).

Two keys. The owner's signs the config -- thresholds, gate packs, bars, adapter promotion -- offline, never by
this process. The harness's signs generation manifests, only after the owner-signed pack passed on them.

The model proposes and phrases. The harness never lets it decide what is true, whether a check passed, which
tool runs, what is written, which world a fact lands in, whether content is allowed, which knowledge a program
reads, how many jumps it may take, or whether to speak after a failed check.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .harness_rules import (BRANCH_ONLY, BRANCH_PREFIX, Budget, DEFAULT_BUDGET, EFFECTS, Exhausted,  # noqa: F401
                            Fact, GRANTS, NEVER_LIVE, NO_EXTERNAL, POSTCONDITIONS, REQUIRED_DEFECTS, Refused,
                            STATUSES, TaskFrame, Tool, _jsonable, _sign, file_sha, sha, sign_config)
from .harness_tools import Generations, Preflight, ToolRegistry  # noqa: F401
from .harness_night import HarnessNight

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

__wiring__ = "STANDALONE"


# ── the harness ─────────────────────────────────────────────────────────────
class RefuseToServe(Exception):
    pass


_current = threading.local()                             # the frame of the turn running on this thread, if any


def current_frame() -> TaskFrame | None:
    return getattr(_current, "frame", None)


class Harness(HarnessNight):
    """One process. `boot()` then `turn()`; `night()` writes the next generation; `replay()` from the ledger.

    `speaker(rec) -> None` is the channel the reply leaves by (a socket, a console); the harness commits the
    intent row before calling it and the spoken row after, and never calls it twice for one turn. `brain` may
    offer `view(stores)` (a read-only view on another generation, for the gate pack), `use_generation(stores)`,
    `hormones()`, `set_hormones(h)`, `replay(user_text, model_outputs)` (the deterministic host over recorded
    model outputs) -- each used when present."""

    def __init__(self, config: dict, ledger, brain, registry: ToolRegistry, preflight: Preflight | None = None,
                 vm_selftest: Callable[[], bool] | None = None, night_fn: Callable[..., dict] | None = None,
                 on_event: Callable[[dict], None] | None = None, config_key: bytes | None = None,
                 owner_key: bytes | None = None, harness_key: bytes | None = None,
                 generations: Generations | None = None, speaker: Callable[[dict], None] | None = None):
        self.config, self.ledger, self.brain, self.registry = config, ledger, brain, registry
        self.preflight, self.vm_selftest, self.night_fn = preflight, vm_selftest, night_fn
        self.on_event = on_event or (lambda ev: None)
        self.owner_key = owner_key or config_key          # `config_key` is v0's name for the owner's key
        self.harness_key = harness_key
        self.generations = generations
        self.speaker = speaker or (lambda rec: None)
        self.serving = False
        self.generation: int | None = None
        self.stores: dict[str, str] = {}
        self.turns: dict[str, dict] = {}
        self.interrupted: list[dict] = []                 # orphan intents resolved at boot: never re-spoken
        self._spoken: set[str] = set()
        self._n = 0
        self._lock = threading.RLock()                    # the turn counter and the speak guard, not the stores
        registry.declare_vm_modules(config.get("vm_modules") or {})   # what a program may `use`, from the signed config

    # ── boot ──────────────────────────────────────────────────────────────
    def boot(self) -> dict:
        checks = []

        def check(name: str, ok: bool, detail: str = ""):
            checks.append({"check": name, "ok": bool(ok), "detail": detail})
            self.on_event({"kind": "boot", "check": name, "ok": bool(ok), "detail": detail})
            return ok

        cfg = self.config
        if self.owner_key is not None:                    # the config is owner-signed; a changed threshold fails here
            body = {k: v for k, v in cfg.items() if k != "signature"}
            check("config.signature", hmac.compare_digest(cfg.get("signature", ""), _sign(self.owner_key, body)))
        if self.generations is not None:                  # the store generation: a manifest THIS process signed
            man = self.generations.current()
            ok, why = (False, "no manifest") if man is None else self.generations.verify(man)
            check("generation", ok, why)
            if ok:
                self.generation, self.stores = man["generation"], self.generations.paths(man)
        for name, spec in cfg.get("stores", {}).items():  # v0: owner-signed store hashes (no generations)
            p = spec.get("path")
            ok = bool(p) and os.path.exists(p) and file_sha(p) == spec.get("sha256")
            check(f"store.{name}", ok, "" if ok else "hash differs or missing")
            if ok:
                self.stores[name] = p
        check("vm.selftest", self.vm_selftest() if self.vm_selftest else True)
        for name, spec in cfg.get("adapters", {}).items():
            p = spec.get("path")
            ok = bool(p) and os.path.exists(p) and file_sha(p) == spec.get("sha256")
            check(f"adapter.{name}", ok, "" if ok else "hash differs or missing")
        live = (cfg.get("emitters") or {}).get("live")
        if live is not None:                              # the live emitter fits the card beside the base, or nothing serves (change 9)
            cap = float(cfg.get("max_live_params_m", 600))
            ok = float(live.get("params_m", 0)) <= cap
            if ok and live.get("path"):
                ok = os.path.exists(live["path"]) and file_sha(live["path"]) == live.get("sha256")
            check("emitter.live", ok, "" if ok else f"{live.get('name')}: {live.get('params_m')}M vs cap {cap:g}M, or hash differs")
        try:
            self.ledger.record("boot", name="boot", kind="boot", fn="ledger.write", input=None,
                               expected=None, got=None, ok=True, verdict="probe")
            check("ledger.write", True)
        except Exception as e:
            check("ledger.write", False, str(e)[:200])
        if self.preflight is not None:
            missing = self.preflight.missing_defects("boot")
            check("preflight.defects", not missing, f"boot pack lacks {missing}" if missing else "")
        if self.preflight is not None and all(c["ok"] for c in checks):
            rep = self.preflight.run(self._preflight_answer, "boot")
            check("preflight", rep["passed"], f"failed families: {rep['failed']}" if rep["failed"] else "")
            checks[-1]["report"] = rep["families"]
        passed = all(c["ok"] for c in checks)
        orphans = self.recover() if passed else []
        if passed:                                        # turn ids continue the ledger's: a restart never reuses one
            self._n = max(self._n, len(self.ledger.rows("turn")))
        row = {"checks": checks, "config_sha256": sha({k: v for k, v in cfg.items() if k != "signature"}),
               "generation": self.generation, "orphan_intents": len(orphans)}
        self.ledger.record(json.dumps(row, sort_keys=True), name="boot", kind="boot", fn=None, input=None,
                           expected="serve", got="serve" if passed else "refuse", ok=passed,
                           verdict="serving" if passed else "REFUSED: " + ", ".join(c["check"] for c in checks if not c["ok"]))
        self.serving = passed
        if not passed:
            raise RefuseToServe(", ".join(c["check"] for c in checks if not c["ok"]))
        return row

    def recover(self) -> list[dict]:
        """Ledger-first speaking, on restart: an intent without a spoken row was interrupted between the commit and
        the bytes leaving. It is resolved as such, and never re-spoken."""
        spoken = {r["name"] for r in self.ledger.rows("speak", fn="spoken")}
        resolved = {r["name"] for r in self.ledger.rows("speak", fn="interrupted")}
        out = []
        for r in self.ledger.rows("speak", fn="intent"):
            if r["name"] in spoken or r["name"] in resolved:
                continue
            h = self.ledger.record(f"speak:{r['name']}", name=r["name"], kind="speak", fn="interrupted", input=None,
                                   expected=r["got"], got=None, ok=False, verdict="interrupted before the bytes left; not re-spoken")
            out.append({"turn": r["name"], "reply_sha256": r["got"], "ledger": h})
            self._spoken.add(r["name"])                   # closed: a replay never speaks it either
        self.interrupted.extend(out)
        return out

    def _preflight_answer(self, ask: str) -> dict:
        rec = self.brain.turn(ask)
        return {"reply": rec.get("reply"), "spoken": self._spoken_rec(rec)}

    @staticmethod
    def _spoken_rec(rec: dict) -> bool:
        return bool(rec.get("reply")) and not rec.get("dont_know", False) and rec.get("kind") != "refused"

    # ── the turn ──────────────────────────────────────────────────────────
    def hormones(self) -> dict:
        b = self.brain
        if hasattr(b, "hormones"):
            return dict(b.hormones() or {})
        chat = getattr(b, "chat", None)
        return dict(getattr(chat, "state", None) or {})

    def frame_for(self, kind: str, goal: str, turn_id: str, plugin: str | None = None,
                  hormones: dict | None = None) -> TaskFrame:
        """The root frame of a turn: the kind's grant, NARROWED by the hormone rules in the config (a rule may
        drop a permission, never add one), the budget, the pinned generation, the postcondition."""
        base = set(GRANTS.get(kind, set()))
        if plugin:
            base.add(f"plugin:{plugin}")
        if kind != "night":
            base -= NEVER_LIVE
        h = dict(hormones if hormones is not None else self.hormones())
        allowed = set(base)
        for rule in (self.config.get("hormones") or {}).get("narrow", []):
            v = h.get(rule.get("axis"))
            if v is None:
                continue
            if ("above" in rule and float(v) > float(rule["above"])) or ("below" in rule and float(v) < float(rule["below"])):
                allowed -= set(rule.get("drop", []))
        assert allowed <= base, "hormones may narrow a grant, never widen it"
        return TaskFrame(turn_id, kind, goal, allowed=allowed, budget=Budget(**self.config.get("budget", {})),
                         generation=self.generation, hormones=h)

    def turn(self, user_text: str, feedback: str | None = None) -> dict:
        if not self.serving:
            raise RefuseToServe("not booted")
        with self._lock:
            self._n += 1
            turn_id = f"t{self._n:06d}"
        route = self.brain.route(user_text) if hasattr(self.brain, "route") else {"cortex": "talk"}
        kind = {"talk": "chat", "reasoning": "task", "memory": "learn", "help": "help"}.get(route.get("cortex"), route.get("cortex"))
        plugin = None if kind in GRANTS else route.get("cortex")
        frame = self.frame_for("plugin" if plugin else kind, user_text, turn_id, plugin=plugin)
        frame.route = route
        self.on_event({"kind": "frame", "turn": turn_id, "frame": frame.frame_id, "turn_kind": frame.kind,
                       "allowed": sorted(frame.allowed), "generation": frame.generation})
        t0 = time.perf_counter()
        _current.frame = frame
        try:
            try:
                rec = self.brain.turn(user_text, feedback, frame=frame) if _takes_frame(self.brain.turn) \
                    else self.brain.turn(user_text, feedback)
                frame.status = frame.judge(rec)
            except Exhausted as e:
                rec = {"reply": None, "kind": "exhausted", "reason": str(e)}
                frame.status = "exhausted"
            except Refused as e:
                rec = {"reply": None, "kind": "refused", "reason": e.reason}
                frame.status = "refused"
        finally:
            _current.frame = None
        frame.verdicts = list(rec.get("verdicts", [])) or ([rec["task"].get("verdict")] if isinstance(rec.get("task"), dict) and rec["task"].get("verdict") else [])
        reply = rec.get("reply") if frame.status in ("answered", "asked") or (frame.status == "refused" and rec.get("reply")) else None
        reply_hash = sha(reply or "")
        cfg = self.config
        row = {"turn": turn_id, "frame": frame.frame_id, "kind": frame.kind, "post": frame.post, "status": frame.status,
               "reply_sha256": reply_hash, "calls": list(frame.calls), "verdicts": frame.verdicts,
               "budget": frame.budget.as_dict(), "route": route, "wall_ms": round((time.perf_counter() - t0) * 1000),
               # everything that bends a turn (change 3): replay needs all of it
               "generation": frame.generation, "hormones": frame.hormones,
               "context_delta": rec.get("context_delta"),
               "model": {"adapters": {n: s.get("sha256") for n, s in (cfg.get("adapters") or {}).items()},
                         "emitter": (cfg.get("emitters") or {}).get("live", {}).get("sha256"),
                         "seed": rec.get("seed", cfg.get("decode_seed"))}}
        h = self.ledger.record(json.dumps(row, sort_keys=True, default=str), name=turn_id, kind="turn", fn=frame.kind,
                               input=user_text, expected=None, got=reply_hash, ok=frame.status == "answered",
                               verdict=frame.status)
        if rec.get("model_outputs") is not None:          # what the model said, verbatim, for audit replay (vaulted)
            self.ledger.record(f"model:{turn_id}", name=turn_id, kind="model", fn=frame.kind,
                               input=json.dumps(rec["model_outputs"], ensure_ascii=False, default=str),
                               expected=None, got=reply_hash, ok=True, verdict="recorded")
        rec = dict(rec, reply=reply, turn_id=turn_id, ledger=h, frame=frame.frame_id, status=frame.status,
                   generation=frame.generation, budget_bound=list(frame.budget.bound))
        self.turns[turn_id] = {"row": row, "hash": h, "user_text": user_text}
        self.on_event({"kind": "turn", "turn": turn_id, "status": frame.status, "ledger": h})
        rec["spoken"] = self._speak(turn_id, rec) if reply else False
        return rec

    def _speak(self, turn_id: str, rec: dict) -> bool:
        """Speak is an effect, ledger-first and at-most-once: the intent row is committed before the bytes leave,
        the spoken row after. A crash between the two leaves an intent `recover()` resolves; a second call for
        the same turn never speaks."""
        with self._lock:
            if turn_id in self._spoken:
                return False
            self._spoken.add(turn_id)
        reply_hash = sha(rec.get("reply") or "")
        self.ledger.record(f"speak:{turn_id}", name=turn_id, kind="speak", fn="intent", input=None,
                           expected=None, got=reply_hash, ok=True, verdict="intent")
        self.speaker(rec)                                 # the bytes leave here; if this raises, the intent stays orphan
        self.ledger.record(f"speak:{turn_id}", name=turn_id, kind="speak", fn="spoken", input=None,
                           expected=reply_hash, got=reply_hash, ok=True, verdict="spoken")
        return True


def _takes_frame(fn) -> bool:
    try:
        import inspect
        return "frame" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
