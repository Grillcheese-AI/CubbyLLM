"""harness -- one process that owns boot, state, the turn and the night (v0, 2026-09-28).

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this). Spec: docs/research/2026-09-28-harness-v0.md;
pre-registered as H-E14. Pinned by standin/tests/test_harness.py with fakes: no model, no VM, no GPU.

What it adds: nothing. What it makes one system: the parts that exist.

    boot     signed config -> stores' hashes checked -> VM self-test -> adapters' hashes checked -> preflight
             (a gate pack with bars) -> serve, or REFUSE, with a ledger row either way. Thresholds come from the
             config; the preflight never moves one (a boot that could lower a threshold to pass its own test is a hole).
    turn     a TaskFrame with a Budget; every tool call through ONE registry (deny by default, effect classes
             enforced at call time, budgets spent, every call ledgered); the brain's own turn does the work;
             the reply's hash goes to the ledger with the frame and its verdicts.
    night    the sleep cycle, run under the STORE LOCK; a turn that arrives while it is held is refused, not
             served from a store mid-write.
    replay   any turn back from the ledger: the row, the calls, the reply hash to check against.

The model proposes and phrases. The harness never lets it decide what is true, whether a check passed, which
tool runs, what is written, whether content is allowed, or whether to speak after a failed check.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

EFFECTS = ("read", "propose", "write")
# what each turn kind may do; a tool whose `needs` is not a subset of the kind's grant is refused before it runs
GRANTS = {
    "fact":   {"read_history", "read_wiki", "read_store", "vm"},
    "task":   {"read_history", "read_wiki", "read_store", "vm", "fetch"},
    "chat":   set(),                                    # no facts at stake: no tool at all
    "help":   set(),
    "learn":  {"read_store", "vm", "write_store"},      # the memory cortex: a gated write
    "plugin": {"read_store", "vm"},                     # plus plugin:<name>, added per mounted plugin
    "night":  {"read_history", "read_wiki", "read_store", "vm", "write_store", "forge"},
}
NEVER_LIVE = {"forge"}                                  # never granted in a live turn, whatever the kind
DEFAULT_BUDGET = dict(tool_calls=8, vm_hops=64, fetches=1, forge_attempts=0, depth=3, wall_s=10.0)


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def file_sha(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ── budgets and frames ──────────────────────────────────────────────────────
class Exhausted(Exception):
    """A budget line ran out. A declared outcome of the frame, never a silent truncation."""


@dataclass
class Budget:
    tool_calls: int = DEFAULT_BUDGET["tool_calls"]
    vm_hops: int = DEFAULT_BUDGET["vm_hops"]
    fetches: int = DEFAULT_BUDGET["fetches"]
    forge_attempts: int = DEFAULT_BUDGET["forge_attempts"]
    depth: int = DEFAULT_BUDGET["depth"]
    wall_s: float = DEFAULT_BUDGET["wall_s"]
    spent: dict = field(default_factory=lambda: {"tool_calls": 0, "vm_hops": 0, "fetches": 0, "forge_attempts": 0})
    t0: float = field(default_factory=time.monotonic)

    def spend(self, line: str, n: int = 1) -> None:
        if time.monotonic() - self.t0 > self.wall_s:
            raise Exhausted("wall_s")
        self.spent[line] = self.spent.get(line, 0) + n
        if self.spent[line] > getattr(self, line):
            raise Exhausted(line)

    def as_dict(self) -> dict:
        return {"limits": {k: getattr(self, k) for k in DEFAULT_BUDGET}, "spent": dict(self.spent),
                "elapsed_s": round(time.monotonic() - self.t0, 3)}


@dataclass
class TaskFrame:
    """One goal with its permissions and budget. A multi-step task is a stack of these under one root."""
    turn_id: str
    kind: str
    goal: str
    allowed: set = field(default_factory=set)
    budget: Budget = field(default_factory=Budget)
    parent: "TaskFrame | None" = None
    status: str = "open"                                 # open | answered | asked | refused | exhausted | failed
    calls: list = field(default_factory=list)            # ledger hashes of this frame's tool calls
    verdicts: list = field(default_factory=list)
    frame_id: str = ""

    def __post_init__(self):
        depth = 0
        p = self.parent
        while p is not None:
            depth += 1
            p = p.parent
        if depth > self.budget.depth:
            raise Exhausted("depth")
        self.frame_id = self.frame_id or f"{self.turn_id}/{depth}/{len(self.parent.calls) if self.parent else 0}"

    def child(self, kind: str, goal: str) -> "TaskFrame":
        # a sub-goal inherits its parent's grant and SHARES its budget: the root bounds the whole task
        return TaskFrame(self.turn_id, kind, goal, allowed=set(self.allowed), budget=self.budget, parent=self)


# ── tools ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Tool:
    name: str
    effect: str                                          # read | propose | write
    needs: frozenset                                     # permissions a call requires
    fn: Callable[..., Any]
    doc: str = ""
    spends: str = "tool_calls"                           # the budget line a call spends besides tool_calls

    def __post_init__(self):
        if self.effect not in EFFECTS:
            raise ValueError(f"tool {self.name}: effect must be one of {EFFECTS}, not {self.effect!r}")


class Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ToolRegistry:
    """The one door. Deny by default; effect classes and budgets enforced at call time; every call ledgered."""

    def __init__(self, ledger, on_event: Callable[[dict], None] | None = None):
        self.tools: dict[str, Tool] = {}
        self.ledger = ledger
        self.on_event = on_event or (lambda ev: None)

    def declare(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name} declared twice")
        self.tools[tool.name] = tool
        return tool

    def call(self, frame: TaskFrame, name: str, **args) -> Any:
        tool = self.tools.get(name)
        args_hash = sha(args)
        try:
            if tool is None:
                raise Refused("no such tool")
            missing = set(tool.needs) - set(frame.allowed)
            if missing:
                raise Refused(f"needs {sorted(missing)}")
            if tool.needs & NEVER_LIVE and frame.kind != "night":
                raise Refused("never in a live turn")
            if tool.effect == "write" and "write_store" not in frame.allowed:
                raise Refused("a write without write_store")
            frame.budget.spend("tool_calls")
            if tool.spends != "tool_calls":
                frame.budget.spend(tool.spends)
        except (Refused, Exhausted) as e:
            reason = e.reason if isinstance(e, Refused) else f"budget: {e}"
            h = self._row(frame, tool, name, args_hash, args, ok=False, got=None, verdict=f"refused: {reason}")
            frame.calls.append(h)
            self.on_event({"kind": "tool", "tool": name, "ok": False, "reason": reason, "frame": frame.frame_id})
            raise
        t0 = time.perf_counter()
        try:
            got = tool.fn(**args)
        except Exception as e:                           # a tool's own failure is a row too, then it propagates
            self._row(frame, tool, name, args_hash, args, ok=False, got=None, verdict=f"error: {type(e).__name__}: {e}"[:300])
            raise
        h = self._row(frame, tool, name, args_hash, args, ok=True, got=got,
                      verdict=f"ok {round((time.perf_counter() - t0) * 1000)}ms")
        frame.calls.append(h)
        self.on_event({"kind": "tool", "tool": name, "ok": True, "effect": tool.effect, "frame": frame.frame_id})
        return got

    def _row(self, frame, tool, name, args_hash, args, ok, got, verdict) -> str:
        # `program` is what the ledger hashes and keeps in the clear: never the arguments themselves (they may hold a
        # person's words) -- their hash, and the frame. The arguments go through `input`, which the ledger vaults.
        program = json.dumps({"tool": name, "effect": getattr(tool, "effect", None), "args_sha256": args_hash,
                              "frame": frame.frame_id, "turn": frame.turn_id}, sort_keys=True)
        return self.ledger.record(program, name=name, kind="tool", fn=frame.kind,
                                  input=json.dumps(args, default=str, ensure_ascii=False),
                                  expected=None, got=None if got is None else sha(got), ok=ok, verdict=verdict)


# ── preflight ───────────────────────────────────────────────────────────────
class Preflight:
    """A gate pack: asks with expected outcomes and pre-registered bars. Pass, or the harness refuses to serve.
    It reads thresholds from the config and never writes one."""

    def __init__(self, pack: dict):
        self.pack = pack

    def run(self, answer: Callable[[str], dict]) -> dict:
        out = {"families": {}, "passed": True, "failed": []}
        for fam, spec in self.pack["families"].items():
            n = ok = 0
            for item in spec["items"]:
                n += 1
                rec = answer(item["ask"])
                ok += int(self._match(item, rec))
            rate = ok / max(1, n)
            out["families"][fam] = {"n": n, "ok": ok, "rate": rate, "bar": spec["bar"]}
            if rate < spec["bar"]:
                out["passed"] = False
                out["failed"].append(fam)
        return out

    @staticmethod
    def _match(item: dict, rec: dict) -> bool:
        want = item["expect"]
        if want == "spoken":
            return bool(rec.get("spoken")) and (item.get("value") is None or item["value"] in str(rec.get("reply", "")))
        if want == "refused":
            return not rec.get("spoken")
        if want == "not_wrong":                          # the sentinel: anything but a spoken wrong answer
            return not (rec.get("spoken") and item["wrong"] in str(rec.get("reply", "")))
        raise ValueError(f"unknown expectation {want!r}")


# ── the harness ─────────────────────────────────────────────────────────────
class RefuseToServe(Exception):
    pass


class Harness:
    """One process. `boot()` then `turn()`; `night()` under the store lock; `replay()` from the ledger."""

    def __init__(self, config: dict, ledger, brain, registry: ToolRegistry, preflight: Preflight | None = None,
                 vm_selftest: Callable[[], bool] | None = None, night_fn: Callable[[], dict] | None = None,
                 on_event: Callable[[dict], None] | None = None, config_key: bytes | None = None):
        self.config, self.ledger, self.brain, self.registry = config, ledger, brain, registry
        self.preflight, self.vm_selftest, self.night_fn = preflight, vm_selftest, night_fn
        self.on_event = on_event or (lambda ev: None)
        self.config_key = config_key
        self.store_lock = threading.RLock()
        self.serving = False
        self.turns: dict[str, dict] = {}
        self._n = 0

    # ── boot ──────────────────────────────────────────────────────────────
    def boot(self) -> dict:
        checks = []

        def check(name: str, ok: bool, detail: str = ""):
            checks.append({"check": name, "ok": bool(ok), "detail": detail})
            self.on_event({"kind": "boot", "check": name, "ok": bool(ok), "detail": detail})
            return ok

        cfg = self.config
        if self.config_key is not None:                  # the config is signed; a changed threshold fails here
            body = {k: v for k, v in cfg.items() if k != "signature"}
            want = hmac.new(self.config_key, sha(body).encode("ascii"), hashlib.sha256).hexdigest()
            check("config.signature", hmac.compare_digest(cfg.get("signature", ""), want))
        for name, spec in cfg.get("stores", {}).items():
            p = spec.get("path")
            ok = bool(p) and os.path.exists(p) and file_sha(p) == spec.get("sha256")
            check(f"store.{name}", ok, "hash differs or missing")
        check("vm.selftest", self.vm_selftest() if self.vm_selftest else True)
        for name, spec in cfg.get("adapters", {}).items():
            p = spec.get("path")
            ok = bool(p) and os.path.exists(p) and file_sha(p) == spec.get("sha256")
            check(f"adapter.{name}", ok, "hash differs or missing")
        try:
            self.ledger.record("boot", name="boot", kind="boot", fn="ledger.write", input=None,
                               expected=None, got=None, ok=True, verdict="probe")
            check("ledger.write", True)
        except Exception as e:
            check("ledger.write", False, str(e)[:200])
        if self.preflight is not None and all(c["ok"] for c in checks):
            rep = self.preflight.run(self._preflight_answer)
            check("preflight", rep["passed"], f"failed families: {rep['failed']}" if rep["failed"] else "")
            checks[-1]["report"] = rep["families"]
        passed = all(c["ok"] for c in checks)
        row = {"checks": checks, "config_sha256": sha({k: v for k, v in cfg.items() if k != "signature"})}
        self.ledger.record(json.dumps(row, sort_keys=True), name="boot", kind="boot", fn=None, input=None,
                           expected="serve", got="serve" if passed else "refuse", ok=passed,
                           verdict="serving" if passed else "REFUSED: " + ", ".join(c["check"] for c in checks if not c["ok"]))
        self.serving = passed
        if not passed:
            raise RefuseToServe(", ".join(c["check"] for c in checks if not c["ok"]))
        return row

    def _preflight_answer(self, ask: str) -> dict:
        rec = self.brain.turn(ask)
        return {"reply": rec.get("reply"), "spoken": self._spoken(rec)}

    @staticmethod
    def _spoken(rec: dict) -> bool:
        return bool(rec.get("reply")) and not rec.get("dont_know", False) and rec.get("kind") != "refused"

    # ── the turn ──────────────────────────────────────────────────────────
    def frame_for(self, kind: str, goal: str, turn_id: str, plugin: str | None = None) -> TaskFrame:
        allowed = set(GRANTS.get(kind, set()))
        if plugin:
            allowed.add(f"plugin:{plugin}")
        allowed -= NEVER_LIVE
        return TaskFrame(turn_id, kind, goal, allowed=allowed, budget=Budget(**self.config.get("budget", {})))

    def turn(self, user_text: str, feedback: str | None = None) -> dict:
        if not self.serving:
            raise RefuseToServe("not booted")
        if not self.store_lock.acquire(blocking=False):  # the night holds it: never serve from a store mid-write
            self.ledger.record("turn", name="turn", kind="turn", fn=None, input=user_text, expected=None, got=None,
                               ok=False, verdict="refused: the night holds the store lock")
            return {"reply": None, "kind": "refused", "reason": "consolidating; try again in a moment"}
        try:
            self._n += 1
            turn_id = f"t{self._n:06d}"
            route = self.brain.route(user_text) if hasattr(self.brain, "route") else {"cortex": "talk"}
            kind = {"talk": "chat", "reasoning": "task", "memory": "learn", "help": "help"}.get(route.get("cortex"), route.get("cortex"))
            plugin = None if kind in GRANTS else route.get("cortex")
            frame = self.frame_for("plugin" if plugin else kind, user_text, turn_id, plugin=plugin)
            self.on_event({"kind": "frame", "turn": turn_id, "frame": frame.frame_id, "turn_kind": frame.kind,
                           "allowed": sorted(frame.allowed)})
            t0 = time.perf_counter()
            try:
                rec = self.brain.turn(user_text, feedback, frame=frame) if _takes_frame(self.brain.turn) \
                    else self.brain.turn(user_text, feedback)
                frame.status = "answered" if self._spoken(rec) else ("asked" if rec.get("kind") == "ask" else "refused")
            except Exhausted as e:
                rec = {"reply": None, "kind": "exhausted", "reason": str(e)}
                frame.status = "exhausted"
            except Refused as e:
                rec = {"reply": None, "kind": "refused", "reason": e.reason}
                frame.status = "refused"
            frame.verdicts = list(rec.get("verdicts", [])) or ([rec["task"].get("verdict")] if isinstance(rec.get("task"), dict) and rec["task"].get("verdict") else [])
            reply_hash = sha(rec.get("reply") or "")
            row = {"turn": turn_id, "frame": frame.frame_id, "kind": frame.kind, "status": frame.status,
                   "reply_sha256": reply_hash, "calls": list(frame.calls), "verdicts": frame.verdicts,
                   "budget": frame.budget.as_dict(), "route": route, "wall_ms": round((time.perf_counter() - t0) * 1000)}
            h = self.ledger.record(json.dumps(row, sort_keys=True, default=str), name=turn_id, kind="turn", fn=frame.kind,
                                   input=user_text, expected=None, got=reply_hash, ok=frame.status == "answered",
                                   verdict=frame.status)
            rec = dict(rec, turn_id=turn_id, ledger=h, frame=frame.frame_id, status=frame.status)
            self.turns[turn_id] = {"row": row, "hash": h}
            self.on_event({"kind": "turn", "turn": turn_id, "status": frame.status, "ledger": h})
            return rec
        finally:
            self.store_lock.release()

    # ── the night ─────────────────────────────────────────────────────────
    def night(self) -> dict:
        if self.night_fn is None:
            return {"skipped": "no night_fn"}
        with self.store_lock:                            # every store write of the night happens under it
            self.ledger.record("night", name="night", kind="night", fn="begin", input=None, expected=None, got=None,
                               ok=True, verdict="lock held")
            rep = self.night_fn()
            self.ledger.record(json.dumps({"night": rep}, sort_keys=True, default=str), name="night", kind="night",
                               fn="end", input=None, expected=None, got=None, ok=True, verdict="lock released")
        return rep

    # ── replay ────────────────────────────────────────────────────────────
    def replay(self, turn_id: str) -> dict:
        """The turn's row from the ledger, its tool rows, and whether the row still verifies."""
        t = self.turns.get(turn_id)
        if t is None:
            return {"turn": turn_id, "found": False}
        d = self.ledger.get(t["hash"])
        ok, why = self.ledger.verify(t["hash"], require_current_vm=False)
        calls = [self.ledger.get(h) for h in t["row"]["calls"]]
        return {"turn": turn_id, "found": True, "verified": ok, "why": why, "row": json.loads(d["program"]),
                "calls": [{"name": c["name"], "ok": c["ok"], "verdict": c["verdict"]} for c in calls if c]}


def _takes_frame(fn) -> bool:
    try:
        import inspect
        return "frame" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def sign_config(config: dict, key: bytes) -> dict:
    """The signed config: thresholds, store and adapter hashes, the gate pack. Calibration writes a new one offline."""
    body = {k: v for k, v in config.items() if k != "signature"}
    return dict(body, signature=hmac.new(key, sha(body).encode("ascii"), hashlib.sha256).hexdigest())
