"""cubby_serve/harness_rules.py -- the harness's rules and records: the effects and grants, what is never live, the
default budget, the statuses and postconditions; hashing and signing a config; the budget, the task frame, a fact,
a tool, a refusal. Split out of harness.py (order 0031), verbatim; harness.py re-exports every name.

Wired: WIRED (the harness).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__wiring__ = "WIRED"


EFFECTS = ("read", "propose", "write", "external")


# what each turn kind may do; a tool whose `needs` is not a subset of the kind's grant is refused before it runs
GRANTS = {
    "fact":   {"read_history", "read_wiki", "read_store", "vm"},
    "task":   {"read_history", "read_wiki", "read_store", "vm", "fetch"},
    "chat":   {"vm"},                                   # no facts at stake: no store, no fetch; `vm` is the ASK -- the speech
    "help":   {"vm"},                                   # exit runs the certified think/act program on the VM (chat.mediate)
    "learn":  {"read_store", "vm", "write_store"},      # the memory cortex: a gated write
    "plugin": {"read_store", "vm"},                     # plus plugin:<name>, added per mounted plugin
    "night":  {"read_history", "read_wiki", "read_store", "vm", "write_store", "fetch", "forge", "promote", "teacher"},
}


NEVER_LIVE = {"forge", "promote", "teacher"}            # never granted in a live turn, whatever the kind


NO_EXTERNAL = {"fact", "chat", "help"}                  # an `external` effect (fetch) is off for these kinds, whatever the grant


DEFAULT_BUDGET = dict(tool_calls=8, reads=256, vm_runs=16, vm_hops=64, fetches=2, gen_tokens=2048,
                      forge_attempts=0, depth=3, wall_s=10.0)


# a fact's epistemic status; a counterfactual or hypothetical never shares a namespace with observed history
STATUSES = ("observed", "asserted", "hypothetical", "counterfactual", "derived")


BRANCH_ONLY = {"hypothetical", "counterfactual"}


BRANCH_PREFIX = "branch:"


# the postcondition a frame declares at creation, by kind -- typed by the host, never chosen after the result
POSTCONDITIONS = {"fact": "answer", "task": "answer", "learn": "learned", "chat": "phrase", "help": "phrase",
                  "plugin": "phrase", "night": "consolidated"}


# the defects a boot pack must carry (round 3, change 8): a contradicting context, an absent role, a verified
# program that answers the wrong role
REQUIRED_DEFECTS = ("contradicting_context", "absent_role", "wrong_role")


def _jsonable(o: Any):
    """A dataclass (a Fact) as its fields; anything else as text."""
    import dataclasses
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_jsonable).encode("utf-8")).hexdigest()


def file_sha(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _sign(key: bytes, body: dict) -> str:
    return hmac.new(key, sha(body).encode("ascii"), hashlib.sha256).hexdigest()


def sign_config(config: dict, key: bytes) -> dict:
    """The OWNER-signed config: thresholds, gate packs, adapter and emitter hashes. Calibration writes a new one
    offline; this process only ever reads it."""
    body = {k: v for k, v in config.items() if k != "signature"}
    return dict(body, signature=_sign(key, body))


# ── budgets and frames ──────────────────────────────────────────────────────
class Exhausted(Exception):
    """A budget line ran out. A declared outcome of the frame, never a silent truncation."""


@dataclass
class Budget:
    tool_calls: int = DEFAULT_BUDGET["tool_calls"]      # propose / write / external calls
    reads: int = DEFAULT_BUDGET["reads"]                # store reads (a walk reads many times; its own line)
    vm_runs: int = DEFAULT_BUDGET["vm_runs"]            # programs run
    vm_hops: int = DEFAULT_BUDGET["vm_hops"]            # jumps inside the VM: the program's `max_jumps`
    fetches: int = DEFAULT_BUDGET["fetches"]            # 2: refuse -> fetch -> a second walk is two by design
    gen_tokens: int = DEFAULT_BUDGET["gen_tokens"]      # generated tokens: at ~40 tok/s the binding line
    forge_attempts: int = DEFAULT_BUDGET["forge_attempts"]
    depth: int = DEFAULT_BUDGET["depth"]
    wall_s: float = DEFAULT_BUDGET["wall_s"]
    spent: dict = field(default_factory=dict)
    bound: list = field(default_factory=list)           # the lines that ran out, in order
    t0: float = field(default_factory=time.monotonic)

    LINES = ("tool_calls", "reads", "vm_runs", "vm_hops", "fetches", "gen_tokens", "forge_attempts")

    def remaining(self, line: str) -> int:
        return int(getattr(self, line)) - int(self.spent.get(line, 0))

    def remaining_wall_ms(self) -> int:
        return max(0, int((self.wall_s - (time.monotonic() - self.t0)) * 1000))

    def spend(self, line: str, n: int = 1) -> None:
        if time.monotonic() - self.t0 > self.wall_s:
            self._bind("wall_s")
        self.spent[line] = self.spent.get(line, 0) + n
        if self.spent[line] > getattr(self, line):
            self._bind(line)

    def _bind(self, line: str) -> None:
        if line not in self.bound:
            self.bound.append(line)
        raise Exhausted(line)

    def vm(self, max_ops: int | None = None, max_queries: int | None = None) -> dict:
        """The VM's Budget for the next program, from what this frame has left: vm_hops -> max_jumps, the wall's
        remainder -> max_wall_ms, plus the process-wide op and query ceilings. The emitter's program never sets
        these (round 3, change 6); the field exists on the wire since VM step 0."""
        b = {"max_jumps": max(0, self.remaining("vm_hops")), "max_wall_ms": self.remaining_wall_ms()}
        if max_ops is not None:
            b["max_ops"] = int(max_ops)
        if max_queries is not None:
            b["max_queries"] = int(max_queries)
        return b

    def as_dict(self) -> dict:
        return {"limits": {k: getattr(self, k) for k in DEFAULT_BUDGET}, "spent": dict(self.spent),
                "bound": list(self.bound), "elapsed_s": round(time.monotonic() - self.t0, 3)}


@dataclass
class TaskFrame:
    """One goal with its permissions, its budget and its postcondition, declared at creation. A multi-step task
    is a TREE of these under one root (a child's grant is the intersection of its parent's and its kind's)."""
    turn_id: str
    kind: str
    goal: str
    allowed: set = field(default_factory=set)
    budget: Budget = field(default_factory=Budget)
    parent: "TaskFrame | None" = None
    post: str = ""                                       # answer | learned | phrase | consolidated
    generation: int | None = None                        # the store generation pinned for this frame's lifetime
    worlds: set | None = None                            # the namespaces this frame may write; None = any mounted
    hormones: dict = field(default_factory=dict)         # the state at frame start, as the row records it
    route: dict | None = None                            # the harness's route, handed to the brain so it is not re-read
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
        self.post = self.post or POSTCONDITIONS.get(self.kind, "phrase")
        self.frame_id = self.frame_id or f"{self.turn_id}/{depth}/{len(self.parent.calls) if self.parent else 0}"

    def child(self, kind: str, goal: str) -> "TaskFrame":
        """A sub-goal: permissions = parent's ∩ kind's (never wider than either), the budget SHARED (the root
        bounds the whole task), the generation and the namespaces inherited."""
        allowed = set(self.allowed) & set(GRANTS.get(kind, set()))
        if self.kind != "night":                         # a live frame's children are live, whatever they are called
            allowed -= NEVER_LIVE
        return TaskFrame(self.turn_id, kind, goal, allowed=allowed, budget=self.budget, parent=self,
                         generation=self.generation, worlds=self.worlds, hormones=self.hormones)

    def judge(self, rec: dict) -> str:
        """The frame's status from its declared postcondition and what came back -- decided by the host, by type,
        never by the model and never after looking for a friendlier reading."""
        spoken = bool(rec.get("reply")) and not rec.get("dont_know", False) and rec.get("kind") != "refused"
        if rec.get("kind") == "ask":
            return "asked"
        if self.post == "answer":
            return "answered" if spoken and (rec.get("verdicts") or self._task_verdict(rec)) else ("refused" if not spoken else "failed")
        if self.post == "learned":
            learn = rec.get("learn") or {}
            return "answered" if learn.get("accepted") or rec.get("learned") else "refused"
        if self.post == "consolidated":
            return "answered" if rec.get("consolidated") is not None else "failed"
        return "answered" if spoken else "refused"                 # phrase: the model spoke, or declined to

    @staticmethod
    def _task_verdict(rec: dict) -> bool:
        t = rec.get("task")
        return isinstance(t, dict) and bool(t.get("speak_ok") or t.get("verdict"))


@dataclass(frozen=True)
class Fact:
    """A fact as the store keeps it: its world (namespace), epistemic status, valid time and provenance."""
    text: str
    world: str
    status: str = "asserted"
    valid_time: dict | None = None                       # {start, end, point}
    provenance: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"fact status must be one of {STATUSES}, not {self.status!r}")


# ── tools ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Tool:
    name: str
    effect: str                                          # read | propose | write | external
    needs: frozenset                                     # permissions a call requires
    fn: Callable[..., Any]
    doc: str = ""
    spends: str = "tool_calls"                           # the budget line a call spends (reads, vm_runs, fetches, ...)
    bind: Callable[["TaskFrame", dict], dict] | None = None   # the arguments the HARNESS sets from the frame (a program's
                                                         # budget and knowledge path); a caller's values for them are dropped

    def __post_init__(self):
        if self.effect not in EFFECTS:
            raise ValueError(f"tool {self.name}: effect must be one of {EFFECTS}, not {self.effect!r}")


class Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
