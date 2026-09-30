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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

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


class ToolRegistry:
    """The one door. Deny by default; effect classes and budgets enforced at call time; every call ledgered.
    A `propose` tool's result goes to QUARANTINE and comes back as {proposal, value}; only `promote` (a gated
    write, night only) moves it into a store. Native VM modules reachable through `use` are declared here too,
    with their own effect classes (`declare_vm_modules`)."""

    def __init__(self, ledger, on_event: Callable[[dict], None] | None = None):
        self.tools: dict[str, Tool] = {}
        self.ledger = ledger
        self.on_event = on_event or (lambda ev: None)
        self.quarantine: dict[str, dict] = {}
        self.vm_modules: dict[str, str] = {}             # module -> effect, for programs that `use` them
        self._n_proposals = 0

    def declare(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name} declared twice")
        self.tools[tool.name] = tool
        return tool

    def declare_vm_modules(self, modules: dict[str, str]) -> None:
        """{module: effect}: what a program's `use <module>` costs. A write module is refused in a turn without
        write_store; a `use` of an undeclared module is refused (deny by default holds inside the VM too)."""
        for m, eff in modules.items():
            if eff not in EFFECTS:
                raise ValueError(f"vm module {m}: effect must be one of {EFFECTS}, not {eff!r}")
        self.vm_modules.update(modules)

    def check_uses(self, frame: TaskFrame, uses: list[str]) -> None:
        for m in uses or []:
            eff = self.vm_modules.get(m)
            if eff is None:
                raise Refused(f"use {m}: undeclared VM module")
            if eff == "write" and "write_store" not in frame.allowed:
                raise Refused(f"use {m}: a write module without write_store")
            if eff == "external" and frame.kind in NO_EXTERNAL:
                raise Refused(f"use {m}: external effect off for {frame.kind}")

    def call(self, frame: TaskFrame, name: str, **args) -> Any:
        tool = self.tools.get(name)
        harness_set: list[str] = []
        try:
            if tool is None:
                raise Refused("no such tool")
            if tool.needs & NEVER_LIVE and frame.kind != "night":
                raise Refused("never in a live turn")
            missing = set(tool.needs) - set(frame.allowed)
            if missing:
                raise Refused(f"needs {sorted(missing)}")
            if tool.effect == "write" and "write_store" not in frame.allowed:
                raise Refused("a write without write_store")
            if tool.effect == "external" and frame.kind in NO_EXTERNAL:
                raise Refused(f"external effect off for {frame.kind}")
            if tool.effect == "write":
                self._check_namespace(frame, args)
            if args.get("uses"):
                self.check_uses(frame, list(args["uses"]))
            if tool.bind is not None:                    # the harness sets these; the caller's values never reach the tool
                bound = tool.bind(frame, args)
                harness_set = sorted(bound)
                args = {**{k: v for k, v in args.items() if k not in bound}, **bound}
            frame.budget.spend(tool.spends)
        except (Refused, Exhausted) as e:
            reason = e.reason if isinstance(e, Refused) else f"budget: {e}"
            h = self._row(frame, tool, name, args, ok=False, got=None, verdict=f"refused: {reason}", harness_set=harness_set)
            frame.calls.append(h)
            self.on_event({"kind": "tool", "tool": name, "ok": False, "reason": reason, "frame": frame.frame_id})
            raise
        t0 = time.perf_counter()
        try:
            got = tool.fn(**args)
        except Refused as e:                             # the tool refused on its own rule (promote's, a store's): a refusal row
            h = self._row(frame, tool, name, args, ok=False, got=None, verdict=f"refused: {e.reason}", harness_set=harness_set)
            frame.calls.append(h)
            raise
        except Exception as e:                           # a tool's own failure is a row too, then it propagates
            self._row(frame, tool, name, args, ok=False, got=None, verdict=f"error: {type(e).__name__}: {e}"[:300], harness_set=harness_set)
            raise
        empty = got is None or got == [] or got == {} or got == () or (
            isinstance(got, dict) and "result" in got and got["result"] in (None, [], {}))   # a VM result with nothing bound
        if tool.effect == "propose":                     # quarantined: a proposal is not a fact until promoted
            self._n_proposals += 1
            pid = f"p{self._n_proposals:06d}"
            self.quarantine[pid] = {"id": pid, "value": got, "tool": name, "frame": frame.frame_id,
                                    "turn": frame.turn_id, "promoted": None}
            got = {"proposal": pid, "value": got}
        # an empty result is a RETURN, not a success of the frame: the frame's postcondition decides that
        h = self._row(frame, tool, name, args, ok=True, got=got,
                      verdict=f"ok {round((time.perf_counter() - t0) * 1000)}ms" + (" empty" if empty else ""), harness_set=harness_set)
        frame.calls.append(h)
        self.on_event({"kind": "tool", "tool": name, "ok": True, "effect": tool.effect, "frame": frame.frame_id, "empty": empty})
        return got

    @staticmethod
    def _check_namespace(frame: TaskFrame, args: dict) -> None:
        """Worlds are namespaces: a fact lands in the world it names, a counterfactual or hypothetical only in a
        branch world, and only in a world this frame may write."""
        for v in args.values():
            if not isinstance(v, Fact):
                continue
            if v.status in BRANCH_ONLY and not v.world.startswith(BRANCH_PREFIX):
                raise Refused(f"a {v.status} fact never shares a namespace with observed history ({v.world})")
            if frame.worlds is not None and v.world not in frame.worlds:
                raise Refused(f"world {v.world} is not this frame's to write")

    def _row(self, frame, tool, name, args, ok, got, verdict, harness_set=()) -> str:
        # `program` is what the ledger hashes and keeps in the clear: never the arguments themselves (they may hold a
        # person's words) -- their hash, and the frame. The arguments go through `input`, which the ledger vaults.
        program = json.dumps({"tool": name, "effect": getattr(tool, "effect", None), "args_sha256": sha(args),
                              "frame": frame.frame_id, "turn": frame.turn_id, "harness_set": list(harness_set)}, sort_keys=True)
        return self.ledger.record(program, name=name, kind="tool", fn=frame.kind,
                                  input=json.dumps(args, default=_jsonable, ensure_ascii=False),
                                  expected=None, got=None if got is None else sha(got), ok=ok, verdict=verdict)

    # -- quarantine -> store: the one promote path ----------------------------------------------------
    def promote_tool(self, writer: Callable[[Fact], Any]) -> Tool:
        """`promote(proposal, world, status)`: a gated write (night only) that moves a quarantined proposal into a
        store through `writer` -- the ONE way a proposal becomes a fact."""
        def promote(proposal: str, world: str, status: str = "derived", provenance: str | None = None):
            q = self.quarantine.get(proposal)
            if q is None:
                raise Refused(f"no such proposal {proposal}")
            if q["promoted"]:
                raise Refused(f"proposal {proposal} already promoted")
            text = q["value"] if isinstance(q["value"], str) else json.dumps(q["value"], sort_keys=True, default=str)
            fact = Fact(text, world, status=status, provenance=provenance or f"{q['tool']}@{q['turn']}")
            if fact.status in BRANCH_ONLY and not world.startswith(BRANCH_PREFIX):
                raise Refused(f"a {status} proposal never lands in observed history ({world})")
            got = writer(fact)
            q["promoted"] = world
            return got
        return self.declare(Tool("promote", "write", frozenset({"write_store", "promote"}), fn=promote,
                                 doc="quarantine -> store, night only"))


# ── preflight ───────────────────────────────────────────────────────────────
class Preflight:
    """The gate packs: asks with expected outcomes and pre-registered bars, rotating (round 3, change 8) --
    `boot` (small, blocking: runs at boot and on every new generation), `regression` (larger, offline),
    `sentinel` (held out, NEVER harvested: `sentinel_asks` is what the harvester excludes). The boot pack must
    carry the real defects (`REQUIRED_DEFECTS`): a family declares which with `defect`. Bars are read from the
    owner-signed config; nothing here writes one."""

    def __init__(self, packs: dict):
        if "families" in packs and "boot" not in packs:  # v0's single-pack form
            packs = {"boot": packs}
        self.packs = packs

    def families(self, pack: str = "boot") -> dict:
        return (self.packs.get(pack) or {}).get("families", {})

    def missing_defects(self, pack: str = "boot") -> list[str]:
        have = {spec.get("defect") for spec in self.families(pack).values()}
        return [d for d in REQUIRED_DEFECTS if d not in have]

    def sentinel_asks(self) -> set[str]:
        return {it["ask"] for spec in self.families("sentinel").values() for it in spec.get("items", [])}

    def exclude_from_harvest(self, records: list[dict], key: str = "question") -> list[dict]:
        """A harvest without the sentinels: what the sleep cycle may learn from."""
        held = self.sentinel_asks()
        return [r for r in records if r.get(key) not in held]

    def run(self, answer: Callable[[str], dict], pack: str = "boot") -> dict:
        out = {"pack": pack, "families": {}, "passed": True, "failed": [], "spoken_values": {}}
        for fam, spec in self.families(pack).items():
            n = ok = 0
            for item in spec["items"]:
                n += 1
                rec = answer(item["ask"])
                ok += int(self._match(item, rec))
                if rec.get("spoken"):                    # what was SAID, for the hormone-extremes gate
                    out["spoken_values"][item["ask"]] = str(rec.get("reply", ""))
            rate = ok / max(1, n)
            out["families"][fam] = {"n": n, "ok": ok, "rate": rate, "bar": spec["bar"], "defect": spec.get("defect")}
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


# ── store generations ───────────────────────────────────────────────────────
class Generations:
    """Store generations on disk: `<root>/gen_<n>/...` and `<root>/manifest.json`, the manifest signed by the
    HARNESS key after the gate pack passed on that generation. The night writes g+1 beside g and never touches
    g; the swap is one atomic rename; a turn pins the generation it started on. Two keys: the owner's never
    signs a manifest, the harness's never signs a config."""

    def __init__(self, root: str | os.PathLike, harness_key: bytes):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = harness_key

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.root / "manifest.json"

    def dir(self, n: int) -> pathlib.Path:
        return self.root / f"gen_{n:04d}"

    def current(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def verify(self, manifest: dict) -> tuple[bool, str]:
        body = {k: v for k, v in manifest.items() if k != "signature"}
        if not hmac.compare_digest(manifest.get("signature", ""), _sign(self.key, body)):
            return False, "manifest signature"
        for name, spec in manifest.get("stores", {}).items():
            p = self.root / spec["path"]
            if not p.exists() or file_sha(p) != spec["sha256"]:
                return False, f"store.{name}"
        return True, "ok"

    def paths(self, manifest: dict) -> dict[str, str]:
        return {name: str(self.root / spec["path"]) for name, spec in manifest.get("stores", {}).items()}

    def next(self) -> tuple[int, pathlib.Path]:
        cur = self.current()
        n = (cur["generation"] + 1) if cur else 1
        while self.dir(n).exists():
            n += 1
        d = self.dir(n)
        d.mkdir(parents=True)
        return n, d

    def promote(self, n: int, stores: dict[str, str], config_sha256: str, gates: dict) -> dict:
        """Sign generation `n`'s manifest and swap it in atomically (write beside, then rename)."""
        body = {"generation": n, "stores": {}, "config_sha256": config_sha256, "gates": gates,
                "signed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for name, p in stores.items():
            p = pathlib.Path(p)
            body["stores"][name] = {"path": str(p.relative_to(self.root)).replace(os.sep, "/"), "sha256": file_sha(p)}
        manifest = dict(body, signature=_sign(self.key, body))
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.manifest_path)              # atomic on the same filesystem
        return manifest

    def seed(self, stores: dict[str, str], config_sha256: str = "") -> dict:
        """Generation 1 from existing store files (copied in): how a deployment starts."""
        import shutil
        n, d = self.next()
        placed = {}
        for name, p in stores.items():
            dst = d / pathlib.Path(p).name
            shutil.copy2(p, dst)
            placed[name] = str(dst)
        return self.promote(n, placed, config_sha256, {"seeded": True})


# ── the harness ─────────────────────────────────────────────────────────────
class RefuseToServe(Exception):
    pass


_current = threading.local()                             # the frame of the turn running on this thread, if any


def current_frame() -> TaskFrame | None:
    return getattr(_current, "frame", None)


class Harness:
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

    # ── the night ─────────────────────────────────────────────────────────
    def night(self) -> dict:
        """The sleep cycle writes generation g+1 BESIDE g (`night_fn(dst_dir, generation) -> {"stores": {name:
        path}, ...}`); the boot pack runs against g+1 through a view of the brain; on a pass the manifest is
        signed with the harness key and swapped atomically and the brain switches; on a fail g+1 stays on disk as
        rejected and g keeps serving. Turns in flight keep the generation they pinned. Without `generations`
        (v0's shape) the night runs in place and is ledgered, nothing more."""
        if self.night_fn is None:
            return {"skipped": "no night_fn"}
        self.ledger.record("night", name="night", kind="night", fn="begin", input=None, expected=None, got=None,
                           ok=True, verdict=f"from generation {self.generation}")
        if self.generations is None:
            rep = self.night_fn()
            self.ledger.record(json.dumps({"night": rep}, sort_keys=True, default=str), name="night", kind="night",
                               fn="end", input=None, expected=None, got=None, ok=True, verdict="in place")
            return rep
        n, dst = self.generations.next()
        rep = dict(self.night_fn(dst, n) or {})
        stores = dict(rep.get("stores") or {})
        # the gate pack against the candidate, through a view that leaves the serving stores alone
        view = self.brain.view(stores) if hasattr(self.brain, "view") else self.brain
        gate = None
        if self.preflight is not None:
            def answer(ask: str) -> dict:
                r = view.turn(ask)
                return {"reply": r.get("reply"), "spoken": self._spoken_rec(r)}
            gate = self.preflight.run(answer, "boot")
        passed = gate is None or gate["passed"]
        if passed:
            man = self.generations.promote(n, stores, sha({k: v for k, v in self.config.items() if k != "signature"}),
                                           {k: v for k, v in (gate or {}).items() if k != "spoken_values"})
            if hasattr(self.brain, "use_generation"):
                self.brain.use_generation(self.generations.paths(man))
            self.generation, self.stores = n, self.generations.paths(man)
            rep.update({"generation": n, "promoted": True})
            verdict = f"generation {n} promoted"
        else:
            rep.update({"generation": n, "promoted": False, "failed": gate["failed"]})
            verdict = f"generation {n} REJECTED: {gate['failed']}; {self.generation} keeps serving"
        self.ledger.record(json.dumps({"night": rep}, sort_keys=True, default=str), name="night", kind="night",
                           fn="end", input=None, expected="promote", got="promote" if passed else "reject",
                           ok=passed, verdict=verdict)
        self.on_event({"kind": "night", "generation": n, "promoted": passed})
        return rep

    # ── replay ────────────────────────────────────────────────────────────
    def replay(self, turn_id: str, mode: str = "row") -> dict:
        """`row`: the turn's row from the ledger, its tool rows, and whether the row still verifies. `audit`: the
        recorded model outputs through the deterministic host (`brain.replay`) must reproduce the reply hash --
        the 100% gate. `regen`: the brain decodes again -- a property of the engine, reported apart."""
        t = self.turns.get(turn_id)
        if t is None:
            return {"turn": turn_id, "found": False}
        d = self.ledger.get(t["hash"])
        ok, why = self.ledger.verify(t["hash"], require_current_vm=False)
        calls = [self.ledger.get(h) for h in t["row"]["calls"]]
        out = {"turn": turn_id, "found": True, "verified": ok, "why": why, "row": json.loads(d["program"]),
               "calls": [{"name": c["name"], "ok": c["ok"], "verdict": c["verdict"]} for c in calls if c]}
        want = t["row"]["reply_sha256"]
        if mode == "audit":
            rows = self.ledger.rows("model", name=turn_id)
            if not rows or not hasattr(self.brain, "replay"):
                out.update(audit=None, audit_why="no recorded model outputs" if not rows else "brain has no replay()")
                return out
            outputs = json.loads(self.ledger.get(rows[-1]["hash"])["input"])
            rec = self.brain.replay(t["user_text"], outputs, hormones=t["row"].get("hormones"),
                                    generation=t["row"].get("generation"))
            out.update(audit=sha(rec.get("reply") or "") == want, audit_reply_sha256=sha(rec.get("reply") or ""))
        elif mode == "regen":
            rec = self.brain.turn(t["user_text"])
            out.update(regen=sha(rec.get("reply") or "") == want)
        return out

    # ── gates the harness can run on itself ───────────────────────────────
    def hormone_gate(self, extremes: list[dict]) -> dict:
        """The gate pack at hormone extremes: the same SPOKEN values everywhere (a refusal may change, a value
        may not), and no extreme widens a grant. Needs `brain.set_hormones`."""
        if self.preflight is None or not hasattr(self.brain, "set_hormones"):
            return {"skipped": "no preflight or the brain has no set_hormones"}
        base = self.hormones()
        values, grants = {}, {}
        try:
            for h in [base] + list(extremes):
                self.brain.set_hormones(h)
                rep = self.preflight.run(self._preflight_answer, "boot")
                values[sha(h)] = rep["spoken_values"]
                grants[sha(h)] = {k: sorted(self.frame_for(k, "probe", "gate", hormones=h).allowed) for k in GRANTS}
        finally:
            self.brain.set_hormones(base)
        keys = list(values)
        asks = set().union(*(set(v) for v in values.values()))
        same = all(len({values[k][a] for k in keys if a in values[k]}) <= 1 for a in asks)
        never_wider = all(set(grants[k][kind]) <= set(grants[keys[0]][kind]) for k in keys for kind in GRANTS)
        return {"same_spoken_values": same, "never_wider": never_wider, "n_extremes": len(extremes), "asks": len(asks)}

    def deny_probe(self, probes: list[tuple[str, str, dict]]) -> dict:
        """Gate 5: (kind, tool, args) calls a turn kind must not make -- 0 executed, every one ledgered."""
        before = self.ledger.count(ok=False)
        executed = 0
        for i, (kind, tool, args) in enumerate(probes):
            f = self.frame_for(kind, "probe", f"probe{i:05d}")
            try:
                self.registry.call(f, tool, **args)
                executed += 1
            except (Refused, Exhausted):
                pass
        return {"n": len(probes), "executed": executed, "ledgered": self.ledger.count(ok=False) - before}


def _takes_frame(fn) -> bool:
    try:
        import inspect
        return "frame" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
