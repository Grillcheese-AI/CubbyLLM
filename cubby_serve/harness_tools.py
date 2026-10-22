"""cubby_serve/harness_tools.py -- the harness's tool registry (declared tools, their namespaces and uses, a call
under the frame's budget, a tool promoted), the preflight, and the generations of the config. Split out of harness.py
(order 0031), verbatim; harness.py re-exports every name.

Wired: WIRED (the harness).
"""
from __future__ import annotations

import hmac
import json
import os
import pathlib
import time
from typing import Any, Callable
from .harness_rules import (BRANCH_ONLY, BRANCH_PREFIX, EFFECTS, Exhausted, Fact, NEVER_LIVE, NO_EXTERNAL,
                            REQUIRED_DEFECTS, Refused, TaskFrame, Tool, _jsonable, _sign, file_sha, sha)

__wiring__ = "WIRED"


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
