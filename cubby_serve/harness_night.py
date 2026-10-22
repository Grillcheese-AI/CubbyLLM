"""cubby_serve/harness_night.py -- the harness at night and under test: the night's run, a replay of a day, the hormone
gate, the deny probe. Split out of harness.py (order 0031) as a mixin of Harness, verbatim.

Wired: WIRED (Harness).
"""
from __future__ import annotations

import json
from .harness_rules import Exhausted, GRANTS, Refused, sha

__wiring__ = "WIRED"


class HarnessNight:
    """Harness's night, replay and probes (a mixin; order 0031)."""

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
