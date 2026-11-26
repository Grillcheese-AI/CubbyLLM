"""arith_records - order 0085's writer around a read (`exp_0078_dev_read.py --records <path>`): the freeze at the
read's start, one record per question (`cubbyllm.records.arith_trace.trace`, carrying the start's freeze), and the
freeze again at the end, written beside the records as `<path>.freeze.json` with what moved (a mismatch voids the
read: line 3). Records are JSON lines, kept off the repository (the report holds counts only).

Wiring: STANDALONE (order 0085).
"""
from __future__ import annotations

import json
import os
import time

from arith_freeze import differences, freeze

__wiring__ = "STANDALONE"


def vm_exe() -> "str | None":
    exe = os.environ.get("CUBELANG_EXE")
    if not exe:
        from cubbyllm import local_paths
        exe = local_paths.value("CUBELANG_EXE")
    return exe


def run_info(name: str, start: dict, calibration: str = "", calibration_at: str = "") -> dict:
    """The CheezML writer's run (`cheez_arith.write`): ids derived from the run's name and the VM's commit (no machine
    name), the time, and the calibration that set m (the dev rows it was chosen on, by hash, and when)."""
    import uuid
    from arith_freeze import sha256_file, _source_sha, ROOT
    ns = uuid.UUID("0c5e4a1e-85a7-4c1e-9b0f-0085a7c0de85")
    commit = (start.get("vm") or {}).get("commit") or ""
    cal = None
    if calibration:
        screen = "".join(_source_sha(os.path.join(ROOT, p)) for p in
                         ("cubbyllm/reasoning/speak_vote.py", "validation/exp_0078_dev_score.py"))
        import hashlib
        cal = {"data sha256": sha256_file(calibration), "screen sha256": hashlib.sha256(screen.encode()).hexdigest(),
               "at": calibration_at}
    return {"run": name, "host": f"urn:uuid:{uuid.uuid5(ns, 'host|' + name)}", "vm": f"urn:uuid:{uuid.uuid5(ns, 'vm|' + commit)}",
            "vm version": commit or None, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "deployment": f"urn:uuid:{uuid.uuid5(ns, 'deployment|' + name)}", "calibration": cal}


class Writer:
    def __init__(self, path: str, adapter: str, settings: dict, rules=(), exe: "str | None" = None,
                 name: str = "read", calibration: str = "", calibration_at: str = ""):
        self.path, self.adapter, self.settings, self.rules = path, adapter, settings, tuple(rules)
        self.exe = exe if exe is not None else vm_exe()
        self.start = freeze(adapter, settings, self.exe)
        self.run = run_info(name, self.start, calibration, calibration_at)
        self.started, self.n, self.seconds = time.strftime("%Y-%m-%dT%H:%M:%S"), 0, 0.0
        self.kinds: dict = {}
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rid: str, i: int, question: str, chains: list) -> float:
        """One question's record (the trace, and its CheezML record or why there is none); returns the seconds the
        writer took (the time it adds per question)."""
        from cubbyllm.records.arith_trace import trace
        from cubbyllm.records.cheez_arith import write as cheez
        t0 = time.perf_counter()
        rec = trace(question, chains, self.settings["k"], self.settings["m"], self.rules, self.start)
        graph, why = cheez(rec, {**self.run, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, rid) \
            if self.run["calibration"] or rec["kind"] == "refusal" else (None, "no calibration given (--calibration)")
        self.f.write(json.dumps({"id": rid, "i": i, **rec, "cheez": graph, "cheez why": why,
                                 "cheez terms": "v0.2" if rec["kind"] == "refusal" else "proposed (0085)"},
                                ensure_ascii=False) + "\n")
        self.f.flush()
        dt = time.perf_counter() - t0
        self.n, self.seconds = self.n + 1, self.seconds + dt
        self.kinds[rec["kind"]] = self.kinds.get(rec["kind"], 0) + 1
        return round(dt, 4)

    def close(self) -> dict:
        self.f.close()
        end = freeze(self.adapter, self.settings, self.exe)
        diff = differences(self.start, end)
        out = {"started": self.started, "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "start": self.start, "end": end,
               "differences": diff, "void": bool(diff), "records": self.n, "kinds": self.kinds,
               "writer seconds a question": round(self.seconds / max(1, self.n), 4)}
        with open(self.path + ".freeze.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"records: {self.n} {self.kinds} | freeze {'VOID: ' + ', '.join(diff) if diff else 'held'}", flush=True)
        return out


__all__ = ["Writer", "vm_exe"]
