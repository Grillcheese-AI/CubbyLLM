"""Order 0067 (Cowork, 2026-11-13): the step 2 and step 3 sets rerun once on a new VM build, every change listed.

Each stored program is run on each of its stored replies with the build ``CUBELANG_EXE`` names now; its verdict is
compared with the one recorded when the set was built (the evaluator's, which the earlier build matched on every reply).
No name-key guard runs here: the build checks its own keys (cubelang P6-9).

  python validation/rerun_vm_sets.py [--out validation/logs/vm_sets_rerun_<date>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE / "builders")]
import view_compile as VC  # noqa: E402
import view_join as VJ  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402

__wiring__ = "STANDALONE"


def run(vm, src):
    try:
        res = vm.run(src, fn="solve").get("result")
    except Exception as e:                                # an error is a change, listed
        return f"error: {str(e)[:160]}"
    return None if res is None else str(res).strip() in ("1", "1.0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    vm, out = cc.CubelangSession(), {"order": "0067", "what": "steps 2-3 rerun on a new VM build", "vm": vm_identity.identity(), "sets": {}}
    sets = {"step2_view_programs": (ds / "view_programs_0067" / "rows.jsonl", lambda r, t: VC.program(r["spec"], t)),
            "step3_long_rows": (ds / "long_rows_0067" / "rows.jsonl", lambda r, t: VJ.program(r["join"], r["parts"], t))}
    for name, (path, make) in sets.items():
        t, changes = Counter(), []
        for r in map(json.loads, open(path, encoding="utf-8")):
            for rep in r["replies"]:
                got = run(vm, make(r, rep["text"]))
                t["runs"] += 1
                if got == rep["held"]:
                    t["same"] += 1
                    continue
                t["changed"] += 1
                if len(changes) < 50:
                    changes.append({"id": r["id"], "recorded": rep["held"], "now": got, "reply": rep["text"][:200]})
        out["sets"][name] = {"counts": dict(t), "changes": changes}
        print(name, dict(t), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
