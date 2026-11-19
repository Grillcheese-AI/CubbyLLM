"""cubelang P6-3 to P6-7's cross-check on order 0067's stored rows (Cowork, 2026-11-14): every row's join, run on its
stored replies, gives the verdict stored when the row was kept.

`long_rows_0067` (`build_long_rows.py`) kept a row only when the VM's verdict on its joined program agreed with the
evaluator's on every chosen reply; each reply carries that verdict (``held``). This reads the rows as written (refused
if ``rows.jsonl`` is not the manifest's SHA-256), rebuilds each reply's program from the stored join and parts
(`view_join.program`, as the builder ran it), runs it on the VM in ``$CUBELANG_EXE``, and counts the replies whose
verdict differs. The gold programs only: no reader's answers. Ids and counts only in the log.

  python validation/exp_p6_crosscheck_0067.py [--set $CUBBY_DATASETS/long_rows_0067] [--out validation/logs/<name>.json]

Wired: STANDALONE (a cross-check for the cubelang fix list).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation" / "builders")]
import view_join as VJ  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402

__wiring__ = "STANDALONE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="the folder of long_rows_0067 (default: $CUBBY_DATASETS/long_rows_0067)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    folder = Path(a.set or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067")
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((folder / "rows.jsonl").read_bytes()).hexdigest()
    if sha != man["rows_sha256"]:
        raise SystemExit(f"rows.jsonl is not the manifest's (sha256 {sha[:12]} against {man['rows_sha256'][:12]})")
    rows = [json.loads(line) for line in open(folder / "rows.jsonl", encoding="utf-8")]
    vm = cc.CubelangSession()
    runs, differ, by_split = 0, [], Counter()
    try:
        for r in rows:
            for i, rep in enumerate(r["replies"]):
                got = str(vm.run(VJ.program(r["join"], r["parts"], rep["text"]), fn="solve").get("result")).strip() in ("1", "1.0")
                runs += 1
                if got != rep["held"]:
                    differ.append({"id": r["id"], "reply": i, "stored": rep["held"], "vm": got})
                    by_split[r["split"]] += 1
    finally:
        vm.close()
    rep = {"what": "cubelang P6-3..7 cross-check: 0067's stored rows, each join on its stored replies, against the "
                   "verdict stored when the row was kept",
           "date": _dt.date.today().isoformat(), "vm": vm_identity.identity(),
           "set": {"rows": len(rows), "rows_sha256": sha, "stored_with_vm": man.get("vm", {}).get("commit"),
                   "stored_runs_agreeing": man.get("vm_runs_agreeing")},
           "runs": runs, "agree": runs - len(differ), "differ": len(differ), "differ_by_split": dict(by_split),
           "differences": differ[:200]}
    out = Path(a.out or ROOT / "validation" / "logs" / f"p6_crosscheck_0067_{rep['date']}.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("vm", "differences")}), "->", out)


if __name__ == "__main__":
    main()
