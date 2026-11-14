"""Order 0073, step 1: the spec language, its evaluator, the compiler and the join moved under `cubbyllm/speech/`
with no change of behaviour. Shown on every stored row, with the core's code only:

- each long row and each pair row (`long_rows_0067/rows.jsonl`, `pair_rows.jsonl`): `view_join.join(parts, groups)`
  is the stored program byte for byte, and `view_spec.evaluate(view_join.spec(parts, groups), views(reply))` is the
  stored verdict of every reply;
- each program row (`view_programs_0067/rows.jsonl`): `view_compile.body(spec)` is the stored body byte for byte, and
  `evaluate(spec, views(reply))` the stored verdict of every reply.

A row that differs stops nothing here: it is counted and its id listed, and the build does not go on until the list is
empty. No row text is written.

  python validation/exp_view_core_rebuild.py [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

SETS = (("long rows", "long_rows_0067/rows.jsonl", "join"), ("pair rows", "long_rows_0067/pair_rows.jsonl", "join"),
        ("program rows", "view_programs_0067/rows.jsonl", "body"))


def rebuilt(row: dict, kind: str) -> tuple[str, dict]:
    """(the program text the core builds for `row`, the spec its replies are judged by)."""
    if kind == "join":
        return VJ.join(row["parts"], row["groups"]), VJ.spec(row["parts"], row["groups"])
    return VC.body(row["spec"]), row["spec"]


def check(path: Path, kind: str) -> dict:
    raw = path.read_bytes()
    t = {"rows": 0, "programs identical": 0, "replies": 0, "verdicts identical": 0, "rows that differ": []}
    for ln in raw.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        text, spec = rebuilt(row, kind)
        same = text == row[kind]
        t["rows"] += 1
        t["programs identical"] += int(same)
        for rep in row["replies"]:
            ok = VS.evaluate(spec, RV.views(rep["text"])) == rep["held"]
            t["replies"] += 1
            t["verdicts identical"] += int(ok)
            same = same and ok
        if not same:
            t["rows that differ"].append(row["id"])
    t["sha256"] = hashlib.sha256(raw).hexdigest()
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(local_paths.value("CUBBY_DATASETS") or ".")
    report = {"order": "0073", "what": "step 1: every stored program rebuilt with the core's compiler and join, every "
              "stored verdict with the core's evaluator", "sets": {}}
    for name, rel, kind in SETS:
        t = check(root / rel, kind)
        report["sets"][name] = dict(t, file=rel)
        print(f"{name}: {t['rows']} rows, programs identical {t['programs identical']}; {t['replies']} replies, "
              f"verdicts identical {t['verdicts identical']}; rows that differ {len(t['rows that differ'])}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
