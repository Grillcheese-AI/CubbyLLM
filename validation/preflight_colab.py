"""Replay the Colab notebook's data contract locally, before burning GPU time.

`notebooks/standin_gen3_masked_sft.ipynb` cell 1 asserts on the manifest and
cell 2 filters and splits the records. Both run *after* `pip install unsloth`
and a model download, so a broken manifest or an empty split costs a session
start to discover. Everything they check that does not need a GPU is
re-checkable in a second here.

    python validation/preflight_colab.py v14_nochain

Checks, in the notebook's own order:
  * both files exist beside each other
  * `manifest['version'] == VERSION`
  * the gen-3 lineage assert
  * the manifest keys cell 1 prints
  * cell 2's vm_ok / gold_match filter leaves records
  * the train/val split is non-empty on both sides
  * every record carries the fields `to_messages` reads
  * `train_rows_after_repeat` agrees with the corpus
"""
from __future__ import annotations

import collections
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "standin" / "data" / "out"
R7 = "cubbyllm/cot_harvest_r7"


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "v14_nochain"
    data = OUT / f"emitter_sft_{version}.jsonl"
    manifest = OUT / f"emitter_sft_{version}.manifest.json"

    print(f"preflight for STANDIN_VERSION={version}\n")
    fail = 0

    for f in (data, manifest):
        ok = f.is_file()
        print(f"  {'ok      ' if ok else 'MISSING '}{f.name}")
        fail += not ok
    if fail:
        print("\nFAIL: cell 1 stops here.")
        return 1

    m = json.loads(manifest.read_text(encoding="utf-8"))

    # cell 1's assert, verbatim in meaning
    lineage = ("gen3" in m) or str(m.get("base", "")).startswith("emitter_sft_v13e")
    if m.get("version") != version:
        print(f"\nFAIL: manifest version {m.get('version')!r} != {version!r}")
        fail += 1
    if not lineage:
        print("\nFAIL: manifest is not recognisably gen-3 lineage "
              "(needs a 'gen3' key or base starting emitter_sft_v13e)")
        fail += 1

    # the keys cell 1 prints straight after the assert
    for k in ("by_task", "n_records", "train_rows_after_repeat", "by_source"):
        if k not in m:
            print(f"\nFAIL: manifest is missing {k!r}, which cell 1 prints")
            fail += 1
    if fail:
        return 1
    print(f"\n  manifest ok: {m['version']} | records {m['n_records']:,} "
          f"| train rows after repeat {m['train_rows_after_repeat']:,}")
    print(f"  by_task: {m['by_task']}")

    # --- cell 2 ---
    recs = []
    with io.open(data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"\n  corpus rows: {len(recs):,}")
    if len(recs) != m["n_records"]:
        print(f"  FAIL: manifest says {m['n_records']:,} records, file has {len(recs):,}")
        fail += 1

    kept = [r for r in recs
            if r.get("vm_ok") in (True, None)
            and (r.get("gold_match") is not False or r.get("source") == R7)]
    print(f"  after cell 2's vm_ok/gold_match filter: {len(kept):,}")

    train = [r for r in kept if r.get("split") == "train"]
    val = [r for r in kept if r.get("split") in ("val", "held")]
    print(f"  train {len(train):,} {dict(collections.Counter(r['task'] for r in train))}")
    print(f"  val   {len(val):,} {dict(collections.Counter(r['task'] for r in val))}")
    if not train:
        print("  FAIL: the train split is empty -- training would have nothing to do")
        fail += 1
    if not val:
        print("  FAIL: the val split is empty")
        fail += 1

    # `to_messages` reads these; a missing one is an AttributeError mid-run
    missing = collections.Counter()
    for r in kept:
        for k in ("prompt", "program", "task", "split"):
            if not r.get(k):
                missing[k] += 1
    if missing:
        print(f"  FAIL: records missing fields to_messages reads: {dict(missing)}")
        fail += 1
    else:
        print("  every record carries prompt/program/task/split")

    rows_after_repeat = sum(int(r.get("repeat", 1) or 1) for r in train)
    print(f"  train rows after repeat (recomputed): {rows_after_repeat:,}")
    if rows_after_repeat != m["train_rows_after_repeat"]:
        print(f"  NOTE: manifest says {m['train_rows_after_repeat']:,}. The notebook prints "
              f"the manifest value and trains on the recomputed one, so a mismatch is "
              f"misleading rather than fatal.")

    # the plan example cell 2 pulls out for display
    if not any(r["task"] == "plan" for r in train):
        print("  FAIL: cell 2 does `next(r for r in train if r['task'] == 'plan')` "
              "and would raise StopIteration")
        fail += 1

    if fail:
        print(f"\nNOT READY: {fail} problem(s) above.")
        return 1
    print("\nREADY: the notebook's data contract is satisfied. Remaining steps are "
          "manual:\n"
          "  1. copy both files into the Drive standin/ folder\n"
          "  2. push the repo -- cell 1 does `git reset --hard origin/master`\n"
          f"  3. set STANDIN_VERSION={version} (and STANDIN_ARM) before cell 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
