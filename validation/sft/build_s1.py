"""build_s1 -- order 0078, S1's set: S0's recipe with all of TinyGSM. Step rows (H-E19's write-back format).

- **The base**, as S0's: ``emitter_sft_v12e_w_slots.jsonl`` with its arithmetic re-cut into step rows
  (`step_loop.recut`), the other families copied. Its GSM8K rows carry ``gsm8k_id`` (matched by question to
  ``$CUBBY_DATASETS/gsm8k/train.jsonl``, as Code A matched dev): the 5,819 train problems in its train split and the
  329 dev problems, which are its val split. S2 finds each gold's ``<<...>>`` text by that id.
- **TinyGSM**, step 1's kept programs (``$CUBBY_DATA/tinygsm_0078_nop6/tinygsm_s<k>.jsonl``), each put in slot form
  (`emitter_data.slot_record`) and re-cut; a program that does not re-cut is counted, not kept. ``--cap-rows N``: every
  program of ``--keep-steps`` steps or more (the rare long plans), and the shorter ones sampled by a hash of their id
  so the rows come to about N (0: all of them).

  python validation/sft/build_s1.py [--cap-rows 0] [--keep-steps 5] [--out-dir $CUBBY_DATA/s1_0078]
      [--log validation/logs/s1_0078_<date>.json]
-> ``base_step.jsonl`` and ``tinygsm_step_s<k>.jsonl`` off-repo; one manifest (counts only) under validation/logs.

Wired: STANDALONE (order 0078's data build).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

_norm = lambda q: " ".join(re.findall(r"[a-z0-9']+", (q or "").lower()))


def gsm8k_ids(datasets: str) -> dict[str, str]:
    return {_norm(r["question"]): r["id"] for r in map(json.loads, open(os.path.join(datasets, "gsm8k", "train.jsonl"),
                                                                       encoding="utf-8"))}


def keep_share(by_steps: dict[int, int], cap: int, keep_steps: int) -> float:
    """The share of short programs to keep so the rows come to about ``cap`` (a program of s steps gives s + 1 rows)."""
    if not cap:
        return 1.0
    long_rows = sum((s + 1) * n for s, n in by_steps.items() if s >= keep_steps)
    short_rows = sum((s + 1) * n for s, n in by_steps.items() if s < keep_steps)
    return max(0.0, min(1.0, (cap - long_rows) / max(1, short_rows)))


def sampled(rid: str, share: float) -> bool:
    return share >= 1.0 or int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def main() -> None:
    from cubbyllm import local_paths
    from cubbyllm.reasoning.step_loop import recut
    from emitter_data import slot_record
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-rows", type=int, default=0)
    ap.add_argument("--keep-steps", type=int, default=5)
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(17)))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    data, datasets = local_paths.value("CUBBY_DATA") or ".", local_paths.value("CUBBY_DATASETS") or "."
    out_dir = a.out_dir or os.path.join(data, "s1_0078")
    os.makedirs(out_dir, exist_ok=True)
    ids, base = gsm8k_ids(datasets), Counter()
    with open(os.path.join(out_dir, "base_step.jsonl"), "w", encoding="utf-8") as f:
        for r in map(json.loads, open(os.path.join(data, "emitter_sft_v12e_w_slots.jsonl"), encoding="utf-8")):
            if r.get("task") != "arithmetic":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                base[f"{r['task']}/{r['split']}"] += 1
                continue
            gid = ids.get(_norm(r.get("question")))
            if gid:
                r["gsm8k_id"] = gid
                base[f"gsm8k_tagged/{r['split']}"] += 1
            rows = recut(r)
            base[f"arithmetic_step_rows/{r['split']}" if rows else f"arithmetic_not_recut/{r['split']}"] += len(rows) or 1
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    step1 = json.load(open(os.path.join(ROOT, "validation", "logs", "tinygsm_0078_nop6_2026-11-19.json"), encoding="utf-8"))
    by_steps: Counter = Counter()
    for m in step1["shards"].values():
        by_steps.update({int(k): v for k, v in m["kept_by_steps"].items()})
    share = keep_share(by_steps, a.cap_rows, a.keep_steps)
    tg, shard_rows = Counter(), {}
    for k in a.shards:
        n = 0
        with open(os.path.join(out_dir, f"tinygsm_step_s{k}.jsonl"), "w", encoding="utf-8") as f:
            for rec in map(json.loads, open(os.path.join(data, "tinygsm_0078_nop6", f"tinygsm_s{k}.jsonl"), encoding="utf-8")):
                steps = rec["difficulty"]["steps"]
                if steps < a.keep_steps and not sampled(rec["id"], share):
                    tg["sampled_out"] += 1
                    continue
                s = slot_record(rec)
                rows = recut(s) if s else []
                if not rows:
                    tg["not_slotted" if not s else "not_recut"] += 1
                    continue
                tg["programs"] += 1
                tg[f"programs_{'long' if steps >= a.keep_steps else 'short'}"] += 1
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += len(rows)
        shard_rows[f"s{k}"] = n
        print(f"s{k}: {n:,} step rows", flush=True)
    rep = {"what": "order 0078 S1's set: v12e_w re-cut (GSM8K rows tagged) + TinyGSM step rows",
           "date": _dt.date.today().isoformat(), "cap_rows": a.cap_rows, "keep_steps": a.keep_steps,
           "short_share_kept": round(share, 6), "base": dict(sorted(base.items())), "tinygsm": dict(tg),
           "tinygsm_step_rows": sum(shard_rows.values()), "by_shard": shard_rows}
    log = a.log or os.path.join(ROOT, "validation", "logs", f"s1_0078_{rep['date']}.json")
    json.dump(rep, open(log, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "by_shard"}), "->", log)


if __name__ == "__main__":
    main()
