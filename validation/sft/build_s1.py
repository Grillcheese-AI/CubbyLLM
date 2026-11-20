"""build_s1 -- order 0078, S1's set: S0's recipe with all of TinyGSM. Step rows (H-E19's write-back format).

- **The base**, as S0's: ``emitter_sft_v12e_w_slots.jsonl`` with its arithmetic re-cut into step rows
  (`step_loop.recut`), the other families copied. Its GSM8K rows carry ``gsm8k_id`` (matched by question to
  ``$CUBBY_DATASETS/gsm8k/train.jsonl``, as Code A matched dev): the 5,819 train problems in its train split and the
  329 dev problems, which are its val split. S2 finds each gold's ``<<...>>`` text by that id.
- **GSM8K train's other problems** (`gsm8k_programs.py`: the golds' annotations as programs, kept when they give the
  gold), tagged ``gsm8k_id``; the misses listed by id and kind.
- **TinyGSM**, step 1's kept programs (``$CUBBY_DATA/tinygsm_0078_nop6/tinygsm_s<k>.jsonl``), each put in slot form
  (`emitter_data.slot_record`) and re-cut; a program that does not re-cut is counted, not kept. Two selections:
  ``--mode mix`` (S1's first cut, Nick 2026-11-20): whole programs sampled (seed 78) to GSM8K train's mix of step
  counts (its golds' ``<<...>>`` steps, outside dev; 10 and more counted as 10) until the rows come to
  ``--rows`` in all; ``--mode cap``: every program of ``--keep-steps`` steps or more, the shorter ones sampled by a hash
  of their id so the rows come to about ``--cap-rows`` (0: all).

  python validation/sft/build_s1.py [--mode mix --rows 1500000 | --mode cap --cap-rows 0 --keep-steps 5]
      [--out-dir $CUBBY_DATA/s1_0078]
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


def gsm8k_mix(datasets: str) -> Counter:
    """GSM8K train's golds (outside dev) by their number of ``<<...>>`` steps, 10 and more as 10."""
    from validation.sft.gsm8k_programs import ANNOT
    c: Counter = Counter()
    for r in map(json.loads, open(os.path.join(datasets, "gsm8k", "train.jsonl"), encoding="utf-8")):
        n = len(ANNOT.findall(r["answer"]))
        if not r.get("dev") and n:
            c[min(n, 10)] += 1
    return c


def mix_counts(mix: Counter, rows: int, avail: Counter) -> dict[int, int]:
    """Programs to draw per step count: GSM8K's shares, scaled so the rows (s + 1 a program) come to ``rows``."""
    tot = sum(mix.values())
    per = sum(n / tot * (s + 1) for s, n in mix.items())
    return {s: min(avail.get(s, 0), round(n / tot * rows / per)) for s, n in sorted(mix.items())}


def _write_rows(f, rec, counter, slot_record, recut) -> int:
    s = slot_record(rec)
    rows = recut(s) if s else []
    if not rows:
        counter["not_slotted" if not s else "not_recut"] += 1
        return 0
    counter["programs"] += 1
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    import random
    from cubbyllm import local_paths
    from cubbyllm.reasoning.step_loop import recut
    from emitter_data import slot_record
    from validation.sft import gsm8k_programs as GP
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mix", "cap"), default="mix")
    ap.add_argument("--rows", type=int, default=1_500_000, help="mix: the step rows in all")
    ap.add_argument("--cap-rows", type=int, default=0)
    ap.add_argument("--keep-steps", type=int, default=5)
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(17)))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    data, datasets = local_paths.value("CUBBY_DATA") or ".", local_paths.value("CUBBY_DATASETS") or "."
    out_dir = a.out_dir or os.path.join(data, "s1_0078")
    os.makedirs(out_dir, exist_ok=True)
    ids, base, seen = gsm8k_ids(datasets), Counter(), set()
    with open(os.path.join(out_dir, "base_step.jsonl"), "w", encoding="utf-8") as f:
        for r in map(json.loads, open(os.path.join(data, "emitter_sft_v12e_w_slots.jsonl"), encoding="utf-8")):
            if r.get("task") != "arithmetic":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                base[f"{r['task']}/{r['split']}"] += 1
                continue
            gid = ids.get(_norm(r.get("question")))
            if gid:
                r["gsm8k_id"] = gid
                seen.add(gid)
                base[f"gsm8k_tagged/{r['split']}"] += 1
            rows = recut(r)
            base[f"arithmetic_step_rows/{r['split']}" if rows else f"arithmetic_not_recut/{r['split']}"] += len(rows) or 1
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra, misses, extra_rows = Counter(), [], 0
    with open(os.path.join(out_dir, "gsm8k_extra_step.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(map(json.loads, open(os.path.join(datasets, "gsm8k", "train.jsonl"), encoding="utf-8"))):
            if r.get("dev") or r["id"] in seen:
                continue
            rec, kind = GP.program(r, i)
            if rec is None:
                misses.append({"id": r["id"], "kind": kind})
                extra[kind] += 1
                continue
            rec["split"] = "train"
            extra_rows += _write_rows(f, rec, extra, slot_record, recut)
    base_rows = sum(v for k, v in base.items() if not k.startswith(("gsm8k_tagged", "arithmetic_not")))
    tg, shard_rows, per_shard = Counter(), {}, {}
    step1 = json.load(open(os.path.join(ROOT, "validation", "logs", "tinygsm_0078_nop6_2026-11-19.json"), encoding="utf-8"))
    avail: Counter = Counter()
    for m in step1["shards"].values():
        avail.update({int(k): v for k, v in m["kept_by_steps"].items()})
    mix, want, share = gsm8k_mix(datasets), {}, 1.0
    if a.mode == "mix":
        want = mix_counts(mix, a.rows - base_rows - extra_rows, avail)
        pools: dict[int, list] = {s: [] for s in want}
        for k in a.shards:
            for ln, line in enumerate(open(os.path.join(data, "tinygsm_0078_nop6", f"tinygsm_s{k}.jsonl"), encoding="utf-8")):
                st = int(line.rsplit('"steps": ', 1)[1].split(",", 1)[0])
                if st in pools:
                    pools[st].append((k, ln))
        rng = random.Random(78)
        for s, n in want.items():
            for k, ln in rng.sample(pools[s], min(n, len(pools[s]))):
                per_shard.setdefault(k, set()).add(ln)
    else:
        share = keep_share(avail, a.cap_rows, a.keep_steps)
    for k in a.shards:
        n = 0
        with open(os.path.join(out_dir, f"tinygsm_step_s{k}.jsonl"), "w", encoding="utf-8") as f:
            for ln, line in enumerate(open(os.path.join(data, "tinygsm_0078_nop6", f"tinygsm_s{k}.jsonl"), encoding="utf-8")):
                if a.mode == "mix" and ln not in per_shard.get(k, ()):
                    continue
                rec = json.loads(line)
                if a.mode == "cap" and rec["difficulty"]["steps"] < a.keep_steps and not sampled(rec["id"], share):
                    continue
                n += _write_rows(f, rec, tg, slot_record, recut)
                tg[f"steps={rec['difficulty']['steps']}"] += 1
        shard_rows[f"s{k}"] = n
        print(f"s{k}: {n:,} step rows", flush=True)
    rep = {"what": "order 0078 S1's set: v12e_w re-cut (GSM8K rows tagged), GSM8K's other golds, TinyGSM step rows",
           "date": _dt.date.today().isoformat(), "mode": a.mode, "rows_wanted": a.rows if a.mode == "mix" else a.cap_rows,
           "base": dict(sorted(base.items())), "base_rows": base_rows,
           "gsm8k_extra": {"kinds": dict(extra), "step_rows": extra_rows, "misses": misses},
           "gsm8k_mix": dict(sorted(mix.items())), "tinygsm_drawn": {str(s): n for s, n in want.items()},
           "tinygsm": dict(sorted(tg.items())), "tinygsm_step_rows": sum(shard_rows.values()),
           "rows_total": base_rows + extra_rows + sum(shard_rows.values()), "by_shard": shard_rows, "seed": 78}
    log = a.log or os.path.join(ROOT, "validation", "logs", f"s1_0078_{rep['date']}.json")
    json.dump(rep, open(log, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_shard", "gsm8k_extra")}), "->", log)


if __name__ == "__main__":
    main()
