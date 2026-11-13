"""Order 0067: the extra row sets `build_rule_rows` adds after its drawn rules: Nick's annotated Alpaca prompts (round
4), the rows answering ``asked`` (round 5), real prompts from OASST1 and Dolly-15k (round 6). Each drops a train or dev
row that shares 5 words with a benchmark prompt, and numbers the rows it adds after the ones it is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__wiring__ = "STANDALONE"


def with_alpaca(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """`rows` plus Nick's annotated Alpaca prompts (`alpaca_rows`): natural length rules and rule-free prompts. A train
    or dev prompt that shares 5 words with a benchmark prompt is dropped (a real prompt is not redrawn). Ids follow."""
    import alpaca_rows
    import bench_overlap as BO
    bench = BO.benchmark_grams()
    extra, dropped = [], 0
    for r in alpaca_rows.rows(seed):
        if r["split"] in ("train", "dev") and bench and BO.overlaps(r["instruction"], bench):
            dropped += 1
            continue
        extra.append(r)
    for i, r in enumerate(extra, start=len(rows)):
        r["id"] = f"r{i:06d}"
    return rows + extra, {"alpaca_rows": len(extra), "alpaca_dropped_for_benchmark_overlap": dropped,
                          "alpaca_licence": "Stanford Alpaca text CC BY-NC 4.0; annotations Nick's own"}


def with_asked(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """`rows` plus the rows whose answer is ``asked`` (`asked_rows`, round 5); train/dev ones sharing 5 words with a
    benchmark prompt are dropped."""
    import asked_rows
    import bench_overlap as BO
    bench = BO.benchmark_grams()
    measurable = [(r["instruction"], r["plan"], r["lang"]) for r in rows
                  if r["split"] == "train" and r.get("template") is not None and "+" not in r["kind"]]
    extra, dropped = [], 0
    for r in asked_rows.rows(seed + 3, measurable):
        if r["split"] in ("train", "dev") and bench and BO.overlaps(r["instruction"], bench):
            dropped += 1
            continue
        extra.append(r)
    for i, r in enumerate(extra, start=len(rows)):
        r["id"] = f"r{i:06d}"
    return rows + extra, {"asked_rows": len(extra), "asked_dropped_for_benchmark_overlap": dropped}


def with_real(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """`rows` plus round 6's real prompts (`real_rows`: OASST1 and Dolly-15k, rules and phrases put inside them);
    train/dev ones sharing 5 words with a benchmark prompt (IFEval, IFBench) are dropped."""
    import bench_overlap as BO
    import real_rows
    bench = BO.benchmark_grams()
    measurable = [(r["instruction"], r["plan"], r["lang"], r["ifeval"]) for r in rows
                  if r["split"] == "train" and r.get("template") is not None and "+" not in r["kind"]]
    extra, dropped = [], 0
    for r in real_rows.rows(seed + 6, measurable):
        if r["split"] in ("train", "dev") and bench and BO.overlaps(r["instruction"], bench):
            dropped += 1
            continue
        extra.append(r)
    for i, r in enumerate(extra, start=len(rows)):
        r["id"] = f"r{i:06d}"
    return rows + extra, {"real_rows": len(extra), "real_dropped_for_benchmark_overlap": dropped,
                          "real_benchmarks_checked": sorted(bench),
                          "real_licences": "OASST1 prompts Apache-2.0; Dolly-15k instructions and passages CC BY-SA 3.0"}


__all__ = ["with_alpaca", "with_asked", "with_real"]
