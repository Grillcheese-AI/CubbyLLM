"""Order 0077, build 3: arm R's SFT set, in the talk trainer's record form (arm M's target: folded, free values).

- **train:** 0074's 8,000 train records (`long_rows_0074`: train rows and minimal pairs, made again here and checked
  record for record against 0074's own set, ``--check``) and 0077's ``train_real`` rows (OASST1 only);
- **held:** 0077's ``dev_real`` (the trainer's held loss). 0074's dev, every test set and every Dolly prompt are in
  no record: an adapter trained on CC BY-SA text is not served.

Before the set is written, as 0073's and 0074's: every answer reads back strictly, also with ``own_text``; no record
shares 5 words with IFEval or IFBench (a real row that does stops the build: its prompt was screened); with
``--tokenizer`` every record fits ``--max-len`` (a record over it stops the build: the trainer would skip it).

  python validation/sft/build_view_real_sft.py --out-dir <dir> [--tokenizer <json>] [--check <view_long_sft_l.jsonl>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation"), str(ROOT / "validation" / "builders"), str(ROOT / "validation" / "sft")]
import bench_overlap as BO  # noqa: E402
import build_view_split_sft as S  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402

__wiring__ = "STANDALONE"

NAME = "view_real_sft_r.jsonl"
SETS = ("train_real", "dev_real", "test_real", "test_dolly")
READ_MAX = 900                                         # exp_view_split_torch.py's limit for arm M


def rows_of(ds: Path) -> tuple[list, list]:
    """(0074's train rows and pairs, in its set's order; 0077's train_real then dev_real)."""
    old = [r for r in S.rows_of(ds / "long_rows_0074") if r["split"] != "dev"]
    new = [json.loads(ln) for ln in open(ds / "long_rows_0077" / "rows.jsonl", encoding="utf-8")]
    return old, [r for r in new if r["split"] == "train_real"] + [r for r in new if r["split"] == "dev_real"]


def record(r: dict) -> dict:
    rec = S.record(r, "m")
    if "source" in r:                                  # a 0077 row
        rec.update(split="held" if r["split"] == "dev_real" else "train", family=f"real_{r['n_parts']}")
    return rec


def build(ds: Path, out_dir: Path, tokenizer: str | None, max_len: int, check: str | None) -> dict:
    old, new = rows_of(ds)
    if any(r.get("source") != "oasst1" for r in new):
        raise SystemExit("a row whose prompt is not OASST1's: no Dolly text is trained on")
    bench = BO.benchmark_grams(ifbench=True)
    leaks = [r["id"] for r in old + new if BO.overlaps(r["instruction"], bench)]
    if leaks:
        raise SystemExit(f"{len(leaks)} rows share 5 words with a benchmark: {leaks[:5]}")
    rows = old + new
    recs = [record(r) for r in rows]
    read = sum("lines" in V.parse(rec["answer"], r["instruction"], "m", own_text=own) for rec, r in zip(recs, rows)
               for own in (False, True))
    if read != 2 * len(recs):
        raise SystemExit(f"{2 * len(recs) - read} answers do not read back")
    same = None
    if check:                                          # 0074's own records, as built then
        theirs = [json.loads(ln) for ln in open(check, encoding="utf-8")]
        theirs = [t for t in theirs if t["split"] == "train"]
        same = len(theirs) == len(old) and all(a == b for a, b in zip(recs[:len(old)], theirs))
        if not same:
            raise SystemExit("0074's train records made again differ from 0074's set")
    sizes = {}
    if tokenizer:
        tk = __import__("tokenizers").Tokenizer.from_file(tokenizer)
        sizes = S.lengths(recs, tk, max_len)
        every = [json.loads(ln) for ln in open(ds / "long_rows_0077" / "rows.jsonl", encoding="utf-8")]
        ids = {s: [len(tk.encode(V.target(r, "m")).ids) for r in every if r["split"] == s] for s in SETS}
        sizes["0077 answers in ids (counts only), most by set"] = {s: max(v) for s, v in ids.items() if v}
        sizes[f"0077 answers over the read's {READ_MAX} ids, by set"] = {s: sum(x > READ_MAX for x in v) for s, v in ids.items()}
        if any(sizes[f"0077 answers over the read's {READ_MAX} ids, by set"].values()):
            raise SystemExit(f"answers longer than the read allows: {sizes}")
    if sizes.get("records_over_max_len"):
        raise SystemExit(f"records over {max_len} tokens: {sizes['records_over_max_len']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / NAME
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs)
    m77 = json.loads((ds / "long_rows_0077" / "manifest.json").read_text(encoding="utf-8"))["sha256"]
    m74 = json.loads((ds / "long_rows_0074" / "manifest.json").read_text(encoding="utf-8"))["sha256"]
    man = {"order": "0077", "what": "arm R's SFT set: 0074's train records and the real-prompt rows (OASST1), arm M's "
           "target", "arm": "m", "records": len(recs), "by_split": dict(Counter(x["split"] for x in recs)),
           "by_family": dict(sorted(Counter(x["family"] for x in recs).items())),
           "by_lang": dict(Counter(x["lang"] for x in recs)),
           "answers that read back strictly, also with own_text": read // 2,
           "0074's train records, the same as 0074's set": same, "rows of test or test_long": 0,
           "rows of test_real or test_dolly": 0, "dolly records": 0, "benchmarks_checked": sorted(bench),
           "records_sharing_5_words_with_a_benchmark_dropped": 0, "max_len": max_len,
           **(sizes or {"records_over_max_len": None}),
           "rows_sha256": {"long_rows_0074": m74, "long_rows_0077": m77},
           "row_licences": {"templates": "this repo's own (Apache-2.0)", "prompts": "OASST1 (Apache-2.0)"},
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--check", default=None, help="0074's view_long_sft_l.jsonl: its train records must be remade alike")
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    print(json.dumps(build(ds, Path(a.out_dir), a.tokenizer, a.max_len, a.check), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
