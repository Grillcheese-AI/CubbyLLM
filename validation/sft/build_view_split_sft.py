"""Order 0073, step 5: the reader's two SFT sets, in the talk trainer's record form
(``validation/train_talk_torch.py``: {"id", "split", "family", "lang", "prompt", "answer"}).

The same records in both sets, in the same order; only the answer's spelling of a negation differs (arm M folded,
arm W as said: `cubbyllm/speech/view_split.target`). From `$CUBBY_DATASETS/long_rows_0073`:
- ``rows.jsonl``: ``train`` -> train (``long_<parts>``), ``dev`` -> held (the trainer's held loss);
- ``pair_rows.jsonl``: the minimal pairs -> train (``pair_<parts>``), a pair's two rows side by side.
``test`` and ``test_long`` are never trained on: they are what the arms are read on. 0067's set also had one-part
instructions and CubeLang programs to continue; this reader writes no program, so neither is here.

Before a set is written: every answer parses back strictly to its row, also with ``own_text`` (a set whose answers
the grammar would refuse cannot be taught); no instruction shares 5 words with IFEval or IFBench. With
``--tokenizer`` the records longer than ``--max-len`` are counted (the trainer skips them; none is expected), and so
is the share of answer tokens that are key (`key_tokens`: each check's tag and description, weighted in training).

  python validation/sft/build_view_split_sft.py --out-dir <dir> [--tokenizer <json>] [--max-len 1280]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation"), str(ROOT / "validation" / "builders")]
import bench_overlap as BO  # noqa: E402
import key_tokens as K  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402

__wiring__ = "STANDALONE"

NAME = "view_split_sft_{arm}.jsonl"


def rows_of(ds: Path) -> list[dict]:
    """The rows a set is made of, in its order: train rows, the pairs, then dev."""
    main = [json.loads(ln) for ln in open(ds / "rows.jsonl", encoding="utf-8")]
    pairs = [json.loads(ln) for ln in open(ds / "pair_rows.jsonl", encoding="utf-8")]
    return [r for r in main if r["split"] == "train"] + pairs + [r for r in main if r["split"] == "dev"]


def record(r: dict, arm: str) -> dict:
    family = r.get("family") or f"long_{r['n_parts']}"
    return {"id": r["id"], "split": "held" if r["split"] == "dev" else "train", "family": family, "lang": r["lang"],
            "prompt": V.prompt(r["instruction"]), "answer": V.target(r, arm)}


def lengths(recs: list[dict], tk, max_len: int, key_weight: float = 4.0) -> dict:
    """Token counts as the trainer makes them (`talk_data.encode_records`), and the key tokens' share."""
    total, over, key, answer = [], Counter(), 0, 0
    for r in recs:
        w = K.weights(tk, r["answer"], key_weight)
        n = len(tk.encode(r["prompt"]).ids) + len(w)
        total.append(n)
        over[r["family"]] += n > max_len
        key += sum(1 for x in w if x == key_weight)
        answer += len(w)
    return {"tokens a record, median": statistics.median(total), "tokens a record, most": max(total),
            "records_over_max_len": {f: n for f, n in over.items() if n}, "answer tokens": answer,
            "answer tokens that are key": key, "share of answer tokens that are key": round(key / answer, 4)}


def build(ds: Path, out_dir: Path, tokenizer: str | None, max_len: int) -> dict:
    rows, bench = rows_of(ds), BO.benchmark_grams(ifbench=True)
    leaks = {r["id"] for r in rows if BO.overlaps(r["instruction"], bench)}
    rows = [r for r in rows if r["id"] not in leaks]
    tk = None
    if tokenizer:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tokenizer)
    sets = {arm: [record(r, arm) for r in rows] for arm in V.ARMS}
    same = all(a["id"] == b["id"] and a["prompt"] == b["prompt"] and a["split"] == b["split"]
               for a, b in zip(sets["m"], sets["w"]))
    source = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))["sha256"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mans = {}
    for arm, recs in sets.items():
        read = sum("lines" in V.parse(rec["answer"], r["instruction"], arm, own_text=own)
                   for rec, r in zip(recs, rows) for own in (False, True))
        if read != 2 * len(recs) or not same:
            raise SystemExit(f"arm {arm}: {2 * len(recs) - read} answers do not read back, or the two sets' records differ")
        out = out_dir / NAME.format(arm=arm)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs)
        mans[arm] = {"order": "0073", "what": "the reader's SFT set (step 5): the split with each check's description",
                     "arm": arm, "records": len(recs), "by_split": dict(Counter(x["split"] for x in recs)),
                     "by_family": dict(sorted(Counter(x["family"] for x in recs).items())),
                     "by_lang": dict(Counter(x["lang"] for x in recs)),
                     "answers that read back strictly, also with own_text": read // 2,
                     "answers that differ from the other arm's": sum(a["answer"] != b["answer"] for a, b in zip(sets["m"], sets["w"])),
                     "the two sets have the same ids, prompts and splits in the same order": same,
                     "rows of test or test_long": 0, "benchmarks_checked": sorted(bench),
                     "records_sharing_5_words_with_a_benchmark_dropped": len(leaks), "max_len": max_len,
                     **(lengths(recs, tk, max_len) if tk else {"records_over_max_len": None}),
                     "rows_sha256": source, "row_licences": {"templates": "this repo's own (Apache-2.0)"},
                     "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        Path(str(out) + ".manifest.json").write_text(json.dumps(mans[arm], indent=1, ensure_ascii=False), encoding="utf-8")
    return mans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-len", type=int, default=1280)
    a = ap.parse_args()
    ds = Path(a.set or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073")
    print(json.dumps(build(ds, Path(a.out_dir), a.tokenizer, a.max_len), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
