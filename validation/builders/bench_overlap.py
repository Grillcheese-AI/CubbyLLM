"""Order 0067: no benchmark prompt reaches a training row. The runs of N consecutive words (N = 5) that a text shares with
a benchmark's prompts, words lowercased and stripped of punctuation.

Benchmarks are read from their off-repo place: IFEval's ``ifeval_input_data.jsonl`` under ``$CUBBY_DATASETS/ifeval``,
M-IFEval's French prompts as any ``*.jsonl`` with a ``prompt`` field under ``$CUBBY_DATASETS/m-ifeval/fr`` (absent on
this machine until its licence is read). `benchmark_grams()` returns {name: grams} for the ones present.

  python validation/builders/bench_overlap.py [--rows <rows.jsonl>]   # prints the train/dev wordings that overlap
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

N = 5


def words(text: str) -> list[str]:
    return re.findall(r"[\w']+", (text or "").lower().replace("’", "'"))


def grams(text: str, n: int = N) -> set[tuple]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _prompts(path: Path) -> list[str]:
    return [json.loads(line).get("prompt", "") for line in open(path, encoding="utf-8") if line.strip()]


def benchmark_grams() -> dict:
    root = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = {}
    ife = root / "ifeval" / "ifeval_input_data.jsonl"
    if ife.exists():
        out["IFEval"] = set().union(*(grams(p) for p in _prompts(ife)))
    fr = root / "m-ifeval" / "fr"
    if fr.exists():
        files = sorted(fr.glob("*.jsonl"))
        if files:
            out["M-IFEval fr"] = set().union(*(grams(p) for f in files for p in _prompts(f)))
    return out


def overlaps(text: str, bench: dict) -> dict:
    """{benchmark: the shared runs} for `text` (empty when clean)."""
    g = grams(text)
    return {name: sorted(" ".join(x) for x in g & bg) for name, bg in bench.items() if g & bg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "rule_rows_0067" / "rows.jsonl")
    bench = benchmark_grams()
    print("benchmarks read:", sorted(bench))
    seen = {}
    for line in open(rows_p, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] not in ("train", "dev"):
            continue
        hit = overlaps(r["instruction"], bench)
        if hit:
            key = (r["kind"], r["lang"], r.get("template"))
            seen.setdefault(key, (r["instruction"], hit))
    for (kind, lang, t), (text, hit) in sorted(seen.items(), key=str):
        print(f"{kind} {lang} template {t}: {text}\n    {hit}")
    print(len(seen), "train/dev wordings overlap")


if __name__ == "__main__":
    main()
