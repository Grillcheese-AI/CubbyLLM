"""Order 0078, step 2: GSM8K itself (MIT, `openai/gsm8k`, the "main" config), off-repo, with ids.

From the two parquet files downloaded to ``$CUBBY_DATASETS/gsm8k`` (with the dataset card) it writes there:
- ``train.jsonl``: the 7,473 train problems, ``{"id": "gsm8k-train-<index>", "question", "answer", "dev"}``;
- ``test.jsonl``: the 1,319 test problems, ``{"id": "gsm8k-test-<index>", ...}`` (read once per arm, at the end);
- ``dev_ids.json``: the 329 problems the emitter holds out of GSM8K train (its val split,
  `standin/data/out/emitter_sft_v12e_w_slots.jsonl`, the ones `exp_he19_beam.py` reads), matched by their question's
  text, every one exactly;
- ``manifest.json``: the SHA-256 of every file, the counts, and two checks: no dev question is a test question, and
  no question of the emitter's training rows is a GSM8K test or dev question (the emitter's own contamination).
The manifest (ids and counts, no text) is copied to ``validation/logs/``. Indexes are the parquet's row order.

  python validation/builders/gsm8k_sets.py [--log validation/logs/gsm8k_2026-11-19_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

EMITTER = ROOT / "standin" / "data" / "out" / "emitter_sft_v12e_w_slots.jsonl"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(d: Path) -> dict:
    import pyarrow.parquet as pq
    src = {s: d / f"{s}-00000-of-00001.parquet" for s in ("train", "test")}
    rows = {s: pq.read_table(str(p)).to_pylist() for s, p in src.items()}
    emitter = [json.loads(ln) for ln in open(EMITTER, encoding="utf-8")]
    emitter = [r for r in emitter if r.get("task") == "arithmetic"]
    train_ix = {norm(r["question"]): i for i, r in enumerate(rows["train"])}
    test_q = {norm(r["question"]) for r in rows["test"]}
    val = [r for r in emitter if r.get("split") == "val"]
    dev = sorted({train_ix[norm(r["question"])] for r in val if norm(r["question"]) in train_ix})
    if len(dev) != len(val):
        raise SystemExit(f"{len(val) - len(dev)} of the emitter's val problems are not GSM8K train questions")
    dev_q = {norm(rows["train"][i]["question"]) for i in dev}
    out = {}
    for s in ("train", "test"):
        p = d / f"{s}.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(rows[s]):
                rec = {"id": f"gsm8k-{s}-{i}", "question": r["question"], "answer": r["answer"]}
                if s == "train":
                    rec["dev"] = i in set(dev)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out[s] = p
    dev_ids = [f"gsm8k-train-{i}" for i in dev]
    (d / "dev_ids.json").write_text(json.dumps(dev_ids, indent=0) + "\n", encoding="utf-8")
    trained = [norm(r["question"]) for r in emitter if r.get("split") != "val"]
    man = {"order": "0078", "what": "step 2: GSM8K (MIT, openai/gsm8k main) off-repo with ids; dev = the emitter's 329 "
           "held-out GSM8K train problems", "licence": "MIT (the dataset card, README.md beside the files)",
           "counts": {"train": len(rows["train"]), "test": len(rows["test"]), "dev (inside train)": len(dev)},
           "checks": {"dev questions that are test questions": len(dev_q & test_q),
                      "emitter training rows (arithmetic) whose question is a GSM8K test question": sum(q in test_q for q in trained),
                      "emitter training rows (arithmetic) whose question is a dev question": sum(q in dev_q for q in trained)},
           "dev ids": dev_ids,
           "sha256": {**{p.name: sha(p) for p in src.values()}, "README.md": sha(d / "README.md"),
                      **{p.name: sha(p) for p in out.values()}, "dev_ids.json": sha(d / "dev_ids.json")}}
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="validation/logs/gsm8k_2026-11-19_manifest.json")
    a = ap.parse_args()
    d = Path(local_paths.value("CUBBY_DATASETS") or ".") / "gsm8k"
    man = build(d)
    Path(a.log).write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("counts", "checks")}, indent=1))


if __name__ == "__main__":
    main()
