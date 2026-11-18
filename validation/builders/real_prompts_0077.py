"""Order 0077, build 1: the real prompts the description reader's rows are built around (off-repo; counts and hashes
only in the repo).

Round 6's split is kept (Nick, 2026-11-17): its rows (`$CUBBY_DATASETS/rule_rows_0067_v4_asked_real/rows.jsonl`) say
which screened prompt went to train, dev or a held-out test, by ``source_id``; the prompt itself is taken again from
`real_rows.pool()` (round 6's screens, its thinning of prompts under 200 characters, Dolly's passage placed by its own
seed), so the text is the prompt as screened, without the rule round 6 put inside it.
- **train_real / dev_real:** OASST1 prompts (English and French) of round 6's train and dev rows. No Dolly prompt:
  an adapter trained on CC BY-SA text is not served. Round 6's French held-out prompts are known by their split
  (``real_*_test``) and kept out of both;
- **test_real:** English OASST1 prompts of round 6's held-out rows, drawn with ``seed`` (77);
- **test_dolly:** every English Dolly prompt of round 6's held-out rows;
- every set: at most ``MAX_CHARS`` characters; train and dev: no prompt sharing 5 words with IFEval or IFBench.

`sets(seed, n_train, n_dev, n_test, bench)` -> ({set: [(source id, lang, prompt)] in its seeded order}, counts); the
builder uses each set's first prompts until it has its rows. No prompt is in two sets.

Wiring: STANDALONE (the 0077 row builder).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import bench_overlap as BO  # noqa: E402
import real_rows  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

MAX_CHARS = 600
ROUND6 = "rule_rows_0067_v4_asked_real"


def round6_splits(ds: Path | None = None) -> dict:
    """{source id: (round 6's split, "train" | "dev" | "held"), lang} for every real prompt round 6 used."""
    ds = ds or Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = {}
    for ln in open(ds / ROUND6 / "rows.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        sid = str(r.get("source_id") or "")
        if sid.startswith(("oasst1:", "dolly:")):
            out[sid] = ("held" if r["split"].endswith("_test") else r["split"], r["lang"])
    return out


def clean(prompt: str) -> str:
    return prompt.replace("\r\n", "\n").replace("\r", "\n").strip()


def sets(seed: int = 77, n_train: int = 2000, n_dev: int = 300, n_test: int = 300, bench: dict | None = None,
         splits: dict | None = None, pool: dict | None = None) -> tuple[dict, dict]:
    splits = round6_splits() if splits is None else splits
    pool = pool if pool is not None else {sid: clean(p) for ps in real_rows.pool().values() for sid, p in ps}
    counts = Counter()
    cand = {"train_real": [], "dev_real": [], "test_real": [], "test_dolly": []}
    for sid in sorted(splits):
        split, lang = splits[sid]
        src = sid.split(":")[0]
        if sid not in pool:
            counts["round 6 prompt not in the pool again (must be 0)"] += 1
            continue
        p = pool[sid]
        if len(p) > MAX_CHARS:
            counts[f"over {MAX_CHARS} characters: {split} {src} {lang}"] += 1
            continue
        if split == "held":
            if lang == "en":
                cand["test_real" if src == "oasst1" else "test_dolly"].append((sid, lang, p))
            else:
                counts["French held-out prompts of round 6 kept out"] += 1
        elif src == "oasst1":
            if bench and BO.overlaps(p, bench):
                counts[f"shares 5 words with a benchmark: {split}"] += 1
                continue
            cand["train_real" if split == "train" else "dev_real"].append((sid, lang, p))
    for name, c in cand.items():
        counts[f"candidates: {name}"] = len(c)
    out = {}
    for i, (name, n) in enumerate((("test_real", n_test), ("train_real", n_train), ("dev_real", n_dev),
                                   ("test_dolly", n_test))):
        c, rng = cand[name], random.Random(seed if i == 0 else seed * 10 + i)    # test_real: seed 77 on its own
        if len(c) < n and name != "test_dolly":
            raise SystemExit(f"{name}: {len(c)} prompts, {n} asked")
        out[name] = rng.sample(c, len(c))                # the seeded order: the builder takes from the front until n
        counts[f"wanted: {name}"] = n                    # rows are made (a prompt no draw fits is replaced by the next)
    ids = [sid for v in out.values() for sid, _, _ in v]
    assert len(ids) == len(set(ids)), "a prompt in two sets"
    return out, dict(counts)


__all__ = ["MAX_CHARS", "clean", "round6_splits", "sets"]
