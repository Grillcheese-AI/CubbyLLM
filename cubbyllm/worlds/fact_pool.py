"""cubbyllm/worlds/fact_pool.py -- the serve stack's fact pool: the deduplicated facts of a seeded sample of the
scaling-law questions (``valid_scaling_law_with_facts.pq`` under ``$CUBBY_DATA_ROOT``), each question's chain as
indices into it; and where the fast-word table the facts world encodes with lives (``$CUBBY_MODELS``). Order 0031
moved them here, verbatim, from ``validation/exp_m3_cot_pipeline.py`` (which re-exports them), since the server
builds its first world from them.

Wired: WIRED (the server's ``build_serve``).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np

from ..local_paths import expand as _lp

__wiring__ = "WIRED"

PQ_FILE = pathlib.Path(_lp(r"$CUBBY_DATA_ROOT/valid_scaling_law_with_facts.pq"))
V4_TABLE = pathlib.Path(os.environ.get("CUBBY_MODELS", _lp(r"$CUBBY_MODELS"))) / "fastword_table_v4.npz"


def load_sample(n: int, seed: int, exclude: set[str] | None = None):
    """-> (questions, answers, hops, chains, store).

    `chains[i]` is question i's own supporting-fact chain, in walk order,
    as indices into `store` (the whitespace-normalized, deduped fact pool
    built from exactly this sample -- same trick as exp_m3_injection).
    """
    import pyarrow.parquet as pq

    t = pq.read_table(PQ_FILE, columns=["question_prompt", "facts", "answer", "n_hop"])
    rows = [(q, f, a, int(h)) for q, f, a, h in zip(
        t.column("question_prompt").to_pylist(), t.column("facts").to_pylist(),
        t.column("answer").to_pylist(), t.column("n_hop").to_pylist())
        if q and f and a and len(q) > 15]
    if exclude:
        rows = [r for r in rows if r[0] not in exclude]

    rng = np.random.default_rng(seed)
    if len(rows) > n:
        pick = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[j] for j in sorted(pick)]

    fact_id: dict[str, int] = {}
    store: list[str] = []

    def fid(s: str) -> int:
        s = " ".join(s.split())
        if s not in fact_id:
            fact_id[s] = len(store)
            store.append(s)
        return fact_id[s]

    chains = [[fid(x) for x in f] for _, f, _, _ in rows]
    return ([r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
            chains, store)


# --------------------------------------------------------------------------
# 2. Retriever: unit-cosine top-k over a precomputed store.
# --------------------------------------------------------------------------
