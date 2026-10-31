"""r2_qfhrr_score -- order 0046, R2's qFHRR arm, the similarity half. It runs under grilly2's interpreter (never grilly
v1, which this repo's own interpreter imports), started by ``r2_qfhrr.score``:

    uv run --project <the grilly2 checkout beside this one> python validation/retrieval_bench/r2_qfhrr_score.py JOB OUT

JOB  {"table": the word table (.npz), "docs": {key: text}, "rows": [{"id", "cues": text, "items": [key, ...]}]}
OUT  {"device", "n_docs", "check": {...}, "wall_s", "scores": {row id: [score per item, in the row's order]}}

The codes are the serving word table's (``cubbyllm.worlds.fast_words``: each word's (k, l) block code, IDF-weighted
and bundled; no model). Every code is scaled to unit length, so a score is the cosine. The prototypes are uploaded
once; each row's candidates are gathered on the card (``index_select``) and scored against the question's code with
``grilly.vsa.similarity_batch``. The first rows are scored again in numpy, and the largest difference is reported.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__wiring__ = "STANDALONE"

CHECK = 50


def unit(code: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(code))
    return (code / n).astype(np.float32) if n > 0 else code.astype(np.float32)


def main() -> None:
    import grilly
    from grilly import vsa
    from cubbyllm.worlds.fast_words import FastWordEncoder
    job_path, out_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    job = json.load(open(job_path, encoding="utf-8"))
    enc = FastWordEncoder.from_npz(job["table"])
    keys = sorted(job["docs"])
    at = {q: i for i, q in enumerate(keys)}
    protos = np.stack([unit(enc.encode(job["docs"][q])) for q in keys]) if keys else np.zeros((0, enc.k, enc.l), np.float32)
    P = grilly.from_numpy(protos)
    print(f"{len(keys)} prototypes on {P.device} ({protos.nbytes / 1e9:.2f} GB), {time.time() - t0:.0f} s", flush=True)
    scores, worst = {}, 0.0
    for i, r in enumerate(job["rows"]):
        q = unit(enc.encode(r["cues"]))
        idx = np.array([at[x] for x in r["items"]], dtype=np.int64)
        got = vsa.similarity_batch(grilly.from_numpy(q), grilly.index_select(P, 0, grilly.from_numpy(idx))).numpy()
        got = got * enc.k                                       # NVSA's sum(a * b) / k, back to the cosine
        if i < CHECK:
            ref = protos[idx].reshape(len(idx), -1) @ q.reshape(-1)
            worst = max(worst, float(np.abs(ref - got).max()) if len(idx) else 0.0)
        scores[r["id"]] = [round(float(x), 6) for x in got]
        if i % 500 == 0:
            print(f"{i}/{len(job['rows'])} rows", flush=True)
    out = {"device": str(P.device), "interpreter": "grilly2", "n_docs": len(keys), "n_rows": len(job["rows"]),
           "check": {"rows": min(CHECK, len(job["rows"])), "max_abs_diff_vs_numpy": worst},
           "wall_s": round(time.time() - t0, 1), "scores": scores}
    json.dump(out, open(out_path, "w", encoding="utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "scores"}), flush=True)


if __name__ == "__main__":
    main()
