"""r3_qfhrr_score -- order 0046, R3's qFHRR arm, the similarity half. It runs under grilly2's interpreter (never grilly
v1, which this repo's own interpreter imports), started by ``r3_qfhrr.score``:

    uv run --project <the grilly2 checkout beside this one> python validation/retrieval_bench/r3_qfhrr_score.py JOB OUT

JOB  {"table": the word table (.npz), "docs": {"P17|en": text}, "rows": [{"id", "lang", "cues": text}], "keep": n}
OUT  {"device", "n_docs", "check": {...}, "wall_s", "ranked": {row id: [[property, score], ...] best first}}

Each code is the serving word table's (``cubbyllm.worlds.fast_words``), scaled to unit length, so a score is the cosine.
One language's codebook is uploaded at a time and every row of that language is scored against all of it with
``grilly.vsa.similarity_batch``; the first rows are scored again in numpy and the largest difference is reported.

Wired: STANDALONE (R3's bench).
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

CHECK = 20


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
    keep = int(job.get("keep", 20))
    ranked, worst, checked, device, n_docs = {}, 0.0, 0, None, 0
    for lang in sorted({r["lang"] for r in job["rows"]}):
        keys = sorted(k for k in job["docs"] if k.endswith("|" + lang))
        names = [k.split("|")[0] for k in keys]
        protos = np.stack([unit(enc.encode(job["docs"][k])) for k in keys])
        P = grilly.from_numpy(protos)
        device, n_docs = str(P.device), n_docs + len(keys)
        print(f"{lang}: {len(keys)} codes on {P.device} ({protos.nbytes / 1e9:.2f} GB), {time.time() - t0:.0f} s", flush=True)
        rows = [r for r in job["rows"] if r["lang"] == lang]
        for i, r in enumerate(rows):
            q = unit(enc.encode(r["cues"]))
            got = vsa.similarity_batch(grilly.from_numpy(q), P).numpy() * enc.k     # NVSA's sum(a * b) / k, back to the cosine
            if checked < CHECK:
                ref = protos.reshape(len(keys), -1) @ q.reshape(-1)
                worst, checked = max(worst, float(np.abs(ref - got).max())), checked + 1
            best = np.argsort(-got, kind="stable")[:keep]
            ranked[r["id"]] = [[names[j], round(float(got[j]), 6)] for j in best]
            if i % 300 == 0:
                print(f"{lang} {i}/{len(rows)} rows", flush=True)
        del P
    out = {"device": device, "interpreter": "grilly2", "n_docs": n_docs, "n_rows": len(job["rows"]),
           "check": {"rows": checked, "max_abs_diff_vs_numpy": worst}, "wall_s": round(time.time() - t0, 1),
           "ranked": ranked}
    json.dump(out, open(out_path, "w", encoding="utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "ranked"}), flush=True)


if __name__ == "__main__":
    main()
