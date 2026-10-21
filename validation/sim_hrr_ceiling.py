"""The far-facts read's ceiling: an FHRR scene with random codes, decoded by an ideal cleanup.

H-K10, order 0001. ``n`` items bound to random unitary keys in ``d`` dims (exact FHRR, Fourier
domain); unbind one; argmax of the dot product over ``V`` candidates (the target and V - 1 others).
That is the best the read can do: quasi-orthogonal codes and a perfect decoder. Learned, correlated
embeddings and a trunk that has to learn the decoding can only do worse. With ``--cache`` it also
counts what the real shards write: items in a document's scene, and the distinct written tokens
(the candidate set a decoder must choose among).

  python validation/sim_hrr_ceiling.py --cache <token cache> --out validation/logs/sim_hrr_ceiling_<date>.json
"""
import argparse
import json
import os
import time

import numpy as np


def ceiling(n: int, V: int, d: int, trials: int, rng) -> float:
    E = rng.standard_normal((V, d)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    hits = 0
    for _ in range(trials):
        idx = rng.choice(V, size=n, replace=False)              # the stored items; idx[0] is asked for
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (n, d // 2 + 1)))
        ph[:, 0] = ph[:, -1] = 1                                # real keys: DC and Nyquist stay real
        scene = (np.fft.rfft(E[idx], axis=1) * ph).sum(0)
        r = np.fft.irfft(scene * np.conj(ph[0]), n=d)           # unbind the asked address
        hits += int(np.argmax(E @ r.astype(np.float32)) == idx[0])
    return hits / trials


def shard_stats(cache: str, name: str) -> dict:
    doc = json.load(open(os.path.join(cache, "_align.json"), encoding="utf-8"))[name]
    r = np.fromfile(os.path.join(cache, f"{name}.role.u8"), dtype=np.uint8)
    ids = np.fromfile(os.path.join(cache, f"{name}.u32"), dtype=np.uint32)
    n = len(r) // doc
    w = (r[:n * doc].reshape(n, doc) & 1).astype(bool)         # WRITE = 1: into the scene
    per = w.sum(1)
    return {"documents": n, "doc": doc, "writes_per_doc_mean": round(float(per.mean()), 1),
            "writes_per_doc_p90": int(np.percentile(per, 90)), "writes_per_doc_max": int(per.max()),
            "distinct_written_tokens": int(len(np.unique(ids[:n * doc].reshape(n, doc)[w])))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--items", default="37,19,9", help="items in the scene (37 = facts_far1k_lead's mean; /2, /4)")
    ap.add_argument("--candidates", default="40,131072", help="40 = the villages' written set; 131,072 = the vocabulary")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="", help="token cache: also count the shards' writes")
    ap.add_argument("--shards", default="facts_far1k_lead,spatial_here")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    out = {"args": vars(a), "top1": {}}
    for n in map(int, a.items.split(",")):
        for V in map(int, a.candidates.split(",")):
            p = ceiling(n, V, a.d, a.trials if V > 1000 else 5 * a.trials, rng)
            out["top1"][f"items{n}_candidates{V}"] = p
            print(f"d {a.d}, {n:3d} items, best of {V:7,d} candidates: top-1 {p:.3f}", flush=True)
    if a.cache:
        out["shards"] = {s: shard_stats(a.cache, s) for s in a.shards.split(",")}
        for s, v in out["shards"].items():
            print(s, v, flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
