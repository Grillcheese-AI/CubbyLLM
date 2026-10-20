"""exp_k8_holo -- a holographic state in place of the KV cache (H-K8): the capacity law of a bound
superposition, and the bytes a token touches, against the KV cache and the episodic row store.

A KV cache is one row per token, re-read by every token: at length N the bytes a token touches
grow as N, and so does the energy. The hypothesis: beyond the window, keep one vector per chunk of
C tokens, the superposition of the chunk's bound (key, value) pairs with unitary keys,

    m_c = sum_t  u(k_t) (*) v_t          read:  m_c (*) u(q)^-1  =  v_t* + noise

and a summary key K_c = sum_t u(k_t)/sqrt(C) to pick the chunk (q . K_c ~ 1/sqrt(C) for a member,
~1/sqrt(d) for a stranger). The read is one Fourier-domain product; no softmax over rows. The
algebra fixes the price: with random unitary keys and unit Gaussian values every decoy's dot with
the read is N(0, C/d), the maximum over N decoys ~ sqrt(2 C ln N / d), so a nearest-value decode
over N candidates holds while C ln N < d/2: chunks of ~60 at d=1024 and N=4,096, ~120 at d=2048
(one vector alone, N = M: M ln M < d/2). One chunk vector per C tokens instead of C rows: ~C/2
times fewer bytes, the read O(d log d) and flat in N.

Measured (numpy, a minute):  1. the law -- hit rate against M pairs in ONE vector, d = 512..2048,
the 90% boundary against the solution of M ln M = d/2;  2. the chunk store -- N = 4,096
beyond-window pairs at d = 1024 (the grown model's width) and 2048, chunks of C, the query a
noised copy of a stored key, decode over all N values, chunk selection by summary key (top-1,
top-4 summed, and all chunks summed = no selection);  3. bytes resident and bytes per token at the
450M and the grown geometry: full attention, the window + row store we have, window + chunks.

Gates (pre-registered): G1 the measured 90% boundary within 25% of the law at each d;
G2 at d=1024, N=4096, C=32, noise sigma=0.5: top-1 decode >= 0.9 over all 4,096 values.
Kill: decode < 0.5 at C=16, d=1024 (then chunks cannot beat rows even 8-fold).

First read (2026-10-20, seed 0): the law's crossing is where the decode sits at 0.68-0.70 at every
d, not 0.90 -- G1 fails as written; 90% holds to about half the law (M = 31/55/97 at d =
512/1024/2048). G2 passes at the line: 0.90 (C=32, sigma 0.5, d=1024); C=16 is 1.00/1.00/0.97
over sigma 0/0.5/1.0, C=64 is 0.61/0.51/0.14; at d=2048 C=32 is 1.00/1.00/0.98 and C=64
0.97/0.95/0.64. Selection is the whole game: summing all chunks decodes nothing (0.00), top-4
summed is worse than top-1. The read is attenuated (cos 0.16-0.25 to the true value) though
decodable. Bytes per generated token at the grown geometry and N = 1M: full attention 262 GB, the
window + row store 2.2 GB, window + chunks 1.15 GB (resident 35 GB against 2.2 GB); at N = 4,096
the 64-layer window attention is the floor of all three (128 MB).

    python validation/exp_k8_holo.py [--out validation/logs/exp_k8_holo.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def unitary(x: np.ndarray) -> np.ndarray:
    """Phase-only copy of x (rows): |X_f| = 1, the same phases; the spatial link's ``unitary``."""
    X = np.fft.rfft(x, axis=-1)
    return np.fft.irfft(np.exp(1j * np.angle(X)), n=x.shape[-1], axis=-1)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1), n=a.shape[-1], axis=-1)


def unbind(m: np.ndarray, u: np.ndarray) -> np.ndarray:
    """m (*) u^-1 for unitary u: the conjugate spectrum."""
    return np.fft.irfft(np.fft.rfft(m, axis=-1) * np.conj(np.fft.rfft(u, axis=-1)), n=m.shape[-1], axis=-1)


def pairs(rng, n: int, d: int):
    keys = unitary(rng.standard_normal((n, d)))
    vals = rng.standard_normal((n, d)) / math.sqrt(d)
    return keys, vals


def noisy(rng, keys: np.ndarray, sigma: float) -> np.ndarray:
    """A query that is a stored key plus relative noise sigma, made unitary again."""
    if sigma == 0:
        return keys
    return unitary(keys + sigma * rng.standard_normal(keys.shape) / math.sqrt(keys.shape[-1]))


def law(d: int, n_log: float) -> float:
    """The C (or M) with C * n_log = d/2; n_log = ln N (or ln M for the one-vector law)."""
    return d / (2 * n_log)


def one_vector_boundary(d: int) -> float:
    """M with M ln M = d/2."""
    lo, hi = 1.0, float(d)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid * math.log(mid) < d / 2 else (lo, mid)
    return lo


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.einsum("id,id->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


def capacity(rng, d: int, grid: list, trials: int) -> dict:
    """ONE vector holding M pairs: decode hit rate (nearest of the M values) and cos(read, true)."""
    rows = []
    for M in grid:
        hit, snr = 0.0, 0.0
        for _ in range(trials):
            k, v = pairs(rng, M, d)
            m = bind(k, v).sum(0)
            read = unbind(m[None, :], k)                                      # (M, d): every key reads
            hit += float(((read @ v.T).argmax(1) == np.arange(M)).mean())
            snr += cosine(read, v)
        rows.append({"M": M, "hit": hit / trials, "cos": snr / trials})
    below = [i for i in range(len(rows)) if rows[i]["hit"] < 0.9]
    return {"d": d, "rows": rows, "law_M": round(one_vector_boundary(d), 1),
            "first_M_below_90": rows[below[0]]["M"] if below else None,
            "last_M_above_90": rows[below[0] - 1]["M"] if below and below[0] > 0 else (rows[-1]["M"] if not below else None)}


def chunk_store(rng, d: int, n: int, chunks: list, sigmas: list, queries: int) -> list:
    """N pairs in chunks of C; a query is a noised stored key; decode over ALL N values."""
    keys, vals = pairs(rng, n, d)
    out = []
    for C in chunks:
        nc = n // C
        kk, vv = keys[:nc * C].reshape(nc, C, d), vals[:nc * C].reshape(nc, C, d)
        content = bind(kk.reshape(-1, d), vv.reshape(-1, d)).reshape(nc, C, d).sum(1)    # (nc, d)
        summary = kk.sum(1) / math.sqrt(C)                                               # (nc, d)
        whole = content.sum(0)                                                           # no selection
        for sigma in sigmas:
            idx = rng.integers(0, nc * C, queries)
            q = noisy(rng, keys[idx], sigma)
            z = q @ summary.T                                                             # (Q, nc)
            order = np.argsort(-z, axis=1)
            pick = order[:, 0]
            read1 = unbind(content[pick], q)
            read4 = sum(unbind(content[order[:, j]], q) for j in range(min(4, nc)))
            read_all = unbind(np.broadcast_to(whole, (queries, d)), q)
            hit = lambda r: float(((r @ vals[:nc * C].T).argmax(1) == idx).mean())
            out.append({"d": d, "C": C, "sigma": sigma, "chunks": nc, "law_C": round(law(d, math.log(nc * C)), 1),
                        "select_ok": float((pick == idx // C).mean()), "hit_top1": hit(read1),
                        "cos_top1": cosine(read1, vals[idx]), "hit_top4": hit(read4), "hit_no_selection": hit(read_all)})
    return out


def bytes_table(geometries: dict, lengths: list, C: int = 32, b: int = 2) -> list:
    """Resident bytes and bytes touched per generated token: full attention; the window plus the
    row store (mem_key-dim keys, D-dim values, top-8 values read); the window plus chunks (a D-dim
    summary and a D-dim content per chunk, the summaries scanned, one content read)."""
    rows = []
    for name, g in geometries.items():
        D, L, W, ML, K = g["D"], g["L"], g["window"], g["mem_layers"], g["mem_key"]
        for N in lengths:
            full = {"resident": 2 * D * L * N * b, "per_token": 2 * D * L * N * b}
            win = 2 * D * L * W * b
            beyond = max(N - W, 0)
            store = {"resident": win + ML * beyond * (K + D) * b, "per_token": win + ML * (beyond * K + 8 * D) * b}
            nc = math.ceil(beyond / C)
            holo = {"resident": win + ML * nc * 2 * D * b, "per_token": win + ML * (nc * D + D) * b}
            rows.append({"model": name, "N": N, "C": C, "full_attention": full, "window_plus_rows": store, "window_plus_chunks": holo})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "exp_k8_holo.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    report = {"args": vars(args)}

    print("1. the law: M pairs in ONE vector, decode hit rate (nearest of M)      [law: M ln M = d/2]")
    report["capacity"] = []
    for d in (512, 1024, 2048):
        grid = sorted({int(round(one_vector_boundary(d) * f)) for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)})
        r = capacity(rng, d, grid, args.trials)
        report["capacity"].append(r)
        print(f"  d={d:5d}  law M*={r['law_M']:6.1f}  90% holds through M={r['last_M_above_90']}, "
              f"first below at M={r['first_M_below_90']}   " + "  ".join(f"{x['M']}:{x['hit']:.2f}" for x in r["rows"]))

    print("\n2. the chunk store: N=4,096 pairs, chunks of C, query = stored key + noise sigma; decode over all N   [law: C ln N = d/2]")
    print("      d     C  sigma  chunks  law C*  select  hit(top-1)  cos   hit(top-4)  hit(no selection)")
    report["chunks"] = []
    for d in (1024, 2048):
        report["chunks"] += chunk_store(rng, d, 4096, [16, 32, 64, 128], [0.0, 0.5, 1.0], args.queries)
    for r in report["chunks"]:
        print(f"  {r['d']:5d}  {r['C']:4d}  {r['sigma']:4.1f}  {r['chunks']:6d}  {r['law_C']:6.1f}  {r['select_ok']:6.2f}  "
              f"{r['hit_top1']:10.2f}  {r['cos_top1']:5.2f}  {r['hit_top4']:10.2f}  {r['hit_no_selection']:17.2f}")

    geos = {"base450m": {"D": 1024, "L": 32, "window": 512, "mem_layers": 8, "mem_key": 64},
            "grow_mem_grown": {"D": 1024, "L": 64, "window": 512, "mem_layers": 16, "mem_key": 64}}
    print("\n3. bytes per generated token (bf16): full attention | window + row store | window + chunk store (C=32)")
    report["bytes"] = bytes_table(geos, [4096, 32768, 262144, 1048576])
    for r in report["bytes"]:
        f, s, h = r["full_attention"]["per_token"], r["window_plus_rows"]["per_token"], r["window_plus_chunks"]["per_token"]
        print(f"  {r['model']:15s} N={r['N']:>9,}   {f / 2**20:10,.0f} MB   {s / 2**20:8,.1f} MB   {h / 2**20:8,.1f} MB"
              f"   (resident: rows {r['window_plus_rows']['resident'] / 2**20:8,.1f} MB, chunks {r['window_plus_chunks']['resident'] / 2**20:7,.1f} MB)")
    report["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}  ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
