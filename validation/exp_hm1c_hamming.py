"""H-M1c: the Rubik's-cube encoding for the n-gram memory -- role-filler binding into bitpacked hypervectors and a
nearest-neighbour Hamming scan (grilly v1 cubemind/cube.h + cache.h), against the bucket addressing of H-M1/H-M1b.

cube.h encodes a cube state as sign(sum_i role_i * filler_i): role = the facelet position (BLAKE3 random), filler = the
colour (random, "red is not more similar to blue"). Here a trigram (a, b, c) is
    key = majority(role_1 XOR f_a, role_2 XOR f_b, role_3 XOR f_c)          (10,240 bits, bitpacked)
with role_p a seeded random bit vector per position and f_w the word's filler bits: sign((1 - b) * proj_w + b * signal_w)
from the fastword table -- at blend 0 the sign of the teacher part (a SimHash of the teacher vector, so Hamming tracks
the teacher's angle); at blend 0.5 the shipped vector, whose signal slot makes it close to a random code (the cube's own
choice). Recall = the nearest stored key by Hamming distance (XOR + popcount over every key), no buckets.

Same query classes as exp_hm1b (one word of a corpus trigram swapped): self, paraphrase, semantic neighbour, entity
sibling (a DIFFERENT fact: a hit is a leak), random word. Reported: the original trigram's rank among ALL stored
trigrams (hit@1, hit@16), the Hamming distance to it, and the scan time per query (CPU numpy here). CPU, no model.

    python validation/exp_hm1c_hamming.py --table <fastword_table.npz> --shard <wiki_full.u32>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hm1_ngram_hv import HERE, ROOT, Table, corpus_trigrams  # noqa: E402
from exp_hm1b_soft_address import make_queries  # noqa: E402

LOG: list[str] = []
KS = (1, 16)
TAUS = (0.05, 0.10, 0.15, 0.20)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def filler_bits(tab: Table, blend: float, batch: int = 4096) -> np.ndarray:
    """(n, D/64) uint64: each word's filler, sign((1 - b) proj + b signal), bitpacked."""
    n, D = len(tab.words), tab.k * tab.l
    out = np.empty((n, D // 64), dtype=np.uint64)
    for s in range(0, n, batch):
        part = np.arange(s, min(n, s + batch))
        X = (1.0 - blend) * tab.proj(part)
        r = np.arange(len(part))[:, None]
        X[r, np.arange(tab.k)[None, :], tab.sig[part]] += blend / np.sqrt(tab.k)
        bits = (X.reshape(len(part), D) > 0)
        out[s:s + len(part)] = np.packbits(bits, axis=1, bitorder="little").view(np.uint64)
    return out


def keys(F: np.ndarray, roles: np.ndarray, tri: np.ndarray, batch: int = 65536) -> np.ndarray:
    """(N, W) uint64 trigram keys: bitwise majority of the three role-bound fillers."""
    out = np.empty((len(tri), F.shape[1]), dtype=np.uint64)
    for s in range(0, len(tri), batch):
        t = tri[s:s + batch]
        x, y, z = F[t[:, 0]] ^ roles[0], F[t[:, 1]] ^ roles[1], F[t[:, 2]] ^ roles[2]
        out[s:s + len(t)] = (x & y) | (x & z) | (y & z)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "grillcheese_bbpe128k.json"))
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--n", type=int, default=500, help="queries per class (each a full scan of every stored key)")
    ap.add_argument("--blends", default="0,0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v4")
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tab = Table(a.table)
    D = tab.k * tab.l
    tri = corpus_trigrams(a.shard, a.tokenizer, tab, a.chunks, a.chunk)
    queries = make_queries(tab, tri, rng, a.n)
    row_of = {tuple(t): i for i, t in enumerate(tri.tolist())}
    roles = np.random.default_rng(1234).integers(0, 2 ** 63, size=(3, D // 64), dtype=np.uint64) * 2 + 1
    log(f"hm1c | {os.path.basename(a.table)} (teacher {tab.meta.get('teacher', '?')}): {len(tab.words)} words, D={D} | "
        f"{len(tri)} stored trigrams ({len(tri) * D // 8 / 1e6:.0f} MB bitpacked) | {a.n} queries per class ({time.time() - t0:.0f}s)")
    report = {}
    for b in [float(x) for x in a.blends.split(",")]:
        F = filler_bits(tab, b)
        K = keys(F, roles, tri)
        log(f"\nblend {b:.2f}: role-filler keys built ({time.time() - t0:.0f}s)")
        log(f"  {'query class':<24} {'n':>4} " + " ".join(f"{f'hit@{k}':>7}" for k in KS) +
            f" {'d(orig)/D':>10} {'d(nearest other)/D':>19} {'ms/query':>9}")
        rows = {}
        for cls, (orig, new) in queries.items():
            Q = keys(F, roles, new)
            ranks, d_orig, d_other, times = [], [], [], []
            for qi in range(len(Q)):
                t1 = time.perf_counter()
                d = np.bitwise_count(K ^ Q[qi]).sum(1, dtype=np.int32)
                times.append(time.perf_counter() - t1)
                o = row_of[tuple(orig[qi].tolist())]
                do = d[o]
                ranks.append(int((d < do).sum()))
                d[o] = D
                d_orig.append(do / D); d_other.append(d.min() / D)
            ranks = np.array(ranks); d_orig = np.array(d_orig)
            hits = [float((ranks < k).mean()) for k in KS]
            acc = {t: float(((ranks == 0) & (d_orig < t)).mean()) for t in TAUS}
            rows[cls] = {"n": len(Q), **{f"hit@{k}": h for k, h in zip(KS, hits)}, "d_orig": float(np.mean(d_orig)),
                         "d_nearest_other": float(np.mean(d_other)), "ms_per_query": 1000 * float(np.median(times)),
                         "accepted_top1_under": {str(t): v for t, v in acc.items()},
                         "d_orig_quartiles": np.quantile(d_orig, [0.25, 0.5, 0.75]).tolist()}
            log(f"  {cls:<24} {len(Q):>4} " + " ".join(f"{h:>7.4f}" for h in hits) +
                f" {np.mean(d_orig):>10.4f} {np.mean(d_other):>19.4f} {1000 * np.median(times):>9.1f}   |"
                + "".join(f" {acc[t]:>6.3f}" for t in TAUS))
        log(f"  {'':<24} {'':>4} {'':>7} {'':>7} {'':>10} {'':>19} {'':>9}   | accepted (top-1 AND d < tau), "
            "tau = " + ", ".join(f"{t:.2f}" for t in TAUS))
        report[str(b)] = rows
    out = os.path.join(HERE, "logs", f"exp_hm1c_hamming_{a.tag}")
    json.dump({"args": vars(a), "report": report}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\n  -> {os.path.relpath(out, ROOT)}.{{log,json}} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
