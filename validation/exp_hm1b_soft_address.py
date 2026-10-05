"""H-M1b screen: SOFT addressing for the qFHRR n-gram memory -- does a multi-probe / product-key read keep the
meaning that exact g-block buckets throw away?

H-M1 (exp_hm1_ngram_hv.py) found the meaning lives in the teacher part (blend 0: a semantic neighbour shares a
block phase 24% of the time, an entity sibling 1.8%, random 0.8%), but an exact g = 3 bucket needs all three
phases to agree, so a neighbour swap lands in the original bucket only 3.9% of the time.

Here the reader probes instead of hashing. The key of a trigram (a, b, c) is (rot2(a) + rot1(b) + c) mod l per
block; per block the reader takes each word's top-t slots (with their block values), scores every phase the three
can sum to (score = the best sum of values that makes it), keeps the top-m phases, and across the g blocks of a
table ranks the m^g buckets by summed score (product-key read). The writer stores at the exact top-1 key.

Query = a corpus trigram with ONE word swapped (random eligible position):
  self (sanity: must hit at K = 1), paraphrase (big -> large), semantic neighbour (the teacher's nearest word),
  entity sibling (france -> germany: a DIFFERENT fact, must NOT hit), random word.
Reported per class: p(top-1) and p(in top-m) per block, then hit@K = the original trigram's bucket is among the
reader's top-K probes of a table (mean over 4 tables) and 'any of 4 tables', for g = 2 and g = 3.
Wanted: paraphrase / neighbour high at small K; entity and random near the uniform K / l^g. CPU, numpy, no model.

    python validation/exp_hm1b_soft_address.py --table <fastword_table_v4.npz> --shard <wiki_full.u32>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hm1_ngram_hv import ENTITY_SETS, HERE, PARAPHRASE, ROOT, Table, corpus_trigrams  # noqa: E402

LOG: list[str] = []
KS = (1, 4, 16, 64)
TABLES = 4


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def top_slots(tab: Table, blend: float, t: int, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """(n, k, t) each word's top-t slots per block (best first) and their values, at signal blend b."""
    n = len(tab.words)
    idx = np.empty((n, tab.k, t), dtype=np.int64)
    val = np.empty((n, tab.k, t), dtype=np.float32)
    for s in range(0, n, batch):
        part = np.arange(s, min(n, s + batch))
        X = (1.0 - blend) * tab.proj(part)
        r = np.arange(len(part))[:, None]
        X[r, np.arange(tab.k)[None, :], tab.sig[part]] += blend / np.sqrt(tab.k)
        I = np.argpartition(-X, t - 1, axis=-1)[..., :t]
        V = np.take_along_axis(X, I, -1)
        o = np.argsort(-V, axis=-1)
        idx[part] = np.take_along_axis(I, o, -1)
        val[part] = np.take_along_axis(V, o, -1)
    return idx, val


def exact_key(C: np.ndarray, tri: np.ndarray, l: int) -> np.ndarray:
    """(Q, k) the writer's key: top-1 phases bound as (rot2(a) + rot1(b) + c) mod l."""
    return (np.roll(C[tri[:, 0]], 2, axis=1) + np.roll(C[tri[:, 1]], 1, axis=1) + C[tri[:, 2]]) % l


def soft_read(idx: np.ndarray, val: np.ndarray, tri: np.ndarray, l: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """(Q, k, m) the reader's top-m key phases per block and their scores (best sum of the words' slot values)."""
    Ia, Va = np.roll(idx[tri[:, 0]], 2, axis=1), np.roll(val[tri[:, 0]], 2, axis=1)
    Ib, Vb = np.roll(idx[tri[:, 1]], 1, axis=1), np.roll(val[tri[:, 1]], 1, axis=1)
    Ic, Vc = idx[tri[:, 2]], val[tri[:, 2]]
    P = (Ia[..., :, None, None] + Ib[..., None, :, None] + Ic[..., None, None, :]) % l
    S = Va[..., :, None, None] + Vb[..., None, :, None] + Vc[..., None, None, :]
    Q, k = P.shape[:2]
    P, S = P.reshape(Q, k, -1), S.reshape(Q, k, -1)
    D = np.full((Q, k, l), -np.inf, dtype=np.float32)
    qi = np.broadcast_to(np.arange(Q)[:, None, None], P.shape)
    ki = np.broadcast_to(np.arange(k)[None, :, None], P.shape)
    np.maximum.at(D, (qi, ki, P), S)
    I = np.argpartition(-D, m - 1, axis=-1)[..., :m]
    return I, np.take_along_axis(D, I, -1)


def hits(I: np.ndarray, V: np.ndarray, K0: np.ndarray, g: int) -> np.ndarray:
    """(len(KS), TABLES, Q) bool: the original bucket is among the reader's top-K of table T (product-key rank)."""
    Q = I.shape[0]
    out = np.zeros((len(KS), TABLES, Q), dtype=bool)
    for T in range(TABLES):
        present = np.ones(Q, dtype=bool)
        so = np.zeros(Q, dtype=np.float32)
        sums = np.zeros((Q, 1), dtype=np.float32)
        for j in range(T * g, T * g + g):
            match = I[:, j, :] == K0[:, j, None]
            present &= match.any(1)
            so = so + np.where(match, V[:, j, :], 0.0).sum(1).astype(np.float32)
            sums = (sums[:, :, None] + V[:, j, None, :]).reshape(Q, -1)
        rank = (sums > so[:, None]).sum(1)
        for i, K in enumerate(KS):
            out[i, T] = present & (rank < K)
    return out


def nearest(tab: Table, words: np.ndarray, batch: int = 256) -> dict[int, int]:
    """The teacher's nearest word (cosine of the proj part, blend 0) for each id in `words`."""
    P = tab.proj(np.arange(len(tab.words))).reshape(len(tab.words), -1)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    out = {}
    for s in range(0, len(words), batch):
        q = words[s:s + batch]
        S = P[q] @ P.T
        S[np.arange(len(q)), q] = -np.inf
        out.update(zip(q.tolist(), S.argmax(1).tolist()))
    del P
    return out


def make_queries(tab: Table, tri: np.ndarray, rng: np.random.Generator, n: int) -> dict:
    """class -> (original trigrams, query trigrams) with one word swapped at a random eligible position."""
    V = len(tab.words)
    content = np.array([w.isalpha() and len(w) >= 3 for w in tab.words])
    para = {}
    for x, y in PARAPHRASE:
        if x in tab.index and y in tab.index:
            para[tab.index[x]], para[tab.index[y]] = tab.index[y], tab.index[x]
    sib = {}
    for ws in ENTITY_SETS.values():
        ids = [tab.index[w] for w in ws if w in tab.index]
        for i in ids:
            sib[i] = [j for j in ids if j != i]

    def positions(eligible: np.ndarray):
        mask = eligible[tri]
        rows = np.flatnonzero(mask.any(1))
        rows = rng.choice(rows, size=min(n, len(rows)), replace=False)
        pos = np.array([rng.choice(np.flatnonzero(mask[r])) for r in rows])
        return tri[rows], pos

    def swapped(orig, pos, fn):
        new = orig.copy()
        new[np.arange(len(orig)), pos] = [fn(int(w)) for w in orig[np.arange(len(orig)), pos]]
        return orig, new

    out = {}
    pick = tri[rng.choice(len(tri), size=min(n, len(tri)), replace=False)]
    out["self (sanity)"] = (pick, pick.copy())
    is_para = np.zeros(V, dtype=bool); is_para[list(para)] = True
    out["paraphrase"] = swapped(*positions(is_para), lambda w: para[w])
    orig, pos = positions(content)
    nn = nearest(tab, np.unique(orig[np.arange(len(orig)), pos]))
    out["semantic neighbour"] = swapped(orig, pos, lambda w: nn[w])
    is_ent = np.zeros(V, dtype=bool); is_ent[list(sib)] = True
    out["entity (different fact)"] = swapped(*positions(is_ent), lambda w: int(rng.choice(sib[w])))
    pool = np.flatnonzero(content)
    out["random word"] = swapped(*positions(content), lambda w: int(rng.choice(pool)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True, help="fastword_table_v4.npz (off-repo)")
    ap.add_argument("--shard", required=True, help="a u32 token shard (wiki_full.u32)")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "grillcheese_bbpe128k.json"))
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--n", type=int, default=4000, help="queries per class")
    ap.add_argument("--t", type=int, default=3, help="slots per word per block the reader combines")
    ap.add_argument("--m", type=int, default=8, help="key phases kept per block")
    ap.add_argument("--blends", default="0,0.1")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tables", type=int, default=4, help="tables (block groups) read per query")
    ap.add_argument("--tag", default="v4")
    a = ap.parse_args(argv)
    global TABLES
    TABLES = a.tables
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tab = Table(a.table)
    l = tab.l
    log(f"hm1b | {os.path.basename(a.table)}: {len(tab.words)} words, k={tab.k} l={l}, teacher "
        f"{tab.meta.get('teacher', '?')} | reader: top-{a.t} slots per word, top-{a.m} key phases per block, "
        f"{TABLES} tables, K in {KS} ({time.time() - t0:.0f}s)")
    tri = corpus_trigrams(a.shard, a.tokenizer, tab, a.chunks, a.chunk)
    queries = make_queries(tab, tri, rng, a.n)
    log(f"  corpus: {len(tri)} distinct trigrams | queries: " +
        ", ".join(f"{c} {len(o)}" for c, (o, _) in queries.items()) + f" ({time.time() - t0:.0f}s)")
    report = {}
    for b in [float(x) for x in a.blends.split(",")]:
        idx, val = top_slots(tab, b, a.t)
        log(f"\nblend {b:.2f} ({time.time() - t0:.0f}s)")
        log(f"  {'query class':<24} {'n':>5} {'p(top-1)':>9} {f'p(top-{a.m})':>9} |"
            + "".join(f" {f'hit@{K}':>8}" for K in KS) + f" {f'any{TABLES}@16':>8}   (per table, mean of {TABLES})")
        rows = {}
        for cls, (orig, new) in queries.items():
            K0 = exact_key(idx[..., 0], orig, l)
            p1, pm, H = [], [], {2: [], 3: []}
            for s in range(0, len(orig), a.batch):
                I, Vv = soft_read(idx, val, new[s:s + a.batch], l, a.m)
                k0 = K0[s:s + a.batch]
                top1 = np.take_along_axis(I, Vv.argmax(-1)[..., None], -1)[..., 0]
                p1.append((top1 == k0).mean(1)); pm.append((I == k0[..., None]).any(-1).mean(1))
                for g in (2, 3):
                    H[g].append(hits(I, Vv, k0, g))
            p1, pm = float(np.concatenate(p1).mean()), float(np.concatenate(pm).mean())
            rows[cls] = {"n": len(orig), "p_top1": p1, f"p_top{a.m}": pm}
            for g in (2, 3):
                h = np.concatenate(H[g], axis=2)
                per = h.mean(axis=(1, 2)); any4 = h.any(axis=1).mean(axis=1)
                rows[cls][f"g{g}"] = {"hit": dict(zip(map(str, KS), per.tolist())),
                                      "any4": dict(zip(map(str, KS), any4.tolist()))}
                lead = f"  {cls:<24} {len(orig):>5} {p1:>9.4f} {pm:>9.4f} |" if g == 2 else f"  {'':<24} {'':>5} {'':>9} {'':>9} |"
                log(lead + "".join(f" {x:>8.4f}" for x in per) + f" {any4[KS.index(16)]:>8.4f}   g={g}")
        for g in (2, 3):
            log(f"  {'(uniform hash)':<24} {'':>5} {1 / l:>9.4f} {a.m / l:>9.4f} |"
                + "".join(f" {min(1.0, K / l ** g):>8.5f}" for K in KS) + f" {'':>8}   g={g}")
        report[str(b)] = rows
    out = os.path.join(HERE, "logs", f"exp_hm1b_soft_address_{a.tag}")
    json.dump({"args": vars(a), "ks": KS, "report": report}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\n  -> {os.path.relpath(out, ROOT)}.{{log,json}} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
