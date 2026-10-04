"""H-M1 screen: can bound fastword codes ADDRESS an n-gram memory (Qwen4's RAM-resident n-gram table, rebuilt
on our qFHRR word table), and does the address keep meaning?

The n-gram key is built from the compact (integer-phase) form of the table: each word = 80 phases (per-block
argmax of its blended vector, l = 128), and a trigram key = (rot2(a) + rot1(b) + c) mod 128 per block, rot =
block-order rotation -- qFHRR binding with position by permutation. A memory table is addressed by the phases of
a group of g blocks (g = 3 -> 128^3 = 2.1M buckets); several groups = several tables (multi-head, Engram-style).

Because the key is a sum, swapping ONE word changes the key exactly where that word's phases change: the
locality of the address is the per-block phase agreement of the two words. So the screen is:

  1. per word pair, by class: paraphrase (big/large), same-type entity (france/germany -- a DIFFERENT fact),
     semantic nearest neighbour, random -- the per-block agreement p and the same-bucket rate for g = 1, 2, 3
  2. load: distinct corpus trigrams into g = 3 buckets -- collisions against a uniform hash's expectation

at signal blends b in {0, 0.1, 0.25, 0.5, 0.75}. The stored table is 0.5 * proj + 0.5 * signal (both unit, not
renormalized: mowm semantic_words.FastWordEncoder.build), so proj = 2v - signal and any blend is rebuilt exactly.
The wanted profile: paraphrase high, entity low, random ~ 1/128^g. CPU, numpy, no model.

    python validation/exp_hm1_ngram_hv.py --table <fastword_table_v4.npz> --shard <wiki_full.u32>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG: list[str] = []

PARAPHRASE = [("big", "large"), ("small", "little"), ("buy", "purchase"), ("start", "begin"), ("end", "finish"),
              ("quick", "fast"), ("car", "automobile"), ("help", "assist"), ("show", "display"), ("job", "work"),
              ("house", "home"), ("child", "kid"), ("ill", "sick"), ("near", "close"), ("rich", "wealthy"),
              ("smart", "clever"), ("angry", "mad"), ("answer", "reply"), ("famous", "renowned"), ("hard", "difficult"),
              ("happy", "glad"), ("shop", "store"), ("road", "street"), ("movie", "film"), ("couch", "sofa"),
              ("gift", "present"), ("speak", "talk"), ("rule", "regulation"), ("error", "mistake"), ("total", "sum"),
              ("city", "town"), ("attempt", "try"), ("enough", "sufficient"), ("often", "frequently"), ("whole", "entire")]
ENTITY_SETS = {
    "country": ["france", "germany", "italy", "spain", "canada", "japan", "china", "india", "brazil", "mexico",
                "egypt", "russia", "sweden", "norway", "poland", "greece", "turkey", "kenya", "chile", "peru"],
    "city": ["paris", "london", "berlin", "rome", "madrid", "tokyo", "moscow", "cairo", "toronto", "chicago",
             "boston", "vienna", "dublin", "lisbon", "athens", "prague", "sydney", "montreal", "quebec", "seattle"],
    "month": ["january", "february", "march", "april", "june", "july", "august", "september", "october",
              "november", "december"],
    "number": ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty", "hundred"],
    "element": ["oxygen", "hydrogen", "carbon", "nitrogen", "iron", "copper", "gold", "silver", "zinc", "lead"],
}
BLENDS = (0.0, 0.1, 0.25, 0.5, 0.75)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def signal_idx(word: str, k: int, l: int) -> np.ndarray:
    """The hot slot per block of mowm's signal_code (same BLAKE2b seeding, bit for bit)."""
    digest = hashlib.blake2b(f"w:{word}".encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**63)
    return np.random.default_rng(seed).integers(0, l, size=k)


class Table:
    """The fastword table, with the blend undone: proj (the teacher part) and the signal slots, apart."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.k, self.l = int(z["k"]), int(z["l"])
        self.words = [str(w) for w in z["words"]]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.v = z["vectors"]                                        # (n, k*l) f16 = 0.5 proj + 0.5 signal
        self.sig = np.stack([signal_idx(w, self.k, self.l) for w in self.words])   # (n, k)
        self.meta = json.loads(str(z["meta"]))

    def proj(self, ids: np.ndarray) -> np.ndarray:
        """(n, k, l) teacher part: proj = 2v - signal (exact: both parts unit, the blend not renormalized)."""
        V = 2.0 * self.v[ids].astype(np.float32).reshape(len(ids), self.k, self.l)
        r = np.arange(len(ids))[:, None]
        V[r, np.arange(self.k)[None, :], self.sig[ids]] -= 1.0 / np.sqrt(self.k)
        return V

    def compact(self, ids: np.ndarray, blend: float, batch: int = 4096) -> np.ndarray:
        """(n, k) phases: per-block argmax of (1 - b) proj + b signal."""
        out = np.empty((len(ids), self.k), dtype=np.int64)
        for s in range(0, len(ids), batch):
            part = ids[s:s + batch]
            X = (1.0 - blend) * self.proj(part)
            r = np.arange(len(part))[:, None]
            X[r, np.arange(self.k)[None, :], self.sig[part]] += blend / np.sqrt(self.k)
            out[s:s + batch] = X.argmax(-1)
        return out


def group_rates(a: np.ndarray, b: np.ndarray, g: int, tables: int = 4) -> tuple[float, float]:
    """Same-bucket rate for g-block groups (mean over all groups), and 'any of the first `tables` groups'."""
    eq = (a == b)
    n_groups = eq.shape[1] // g
    G = eq[:, :n_groups * g].reshape(len(eq), n_groups, g).all(-1)
    return float(G.mean()), float(G[:, :tables].any(-1).mean())


def pairs_by_class(tab: Table, rng: np.random.Generator, n_nn: int = 2000):
    have = lambda w: w in tab.index
    out = {"paraphrase": [(a, b) for a, b in PARAPHRASE if have(a) and have(b)]}
    ent = []
    for ws in ENTITY_SETS.values():
        ws = [w for w in ws if have(w)]
        ent += [(ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))]
    out["entity (different fact)"] = ent
    pool = np.array([i for i, w in enumerate(tab.words) if w.isalpha() and len(w) >= 3])
    q = rng.choice(pool, size=min(n_nn, len(pool)), replace=False)
    P = tab.proj(np.arange(len(tab.words))).reshape(len(tab.words), -1)
    Q = P[q]
    nn = []
    for s in range(0, len(q), 256):
        S = Q[s:s + 256] @ P.T
        S[np.arange(len(S)), q[s:s + 256]] = -np.inf
        nn += list(S.argmax(1))
    out["semantic neighbour"] = [(tab.words[i], tab.words[j]) for i, j in zip(q, nn)]
    rnd = rng.choice(pool, size=(n_nn, 2))
    out["random"] = [(tab.words[i], tab.words[j]) for i, j in rnd if i != j]
    del P
    return out


def corpus_trigrams(shard: str, tokenizer: str, tab: Table, n_chunks: int, chunk: int) -> np.ndarray:
    """Distinct in-vocabulary word trigrams from evenly spaced token chunks of a u32 shard, as word ids."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer)
    data = np.memmap(shard, dtype=np.uint32, mode="r")
    starts = np.linspace(0, len(data) - chunk - 1, n_chunks).astype(np.int64)
    seen = set()
    for s in starts:
        words = re.findall(r"[a-z0-9]+", tk.decode(data[s:s + chunk].tolist()).lower())
        ids = [tab.index.get(w, -1) for w in words]
        for i in range(len(ids) - 2):
            if ids[i] >= 0 and ids[i + 1] >= 0 and ids[i + 2] >= 0:
                seen.add((ids[i], ids[i + 1], ids[i + 2]))
    return np.array(sorted(seen), dtype=np.int64)


def trigram_keys(C: np.ndarray, tri: np.ndarray, l: int) -> np.ndarray:
    """(rot2(a) + rot1(b) + c) mod l per block: qFHRR binding, position by block-order rotation."""
    return (np.roll(C[tri[:, 0]], 2, axis=1) + np.roll(C[tri[:, 1]], 1, axis=1) + C[tri[:, 2]]) % l


def load_stats(K: np.ndarray, l: int, g: int = 3, tables: int = 4) -> dict:
    n = len(K)
    B = l ** g
    shared, maxload = [], []
    for t in range(tables):
        cols = K[:, t * g:(t + 1) * g]
        bucket = sum(cols[:, j] * l ** (g - 1 - j) for j in range(g))
        _, inv, cnt = np.unique(bucket, return_inverse=True, return_counts=True)
        shared.append(float((cnt[inv] > 1).mean()))
        maxload.append(int(cnt.max()))
    expected = 1.0 - (1.0 - 1.0 / B) ** (n - 1)
    full_dup = 1.0 - len(np.unique(K, axis=0)) / n
    return {"n": n, "buckets": B, "shared": float(np.mean(shared)), "expected_uniform": expected,
            "max_load": int(max(maxload)), "full_key_duplicates": full_dup}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True, help="fastword_table_v4.npz (off-repo)")
    ap.add_argument("--shard", required=True, help="a u32 token shard (wiki_full.u32)")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "grillcheese_bbpe128k.json"))
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v4")
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tab = Table(a.table)
    log(f"hm1 | {os.path.basename(a.table)}: {len(tab.words)} words, k={tab.k} l={tab.l}, "
        f"teacher {tab.meta.get('teacher', '?')}, stored blend {tab.meta.get('signal_blend', '?')} ({time.time() - t0:.0f}s)")
    pairs = pairs_by_class(tab, rng)
    log("  pairs: " + ", ".join(f"{c} {len(p)}" for c, p in pairs.items()))
    tri = corpus_trigrams(a.shard, a.tokenizer, tab, a.chunks, a.chunk)
    log(f"  corpus: {len(tri)} distinct in-vocabulary trigrams from {a.chunks} x {a.chunk} tokens ({time.time() - t0:.0f}s)")
    report = {"pairs": {}, "load": {}}
    for b in BLENDS:
        C = tab.compact(np.arange(len(tab.words)), b)
        sig_share = float((C == tab.sig).mean())
        log(f"\nblend {b:.2f}: the phase is the word's signal slot in {sig_share:.3f} of blocks")
        log(f"  {'pair class':<26} {'n':>5} {'p(block)':>9} {'g=1':>7} {'g=2':>8} {'g=3':>9} {'g=3, any of 4':>14}")
        rows = {}
        for cls, ps in pairs.items():
            ia = np.array([tab.index[x] for x, _ in ps]); ib = np.array([tab.index[y] for _, y in ps])
            p = float((C[ia] == C[ib]).mean())
            r1, _ = group_rates(C[ia], C[ib], 1); r2, _ = group_rates(C[ia], C[ib], 2); r3, any3 = group_rates(C[ia], C[ib], 3)
            rows[cls] = {"n": len(ps), "p": p, "g1": r1, "g2": r2, "g3": r3, "g3_any4": any3}
            log(f"  {cls:<26} {len(ps):>5} {p:>9.4f} {r1:>7.4f} {r2:>8.5f} {r3:>9.6f} {any3:>14.4f}")
        log(f"  {'(uniform hash)':<26} {'':>5} {1 / tab.l:>9.4f} {1 / tab.l:>7.4f} {1 / tab.l ** 2:>8.5f} {1 / tab.l ** 3:>9.6f}")
        ld = load_stats(trigram_keys(C, tri, tab.l), tab.l)
        log(f"  load, g=3 ({ld['buckets']:,} buckets, {ld['n']:,} trigrams): {ld['shared']:.4f} share a bucket "
            f"(uniform hash {ld['expected_uniform']:.4f}), max load {ld['max_load']}, full-key duplicates {ld['full_key_duplicates']:.5f}")
        report["pairs"][str(b)] = rows
        report["load"][str(b)] = ld
    out = os.path.join(HERE, "logs", f"exp_hm1_ngram_hv_{a.tag}")
    json.dump({"args": vars(a), **report}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\n  -> {os.path.relpath(out, ROOT)}.{{log,json}} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
