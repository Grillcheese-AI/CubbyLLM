"""build_word_table -- a FastWordEncoder word table from the MoWM axiom codebook, and the
256-bit word codes the hippocampus reads from it.

Wired: STANDALONE (offline builder; run once; outputs are off-repo, under standin/data/out/).

The MoWM codebook (a checkout beside this one; ``mowm_nvembed_codebook.npz``) holds 166,894
axiom texts with their NV-Embed-v2 embeddings (4,096-d), the 16 domain centroids and the
orthonormal projection into the worlds' (k=80, l=128) block space. No teacher runs here: a
WORD's vector is the mean of the embeddings of the texts it occurs in (its contexts), with
the anisotropy direction removed, projected through the codebook's own matrix -- so word
codes, episode codes and the worlds' centroids share one block space -- then blended 0.5/0.5
with the word's seeded signal code, exactly FastWordEncoder.build's recipe (semantic_words.py,
loaded by path; never imported as a package: ported, not linked). Coverage measured
2026-09-12: 96% of the wiki world's 2,134 relation content words occur in >= 3 codebook
texts; 98% of SimpleQA's 400 most frequent question words. The misses are entity names and
misspellings, which fall back to the signal code -- surface identity, pattern-separated.

Outputs (both off-repo):
  fastword_table_mowm.npz   FastWordEncoder.from_npz-loadable (float16 vectors)
  word_bits_mowm.json       {word: 256-bit code as hex}: SimHash of the word's block vector
                            through one fixed Gaussian projection -- the hippocampus's DG

  python standin/data/build_word_table.py [--codebook PATH] [--semantic-words PATH]
         [--min-df 3] [--max-words 80000] [--max-texts-per-word 2000]
"""
from __future__ import annotations

import argparse, collections, hashlib, importlib.util, json, math, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "standin" / "data" / "out"
MOWM = ROOT.parent / "mowm"
N_BITS = 256
SIMHASH_SEED = 0x5EED_B175


def load_semantic_words(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("semantic_words", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)     # type: ignore[union-attr]
    return mod


def simhash_matrix(dim: int, n_bits: int = N_BITS, seed: int = SIMHASH_SEED) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((dim, n_bits)).astype(np.float32)


def bits_of(v: np.ndarray, R: np.ndarray) -> int:
    s = (v.astype(np.float32) @ R) > 0
    return int("".join("1" if b else "0" for b in s[::-1]), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default=str(MOWM / "mowm_nvembed_codebook.npz"))
    ap.add_argument("--semantic-words", default=str(MOWM / "mowm" / "encoding" / "semantic_words.py"))
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=80_000)
    ap.add_argument("--max-texts-per-word", type=int, default=2000)
    ap.add_argument("--signal-blend", type=float, default=0.5)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.perf_counter()
    sw = load_semantic_words(pathlib.Path(a.semantic_words))
    d = np.load(a.codebook, allow_pickle=True)
    texts = [str(t) for t in d["texts"]]
    E = d["embeddings"]                                    # (n, 4096) float32
    P = d["projection_matrix"]                             # (k*l, 4096)
    k, l = int(d["k_vsa"]), int(d["l_vsa"])
    print(f"codebook: {len(texts)} texts, embeddings {E.shape}, projection {P.shape}, block ({k},{l}) | {time.perf_counter() - t0:.0f}s", flush=True)

    # inverted index: word -> text ids (a text counts once per word)
    inv: dict[str, list[int]] = collections.defaultdict(list)
    for i, t in enumerate(texts):
        for w in sw.split_words(t):
            inv[w].append(i)
    df = {w: len(ids) for w, ids in inv.items()}
    words = sorted((w for w, n in df.items() if n >= a.min_df), key=lambda w: (-df[w], w))[:a.max_words]
    print(f"{len(df)} distinct words; {len(words)} with df >= {a.min_df} kept (cap {a.max_words}) | {time.perf_counter() - t0:.0f}s", flush=True)

    # each word: the mean of its contexts' embeddings (capped, deterministic sample)
    rng = np.random.default_rng(0)
    W = np.zeros((len(words), E.shape[1]), dtype=np.float32)
    for j, w in enumerate(words):
        ids = inv[w]
        if len(ids) > a.max_texts_per_word:
            ids = rng.choice(ids, size=a.max_texts_per_word, replace=False)
        W[j] = E[np.asarray(ids)].mean(axis=0)
        if (j + 1) % 10000 == 0:
            print(f"  {j + 1}/{len(words)} word vectors ({time.perf_counter() - t0:.0f}s)", flush=True)
    W = sw._unit_rows(W)
    mu, pcs = sw.fit_anisotropy(W)                          # the table's own statistics (in-domain form)
    Wc = sw.remove_anisotropy(W, mu, pcs)
    proj = sw._unit_rows(Wc @ P.T)                          # (n, k*l): the worlds' block space, cosine-exact
    vectors = np.stack([(1.0 - a.signal_blend) * proj[i] + a.signal_blend * sw.signal_code(w, k, l)
                        for i, w in enumerate(words)]).astype(np.float32)
    vectors = sw._unit_rows(vectors)
    n = len(texts)
    idf = {w: float(math.log(n / df[w])) for w in words}
    enc = sw.FastWordEncoder(words, vectors, idf, float(math.log(n)), k, l,
                             meta={"source": "mowm_nvembed_codebook", "teacher": str(d["model_id"]), "recipe": "context-mean",
                                   "signal_blend": a.signal_blend, "min_df": a.min_df, "n_words": len(words), "built": time.strftime("%Y-%m-%d")})
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    enc.save_npz(out / "fastword_table_mowm.npz", dtype=np.float16)
    print(f"wrote fastword_table_mowm.npz ({len(words)} words) | {time.perf_counter() - t0:.0f}s", flush=True)

    # the hippocampus's DG: one fixed SimHash of each word's block vector
    R = simhash_matrix(k * l)
    S = (vectors @ R) > 0                                   # (n, 256) bool
    bits = {}
    for i, w in enumerate(words):
        row = S[i]
        code = 0
        for j in range(N_BITS):
            if row[j]:
                code |= 1 << j
        bits[w] = format(code, "x")
    (out / "word_bits_mowm.json").write_text(json.dumps({"n_bits": N_BITS, "seed": SIMHASH_SEED, "signal_blend": a.signal_blend,
                                                         "source": "fastword_table_mowm.npz", "bits": bits}), encoding="utf-8")
    print(f"wrote word_bits_mowm.json ({len(bits)} words) | {time.perf_counter() - t0:.0f}s", flush=True)

    # a look at what the table thinks (cosine in block space; SimHash agreement in bits)
    def cos(x, y): return float(enc._vec[x] @ enc._vec[y]) if x in enc._vec and y in enc._vec else float("nan")
    def agree(x, y): return N_BITS - bin(int(bits[x], 16) ^ int(bits[y], 16)).count("1") if x in bits and y in bits else -1
    for x, y in (("born", "birth"), ("founded", "inception"), ("founded", "established"), ("country", "nation"),
                 ("capital", "city"), ("death", "died"), ("spouse", "married"), ("james", "john"), ("capital", "banana")):
        print(f"  cos({x},{y}) = {cos(x, y):.3f}   bits agreeing {agree(x, y)}/{N_BITS}")


if __name__ == "__main__":
    main()
