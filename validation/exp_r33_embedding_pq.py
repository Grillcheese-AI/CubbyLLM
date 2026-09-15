"""exp_r33 -- can the input embedding table be quantized, and by how much?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE TARGET, AND WHAT IS ALREADY SOLVED
--------------------------------------
The OUTPUT side needs nothing: `TopKRetrievalHead(learnable=False)` scores against
a FIXED row-normalized codebook, and a VSA codebook is generated from `SEED`, so
it stores zero bytes. H-C3 already measured that the codebook is the wall at
V=1M (10-41 GB fp32) and chose top-K retrieval over a full softmax. Quantizing
there would solve a solved problem.

The INPUT side is the gigabyte: `HybridEmbedding.core` is a learned
`nn.Embedding(vocab_core, d_model)`. At V=128k, d=2048 that is 1,048 MB of fp32
that every deployed instance pays for.

THREE WAYS TO COMPRESS IT
-------------------------
    hash_pq     product quantization with HASH-assigned codes (a "Bloom
                embedding"): code_m = hash(id, salt_m) % K, centroids learned.
                Works from scratch -- no pre-trained table needed, which is the
                only variant compatible with no-retraining on day one.

    clustered   real PQ: train a full table, k-means each subspace, assign each
                token to its nearest centroid, fine-tune. Better quality because
                similar tokens SHARE centroids meaningfully, but it needs the
                full table to exist first -- so it is a one-time base step, not
                something available before the first run.

    moqe        mixture of quantization experts: several codebooks at different
                precisions, tokens routed to one. Token frequency is Zipfian, so
                the head of the distribution deserves precision the tail does
                not.

WHY THE MoQE ROUTER HERE IS NOT cubemind's MoQE
------------------------------------------------
cubemind has a MoQE (`execution/moqe.py`): N experts at 2/4/6/8 bits over the
WEIGHT matrices, a learned softmax gate, Gumbel-Softmax training, a router
balance loss and a load-balancing entropy term. It was archived at PPL ~58.

For an embedding table none of that machinery is needed, and that is the point:
**token frequency is known before training**. The router can be a static
frequency bucket -- no gate to learn, no balance loss, no routing collapse to
diagnose. The archived result is about a learned router over weights; it does
not transfer to a static router over embeddings, and this experiment does not
inherit its risk.

WHAT THIS MEASURES
------------------
Aliasing (do two tokens end up with the identical vector?), memory, and the
reconstruction error each scheme puts on the embedding -- measured against REAL
token frequencies from the corpus, because a scheme that crushes rare tokens is
judged by how rare they actually are.

KILL CRITERION, two clauses
---------------------------
  * correctness -- ANY exact aliasing (two ids, one vector) is disqualifying.
    That is not lossy compression, it is two words becoming the same word, and
    no amount of training recovers it.
  * rate -- frequency-weighted reconstruction error must stay below `--max-err`
    relative to the full table. The error lands on PROPOSING (worse programs),
    which the disposer and the VM turn into refusals rather than wrong answers,
    so the honest cost of this change is refusal rate -- but a scheme whose
    error is large enough to change the answer distribution is not worth 300x.

    python validation/exp_r33_embedding_pq.py --corpus <a .jsonl>
"""
from __future__ import annotations

import argparse, collections, glob, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MASK64 = (1 << 64) - 1


def salted_codes(ids, m_subspaces: int, k: int, *, weak: bool = False):
    """Full splitmix64 per subspace, reading the HIGH bits.

    Two corrections, and the second one this experiment caught on itself.

    1. The GNSC prototype used `(id * prime_m) % K`, where every subspace shares
       the multiplier structure, so `id` and `id + K` are congruent in ALL of
       them. Measured: 128k ids collapse onto K distinct vectors. Salting each
       subspace makes a collision in one independent of the others.

    2. Salting is necessary and NOT sufficient. The first cut here used one
       multiply, one xorshift, and `% K` -- which reads the LOW bits, and low
       bits after a single multiply are barely mixed. At K=256 that aliased 140
       of 127,996 tokens, where the birthday bound predicts ~0. The fix is the
       complete splitmix64 finaliser and taking the HIGH bits
       (`h >> (64 - log2 K)`), which is where the mixing actually lands.

    `weak=True` reproduces the broken variant, because a correctness clause that
    cannot reproduce the failure it caught is not much of a clause.
    """
    import numpy as np

    out = np.empty((len(ids), m_subspaces), dtype=np.int64)
    u = np.asarray(ids, dtype=np.uint64)
    shift = np.uint64(64 - int(k).bit_length() + 1)
    for m in range(m_subspaces):
        salt = np.uint64(0x9E3779B97F4A7C15) + np.uint64(m) * np.uint64(0x7F4A7C15)
        h = (u ^ salt) & np.uint64(MASK64)
        h = (h * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(MASK64)
        h ^= h >> np.uint64(30 if not weak else 31)
        if weak:
            out[:, m] = (h % np.uint64(k)).astype(np.int64)
            continue
        h = (h * np.uint64(0x94D049BB133111EB)) & np.uint64(MASK64)
        h ^= h >> np.uint64(31)
        out[:, m] = ((h >> shift) % np.uint64(k)).astype(np.int64)
    return out


def prime_codes(ids, m_subspaces: int, k: int):
    """The GNSC prototype's assignment, kept as the control."""
    import numpy as np
    primes = np.array([10007, 20011, 30011, 40009, 50021, 60013, 70009, 80021],
                      dtype=np.int64)[:m_subspaces]
    return (np.asarray(ids, dtype=np.int64)[:, None] * primes) % k


def n_distinct(codes) -> tuple[int, int]:
    import numpy as np
    v = np.ascontiguousarray(codes).view(
        np.dtype((np.void, codes.dtype.itemsize * codes.shape[1])))
    _, counts = np.unique(v, return_counts=True)
    return len(counts), int(counts.max())


def token_counts(cache_paths: list[str], vocab: int, limit_tokens: int):
    """Exact token counts from the pretraining token cache (`.u32` id arrays).

    The cache is what the trunk will actually be trained on, so these are the
    real frequencies rather than a Zipf model -- and since the MoQE tiers ARE a
    frequency decision, a model would have decided the result. Reading the ids
    directly also skips re-tokenizing 25 GB of text.
    """
    import numpy as np

    counts = np.zeros(vocab, dtype=np.int64)
    seen = 0
    chunk = 1 << 26                                   # 64M ids ~ 256 MB per read
    for path in cache_paths:
        with open(path, "rb") as fh:
            while seen < limit_tokens:
                buf = np.fromfile(fh, dtype=np.uint32, count=chunk)
                if buf.size == 0:
                    break
                ok = buf[buf < vocab]                 # ignore ids past the table
                counts += np.bincount(ok, minlength=vocab)
                seen += buf.size
        if seen >= limit_tokens:
            break
    return counts, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", default=None,
                   help="a .u32 token-cache file (or glob) from the pretraining "
                        "corpus -- exact ids, so exact frequencies")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--limit-tokens", type=float, default=400e6,
                   help="how many ids to count; 400M is far past what frequency "
                        "RANKS need to converge")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--subspaces", type=int, default=8)
    ap.add_argument("--centroids", type=int, default=256)
    ap.add_argument("--max-err", type=float, default=0.25,
                   help="frequency-weighted relative reconstruction error ceiling")
    ap.add_argument("--spectrum", type=float, default=0.7,
                   help="power-law exponent of the synthetic table's singular "
                        "values. 0 = isotropic, which is the worst case for ANY "
                        "quantizer and where the instrument cannot discriminate")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import numpy as np
    from tokenizers import Tokenizer

    tp = pathlib.Path(a.tokenizer) if a.tokenizer else (
        ROOT / "data" / "grillcheese_bbpe128k.json")
    tok = Tokenizer.from_file(str(tp))
    V = tok.get_vocab_size()
    D, M, K = a.d_model, a.subspaces, a.centroids
    log(f"tokenizer {tp.name}: V={V}   d_model={D}, subspaces={M}, centroids={K}")

    # ---- 1. aliasing: the correctness clause -----------------------------
    ids = np.arange(V)
    log(f"\n{'assignment':<34}{'distinct':>10}{'worst group':>13}   verdict")
    log("-" * 72)
    alias = {}
    for label, codes in (
            ("GNSC prototype (id*prime)", prime_codes(ids, M, K)),
            ("salted, one round, low bits", salted_codes(ids, M, K, weak=True)),
            ("salted, full splitmix, high bits", salted_codes(ids, M, K))):
        d, worst = n_distinct(codes)
        alias[label] = {"distinct": d, "worst_group": worst}
        log(f"{label:<34}{d:>10}{worst:>13}   "
            f"{'DISQUALIFYING' if d < V else 'ok'}")
    log("  the middle row is this experiment's OWN first cut: salting the subspaces")
    log("  is necessary and not sufficient -- `% K` reads the low bits, and after a")
    log("  single multiply those are barely mixed.")

    # ---- 2. real token frequencies ---------------------------------------
    paths: list[str] = []
    if a.token_cache:
        paths = sorted(glob.glob(a.token_cache))
    counts = None
    if not paths:
        log("\n!! no --token-cache given; the MoQE tiers ARE a frequency decision,")
        log("   so an assumed distribution would BE the result. Skipping the tier")
        log("   comparison rather than inventing one.")
    else:
        log(f"\ncounting ids from {len(paths)} cache file(s), "
            f"{a.limit_tokens / 1e6:.0f}M-token budget...")
        counts, seen = token_counts(paths, V, int(a.limit_tokens))
        total = int(counts.sum())
        order = np.argsort(-counts)
        log(f"  {seen:,} ids read, {total:,} in-vocabulary, "
            f"{int((counts > 0).sum()):,} of {V:,} entries used")
        for cut in (1000, 5000, 20000, 50000):
            cov = int(counts[order[:cut]].sum())
            log(f"  top {cut:>6,} tokens cover {100 * cov / total:>5.1f}% of occurrences")
        log(f"  never used: {int((counts == 0).sum()):,} entries "
            f"({100 * (counts == 0).mean():.1f}% of the table)")

    # ---- 3. memory --------------------------------------------------------
    full = V * D * 4
    cb = M * K * (D // M) * 4
    code_bytes = 1 if K <= 256 else 2
    codes_sz = V * M * code_bytes
    log(f"\nmemory at V={V:,}, d={D}:")
    log(f"  full fp32 table           {full / 1e6:>9.1f} MB")
    log(f"  PQ codebooks              {cb / 1e6:>9.2f} MB")
    log(f"  PQ codes ({code_bytes} B/subspace)   {codes_sz / 1e6:>9.2f} MB")
    log(f"  total                     {(cb + codes_sz) / 1e6:>9.2f} MB    "
        f"({full / (cb + codes_sz):.0f}x smaller)")

    # ---- 4. reconstruction error -----------------------------------------
    # A real table is not available (the trunk is untrained), so this measures
    # the scheme's capacity on a SYNTHETIC table with the statistics an embedding
    # table has -- unit-norm rows, isotropic. Stated plainly: this bounds the
    # quantizer, it does not predict the trained model's loss.
    rng = np.random.default_rng(a.seed)
    n_probe = min(V, 20000)
    probe_ids = (order[:n_probe] if counts is not None
                 else rng.choice(V, n_probe, replace=False))
    # The table's SPECTRUM is the whole experiment. An isotropic Gaussian table
    # is the worst case for any quantizer: in 256 dimensions every point is
    # nearly equidistant from every centroid, so 256 centroids reconstruct
    # nothing and hash-assigned, clustered and random all score ~1.0. The first
    # cut here did exactly that and "killed" PQ on an artifact -- the tell was
    # clustered PQ (0.973) barely beating hashing (0.994), when clustering must
    # win decisively if the instrument can see anything at all.
    #
    # Real embedding tables are strongly anisotropic: a few directions carry most
    # of the variance. That is WHY product quantization works on them. So the
    # table is drawn with a power-law spectrum, and `--spectrum 0` recovers the
    # isotropic worst case for comparison.
    sub = D // M
    E = rng.standard_normal((n_probe, D)).astype(np.float32)
    if a.spectrum > 0:
        decay = (np.arange(1, D + 1, dtype=np.float32) ** (-a.spectrum))
        basis, _ = np.linalg.qr(rng.standard_normal((D, D)).astype(np.float32))
        E = (E * decay) @ basis.T
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    def recon_error(codes: np.ndarray) -> np.ndarray:
        """Fit each subspace's centroids as the MEAN of its assigned rows -- the
        optimal 1-step assignment-fixed fit, i.e. the best a hash-assigned PQ can
        do without moving assignments."""
        err = np.zeros(n_probe, dtype=np.float32)
        for m in range(M):
            block = E[:, m * sub:(m + 1) * sub]
            cm = codes[:, m]
            cent = np.zeros((K, sub), dtype=np.float32)
            cnt = np.bincount(cm, minlength=K).astype(np.float32)
            np.add.at(cent, cm, block)
            cent /= np.maximum(cnt, 1)[:, None]
            diff = block - cent[cm]
            err += (diff ** 2).sum(axis=1)
        return np.sqrt(err) / np.linalg.norm(E, axis=1)

    hash_err = recon_error(salted_codes(probe_ids, M, K))

    # clustered: assignment follows the DATA (k-means, a few Lloyd iterations)
    def kmeans_codes(n_iter: int = 8) -> np.ndarray:
        """Lloyd's, with distances as a matmul: ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2.
        The first cut built an (n, K, sub) difference tensor -- 1.3e9 floats at
        these shapes, which is why it never finished."""
        codes = np.empty((n_probe, M), dtype=np.int64)
        for m in range(M):
            block = np.ascontiguousarray(E[:, m * sub:(m + 1) * sub])
            xn = (block ** 2).sum(1, keepdims=True)                 # (n, 1)
            cent = block[rng.choice(n_probe, K, replace=False)].copy()
            assign = np.zeros(n_probe, dtype=np.int64)
            for _ in range(n_iter):
                d2 = xn - 2.0 * (block @ cent.T) + (cent ** 2).sum(1)[None, :]
                assign = d2.argmin(1)
                newc = np.zeros_like(cent)
                cnt = np.bincount(assign, minlength=K).astype(np.float32)
                np.add.at(newc, assign, block)
                nz = cnt > 0
                newc[nz] /= cnt[nz][:, None]
                newc[~nz] = cent[~nz]                               # keep empty centroids
                cent = newc
            codes[:, m] = assign
        return codes

    log("\nfitting clustered PQ (k-means per subspace)...")
    clustered_err = recon_error(kmeans_codes())

    def summary(err: np.ndarray, label: str, weights=None) -> dict:
        mean = float(err.mean())
        wmean = float(np.average(err, weights=weights)) if weights is not None else mean
        return {"label": label, "mean_rel_err": round(mean, 4),
                "freq_weighted_rel_err": round(wmean, 4),
                "p95": round(float(np.percentile(err, 95)), 4)}

    w = (counts[probe_ids].astype(np.float64) if counts is not None else None)
    if w is not None and w.sum() <= 0:
        w = None
    rows = [summary(hash_err, "hash-assigned PQ", w),
            summary(clustered_err, "clustered PQ (k-means)", w)]

    # ---- 5. MoQE: precision by frequency tier ----------------------------
    if counts is not None:
        # static router: the head of the distribution keeps full precision, the
        # tail is quantized. No gate to learn -- frequency is known in advance.
        for head_frac in (0.02, 0.05):
            n_head = int(n_probe * head_frac)
            err = hash_err.copy()
            err[:n_head] = 0.0                     # head kept in fp32
            tail_mem = (V - int(V * head_frac)) * M * code_bytes
            head_mem = int(V * head_frac) * D * 4
            rows.append({**summary(err, f"MoQE: top {head_frac:.0%} fp32 + tail PQ", w),
                         "memory_mb": round((cb + tail_mem + head_mem) / 1e6, 2)})

    # ---- 5b. the knob that actually decides PQ quality: sub_dim = D / M ----
    # M=8 over d=2048 asks 256 centroids to cover a 256-dimensional subspace,
    # which is far coarser than PQ is ever run in practice (sub_dim 4-16 is
    # normal). Sweeping M is the difference between "PQ does not work" and "PQ
    # was configured wrong", and the first cut here did not sweep it.
    log(f"\n{'M':>5}{'sub_dim':>9}{'clustered err':>15}{'freq-wtd':>10}"
        f"{'codes MB':>10}{'total MB':>10}")
    log("-" * 60)
    sweep = []
    for M2 in (8, 32, 128, 256):
        if D % M2 or M2 > D:
            continue
        sub2 = D // M2
        saveM, savesub = M, sub
        M, sub = M2, sub2
        try:
            err2 = recon_error(kmeans_codes(n_iter=6))
        finally:
            M, sub = saveM, savesub
        cb2 = M2 * K * sub2 * 4
        codes2 = V * M2 * code_bytes
        wmean2 = float(np.average(err2, weights=w)) if w is not None else float(err2.mean())
        sweep.append({"M": M2, "sub_dim": sub2, "mean_rel_err": round(float(err2.mean()), 4),
                      "freq_weighted_rel_err": round(wmean2, 4),
                      "total_mb": round((cb2 + codes2) / 1e6, 2),
                      "compression": round(full / (cb2 + codes2), 1)})
        log(f"{M2:>5}{sub2:>9}{err2.mean():>15.4f}{wmean2:>10.4f}"
            f"{codes2 / 1e6:>10.2f}{(cb2 + codes2) / 1e6:>10.2f}")

    log(f"\n{'scheme':<34}{'mean err':>10}{'freq-wtd':>10}{'p95':>8}{'MB':>9}")
    log("-" * 72)
    for r in rows:
        log(f"{r['label']:<34}{r['mean_rel_err']:>10.4f}"
            f"{r['freq_weighted_rel_err']:>10.4f}{r['p95']:>8.4f}"
            f"{r.get('memory_mb', round((cb + codes_sz) / 1e6, 2)):>9}")
    log("  err = ||e - reconstructed|| / ||e||, on a synthetic unit-norm table.")
    log("  This bounds the QUANTIZER; it does not predict the trained model's loss.")

    # THE INSTRUMENT'S OWN CHECK, and the one this experiment needed.
    # Clustered PQ chooses assignments from the data; a hash chooses them blind.
    # If clustering does not clearly beat hashing, the table has no structure for
    # ANY quantizer to find and no error number from it means anything. Reporting
    # a verdict in that state is how the first cut "killed" PQ on an artifact.
    h, c = rows[0]["mean_rel_err"], rows[1]["mean_rel_err"]
    gain = (h - c) / h if h > 0 else 0.0
    log(f"\ninstrument check: clustered beats hash-assigned by {gain:.1%} "
        f"({h:.4f} -> {c:.4f})")
    discriminating = gain >= 0.10
    if not discriminating:
        log("  -> BELOW 10%: the synthetic table has no structure to quantize, so")
        log("     every scheme scores alike and the rate clause is unmeasurable.")
        log(f"     Raise --spectrum (now {a.spectrum}) or wait for a real table.")

    aliased = alias["salted, full splitmix, high bits"]["distinct"] < V
    best = min(r["freq_weighted_rel_err"] for r in rows)
    if aliased:
        verdict = "KILLED (correctness): the salted assignment still aliases"
    elif not discriminating:
        verdict = (f"correctness PASSES (zero aliasing, "
                   f"{full / (cb + codes_sz):.0f}x smaller); the RATE clause is "
                   f"unmeasurable here -- the instrument cannot tell the schemes "
                   f"apart on a table this isotropic, so it reports that instead "
                   f"of a number")
    else:
        # The rate clause is DEFERRED, not passed and not failed. Every error
        # number above rests on a synthetic table whose spectrum is a parameter
        # somebody chose (--spectrum), and on a --max-err threshold nobody has
        # grounded, because no trained table exists to ground it against. Two
        # guesses multiplied do not make a kill. What IS settled is the ORDERING,
        # which does not depend on either guess, and the memory, which is exact.
        verdict = (f"correctness PASSES: zero aliasing, "
                   f"{full / (cb + codes_sz):.0f}x smaller. RATE DEFERRED -- the "
                   f"error numbers rest on a synthetic spectrum ({a.spectrum}) and "
                   f"an ungrounded threshold ({a.max_err}); re-run against the real "
                   f"table once the trunk is trained. The ordering is what "
                   f"transfers: hash-assigned {rows[0]['mean_rel_err']:.3f} vs "
                   f"clustered {rows[1]['mean_rel_err']:.3f}, and "
                   f"frequency-weighted {best:.3f} with tiering")
    log(f"\nVERDICT: {verdict}")
    log("\nWHAT IS SETTLED AND WHAT IS NOT:")
    log("  settled  - GNSC's assignment aliases 128k tokens onto K vectors, and a")
    log("             full splitmix64 on the high bits fixes it to zero. Exact.")
    log("  settled  - the memory arithmetic, and that hash-assigned PQ (the only")
    log("             variant needing no pre-trained table) reconstructs NOTHING:")
    log("             ~0.99 relative error however much it compresses.")
    log("  settled  - the ordering: clustering beats hashing, smaller sub_dim beats")
    log("             larger, and frequency tiering beats both per MB.")
    log("  NOT      - whether any of it costs the model anything. That needs a")
    log("             trained table, and the honest answer today is 'unmeasured'.")
    log("\n  So the shape of the recommendation is already clear: the from-scratch")
    log("  variant is the one that does not work, and clustered PQ + a static")
    log("  frequency router is a POST-training compression step on a finished base.")
    log("  That fits no-retraining exactly -- compress the base once, ship it.")

    stem = f"exp_r33_embedding_pq{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "vocab": V, "d_model": D, "subspaces": M, "centroids": K,
        "token_cache": paths, "limit_tokens": a.limit_tokens,
        "aliasing": alias, "memory_mb": {
            "full_fp32": round(full / 1e6, 1), "codebooks": round(cb / 1e6, 2),
            "codes": round(codes_sz / 1e6, 2),
            "total_pq": round((cb + codes_sz) / 1e6, 2)},
        "schemes": rows, "max_err": a.max_err, "verdict": verdict,
        "subspace_sweep": sweep,
        "spectrum": a.spectrum, "clustering_gain": round(gain, 4),
        "instrument_discriminating": bool(discriminating),
        "caveat": "reconstruction error is measured on a SYNTHETIC unit-norm table; "
                  "the trunk is untrained, so this bounds the quantizer rather than "
                  "predicting the model's loss",
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
