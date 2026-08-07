"""exp_f2m2_semantic_routing — H-F2 M2: can semantic block codes route?

Stage 1 ranks embedders (HASH / NGRAM / T-bag / T-ctx / REF) under a dense,
cosine-exact orthonormal projection into (80,128) (HASH/NGRAM are born directly
in block space, so they skip the projection). Stage 2 settles discretization
(dense / argmax / PQ / PQ-ADC) for the winner and recalibrates tau_match.
--leakage-check controls Stage 2's codebook; --null-baseline measures what each
statistic scores under a PURE-NOISE encoder, which is the only honest thing to
read the results against.

Standalone experiment: NEVER imported by cubbyllm/.

RESULT AND AUTHORITY. The Stage-1 kill criterion CLOSED (bars 2/3a/3b failed),
so the design's SS4.3 live confirm and its mowm-side path were never built, and
Stage 2's "quantization improves routing" was RETRACTED as codebook leakage.
The authoritative record is CUBBYLLM_HYPOTHESES.md, entry "H-F2 M2 screened
(2026-08-06)" -- every figure there links a file in validation/logs/. The design
doc below is amended-in-place with dated notes rather than rewritten, so it
still carries forward-looking prose that the run overtook; read it for intent,
read the hypotheses entry for what happened:
docs/superpowers/specs/2026-08-06-hf2-m2-semantic-context-encoding-design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

K, L = 80, 128
VSA_DIM = K * L          # 10240
PROJ_SEED = 0

# Stamped INSIDE data/axiom_embeddings_*.npz. Those files record what a Stage-2
# run produced; they are NOT a configuration to load a threshold or a codebook
# from. Nothing in this repo reads them, and the only place they were ever meant
# to be consumed (the plan's Task 7, the mowm-side semantic path) was gated out
# and never built -- exactly the "looks live, documented only by the killed
# path's instructions" shape this repo names as an anti-pattern.
NPZ_RETRACTED = (
    "discretization + tau_match are RETRACTED: the pq variant won only because "
    "its codebook was fit on the items leave-one-out excludes (leakage). See "
    "validation/logs/exp_f2m2_leakage_*.json."
)
NPZ_NOTE = (
    "RECORD, NOT CONFIGURATION. Produced by validation/exp_f2m2_semantic_routing.py "
    "--stage2. Do NOT load `discretization`, `tau_match` or `centroids` as settings: "
    "(1) `discretization='pq'` and its `tau_match` come from the variant whose "
    "advantage was later shown to be a codebook-leakage artifact and is RETRACTED "
    "(validation/logs/exp_f2m2_leakage_ref.json / _t-ctx.json); (2) `centroids` IS "
    "that leakage-contaminated codebook, fit on all 280 formulas including the "
    "held-out item; (3) `roc_auc` is that same pq variant's and inherits the same "
    "retraction -- on t-ctx it sits BELOW its measured null (0.0915 vs the one-hot "
    "null 0.1605 +/- 0.0162, validation/logs/exp_f2m2_null_baseline.json), and even "
    "where the AUC does clear its null the threshold does not transfer, because "
    "these taus are one-hot block-code cosines (81 levels at K=80) from a "
    "CROSS-MODAL corpus while M1's 0.35 is a k=4/l=32 cosine. M1's tau_match = 0.35 "
    "is NOT replaced by any M2 number. "
    "Nothing in CubbyLLM or mowm reads this file: the consumer it was written for "
    "(the mowm-side semantic path) was gated out by the Stage-1 kill criterion and "
    "never built. Kept as a record of the screen. See CUBBYLLM_HYPOTHESES.md, "
    "H-F2 M2."
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Make the repo root importable so `import cubbyllm` works without an editable
# install — mirrors tests/conftest.py; needed because running this file
# directly (`python validation/exp_f2m2_semantic_routing.py`) puts this file's
# own directory on sys.path[0], not the repo root, and tbag_encode/_rebuild_trunk
# (Task 3) are the first things in this file to actually import cubbyllm.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "mowm_axioms.json"


def load_corpus() -> tuple[list[str], list[str], list[str]]:
    """-> (names, formulas, top-level domains), aligned by index."""
    axioms = json.loads(CORPUS.read_text(encoding="utf-8"))["axioms"]
    names = [a["name"] for a in axioms]
    formulas = [a["formula_str"] for a in axioms]
    domains = [a["domain"].split(".")[0] for a in axioms]
    return names, formulas, domains


def make_projection(latent_dim: int, seed: int = PROJ_SEED) -> np.ndarray:
    """(VSA_DIM, latent_dim) with ORTHONORMAL COLUMNS.

    Orthonormality makes the projection cosine-exact: (Px).(Py) = x.y and
    ||Px|| = ||x||, so nothing is lost moving embeddings into VSA space.
    """
    if VSA_DIM < latent_dim:
        raise ValueError(f"need k*l={VSA_DIM} >= latent_dim={latent_dim}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((VSA_DIM, latent_dim)))
    return Q.astype(np.float32)


def project(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """(n, latent_dim) embeddings -> (n, K, L) dense block codes."""
    return (np.asarray(E, dtype=np.float32) @ P.T).reshape(-1, K, L)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine between every row of A and every row of B, flattened. -> (n, m)."""
    a = A.reshape(len(A), -1).astype(np.float64)
    b = B.reshape(len(B), -1).astype(np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def hash_encode(texts: list[str]) -> np.ndarray:
    """M1's status quo, ported (not imported): BLAKE2b-seeded one-hot per block.

    Mirrors cubemind's WorldEncoder._hash_to_vec semantics — a deterministic
    one-hot-per-block code with no relationship between similar texts.
    """
    out = np.zeros((len(texts), K, L), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**63)
        idx = np.random.default_rng(seed).integers(0, L, size=K)
        out[i, np.arange(K), idx] = 1.0
    return out


def ngram_encode(texts: list[str], n: int = 3) -> np.ndarray:
    """Training-free lexical sketch: bundle seeded block codes of character n-grams.

    Each unique character n-gram gets a permanent BLAKE2b-seeded one-hot-per-block
    code; bundling (superposing) them makes a string similar to any string sharing
    substrings -- bundling preserves similarity-to-parts, unlike binding. Born in
    block space, so no projection step is involved.
    """
    out = np.zeros((len(texts), K, L), dtype=np.float32)
    for i, text in enumerate(texts):
        s = text.lower()
        grams = {s[j:j + n] for j in range(max(1, len(s) - n + 1))}
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(digest, "little") % (2**63)
            idx = np.random.default_rng(seed).integers(0, L, size=K)
            out[i, np.arange(K), idx] += 1.0
    return out


_GRAM_CACHE: dict[str, np.ndarray] = {}


def _grams(text: str, ns: tuple[int, ...] = (2, 3, 4), words: bool = False) -> set[str]:
    """Namespaced gram set for a text: char n-grams per n, optional word tokens."""
    s = text.lower()
    out: set[str] = set()
    for n in ns:
        for j in range(max(1, len(s) - n + 1)):
            out.add(f"{n}:{s[j:j + n]}")
    if words:
        for tok in re.split(r"[^a-z0-9]+", s):
            if len(tok) >= 2:
                out.add(f"w:{tok}")
    return out


def build_idf(texts: list[str], ns: tuple[int, ...] = (2, 3, 4), words: bool = False):
    """OFFLINE corpus statistics for gram weighting -> (idf dict, unseen-gram default).

    Nothing here runs per-challenge: the serving-path cost of using the weights
    is one dict lookup per gram. Unseen grams get the maximum weight (they are
    by definition distinctive).
    """
    df: dict[str, int] = {}
    for t in texts:
        for g in _grams(t, ns, words):
            df[g] = df.get(g, 0) + 1
    n_docs = max(1, len(texts))
    idf = {g: float(np.log(n_docs / c)) for g, c in df.items()}
    return idf, float(np.log(n_docs))


def ngram_encode_v2(
    texts: list[str],
    ns: tuple[int, ...] = (2, 3, 4),
    words: bool = False,
    idf: dict[str, float] | None = None,
    default_idf: float | None = None,
) -> np.ndarray:
    """The fast-path challenge encoder: weighted bundle of seeded block codes.

    Same BLAKE2b primitive as hash_encode/ngram_encode; the upgrades measured
    by --fast-arms are variable-length grams, word tokens, and IDF weighting.
    The weight scales the bundle CONTRIBUTION (a real scalar on magnitude) --
    NOT a phase rotation, which would compress common grams into one arc and
    amplify exactly what it means to attenuate. Per-gram codes are memoized in
    _GRAM_CACHE, since a serving vocabulary of grams stabilizes quickly.
    """
    out = np.zeros((len(texts), K, L), dtype=np.float32)
    rows = np.arange(K)
    for i, text in enumerate(texts):
        for g in _grams(text, ns, words):
            idx = _GRAM_CACHE.get(g)
            if idx is None:
                digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
                seed = int.from_bytes(digest, "little") % (2**63)
                idx = np.random.default_rng(seed).integers(0, L, size=K)
                _GRAM_CACHE[g] = idx
            w = 1.0 if idf is None else idf.get(g, default_idf)
            out[i, rows, idx] += w
    return out


def token_encode(
    texts: list[str],
    encode_fn,
    idf: dict[int, float] | None = None,
    default_idf: float | None = None,
) -> np.ndarray:
    """Token-ID bundle: use the numbers the pipeline already has.

    In serving, the trunk has ALREADY tokenized the challenge, so this
    encoder's marginal cost is scatter-adds over ids that exist for free. A
    token id's numeric VALUE carries no meaning (43706 and 43707 are unrelated)
    -- the id is a stable KEY seeding a block code, never a coordinate, so no
    arithmetic (and no fractional binding) over ids. Structurally this is
    t-bag with the trained embedding rows replaced by random block codes.
    --fast-arms timing INCLUDES tokenization, overstating the marginal cost.
    """
    out = np.zeros((len(texts), K, L), dtype=np.float32)
    rows = np.arange(K)
    for i, text in enumerate(texts):
        for tid in set(encode_fn(text.lower())):
            key = f"t:{tid}"
            idx = _GRAM_CACHE.get(key)
            if idx is None:
                idx = np.random.default_rng(1_000_003 + tid).integers(0, L, size=K)
                _GRAM_CACHE[key] = idx
            w = 1.0 if idf is None else idf.get(tid, default_idf)
            out[i, rows, idx] += w
    return out


def ref_encode(texts: list[str]) -> np.ndarray:
    """Reference arm: a small pretrained sentence embedder (384-D).

    Present ONLY to size the gap against our own trunk. Requires
    `pip install sentence-transformers` and one ~90MB model download.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=False),
        dtype=np.float32,
    )


CKPT = pathlib.Path(r"D:\CUBBY-TRAINED-MODELS\hd5_mem21.pt")
TOKENIZER = ROOT / "data" / "grillcheese_bbpe128k.json"


def load_tokenizer():
    """-> (encode, vocab) for the BBPE-128k tokenizer the trunk was trained on."""
    from cubbyllm.training.data import _load_tokenizer

    encode, _decode, _eos_id, vocab = _load_tokenizer(str(TOKENIZER))
    return encode, vocab


def load_checkpoint() -> dict:
    import torch

    return torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)


def tbag_encode(texts: list[str]) -> np.ndarray:
    """Ours, cheapest: mean of the TRAINED token-embedding rows. No model rebuild.

    params[0] is the (vocab, d) embedding table; averaging its rows over a
    text's tokens is a bag-of-embeddings sentence vector.
    """
    encode, vocab = load_tokenizer()
    ck = load_checkpoint()
    emb = ck["params"][0].float().numpy()                       # (127996, 512)
    assert emb.shape[0] == ck["meta"]["vocab"] == vocab, (
        emb.shape, ck["meta"]["vocab"], vocab
    )
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [t for t in encode(text) if 0 <= t < emb.shape[0]]
        if ids:
            out[i] = emb[ids].mean(axis=0)
    return out


def _rebuild_trunk():
    """Rebuild the trained model exactly and load its params.

    train_colab.py reads its config from the environment at import time, so
    setting the env from the checkpoint's own `meta` reproduces the exact
    architecture; the tensor-count and shape assertions below are the same
    guard train_colab's resume path uses (:414-420).
    """
    import importlib.util
    import os

    import torch

    ck = load_checkpoint()
    meta = ck["meta"]
    os.environ.update({
        "CB_D": str(meta["D"]), "CB_L": str(meta["L"]),
        "CB_ATTN_EVERY": str(meta["attn_every"]), "CB_WINDOW": str(meta["window"]),
        "CB_HEADS": str(meta["heads"]), "CB_MEM_EVERY": str(meta["mem_every"]),
        "CB_MEM_TOPK": str(meta["mem_topk"]), "CB_MEM_KEY": str(meta["mem_key"]),
        "CB_BACKBONE": str(meta["backbone"]),
        # NOTE: the generator kind is CB_GEN_KIND, not CB_GEN -- CB_GEN is an
        # unrelated integer (GEN_EVERY, "sample generations every N steps"), and
        # setting it to "basis" raises ValueError at train_colab import time.
        "CB_GEN_KIND": str(meta["gen"]),
        "CB_AMP": "0", "CUBBY_SPM": str(TOKENIZER),
    })
    spec = importlib.util.spec_from_file_location(
        "_m2_train_colab", ROOT / "validation" / "train_colab.py"
    )
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)

    model = tc.build(meta["vocab"], torch.device("cpu"))
    params = list(model.parameters())
    saved = ck["params"]
    if len(params) != len(saved):
        raise SystemExit(
            f"rebuild mismatch: model has {len(params)} tensors, checkpoint has "
            f"{len(saved)}. Do NOT fall back silently — report this."
        )
    for i, (p, s) in enumerate(zip(params, saved)):
        if tuple(p.shape) != tuple(s.shape):
            raise SystemExit(f"shape mismatch at tensor {i}: {tuple(p.shape)} vs {tuple(s.shape)}")
        with torch.no_grad():
            p.copy_(s)
    return model, tc


def tctx_encode(texts: list[str], batch: int = 16) -> np.ndarray:
    """Ours, contextual: mean-pooled pre-head hidden states from the real trunk."""
    import torch

    encode, _vocab = load_tokenizer()
    model, _tc = _rebuild_trunk()
    window = load_checkpoint()["meta"]["window"]

    out = np.zeros((len(texts), load_checkpoint()["meta"]["D"]), dtype=np.float32)
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        seqs = [encode(t)[:window] or [0] for t in chunk]
        width = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width, 1))
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[r, : len(s), 0] = 1.0
        with torch.no_grad():
            h = model.features(ids)                       # (B, S, d)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        out[start : start + len(chunk)] = pooled.float().numpy()
    return out


def majority_class(domains: list[str]) -> float:
    """MICRO chance: what "always guess the biggest domain" scores (physics is 84/280)."""
    counts = {}
    for d in domains:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.values()) / len(domains)


def macro_chance(domains: list[str]) -> float:
    """MACRO chance: 1/n_domains.

    Macro-averaging removes class imbalance by construction, so EVERY constant
    classifier -- the majority-class strategy included -- and uniform-random
    guessing all score exactly 1/n_domains in macro. Comparing a macro accuracy
    against the majority-class rate is a category error: they are baselines for
    different metrics.
    """
    return 1.0 / len(set(domains))


# --- MATCHED nulls -----------------------------------------------------------
# majority_class / macro_chance are baselines for a classifier that PICKS A
# LABEL. Our metric does not pick a label -- it takes an argmax over the other
# n-1 CANDIDATE ITEMS and reads off that item's domain. The matched null for
# that procedure is "pick a uniformly random other item", which is a different
# number, because a domain is hit in proportion to how many candidates it owns.

def per_domain_null(domains: list[str]) -> dict:
    """Per-domain chance for argmax-over-the-other-(n-1)-items: (n_d - 1)/(n - 1).

    NOT flat. On this corpus it ranges from 9/279 = 0.032 for a 10-member domain
    to 83/279 = 0.297 for physics (84 members), so a per-domain score must be
    read against its OWN domain's baseline, never against one shared number.
    """
    n = len(domains)
    sizes = {d: domains.count(d) for d in sorted(set(domains))}
    return {d: (nd - 1) / (n - 1) for d, nd in sizes.items()}


def matched_micro_null(domains: list[str]) -> float:
    """MICRO null for argmax-over-candidates: sum_d n_d(n_d - 1) / (n(n - 1)).

    The probability that a uniformly random OTHER item shares the query's
    domain. This is the random-encoder floor; majority_class() is a different,
    also-legitimate baseline (a strong constant strategy). Report both, labelled.
    """
    n = len(domains)
    sizes = {d: domains.count(d) for d in set(domains)}
    return sum(nd * (nd - 1) for nd in sizes.values()) / (n * (n - 1))


def matched_macro_null(domains: list[str]) -> float:
    """MACRO null for argmax-over-candidates: mean_d (n_d - 1)/(n - 1).

    Equals sum_d (n_d - 1) / (n_domains * (n - 1)) = (n - n_domains)/(n_d*(n-1)).
    Close to 1/n_domains but NOT equal to it (0.0633 vs 0.0667 here), because
    excluding the query itself shrinks its own domain's candidate pool.
    """
    nulls = per_domain_null(domains)
    return sum(nulls.values()) / len(nulls)


def self_retrieval_top1(name_codes: np.ndarray, formula_codes: np.ndarray) -> float:
    """Encoder sanity: does name_i retrieve formula_i as top-1 among all formulas?

    Ceiling on this corpus is 278/280 = 0.993, not 1.0: two `name` values
    (De Broglie Wavelength, Boltzmann Distribution) each appear twice with
    different formulas, so the duplicate-name pair embeds identically, shares
    a single argmax, and at most one of the two can be retrieved correctly.
    """
    sims = cosine_matrix(name_codes, formula_codes)
    return float((sims.argmax(axis=1) == np.arange(len(name_codes))).mean())


def loo_domain_routing(
    name_codes: np.ndarray, formula_codes: np.ndarray, domains: list[str]
) -> dict:
    """Leave-one-out domain routing, mirroring CubbyBridge._best_match.

    For each axiom i, pose name_i and route it to the domain owning the single
    highest-cosine INDIVIDUAL formula vector, with formula_i itself excluded --
    a real challenge must reach the right world via OTHER axioms, not itself.
    """
    sizes = {d: domains.count(d) for d in set(domains)}
    too_small = sorted(d for d, n in sizes.items() if n < 2)
    if too_small:
        raise ValueError(
            f"leave-one-out needs >=2 axioms per domain; got singletons: {too_small}"
        )

    sims = cosine_matrix(name_codes, formula_codes)
    np.fill_diagonal(sims, -np.inf)                    # leave-one-out
    doms = np.asarray(domains)
    picked = doms[sims.argmax(axis=1)]
    hit = picked == doms

    per_domain = {}
    for d in sorted(set(domains)):
        sel = doms == d
        per_domain[d] = float(hit[sel].mean())

    best = sims.max(axis=1)
    same = np.array([sims[i][doms == doms[i]].max() for i in range(len(doms))])
    diff = np.array([sims[i][doms != doms[i]].max() for i in range(len(doms))])
    return {
        "micro": float(hit.mean()),
        "macro": float(np.mean(list(per_domain.values()))),
        "per_domain": per_domain,
        "best_cos_mean": float(best.mean()),
        "same_cos": same,
        "diff_cos": diff,
    }


STAGE1_ARMS = ("hash", "ngram", "t-bag", "t-ctx", "ref")


def _encode_arm(arm: str, names: list[str], formulas: list[str]):
    """-> (name_codes, formula_codes) as (n, K, L) dense block codes."""
    if arm == "hash":
        return hash_encode(names), hash_encode(formulas)
    if arm == "ngram":
        return ngram_encode(names), ngram_encode(formulas)
    encoder = {"t-bag": tbag_encode, "t-ctx": tctx_encode, "ref": ref_encode}[arm]
    name_emb, formula_emb = encoder(names), encoder(formulas)
    P = make_projection(name_emb.shape[1])
    return project(P, name_emb), project(P, formula_emb)


def run_stage1() -> dict:
    names, formulas, domains = load_corpus()
    micro_chance = majority_class(domains)
    mac_chance = macro_chance(domains)
    print(f"corpus: {len(names)} axioms, {len(set(domains))} domains | "
          f"micro chance (majority-class) = {micro_chance:.3f} | "
          f"macro chance (1/{len(set(domains))}) = {mac_chance:.3f}\n")

    results = {"micro_chance": micro_chance, "macro_chance": mac_chance, "arms": {}}
    for arm in STAGE1_ARMS:
        name_codes, formula_codes = _encode_arm(arm, names, formulas)
        top1 = self_retrieval_top1(name_codes, formula_codes)
        routed = loo_domain_routing(name_codes, formula_codes, domains)
        results["arms"][arm] = {
            "self_retrieval_top1": top1,
            "micro": routed["micro"],
            "macro": routed["macro"],
            "per_domain": routed["per_domain"],
        }
        print(f"{arm:>6}: self-retrieval top1={top1:.3f} | "
              f"routing micro={routed['micro']:.3f} (vs {micro_chance:.3f}) "
              f"macro={routed['macro']:.3f} "
              f"({routed['macro'] / mac_chance:.2f}x macro-chance)")

    hash_macro = results["arms"]["hash"]["macro"]
    ref_macro = results["arms"]["ref"]["macro"]
    trunk = max(("t-bag", "t-ctx"), key=lambda a: results["arms"][a]["macro"])
    trunk_macro = results["arms"][trunk]["macro"]
    results["best_trunk"] = trunk

    print("\n--- kill criterion ---")
    # NOTE: every macro figure is compared against MACRO chance (1/n_domains).
    # Comparing macro against the majority-class rate would be a category error.
    checks = {
        "1 baseline sanity (hash macro <= 1.2x macro-chance)": hash_macro <= 1.2 * mac_chance,
        "2 encoder sanity (best trunk self-retrieval >= 0.50)":
            results["arms"][trunk]["self_retrieval_top1"] >= 0.50,
        "3a routing (best trunk macro >= 2x macro-chance)": trunk_macro >= 2 * mac_chance,
        "3b routing (best trunk >= 0.60x ref)": trunk_macro >= 0.60 * ref_macro,
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    results["checks"] = {k: bool(v) for k, v in checks.items()}

    ratio = trunk_macro / ref_macro if ref_macro else float("nan")
    verdict = ("ship the trunk" if ratio >= 0.90 else
               "fine-tune (M3) — gap is large" if ratio < 0.60 else
               "judgment call — document it")
    results["trunk_vs_ref"] = ratio
    results["decision"] = verdict
    print(f"\nbest trunk = {trunk}; trunk/ref = {ratio:.3f} -> {verdict}")
    return results


def argmax_onehot(codes: np.ndarray) -> np.ndarray:
    """One-hot at the max signed value per block — LSH-style on a random
    orthonormal projection. Cosine between two such codes is
    (#matching blocks)/K, i.e. 81 distinct levels at K=80."""
    idx = codes.argmax(axis=2)                                   # (n, K)
    out = np.zeros_like(codes, dtype=np.float32)
    n_idx, k_idx = np.meshgrid(
        np.arange(codes.shape[0]), np.arange(K), indexing="ij"
    )
    out[n_idx, k_idx, idx] = 1.0
    return out


def fit_pq(codes: np.ndarray, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Per-block k-means (hand-rolled Lloyd's; no scikit-learn) -> (K, L, L).

    Each block's L-dim sub-vectors are clustered into L centroids, so a code
    becomes one-hot at its nearest centroid per block.
    """
    if len(codes) < L:
        raise ValueError(
            f"fit_pq needs >= L={L} points to seed every centroid; got {len(codes)}. "
            f"Fewer points than centroids would silently leave {L - len(codes)} "
            f"centroids at an all-zero init -- report this, don't work around it."
        )
    rng = np.random.default_rng(seed)
    centroids = np.zeros((K, L, L), dtype=np.float32)
    for b in range(K):
        X = codes[:, b, :].astype(np.float32)                    # (n, L)
        start = rng.choice(len(X), size=min(L, len(X)), replace=False)
        C = np.zeros((L, L), dtype=np.float32)
        C[: len(start)] = X[start]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            assign = d.argmin(axis=1)
            for j in range(L):
                sel = assign == j
                if sel.any():
                    C[j] = X[sel].mean(axis=0)
        centroids[b] = C
    return centroids


def pq_onehot(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(codes, dtype=np.float32)
    for b in range(K):
        d = ((codes[:, b, :][:, None, :] - centroids[b][None, :, :]) ** 2).sum(axis=2)
        out[np.arange(len(codes)), b, d.argmin(axis=1)] = 1.0
    return out


def pq_reconstruct(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Replace each block with its NEAREST CENTROID VECTOR (not a one-hot label).

    This is what PQ retrieval actually stores-and-reconstructs: the code is an
    index, and the reconstruction is the centroid it points at. Comparing a dense
    query against the one-hot INDICATOR instead is meaningless -- the indicator's
    direction (a basis vector at the centroid's index) is unrelated to the
    centroid's direction.
    """
    idx = pq_onehot(codes, centroids).argmax(axis=2)                  # (n, K)
    return np.stack([centroids[b][idx[:, b]] for b in range(K)], axis=1).astype(np.float32)


def recalibrate_tau(same_cos: np.ndarray, diff_cos: np.ndarray) -> tuple[float, float]:
    """Pick tau at max Youden's J over same-domain vs different-domain maxima.
    Returns (tau, roc_auc). M1's 0.35 does not transfer to this distribution."""
    scores = np.concatenate([same_cos, diff_cos])
    labels = np.concatenate([np.ones(len(same_cos)), np.zeros(len(diff_cos))])
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    j = tpr - fpr
    return float(s[int(j.argmax())]), auc


def run_stage2(arm: str) -> tuple[dict, np.ndarray]:
    names, formulas, domains = load_corpus()
    # Report BOTH baselines, each labelled, so no reader can repeat the
    # macro-vs-majority-class category error. Bar 4 is dense-relative and does
    # not use either.
    micro_chance = majority_class(domains)
    mac_chance = macro_chance(domains)
    name_codes, formula_codes = _encode_arm(arm, names, formulas)

    variants = {"dense": (name_codes, formula_codes)}
    variants["argmax"] = (argmax_onehot(name_codes), argmax_onehot(formula_codes))
    cent = fit_pq(formula_codes)
    variants["pq"] = (pq_onehot(name_codes, cent), pq_onehot(formula_codes, cent))
    # ADC (Asymmetric Distance Computation): the query stays DENSE and only the
    # stored side is quantized -- then RECONSTRUCTED to its centroid vectors.
    # Mirrors the real bridge: axioms must be one-hot to participate in
    # bind/bundle, but a routing comparison never requires quantizing the query.
    variants["pq-adc"] = (name_codes, pq_reconstruct(formula_codes, cent))

    out = {"arm": arm, "micro_chance": micro_chance, "macro_chance": mac_chance,
           "variants": {}}
    for label, (nc, fc) in variants.items():
        routed = loo_domain_routing(nc, fc, domains)
        tau, auc = recalibrate_tau(routed["same_cos"], routed["diff_cos"])
        out["variants"][label] = {
            "micro": routed["micro"], "macro": routed["macro"],
            "tau_match": tau, "roc_auc": auc,
        }
        print(f"{label:>7}: micro={routed['micro']:.3f} macro={routed['macro']:.3f} "
              f"| tau={tau:.3f} auc={auc:.3f}")

    dense_macro = out["variants"]["dense"]["macro"]
    best = max(("argmax", "pq"), key=lambda v: out["variants"][v]["macro"])
    ok = out["variants"][best]["macro"] >= 0.80 * dense_macro
    print(f"\n--- kill criterion ---\n  [{'PASS' if ok else 'FAIL'}] "
          f"4 discretization (best one-hot '{best}' >= 0.80x dense)")
    if not ok:
        print("  => FINDING: semantic routing works only in DENSE space; the "
              "dense-vs-one-hot algebra choice is now explicit, not silent.")
    out["winner"] = best if ok else "dense"
    out["discretization_passed"] = bool(ok)
    return out, cent


def _stratified_half(domains: list[str], seed: int = 0):
    """Split indices in two, stratified by domain so every domain keeps >=2 members
    in each half (smallest domain here is 10, so each half gets >=5)."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for d in sorted(set(domains)):
        idx = [i for i, x in enumerate(domains) if x == d]
        rng.shuffle(idx)
        cut = len(idx) // 2
        a.extend(idx[:cut]); b.extend(idx[cut:])
    return np.array(sorted(a)), np.array(sorted(b))


def run_leakage_check(arm: str) -> dict:
    """Is 'quantized beats dense' real, or an artifact of fitting the codebook on
    the very items LOO excludes?

    Control A (disjoint SOURCE, same n): fit the codebook on the NAME side. Same
    280 points and same saturation regime as the shipped run, but zero formula
    content, so a surviving gain cannot come from formula leakage. Confound:
    names are a different surface modality.
    Control B (disjoint ITEMS, same modality): fit on formula half A, score only
    on half B, so no fitted item is ever a scored candidate. Confound: the
    codebook is fit on ~140 points, a more saturated regime.
    """
    names, formulas, domains = load_corpus()
    name_codes, formula_codes = _encode_arm(arm, names, formulas)
    out = {"arm": arm}

    # --- shipped (leaky) reference, recomputed here so the table is self-contained
    cent_all = fit_pq(formula_codes)
    out["shipped"] = {
        "dense": loo_domain_routing(name_codes, formula_codes, domains)["macro"],
        "argmax": loo_domain_routing(argmax_onehot(name_codes), argmax_onehot(formula_codes), domains)["macro"],
        "pq": loo_domain_routing(pq_onehot(name_codes, cent_all), pq_onehot(formula_codes, cent_all), domains)["macro"],
        "pq-adc": loo_domain_routing(name_codes, pq_reconstruct(formula_codes, cent_all), domains)["macro"],
    }

    # --- Control A: codebook fit on the NAME side (no formula content at all)
    cent_names = fit_pq(name_codes)
    out["control_a_name_fit"] = {
        "dense": out["shipped"]["dense"],
        "pq": loo_domain_routing(pq_onehot(name_codes, cent_names), pq_onehot(formula_codes, cent_names), domains)["macro"],
        "pq-adc": loo_domain_routing(name_codes, pq_reconstruct(formula_codes, cent_names), domains)["macro"],
    }

    # --- Control B: fit on formula half A, score only on half B
    A, B = _stratified_half(domains)
    cent_a = fit_pq(formula_codes[A])
    doms_b = [domains[i] for i in B]
    nb, fb = name_codes[B], formula_codes[B]
    out["control_b_halfsplit"] = {
        "dense": loo_domain_routing(nb, fb, doms_b)["macro"],
        "argmax": loo_domain_routing(argmax_onehot(nb), argmax_onehot(fb), doms_b)["macro"],
        "pq": loo_domain_routing(pq_onehot(nb, cent_a), pq_onehot(fb, cent_a), doms_b)["macro"],
        "pq-adc": loo_domain_routing(nb, pq_reconstruct(fb, cent_a), doms_b)["macro"],
        "n_scored": int(len(B)),
    }
    return out


NULL_KINDS = ("gauss", "onehot")


def _null_codes(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """A PURE-NOISE encoder: (n, K, L) codes carrying zero information.

    `gauss` matches the dense arms' format (t-bag / t-ctx / ref after the
    orthonormal projection, and ngram's multi-hot bundles); `onehot` matches the
    one-hot-per-block format of the `hash` arm and of the argmax/pq discretizers,
    where cosine quantizes to (#matching blocks)/K and exact ties are common.
    Both are needed: a tie-heavy code interacts with np.argmax's low-index
    tie-breaking in a way a continuous code does not.
    """
    if kind == "gauss":
        return rng.standard_normal((n, K, L)).astype(np.float32)
    if kind == "onehot":
        out = np.zeros((n, K, L), dtype=np.float32)
        idx = rng.integers(0, L, size=(n, K))
        out[np.arange(n)[:, None], np.arange(K)[None, :], idx] = 1.0
        return out
    raise ValueError(f"unknown null kind {kind!r}; expected one of {NULL_KINDS}")


def _tie_stats(name_codes: np.ndarray, formula_codes: np.ndarray) -> dict:
    """How often is the routing argmax a TIE, and how big is the tied group?

    One-hot-per-block cosine is (#matching blocks)/K, i.e. 81 discrete levels at
    K=80, so exact ties are routine -- and `np.argmax` silently breaks every tie
    toward the LOWEST index. In a domain-GROUPED corpus (data/mowm_axioms.json
    stores all 15 domains as contiguous runs) that is a systematic pull toward
    the first-listed domains. Continuous codes have essentially no ties, which is
    why the one-hot null and the dense null differ at all.
    """
    sims = cosine_matrix(name_codes, formula_codes)
    np.fill_diagonal(sims, -np.inf)
    groups = (np.abs(sims - sims.max(axis=1, keepdims=True)) < 1e-9).sum(axis=1)
    return {
        "tied_argmax_frac": float((groups > 1).mean()),
        "tie_group_mean": float(groups.mean()),
    }


def run_null_baseline(seeds: int = 10) -> dict:
    """Measure the NULL of every Stage-1/Stage-2 statistic, empirically.

    WHY THIS EXISTS. Three of this screen's statistics have nulls that are NOT
    the baselines they were first read against:

      * `micro` was read against the majority-class rate (0.300). That is a
        legitimate strong baseline, but the RANDOM-ENCODER floor for an
        argmax-over-279-candidates metric is matched_micro_null() = 0.1226.
      * `macro` was read against 1/n_domains = 0.0667; the matched null is
        matched_macro_null() = 0.0633.
      * `roc_auc` was read against 0.5. It cannot be: same_cos[i] is a max over
        (n_d - 1) same-domain candidates while diff_cos[i] is a max over
        (n - n_d) different-domain candidates, and the max of more draws is
        stochastically larger. On this corpus that is ~14 vs ~265 draws, so the
        pooled unpaired statistic has a null FAR below 0.5 BY CONSTRUCTION.

    Method: feed pure noise through the SHIPPED loo_domain_routing /
    self_retrieval_top1 / recalibrate_tau (unchanged) over the real corpus's
    domain structure, several seeds, and report mean +/- sd. The `balanced`
    structure (two equal domains, so ~139 vs 140 candidates) is the control that
    shows the sub-0.5 AUC null is a CANDIDATE-SET-SIZE artifact and not a defect
    of the encoders: it should land near 0.5.
    """
    names, formulas, domains = load_corpus()
    n = len(domains)
    balanced = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    structures = {"corpus": domains, "balanced": balanced}

    out = {"n": n, "seeds": seeds, "structures": {}}
    for sname, doms in structures.items():
        entry = {
            "n_domains": len(set(doms)),
            "analytic": {
                "matched_micro_null": matched_micro_null(doms),
                "matched_macro_null": matched_macro_null(doms),
                "majority_class": majority_class(doms),
                "macro_chance_1_over_n": macro_chance(doms),
                "per_domain_null": per_domain_null(doms),
            },
            "kinds": {},
        }
        print(f"\n=== structure: {sname} ({entry['n_domains']} domains, n={n}) ===")
        a = entry["analytic"]
        print(f"  analytic matched nulls: micro={a['matched_micro_null']:.4f} "
              f"macro={a['matched_macro_null']:.4f} "
              f"| for contrast: majority-class={a['majority_class']:.4f}, "
              f"1/n_domains={a['macro_chance_1_over_n']:.4f}")

        for kind in NULL_KINDS:
            rows = []
            for s in range(seeds):
                rng = np.random.default_rng(1000 + s)
                # name and formula codes drawn INDEPENDENTLY: a "name" carries
                # zero information about its own "formula", which is the null.
                nc = _null_codes(n, kind, rng)
                fc = _null_codes(n, kind, rng)
                routed = loo_domain_routing(nc, fc, doms)
                tau, auc = recalibrate_tau(routed["same_cos"], routed["diff_cos"])
                rows.append({
                    "seed": 1000 + s,
                    "self_retrieval_top1": self_retrieval_top1(nc, fc),
                    "micro": routed["micro"],
                    "macro": routed["macro"],
                    "roc_auc": auc,
                    "tau_match": tau,
                    "per_domain": routed["per_domain"],
                    **_tie_stats(nc, fc),
                })
            summary = {}
            for key in ("self_retrieval_top1", "micro", "macro", "roc_auc", "tau_match",
                        "tied_argmax_frac", "tie_group_mean"):
                v = np.array([r[key] for r in rows], dtype=np.float64)
                summary[key] = {
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()),
                }
            per_dom_mean = {
                d: float(np.mean([r["per_domain"][d] for r in rows]))
                for d in sorted(set(doms))
            }
            entry["kinds"][kind] = {
                "summary": summary,
                "per_domain_mean": per_dom_mean,
                "per_seed": rows,
            }
            m = summary
            print(f"  [{kind:>6}] micro={m['micro']['mean']:.4f}+/-{m['micro']['sd']:.4f} "
                  f"macro={m['macro']['mean']:.4f}+/-{m['macro']['sd']:.4f} "
                  f"auc={m['roc_auc']['mean']:.4f}+/-{m['roc_auc']['sd']:.4f} "
                  f"self-top1={m['self_retrieval_top1']['mean']:.4f}"
                  f"+/-{m['self_retrieval_top1']['sd']:.4f}")
        out["structures"][sname] = entry

    # Per-domain detail for the corpus structure: the baseline SCALES WITH
    # DOMAIN SIZE, so a flat per-domain baseline does not exist.
    print("\n--- per-domain null, corpus structure (baseline is NOT flat) ---")
    print(f"{'domain':>12}{'n_d':>6}{'(n_d-1)/279':>14}{'gauss':>10}{'onehot':>10}")
    corpus = out["structures"]["corpus"]
    for d, null in sorted(corpus["analytic"]["per_domain_null"].items()):
        nd = domains.count(d)
        g = corpus["kinds"]["gauss"]["per_domain_mean"][d]
        o = corpus["kinds"]["onehot"]["per_domain_mean"][d]
        print(f"{d:>12}{nd:>6}{null:>14.4f}{g:>10.4f}{o:>10.4f}")

    # The `hash` arm IS a draw from the one-hot null by construction (same
    # BLAKE2b-seeded one-hot-per-block format, and a name's code carries no
    # information about its own formula's code), so its own tie statistics
    # belong in this characterization.
    hash_ties = _tie_stats(hash_encode(names), hash_encode(formulas))
    out["hash_arm_tie_stats"] = hash_ties
    print(f"\nhash arm on the real corpus: tied argmax on "
          f"{hash_ties['tied_argmax_frac']:.3f} of rows, mean tie-group "
          f"{hash_ties['tie_group_mean']:.2f} candidates")

    print("\n--- read ---")
    cg = corpus["kinds"]["gauss"]["summary"]
    co = corpus["kinds"]["onehot"]["summary"]
    bg = out["structures"]["balanced"]["kinds"]["gauss"]["summary"]
    print(f"  AUC null on the corpus structure is {cg['roc_auc']['mean']:.4f} "
          f"(gauss) / {co['roc_auc']['mean']:.4f} (onehot), NOT 0.5.")
    print(f"  AUC null on a BALANCED structure is {bg['roc_auc']['mean']:.4f} "
          f"(gauss) -> the sub-0.5 null is a candidate-set-size artifact.")
    print(f"  Empirical micro null (gauss) {cg['micro']['mean']:.4f} vs analytic "
          f"{corpus['analytic']['matched_micro_null']:.4f}.")
    print("  A one-hot noise encoder scores ABOVE the analytic macro null "
          f"({co['macro']['mean']:.4f} vs "
          f"{corpus['analytic']['matched_macro_null']:.4f}) because one-hot "
          "cosines tie constantly and np.argmax breaks ties toward LOW indices, "
          "which in a domain-GROUPED corpus means the first-listed domain.")
    return out


def run_fast_arms() -> dict:
    """Latency-constrained routing: how far does a MODEL-FREE challenge encoder get?

    Axioms are encoded OFFLINE regardless (H0's frozen shape), so latency only
    binds the CHALLENGE side. Every arm below is BLAKE2b hashing plus
    scatter-adds -- no torch, no model in the hot path, the same primitive the
    Rust VM / Vulkan shaders can reimplement. The `ngram3` row is M2's shipped
    encoder unchanged and must reproduce Stage 1's 0.1400 macro exactly (the
    harness regression check). The IDF table + gram vocabulary is the fast
    path's offline "codebook"; MiniLM appears only as the quality/latency
    reference. The K-sweep at the end asks whether dimensionality is a binding
    constraint for the bundled codes (one-hot cosine has K+1 levels; bundle
    crosstalk ~ sqrt(m/D)) -- or whether the ceiling is the lexical signal.
    """
    import platform
    import time

    names, formulas, domains = load_corpus()
    mac_null = matched_macro_null(domains)
    mic_null = matched_micro_null(domains)
    texts = names + formulas

    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print("mode: --fast-arms -- model-free challenge encoders vs latency; no checkpoint,")
    print("      no tokenizer; MiniLM loaded ONLY for the reference row's timing")
    print(f"corpus: {len(names)} axioms, {len(set(domains))} domains | "
          f"matched macro null {mac_null:.4f} | matched micro null {mic_null:.4f}")
    print(f"self-retrieval ceiling 278/280 = 0.993 (two duplicate names)\n")

    arms: dict[str, dict | None] = {
        "ngram3 (M2 baseline)": None,
        "var-n 2-4": {"ns": (2, 3, 4), "words": False, "idf": False},
        "var-n 2-4 + idf": {"ns": (2, 3, 4), "words": False, "idf": True},
        "var-n 2-4 + words + idf": {"ns": (2, 3, 4), "words": True, "idf": True},
        "words only + idf": {"ns": (), "words": True, "idf": True},
    }

    print(f"{'arm':>26} {'self':>6} {'macro':>7} {'xnull':>6} {'micro':>7} "
          f"{'cold-us':>8} {'warm-us':>8}")
    out: dict = {"matched_macro_null": mac_null, "matched_micro_null": mic_null,
                 "l": L, "k": K, "arms": {}}
    for label, cfg in arms.items():
        _GRAM_CACHE.clear()
        idf = default = None
        if cfg is not None and cfg["idf"]:
            idf, default = build_idf(texts, cfg["ns"], cfg["words"])   # offline cost
        t0 = time.perf_counter()
        if cfg is None:
            nc, fc = ngram_encode(names), ngram_encode(formulas)
        else:
            nc = ngram_encode_v2(names, cfg["ns"], cfg["words"], idf, default)
            fc = ngram_encode_v2(formulas, cfg["ns"], cfg["words"], idf, default)
        cold = (time.perf_counter() - t0) / len(texts) * 1e6
        t0 = time.perf_counter()
        if cfg is None:
            ngram_encode(names)
        else:
            ngram_encode_v2(names, cfg["ns"], cfg["words"], idf, default)
        warm = (time.perf_counter() - t0) / len(names) * 1e6
        top1 = self_retrieval_top1(nc, fc)
        routed = loo_domain_routing(nc, fc, domains)
        out["arms"][label] = {
            "self_retrieval_top1": top1,
            "macro": routed["macro"], "macro_x_null": routed["macro"] / mac_null,
            "micro": routed["micro"], "per_domain": routed["per_domain"],
            "cold_us_per_text": cold, "warm_us_per_text": warm,
            "config": cfg,
        }
        print(f"{label:>26} {top1:6.3f} {routed['macro']:7.4f} "
              f"{routed['macro'] / mac_null:6.2f} {routed['micro']:7.4f} "
              f"{cold:8.0f} {warm:8.0f}")

    # -- token-ID arms: use the numbers the serving pipeline ALREADY has -------
    #    (the trunk tokenizes every challenge anyway; ids are free at routing
    #    time, so measured timing -- which includes tokenization -- overstates
    #    the marginal cost)
    encode_fn, _tok_vocab = load_tokenizer()
    token_sets = [set(encode_fn(t.lower())) for t in texts]
    df_tok: dict[int, int] = {}
    for s_ in token_sets:
        for tid in s_:
            df_tok[tid] = df_tok.get(tid, 0) + 1
    idf_tok = {tid: float(np.log(len(texts) / c)) for tid, c in df_tok.items()}
    default_tok = float(np.log(len(texts)))
    for label, use_idf in (("bpe tokens (ids as keys)", False),
                           ("bpe tokens + idf", True)):
        _GRAM_CACHE.clear()
        t0 = time.perf_counter()
        nc = token_encode(names, encode_fn, idf_tok if use_idf else None, default_tok)
        fc = token_encode(formulas, encode_fn, idf_tok if use_idf else None, default_tok)
        cold = (time.perf_counter() - t0) / len(texts) * 1e6
        t0 = time.perf_counter()
        token_encode(names, encode_fn, idf_tok if use_idf else None, default_tok)
        warm = (time.perf_counter() - t0) / len(names) * 1e6
        top1 = self_retrieval_top1(nc, fc)
        routed = loo_domain_routing(nc, fc, domains)
        out["arms"][label] = {
            "self_retrieval_top1": top1,
            "macro": routed["macro"], "macro_x_null": routed["macro"] / mac_null,
            "micro": routed["micro"], "per_domain": routed["per_domain"],
            "cold_us_per_text": cold, "warm_us_per_text": warm,
            "config": {"tokenizer": "grillcheese_bbpe128k", "idf": use_idf,
                       "note": "timing includes tokenization; ids are free in serving"},
        }
        print(f"{label:>26} {top1:6.3f} {routed['macro']:7.4f} "
              f"{routed['macro'] / mac_null:6.2f} {routed['micro']:7.4f} "
              f"{cold:8.0f} {warm:8.0f}")

    # -- reference row: quality from the committed Stage-1 artifact; latency
    #    measured here (model pre-loaded; batched AND single-query, since a
    #    routing challenge arrives alone in serving).
    stage1 = json.loads(
        (ROOT / "validation" / "logs" / "exp_f2m2_stage1.json").read_text(encoding="utf-8"))
    ref = stage1["arms"]["ref"]
    ref_row: dict = {"self_retrieval_top1": ref["self_retrieval_top1"],
                     "macro": ref["macro"], "macro_x_null": ref["macro"] / mac_null,
                     "micro": ref["micro"], "quality_source": "exp_f2m2_stage1.json"}
    try:
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        load_s = time.perf_counter() - t0
        model.encode(names[:4], show_progress_bar=False)                 # warmup
        t0 = time.perf_counter()
        model.encode(names, batch_size=64, show_progress_bar=False)
        batched_us = (time.perf_counter() - t0) / len(names) * 1e6
        t0 = time.perf_counter()
        for t in names[:32]:
            model.encode([t], show_progress_bar=False)
        single_us = (time.perf_counter() - t0) / 32 * 1e6
        ref_row.update({"model_load_s": load_s, "batched_us_per_text": batched_us,
                        "single_query_us_per_text": single_us})
        print(f"{'MiniLM ref (stage1)':>26} {ref['self_retrieval_top1']:6.3f} "
              f"{ref['macro']:7.4f} {ref['macro'] / mac_null:6.2f} {ref['micro']:7.4f} "
              f"{'--':>8} {batched_us:8.0f}   "
              f"(single-query {single_us / 1000:.1f} ms; +{load_s:.1f}s model load)")
    except Exception as exc:                                             # pragma: no cover
        print(f"{'MiniLM ref (stage1)':>26} latency not measured ({exc})")
    out["arms"]["MiniLM ref (stage1)"] = ref_row

    # -- K-sweep on the best lexical arm: is dimensionality the constraint? ----
    best_label, best_cfg = max(
        ((lb, c) for lb, c in arms.items() if c is not None),
        key=lambda lc: out["arms"][lc[0]]["macro"])
    print(f"\nK-sweep on best lexical arm ({best_label}); l={L} fixed, one seed per K:")
    k0 = K
    sweep: dict = {}
    for k_try in (80, 160, 320):
        globals()["K"] = k_try
        _GRAM_CACHE.clear()
        idf = default = None
        if best_cfg["idf"]:
            idf, default = build_idf(texts, best_cfg["ns"], best_cfg["words"])
        nc = ngram_encode_v2(names, best_cfg["ns"], best_cfg["words"], idf, default)
        fc = ngram_encode_v2(formulas, best_cfg["ns"], best_cfg["words"], idf, default)
        r = loo_domain_routing(nc, fc, domains)
        sweep[str(k_try)] = {"d": k_try * L, "macro": r["macro"], "micro": r["micro"]}
        print(f"  K={k_try:>3} (d={k_try * L:>6}): macro {r['macro']:.4f} "
              f"({r['macro'] / mac_null:.2f}x null)  micro {r['micro']:.4f}")
    globals()["K"] = k0
    _GRAM_CACHE.clear()
    out["k_sweep"] = {"arm": best_label, "l": L, "results": sweep,
                      "note": "different K redraws every gram layout, so small deltas "
                              "are seed-level noise; only a clear monotone trend counts"}

    fast_labels = [lb for lb in out["arms"] if lb != "MiniLM ref (stage1)"]
    top_label = max(fast_labels, key=lambda lb: out["arms"][lb]["macro"])
    fb = out["arms"][top_label]
    print(f"\nbest model-free arm: {top_label} -- macro {fb['macro']:.4f} "
          f"({fb['macro_x_null']:.2f}x null, {fb['macro'] / ref['macro']:.2f}x MiniLM) "
          f"at ~{fb['warm_us_per_text']:.0f} us/challenge warm")
    out["best_fast_arm"] = top_label

    (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "validation" / "logs" / "exp_f2m2_fast_arms.json"
    json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {json_path}")
    return out


# ---------------------------------------------------------------------------
# BEAGLE-style distributional memory (user-proposed; Jones & Mewhort family):
# a token's MEMORY vector = superposition of the SIGNAL codes of tokens it
# co-occurs with over a real corpus. Semantics from statistics, no gradients,
# and the corpus pass is OFFLINE -- serving stays table-lookup + scatter-adds.
# ---------------------------------------------------------------------------

_SPECIALS_FROM = 127933          # <pad>/<unk>/<s>/</s>/chat specials start here
_FULL_VOCAB = 127996


def beagle_build_memory(
    interest_ids: np.ndarray,
    shard_path: str,
    n_tokens: int = 200_000_000,
    window: int = 5,
    max_positions: int = 4_000_000,
    top_neighbors: int = 8000,
    chunk: int = 20_000_000,
    seed: int = 0,
) -> dict:
    """One streaming pass of BEAGLE context accumulation for the interest ids.

    Counting is exact over a position subsample; the projection through signal
    codes runs once per unique (row, neighbor) pair via per-row bincount, so
    cost scales with unique pairs, not corpus length. Returns three memory
    tables (raw counts, log1p-damped, PPMI-weighted), each (n_interest, K, L).
    """
    import time

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lut_row = np.full(_FULL_VOCAB, -1, dtype=np.int64)
    lut_row[np.asarray(interest_ids, dtype=np.int64)] = np.arange(len(interest_ids))

    corpus = np.memmap(shard_path, dtype=np.uint32, mode="r")
    n_tokens = int(min(n_tokens, len(corpus)))
    offsets = [d for d in range(-window, window + 1) if d != 0]
    n_chunks = max(1, n_tokens // chunk)
    budget_per_chunk = max(1, max_positions // n_chunks)

    key_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []
    seen = used = 0
    for a in range(0, n_tokens, chunk):
        seg = np.asarray(corpus[a:min(a + chunk, n_tokens)]).astype(np.int64)
        rows_here = lut_row[np.clip(seg, 0, _FULL_VOCAB - 1)]
        pos = np.nonzero(rows_here >= 0)[0]
        pos = pos[(pos >= window) & (pos < len(seg) - window)]
        seen += len(pos)
        if len(pos) > budget_per_chunk:
            pos = rng.choice(pos, size=budget_per_chunk, replace=False)
        used += len(pos)
        if not len(pos):
            continue
        tgt = rows_here[pos]
        for d in offsets:
            nb = seg[pos + d]
            ok = nb < _SPECIALS_FROM
            keys = tgt[ok] * _FULL_VOCAB + nb[ok]
            uk, uc = np.unique(keys, return_counts=True)
            key_parts.append(uk)
            cnt_parts.append(uc.astype(np.int64))
        print(f"  scanned {min(a + chunk, n_tokens) / 1e6:6.0f}M tokens | "
              f"interest positions seen {seen / 1e6:.2f}M used {used / 1e6:.2f}M", flush=True)

    keys = np.concatenate(key_parts); del key_parts
    cnts = np.concatenate(cnt_parts); del cnt_parts
    uk, inv = np.unique(keys, return_inverse=True)
    counts = np.zeros(len(uk), dtype=np.int64)
    np.add.at(counts, inv, cnts)
    del keys, cnts, inv
    pair_rows = (uk // _FULL_VOCAB).astype(np.int64)
    pair_nbs = (uk % _FULL_VOCAB).astype(np.int64)
    del uk

    # cap unique neighbors per row (largest counts first) so no row's projection
    # is dominated by an unbounded stop-token tail
    order = np.lexsort((-counts, pair_rows))
    pair_rows, pair_nbs, counts = pair_rows[order], pair_nbs[order], counts[order]
    starts = np.nonzero(np.r_[True, np.diff(pair_rows) > 0])[0]
    seg_len = np.diff(np.r_[starts, len(pair_rows)])
    rank = np.arange(len(pair_rows)) - np.repeat(starts, seg_len)
    keep = rank < top_neighbors
    pair_rows, pair_nbs, counts = pair_rows[keep], pair_nbs[keep], counts[keep]

    # signal codes for every unique neighbor (same seeding as token_encode)
    unb = np.unique(pair_nbs)
    sig_lut = np.zeros((len(unb), K), dtype=np.int64)
    for j, tid in enumerate(unb):
        sig_lut[j] = np.random.default_rng(1_000_003 + int(tid)).integers(0, L, size=K)
    nb_slot = np.searchsorted(unb, pair_nbs)
    sig_pairs = sig_lut[nb_slot]                                    # (P, K)

    # weights: raw counts / log damping / PPMI (the count -> semantics transform)
    total = float(counts.sum())
    row_tot = np.zeros(len(interest_ids)); np.add.at(row_tot, pair_rows, counts.astype(np.float64))
    nb_tot = np.zeros(len(unb)); np.add.at(nb_tot, nb_slot, counts.astype(np.float64))
    w_raw = counts.astype(np.float64)
    w_log = np.log1p(w_raw)
    with np.errstate(divide="ignore"):
        pmi = np.log((w_raw * total) / (row_tot[pair_rows] * nb_tot[nb_slot]))
    w_ppmi = np.maximum(0.0, pmi)

    base = (np.arange(K) * L)[None, :]
    flat_all = (base + sig_pairs)                                   # (P, K)

    def _project(w: np.ndarray) -> np.ndarray:
        mem = np.zeros((len(interest_ids), K * L), dtype=np.float64)
        for r in range(len(interest_ids)):
            lo = np.searchsorted(pair_rows, r)
            hi = np.searchsorted(pair_rows, r + 1)
            if lo == hi:
                continue
            mem[r] = np.bincount(flat_all[lo:hi].ravel(),
                                 weights=np.repeat(w[lo:hi], K), minlength=K * L)
        return mem.reshape(len(interest_ids), K, L).astype(np.float32)

    out = {
        "raw": _project(w_raw), "log": _project(w_log), "ppmi": _project(w_ppmi),
        "lut_row": lut_row,
        "stats": {
            "shard": str(shard_path), "n_tokens": n_tokens, "window": window,
            "positions_seen": int(seen), "positions_used": int(used),
            "unique_pairs_kept": int(len(pair_rows)),
            "rows_total": int(len(interest_ids)),
            "rows_with_mass": int((row_tot > 0).sum()),
            "top_neighbors_cap": top_neighbors,
            "elapsed_s": time.perf_counter() - t0,
        },
    }
    return out


def beagle_encode(
    texts: list[str],
    encode_fn,
    mem: np.ndarray | None,
    lut_row: np.ndarray,
    idf: dict[int, float],
    default_idf: float,
    blend_signal: bool = False,
) -> np.ndarray:
    """IDF-weighted bundle of MEMORY vectors of a text's token ids (RAW case,
    matching the corpus). Memory rows are L2-normalized so corpus-frequent
    tokens don't dominate by mass; a token with no accumulated memory falls
    back to its SIGNAL code, so surface identity still contributes. mem=None
    is the pure-signal control at matched case handling; blend_signal=True
    adds each token's own signal code alongside its memory (surface + meaning).
    """
    out = np.zeros((len(texts), K, L), dtype=np.float32)
    rows_k = np.arange(K)
    inv_sqrt_k = 1.0 / np.sqrt(K)
    for i, text in enumerate(texts):
        acc = np.zeros(K * L, dtype=np.float64)
        for tid in set(encode_fn(text)):
            if tid >= _SPECIALS_FROM:
                continue
            w = idf.get(tid, default_idf)
            sig = np.random.default_rng(1_000_003 + int(tid)).integers(0, L, size=K)
            sig_vec = np.zeros(K * L)
            sig_vec[rows_k * L + sig] = inv_sqrt_k                  # unit norm
            v = None
            row = lut_row[tid] if 0 <= tid < len(lut_row) else -1
            if mem is not None and row >= 0:
                m = mem[row].ravel().astype(np.float64)
                nrm = np.linalg.norm(m)
                if nrm > 0:
                    v = m / nrm
            if v is None:
                v = sig_vec
            elif blend_signal:
                v = 0.5 * (v + sig_vec)
            acc += w * v
        out[i] = acc.reshape(K, L).astype(np.float32)
    return out


def run_beagle(shard_path: str, n_tokens: int, window: int) -> dict:
    """The BEAGLE arm: does distributional superposition close the semantic gap
    the surface-lexical arms cap at -- while keeping serving at table lookups?
    """
    import platform
    import time

    from cubbyllm.training.data import _load_tokenizer

    names, formulas, domains = load_corpus()
    mac_null = matched_macro_null(domains)
    texts = names + formulas
    encode_fn, decode_fn, _eos, _vocab = _load_tokenizer(str(TOKENIZER))

    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"mode: --beagle -- distributional memory from {pathlib.Path(shard_path).name}, "
          f"first {n_tokens / 1e6:.0f}M tokens, window +/-{window}")
    print("case handling: RAW (matches the corpus; the earlier token arm lowercased)\n")

    # interest ids + raw-case IDF over the 560 texts
    tok_sets = [set(t for t in encode_fn(x) if t < _SPECIALS_FROM) for x in texts]
    interest = np.array(sorted(set().union(*tok_sets)), dtype=np.int64)
    df_tok: dict[int, int] = {}
    for s_ in tok_sets:
        for tid in s_:
            df_tok[tid] = df_tok.get(tid, 0) + 1
    idf = {tid: float(np.log(len(texts) / c)) for tid, c in df_tok.items()}
    default_idf = float(np.log(len(texts)))
    print(f"interest tokens: {len(interest)} unique ids across {len(texts)} texts")

    built = beagle_build_memory(interest, shard_path, n_tokens=n_tokens, window=window)
    st = built["stats"]
    print(f"memory built in {st['elapsed_s']:.0f}s | positions used {st['positions_used'] / 1e6:.2f}M "
          f"| pairs kept {st['unique_pairs_kept'] / 1e6:.2f}M "
          f"| rows with mass {st['rows_with_mass']}/{st['rows_total']}\n")

    # neighborhood demo: are the memory vectors semantic at all?
    probe_words = ["force", "energy", "probability"]
    ppmi = built["ppmi"]
    norms = np.linalg.norm(ppmi.reshape(len(interest), -1), axis=1)
    normed = ppmi.reshape(len(interest), -1) / np.maximum(norms, 1e-12)[:, None]
    for w_ in probe_words:
        ids = [t for t in encode_fn(w_) if t < _SPECIALS_FROM]
        if not ids or built["lut_row"][ids[0]] < 0:
            continue
        r = int(built["lut_row"][ids[0]])
        sims = normed @ normed[r]
        top = np.argsort(-sims)[1:6]
        nbs = ", ".join(f"{decode_fn([int(interest[j])])!r}({sims[j]:.2f})" for j in top)
        print(f"  nearest to {decode_fn([ids[0]])!r}: {nbs}")
    print()

    fast = json.loads((ROOT / "validation" / "logs" / "exp_f2m2_fast_arms.json")
                      .read_text(encoding="utf-8"))
    print(f"{'arm':>26} {'self':>6} {'macro':>7} {'xnull':>6} {'micro':>7}")
    for lb in ("words only + idf", "bpe tokens + idf", "MiniLM ref (stage1)"):
        fa = fast["arms"][lb]
        print(f"{lb:>26} {fa['self_retrieval_top1']:6.3f} {fa['macro']:7.4f} "
              f"{fa['macro'] / mac_null:6.2f} {fa['micro']:7.4f}   (prior run)")

    out: dict = {"stats": st, "matched_macro_null": mac_null, "arms": {}}
    arms = {
        "signal only (raw case)": (None, False),
        "beagle raw": (built["raw"], False),
        "beagle log": (built["log"], False),
        "beagle ppmi": (built["ppmi"], False),
        "beagle ppmi + signal": (built["ppmi"], True),
    }
    for label, (mem, blend) in arms.items():
        t0 = time.perf_counter()
        nc = beagle_encode(names, encode_fn, mem, built["lut_row"], idf, default_idf, blend)
        fc = beagle_encode(formulas, encode_fn, mem, built["lut_row"], idf, default_idf, blend)
        us = (time.perf_counter() - t0) / len(texts) * 1e6
        top1 = self_retrieval_top1(nc, fc)
        routed = loo_domain_routing(nc, fc, domains)
        out["arms"][label] = {
            "self_retrieval_top1": top1, "macro": routed["macro"],
            "macro_x_null": routed["macro"] / mac_null, "micro": routed["micro"],
            "per_domain": routed["per_domain"], "encode_us_per_text": us,
        }
        print(f"{label:>26} {top1:6.3f} {routed['macro']:7.4f} "
              f"{routed['macro'] / mac_null:6.2f} {routed['micro']:7.4f}   ({us:.0f} us/text)")

    best = max(out["arms"], key=lambda lb: out["arms"][lb]["macro"])
    out["best_arm"] = best
    print(f"\nbest beagle-mode arm: {best} -- macro {out['arms'][best]['macro']:.4f} "
          f"({out['arms'][best]['macro_x_null']:.2f}x null) vs words+idf 0.1730 / MiniLM 0.3580")

    (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "validation" / "logs" / "exp_f2m2_beagle.json"
    json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {json_path}")
    return out


# ---------------------------------------------------------------------------
# model2vec-style static distillation: run the TEACHER once, offline, over the
# vocabulary's strings; serve from the resulting table with lookup + adds.
# The model never enters the hot path. Ablation against the random-code token
# arm isolates exactly what learned vectors buy over seeded noise.
# ---------------------------------------------------------------------------

def _remove_top_pc(X: np.ndarray, n_pc: int = 1) -> np.ndarray:
    """All-but-the-top: mean-center, strip the top principal component(s),
    re-normalize rows. The classic anisotropy fix, applied to the TABLE."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(n_pc):
        Xc = Xc - np.outer(Xc @ Vt[i], Vt[i])
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _bundle_from_table(
    texts: list[str],
    keys_of,                       # text -> iterable of table keys
    vec_of: dict,                  # key -> unit vector (all same dim)
    weight_of,                     # key -> float
    dim: int,
) -> np.ndarray:
    """The one serving primitive every fast arm shares: lookup + weighted add."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for k_ in keys_of(t):
            v = vec_of.get(k_)
            if v is not None:
                out[i] += weight_of(k_) * v
    return out


def run_m2v() -> dict:
    """Distill MiniLM into static per-token / per-word tables and route with
    the same lookup+add path as the random-code arms. Teacher cost is OFFLINE
    (embed ~2k vocabulary strings once); serving never touches the model."""
    import platform
    import time

    from cubbyllm.training.data import _load_tokenizer
    from sentence_transformers import SentenceTransformer

    names, formulas, domains = load_corpus()
    mac_null = matched_macro_null(domains)
    texts = names + formulas
    encode_fn, decode_fn, _e, _v = _load_tokenizer(str(TOKENIZER))

    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print("mode: --m2v -- static distillation of MiniLM into token/word tables;")
    print("      teacher runs ONCE offline, serving is table lookup + adds\n")

    # -- vocabulary of this corpus: lowered BPE ids and lowered words ----------
    tok_sets = [set(t for t in encode_fn(x.lower()) if t < _SPECIALS_FROM) for x in texts]
    tok_ids = sorted(set().union(*tok_sets))
    df_tok: dict[int, int] = {}
    for s_ in tok_sets:
        for tid in s_:
            df_tok[tid] = df_tok.get(tid, 0) + 1
    idf_tok = {tid: float(np.log(len(texts) / c)) for tid, c in df_tok.items()}
    dflt = float(np.log(len(texts)))

    word_sets = [{w for w in re.split(r"[^a-z0-9]+", x.lower()) if len(w) >= 2} for x in texts]
    words = sorted(set().union(*word_sets))
    df_w: dict[str, int] = {}
    for s_ in word_sets:
        for w_ in s_:
            df_w[w_] = df_w.get(w_, 0) + 1
    idf_w = {w_: float(np.log(len(texts) / c)) for w_, c in df_w.items()}

    # -- the offline distillation pass ----------------------------------------
    t0 = time.perf_counter()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tok_strings = [decode_fn([int(t)]) for t in tok_ids]
    E_tok = _unit_rows(np.asarray(model.encode(tok_strings, batch_size=256,
                                               show_progress_bar=False), dtype=np.float32))
    E_wrd = _unit_rows(np.asarray(model.encode(words, batch_size=256,
                                               show_progress_bar=False), dtype=np.float32))
    build_s = time.perf_counter() - t0
    print(f"distilled tables: {len(tok_ids)} token strings + {len(words)} words "
          f"in {build_s:.1f}s (one-off, offline)\n")

    E_tok_pc = _remove_top_pc(E_tok)
    E_wrd_pc = _remove_top_pc(E_wrd)

    # block-space forms for the surface-blend arms (cosine-exact projection)
    P384 = make_projection(384)
    rows_k = np.arange(K)
    inv_sqrt_k = 1.0 / np.sqrt(K)

    def _tok_signal(tid: int) -> np.ndarray:
        sig = np.random.default_rng(1_000_003 + int(tid)).integers(0, L, size=K)
        v = np.zeros(K * L, dtype=np.float32)
        v[rows_k * L + sig] = inv_sqrt_k
        return v

    def _word_signal(w_: str) -> np.ndarray:
        digest = hashlib.blake2b(f"w:{w_}".encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**63)
        sig = np.random.default_rng(seed).integers(0, L, size=K)
        v = np.zeros(K * L, dtype=np.float32)
        v[rows_k * L + sig] = inv_sqrt_k
        return v

    tok_keys = lambda t: set(x for x in encode_fn(t.lower()) if x < _SPECIALS_FROM)  # noqa: E731
    wrd_keys = lambda t: {w for w in re.split(r"[^a-z0-9]+", t.lower()) if len(w) >= 2}  # noqa: E731

    def _blend(table_pc: np.ndarray, keys, signal_fn) -> dict:
        proj = _unit_rows(table_pc @ P384.T)                       # (n, K*L), unit
        return {k_: 0.5 * proj[j] + 0.5 * signal_fn(k_) for j, k_ in enumerate(keys)}

    arms: dict[str, tuple] = {
        "m2v tokens + idf": ({t: E_tok[j] for j, t in enumerate(tok_ids)},
                             tok_keys, lambda k_: idf_tok.get(k_, dflt), 384),
        "m2v tokens -pc1 + idf": ({t: E_tok_pc[j] for j, t in enumerate(tok_ids)},
                                  tok_keys, lambda k_: idf_tok.get(k_, dflt), 384),
        "m2v words + idf": ({w_: E_wrd[j] for j, w_ in enumerate(words)},
                            wrd_keys, lambda k_: idf_w.get(k_, dflt), 384),
        "m2v words -pc1 + idf": ({w_: E_wrd_pc[j] for j, w_ in enumerate(words)},
                                 wrd_keys, lambda k_: idf_w.get(k_, dflt), 384),
        "m2v tok -pc1 + signal": (_blend(E_tok_pc, tok_ids, _tok_signal),
                                  tok_keys, lambda k_: idf_tok.get(k_, dflt), K * L),
        "m2v word -pc1 + signal": (_blend(E_wrd_pc, words, _word_signal),
                                   wrd_keys, lambda k_: idf_w.get(k_, dflt), K * L),
    }

    fast = json.loads((ROOT / "validation" / "logs" / "exp_f2m2_fast_arms.json")
                      .read_text(encoding="utf-8"))
    print(f"{'arm':>26} {'self':>6} {'macro':>7} {'xnull':>6} {'micro':>7}")
    for lb in ("bpe tokens + idf", "words only + idf", "MiniLM ref (stage1)"):
        fa = fast["arms"][lb]
        print(f"{lb:>26} {fa['self_retrieval_top1']:6.3f} {fa['macro']:7.4f} "
              f"{fa['macro'] / mac_null:6.2f} {fa['micro']:7.4f}   (prior run)")

    out: dict = {"matched_macro_null": mac_null, "build_s": build_s,
                 "n_token_strings": len(tok_ids), "n_words": len(words), "arms": {}}
    for label, (vec_of, keys_of, weight_of, dim) in arms.items():
        t0 = time.perf_counter()
        nc = _bundle_from_table(names, keys_of, vec_of, weight_of, dim)
        fc = _bundle_from_table(formulas, keys_of, vec_of, weight_of, dim)
        us = (time.perf_counter() - t0) / len(texts) * 1e6
        top1 = self_retrieval_top1(nc, fc)
        routed = loo_domain_routing(nc, fc, domains)
        out["arms"][label] = {
            "self_retrieval_top1": top1, "macro": routed["macro"],
            "macro_x_null": routed["macro"] / mac_null, "micro": routed["micro"],
            "per_domain": routed["per_domain"], "encode_us_per_text": us,
        }
        print(f"{label:>26} {top1:6.3f} {routed['macro']:7.4f} "
              f"{routed['macro'] / mac_null:6.2f} {routed['micro']:7.4f}   ({us:.0f} us/text)")

    best = max(out["arms"], key=lambda lb: out["arms"][lb]["macro"])
    fb = out["arms"][best]
    ref_macro = fast["arms"]["MiniLM ref (stage1)"]["macro"]
    out["best_arm"] = best
    print(f"\nbest distilled arm: {best} -- macro {fb['macro']:.4f} "
          f"({fb['macro_x_null']:.2f}x null, {fb['macro'] / ref_macro:.2f}x the live teacher) "
          f"at ~{fb['encode_us_per_text']:.0f} us/challenge")

    (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "validation" / "logs" / "exp_f2m2_m2v.json"
    json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {json_path}")
    return out


def _fit_pc(X: np.ndarray, n_pc: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fit the anisotropy stats (mean + top principal components) on X."""
    mu = X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:n_pc]


def _apply_pc(X: np.ndarray, mu: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Apply externally-fit anisotropy removal to X; _remove_top_pc(X) is the
    self-fit special case _apply_pc(X, *_fit_pc(X))."""
    Xc = X - mu
    for v in pcs:
        Xc = Xc - np.outer(Xc @ v, v)
    return _unit_rows(Xc)


def run_m2v_ood(n_docs: int = 20_000, doc_len: int = 200, top_words: int = 2000) -> dict:
    """Does the winning m2v arm survive OUT-OF-DOMAIN statistics?

    The teacher table is corpus-independent, but the shipped winner's IDF and
    anisotropy direction (mean+PC1) were fit on the 560 eval texts themselves
    -- the same-corpus-statistics error class this project has been burned by
    twice. Here both are refit on wiki_full text the eval corpus never saw:
    IDF from word document-frequencies over sampled decoded windows, PC1 from
    teacher embeddings of the corpus's most frequent words. A 2x2 grid
    (in/ext IDF x in/ext PC1) attributes any change to its ingredient.
    """
    import platform
    import time

    from cubbyllm.training.data import _load_tokenizer
    from sentence_transformers import SentenceTransformer

    names, formulas, domains = load_corpus()
    mac_null = matched_macro_null(domains)
    texts = names + formulas
    _enc, decode_fn, _e, _v = _load_tokenizer(str(TOKENIZER))

    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    print(f"mode: --m2v-ood -- refit IDF + PC1 on wiki_full ({n_docs} docs x {doc_len} tokens); "
          f"teacher table unchanged\n")

    # -- in-domain ingredients (reproduce the shipped winner exactly) ----------
    word_sets = [{w for w in re.split(r"[^a-z0-9]+", x.lower()) if len(w) >= 2} for x in texts]
    words = sorted(set().union(*word_sets))
    df_in: dict[str, int] = {}
    for s_ in word_sets:
        for w_ in s_:
            df_in[w_] = df_in.get(w_, 0) + 1
    idf_in = {w_: float(np.log(len(texts) / c)) for w_, c in df_in.items()}
    dflt_in = float(np.log(len(texts)))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E_wrd = _unit_rows(np.asarray(model.encode(words, batch_size=256,
                                               show_progress_bar=False), dtype=np.float32))

    # -- external statistics from wiki_full ------------------------------------
    t0 = time.perf_counter()
    shard = np.memmap(r"D:\grillcheese_training_data\token_cache\wiki_full.u32",
                      dtype=np.uint32, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(shard) - doc_len, size=n_docs)
    df_ext: dict[str, int] = {}
    freq_ext: dict[str, int] = {}
    for s_ in starts:
        doc = decode_fn([int(t) for t in shard[s_:s_ + doc_len]]).lower()
        ws = {w for w in re.split(r"[^a-z0-9]+", doc) if len(w) >= 2}
        for w_ in ws:
            df_ext[w_] = df_ext.get(w_, 0) + 1
            freq_ext[w_] = freq_ext.get(w_, 0) + 1
    idf_ext = {w_: float(np.log(n_docs / c)) for w_, c in df_ext.items()}
    dflt_ext = float(np.log(n_docs))
    ext_vocab = sorted(freq_ext, key=lambda w_: -freq_ext[w_])[:top_words]
    E_ext = _unit_rows(np.asarray(model.encode(ext_vocab, batch_size=256,
                                               show_progress_bar=False), dtype=np.float32))
    mu_ext, pc_ext = _fit_pc(E_ext)
    print(f"external stats built in {time.perf_counter() - t0:.0f}s: "
          f"{len(df_ext)} corpus words; PC1 fit on top {len(ext_vocab)}; "
          f"axiom words seen in corpus: "
          f"{sum(1 for w_ in words if w_ in idf_ext)}/{len(words)}\n")

    mu_in, pc_in = _fit_pc(E_wrd)
    P384 = make_projection(384)
    rows_k = np.arange(K)
    inv_sqrt_k = 1.0 / np.sqrt(K)

    def _word_signal(w_: str) -> np.ndarray:
        digest = hashlib.blake2b(f"w:{w_}".encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**63)
        sig = np.random.default_rng(seed).integers(0, L, size=K)
        v = np.zeros(K * L, dtype=np.float32)
        v[rows_k * L + sig] = inv_sqrt_k
        return v

    wrd_keys = lambda t: {w for w in re.split(r"[^a-z0-9]+", t.lower()) if len(w) >= 2}  # noqa: E731

    variants = {
        "in-IDF / in-PC1 (shipped)": (idf_in, dflt_in, mu_in, pc_in),
        "in-IDF / ext-PC1": (idf_in, dflt_in, mu_ext, pc_ext),
        "ext-IDF / in-PC1": (idf_ext, dflt_ext, mu_in, pc_in),
        "ext-IDF / ext-PC1 (honest)": (idf_ext, dflt_ext, mu_ext, pc_ext),
    }
    out: dict = {"matched_macro_null": mac_null, "n_docs": n_docs,
                 "doc_len": doc_len, "top_words": top_words, "arms": {}}
    print(f"{'variant':>28} {'self':>6} {'macro':>7} {'xnull':>6} {'micro':>7}")
    for label, (idf, dflt, mu, pcs) in variants.items():
        E_pc = _apply_pc(E_wrd, mu, pcs)
        proj = _unit_rows(E_pc @ P384.T)
        vec_of = {w_: 0.5 * proj[j] + 0.5 * _word_signal(w_) for j, w_ in enumerate(words)}
        nc = _bundle_from_table(names, wrd_keys, vec_of, lambda k_: idf.get(k_, dflt), K * L)
        fc = _bundle_from_table(formulas, wrd_keys, vec_of, lambda k_: idf.get(k_, dflt), K * L)
        top1 = self_retrieval_top1(nc, fc)
        routed = loo_domain_routing(nc, fc, domains)
        out["arms"][label] = {
            "self_retrieval_top1": top1, "macro": routed["macro"],
            "macro_x_null": routed["macro"] / mac_null, "micro": routed["micro"],
        }
        print(f"{label:>28} {top1:6.3f} {routed['macro']:7.4f} "
              f"{routed['macro'] / mac_null:6.2f} {routed['micro']:7.4f}")

    (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "validation" / "logs" / "exp_f2m2_m2v_ood.json"
    json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {json_path}")
    return out


def _env_stamp() -> str:
    """The environment stamp every exp_f2m2_* log carries, generated in-repo.

    The earlier logs' stamps came from a one-off script that lived outside the
    repo, which made them regenerable only by hand. This mode generates its own.
    """
    import platform

    parts = [f"python {platform.python_version()}", platform.platform()]
    parts.append(f"numpy {np.__version__}")
    try:
        import torch
        parts.append(f"torch {torch.__version__}")
    except Exception:                                          # pragma: no cover
        parts.append("torch not-imported")
    try:
        import sentence_transformers as st
        parts.append(f"sentence-transformers {st.__version__}")
    except Exception:                                          # pragma: no cover
        parts.append("sentence-transformers not-imported")
    stamp = " | ".join(parts)
    # This mode is PURE NOISE end to end: no checkpoint, no tokenizer, no
    # pretrained model is opened, so there is no checkpoint identity to pin.
    return (stamp + "\ncheckpoint: NOT LOADED — --null-baseline uses a pure-noise "
            "encoder (no trunk, no tokenizer, no pretrained model)")


def self_test() -> None:
    names, formulas, domains = load_corpus()
    assert len(names) == len(formulas) == len(domains) == 280
    assert len(set(domains)) == 15

    # The load-bearing analytic claim: the projection is cosine-EXACT.
    rng = np.random.default_rng(1)
    E = rng.standard_normal((16, 512)).astype(np.float32)
    P = make_projection(512)
    err = np.abs(cosine_matrix(E, E) - cosine_matrix(project(P, E), project(P, E))).max()
    assert err < 1e-5, f"projection is not cosine-exact: max|delta|={err:.2e}"

    # Determinism.
    assert np.array_equal(make_projection(512), make_projection(512))

    # 384 is the `ref` arm's latent dim -- every `ref` headline number came from
    # this projection, so it is exercised explicitly and not left to inference
    # from the 512 case.
    E384 = rng.standard_normal((16, 384)).astype(np.float32)
    P384 = make_projection(384)
    err384 = np.abs(
        cosine_matrix(E384, E384) - cosine_matrix(project(P384, E384), project(P384, E384))
    ).max()
    assert err384 < 1e-5, f"384-D projection is not cosine-exact: max|delta|={err384:.2e}"

    # The hash arm emits valid, deterministic one-hot-per-block codes.
    h = hash_encode(["a", "b", "a"])
    assert h.shape == (3, K, L)
    assert np.array_equal(h.sum(axis=2), np.ones((3, K), dtype=np.float32))
    assert np.array_equal(h[0], h[2]) and not np.array_equal(h[0], h[1])

    # The ngram arm. Every other encoder and discretizer is guarded here; this
    # one produced the result the write-up calls uncomfortable (a training-free
    # sketch beating the trained trunk), and M3's central question rests on it,
    # so it gets asserts too -- not just shape and determinism, but THE PROPERTY
    # THAT ACTUALLY MATTERS: bundling makes substring overlap show up as cosine.
    g_over = "the quick brown fox"          # shares most 3-grams with g_near
    g_near = "the quick brown dog"
    g_far = "12345 67890"                   # 3-grams disjoint from both
    ng = ngram_encode([g_over, g_near, g_far])
    assert ng.shape == (3, K, L), ng.shape
    assert np.array_equal(ng, ngram_encode([g_over, g_near, g_far]))    # deterministic
    # Structural invariant of bundling: each unique n-gram adds 1.0 to exactly
    # one index in EVERY block, so every block's mass equals the n-gram count.
    n_grams = len({g_over[j:j + 3] for j in range(len(g_over) - 2)})
    assert np.array_equal(ng[0].sum(axis=1), np.full(K, float(n_grams), dtype=np.float32))
    # Short strings (len < n) must still emit one gram, not zero.
    assert np.array_equal(ngram_encode(["ab"])[0].sum(axis=1),
                          np.ones(K, dtype=np.float32))
    ng_near = float(cosine_matrix(ng[:1], ng[1:2])[0, 0])
    ng_far = float(cosine_matrix(ng[:1], ng[2:3])[0, 0])
    assert ng_near > ng_far, f"ngram lost substring locality: {ng_near} !> {ng_far}"
    assert ng_near > 0.5, f"ngram overlap cosine too low: {ng_near}"
    # A hashed one-hot encoder has NO such property -- this is what separates
    # the ngram arm from the hash arm, and it is the whole point of the control.
    hh = hash_encode([g_over, g_near, g_far])
    assert float(cosine_matrix(hh[:1], hh[1:2])[0, 0]) < ng_near

    # Trunk arms: shapes, determinism, and that they are NOT degenerate.
    probe = ["force equals mass times acceleration", "a cat sat on the mat"]
    bag = tbag_encode(probe)
    assert bag.shape == (2, 512), bag.shape
    assert np.isfinite(bag).all()
    assert np.array_equal(bag, tbag_encode(probe))          # deterministic
    bag_cos = float(cosine_matrix(bag[:1], bag[1:])[0, 0])
    assert abs(bag_cos) < 0.999, f"t-bag collapsed: cos={bag_cos:.4f}"

    ctx = tctx_encode(probe)
    assert ctx.shape == (2, 512), ctx.shape
    assert np.isfinite(ctx).all()
    assert np.array_equal(ctx, tctx_encode(probe))          # deterministic
    ctx_cos = float(cosine_matrix(ctx[:1], ctx[1:])[0, 0])
    assert abs(ctx_cos) < 0.999, f"t-ctx collapsed: cos={ctx_cos:.4f}"
    print(f"trunk arms OK — t-bag cos(unrelated)={bag_cos:.3f}, t-ctx={ctx_cos:.3f}")

    # A synthetic corpus with two well-separated clusters: both metrics must be
    # perfect, which pins the metrics' orientation (name/formula views built below).
    # NOTE: the clusters must genuinely cluster — an orthonormal basis would NOT
    # work, because with every off-diagonal cosine equal to 0 the leave-one-out
    # argmax is decided by tie-breaking, not by domain.
    fake_rng = np.random.default_rng(7)
    centers = fake_rng.standard_normal((2, 32)).astype(np.float32)
    fake = np.concatenate([np.tile(centers[0], (4, 1)), np.tile(centers[1], (4, 1))])
    fake = (fake + 0.01 * fake_rng.standard_normal((8, 32))).astype(np.float32).reshape(8, 32, 1)
    fake_dom = ["a"] * 4 + ["b"] * 4
    # Two DIFFERENT views of the same 8 items (name-view vs formula-view), so
    # this catches a transposed/axis-flipped metric -- a symmetric fake=fake
    # input cannot, because cosine_matrix(X, X) is symmetric.
    fake_n = (fake + 0.005 * fake_rng.standard_normal(fake.shape)).astype(np.float32)
    fake_f = (fake + 0.005 * fake_rng.standard_normal(fake.shape)).astype(np.float32)
    assert self_retrieval_top1(fake_n, fake_f) == 1.0
    routed = loo_domain_routing(fake_n, fake_f, fake_dom)
    assert routed["macro"] == 1.0, routed

    # Orientation guard: a DELIBERATELY ASYMMETRIC pair. name_0 matches its own
    # formula best (so the correct orientation scores 1.0), but name_0 is also
    # closer to formula_1 than name_1 is -- so the COLUMN-wise (transposed)
    # argmax picks the wrong row for item 1. The clustered case above cannot
    # catch this: both views perturb the same item-level point, leaving it
    # diagonal-dominant on both axes.
    orient_f = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32).reshape(2, 3, 1)
    orient_n = np.array([[1.0, 0.9, 0.0], [0.0, 0.6, 0.8]], dtype=np.float32).reshape(2, 3, 1)
    assert self_retrieval_top1(orient_n, orient_f) == 1.0          # names -> formulas
    assert self_retrieval_top1(orient_f, orient_n) < 1.0           # transposed must FAIL

    # Orientation guard for the ROUTING metric (the one the kill criterion uses).
    # 4 items, 2 domains, asymmetric by design: name_0 (domain a) is also the
    # best match for formula_2 (domain b), but in the correct direction name_0's
    # own best NON-SELF formula is still f_1 (domain a) -- so correct scores
    # macro 1.0, while the transposed direction misroutes formula_2 to a
    # domain-a name and drops to 0.75. The clustered fake_n/fake_f pair cannot
    # catch this: it is diagonal-dominant on both axes.
    route_f = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                       dtype=np.float32).reshape(4, 4, 1)
    route_n = np.array([[0, 0.9, 0.8, 0], [0.8, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0.5, 1]],
                       dtype=np.float32).reshape(4, 4, 1)
    route_dom = ["a", "a", "b", "b"]
    assert loo_domain_routing(route_n, route_f, route_dom)["macro"] == 1.0
    assert loo_domain_routing(route_f, route_n, route_dom)["macro"] < 1.0   # transposed must FAIL

    assert abs(majority_class(fake_dom) - 0.5) < 1e-9

    # micro and macro chance are DIFFERENT baselines and must not be conflated:
    # on a skewed corpus the majority-class rate is high while macro-chance stays
    # 1/n_domains, because macro-averaging removes the imbalance by construction.
    skew = ["a"] * 9 + ["b"]
    assert abs(majority_class(skew) - 0.9) < 1e-9
    assert abs(macro_chance(skew) - 0.5) < 1e-9

    # MATCHED nulls (argmax over the other n-1 items) are a THIRD thing again,
    # distinct from both baselines above. On the real corpus they are exact
    # rationals: micro = 9576/78120 = 19/155, macro = 265/(15*279) = 53/837.
    assert abs(matched_micro_null(domains) - 19 / 155) < 1e-12
    assert abs(matched_macro_null(domains) - 53 / 837) < 1e-12
    # ... and NOT equal to the baselines they were first read against.
    assert matched_micro_null(domains) < majority_class(domains)
    assert matched_macro_null(domains) < macro_chance(domains)
    # The per-domain null SCALES WITH DOMAIN SIZE -- physics (84) is ~9x
    # identity (10), so a flat per-domain baseline does not exist.
    pdn = per_domain_null(domains)
    assert abs(pdn["physics"] - 83 / 279) < 1e-12
    assert abs(pdn["identity"] - 9 / 279) < 1e-12
    assert pdn["physics"] > 9 * pdn["identity"]
    # On a BALANCED corpus micro and macro nulls coincide, which is the property
    # that makes the balanced structure a clean control in --null-baseline.
    bal = ["A"] * 140 + ["B"] * 140
    assert abs(matched_micro_null(bal) - matched_macro_null(bal)) < 1e-12
    # Pure-noise codes are valid inputs to the shipped metrics.
    for kind in NULL_KINDS:
        nz = _null_codes(6, kind, np.random.default_rng(0))
        assert nz.shape == (6, K, L) and np.isfinite(nz).all()
    assert np.array_equal(
        _null_codes(6, "onehot", np.random.default_rng(0)).sum(axis=2),
        np.ones((6, K), dtype=np.float32),
    )
    # Tie diagnostics: one-hot codes tie constantly, continuous codes never do.
    # That difference is the whole reason the two nulls differ.
    oh_a = _null_codes(60, "onehot", np.random.default_rng(11))
    oh_b = _null_codes(60, "onehot", np.random.default_rng(12))
    gz_a = _null_codes(60, "gauss", np.random.default_rng(11))
    gz_b = _null_codes(60, "gauss", np.random.default_rng(12))
    assert _tie_stats(oh_a, oh_b)["tied_argmax_frac"] > 0.2
    assert _tie_stats(gz_a, gz_b)["tied_argmax_frac"] == 0.0
    assert _tie_stats(gz_a, gz_b)["tie_group_mean"] == 1.0

    # Discretizers emit valid one-hot-per-block codes; PQ is deterministic.
    # n_probe >= L=128: fit_pq now raises below that (see fit_pq's own guard), so
    # the probe must be at least as large as the real corpus's saturation regime,
    # not the small n=5 this used before that guard existed.
    n_probe = 150
    dense = np.random.default_rng(3).standard_normal((n_probe, K, L)).astype(np.float32)
    am = argmax_onehot(dense)
    assert am.shape == dense.shape
    assert np.array_equal(am.sum(axis=2), np.ones((n_probe, K), dtype=np.float32))
    cent = fit_pq(dense)
    assert cent.shape == (K, L, L)
    pq = pq_onehot(dense, cent)
    assert np.array_equal(pq.sum(axis=2), np.ones((n_probe, K), dtype=np.float32))
    assert np.array_equal(pq, pq_onehot(dense, fit_pq(dense)))       # deterministic

    # fit_pq must refuse fewer points than centroids rather than silently
    # zero-padding the codebook.
    try:
        fit_pq(dense[:5])
        raise AssertionError("fit_pq should have rejected n=5 < L=128")
    except ValueError:
        pass

    # pq_reconstruct must return centroid VECTORS, not one-hot labels: every
    # reconstructed row must equal some actual row of the codebook.
    rec = pq_reconstruct(dense, cent)
    assert rec.shape == dense.shape
    assert (np.abs(rec).sum(axis=2) > 0).all()                        # every block populated
    for b in range(K):
        matches = np.all(rec[:, b, :][:, None, :] == cent[b][None, :, :], axis=2)
        assert matches.any(axis=1).all(), f"block {b}: a row of rec is not a row of cent"

    # Threshold recalibration separates a trivially separable pair of populations.
    tau, auc = recalibrate_tau(np.full(20, 0.9), np.full(20, 0.1))
    assert 0.1 < tau <= 0.9 and auc == 1.0, (tau, auc)   # tau=0.9 separates perfectly

    print(f"self-test OK — corpus 280/15 domains; cosine-exact max|delta|={err:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run invariants and exit")
    ap.add_argument("--stage1", action="store_true", help="rank embedders (dense)")
    ap.add_argument("--fast-arms", action="store_true",
                    help="model-free challenge encoders vs latency (the fast path)")
    ap.add_argument("--beagle", action="store_true",
                    help="BEAGLE distributional-memory arm (offline corpus pass)")
    ap.add_argument("--beagle-shard",
                    default=r"D:\grillcheese_training_data\token_cache\wiki_full.u32",
                    help="uint32 token shard to accumulate co-occurrence from")
    ap.add_argument("--beagle-tokens", type=int, default=200_000_000,
                    help="corpus-slice length in tokens")
    ap.add_argument("--beagle-window", type=int, default=5,
                    help="co-occurrence half-window")
    ap.add_argument("--m2v", action="store_true",
                    help="static distillation of the teacher into token/word tables")
    ap.add_argument("--m2v-ood", action="store_true",
                    help="refit IDF + PC1 on external text (the honesty check)")
    ap.add_argument("--stage2", metavar="ARM", help="settle discretization for ARM")
    ap.add_argument("--leakage-check", metavar="ARM", help="check PQ codebook leakage for ARM")
    ap.add_argument("--null-baseline", action="store_true",
                    help="measure the null of macro/micro/ROC-AUC with a pure-noise encoder")
    ap.add_argument("--annotate-npz", action="store_true",
                    help="stamp the retraction note into the exported data/axiom_embeddings_*.npz")
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of noise seeds for --null-baseline (default 10)")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.null_baseline:
        print(_env_stamp())
        out = run_null_baseline(seeds=args.seeds)
        (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
        path = ROOT / "validation" / "logs" / "exp_f2m2_null_baseline.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
        return
    if args.annotate_npz:
        # Retro-annotates the two ALREADY-EXPORTED artifacts rather than
        # re-running --stage2, so every pre-existing array stays bit-identical
        # (asserted below) and no other measurement is disturbed. Future
        # --stage2 runs carry the same note from the savez call.
        for arm in ("ref", "t-ctx"):
            p = ROOT / "data" / f"axiom_embeddings_{arm}.npz"
            stamped = ("note", "retracted")
            with np.load(p, allow_pickle=False) as z:
                before = {k: z[k] for k in z.files}
            # The DATA keys must survive byte-for-byte; the stamp keys are the
            # ones being (re)written, so they are excluded from that check and
            # verified against the current constants instead. Re-annotating with
            # a corrected note must stay possible without tripping the guard.
            data_keys = {k: v for k, v in before.items() if k not in stamped}
            np.savez_compressed(p, **(data_keys | {"note": np.array(NPZ_NOTE),
                                                   "retracted": np.array(NPZ_RETRACTED)}))
            with np.load(p, allow_pickle=False) as z:
                assert set(z.files) == set(data_keys) | set(stamped), z.files
                for k, v in data_keys.items():
                    assert np.array_equal(z[k], v), f"{p.name}: {k} changed"
                assert str(z["note"]) == NPZ_NOTE and str(z["retracted"]) == NPZ_RETRACTED
            print(f"annotated {p} ({len(data_keys)} data keys verified unchanged)")
        return
    if args.stage1:
        out = run_stage1()
        (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
        path = ROOT / "validation" / "logs" / "exp_f2m2_stage1.json"
        serialisable = {k: v for k, v in out.items() if k != "arms"} | {
            "arms": {a: {k: v for k, v in d.items()} for a, d in out["arms"].items()}
        }
        path.write_text(json.dumps(serialisable, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
        return
    if args.stage2:
        # FIX (review): --stage2 hash / --stage2 ngram used to crash with KeyError
        # AFTER a full run_stage2() completed, because the export encoder dict was
        # never extended when ngram joined STAGE1_ARMS. Validate up front instead.
        if args.stage2 not in STAGE1_ARMS:
            raise SystemExit(f"--stage2 {args.stage2!r} not in STAGE1_ARMS={STAGE1_ARMS}")
        res, cent = run_stage2(args.stage2)
        names, formulas, domains = load_corpus()
        # Branch the export the same way _encode_arm does: hash/ngram are born
        # directly in block space (no separate pre-projection embedding exists),
        # so their "raw embedding" IS their (n, K, L) block code.
        if args.stage2 in ("hash", "ngram"):
            raw_encoder = {"hash": hash_encode, "ngram": ngram_encode}[args.stage2]
            name_emb, formula_emb = raw_encoder(names), raw_encoder(formulas)
        else:
            encoder = {"t-bag": tbag_encode, "t-ctx": tctx_encode, "ref": ref_encode}[args.stage2]
            name_emb, formula_emb = encoder(names), encoder(formulas)
        # DEVIATION from task-5-brief.md (see task brief header): output paths are
        # suffixed with the arm name so the t-ctx and ref runs do not clobber
        # each other -- the brief's un-suffixed `axiom_embeddings.npz` /
        # `exp_f2m2_stage2.json` assumed a single winning-arm run.
        npz_path = ROOT / "data" / f"axiom_embeddings_{args.stage2}.npz"
        json_path = ROOT / "validation" / "logs" / f"exp_f2m2_stage2_{args.stage2}.json"
        winner = res["winner"]
        # FIX (review): the artifact used to store tau_match from the WINNER
        # variant (measured in 10240-D one-hot block-code space) alongside
        # name_emb/formula_emb in the RAW pre-projection space (384/512-D) -- a
        # threshold that doesn't apply to the vectors actually stored. Now also
        # exports the codebook (centroids) so the winner's space is reachable,
        # the winner's roc_auc alongside its tau_match, the DENSE tau under a
        # distinct key (the threshold that DOES apply to name_emb/formula_emb as
        # stored), and the projection/PQ seeds so both are exactly re-derivable.
        np.savez_compressed(
            npz_path,
            name_emb=name_emb, formula_emb=formula_emb,
            domains=np.array(domains), names=np.array(names),
            arm=np.array(args.stage2), discretization=np.array(winner),
            tau_match=np.array(res["variants"][winner]["tau_match"], dtype=np.float32),
            roc_auc=np.array(res["variants"][winner]["roc_auc"], dtype=np.float32),
            tau_match_dense=np.array(res["variants"]["dense"]["tau_match"], dtype=np.float32),
            centroids=cent,
            proj_seed=np.array(PROJ_SEED),
            pq_iters=np.array(25),
            pq_seed=np.array(0),
            # FIX (whole-branch review): the artifact shipped the RETRACTED
            # conclusion as machine-readable config with nothing in it saying so.
            note=np.array(NPZ_NOTE), retracted=np.array(NPZ_RETRACTED),
        )
        json_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {npz_path} and {json_path}")
        return
    if args.leakage_check:
        if args.leakage_check not in STAGE1_ARMS:
            raise SystemExit(f"--leakage-check {args.leakage_check!r} not in STAGE1_ARMS={STAGE1_ARMS}")
        res = run_leakage_check(args.leakage_check)
        print(f"=== leakage check: {args.leakage_check} ===")
        print(f"{'':>26}{'dense':>10}{'argmax':>10}{'pq':>10}{'pq-adc':>10}")
        s = res["shipped"]
        print(f"{'shipped (leaky)':>26}{s['dense']:>10.4f}{s['argmax']:>10.4f}"
              f"{s['pq']:>10.4f}{s['pq-adc']:>10.4f}")
        a = res["control_a_name_fit"]
        print(f"{'control A (name-fit)':>26}{a['dense']:>10.4f}{'--':>10}"
              f"{a['pq']:>10.4f}{a['pq-adc']:>10.4f}")
        b = res["control_b_halfsplit"]
        print(f"{'control B (half-split)':>26}{b['dense']:>10.4f}{b['argmax']:>10.4f}"
              f"{b['pq']:>10.4f}{b['pq-adc']:>10.4f}   (n_scored={b['n_scored']})")

        checks = {
            "control A pq > dense": a["pq"] > a["dense"],
            "control A pq-adc > dense": a["pq-adc"] > a["dense"],
            "control B pq > dense": b["pq"] > b["dense"],
            "control B pq-adc > dense": b["pq-adc"] > b["dense"],
        }
        print("\n--- read ---")
        for label, ok in checks.items():
            print(f"  [{'HOLDS' if ok else 'FAILS'}] {label}")
        a_holds = checks["control A pq > dense"] and checks["control A pq-adc > dense"]
        b_holds = checks["control B pq > dense"] and checks["control B pq-adc > dense"]
        if a_holds and b_holds:
            verdict = "gain SURVIVES both controls -- cluster-smoothing explanation stands"
        elif not a_holds and not b_holds:
            verdict = "gain FAILS both controls -- shipped result was leakage"
        else:
            verdict = "SPLIT verdict -- survives one control, not the other"
        print(f"  => {verdict}")
        res["checks"] = checks
        res["verdict"] = verdict

        (ROOT / "validation" / "logs").mkdir(parents=True, exist_ok=True)
        json_path = ROOT / "validation" / "logs" / f"exp_f2m2_leakage_{args.leakage_check}.json"
        json_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {json_path}")
        return
    if args.fast_arms:
        run_fast_arms()
        return
    if args.beagle:
        run_beagle(args.beagle_shard, args.beagle_tokens, args.beagle_window)
        return
    if args.m2v:
        run_m2v()
        return
    if args.m2v_ood:
        run_m2v_ood()
        return
    raise SystemExit("pass --self-test, --stage1, --stage2 ARM, --leakage-check ARM, "
                     "--null-baseline, --fast-arms, --beagle, --m2v, --m2v-ood, "
                     "or --annotate-npz")


if __name__ == "__main__":
    main()
