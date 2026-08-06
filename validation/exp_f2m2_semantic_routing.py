"""exp_f2m2_semantic_routing — H-F2 M2: can semantic block codes route?

Stage 1 ranks embedders (HASH / T-bag / T-ctx / REF) under a dense,
cosine-exact orthonormal projection into (80,128). Stage 2 settles
discretization (dense / argmax / PQ) for the winner and recalibrates
tau_match.

Standalone experiment: NEVER imported by cubbyllm/. Design:
docs/superpowers/specs/2026-08-06-hf2-m2-semantic-context-encoding-design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

K, L = 80, 128
VSA_DIM = K * L          # 10240
PROJ_SEED = 0

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


STAGE1_ARMS = ("hash", "t-bag", "t-ctx", "ref")


def _encode_arm(arm: str, names: list[str], formulas: list[str]):
    """-> (name_codes, formula_codes) as (n, K, L) dense block codes."""
    if arm == "hash":
        return hash_encode(names), hash_encode(formulas)
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

    # The hash arm emits valid, deterministic one-hot-per-block codes.
    h = hash_encode(["a", "b", "a"])
    assert h.shape == (3, K, L)
    assert np.array_equal(h.sum(axis=2), np.ones((3, K), dtype=np.float32))
    assert np.array_equal(h[0], h[2]) and not np.array_equal(h[0], h[1])

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
    assert abs(majority_class(fake_dom) - 0.5) < 1e-9

    # micro and macro chance are DIFFERENT baselines and must not be conflated:
    # on a skewed corpus the majority-class rate is high while macro-chance stays
    # 1/n_domains, because macro-averaging removes the imbalance by construction.
    skew = ["a"] * 9 + ["b"]
    assert abs(majority_class(skew) - 0.9) < 1e-9
    assert abs(macro_chance(skew) - 0.5) < 1e-9

    print(f"self-test OK — corpus 280/15 domains; cosine-exact max|delta|={err:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run invariants and exit")
    ap.add_argument("--stage1", action="store_true", help="rank embedders (dense)")
    args = ap.parse_args()
    if args.self_test:
        self_test()
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
    raise SystemExit("pass --self-test or --stage1")


if __name__ == "__main__":
    main()
