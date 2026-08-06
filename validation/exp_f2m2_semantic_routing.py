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

    print(f"self-test OK — corpus 280/15 domains; cosine-exact max|delta|={err:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="run invariants and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit("nothing to run yet — see --self-test")


if __name__ == "__main__":
    main()
