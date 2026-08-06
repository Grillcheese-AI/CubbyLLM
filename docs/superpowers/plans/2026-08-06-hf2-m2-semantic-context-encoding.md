# H-F2 M2 — Semantic Context Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give challenges and axioms a shared semantic space so M1's similarity-routing tier fires on organic contexts — measured against our own trunk, with a kill criterion that also decides whether fine-tuning is worth doing.

**Architecture:** A standalone CubbyLLM validation screen runs two stages over mowm's 280 exported axioms: Stage 1 ranks embedders (BLAKE2b hash baseline / trunk bag-of-embeddings / trunk contextual / small pretrained reference) at dense, cosine-exact orthonormal projection into (80,128); Stage 2 settles discretization (dense / argmax one-hot / PQ one-hot) for the winner and recalibrates `tau_match`. If the kill criterion passes, mowm gains an **additive** semantic axiom path and a live confirm through the real `CubbyBridge`.

**Tech Stack:** Python 3.12; numpy (screen core — no scikit-learn, k-means is hand-rolled); torch (trunk arms only, lazy); `tokenizers` via `cubbyllm.training.data._load_tokenizer`; `sentence-transformers` (reference arm only); mowm (`uv`-managed) for the live confirm.

## Global Constraints

- **Dimensions `k=80, l=128` (`VSA_DIM = 10240`)** — the decided H-B5 algebra, never M1's toy `k=4/l=32`.
- **The projection must have orthonormal columns** (full QR, `k*l >= latent_dim`) so cosine is preserved exactly; this is asserted, not assumed.
- **The screen imports neither `mowm` nor `cubemind`.** It reads `data/mowm_axioms.json` and re-implements the 5-line BLAKE2b baseline locally.
- **`validation/` is not package code** — nothing under `cubbyllm/` may import it.
- **Determinism:** every RNG is seeded (`PROJ_SEED = 0`); reruns must reproduce numbers exactly.
- **Green suites stay green:** CubbyLLM `python -m pytest tests -q` (108 before this plan) and mowm `uv run pytest tests/ -q` (741 before this plan). Changes are additive; counts may only go up.
- **Every number quoted in docs links the `validation/logs/` file that produced it**, and runs are teed to `validation/logs/` with an environment stamp.
- **mowm changes are additive only:** never modify `make_axiom`, `WorldEncoder`, or any hash behavior.
- **Commit trailers.** CubbyLLM commits end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav`
  **mowm commits use mowm's own conventional style with NO CubbyLLM trailers.**
- **The user's uncommitted mowm working-tree files (10 entries) are never staged, stashed, or modified.** In mowm, always `git add` explicit paths; never `git add -A/./-u`.

## File Structure

| File | Responsibility |
|---|---|
| `data/mowm_axioms.json` (create) | The 280-axiom corpus exported from mowm, with provenance. Keeps the screen cross-repo-import-free. |
| `tests/data/test_axiom_corpus.py` (create) | Guards the committed corpus artifact (count, domains, fields). |
| `validation/exp_f2m2_semantic_routing.py` (create) | The whole screen: corpus loading, projection, all encoders, discretizers, LOO metric, kill criterion, `tau` recalibration, artifact export. Standalone. |
| `validation/logs/exp_f2m2_semantic_routing.log` (create) | Teed run output with env stamp. |
| `data/axiom_embeddings.npz` (create, Task 5) | 280×512 name-view and formula-view embeddings for the winning arm — lets mowm consume semantics without owning the trunk. |
| `CUBBYLLM_HYPOTHESES.md` (modify) | The honest H-F2 M2 result. |
| `mowm/encoding/__init__.py`, `mowm/encoding/semantic_axioms.py` (create) | Additive: build `Axiom`s carrying semantic vectors. |
| `mowm/tests/test_semantic_routing.py` (create) | The live confirm through the real `CubbyBridge`. |

---

## Task 1: Export the axiom corpus

**Files:**
- Create: `data/mowm_axioms.json`
- Test: `tests/data/test_axiom_corpus.py`

**Interfaces:**
- Produces: `data/mowm_axioms.json` = `{"_provenance": {...}, "axioms": [{"name": str, "domain": str, "formula_str": str}, ...]}` with 280 entries.

- [ ] **Step 1: Write the failing test.** Create `tests/data/test_axiom_corpus.py`:

```python
"""Guards the committed mowm axiom corpus (the M2 screen's only data input)."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "mowm_axioms.json"


def _corpus():
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_corpus_has_280_axioms_with_all_fields():
    d = _corpus()
    ax = d["axioms"]
    assert len(ax) == 280
    assert d["_provenance"]["n"] == len(ax)
    for a in ax:
        assert a["name"] and a["domain"] and a["formula_str"]


def test_corpus_covers_fifteen_domains_with_physics_largest():
    top = collections.Counter(
        a["domain"].split(".")[0] for a in _corpus()["axioms"]
    )
    assert len(top) == 15
    assert top["physics"] == 84
    assert sum(top.values()) == 280


def test_name_and_formula_are_distinct_views():
    # The screen poses `name` against worlds seeded from `formula_str`; if the
    # two views were ever identical the cross-view metric would be trivial.
    ax = _corpus()["axioms"]
    assert sum(a["name"] == a["formula_str"] for a in ax) == 0
```

- [ ] **Step 2: Run it, verify it fails.** `python -m pytest tests/data/test_axiom_corpus.py -q` → FAIL (`FileNotFoundError`).

- [ ] **Step 3: Export the corpus.** Run from the **mowm** directory (mowm is `uv`-managed; this is a one-off data-prep step, which is why no committed CubbyLLM code imports mowm):

```bash
cd /c/Users/grill/Documents/GitHub/mowm
uv run python - <<'EOF'
import importlib, json, pkgutil, subprocess
import mowm.domains as D
from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all

for m in pkgutil.iter_modules(D.__path__):
    if not m.name.startswith("_"):
        importlib.import_module(f"mowm.domains.{m.name}")

lib = AxiomLibrary(k=8, l=32)   # k/l irrelevant: only text is exported
seed_all(lib)
ax = list(lib._axioms.values())
recs = [{"name": a.name, "domain": a.domain, "formula_str": a.formula_str} for a in ax]
sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
out = {
    "_provenance": {
        "source": "mowm/domains/*.py via seed_all()",
        "mowm_commit": sha,
        "exported": "2026-08-06",
        "n": len(recs),
    },
    "axioms": recs,
}
path = r"C:\Users\grill\Documents\GitHub\CubbyLLM\data\mowm_axioms.json"
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print("wrote", len(recs), "axioms ->", path)
EOF
```

Expected: `wrote 280 axioms`.

- [ ] **Step 4: Run the tests, verify they pass.** `python -m pytest tests/data/test_axiom_corpus.py -q` → 3 passed. Then the full suite: `python -m pytest tests -q` → 111 passed (108 + 3).

- [ ] **Step 5: Commit.**

```bash
git add data/mowm_axioms.json tests/data/test_axiom_corpus.py
git commit -m "feat(data): export mowm's 280-axiom corpus for the H-F2 M2 screen

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 2: Screen scaffold — projection, hash baseline, reference encoder

**Files:**
- Create: `validation/exp_f2m2_semantic_routing.py`

**Interfaces:**
- Consumes: `data/mowm_axioms.json` (Task 1).
- Produces: `K=80`, `L=128`, `VSA_DIM=10240`, `PROJ_SEED=0`; `load_corpus() -> (names, formulas, domains)`; `make_projection(latent_dim, seed=PROJ_SEED) -> (VSA_DIM, latent_dim)`; `project(P, E) -> (n, K, L)`; `cosine_matrix(A, B) -> (n, m)`; `hash_encode(texts) -> (n, K, L)`; `ref_encode(texts) -> (n, 384)`; `self_test()`; a `--self-test` CLI flag.

- [ ] **Step 1: Write the script with its self-test.** Create `validation/exp_f2m2_semantic_routing.py`:

```python
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

import numpy as np

K, L = 80, 128
VSA_DIM = K * L          # 10240
PROJ_SEED = 0

ROOT = pathlib.Path(__file__).resolve().parents[1]
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
```

- [ ] **Step 2: Run the self-test.** `python validation/exp_f2m2_semantic_routing.py --self-test`
Expected: `self-test OK — corpus 280/15 domains; cosine-exact max|delta|=<~1e-7>`. A failure of the cosine-exactness assertion invalidates the spec's central analytic claim — stop and report rather than loosening the tolerance.

- [ ] **Step 3: Verify the reference arm loads.** `pip install sentence-transformers`, then:
`python -c "import sys; sys.path.insert(0,'validation'); from exp_f2m2_semantic_routing import ref_encode; import numpy as np; e=ref_encode(['force equals mass times acceleration','a cat sat on the mat']); print(e.shape, float(np.dot(e[0],e[1])/(np.linalg.norm(e[0])*np.linalg.norm(e[1]))))"`
Expected: `(2, 384)` and a cosine well below 1.0 (unrelated sentences).

- [ ] **Step 4: Confirm the package suite is untouched.** `python -m pytest tests -q` → 111 passed (validation/ is not package code, so this must not change).

- [ ] **Step 5: Commit.**

```bash
git add validation/exp_f2m2_semantic_routing.py
git commit -m "feat(validation): H-F2 M2 screen scaffold — cosine-exact projection + hash/ref arms

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 3: The trunk arms (our own embeddings)

**Files:**
- Modify: `validation/exp_f2m2_semantic_routing.py` (add the two trunk encoders + extend `self_test`)

**Interfaces:**
- Consumes: `load_corpus`, `make_projection`, `project`, `cosine_matrix` (Task 2).
- Produces: `CKPT`, `TOKENIZER` paths; `load_tokenizer() -> (encode, vocab)`; `load_checkpoint() -> dict`; `tbag_encode(texts) -> (n, 512)`; `tctx_encode(texts, batch=16) -> (n, 512)`.

**Context the implementer needs:** the checkpoint stores `params` as a plain list in `model.parameters()` order (`validation/train_colab.py:401-402`), and the resume path rebuilds the model then copies in order after asserting `meta` equality and tensor count (`:414-420`). `params[0]` is the `(127996, 512)` token-embedding table. `meta` is `{'D':512,'L':8,'attn_every':3,'backbone':'hybrid','gen':'basis','heads':8,'mem_every':2,'mem_key':64,'mem_topk':8,'vocab':127996,'window':512}`. `CubbyModel.features(tokens)` returns pre-head hidden states `(B, S, d)` (`cubbyllm/model/assembly.py:89-96`).

- [ ] **Step 1: Add the failing self-test assertions.** Append to `self_test()` in `validation/exp_f2m2_semantic_routing.py`, just before its final `print`:

```python
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
```

- [ ] **Step 2: Run it, verify it fails.** `python validation/exp_f2m2_semantic_routing.py --self-test` → FAIL (`NameError: tbag_encode`).

- [ ] **Step 3: Implement both trunk encoders.** Add above `self_test()`:

```python
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
        "CB_BACKBONE": str(meta["backbone"]), "CB_GEN": str(meta["gen"]),
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
```

- [ ] **Step 4: Run the self-test, verify it passes.** `python validation/exp_f2m2_semantic_routing.py --self-test`
Expected: `trunk arms OK — t-bag cos(unrelated)=<value>, t-ctx=<value>`, both well below 1.0. If `_rebuild_trunk` raises a mismatch, **stop and report** — do not silently fall back to `tbag` only; the controller decides.

- [ ] **Step 5: Commit.**

```bash
git add validation/exp_f2m2_semantic_routing.py
git commit -m "feat(validation): H-F2 M2 trunk arms — bag-of-embeddings + contextual pooling

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 4: The LOO metric and the Stage-1 run

**Files:**
- Modify: `validation/exp_f2m2_semantic_routing.py`
- Create: `validation/logs/exp_f2m2_stage1.log`

**Interfaces:**
- Consumes: all encoders (Tasks 2–3).
- Produces: `self_retrieval_top1(name_codes, formula_codes) -> float`; `loo_domain_routing(name_codes, formula_codes, domains) -> dict` with keys `micro`, `macro`, `per_domain`, `same_cos`, `diff_cos`; `majority_class(domains) -> float`; `--stage1` CLI flag.

- [ ] **Step 1: Write the failing metric self-test.** Append to `self_test()` before its final `print`:

```python
    # A synthetic corpus where name==formula per item and domains are separable:
    # both metrics must be perfect, which pins the metric's orientation.
    fake_dom = ["a"] * 4 + ["b"] * 4
    base = np.eye(8, dtype=np.float32).reshape(8, 8, 1) * np.ones((1, 1, 4), dtype=np.float32)
    assert self_retrieval_top1(base, base) == 1.0
    routed = loo_domain_routing(base, base, fake_dom)
    assert routed["macro"] == 1.0, routed
    assert abs(majority_class(fake_dom) - 0.5) < 1e-9
```

- [ ] **Step 2: Run it, verify it fails.** `python validation/exp_f2m2_semantic_routing.py --self-test` → FAIL (`NameError: self_retrieval_top1`).

- [ ] **Step 3: Implement the metrics and the Stage-1 driver.** Add above `self_test()`:

```python
def majority_class(domains: list[str]) -> float:
    """The honest chance level for an imbalanced corpus (physics is 84/280)."""
    counts = {}
    for d in domains:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.values()) / len(domains)


def self_retrieval_top1(name_codes: np.ndarray, formula_codes: np.ndarray) -> float:
    """Encoder sanity: does name_i retrieve formula_i as top-1 among all formulas?"""
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
    chance = majority_class(domains)
    print(f"corpus: {len(names)} axioms, {len(set(domains))} domains, "
          f"majority-class chance = {chance:.3f}\n")

    results = {"chance": chance, "arms": {}}
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
              f"routing micro={routed['micro']:.3f} macro={routed['macro']:.3f} "
              f"({routed['macro'] / chance:.2f}x chance)")

    hash_macro = results["arms"]["hash"]["macro"]
    ref_macro = results["arms"]["ref"]["macro"]
    trunk = max(("t-bag", "t-ctx"), key=lambda a: results["arms"][a]["macro"])
    trunk_macro = results["arms"][trunk]["macro"]
    results["best_trunk"] = trunk

    print("\n--- kill criterion ---")
    checks = {
        "1 baseline sanity (hash <= 1.2x chance)": hash_macro <= 1.2 * chance,
        "2 encoder sanity (best trunk self-retrieval >= 0.50)":
            results["arms"][trunk]["self_retrieval_top1"] >= 0.50,
        "3a routing (best trunk macro >= 2x chance)": trunk_macro >= 2 * chance,
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
```

Then extend `main()` — replace its `raise SystemExit(...)` line with:

```python
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
```

and add the flag next to `--self-test`:

```python
    ap.add_argument("--stage1", action="store_true", help="rank embedders (dense)")
```

- [ ] **Step 4: Run the self-test, then Stage 1 with a full tee.**

```bash
python validation/exp_f2m2_semantic_routing.py --self-test
python -c "import sys,platform,numpy;print('python',sys.version.split()[0],'|',platform.platform(),'| numpy',numpy.__version__)" > validation/logs/exp_f2m2_stage1.log
python validation/exp_f2m2_semantic_routing.py --stage1 2>&1 | tee -a validation/logs/exp_f2m2_stage1.log
```

Expected: four arms reported, each kill-criterion check printed PASS/FAIL, a decision line, and `exp_f2m2_stage1.json` written. **Report the numbers as they come out** — a FAIL is a result, not a bug to engineer around.

- [ ] **Step 5: Commit.**

```bash
git add validation/exp_f2m2_semantic_routing.py validation/logs/exp_f2m2_stage1.log validation/logs/exp_f2m2_stage1.json
git commit -m "feat(validation): H-F2 M2 stage 1 — LOO domain routing across four embedder arms

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 5: Stage 2 — discretization, threshold recalibration, artifact export

**Files:**
- Modify: `validation/exp_f2m2_semantic_routing.py`
- Create: `validation/logs/exp_f2m2_stage2.log`, `data/axiom_embeddings.npz`

**Interfaces:**
- Consumes: Stage-1 results (`results["best_trunk"]`), all encoders, the metrics.
- Produces: `argmax_onehot(codes) -> (n, K, L)`; `fit_pq(codes, iters=25, seed=0) -> (K, L, L)`; `pq_onehot(codes, centroids) -> (n, K, L)`; `recalibrate_tau(same_cos, diff_cos) -> (tau, auc)`; `--stage2` CLI flag; `data/axiom_embeddings.npz` with arrays `name_emb`, `formula_emb`, `domains`, `names`, and scalars `arm`, `tau_match`, `discretization`.

- [ ] **Step 1: Write the failing self-test.** Append to `self_test()` before its final `print`:

```python
    # Discretizers emit valid one-hot-per-block codes; PQ is deterministic.
    dense = np.random.default_rng(3).standard_normal((5, K, L)).astype(np.float32)
    am = argmax_onehot(dense)
    assert am.shape == dense.shape
    assert np.array_equal(am.sum(axis=2), np.ones((5, K), dtype=np.float32))
    cent = fit_pq(dense)
    assert cent.shape == (K, L, L)
    pq = pq_onehot(dense, cent)
    assert np.array_equal(pq.sum(axis=2), np.ones((5, K), dtype=np.float32))
    assert np.array_equal(pq, pq_onehot(dense, fit_pq(dense)))       # deterministic

    # Threshold recalibration separates a trivially separable pair of populations.
    tau, auc = recalibrate_tau(np.full(20, 0.9), np.full(20, 0.1))
    assert 0.1 < tau < 0.9 and auc == 1.0, (tau, auc)
```

- [ ] **Step 2: Run it, verify it fails.** `python validation/exp_f2m2_semantic_routing.py --self-test` → FAIL (`NameError: argmax_onehot`).

- [ ] **Step 3: Implement Stage 2.** Add above `self_test()`:

```python
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


def run_stage2(arm: str) -> dict:
    names, formulas, domains = load_corpus()
    chance = majority_class(domains)
    name_codes, formula_codes = _encode_arm(arm, names, formulas)

    variants = {"dense": (name_codes, formula_codes)}
    variants["argmax"] = (argmax_onehot(name_codes), argmax_onehot(formula_codes))
    cent = fit_pq(formula_codes)
    variants["pq"] = (pq_onehot(name_codes, cent), pq_onehot(formula_codes, cent))

    out = {"arm": arm, "chance": chance, "variants": {}}
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
    return out
```

Extend `main()` with the flag and branch:

```python
    ap.add_argument("--stage2", metavar="ARM", help="settle discretization for ARM")
```

```python
    if args.stage2:
        res = run_stage2(args.stage2)
        names, formulas, domains = load_corpus()
        encoder = {"t-bag": tbag_encode, "t-ctx": tctx_encode, "ref": ref_encode}[args.stage2]
        np.savez_compressed(
            ROOT / "data" / "axiom_embeddings.npz",
            name_emb=encoder(names), formula_emb=encoder(formulas),
            domains=np.array(domains), names=np.array(names),
            arm=np.array(args.stage2), discretization=np.array(res["winner"]),
            tau_match=np.array(res["variants"][res["winner"]]["tau_match"], dtype=np.float32),
        )
        (ROOT / "validation" / "logs" / "exp_f2m2_stage2.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote data/axiom_embeddings.npz and the stage-2 json")
        return
```

- [ ] **Step 4: Run the self-test, then Stage 2 on Stage 1's best trunk arm.**

```bash
python validation/exp_f2m2_semantic_routing.py --self-test
python -c "import sys,platform,numpy;print('python',sys.version.split()[0],'|',platform.platform(),'| numpy',numpy.__version__)" > validation/logs/exp_f2m2_stage2.log
python validation/exp_f2m2_semantic_routing.py --stage2 <BEST_TRUNK_ARM_FROM_STAGE1> 2>&1 | tee -a validation/logs/exp_f2m2_stage2.log
```

Substitute the `best_trunk` value recorded in `validation/logs/exp_f2m2_stage1.json` (`t-bag` or `t-ctx`). Expected: three variants reported with `tau`/AUC, the discretization check PASS or FAIL, and the two artifacts written.

- [ ] **Step 5: Commit.**

```bash
git add validation/exp_f2m2_semantic_routing.py validation/logs/exp_f2m2_stage2.log validation/logs/exp_f2m2_stage2.json data/axiom_embeddings.npz
git commit -m "feat(validation): H-F2 M2 stage 2 — discretization arms + tau recalibration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

## Task 6: Record the result in the hypotheses doc

**Files:**
- Modify: `CUBBYLLM_HYPOTHESES.md` (the H-F2 section, after the `**H-F2 M1 built (2026-08-06)**` entry)

- [ ] **Step 1: Append the dated M2 result.** Write a new paragraph immediately after the H-F2 M1 "Honest scope" paragraph. It must state, using the **actual numbers** from `validation/logs/exp_f2m2_stage1.json` and `exp_f2m2_stage2.json` (never invented ones): the majority-class chance level; each arm's self-retrieval top-1 and macro routing accuracy; which trunk arm won; the trunk/ref ratio and which decision-rule branch it landed in; the winning discretization and whether bar 4 passed; and the recalibrated `tau_match` with its ROC-AUC. Quote each figure with the log file that produced it. If any kill-criterion bar FAILED, say so plainly in the first sentence — a negative result is the finding, and the M1 entry's "honest scope" precedent applies.

- [ ] **Step 2: Verify the docs claim nothing the code lacks.** Re-read the M1 entry above it: M2 must not describe the mowm-side semantic path as existing unless Tasks 7–8 actually landed.

- [ ] **Step 3: Commit.**

```bash
git add CUBBYLLM_HYPOTHESES.md
git commit -m "docs(hypotheses): H-F2 M2 — semantic context encoding screen result

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JNaPeo6tU6xubkEfWTTLav"
```

---

> ## GATE — Tasks 7 and 8 run only if the kill criterion passed
>
> If Stage 1 bars 1–3 passed, proceed. If any failed, **stop after Task 6**: the
> screen's negative result is M2's deliverable, and the next cycle is M3
> (fine-tune a small embedder) with a measured motivation. Do not wire a
> semantic path that the screen showed does not route.

---

## Task 7: mowm's additive semantic axiom path

**Files:**
- Create: `mowm/encoding/__init__.py`, `mowm/encoding/semantic_axioms.py` (in `C:\Users\grill\Documents\GitHub\mowm`)
- Create: `mowm/tests/data/axiom_embeddings.npz` (copy of the Task-5 artifact, ~600 KB)
- Test: `mowm/tests/test_semantic_axioms.py`

**Interfaces:**
- Consumes: `data/axiom_embeddings.npz` from Task 5 (arrays `name_emb`, `formula_emb`, `domains`, `names`; scalars `arm`, `discretization`, `tau_match`).
- Produces: `make_projection(latent_dim, k=80, l=128, seed=0) -> np.ndarray`; `encode_semantic(emb, k=80, l=128, seed=0, discretization="dense") -> (n, k, l)`; `semantic_axioms(npz_path, view="formula") -> list[Axiom]`.

**Note:** mowm must NOT own the trunk. It consumes precomputed 512-D embeddings and re-derives the same orthonormal projection from the same fixed seed, so vectors match the screen exactly.

- [ ] **Step 1: Write the failing test.** Create `mowm/tests/test_semantic_axioms.py`:

```python
"""Semantic axiom encoding — additive path, hash encoding untouched."""
import numpy as np
import pathlib

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.encoding.semantic_axioms import encode_semantic, make_projection, semantic_axioms

NPZ = pathlib.Path(__file__).parent / "data" / "axiom_embeddings.npz"


def test_projection_is_orthonormal_and_cosine_exact():
    P = make_projection(512)
    assert P.shape == (80 * 128, 512)
    rng = np.random.default_rng(0)
    E = rng.standard_normal((8, 512)).astype(np.float32)
    proj = (E @ P.T)
    raw = E @ E.T / np.outer(np.linalg.norm(E, axis=1), np.linalg.norm(E, axis=1))
    got = proj @ proj.T / np.outer(np.linalg.norm(proj, axis=1), np.linalg.norm(proj, axis=1))
    assert np.abs(raw - got).max() < 1e-4


def test_semantic_axioms_load_with_expected_shape():
    ax = semantic_axioms(NPZ, view="formula")
    assert len(ax) == 280
    assert all(isinstance(a, Axiom) for a in ax)
    assert all(a.vector.shape == (80, 128) for a in ax)
    assert len({a.domain for a in ax}) == 15


def test_semantic_vectors_are_not_hash_vectors():
    # The whole point of M2: semantically related axioms must NOT be orthogonal
    # the way hash codes are.
    ax = semantic_axioms(NPZ, view="formula")
    phys = [a.vector.ravel() for a in ax if a.domain.split(".")[0] == "physics"][:20]
    M = np.stack(phys)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    off = (M @ M.T)[~np.eye(len(M), dtype=bool)]
    assert off.mean() > 0.01, f"within-domain cosine {off.mean():.4f} looks hash-like"


def test_hash_path_is_untouched():
    # An AxiomLibrary built the old way still produces hash-derived vectors.
    lib = AxiomLibrary(k=8, l=32)
    assert hasattr(lib, "encoder") and hasattr(lib.encoder, "_hash_to_vec")
```

- [ ] **Step 2: Run it, verify it fails.** From the mowm dir: `uv run pytest tests/test_semantic_axioms.py -q` → FAIL (`ModuleNotFoundError: mowm.encoding`).

- [ ] **Step 3: Copy the artifact and implement.** Copy `CubbyLLM/data/axiom_embeddings.npz` to `mowm/tests/data/axiom_embeddings.npz`, then create `mowm/encoding/__init__.py`:

```python
from .semantic_axioms import encode_semantic, make_projection, semantic_axioms

__all__ = ["encode_semantic", "make_projection", "semantic_axioms"]
```

and `mowm/encoding/semantic_axioms.py`:

```python
"""semantic_axioms — build Axioms whose vectors carry semantics, not hashes.

ADDITIVE: this does not touch AxiomLibrary.make_axiom or WorldEncoder, whose
BLAKE2b hash encoding stays exactly as it was. It consumes precomputed
embeddings (so mowm never owns the trunk) and re-derives the SAME orthonormal
projection from the same fixed seed used by CubbyLLM's M2 screen
(validation/exp_f2m2_semantic_routing.py), so vectors match bit-for-bit.
"""
from __future__ import annotations

import numpy as np

from ..axiom_library import Axiom


def make_projection(latent_dim: int, k: int = 80, l: int = 128, seed: int = 0) -> np.ndarray:  # noqa: E741
    """(k*l, latent_dim) with orthonormal columns -> preserves cosine exactly."""
    vsa_dim = k * l
    if vsa_dim < latent_dim:
        raise ValueError(f"need k*l={vsa_dim} >= latent_dim={latent_dim}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((vsa_dim, latent_dim)))
    return Q.astype(np.float32)


def encode_semantic(
    emb: np.ndarray, k: int = 80, l: int = 128, seed: int = 0,  # noqa: E741
    discretization: str = "dense",
) -> np.ndarray:
    """(n, latent) embeddings -> (n, k, l) block codes."""
    emb = np.asarray(emb, dtype=np.float32)
    P = make_projection(emb.shape[1], k=k, l=l, seed=seed)
    codes = (emb @ P.T).reshape(-1, k, l)
    if discretization == "dense":
        return codes
    if discretization == "argmax":
        idx = codes.argmax(axis=2)
        out = np.zeros_like(codes)
        n_idx, k_idx = np.meshgrid(np.arange(len(codes)), np.arange(k), indexing="ij")
        out[n_idx, k_idx, idx] = 1.0
        return out
    raise ValueError(f"unknown discretization {discretization!r}")


def semantic_axioms(npz_path, view: str = "formula") -> list[Axiom]:
    """Load the M2 artifact and build Axioms carrying semantic vectors."""
    data = np.load(npz_path, allow_pickle=True)
    emb = data["formula_emb"] if view == "formula" else data["name_emb"]
    codes = encode_semantic(emb, discretization=str(data["discretization"]))
    names = [str(x) for x in data["names"]]
    domains = [str(x) for x in data["domains"]]
    return [
        Axiom(name=n, domain=d, formula_str=n, vector=c, operators=[])
        for n, d, c in zip(names, domains, codes)
    ]
```

- [ ] **Step 4: Run the tests.** `uv run pytest tests/test_semantic_axioms.py -q` → 4 passed. Then the full suite `uv run pytest tests/ -q` → 745 passed (741 + 4), **0 new failures**.

- [ ] **Step 5: Commit** (mowm style, NO CubbyLLM trailers; stage ONLY these paths):

```bash
git -C ../mowm add mowm/encoding/__init__.py mowm/encoding/semantic_axioms.py tests/test_semantic_axioms.py tests/data/axiom_embeddings.npz
git -C ../mowm commit -m "feat(encoding): additive semantic axiom path over precomputed embeddings"
```

---

## Task 8: The live confirm through the real CubbyBridge

**Files:**
- Create: `mowm/tests/test_semantic_routing.py`

**Interfaces:**
- Consumes: `semantic_axioms` (Task 7); `AxiomLibrary`, `MoWMRouter`, `World` (mowm); `CubbyBridge` (M1); `Challenge` (cubbyllm).

**Note:** at k=80/l=128 each `World` builds HYLAs with `d_out=10240`, far larger than M1's toy k=4/l=32. Build the smallest world set that proves the point and keep `n_hylas=2`; if construction is slow or memory-heavy, reduce the number of seeded domains and say so in the test's docstring. Routing itself never touches HYLA — only `predict` does.

- [ ] **Step 1: Write the failing test.** Create `mowm/tests/test_semantic_routing.py`:

```python
"""H-F2 M2 live confirm: organic challenges route instead of spawning.

M1's fuzzy test had to build its context by perturbing an axiom, because hash
encoding made real similarity impossible. Here the challenge is a DIFFERENT
SURFACE FORM (the axiom's name) of a concept the world knows only through
OTHER axioms' formulas.
"""
from __future__ import annotations

import pathlib

import numpy as np
from cubbyllm.bridges.world_model import Challenge

from mowm.axiom_library import AxiomLibrary
from mowm.bridges import CubbyBridge
from mowm.encoding.semantic_axioms import encode_semantic, semantic_axioms
from mowm.router import MoWMRouter
from mowm.world import World

K, L = 80, 128
NPZ = pathlib.Path(__file__).parent / "data" / "axiom_embeddings.npz"
DOMAINS = ("physics", "logic", "bio")          # small, distinct, well populated


def _setup():
    data = np.load(NPZ, allow_pickle=True)
    tau = float(data["tau_match"])
    axioms = semantic_axioms(NPZ, view="formula")
    name_codes = encode_semantic(data["name_emb"], discretization=str(data["discretization"]))
    domains = [str(x) for x in data["domains"]]

    lib = AxiomLibrary(k=K, l=L)
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    for wid, dom in enumerate(DOMAINS, start=100):
        picked = [a for a, d in zip(axioms, domains) if d.split(".")[0] == dom][:8]
        for a in picked:
            lib.register(a)
        router.add_world(World(world_id=wid, k=K, l=L, n_hylas=2, z_depth=0,
                               z_max=2, tau=0.5, axioms=picked, seed=wid))
    return CubbyBridge(lib, router, tau_match=tau), name_codes, domains, axioms


def test_organic_challenge_routes_instead_of_spawning():
    bridge, name_codes, domains, axioms = _setup()
    before = bridge.num_worlds
    routed = total = 0
    for i, (code, dom) in enumerate(zip(name_codes, domains)):
        top = dom.split(".")[0]
        if top not in DOMAINS:
            continue
        total += 1
        r = bridge.attempt(Challenge(context=code, tag=f"organic-{i}"))
        if not r.spawned:
            routed += 1
    assert total >= 10
    assert routed / total >= 0.5, f"only {routed}/{total} organic challenges routed"
    assert bridge.num_worlds <= before + (total - routed)


def test_out_of_domain_challenge_still_spawns():
    bridge, name_codes, domains, _ = _setup()
    before = bridge.num_worlds
    idx = next(i for i, d in enumerate(domains) if d.split(".")[0] not in DOMAINS)
    r = bridge.attempt(Challenge(context=name_codes[idx], tag="far-out"))
    assert r.spawned is True
    assert bridge.num_worlds == before + 1


def test_exact_tag_repeat_still_reuses():
    bridge, name_codes, domains, _ = _setup()
    idx = next(i for i, d in enumerate(domains) if d.split(".")[0] == "physics")
    first = bridge.attempt(Challenge(context=name_codes[idx], tag="repeat"))
    after = bridge.num_worlds
    second = bridge.attempt(Challenge(context=name_codes[idx], tag="repeat"))
    assert second.spawned is False
    assert second.world_id == first.world_id
    assert bridge.num_worlds == after
```

- [ ] **Step 2: Run it, verify it fails.** From the mowm dir: `uv run pytest tests/test_semantic_routing.py -q` → FAIL (the routing assertion, or a collection error until Task 7 landed).

- [ ] **Step 3: Make it pass.** No new production code should be needed — M1's bridge and Task 7's encoder are the implementation. If `test_organic_challenge_routes_instead_of_spawning` fails on the ratio, that is a **real finding about the recalibrated `tau_match`**, not a test to weaken: record the observed routed/total and the tau, and report it to the controller before changing any threshold.

- [ ] **Step 4: Run the tests.** `uv run pytest tests/test_semantic_routing.py -q` → 3 passed. Then `uv run pytest tests/ -q` → 748 passed (745 + 3), **0 new failures**. Note the wall-clock of world construction at k=80/l=128 in the report.

- [ ] **Step 5: Commit** (mowm style, NO CubbyLLM trailers):

```bash
git -C ../mowm add tests/test_semantic_routing.py
git -C ../mowm commit -m "test(bridges): H-F2 M2 live confirm — organic challenges route, out-of-domain spawns"
```

---

## Self-review notes

- **Spec coverage.** §1 finding → Task 4's hash baseline bar. §2 assets → Tasks 1 (corpus), 3 (trunk + tokenizer + reconstruction recipe). §3 gaps → gap 1 Tasks 3–4, gap 2 Task 5 (`recalibrate_tau`), gap 3 Task 5 (argmax/PQ), gap 4 Task 4 (REF arm). §4 stages → Tasks 4 and 5. §4.1 metrics (self-retrieval, LOO domain routing, micro/macro, majority-class chance) → Task 4. §4.2 kill criterion bars 1–4 + decision rule → Tasks 4–5, enforced in code and printed. §4.3 live confirm → Task 8. §5 flow → Tasks 2–5. §6 file structure → the File Structure table. §7 constraints → Global Constraints. §8 risks → Task 3 (reconstruction, stop-don't-fallback), Task 8 (World cost), Task 5 (dense-vs-one-hot finding). §9 deferrals → not tasked, correct.
- **Type consistency.** `_encode_arm(arm, names, formulas) -> (name_codes, formula_codes)` returns `(n, K, L)` in every arm, so `self_retrieval_top1`, `loo_domain_routing`, `argmax_onehot`, `fit_pq`, and `pq_onehot` all take the same shape. `loo_domain_routing` returns `same_cos`/`diff_cos` arrays, which are exactly `recalibrate_tau`'s two inputs. `make_projection(latent_dim, seed=0)` uses identical seed and QR construction in the screen and in `mowm/encoding/semantic_axioms.py`, so both produce bit-identical vectors. `tau_match` flows checkpoint→`data/axiom_embeddings.npz`→`CubbyBridge(..., tau_match=...)`.
- **Gate.** Tasks 7–8 are explicitly gated on the Stage-1 kill criterion; a failed screen ends M2 at Task 6 with a negative result, which is a legitimate outcome, not an incomplete plan.
