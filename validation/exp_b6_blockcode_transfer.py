"""H-B6 transfer probe: does MAP-trained bindability carry over to BlockCodeOps?

The 2026-07-30 live 2B run trained the trunk with a **MAP** binding aux loss
(bipolar elementwise product) because MAP is differentiable. But the DECIDED
production algebra for the VM/world-model channel is **BlockCodeOps** (H-B5) —
a different VSA family (sparse block codes, per-block circular convolution).
Nothing so far shows the two transfer. This script answers that, offline, with
no training: load the trained checkpoint, pull real trunk features h, and test
role/filler bundle-then-unbind recovery under BOTH algebras.

The load-bearing comparison is against an UNTRAINED control. "BlockCode recovery
= 0.8" means nothing on its own; what matters is trained-vs-random-init on
identical inputs.

REPRODUCING THE TRAINED OBJECTIVE IS NOT OPTIONAL (v2, 2026-08-02). The v1 probe
scored MAP at cos 0.247 on trained h when the run's own loss said 0.842 — it
generated its own roles with numpy and pooled fillers globally. Both were wrong,
because the mechanism under test is that **the trunk co-adapts to one specific
fixed role set**. Probe with different roles and that structure is invisible: you
measure the 1/sqrt(n) crosstalk floor and conclude nothing happened. So:

  * roles come from the package's own ``make_roles`` (torch Generator, seed 0) —
    literally the vectors TrainLoop bound against, not a same-seeded numpy copy;
  * fillers are drawn WITHIN one sequence, as ``binding_aux_loss`` does, not
    pooled across the whole batch set;
  * the MAP path is cross-checked against ``binding_aux_loss`` itself each run.

The MAP-on-trained number is therefore a live self-test. If it does not land near
the run's final cos (~0.842 at n=16), the probe is not measuring the thing that
was trained and **every downstream number is void** — the verdict below refuses
to render in that case rather than reporting a transfer that isn't there.

DO NOT COMPARE ABSOLUTE COSINES ACROSS THE TWO ALGEBRAS. Measured on random h:
the MAP floor at n=16 is ~0.25 (= the theoretical 1/sqrt(16)), but BlockCode's is
~0.65 — per-block softmax makes every filler a non-negative distribution, so any
two are positively correlated by construction and cosine cannot approach 0. Only
the trained-minus-untrained DELTA and the retrieval accuracy carry information.

Usage (env knobs mirror train_colab.py; D/L/vocab are read from the ckpt meta):

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt.pt \
  CB_CORPUS=/content/token_cache CB_SOURCES=.../corpus_sources.json \
  python validation/exp_b6_blockcode_transfer.py

Without CB_CORPUS it falls back to RANDOM token ids and says so loudly.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import move_model  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.binding.torch_ops import binding_aux_loss, make_roles  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.ops import BlockCodeVSA  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


CKPT = os.environ.get("CB_CKPT", "")
CORPUS = os.environ.get("CB_CORPUS", "")
SOURCES = os.environ.get("CB_SOURCES", "")
SPM = os.environ.get("CUBBY_SPM", "")
CTX = _env("CB_CTX", 32); N_SLOTS = _env("CB_SLOTS", 8)
BATCH = _env("CB_B", 4); SEQ = _env("CB_S", 256)
N_BATCH = _env("CB_NBATCH", 8)
BLOCK_L = _env("CB_BLOCK_L", 128)
SEED = _env("CB_SEED", 0)
N_PAIRS = [int(x) for x in os.environ.get("CB_NPAIRS", "4,8,16,32").split(",")]
TRAINED_COS = _env("CB_TRAINED_COS", 0.842, float)   # run's final: bind 0.158 -> cos 0.842


def build(vocab, d, layers, dev):
    torch.manual_seed(0)                                  # the run's exact init
    cfg = CubbyConfig(d_model=d, n_layers=layers, ctx_dim=CTX, vocab_core=vocab)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=d, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(d, layers, grad_checkpoint=False),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=d * d), SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, d),
        head=TopKRetrievalHead(torch.randn(vocab, d) * 0.02, learnable=True),
        retrieval_k=vocab)
    move_model(model, dev)
    return model


def load_params(path, model, dev):
    ck = torch.load(path, map_location=dev)
    ps = list(model.parameters())
    if len(ck["params"]) != len(ps):
        raise SystemExit(f"param count mismatch: ckpt {len(ck['params'])} vs model {len(ps)}")
    with torch.no_grad():
        for p, s in zip(ps, ck["params"]):
            p.copy_(s.to(p.device))
    return ck


@torch.no_grad()
def collect_h(model, batches, dev):
    """Trunk features kept as (n_seq, S, d) — sequence structure PRESERVED, because
    the trained objective samples its fillers from within a single sequence."""
    out = [model.features(x.to(dev)).float().cpu() for x in batches]
    return torch.cat(out, dim=0)


def token_batches(vocab):
    if CORPUS:
        from cubbyllm.training import WeightedCorpusPipeline
        if SOURCES:
            srcs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
        else:
            srcs = [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                    for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))]
        pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=SEED, cache_only=True).prepare()
        it = pipe.batches(BATCH, SEQ)
        return [next(it)[0] for _ in range(N_BATCH)], "real corpus"
    g = torch.Generator().manual_seed(SEED)
    return ([torch.randint(0, vocab, (BATCH, SEQ), generator=g) for _ in range(N_BATCH)],
            "RANDOM ids (off-distribution — smoke test only)")


def _positions(n_seq, S, n, seed):
    """Within-sequence filler positions — identical scheme to binding_aux_loss,
    seeded so trained and control see the exact same draw."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, S, (n_seq, n), generator=g)


def map_probe(h, n, seed):
    """MAP — an exact mirror of the trained objective (cross-checked below)."""
    n_seq, S, d = h.shape
    roles = make_roles(n, d, h.device)                    # THE roles TrainLoop used
    idx = _positions(n_seq, S, n, seed)
    fillers = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d))   # (n_seq, n, d)
    r = roles.unsqueeze(0)
    composite = (r * fillers).sum(dim=1, keepdim=True)    # bundle
    recovered = composite * r                             # unbind (self-inverse)
    rn, fn = F.normalize(recovered, dim=-1), F.normalize(fillers, dim=-1)
    cos = (rn * fn).sum(-1).mean()
    sim = torch.bmm(rn, fn.transpose(1, 2))               # (n_seq, n, n)
    acc = (sim.argmax(-1) == torch.arange(n)).float().mean()
    return float(cos), float(acc)


def blockcode_probe(h, n, vsa, seed):
    """BlockCodeOps — same fillers/positions as MAP, production algebra.

    h is dense/real; block codes are per-block distributions, so features are
    reshaped to (k, l) and softmaxed per block (the continuous relaxation the ops
    facade documents). Roles are DISCRETE atoms with orthogonal=False — the ops
    docstring is explicit that the structured set is for resonator factorization
    and produces systematic crosstalk on role/filler duty.
    """
    n_seq, S, d = h.shape
    k, l = vsa.k, vsa.l
    idx = _positions(n_seq, S, n, seed)
    f = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d)).numpy()  # (n_seq, n, d)
    f = f.reshape(n_seq, n, k, l)
    f = f - f.max(axis=-1, keepdims=True)
    f = np.exp(f); f /= f.sum(axis=-1, keepdims=True)
    roles = vsa.codebook(n, orthogonal=False)             # (n, k, l) discrete atoms
    cos_all, acc_all = [], []
    for b in range(n_seq):
        comp = vsa.bundle([vsa.bind(roles[i], f[b, i]) for i in range(n)])
        rec = np.stack([vsa.unbind(comp, roles[i]) for i in range(n)]).reshape(n, -1)
        tgt = f[b].reshape(n, -1)
        rn = rec / (np.linalg.norm(rec, axis=-1, keepdims=True) + 1e-12)
        tn = tgt / (np.linalg.norm(tgt, axis=-1, keepdims=True) + 1e-12)
        cos_all.append((rn * tn).sum(-1).mean())
        acc_all.append(((rn @ tn.T).argmax(1) == np.arange(n)).mean())
    return float(np.mean(cos_all)), float(np.mean(acc_all))


def main():
    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT to the trained checkpoint (see module docstring)")

    meta = torch.load(CKPT, map_location="cpu")["meta"]
    d, layers, vocab = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    k = d // BLOCK_L
    if k * BLOCK_L != d:
        raise SystemExit(f"CB_BLOCK_L={BLOCK_L} does not divide d_model={d}")

    print("H-B6 transfer probe v2 — MAP (trained) vs BlockCodeOps (production)")
    print(f"  ckpt   {CKPT}")
    print(f"  model  d={d} L={layers} vocab={vocab} | device {dev} | torch {torch.__version__}")
    vsa = BlockCodeVSA(k=k, l=BLOCK_L)
    print(f"  blocks k={k} x l={BLOCK_L} (=d) | VSA backend: {vsa.backend()}")
    batches, src = token_batches(vocab)
    print(f"  tokens {BATCH}x{SEQ} x{N_BATCH} batches from {src}")
    print("  roles: package make_roles (torch seed 0) — the vectors TrainLoop bound against")
    print("  fillers: sampled WITHIN each sequence, as binding_aux_loss does")

    print("\n  building trained model ...", flush=True)
    trained = build(vocab, d, layers, dev)
    load_params(CKPT, trained, dev)
    h_tr = collect_h(trained, batches, dev)
    del trained
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print("  building untrained control ...", flush=True)
    control = build(vocab, d, layers, dev)
    h_ct = collect_h(control, batches, dev)
    del control
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  h: trained {tuple(h_tr.shape)}, control {tuple(h_ct.shape)}")

    # norm diagnostics — v1 saw trained MAP fall BELOW the crosstalk floor, which
    # unequal filler norms would explain. Record it either way.
    for nm, hh in (("trained", h_tr), ("control", h_ct)):
        nrm = hh.reshape(-1, d).norm(dim=-1)
        print(f"  ||h|| {nm:<8} mean {nrm.mean():8.3f}  std {nrm.std():8.3f}"
              f"  min {nrm.min():8.3f}  max {nrm.max():9.3f}")

    # ── self-test: does the MAP path reproduce binding_aux_loss exactly? ──────
    roles16 = make_roles(16, d, h_tr.device)
    torch.manual_seed(SEED)
    ref = 1.0 - float(binding_aux_loss(h_tr, roles16))
    mine, _ = map_probe(h_tr, 16, SEED)
    print(f"\n  self-test: binding_aux_loss says cos {ref:.3f}, this probe says {mine:.3f}"
          f" (differ only by position draw)")

    print("\n  n_pairs | algebra    | trained cos / acc | untrained cos / acc | delta cos")
    print("  --------+------------+-------------------+---------------------+----------")
    res = {}
    for n in N_PAIRS:
        for name, fn in (("MAP", map_probe), ("BlockCode", blockcode_probe)):
            if name == "MAP":
                c_tr, a_tr = fn(h_tr, n, SEED)
                c_ct, a_ct = fn(h_ct, n, SEED)
            else:
                c_tr, a_tr = fn(h_tr, n, vsa, SEED)
                c_ct, a_ct = fn(h_ct, n, vsa, SEED)
            res[(n, name)] = (c_tr, a_tr, c_ct, a_ct)
            print(f"  {n:>7} | {name:<10} |    {c_tr:.3f} / {a_tr:6.1%} |"
                  f"     {c_ct:.3f} / {a_ct:6.1%} |   {c_tr - c_ct:+.3f}")

    # ── verdict, GATED on the probe first proving it measures the trained thing ─
    n16 = 16 if 16 in N_PAIRS else N_PAIRS[len(N_PAIRS) // 2]
    m_tr, _, m_ct, _ = res[(n16, "MAP")]
    b_tr, b_acc, b_ct, _ = res[(n16, "BlockCode")]
    print(f"\n  NOTE: absolute cosines are NOT comparable across algebras (MAP floor"
          f" ~1/sqrt(n); BlockCode's is far higher — non-negative fillers). Read DELTA.")
    print(f"  GATE (n={n16}): MAP on trained h = {m_tr:.3f}; run's final was ~{TRAINED_COS:.3f};"
          f" untrained floor {m_ct:.3f} (theory {1/np.sqrt(n16):.3f}).")

    if m_tr - m_ct < 0.10:
        print("  => VOID. The probe does NOT reproduce the trained binding signal, so"
              " the BlockCode column measures nothing about transfer. Fix the probe"
              " before reading any delta. (v1 failed exactly here: wrong RNG for the"
              " roles + globally-pooled fillers => trained scored the crosstalk floor.)")
    else:
        print(f"  MAP gain from training:       {m_tr - m_ct:+.3f} cos  [gate PASSED]")
        print(f"  BlockCode gain from training: {b_tr - b_ct:+.3f} cos (retrieval {b_acc:.1%})")
        if b_tr - b_ct > 0.05:
            print("  => TRANSFERS: MAP-trained h is measurably more BlockCode-bindable than"
                  " untrained. H-B6 open item (2) resolves POSITIVE.")
        elif b_tr - b_ct < 0.01:
            print("  => DOES NOT TRANSFER: BlockCode recovery is no better than an untrained"
                  " trunk. H-B6 open item (2) resolves NEGATIVE — the aux loss shapes h for"
                  " MAP only; the production channel needs its own objective (or a trained"
                  " projection into the block-code basis).")
        else:
            print("  => WEAK/AMBIGUOUS transfer. Report the number, do not round it up.")
    print(f"\n  wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
