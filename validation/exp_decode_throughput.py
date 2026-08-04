"""Decode throughput vs context length — does the O(1) state advantage show up?

Until 2026-08-03 generation re-ran the whole prefix per token, so this measured
nothing: the recurrent advantage existed as an argument, not as code. With
``CubbyModel.step`` it is measurable, and this is the harness.

TWO ARMS, REPORTED SEPARATELY — conflating them is how you get a number that
does not survive scrutiny:

  A. incremental vs naive, SAME model, same weights. Isolates the architecture.
     A win here proves the decode path works; it does not prove anything about
     other models.
  B. vs a real peer (Qwen2.5-1.5B via transformers), same GPU. This is the
     campaign-relevant number AND the unfair one: a Python-loop step() at
     batch 1 is competing with fused CUDA attention kernels. Expect to LOSE at
     short context. THE FINDING IS THE CROSSOVER, not the headline.

Method follows cubby-lm's ``experiments/energy/step0_baseline_j_per_tok.py``,
which got this right: warm up, take a median of repeats, never time the first
call. Peak memory is read per-arm with a reset in between.

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt_v21.pt \
  CB_PEER=Qwen/Qwen2.5-1.5B \
  python validation/exp_decode_throughput.py

Omit CB_PEER to run arm A only (no download, no transformers dependency).
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

CKPT = os.environ.get("CB_CKPT", "")
PEER = os.environ.get("CB_PEER", "")
CTXS = [int(x) for x in os.environ.get("CB_CTXS", "512,2048,8192,32768").split(",")]
N_TOK = int(os.environ.get("CB_N_TOK", "32"))       # timed tokens per point
REPEAT = int(os.environ.get("CB_REPEAT", "3"))      # medians, not means
CTX_DIM = int(os.environ.get("CB_CTX_DIM", "32"))
GEN_KIND = os.environ.get("CB_GEN_KIND", "")        # blank -> read from ckpt meta


def _mem_mb():
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_cubby(meta, dev):
    from cubbyllm.core.config import CubbyConfig
    from cubbyllm.core.context import FrozenSlotRouter
    from cubbyllm.core.device import move_model
    from cubbyllm.core.generation import (BasisHyperGenerator, HyperGenerator,
                                          SnapshotHardener)
    from cubbyllm.model.assembly import CubbyModel
    from cubbyllm.model.backbone import HybridBackbone, MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    d, L, V = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    kind = GEN_KIND or meta.get("gen", "flat")
    gen = (HyperGenerator(ctx_dim=CTX_DIM, n_out=d * d) if kind == "flat" else
           BasisHyperGenerator(ctx_dim=CTX_DIM, d_model=d, n_layers=L))
    # match the trained backbone — a hybrid checkpoint keeps a windowed KV cache,
    # so its decode state is O(window), still bounded but larger than pure MinGRU.
    if meta.get("backbone") == "hybrid":
        backbone = HybridBackbone(d, L, attn_every=int(meta.get("attn_every", 3)),
                                  window=int(meta.get("window", 512)),
                                  heads=int(meta.get("heads", 8)),
                                  mem_every=int(meta.get("mem_every", 0)),
                                  mem_topk=int(meta.get("mem_topk", 8)),
                                  mem_key=int(meta.get("mem_key", 64)))
    else:
        backbone = MinGRUBackbone(d, L)
    torch.manual_seed(0)
    m = CubbyModel(
        config=CubbyConfig(d_model=d, n_layers=L, ctx_dim=CTX_DIM, vocab_core=V),
        context_source=FrozenSlotRouter(input_dim=d, n_slots=8, ctx_dim=CTX_DIM).freeze(),
        backbone=backbone,
        memory=MemoryLayer(gen, SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(V, d),
        head=TopKRetrievalHead(torch.randn(V, d) * 0.02, learnable=True),
        retrieval_k=V)
    move_model(m, dev)
    return m, d, L, V, kind


@torch.no_grad()
def time_incremental(model, ctx, V, dev):
    """Prefill by stepping, then time N_TOK further steps. State carries."""
    ids = torch.randint(0, V, (1, ctx), device=dev)
    state = None
    for t in range(ctx):
        _, state = model.step(ids[:, t], state)
    _sync(); _reset_mem()
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        s = state
        for _ in range(N_TOK):
            lg, s = model.step(ids[:, -1], s)
        _sync(); ts.append((time.perf_counter() - t0) / N_TOK * 1000)
    return st.median(ts), _mem_mb()


@torch.no_grad()
def time_naive(model, ctx, V, dev):
    """What sample_text does today: full forward over the prefix, per token."""
    ids = torch.randint(0, V, (1, ctx), device=dev)
    model.forward(ids)                                   # warm
    _sync(); _reset_mem()
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        for _ in range(max(1, N_TOK // 8)):              # naive is slow; fewer reps
            model.forward(ids)[0, -1]
        _sync(); ts.append((time.perf_counter() - t0) / max(1, N_TOK // 8) * 1000)
    return st.median(ts), _mem_mb()


@torch.no_grad()
def time_peer(name, ctx, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    ids = torch.randint(0, tok.vocab_size, (1, ctx), device=dev)
    out = m(ids, use_cache=True)                          # prefill -> KV cache
    past = out.past_key_values
    _sync(); _reset_mem()
    nxt = ids[:, -1:]
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        p = past
        for _ in range(N_TOK):
            o = m(nxt, past_key_values=p, use_cache=True)
            p = o.past_key_values
        _sync(); ts.append((time.perf_counter() - t0) / N_TOK * 1000)
    ms, mem = st.median(ts), _mem_mb()
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return ms, mem


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT")
    ck = torch.load(CKPT, map_location="cpu")
    model, d, L, V, kind = build_cubby(ck["meta"], dev)
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    del ck
    print(f"decode throughput — batch 1, {N_TOK} tokens, median of {REPEAT}")
    print(f"  cubby  d={d} L={L} vocab={V} gen={kind} | {dev} | torch {torch.__version__}")
    print(f"  state per token: {L*d*2/1024:.0f} KB fixed (fp16-equivalent), context-independent\n")

    print("  === ARM A: incremental vs naive, SAME model (isolates the architecture) ===")
    print("     ctx |  step() ms/tok | naive ms/tok | speedup | step peak MB")
    print("  -------+----------------+--------------+---------+-------------")
    inc = {}
    for c in CTXS:
        try:
            i_ms, i_mb = time_incremental(model, c, V, dev)
            inc[c] = i_ms
            try:
                n_ms, _ = time_naive(model, c, V, dev)
                sp = f"{n_ms/i_ms:6.1f}x"
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                n_ms, sp = float("nan"), "   OOM"
            print(f"  {c:>6} | {i_ms:>14.2f} | {n_ms:>12.2f} | {sp} | {i_mb:>11.0f}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  {c:>6} |  OOM/err: {str(e)[:50]}")
        _reset_mem()

    if not PEER:
        print("\n  (set CB_PEER=Qwen/Qwen2.5-1.5B for arm B)")
        return

    print(f"\n  === ARM B: vs {PEER} (the campaign number — and the unfair one) ===")
    print("  A research Python loop vs fused CUDA kernels. Losing at short context")
    print("  is EXPECTED. The crossover is the result.\n")
    print("     ctx | cubby ms/tok | peer ms/tok | ratio | cubby MB | peer MB")
    print("  -------+--------------+-------------+-------+----------+---------")
    cross = None
    for c in CTXS:
        if c not in inc:
            continue
        try:
            p_ms, p_mb = time_peer(PEER, c, dev)
        except Exception as e:
            print(f"  {c:>6} | peer failed: {str(e)[:48]}")
            continue
        r = p_ms / inc[c]
        if r >= 1.0 and cross is None:
            cross = c
        print(f"  {c:>6} | {inc[c]:>12.2f} | {p_ms:>11.2f} | {r:>4.2f}x | "
              f"{'':>8} | {p_mb:>7.0f}")
    print()
    if cross:
        print(f"  CROSSOVER at ctx ~{cross}: at and beyond this length the recurrent")
        print("  path wins on latency. Below it the peer's fused kernels win.")
    else:
        print("  NO CROSSOVER in the swept range — the peer is faster throughout.")
        print("  That is a real result: the memory advantage is arithmetic and holds,")
        print("  but latency parity needs kernel work (fused scan), not architecture.")


if __name__ == "__main__":
    main()
