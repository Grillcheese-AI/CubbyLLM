"""DDP multi-GPU pretraining for the 2B CubbyModel. Launch with torchrun:

  torchrun --standalone --nproc_per_node=8 validation/train_ddp.py

Same stack as validation/train_colab.py (BBPE cache -> 2B trunk -> bf16 + grad
checkpointing + clip + warmup/cosine + resume), scaled to N GPUs:
  * one rank per GPU (NCCL); gloo+CPU fallback so the wiring is smoke-testable.
  * PER-RANK data seeding — each GPU samples different windows (seed + rank).
  * Manual gradient all-reduce (``TrainLoop(reduce_grads=True)``): CubbyModel is
    NOT an nn.Module, so we average grads over ``model.parameters()`` instead of
    DDP-wrapping it. Weights start identical (broadcast from rank 0) and stay in
    sync because every rank applies the same averaged gradient.
  * LR scaled for the larger effective batch (default sqrt(world_size)).
  * Rank 0 only: logging, sample generations, checkpointing.

Assumes a SINGLE multi-GPU node whose ranks share the filesystem (all read the
local token_cache; rank 0 writes the checkpoint). Env knobs mirror train_colab
(CB_D/L/CTX/SLOTS/B/S/STEPS/LR/WARMUP/CLIP/MIN_LR/GRAD_CKPT/AMP/EVAL/GEN/GEN_N/
CKPT/CKPT_EVERY + CB_CORPUS/CB_SOURCES/CUBBY_SPM) plus CB_SEED and CB_LR_SCALE
(sqrt|linear|<float>). Put the cache on fast local disk (no Drive on a real box).
"""
from __future__ import annotations
import contextlib
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402,F401  (kept for parity / future val split)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import move_model  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.training import TrainLoop, WeightedCorpusPipeline  # noqa: E402
from cubbyllm.training.data import _load_tokenizer  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


D = _env("CB_D", 2048); N_LAYERS = _env("CB_L", 32); CTX = _env("CB_CTX", 32)
N_SLOTS = _env("CB_SLOTS", 8); BATCH = _env("CB_B", 24); SEQ = _env("CB_S", 1024)
STEPS = _env("CB_STEPS", 1_630_000); LR = _env("CB_LR", 3e-4, float)
WARMUP = _env("CB_WARMUP", 2000); CLIP = _env("CB_CLIP", 1.0, float)
MIN_LR = _env("CB_MIN_LR", 0.1, float); EVAL = _env("CB_EVAL", 50)
GEN = _env("CB_GEN", 500); N_GEN = _env("CB_GEN_N", 5)
AMP = bool(_env("CB_AMP", 1)); GRAD_CKPT = bool(_env("CB_GRAD_CKPT", 1))
BIND_W = _env("CB_BIND_W", 0.0, float); BIND_N = _env("CB_BIND_N", 16)
REP_PEN = _env("CB_REP_PEN", 1.3, float); NO_REPEAT = _env("CB_NO_REPEAT", 3)
SEED = _env("CB_SEED", 0); LR_SCALE = os.environ.get("CB_LR_SCALE", "sqrt")
CKPT = os.environ.get("CB_CKPT", ""); CKPT_EVERY = _env("CB_CKPT_EVERY", 500)
CORPUS = os.environ.get("CB_CORPUS", ""); SOURCES = os.environ.get("CB_SOURCES", "")
SPM = os.environ.get("CUBBY_SPM", "")


def build(vocab, dev):
    torch.manual_seed(0)                             # identical init on every rank
    cfg = CubbyConfig(d_model=D, n_layers=N_LAYERS, ctx_dim=CTX, vocab_core=vocab)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=D, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(D, N_LAYERS, grad_checkpoint=GRAD_CKPT),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=D * D), SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, D),
        head=TopKRetrievalHead(torch.randn(vocab, D) * 0.02, learnable=True),
        retrieval_k=vocab)
    move_model(model, dev)
    return model


def save_ckpt(path, model, opt, step, meta):
    ck = {"step": step, "opt": opt.state_dict(), "meta": meta,
          "params": [p.detach().cpu() for p in model.parameters()]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"; torch.save(ck, tmp); os.replace(tmp, path)


def load_ckpt(path, model, opt, dev, meta):
    ck = torch.load(path, map_location=dev)
    ps = list(model.parameters())
    if ck.get("meta") != meta or len(ck["params"]) != len(ps):
        raise SystemExit(f"checkpoint mismatch {ck.get('meta')} != {meta}")
    with torch.no_grad():
        for p, s in zip(ps, ck["params"]):
            p.copy_(s.to(p.device))
    opt.load_state_dict(ck["opt"])
    return int(ck["step"])


@torch.no_grad()
def sample_text(model, decode, dev, vocab, n, eos, use_amp, max_new=48, temp=0.8, top_k=40,
                rep_pen=1.3, no_repeat=3):
    # Anti-repetition (DISPLAY-only, never touches training): a frequency-aware
    # repetition penalty + an n-gram block, so early-training loops ("and and",
    # "m/m/m") don't make the samples look worse than the model is. rep_pen=1.0 /
    # no_repeat=0 disables. Mirrors validation/train_colab.py::sample_text.
    seed = eos if eos >= 0 else 0
    outs = []
    for _ in range(n):
        ids = [seed]
        for _ in range(max_new):
            x = torch.tensor([ids[-SEQ:]], dtype=torch.long, device=dev)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_amp else contextlib.nullcontext())
            with ctx:
                logits = model.forward(x)[0, -1]
            logits = logits.float() / max(temp, 1e-6)
            gen = ids[1:]                               # generated so far (skip seed/EOS)
            if rep_pen and rep_pen != 1.0 and gen:
                counts = torch.bincount(torch.tensor(gen, device=logits.device),
                                        minlength=logits.shape[0]).float()
                factor = torch.pow(rep_pen, counts)     # 1.0 unseen; grows with repeats
                logits = torch.where(logits > 0, logits / factor, logits * factor)
            if no_repeat and len(ids) >= no_repeat:
                prefix = tuple(ids[-(no_repeat - 1):])
                for i in range(len(ids) - no_repeat + 1):
                    if tuple(ids[i:i + no_repeat - 1]) == prefix:
                        logits[ids[i + no_repeat - 1]] = float("-inf")
            if top_k and top_k < vocab:
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            if nxt == seed:
                break
            ids.append(nxt)
        outs.append(decode([t for t in ids if t != seed]).replace("\n", " ").strip()[:220])
    return outs


def main():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local); dev = torch.device(f"cuda:{local}")
    else:
        dev = torch.device("cpu")
    is0 = rank == 0
    amp = AMP and use_cuda

    def log(m):
        if is0:
            print(m, flush=True)

    encode, decode, eos, vocab = _load_tokenizer(SPM)
    if SOURCES:
        srcs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
    else:
        srcs = [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))]
    pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=SEED + rank,
                                  cache_only=True).prepare()

    scale = (math.sqrt(world) if LR_SCALE == "sqrt"
             else world if LR_SCALE == "linear" else float(LR_SCALE))
    lr = LR * scale
    log(f"DDP world={world} backend={'nccl' if use_cuda else 'gloo'} | per-gpu B={BATCH} "
        f"S={SEQ} -> global {world * BATCH * SEQ:,} tok/step")
    log(f"config d={D} L={N_LAYERS} vocab={vocab} | lr {LR:.1e} x{scale:.2f}={lr:.1e} | "
        f"amp {'bf16' if amp else 'off'} grad_ckpt {GRAD_CKPT}")

    model = build(vocab, dev)
    for p in model.parameters():                         # exact identical start
        dist.broadcast(p.data, src=0)

    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=lr, batch_size=BATCH, seq_len=SEQ,
                     device=dev, amp=amp, grad_clip=CLIP, warmup=WARMUP,
                     total_steps=STEPS, min_lr_ratio=MIN_LR, reduce_grads=True,
                     bind_weight=BIND_W, bind_n=BIND_N)
    log(f"trainable params {sum(p.numel() for p in model.parameters()):,} | "
        f"manifest {pipe.manifest_hash()[:16]}")

    meta = {"D": D, "L": N_LAYERS, "vocab": vocab}
    start = 0
    if CKPT and os.path.exists(CKPT):
        start = load_ckpt(CKPT, model, loop.opt, dev, meta)   # all ranks read same file
        for p in model.parameters():
            dist.broadcast(p.data, src=0)                     # guarantee exact sync
        loop._nstep = start
        pipe.seed = SEED + rank + start; loop._batches = pipe.batches(BATCH, SEQ)
        log(f"resumed @ step {start}")

    if use_cuda:
        torch.cuda.synchronize()
    t0 = last_t = time.perf_counter(); toks = 0; last_step = start; recent = []
    for s in range(start + 1, STEPS + 1):
        loss = loop.step(); toks += world * BATCH * SEQ; recent.append(loss)
        if is0 and (s % EVAL == 0 or s == 1):
            if use_cuda:
                torch.cuda.synchronize()
            now = time.perf_counter(); dt = now - t0
            sps = (now - last_t) / max(s - last_step, 1); last_t, last_step = now, s
            ce = sum(recent[-EVAL:]) / len(recent[-EVAL:]); lrn = loop.opt.param_groups[0]["lr"]
            log(f"  step {s:>6}  loss {ce:6.3f}  ppl {math.exp(min(ce, 80)):10.1f}  "
                f"lr {lrn:.2e}  {sps:6.3f} s/step  {toks/dt:>11,.0f} tok/s")
        if is0 and GEN and (s % GEN == 0 or s == 1):
            log(f"  — {N_GEN} sample generations @ step {s} —")
            for j, t in enumerate(sample_text(model, decode, dev, vocab, N_GEN, eos, amp,
                                              rep_pen=REP_PEN, no_repeat=NO_REPEAT), 1):
                log(f"    [{j}] {t}")
        if CKPT and CKPT_EVERY and s % CKPT_EVERY == 0 and is0:
            save_ckpt(CKPT, model, loop.opt, s, meta); log(f"  ✓ ckpt @ {s} -> {CKPT}")
    if is0 and CKPT:
        save_ckpt(CKPT, model, loop.opt, STEPS, meta)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
