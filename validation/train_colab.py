"""GPU training run for the assembled CubbyModel — built for Colab + CUDA.

Runs the REAL stack (frozen router -> hybrid embedding -> MinGRU backbone ->
hardened theta=f(c) memory -> retrieval head) through the REAL device-aware
``TrainLoop`` (hardener penalty included) on the GPU. On CUDA the backbone uses
the fast O(log T) parallel scan automatically. Shapes and data paths are env-
configurable so the same file scales from a laptop CPU to a 96 GB card.

  # Colab (after uploading the package + data — see docs/COLAB.md):
  CUBBY_SPM=/content/grillcheese_spm32k_v2.model \
  CUBBY_STORIES=/content/tinystory_50k.json \
  CB_D=512 CB_L=8 CB_B=64 CB_S=256 CB_STEPS=1000 \
  python validation/train_colab.py

Every CB_EVAL steps it prints loss/ppl/bpc; every CB_GEN steps it prints CB_GEN_N
sample generations (temperature+top-k). Set CB_CKPT to a path (on Colab: a Drive
path) to enable **resume**: it saves every CB_CKPT_EVERY steps and, on restart,
picks up exactly where it left off (weights + optimizer + step) — so a Colab
timeout costs at most CB_CKPT_EVERY steps.

For a real corpus set CB_STAGE to a local dir — it copies the Drive shards to
local disk once (idempotent) so random-window reads don't crawl over the FUSE mount.

Env knobs (all optional): CB_D CB_L CB_CTX CB_SLOTS CB_B CB_S CB_STEPS CB_LR
CB_EVAL CB_GEN CB_GEN_N CB_REP_PEN CB_NO_REPEAT CB_CKPT CB_CKPT_EVERY CB_STORIES_N ; corpus: CB_CORPUS
CB_SOURCES CB_STAGE ; data: CUBBY_SPM CUBBY_STORIES ; device: CB_DEVICE.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import describe, move_model, resolve_device  # noqa: E402
from cubbyllm.core.generation import (BasisHyperGenerator, HyperGenerator,  # noqa: E402
                                      SnapshotHardener)
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402
from cubbyllm.training import InMemoryDataPipeline, TrainLoop, WeightedCorpusPipeline  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


D = _env("CB_D", 512)
N_LAYERS = _env("CB_L", 8)
CTX = _env("CB_CTX", 32)
N_SLOTS = _env("CB_SLOTS", 8)
BATCH = _env("CB_B", 64)
SEQ = _env("CB_S", 256)
STEPS = _env("CB_STEPS", 1000)
LR = _env("CB_LR", 3e-3, float)
EVAL_EVERY = _env("CB_EVAL", 100)
GEN_EVERY = _env("CB_GEN", 500)              # sample generations every N steps
N_GEN = _env("CB_GEN_N", 5)                  # how many sample sentences
CKPT = os.environ.get("CB_CKPT", "")         # checkpoint path (Drive) -> enables resume
CKPT_EVERY = _env("CB_CKPT_EVERY", 500)      # save a checkpoint every N steps
AMP = bool(_env("CB_AMP", 1))                # bf16 mixed precision (CUDA only)
CLIP = _env("CB_CLIP", 1.0, float)           # grad-norm clip (stability) — 0 disables
WARMUP = _env("CB_WARMUP", 0)                # linear LR warmup steps (early-divergence guard)
MIN_LR = _env("CB_MIN_LR", 0.1, float)       # cosine-decay floor as a fraction of CB_LR
GRAD_CKPT = bool(_env("CB_GRAD_CKPT", 0))    # backbone activation checkpointing (fit bigger batch)
# CB_BIND_W: LEAVE AT 0. H-B6 was falsified 2026-08-02 — this loss is Goodhart-able
# and the trunk takes the degenerate solution: it drove retrieval 97% -> 11.5%
# (chance 6.25%) while its own number looked like clean progress. See H-B6.
BIND_W = _env("CB_BIND_W", 0.0, float)       # VSA binding aux-loss weight (0 = off) — MAP scheme
# CB_GEN: 'basis' = BasisHyperGenerator (context+layer -> coefficients over low-rank
# bases, 676,880 params). 'flat' = the original dense HyperGenerator, which emits a
# d*d block and costs 272,631,872 params — 13.6% of a 2B model. 403x difference.
GEN_KIND = os.environ.get("CB_GEN_KIND", "basis")
GEN_BASIS = _env("CB_GEN_BASIS", 16)         # number of low-rank bases
GEN_RANK = _env("CB_GEN_RANK", 8)            # rank of each basis adapter
HEALTH_EVERY = _env("CB_HEALTH", 250)        # representation-health probe cadence (0 = off)
PROMPTED = bool(_env("CB_GEN_PROMPT", 1))    # continue REAL text instead of cold-starting from EOS
PROMPT_LEN = _env("CB_PROMPT_LEN", 24)       # tokens of real prefix to condition on
BIND_N = _env("CB_BIND_N", 16)               # role/filler pairs bundled per step
REP_PEN = _env("CB_REP_PEN", 1.3, float)     # sampling repetition penalty (1.0 = off); freq-aware, breaks loops
NO_REPEAT = _env("CB_NO_REPEAT", 3)          # block repeating any n-gram of this size while sampling (0 = off)
STORIES_N = _env("CB_STORIES_N", 20000)
SPM = os.environ.get("CUBBY_SPM", r"C:\Users\grill\Documents\GitHub\cubby-lm"
                     r"\cubby\tokenizers\spm32k_1p7b\grillcheese_spm32k_v2.model")
STORIES = os.environ.get("CUBBY_STORIES",
                         r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json")
CORPUS = os.environ.get("CB_CORPUS", "")     # token-cache dir (uploaded) -> WeightedCorpusPipeline
SOURCES = os.environ.get("CB_SOURCES", "")   # corpus_sources.json for names+weights (optional)
STAGE = os.environ.get("CB_STAGE", "")       # local dir to copy the cache into (Drive random reads are slow)


def check_gpu(dev):
    print(f"device: {describe(dev)}")
    if dev.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"  {torch.cuda.get_device_name()} | compute capability {cap[0]}.{cap[1]}"
              f" | {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
        if cap[0] >= 12:  # Blackwell sm_120
            print("  (Blackwell sm_120 — needs a cu128 torch build; if you see "
                  "'no kernel image is available', install torch for CUDA 12.8.)")
        print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")


def _make_generator():
    """The theta=f(c) generator. 'basis' is the default: 676,880 params against
    the flat form's 272,631,872 at d=2048 (13.6% of a 2B model), and it conditions
    on layer identity as well as context. 'flat' keeps the original for A/B."""
    if GEN_KIND == "flat":
        return HyperGenerator(ctx_dim=CTX, n_out=D * D)
    return BasisHyperGenerator(ctx_dim=CTX, d_model=D, n_layers=N_LAYERS,
                               n_basis=GEN_BASIS, rank=GEN_RANK)


def build(vocab, dev):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=D, n_layers=N_LAYERS, ctx_dim=CTX, vocab_core=vocab)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=D, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(D, N_LAYERS, grad_checkpoint=GRAD_CKPT),
        memory=MemoryLayer(_make_generator(), SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, D),
        head=TopKRetrievalHead(torch.randn(vocab, D) * 0.02, learnable=True),
        retrieval_k=vocab,
    )
    move_model(model, dev)                       # BEFORE the optimizer is built
    return model


def _autocast(dev):
    """bf16 autocast on CUDA when AMP is on, else a no-op — for eval/generation
    forwards (TrainLoop.step handles its own)."""
    import contextlib
    if AMP and dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def eval_ce(model, val, vocab, dev, n_batches=12):
    losses, rng = [], np.random.default_rng(123)
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - SEQ - 1, size=min(BATCH, 32))
        x = torch.from_numpy(np.stack([val[i:i + SEQ] for i in ix])).to(dev)
        y = torch.from_numpy(np.stack([val[i + 1:i + SEQ + 1] for i in ix])).to(dev)
        with _autocast(dev):
            ce = F.cross_entropy(model.forward(x).reshape(-1, vocab), y.reshape(-1))
        losses.append(float(ce))
    return float(np.mean(losses))


@torch.no_grad()
def sample_text(model, decode, dev, vocab, n=5, max_new=48, temp=0.8, top_k=40, seed_id=2,
                rep_pen=1.3, no_repeat=3):
    """Autoregressively sample ``n`` short generations (temperature + top-k).

    Each starts from EOS (the document-boundary token the corpus is EOS-separated
    on, so this reads as "begin a fresh document") and stops on EOS or max_new.

    Two anti-repetition guards make the samples reflect the model, not the decoder
    (early training loves to fall into "and and and" / "m/m/m" loops): a
    frequency-aware repetition penalty (``rep_pen``, CTRL-style but scaled by how
    many times a token already fired, so tight loops get hit progressively) and an
    ``no_repeat``-gram block (a token that would repeat any n-gram is forbidden).
    Both are DISPLAY-only — they touch sampling, never training. Set rep_pen=1.0 /
    no_repeat=0 to disable."""
    seed = seed_id if seed_id is not None and seed_id >= 0 else 0
    outs = []
    for _ in range(n):
        ids = [seed]
        for _ in range(max_new):
            x = torch.tensor([ids[-SEQ:]], dtype=torch.long, device=dev)
            with _autocast(dev):
                logits = model.forward(x)[0, -1]
            logits = logits.float() / max(temp, 1e-6)   # fp32 for stable sampling
            gen = ids[1:]                               # generated so far (skip the seed/EOS token)
            if rep_pen and rep_pen != 1.0 and gen:      # freq-aware repetition penalty
                counts = torch.bincount(torch.tensor(gen, device=logits.device),
                                        minlength=logits.shape[0]).float()
                factor = torch.pow(rep_pen, counts)     # 1.0 where unseen; >1 grows with repeats
                logits = torch.where(logits > 0, logits / factor, logits * factor)
            if no_repeat and len(ids) >= no_repeat:     # ban tokens that would repeat an n-gram
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


@torch.no_grad()
def representation_health(model, pipe, dev, n=16, seqs=4, seq=256):
    """Is the trunk's h still a usable VSA substrate? Returns (retrieval, ff_cos).

    THE METRIC THAT CAN FAIL. On 2026-08-02 a 20,000-step run drove its binding
    loss 0.735 -> 0.158 while silently collapsing every hidden state in a sequence
    onto one direction: filler-filler cosine 0.006 -> 0.974, retrieval 97% -> 11.5%
    (chance 6.25%). Nothing in the training loss could see it, because the loss
    rewarded cosine-to-own-filler and never penalised cosine-to-OTHER-fillers.

    Retrieval is the property that actually matters — can you tell WHICH item was
    bound — and it costs one bundle/unbind on features we already computed.
    Healthy: retrieval ~95%+, ff ~0.2-0.3 on real text. Collapsed: retrieval near
    chance, ff -> 1.0. An UNTRAINED trunk scores ~97%, so this can only get worse.
    """
    import torch.nn.functional as F

    from cubbyllm.model.binding.torch_ops import make_roles

    it = pipe.batches(seqs, seq)
    x, _ = next(it)
    h = model.features(x.to(dev)).float()
    ns, S, d = h.shape
    roles = make_roles(n, d, h.device)
    idx = torch.randint(0, S, (ns, n), device=h.device)
    f = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d))
    r = roles.unsqueeze(0)
    rec = (r * f).sum(dim=1, keepdim=True) * r
    rn, fn = F.normalize(rec, dim=-1), F.normalize(f, dim=-1)
    sim = torch.bmm(rn, fn.transpose(1, 2))
    acc = (sim.argmax(-1) == torch.arange(n, device=h.device)).float().mean()
    ff = torch.bmm(fn, fn.transpose(1, 2))
    off = ff[:, ~torch.eye(n, dtype=bool, device=h.device)].mean()
    return float(acc), float(off)


@torch.no_grad()
def sample_continuations(model, decode, dev, vocab, pipe, n, max_new=64,
                         temp=0.8, top_k=40, rep_pen=1.3, no_repeat=3):
    """Continue REAL corpus text instead of cold-starting from EOS.

    Cold-starting exposes only the EOS->next distribution, which is sharply
    peaked: at step 500 all five samples opened with the same token, and the
    previous run kept landing on the same wikibooks title stub. That is a
    CROSS-sample diversity failure, and the repetition penalty cannot touch it —
    the penalty only sees within one generation. Conditioning each sample on a
    different real prefix both fixes the artefact and shows the thing actually
    worth judging: whether the model continues real text coherently.

    Returns (prompt_tail, continuation) pairs so the seam is visible.
    """
    x, _ = next(pipe.batches(n, PROMPT_LEN))
    outs = []
    for row in x:
        ids = [int(v) for v in row]
        prompt = list(ids)
        for _ in range(max_new):
            xb = torch.tensor([ids[-SEQ:]], dtype=torch.long, device=dev)
            with _autocast(dev):
                logits = model.forward(xb)[0, -1]
            logits = logits.float() / max(temp, 1e-6)
            gen = ids[len(prompt):]
            if rep_pen and rep_pen != 1.0 and gen:
                counts = torch.bincount(torch.tensor(gen, device=logits.device),
                                        minlength=logits.shape[0]).float()
                factor = torch.pow(rep_pen, counts)
                logits = torch.where(logits > 0, logits / factor, logits * factor)
            if no_repeat and len(ids) >= no_repeat:
                pre = tuple(ids[-(no_repeat - 1):])
                for i in range(len(ids) - no_repeat + 1):
                    if tuple(ids[i:i + no_repeat - 1]) == pre:
                        logits[ids[i + no_repeat - 1]] = float("-inf")
            if top_k and top_k < vocab:
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
            ids.append(int(torch.multinomial(torch.softmax(logits, dim=-1), 1)))
        outs.append((decode(prompt).replace("\n", " ").strip()[-90:],
                     decode(ids[len(prompt):]).replace("\n", " ").strip()[:200]))
    return outs


def stage_cache(src_dir, dst_dir):
    """Copy the ``.u32`` shards from a slow Drive mount to fast local disk.

    Training does random-window reads via memmap; on a Drive FUSE mount each is a
    network round-trip and throughput collapses, so we copy once to local disk.
    Idempotent — skips shards already present at the right size, so an interrupted
    copy (Colab timeout) just resumes on rerun. Returns the local dir to read from.
    """
    import glob
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src_dir, "*.u32")))
    for f in shards:
        dst = os.path.join(dst_dir, os.path.basename(f))
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f):
            continue                                 # already staged
        print(f"  staging {os.path.basename(f)} ({os.path.getsize(f)/1e9:.1f} GB)...",
              flush=True)
        shutil.copy2(f, dst + ".tmp")                # .tmp so a kill can't leave a
        os.replace(dst + ".tmp", dst)                # half-file that looks complete
    return dst_dir


# ── checkpoint / resume ──────────────────────────────────────────────────────
# CubbyModel and most of its pieces (HybridEmbedding, TopKRetrievalHead,
# HyperGenerator) are custom classes with a hand-rolled ``parameters()`` but NO
# ``state_dict()`` — so we checkpoint the raw parameter tensors in the stable
# order ``model.parameters()`` yields them (same order the optimizer was built
# over). A rebuilt model with the same config reproduces that order exactly.
def save_ckpt(path, model, opt, step, meta):
    """Atomically save every trainable tensor + optimizer + step (write to .tmp
    then os.replace, so a Colab kill mid-write can't corrupt the checkpoint)."""
    ck = {"step": step, "opt": opt.state_dict(), "meta": meta,
          "params": [p.detach().cpu() for p in model.parameters()]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, dev, meta):
    """Restore into an already-built+moved model and its optimizer (in place, so
    the optimizer keeps optimizing the same tensors). Returns the step to resume
    AFTER. Refuses an architecture-mismatched checkpoint."""
    ck = torch.load(path, map_location=dev)
    params = list(model.parameters())
    if ck.get("meta") != meta or len(ck["params"]) != len(params):
        raise SystemExit(f"checkpoint {path} (meta {ck.get('meta')}, "
                         f"{len(ck['params'])} tensors) != current model "
                         f"(meta {meta}, {len(params)} tensors); not resuming.")
    with torch.no_grad():
        for p, saved in zip(params, ck["params"]):
            p.copy_(saved.to(p.device))
    opt.load_state_dict(ck["opt"])
    return int(ck["step"])


def main():
    import glob

    # Generated samples (esp. early in training) contain arbitrary scripts; make
    # stdout tolerate them on any console (Windows cp1252, etc.). No-op on Colab.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from cubbyllm.training.data import _load_tokenizer

    dev = resolve_device(os.environ.get("CB_DEVICE", "auto"))
    check_gpu(dev)
    encode, decode, eos_id, vocab = _load_tokenizer(SPM)  # .json BBPE or .model SP

    corpus = CORPUS
    if CORPUS and STAGE and os.path.abspath(STAGE) != os.path.abspath(CORPUS):
        print(f"staging cache {CORPUS} -> {STAGE} (local disk; Drive random reads are slow)")
        t_stage = time.perf_counter()
        corpus = stage_cache(CORPUS, STAGE)
        print(f"  staged in {time.perf_counter()-t_stage:.0f}s")
    val = None
    if corpus:                                   # real corpus from an uploaded token cache
        if SOURCES:
            srcs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
        else:                                    # infer sources from the shards present
            srcs = [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                    for p in sorted(glob.glob(os.path.join(corpus, "*.u32")))]
        pipe = WeightedCorpusPipeline(srcs, SPM, corpus, seed=0, cache_only=True).prepare()
        cpt = 4.8                                # ~128k-BPE avg chars/tok (raw text absent in cache mode)
        print(f"corpus cache: {corpus}\n  mix {pipe.source_token_counts()}")
    else:                                        # tinystories smoke corpus
        text = "\n".join(json.load(open(STORIES, encoding="utf-8"))[:STORIES_N])
        data = np.array(encode(text), dtype=np.int64)
        cpt = len(text) / len(data)
        split = int(len(data) * 0.95)
        train, val = data[:split], data[split:]
        pipe = InMemoryDataPipeline(train, manifest=f"tinystory/{STORIES_N}/{vocab}", seed=0)

    print(f"config: d={D} L={N_LAYERS} B={BATCH} S={SEQ} steps={STEPS} | vocab {vocab}"
          f" | amp {'bf16' if (AMP and dev.type == 'cuda') else 'off'}")
    model = build(vocab, dev)
    n_params = sum(p.numel() for p in model.parameters())
    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=LR,
                     batch_size=BATCH, seq_len=SEQ, device=dev, amp=AMP,
                     grad_clip=CLIP, warmup=WARMUP, total_steps=STEPS, min_lr_ratio=MIN_LR,
                     bind_weight=BIND_W, bind_n=BIND_N)
    print(f"trainable params: {n_params:,} | manifest {pipe.manifest_hash()[:16]}\n")

    # gen kind is part of the architecture identity: swapping generators changes
    # the parameter list, so a stale checkpoint must be refused, not half-loaded.
    meta = {"D": D, "L": N_LAYERS, "vocab": vocab, "gen": GEN_KIND}
    start_step = 0
    if CKPT and os.path.exists(CKPT):
        start_step = load_ckpt(CKPT, model, loop.opt, dev, meta)
        loop._nstep = start_step                     # continue the LR schedule, don't restart it
        # advance the data stream so the resumed run draws fresh windows instead
        # of replaying the exact pseudo-random sequence the pre-resume run saw
        pipe.seed = int(getattr(pipe, "seed", 0)) + start_step
        loop._batches = pipe.batches(BATCH, SEQ)
        print(f"resumed from {CKPT} @ step {start_step} (train to {STEPS})\n")
    elif CKPT:
        print(f"no checkpoint at {CKPT} yet — fresh start, saving there every "
              f"{CKPT_EVERY} steps\n")

    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = last_t = time.perf_counter(); toks = 0; recent = []; last_step = start_step
    for s in range(start_step + 1, STEPS + 1):
        loss = loop.step(); toks += BATCH * SEQ; recent.append(loss)
        if s % EVAL_EVERY == 0 or s == 1:
            if dev.type == "cuda":
                torch.cuda.synchronize()                     # accurate wall time (GPU is async)
            now = time.perf_counter()
            dt = now - t0
            # avg over this log interval -> exactly seconds/step when CB_EVAL=1
            sec_step = (now - last_t) / max(s - last_step, 1)
            last_t, last_step = now, s
            # held-out eval for tinystories; recent train-loss avg for the corpus mode
            ce = (eval_ce(model, val, vocab, dev) if val is not None
                  else sum(recent[-EVAL_EVERY:]) / len(recent[-EVAL_EVERY:]))
            tag = "val" if val is not None else "train"
            lr_now = loop.opt.param_groups[0]["lr"]
            bind = f"  bind {loop._last_bind:5.3f}" if BIND_W > 0 else ""
            health = ""
            if HEALTH_EVERY and s % HEALTH_EVERY == 0:
                acc, ff = representation_health(model, pipe, dev)
                # RETRIEVAL ONLY drives the alarm. ff is printed as context and
                # must NOT gate it: measured 2026-08-02, a perfectly healthy run at
                # step 250 read ret 96.9% / ff 0.680, because early in training the
                # hidden states share a large common component and mean pairwise
                # cosine is high while residual differences stay discriminable. ff
                # falls as the model differentiates (a 7.5k-step control read 0.276).
                # Collapse is ret 11.5% / ff 0.934 — retrieval separates the cases,
                # a cosine statistic does not. Chance at n=16 is 6.25%.
                flag = "  <<< COLLAPSING" if acc < 0.75 else ""
                health = f"  ret {acc:5.1%} ff {ff:+.3f}{flag}"
            print(f"  step {s:>5}  {tag} loss {ce:6.3f}  ppl {math.exp(ce):8.1f}  "
                  f"bpc {ce/math.log(2)/cpt:5.3f}  lr {lr_now:.2e}{bind}{health}  "
                  f"{sec_step:6.3f} s/step  {toks/dt:>9,.0f} tok/s")
        if GEN_EVERY and (s % GEN_EVERY == 0 or s == 1):
            if PROMPTED and pipe is not None:
                print(f"  — {N_GEN} continuations @ step {s} (prompt -> generated) —")
                for j, (pr, co) in enumerate(sample_continuations(
                        model, decode, dev, vocab, pipe, N_GEN,
                        rep_pen=REP_PEN, no_repeat=NO_REPEAT), 1):
                    print(f"    [{j}] ...{pr}  ||  {co}")
            else:
                print(f"  — {N_GEN} sample generations @ step {s} —")
                for j, txt in enumerate(sample_text(model, decode, dev, vocab,
                                                    N_GEN, seed_id=eos_id,
                                                    rep_pen=REP_PEN, no_repeat=NO_REPEAT), 1):
                    print(f"    [{j}] {txt}")
        if CKPT and CKPT_EVERY and s % CKPT_EVERY == 0:
            save_ckpt(CKPT, model, loop.opt, s, meta)
            print(f"  ✓ checkpoint @ step {s} -> {CKPT}")
    if CKPT:                                          # final save (covers non-multiples)
        save_ckpt(CKPT, model, loop.opt, STEPS, meta)
        print(f"final checkpoint -> {CKPT}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
        print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
