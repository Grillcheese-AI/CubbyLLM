"""train_base — the 450M CubbyLLM base: one Colab A100, about 20 hours.

Route A, phase 1 of ``docs/research/2026-09-23-training-acceleration.md``:
a d1024/L32 hybrid (MinGRU, window-512 attention every 3rd layer) trained on
~3B tokens, later width-cloned to the d2048/L32 2B shape on an H100. Four
heads of 256, so doubling the width gives the 2B's eight heads of 256.

What differs from ``train_colab.py``, and why:

* **AdamW + WSD.** Decoupled weight decay 0.1 on matrices; linear warmup,
  flat peak, then a 1-sqrt decay to zero over the last 20%. Any stable-phase
  checkpoint can be cooled down on its own (``CB_DECAY_START``), so a run that
  loses its session still ends as a finished model.
* **~131k tokens per optimizer step** by gradient accumulation.
* **torch.compile of the whole loss** (FlexAttention needs it on CUDA).
* **Vocabulary padded to a multiple of 128.** The head is ~30% of this
  model's FLOPs; padded ids never occur in data and are masked in samples.
* **Size-proportional mix x quality multipliers.** The pinned
  ``corpus_sources.json`` weights are per *source* (all 1.0), so taken
  literally they draw as many windows from the 0.9M-token ``pretrain_ext`` as
  from the 6.4B-token ``unified`` (~300 epochs of one, 0.05 of the other).
  ``CB_MIX=pinned`` reproduces that anyway. The decay phase reweights toward
  the best sources (``DECAY``).
* **Held-out tail of every shard = validation**, scored per source.
* **Deterministic sampling** from (seed, step, micro-batch): a resumed run
  draws exactly the windows the uninterrupted run would have.
* **Checkpoints**: two alternating slots on Drive plus, unless
  ``CB_STABLE_END=0``, the stable-phase end (to re-run the decay), written
  from a local copy in a background thread. About 7 GB each with the
  optimizer state; the final export is weights only (~2.3 GB).
* **Causal context.** Needs the 2026-09-23 ``infer_context`` (per-position
  running mean) and per-row ``BasisHyperGenerator.apply``; training and
  decode see the same context.
* No hardener term: ``SnapshotHardener`` holds no snapshots in a from-scratch
  run, so its penalty is zero (as it is in train_colab).

Modes (``CB_MODE``): ``train`` (default), ``bench`` (micro-batch throughput on
random tokens, no corpus), ``lrprobe`` (short equal-data runs at several LRs).
Every knob is an env var, echoed at start.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import queue
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.decoding import DecodeConfig, generate  # noqa: E402
from cubbyllm.core.device import move_model  # noqa: E402
from cubbyllm.core.generation import BasisHyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import HybridBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402


def _env(k, d, cast=int):
    v = os.environ.get(k)
    return d if v in (None, "") else cast(v)


MODE = _env("CB_MODE", "train", str)
# ── shape ──────────────────────────────────────────────────────────────────
D = _env("CB_D", 1024)
L = _env("CB_L", 32)
HEADS = _env("CB_HEADS", 4)                 # 1024/4 = 256 per head, as the 2B's 2048/8
WINDOW = _env("CB_WINDOW", 512)
ATTN_EVERY = _env("CB_ATTN_EVERY", 3)
CTX = _env("CB_CTX", 32)
N_SLOTS = _env("CB_SLOTS", 8)
GEN_BASIS = _env("CB_GEN_BASIS", 16)
GEN_RANK = _env("CB_GEN_RANK", 8)
VOCAB_PAD = _env("CB_VOCAB_PAD", 128)
# ── batch / optimizer / schedule ───────────────────────────────────────────
SEQ = _env("CB_S", 1024)
MICRO = _env("CB_MICRO", 16)                 # sequences per forward/backward
TOK_PER_STEP = _env("CB_TOKENS_PER_STEP", 131072)
LR = _env("CB_LR", 6e-4, float)              # peak; CB_MODE=lrprobe to choose it
BETA2 = _env("CB_BETA2", 0.95, float)
WD = _env("CB_WD", 0.1, float)
CLIP = _env("CB_CLIP", 1.0, float)
WARMUP = _env("CB_WARMUP", 300)              # optimizer steps (~39M tokens)
DECAY_FRAC = _env("CB_DECAY_FRAC", 0.2, float)
STEPS_ENV = _env("CB_STEPS", 0)              # 0 = size the run to CB_HOURS after measuring
DECAY_START_ENV = _env("CB_DECAY_START", 0)  # >0: start the decay here (cooldown branch)
HOURS = _env("CB_HOURS", 20.0, float)        # training budget, from script start
WALL_H = _env("CB_WALL_H", 22.5, float)      # hard stop (save + exit) before Colab's 24 h
OVERHEAD = _env("CB_OVERHEAD", 0.05, float)  # eval/gen/ckpt share of wall time
SEED = _env("CB_SEED", 0)
COMPILE = bool(_env("CB_COMPILE", 1))
# ── data ───────────────────────────────────────────────────────────────────
CORPUS = _env("CB_CORPUS", "", str)          # dir of <source>.u32 shards (local SSD)
SOURCES = _env("CB_SOURCES", "", str)        # corpus_sources.json (names + pinned weights)
MIX = _env("CB_MIX", "base", str)            # base | pinned
MIX_JSON = _env("CB_MIX_JSON", "", str)      # {"name": mult} overrides for the stable phase
DECAY_JSON = _env("CB_DECAY_JSON", "", str)  # {"name": mult} overrides for the decay phase
VAL_TOKENS = _env("CB_VAL_TOKENS", 2_000_000)  # held-out tail per shard (<= 5% of it)
TOKENIZER = _env("CUBBY_SPM", "", str)
# ── cadence / io ───────────────────────────────────────────────────────────
LOG_EVERY = _env("CB_LOG", 10)
EVAL_EVERY = _env("CB_EVAL", 250)
EVAL_WINDOWS = _env("CB_EVAL_WINDOWS", 8)    # windows per source per eval
GEN_EVERY = _env("CB_GEN", 500)
GEN_TOKENS = _env("CB_GEN_TOKENS", 48)
CKPT_MIN = _env("CB_CKPT_MIN", 45.0, float)
LOCAL_DIR = _env("CB_LOCAL_DIR", "/content/ckpt_base", str)
DRIVE_DIR = _env("CB_DRIVE_DIR", "", str)    # where checkpoints + metrics persist
NAME = _env("CB_NAME", "base450m", str)
RESUME_FROM = _env("CB_RESUME_FROM", "", str)  # a specific checkpoint (else the newest slot)
# Weights only, for a run that starts from another model's weights (a grown base,
# validation/grow_base.py): step 0, a fresh optimizer and schedule. Used only
# when there is nothing to resume, so a restarted run continues from its own
# slots rather than starting over.
INIT_FROM = _env("CB_INIT_FROM", "", str)
FFN_MULT = _env("CB_FFN_MULT", 2)            # SwiGLU width / D; 2 = every base so far
STABLE_END = bool(_env("CB_STABLE_END", 1))  # also keep the pre-decay state (+7 GB on Drive)
# ── bench / probe ──────────────────────────────────────────────────────────
BENCH_MICRO = _env("CB_BENCH_MICRO", "8,16,24,32", str)
PROBE_LRS = _env("CB_PROBE_LRS", "3e-4,6e-4,1.2e-3", str)
PROBE_STEPS = _env("CB_PROBE_STEPS", 150)
PROBE_WARMUP = _env("CB_PROBE_WARMUP", 30)

# Stable-phase multipliers on the size-proportional base. Knowledge lives in
# the external store, so encyclopedic/legal/fact shards are down-weighted and
# the educational, book, QA and code shards (fluency, format, programs) up.
QUALITY = {"fineweb_edu": 2.0, "books": 1.5, "wikibooks": 1.5, "kbtxt": 1.5,
           "nemotron_qa": 1.0, "nemotron_code": 1.0, "unified": 1.0,
           "wikipedia": 1.0, "arxiv": 1.0, "pretrain_ext": 1.0,
           "wiki_full": 0.75, "verified_facts": 0.5, "nemotron_legal": 0.5}
# Decay phase (last CB_DECAY_FRAC of steps): the best data last.
DECAY = {"fineweb_edu": 3.0, "books": 2.0, "wikibooks": 2.0, "kbtxt": 2.0,
         "nemotron_qa": 2.0, "nemotron_code": 2.0, "unified": 1.0,
         "wikipedia": 1.0, "arxiv": 1.0, "pretrain_ext": 1.0,
         "wiki_full": 0.5, "verified_facts": 0.25, "nemotron_legal": 0.25}

PROMPTS = [
    "The history of Quebec City begins",
    "La ville de Lévis est située sur la rive sud",
    "Question: Why does ice float on water?\nAnswer:",
    "def is_prime(n):\n",
]

PEAK_TFLOPS = {"A100": 312, "H100": 989, "H200": 989, "B200": 2250, "L4": 121,
               "A10": 125, "T4": 65, "RTX PRO 6000": 503, "4090": 165}


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
class Corpus:
    """Memmapped ``<source>.u32`` shards; the tail of each is validation."""

    def __init__(self, cache_dir: str, seq: int, val_tokens: int, seed: int):
        self.seq, self.seed = int(seq), int(seed)
        self.arrays, self.train_len, self.val_start = {}, {}, {}
        # a subset built by make_base_cache.py keeps each shard's ORIGINAL
        # held-out tail and records its length here
        hp = os.path.join(cache_dir, "_holdout.json")
        holds = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else {}
        sp = os.path.join(cache_dir, "_sizes.json")                 # original sizes
        self.orig = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
        for p in sorted(glob.glob(os.path.join(cache_dir, "*.u32"))):
            name = os.path.splitext(os.path.basename(p))[0]
            arr = np.memmap(p, dtype=np.uint32, mode="r")
            n = len(arr)
            hold = int(holds[name]) if name in holds else min(int(val_tokens), int(0.05 * n))
            if n - hold < 4 * (seq + 1) or hold < seq + 1:
                print(f"  skip shard {name}: {n:,} tokens is too small")
                continue
            self.arrays[name] = arr
            self.train_len[name] = n - hold
            self.val_start[name] = n - hold
        if not self.arrays:
            raise SystemExit(f"no usable .u32 shards in {cache_dir}")
        self.names = list(self.arrays)
        pinned = {}
        if SOURCES and os.path.exists(SOURCES):
            pinned = {s["name"]: float(s.get("weight", 1.0))
                      for s in json.load(open(SOURCES, encoding="utf-8"))["sources"]}
        stable, decay = dict(QUALITY), dict(DECAY)
        if MIX_JSON:
            stable.update(json.loads(MIX_JSON))
        if DECAY_JSON:
            decay.update(json.loads(DECAY_JSON))
        if MIX == "pinned":
            base = {n: pinned.get(n, 1.0) for n in self.names}
            self.probs = {"stable": self._norm(base), "decay": self._norm(base)}
        else:
            size = {n: float(self.orig.get(n, self.train_len[n])) for n in self.names}
            self.probs = {
                "stable": self._norm({n: size[n] * stable.get(n, 1.0) for n in self.names}),
                "decay": self._norm({n: size[n] * decay.get(n, 1.0) for n in self.names}),
            }

    def _norm(self, w: dict) -> np.ndarray:
        v = np.array([max(float(w[n]), 0.0) for n in self.names], dtype=np.float64)
        return v / v.sum()

    def batch(self, step: int, micro: int, b: int, phase: str):
        rng = np.random.default_rng([self.seed, int(step), int(micro)])
        picks = rng.choice(len(self.names), size=b, p=self.probs[phase])
        out = np.empty((b, self.seq + 1), dtype=np.int64)
        for j, k in enumerate(picks):
            name = self.names[k]
            i = int(rng.integers(0, self.train_len[name] - self.seq - 1))
            out[j] = self.arrays[name][i:i + self.seq + 1]
        t = torch.from_numpy(out)
        return t[:, :-1], t[:, 1:]

    def val_batch(self, name: str, n: int):
        """``n`` fixed, evenly spaced windows from ``name``'s held-out tail."""
        arr, s0 = self.arrays[name], self.val_start[name]
        span = len(arr) - s0 - self.seq - 1
        starts = [s0 + (span * i) // max(n - 1, 1) for i in range(n)] if span > 0 else [s0] * n
        out = np.stack([np.asarray(arr[s:s + self.seq + 1], dtype=np.int64) for s in starts])
        t = torch.from_numpy(out)
        return t[:, :-1], t[:, 1:]

    def report(self, stable_tokens: float, decay_tokens: float) -> str:
        rows = ["  source            tokens(train)  stable%  decay%  epochs this run"]
        for i, n in enumerate(self.names):
            drawn = stable_tokens * self.probs["stable"][i] + decay_tokens * self.probs["decay"][i]
            ep = drawn / self.train_len[n]
            flag = "  <-- repeats" if ep > 4 else ""
            rows.append(f"  {n:16s} {self.train_len[n]:>14,}  {100*self.probs['stable'][i]:6.2f}"
                        f"  {100*self.probs['decay'][i]:6.2f}  {ep:8.2f}{flag}")
        return "\n".join(rows)


class Prefetch:
    """Builds micro-batches ahead on a thread; the phase comes from ``sched``
    at build time, so a schedule change reaches the queue within its depth."""

    def __init__(self, corpus, sched, start_step, accum, micro, dev, depth=8):
        self.q = queue.Queue(maxsize=depth)
        self.stop = False
        pin = dev.type == "cuda"

        def run():
            try:
                produce()
            except BaseException as e:            # surface it in get(), don't hang
                self.q.put(e)

        def produce():
            step = start_step
            while not self.stop:
                step += 1
                phase = "decay" if step > sched["decay_start"] else "stable"
                for m in range(accum):
                    x, y = corpus.batch(step, m, micro, phase)
                    if pin:
                        x, y = x.pin_memory(), y.pin_memory()
                    while not self.stop:
                        try:
                            self.q.put((step, m, x, y), timeout=0.5)
                            break
                        except queue.Full:
                            continue
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.stop = True


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
def arch_meta(vocab_real: int, vocab: int) -> dict:
    meta = {"kind": "cubby_base", "D": D, "L": L, "heads": HEADS, "window": WINDOW,
            "attn_every": ATTN_EVERY, "ctx": CTX, "slots": N_SLOTS,
            "gen_basis": GEN_BASIS, "gen_rank": GEN_RANK, "vocab": vocab,
            "vocab_real": vocab_real, "mem_every": 0, "tied": False,
            "causal_ctx": True}
    if FFN_MULT != 2:                 # absent at 2, so every earlier checkpoint's meta still matches
        meta["ffn_mult"] = FFN_MULT
    return meta


def build_model(meta: dict, dev, seed: int = 0) -> CubbyModel:
    torch.manual_seed(seed)
    d, n_layers, v = meta["D"], meta["L"], meta["vocab"]
    cfg = CubbyConfig(d_model=d, n_layers=n_layers, ctx_dim=meta["ctx"], vocab_core=v)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=d, n_slots=meta["slots"],
                                        ctx_dim=meta["ctx"]).freeze(),
        backbone=HybridBackbone(d, n_layers, attn_every=meta["attn_every"],
                                ffn_mult=meta.get("ffn_mult", 2),
                                window=meta["window"], heads=meta["heads"]),
        memory=MemoryLayer(BasisHyperGenerator(ctx_dim=meta["ctx"], d_model=d,
                                               n_layers=n_layers, n_basis=meta["gen_basis"],
                                               rank=meta["gen_rank"]),
                           SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(v, d),
        head=TopKRetrievalHead(torch.randn(v, d) * 0.02, learnable=True),
        retrieval_k=v,
    )
    move_model(model, dev)
    return model


def flops_per_token(model: CubbyModel, meta: dict, seq: int) -> float:
    """Model FLOPs per token, forward + backward (6 per matmul weight, plus
    windowed attention's score and value products)."""
    n_mm = sum(p.numel() for p in model.backbone.parameters() if p.dim() >= 2)
    n_mm += meta["vocab"] * meta["D"]                                   # head
    n_mm += 2 * meta["D"] * meta["gen_basis"] * meta["gen_rank"]        # adapter
    n_attn = sum(1 for i in range(meta["L"]) if i % meta["attn_every"] == 0)
    w_avg = sum(min(t + 1, meta["window"]) for t in range(seq)) / seq
    return 6.0 * n_mm + 12.0 * meta["D"] * w_avg * n_attn


def peak_flops(dev) -> float:
    if os.environ.get("CB_PEAK_TFLOPS"):
        return float(os.environ["CB_PEAK_TFLOPS"]) * 1e12
    if dev.type != "cuda":
        return float("nan")
    name = torch.cuda.get_device_name()
    for k, v in PEAK_TFLOPS.items():
        if k in name:
            return v * 1e12
    return float("nan")


def make_opt(model: CubbyModel, lr: float, dev):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": WD},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=lr, betas=(0.9, BETA2), eps=1e-8)
    if dev.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, **kw)


def make_fns(model: CubbyModel, vocab: int, dev):
    amp = dev.type == "cuda"

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model.forward(x)
        return F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))

    def tok_loss_fn(x, y):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model.forward(x)
            return F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1),
                                   reduction="none").reshape(y.shape)

    if COMPILE and dev.type == "cuda":
        return torch.compile(loss_fn), torch.compile(tok_loss_fn)
    return loss_fn, tok_loss_fn


def lr_at(step: int, sched: dict, peak: float) -> float:
    w, t, ds = sched["warmup"], sched["steps"], sched["decay_start"]
    if w and step <= w:
        return peak * step / w
    if step <= ds:
        return peak
    frac = min(1.0, (step - ds) / max(t - ds, 1))
    return peak * max(0.0, 1.0 - math.sqrt(frac))                     # 1-sqrt to zero


# ─────────────────────────────────────────────────────────────────────────────
# checkpoints
# ─────────────────────────────────────────────────────────────────────────────
def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class Checkpoints:
    """Local write, then a background copy to one of two alternating Drive
    slots. A slot's sidecar json is invalidated before its file is
    overwritten and rewritten after the copy completes, so a kill mid-copy
    leaves the other slot as the newest valid checkpoint."""

    def __init__(self, local_dir: str, drive_dir: str, name: str):
        self.local_dir, self.drive_dir, self.name = local_dir, drive_dir, name
        os.makedirs(local_dir, exist_ok=True)
        if drive_dir:
            os.makedirs(drive_dir, exist_ok=True)
        self._copy: "threading.Thread | None" = None

    def _side(self, base):
        return base + ".json"

    def _slots(self):
        return [os.path.join(self.drive_dir, f"{self.name}_slot{s}.pt") for s in "ab"]

    def _read_side(self, path):
        try:
            meta = json.load(open(self._side(path), encoding="utf-8"))
            if meta.get("step", -1) >= 0 and os.path.getsize(path) == meta.get("bytes"):
                return meta
        except (OSError, ValueError):
            pass
        return None

    def wait(self):
        if self._copy is not None:
            self._copy.join()
            self._copy = None

    def save(self, payload: dict, tag: str = "") -> str:
        self.wait()                                     # one copy in flight at a time
        local = os.path.join(self.local_dir, f"{self.name}_{tag or 'latest'}.pt")
        torch.save(payload, local + ".tmp")
        os.replace(local + ".tmp", local)
        side = {"step": int(payload["step"]), "bytes": os.path.getsize(local),
                "time": time.strftime("%F %T")}
        _write_json(self._side(local), side)
        if not self.drive_dir:
            return local
        if tag:
            dst = os.path.join(self.drive_dir, f"{self.name}_{tag}.pt")
        else:                                           # the older (or invalid) slot
            a, b = self._slots()
            sa, sb = self._read_side(a), self._read_side(b)
            dst = a if (sa is None or (sb is not None and sa["step"] <= sb["step"])) else b

        def copy():
            try:
                _write_json(self._side(dst), {"step": -1})
                shutil.copyfile(local, dst)
                _write_json(self._side(dst), side)
                print(f"  [ckpt] step {side['step']} -> {dst}", flush=True)
            except OSError as e:
                print(f"  [ckpt] copy to Drive FAILED ({e}); local copy kept at {local}",
                      flush=True)
        self._copy = threading.Thread(target=copy, daemon=True)
        self._copy.start()
        return local

    def latest(self) -> "str | None":
        cands = [os.path.join(self.local_dir, f"{self.name}_latest.pt")]
        if self.drive_dir:
            cands += self._slots()
        best, best_step = None, -1
        for c in cands:
            meta = self._read_side(c) if os.path.exists(c) else None
            if meta and meta["step"] > best_step:
                best, best_step = c, meta["step"]
        return best


def payload(model, opt, step, meta, sched, extra) -> dict:
    return {"step": int(step), "meta": meta, "sched": dict(sched),
            "params": [p.detach() for p in model.parameters()],
            "router": [p.detach() for p in model.context_source.parameters()],
            "opt": opt.state_dict() if opt is not None else None, **extra}


def restore(path, model, opt, dev, meta) -> dict:
    ck = torch.load(path, map_location=dev, weights_only=False)
    params = list(model.parameters())
    if ck["meta"] != meta or len(ck["params"]) != len(params):
        raise SystemExit(f"checkpoint {path} is a different model:\n  {ck['meta']}\n  vs {meta}")
    with torch.no_grad():
        for p, s in zip(params, ck["params"]):
            p.copy_(s.to(p.device))
        for p, s in zip(model.context_source.parameters(), ck["router"]):
            p.copy_(s.to(p.device))
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    return ck


def load_base(path: str, device="cpu") -> CubbyModel:
    """Rebuild a trained base from a checkpoint or final export, for decode,
    evaluation or width-cloning. Returns a CubbyModel on ``device``."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["meta"], torch.device(device))
    restore(path, model, None, torch.device(device), ck["meta"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# evaluation and samples
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(corpus, tok_loss_fn, dev, probs) -> dict:
    out, mix = {}, 0.0
    for i, n in enumerate(corpus.names):
        x, y = corpus.val_batch(n, EVAL_WINDOWS)
        out[n] = float(tok_loss_fn(x.to(dev), y.to(dev)).mean())
        mix += probs[i] * out[n]
    out["_mix"] = mix
    # in-context copy: a random half-sequence, then the same again. Loss on the
    # repeat falls toward 0 once an induction/copy circuit exists.
    g = torch.Generator().manual_seed(1234)
    half = SEQ // 2
    a = torch.randint(1000, 30000, (EVAL_WINDOWS, half + 1), generator=g)
    seq = torch.cat([a[:, :half], a[:, :half + 1]], dim=1)[:, :SEQ + 1]
    tl = tok_loss_fn(seq[:, :-1].to(dev), seq[:, 1:].to(dev))
    out["_copy"] = float(tl[:, half:].mean())
    return out


def samples(model, dev, encode, decode, vocab_real, step) -> list:
    rows = []
    cfg = DecodeConfig(max_new_tokens=GEN_TOKENS, temperature=0.8, top_k=40,
                       repetition_penalty=1.3, no_repeat_ngram=3, seed=int(step))

    def step_fn(tok, state):
        return model.step(tok.to(dev), state)
    with torch.no_grad():
        for p in PROMPTS:
            ids = encode(p)
            out, _ = generate(model, ids, cfg, state=model.init_state(1, dev), step=step_fn)
            rows.append((p, decode([t for t in out if t < vocab_real])))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# modes
# ─────────────────────────────────────────────────────────────────────────────
def device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"device: {torch.cuda.get_device_name()} | "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB | torch {torch.__version__}")
    else:
        print(f"device: cpu | torch {torch.__version__}")
    return dev


def vocab_sizes():
    if TOKENIZER:
        from cubbyllm.training.data import _load_tokenizer
        enc, dec, eos, v = _load_tokenizer(TOKENIZER)
    else:
        enc = dec = None
        v = _env("CB_VOCAB", 127996)
    padded = ((v + VOCAB_PAD - 1) // VOCAB_PAD) * VOCAB_PAD
    return enc, dec, v, padded


def bench(dev):
    _, _, v, vp = vocab_sizes()
    meta = arch_meta(v, vp)
    pk = peak_flops(dev)
    print(f"bench: D{D}/L{L} heads {HEADS} S{SEQ} vocab {v}->{vp} | micro {BENCH_MICRO}")
    rows = []
    for mb in [int(s) for s in BENCH_MICRO.split(",")]:
        model = opt = None
        try:
            model = build_model(meta, dev, SEED)
            opt = make_opt(model, 1e-4, dev)
            loss_fn, _ = make_fns(model, vp, dev)
            fpt = flops_per_token(model, meta, SEQ)
            n_params = sum(p.numel() for p in model.parameters())
            x = torch.randint(0, v, (mb, SEQ), device=dev)
            y = torch.randint(0, v, (mb, SEQ), device=dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t_c = time.perf_counter()
            for _ in range(3):                                      # compile + warm
                loss_fn(x, y).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_c = time.perf_counter() - t_c
            n = 12
            t0 = time.perf_counter()
            for i in range(n):
                loss_fn(x, y).backward()
                if i % 4 == 3:
                    opt.step(); opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n
            tps = mb * SEQ / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
            mfu = tps * fpt / pk
            rows.append((mb, tps, mfu, mem))
            print(f"  micro {mb:3d}: {1e3*dt:7.1f} ms/micro  {tps:9,.0f} tok/s  MFU {100*mfu:5.1f}%"
                  f"  peak {mem:5.1f} GB  | {n_params/1e6:.0f}M params, {fpt/1e9:.2f} GFLOP/tok,"
                  f" compile+warm {t_c:.0f}s", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  micro {mb:3d}: out of memory")
        finally:
            del model, opt
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            torch._dynamo.reset()
    if rows:
        mb, tps, mfu, mem = max(rows, key=lambda r: r[1])
        tokens = tps * HOURS * 3600 * (1 - OVERHEAD)
        print(f"\nfastest: CB_MICRO={mb} ({tps:,.0f} tok/s, MFU {100*mfu:.1f}%, {mem:.0f} GB)."
              f" {HOURS:.0f} h ~ {tokens/1e9:.2f}B tokens ~ {int(tokens // TOK_PER_STEP):,}"
              f" optimizer steps of {TOK_PER_STEP:,} tokens.")


def lrprobe(dev):
    enc, dec, v, vp = vocab_sizes()
    meta = arch_meta(v, vp)
    corpus = Corpus(CORPUS, SEQ, VAL_TOKENS, SEED)
    accum = max(1, TOK_PER_STEP // (MICRO * SEQ))
    lrs = [float(s) for s in PROBE_LRS.split(",")]
    sched = {"warmup": PROBE_WARMUP, "steps": 10**9, "decay_start": 10**9}
    print(f"lrprobe: {lrs} x {PROBE_STEPS} steps x {accum*MICRO*SEQ:,} tokens, same windows per arm")
    res = []
    for lr in lrs:
        model = build_model(meta, dev, SEED)
        opt = make_opt(model, lr, dev)
        loss_fn, tok_loss_fn = make_fns(model, vp, dev)
        t0, losses, dead = time.perf_counter(), [], False
        for step in range(1, PROBE_STEPS + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, sched, lr)
            tot = 0.0
            for m in range(accum):
                x, y = corpus.batch(step, m, MICRO, "stable")
                loss = loss_fn(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)) / accum
                loss.backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(tot)
            if not math.isfinite(tot):
                dead = True
                break
            if step % 25 == 0:
                print(f"  lr {lr:.1e} step {step:4d} loss {np.mean(losses[-25:]):.3f}", flush=True)
        ev = evaluate(corpus, tok_loss_fn, dev, corpus.probs["stable"]) if not dead else {"_mix": float("nan")}
        res.append((lr, ev["_mix"], dead, time.perf_counter() - t0))
        print(f"  lr {lr:.1e}: val_mix {ev['_mix']:.4f}{'  DIVERGED' if dead else ''}"
              f"  ({res[-1][3]/60:.1f} min)", flush=True)
        del model, opt
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        torch._dynamo.reset()
    ok = [r for r in res if not r[2] and math.isfinite(r[1])]
    if ok:
        best = min(ok, key=lambda r: r[1])
        print(f"\nbest at {PROBE_STEPS} steps: CB_LR={best[0]:g} (val_mix {best[1]:.4f})."
              " Short probes favour high LRs; if the next-lower LR is within ~0.02,"
              " take the lower one for the full run.")


def train(dev):
    enc, dec, v, vp = vocab_sizes()
    meta = arch_meta(v, vp)
    corpus = Corpus(CORPUS, SEQ, VAL_TOKENS, SEED)
    accum = max(1, TOK_PER_STEP // (MICRO * SEQ))
    tps_step = accum * MICRO * SEQ
    print(f"config: D{D}/L{L} heads {HEADS} window {WINDOW} 1:{ATTN_EVERY} | vocab {v}->{vp}"
          f" | S{SEQ} micro {MICRO} x accum {accum} = {tps_step:,} tok/step"
          f" | lr {LR:g} b2 {BETA2} wd {WD} clip {CLIP} warmup {WARMUP}"
          f" decay {DECAY_FRAC:.0%} | mix {MIX} | hours {HOURS} wall {WALL_H}")

    model = build_model(meta, dev, SEED)
    opt = make_opt(model, LR, dev)
    n_params = sum(p.numel() for p in model.parameters())
    fpt = flops_per_token(model, meta, SEQ)
    pk = peak_flops(dev)
    print(f"params {n_params/1e6:.1f}M (trainable; embedding + head "
          f"{2*vp*D/1e6:.0f}M) | {fpt/1e9:.2f} GFLOP/token")

    ck = Checkpoints(LOCAL_DIR, DRIVE_DIR, NAME)
    sched = {"warmup": WARMUP, "steps": STEPS_ENV or 10**9,
             "decay_start": 10**9, "tokens_per_step": tps_step, "sized": bool(STEPS_ENV)}
    if STEPS_ENV:
        sched["decay_start"] = int(STEPS_ENV * (1 - DECAY_FRAC))
    start, tokens_seen, hours_before = 0, 0, 0.0
    path = RESUME_FROM or ck.latest()
    if path:
        c = restore(path, model, opt, dev, meta)
        start, tokens_seen = c["step"], c.get("tokens_seen", 0)
        hours_before = c.get("train_hours", 0.0)
        sched = dict(c["sched"])
        if c["sched"].get("tokens_per_step") != tps_step:
            raise SystemExit(f"tokens/step changed ({c['sched'].get('tokens_per_step')} -> "
                             f"{tps_step}); keep CB_MICRO x accum the same on resume.")
        print(f"resumed {path} @ step {start} ({tokens_seen/1e9:.2f}B tokens, "
              f"{hours_before:.1f} h trained)")
        del c
    elif INIT_FROM:
        c = restore(INIT_FROM, model, None, dev, meta)       # weights only; meta must match
        print(f"initialised from {INIT_FROM} (its step {c.get('step')}; "
              f"{'grown from ' + str(c['grown_from']['checkpoint']) if c.get('grown_from') else 'not grown'})"
              f" -> step 0, fresh optimizer and schedule")
        del c
    if STEPS_ENV and STEPS_ENV != sched["steps"]:
        sched.update(steps=STEPS_ENV, sized=True)
        sched["decay_start"] = min(sched["decay_start"], int(STEPS_ENV * (1 - DECAY_FRAC)))
    if DECAY_START_ENV:
        if DECAY_START_ENV < start:
            raise SystemExit(f"CB_DECAY_START={DECAY_START_ENV} is before the resumed step {start};"
                             f" a cooldown branch starts at or after the checkpoint it resumes.")
        sched["decay_start"] = DECAY_START_ENV
        if not STEPS_ENV:
            sched["steps"] = DECAY_START_ENV + max(1, int(DECAY_START_ENV * DECAY_FRAC / (1 - DECAY_FRAC)))
        sched["sized"] = True
        print(f"cooldown branch: decay {sched['decay_start']} -> {sched['steps']}")
    if sched["sized"]:
        print(f"schedule: {sched['steps']:,} steps, decay from {sched['decay_start']:,}")
        est = corpus.report((sched["decay_start"]) * tps_step,
                            (sched["steps"] - sched["decay_start"]) * tps_step)
        print(est)

    loss_fn, tok_loss_fn = make_fns(model, vp, dev)
    metrics_path = os.path.join(DRIVE_DIR or LOCAL_DIR, f"{NAME}_metrics.jsonl")
    mlog = open(metrics_path, "a", encoding="utf-8")

    def log(rec):
        mlog.write(json.dumps(rec) + "\n")
        mlog.flush()

    t_start = time.perf_counter()
    last_ckpt = t_start
    stable_saved = start >= sched["decay_start"]
    last_saved = start
    pf = Prefetch(corpus, sched, start, accum, MICRO, dev)
    win_loss, win_t, win_steps = [], time.perf_counter(), 0
    measure_from = None
    step, poisoned, done = start, False, start       # done = last COMPLETED optimizer step
    try:
        while step < sched["steps"]:
            step += 1
            lr = lr_at(step, sched, LR)
            for g in opt.param_groups:
                g["lr"] = lr
            tot = 0.0
            for m in range(accum):
                s_, m_, x, y = pf.get()
                assert (s_, m_) == (step, m), f"prefetch out of order: {(s_, m_)} != {(step, m)}"
                loss = loss_fn(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)) / accum
                loss.backward()
                tot += loss.detach()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            done = step
            tokens_seen += tps_step
            win_steps += 1
            if step % LOG_EVERY == 0 or step == start + 1:
                tot, gn = float(tot), float(gnorm)
                if not math.isfinite(tot):
                    print(f"  step {step}: loss {tot} NON-FINITE. Stopping WITHOUT saving; the "
                          f"last checkpoint is intact. Resume with a lower CB_LR.", flush=True)
                    poisoned = True
                    break
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                tps = win_steps * tps_step / (now - win_t)
                el = (now - t_start) / 3600
                # size the run from measured speed (once, fresh runs only)
                if not sched["sized"]:
                    if measure_from is None and step >= start + 10:
                        measure_from = (step, now)
                    elif measure_from and step >= measure_from[0] + 30:
                        sps = (now - measure_from[1]) / (step - measure_from[0])
                        left = HOURS * 3600 * (1 - OVERHEAD) - (now - t_start)
                        sched["steps"] = step + max(1, int(left / sps))
                        sched["decay_start"] = max(step, int(sched["steps"] * (1 - DECAY_FRAC)))
                        sched["sized"] = True
                        print(f"  sized: {sps:.2f} s/step -> {sched['steps']:,} steps "
                              f"({sched['steps']*tps_step/1e9:.2f}B tokens), decay from "
                              f"{sched['decay_start']:,}", flush=True)
                        print(corpus.report(sched["decay_start"] * tps_step,
                                            (sched["steps"] - sched["decay_start"]) * tps_step))
                        log({"event": "sized", **sched})
                eta = ((sched["steps"] - step) * (now - win_t) / win_steps / 3600
                       if sched["sized"] else float("nan"))
                phase = "decay" if step > sched["decay_start"] else ("warm" if step <= WARMUP else "stable")
                print(f"  step {step:6d} {phase:6s} loss {tot:6.3f}  lr {lr:.2e}  gnorm {gn:5.2f}"
                      f"  {tps:8,.0f} tok/s  MFU {100*tps*fpt/pk:4.1f}%  {tokens_seen/1e9:6.3f}B tok"
                      f"  {el:5.2f} h  eta {eta:5.2f} h", flush=True)
                log({"step": step, "loss": tot, "lr": lr, "gnorm": gn, "tok_s": tps,
                     "tokens": tokens_seen, "hours": hours_before + el, "phase": phase})
                win_t, win_steps = now, 0
            if step % EVAL_EVERY == 0 or step == sched["steps"]:
                ev = evaluate(corpus, tok_loss_fn, dev, corpus.probs["stable"])
                per = "  ".join(f"{k}:{v:.2f}" for k, v in ev.items() if not k.startswith("_"))
                print(f"  [val] step {step}  mix {ev['_mix']:.4f}  copy {ev['_copy']:.3f}"
                      f"  | {per}", flush=True)
                log({"step": step, "val": ev})
            if GEN_EVERY and enc is not None and (step % GEN_EVERY == 0 or step == sched["steps"]):
                for p, text in samples(model, dev, enc, dec, v, step):
                    print(f"  [gen {step}] {p!r} -> {text!r}", flush=True)
            now = time.perf_counter()
            extra = {"tokens_seen": tokens_seen,
                     "train_hours": hours_before + (now - t_start) / 3600}
            if STABLE_END and not stable_saved and step >= sched["decay_start"]:
                ck.save(payload(model, opt, step, meta, sched, extra), tag="stable_end")
                stable_saved = True
                last_ckpt = now
            elif (now - last_ckpt) / 60 >= CKPT_MIN:
                ck.save(payload(model, opt, step, meta, sched, extra))
                last_ckpt, last_saved = now, step
            if (now - t_start) / 3600 >= WALL_H:
                print(f"  wall limit {WALL_H} h reached at step {step}; saving and stopping."
                      f" Rerun to resume.", flush=True)
                break
    except KeyboardInterrupt:
        step = done                                  # a half-accumulated step is dropped
        print(f"  interrupted; saving the last completed step, {step}.", flush=True)
    finally:
        pf.close()
    if poisoned:
        ck.wait()
        mlog.close()
        return
    extra = {"tokens_seen": tokens_seen,
             "train_hours": hours_before + (time.perf_counter() - t_start) / 3600}
    if step != last_saved:
        ck.save(payload(model, opt, step, meta, sched, extra))
    if step >= sched["steps"]:
        final = payload(model, None, step, meta, sched, extra)
        final.pop("opt")
        ck.save(final, tag="final")
        print(f"finished: {step:,} steps, {tokens_seen/1e9:.2f}B tokens -> {NAME}_final.pt")
    ck.wait()
    mlog.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import signal

    def _stop(signum, frame):                       # SIGTERM (a stopped cell) saves like Ctrl-C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    dev = device()
    if MODE == "bench":
        bench(dev)
    elif MODE == "lrprobe":
        lrprobe(dev)
    else:
        train(dev)


if __name__ == "__main__":
    main()
