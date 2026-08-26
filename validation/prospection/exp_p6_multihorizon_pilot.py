"""H-P6 — the multi-horizon prediction head as a pilot arm (Colab, GPU).

THE RECOMMENDATION THIS IMPLEMENTS. The Group P checkpoint arms showed the
trained hybrid's recurrence is a 1-2-step mixer (H-P3): there is no slow
subspace in the substrate to hang a multi-timescale scheme on, so the three
horizons the "5th dimension" thread wants map onto the mechanisms that exist —
recurrence for short, the attention window for medium, the episodic store for
beyond-window. This script is the medium-horizon piece: an auxiliary head that
predicts, from the trunk feature h_t, the discounted average of the FUTURE
tokens' embeddings at several time constants (successor-feature style, Dayan
1993 / Barreto et al. 2017), trained alongside next-token CE. Two matched arms:

    baseline      CB_MH_W=0      plain next-token training
    multihorizon  CB_MH_W=0.1    + the multi-horizon head

WHAT IT MEASURES, and the kill criteria:
  * bpc / held-out CE of both arms at matched tokens — the head must be
    (near-)free. Kill: the aux arm is worse by more than run-to-run noise.
  * per-horizon "skill": cosine(prediction, target) minus the trivial predictor
    (the mean target direction). Kill: no skill beyond the next-token horizon —
    then the trunk cannot express anything about the medium-term future and
    the multi-γ design is dead at this scale.
  * a LINEAR PROBE fitted post hoc on frozen features of BOTH arms, same
    horizons — how much future information the trunk carries anyway vs. how
    much the explicit objective adds. This is the number that says whether the
    objective changes the representation or only decorates it.
  * the H-P3 gate spectrum of both arms — does an explicit long-horizon
    objective lengthen any recurrent time constant? (H-P3 named this as one
    of the three ways long time constants could be *made*.)
  * the representation-health and copy-floor gates from train_colab, so a
    Goodharting aux loss (the H-B6 failure) is caught the way it was last time.

Reuses ``validation/train_colab.py`` wholesale (build, staging, checkpoint
layout, health probes) so a checkpoint from here loads in every other script.
Checkpoints carry train_colab's exact ``meta``; the head is saved separately
as ``<ckpt>.mhhead.pt`` and the results as ``CB_MH_OUT`` (JSON).

Data: CB_CORPUS with ``<name>.u32`` shards (staged from CB_CACHE_MIRROR if
present) — or, when neither exists, CB_JSONL_DIR (the domain-tagged Wikipedia
jsonl next to the checkpoints) is tokenized ONCE into CB_CORPUS and mirrored
to CB_CACHE_MIRROR for the next session. ~970 MB of text -> ~200M tokens.

Env (on top of train_colab's): CB_MH_W (aux weight; 0 = baseline arm)
CB_MH_GAMMAS (default 0,0.875,0.98,0.998 — mean horizons 1, 8, 50, 500 tokens)
CB_MH_TAG CB_MH_OUT CB_MH_PROBE_STEPS CB_MH_EVAL_BATCHES CB_JSONL_DIR
CB_CACHE_MIRROR CB_MH_CAP (per-source token cap when tokenizing — smoke runs)
CB_COMPILE (torch.compile the backbone; off by default).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(VALIDATION_DIR)
for p in (ROOT, VALIDATION_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_colab as tc  # noqa: E402  (parses the CB_* env at import)
from cubbyllm.core.generation import SnapshotHardener  # noqa: E402
from cubbyllm.model.backbone.mingru import _log_domain_scan  # noqa: E402
from cubbyllm.training import TrainLoop, WeightedCorpusPipeline  # noqa: E402

MH_W = float(os.environ.get("CB_MH_W", "0"))
GAMMAS = [float(g) for g in os.environ.get("CB_MH_GAMMAS", "0,0.875,0.98,0.998").split(",")]
TAG = os.environ.get("CB_MH_TAG", "multihorizon" if MH_W > 0 else "baseline")
OUT = os.environ.get("CB_MH_OUT", "")
PROBE_STEPS = int(os.environ.get("CB_MH_PROBE_STEPS", "300"))
EVAL_BATCHES = int(os.environ.get("CB_MH_EVAL_BATCHES", "16"))
JSONL_DIR = os.environ.get("CB_JSONL_DIR", "")
MIRROR = os.environ.get("CB_CACHE_MIRROR", "")
CAP = int(os.environ.get("CB_MH_CAP", "0")) or None
COMPILE = os.environ.get("CB_COMPILE", "0") == "1"


# ── the head ─────────────────────────────────────────────────────────────────

class MultiHorizonHead(nn.Module):
    """One linear map per horizon: h_t -> predicted discounted future feature.

    Target for horizon γ at position t (computed inside the training window):
        y_t = Σ_{k>=1} γ^{k-1} φ(x_{t+k})  /  Σ_{k=1}^{n_future(t)} γ^{k-1}
    with φ = the model's own core token embedding, unit-normalised and
    DETACHED (the objective shapes the trunk, it can never move the target —
    the H-B6 lesson: an aux loss that can reach its own target Goodharts).
    γ=0 is the next-token embedding; 0.875 / 0.98 / 0.998 average over ~8 /
    ~50 / ~500 future tokens. Loss = 1 - cos per horizon, summed.
    """

    def __init__(self, d: int, gammas: list[float]):
        super().__init__()
        self.gammas = list(gammas)
        self.proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in gammas])

    @staticmethod
    @torch.no_grad()
    def targets(phi: torch.Tensor, gamma: float) -> torch.Tensor:
        """phi (B, S, d) unit features of x_0..x_{S-1} -> (B, S-1, d) targets
        for positions 0..S-2 (position S-1 has no future). Uses the package's
        own log-domain scan on the time-reversed sequence, so the O(S) recursion
        z_t = φ_{t+1} + γ z_{t+1} costs one parallel scan, not a Python loop."""
        phi = phi.float()
        B, S, d = phi.shape
        if gamma <= 0.0:
            return phi[:, 1:]
        rev = phi.flip(1)                                    # rev[j] = φ_{S-1-j}
        z = _log_domain_scan(rev, torch.full_like(rev, gamma)).flip(1)   # z_t = Σ_{s>=t} γ^{s-t} φ_s
        y = z[:, 1:]                                         # Σ_{k>=1} γ^{k-1} φ_{t+k}
        n_future = torch.arange(S - 1, 0, -1, device=phi.device, dtype=torch.float32)
        mass = (1.0 - gamma ** n_future) / (1.0 - gamma)     # truncated geometric mass
        return y / mass.view(1, -1, 1)

    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        return [p(h[:, :-1]) for p in self.proj]             # predictions for positions 0..S-2

    def losses(self, h: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        preds = self.forward(h)
        total, per = 0.0, []
        for pred, g in zip(preds, self.gammas):
            y = self.targets(phi, g)
            cos = F.cosine_similarity(pred.float(), y, dim=-1)
            loss = (1.0 - cos).mean()
            total = total + loss
            per.append(float(cos.mean().detach()))
        return total, per


def unit_features(model, x: torch.Tensor) -> torch.Tensor:
    """φ(x): the model's core token embedding, CENTERED on the batch mean and
    unit-normalised, detached. Centering matters for reading the numbers: the
    raw embedding table shares a large common direction, so an uncentered
    "predict the future" target is ~50% solved by always emitting that
    direction. Centered, the trivial mean-direction predictor scores ~0 and
    the reported cosine is the horizon information itself."""
    with torch.no_grad():
        e = model.embedding.embed(x, ctx=None).float()
        e = e - e.mean(dim=(0, 1), keepdim=True)
        return F.normalize(e, dim=-1)


# ── the loop: train_colab's TrainLoop + the aux term ────────────────────────

class MultiHorizonLoop(TrainLoop):
    def __init__(self, model, data, hardener, head, weight, **kw):
        super().__init__(model, data, hardener, **kw)
        self.head, self.weight = head, float(weight)
        self.last_aux: list[float] = []
        if head is not None:
            self.opt.add_param_group({"params": list(head.parameters())})

    def _all_params(self):
        yield from self.model.parameters()
        if self.head is not None:
            yield from self.head.parameters()

    def step(self) -> float:
        import contextlib
        x, y = next(self._batches)
        if self.device is not None:
            x, y = x.to(self.device), y.to(self.device)
        use_amp = self.amp and self.device is not None and self.device.type == "cuda"
        actx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext())
        with actx:
            h = self.model.features(x)
            logits = self.model.logits_from(h)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            gen = self._generator()
            if gen is not None:
                pen = self.hardener.penalty(gen)
                if torch.is_tensor(pen):
                    loss = loss + pen
        if self.head is not None and self.weight > 0:
            aux, self.last_aux = self.head.losses(h.float(), unit_features(self.model, x))
            loss = loss + self.weight * aux
        self._nstep += 1
        lr = self._lr_at(self._nstep)
        if lr is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self._all_params()), self.grad_clip)
        self.opt.step()
        return float(loss.detach())


# ── data: cache -> mirror -> tokenize-from-jsonl ─────────────────────────────

def ensure_corpus(spm: str) -> tuple[str, list[dict]]:
    """Returns (cache_dir, sources) with <name>.u32 shards present locally.

    Order: CB_CORPUS already has shards -> use them (staging from CB_STAGE is
    train_colab's job and happens before this). Else CB_CACHE_MIRROR (Drive)
    has shards -> copy them local. Else CB_JSONL_DIR -> tokenize every jsonl
    there into CB_CORPUS (one shard per file, EOS between documents) and mirror
    the shards to CB_CACHE_MIRROR so the next session skips this step."""
    cache = tc.STAGE or tc.CORPUS
    if not cache:
        raise SystemExit("set CB_CORPUS (and CB_STAGE on Colab) — where token shards live")
    os.makedirs(cache, exist_ok=True)
    if glob.glob(os.path.join(cache, "*.u32")):
        if tc.SOURCES:
            return cache, json.load(open(tc.SOURCES, encoding="utf-8"))["sources"]
        return cache, [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                       for p in sorted(glob.glob(os.path.join(cache, "*.u32")))]
    if MIRROR and glob.glob(os.path.join(MIRROR, "*.u32")):
        print(f"staging token shards {MIRROR} -> {cache}", flush=True)
        tc.stage_cache(MIRROR, cache)
        return ensure_corpus(spm)
    if not JSONL_DIR:
        raise SystemExit(f"no token shards in {cache}, no CB_CACHE_MIRROR, no CB_JSONL_DIR")
    files = sorted(glob.glob(os.path.join(JSONL_DIR, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl under {JSONL_DIR}")
    sources = [{"name": os.path.splitext(os.path.basename(f))[0], "paths": [f],
                "format": "jsonl", "text_key": "text", "weight": 1.0} for f in files]
    print(f"tokenizing {len(files)} jsonl files from {JSONL_DIR} -> {cache} "
          f"({'cap ' + str(CAP) + ' tok/source' if CAP else 'full'}) ...", flush=True)
    t0 = time.perf_counter()
    pipe = WeightedCorpusPipeline(sources, spm, cache, seed=0, max_tokens_per_source=CAP).prepare()
    counts = pipe.source_token_counts()
    print(f"  {sum(counts.values()):,} tokens in {time.perf_counter()-t0:.0f}s", flush=True)
    if MIRROR:
        import shutil
        os.makedirs(MIRROR, exist_ok=True)
        for p in glob.glob(os.path.join(cache, "*.u32")):
            dst = os.path.join(MIRROR, os.path.basename(p))
            if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(p)):
                shutil.copy2(p, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        print(f"  mirrored shards to {MIRROR}", flush=True)
    return cache, [{"name": s["name"], "weight": 1.0} for s in sources]


# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def held_out(model, head, pipe_eval, dev, gammas, n_batches):
    """Held-out CE plus per-horizon cosine of (head prediction | mean-target
    baseline) vs the target, over fresh windows from a differently-seeded stream."""
    ce, cos_head, cos_mean = [], [[] for _ in gammas], [[] for _ in gammas]
    for _ in range(n_batches):
        x, y = next(pipe_eval)
        x, y = x.to(dev), y.to(dev)
        with tc._autocast(dev):
            h = model.features(x)
            logits = model.logits_from(h)
            ce.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
        phi = unit_features(model, x)
        preds = head.forward(h.float()) if head is not None else None
        for i, g in enumerate(gammas):
            t = MultiHorizonHead.targets(phi, g)
            mean_dir = F.normalize(t.mean(dim=(0, 1)), dim=-1)
            cos_mean[i].append(float(F.cosine_similarity(t, mean_dir.expand_as(t), dim=-1).mean()))
            if preds is not None:
                cos_head[i].append(float(F.cosine_similarity(preds[i].float(), t, dim=-1).mean()))
    return dict(ce=float(np.mean(ce)),
                cos_mean=[float(np.mean(c)) for c in cos_mean],
                cos_head=[float(np.mean(c)) if c else None for c in cos_head])


def linear_probe(model, pipe_train, pipe_eval, dev, gammas, steps, n_eval):
    """Fit a fresh per-horizon linear map on FROZEN trunk features, then score
    it on held-out windows. Measures the future information the trunk carries
    regardless of whether it was trained to expose it."""
    d = model.config.d_model
    probe = MultiHorizonHead(d, gammas).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(steps):
        x, _ = next(pipe_train)
        x = x.to(dev)
        with torch.no_grad(), tc._autocast(dev):
            h = model.features(x).float()
        loss, _ = probe.losses(h, unit_features(model, x))
        opt.zero_grad(); loss.backward(); opt.step()
    r = held_out(model, probe, pipe_eval, dev, gammas, n_eval)
    return r["cos_head"]


def gate_spectrum(model, pipe_eval, dev):
    """H-P3 on this arm: per recurrent layer tau p50 and slow fractions."""
    import _common as C
    x, _ = next(pipe_eval)
    prof = C.retention_profile(model, x[:2].to(dev))
    if not prof:
        return {}
    taus = C.unit_time_constants({i: a.float().cpu() for i, a in prof.items()})
    return {str(i): dict(p50=C.pct(t, .5), p90=C.pct(t, .9),
                         slow8=float((t >= 8).float().mean()),
                         slow100=float((t >= 100).float().mean())) for i, t in taus.items()}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from cubbyllm.core.device import resolve_device
    from cubbyllm.training.data import _load_tokenizer

    dev = resolve_device(os.environ.get("CB_DEVICE", "auto"))
    tc.check_gpu(dev)
    _, _, _, vocab = _load_tokenizer(tc.SPM)
    if tc.CORPUS and tc.STAGE and os.path.abspath(tc.STAGE) != os.path.abspath(tc.CORPUS) \
            and glob.glob(os.path.join(tc.CORPUS, "*.u32")):
        print(f"staging cache {tc.CORPUS} -> {tc.STAGE}", flush=True)
        tc.stage_cache(tc.CORPUS, tc.STAGE)
    cache, sources = ensure_corpus(tc.SPM)
    pipe = WeightedCorpusPipeline(sources, tc.SPM, cache, seed=0, cache_only=True).prepare()
    pipe_eval = WeightedCorpusPipeline(sources, tc.SPM, cache, seed=777, cache_only=True) \
        .prepare().batches(min(tc.BATCH, 16), tc.SEQ)
    counts = pipe.source_token_counts()
    print(f"corpus: {cache} | {len(counts)} sources | {sum(counts.values()):,} tokens | "
          f"manifest {pipe.manifest_hash()[:16]}")

    model = tc.build(vocab, dev)
    if COMPILE and dev.type == "cuda":
        model.backbone = torch.compile(model.backbone)
    head = MultiHorizonHead(tc.D, GAMMAS).to(dev) if MH_W > 0 else None
    loop = MultiHorizonLoop(model, pipe, SnapshotHardener(), head, MH_W,
                            lr=tc.LR, batch_size=tc.BATCH, seq_len=tc.SEQ, device=dev,
                            amp=tc.AMP, grad_clip=tc.CLIP, warmup=tc.WARMUP,
                            total_steps=tc.STEPS, min_lr_ratio=tc.MIN_LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arm: {TAG} | aux weight {MH_W} | gammas {GAMMAS} (mean horizons "
          f"{[round(1/(1-g)) if g < 1 else 'inf' for g in GAMMAS]} tokens)")
    print(f"config: d={tc.D} L={tc.N_LAYERS} backbone={tc.BACKBONE} B={tc.BATCH} S={tc.SEQ} "
          f"steps={tc.STEPS} lr={tc.LR:.1e} warmup={tc.WARMUP} | params {n_params:,} "
          f"(+ head {sum(p.numel() for p in head.parameters()) if head else 0:,})\n", flush=True)

    meta = {"D": tc.D, "L": tc.N_LAYERS, "vocab": vocab, "gen": tc.GEN_KIND,
            "backbone": tc.BACKBONE, "attn_every": tc.ATTN_EVERY,
            "window": tc.ATTN_WINDOW, "heads": tc.ATTN_HEADS,
            "mem_every": tc.MEM_EVERY, "mem_topk": tc.MEM_TOPK, "mem_key": tc.MEM_KEY}
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); recent, curve, first = [], [], None
    for s in range(1, tc.STEPS + 1):
        loss = loop.step(); recent.append(loss)
        if not math.isfinite(loss):
            print(f"  step {s}: non-finite loss — stopping", flush=True); break
        if first is None:
            first = loss
        if s > 3 and loss > 1.8 * first:
            print(f"  step {s}: loss {loss:.2f} diverging from init {first:.2f} — stopping", flush=True); break
        if s % tc.EVAL_EVERY == 0 or s == 1:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tr = float(np.mean(recent[-tc.EVAL_EVERY:]))
            aux = (" aux " + " ".join(f"{c:.3f}" for c in loop.last_aux)) if loop.last_aux else ""
            health = ""
            if tc.HEALTH_EVERY and s % tc.HEALTH_EVERY == 0:
                acc, ff = tc.representation_health(model, pipe, dev)
                cp_r, cp_f = tc.copy_floor(model, dev, vocab, pipe)
                health = (f"  ret {acc:5.1%} ff {ff:+.3f} copy r{cp_r:3.0%}/f{cp_f:3.0%}"
                          + ("  <<< COLLAPSING" if acc < 0.75 else ""))
            el = time.perf_counter() - t0
            print(f"  step {s:>5}  train {tr:6.3f}  lr {loop.opt.param_groups[0]['lr']:.2e}{aux}"
                  f"{health}  {s*tc.BATCH*tc.SEQ/el:>9,.0f} tok/s", flush=True)
            curve.append((s, tr))
        if tc.CKPT and tc.CKPT_EVERY and s % tc.CKPT_EVERY == 0:
            tc.save_ckpt(tc.CKPT, model, loop.opt, s, meta)
    wall = time.perf_counter() - t0
    steps_done = len(recent)
    if tc.CKPT:
        tc.save_ckpt(tc.CKPT, model, loop.opt, steps_done, meta)
        if head is not None:
            torch.save({"gammas": GAMMAS, "state": head.state_dict()}, tc.CKPT + ".mhhead.pt")
        print(f"checkpoint -> {tc.CKPT}")

    print("\nheld-out evaluation ...", flush=True)
    ev = held_out(model, head, pipe_eval, dev, GAMMAS, EVAL_BATCHES)
    probe = linear_probe(model, pipe.batches(min(tc.BATCH, 16), tc.SEQ), pipe_eval, dev,
                         GAMMAS, PROBE_STEPS, EVAL_BATCHES) if PROBE_STEPS > 0 else None
    spec = gate_spectrum(model, pipe_eval, dev)
    acc, ff = tc.representation_health(model, pipe, dev)
    cp_r, cp_f = tc.copy_floor(model, dev, vocab, pipe)
    res = dict(arm=TAG, aux_weight=MH_W, gammas=GAMMAS, steps=steps_done,
               tokens=steps_done * tc.BATCH * tc.SEQ, wall_s=wall,
               config=dict(D=tc.D, L=tc.N_LAYERS, backbone=tc.BACKBONE, B=tc.BATCH, S=tc.SEQ,
                           lr=tc.LR, warmup=tc.WARMUP),
               train_loss_last=float(np.mean(recent[-max(1, tc.EVAL_EVERY):])),
               heldout_ce=ev["ce"], heldout_bits_per_token=ev["ce"] / math.log(2),
               cos_mean_baseline=ev["cos_mean"], cos_head=ev["cos_head"], cos_probe=probe,
               skill_head=[None if c is None else c - m for c, m in zip(ev["cos_head"], ev["cos_mean"])],
               skill_probe=None if probe is None else [c - m for c, m in zip(probe, ev["cos_mean"])],
               gate_spectrum=spec, retrieval=acc, ff_cos=ff, copy_rand=cp_r, copy_freq=cp_f,
               curve=curve, manifest=pipe.manifest_hash(),
               peak_vram_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None))
    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"results -> {OUT}")

    hz = [round(1 / (1 - g)) if g < 1 else "inf" for g in GAMMAS]
    print(f"\n[{TAG}] {steps_done} steps, {res['tokens']:,} tokens, {wall/60:.1f} min"
          f"{', peak VRAM %.1f GB' % res['peak_vram_gb'] if res['peak_vram_gb'] else ''}")
    print(f"  held-out CE {ev['ce']:.4f} nats ({res['heldout_bits_per_token']:.3f} bits/token)"
          f" | train {res['train_loss_last']:.3f}")
    print(f"  health: retrieval {acc:.1%} ff {ff:+.3f} | copy r{cp_r:.0%}/f{cp_f:.0%}")
    print("  horizon (mean tokens):  " + "".join(f"{str(h):>9}" for h in hz))
    print("  cos trivial (mean dir): " + "".join(f"{c:9.3f}" for c in ev["cos_mean"]))
    if head is not None:
        print("  cos head:               " + "".join(f"{c:9.3f}" for c in ev["cos_head"]))
        print("  skill head:             " + "".join(f"{c:+9.3f}" for c in res["skill_head"]))
    if probe is not None:
        print("  cos probe (frozen h):   " + "".join(f"{c:9.3f}" for c in probe))
        print("  skill probe:            " + "".join(f"{c:+9.3f}" for c in res["skill_probe"]))
    if spec:
        print("  gate spectrum (recurrent layers): " + ", ".join(
            f"L{i}: tau p50 {v['p50']:.1f} p90 {v['p90']:.1f} slow8 {v['slow8']:.2f}"
            for i, v in spec.items()))


if __name__ == "__main__":
    main()
