"""H-B6 follow-up: the ckpt_1 A/B that was never captured numerically.

Two questions the training curves structurally could not answer, because both
are about metrics the loss never contained:

  1. Did the binding aux loss cost LANGUAGE MODELLING? Throughout the run bpc
     appeared to track baseline, which is what "binding is free" rested on — but
     that was eyeballed against a remembered curve, never a matched-step number.
  2. Does the no-binding checkpoint retain the RETRIEVAL that the binding-trained
     one lost (11.5% vs an untrained control's 97.5%)? If ckpt_1 sits near the
     untrained control, that pins the loss as the cause beyond argument.

Both models are scored on IDENTICAL held-out batches, same seed, same order.

CONFOUND THIS SCRIPT REFUSES TO HIDE: ckpt_1 was saved when the run was
restarted, so it is probably from a DIFFERENT step than the final checkpoint. A
bpc gap between checkpoints trained for different numbers of steps says nothing
about the binding loss. The script reads the step counter out of each checkpoint
and, if they differ materially, reports the comparison as CONFOUNDED rather than
letting a step-count artefact be read as a binding effect.

"Held-out" is approximate and stated as such: the pipeline samples random
windows, so a distinct seed draws windows the run almost certainly never saw
(~123M tokens consumed out of ~18B, i.e. <1%), but it is not a reserved split.

  CB_CKPT_A=/content/drive/MyDrive/cubbyllm/ckpt.pt \
  CB_CKPT_B=/content/drive/MyDrive/cubbyllm/ckpt_1.pt \
  CB_CORPUS=/content/token_cache CB_SOURCES=.../corpus_sources.json \
  python validation/exp_b6b_ckpt_ab.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.core.config import CubbyConfig  # noqa: E402
from cubbyllm.core.context import FrozenSlotRouter  # noqa: E402
from cubbyllm.core.device import move_model  # noqa: E402
from cubbyllm.core.generation import HyperGenerator, SnapshotHardener  # noqa: E402
from cubbyllm.model.assembly import CubbyModel  # noqa: E402
from cubbyllm.model.backbone import MinGRUBackbone  # noqa: E402
from cubbyllm.model.binding import BindingHead  # noqa: E402
from cubbyllm.model.binding.torch_ops import make_roles  # noqa: E402
from cubbyllm.model.memory import MemoryLayer  # noqa: E402
from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead  # noqa: E402


def _env(k, d, cast=int):
    return cast(os.environ.get(k, d))


CKPT_A = os.environ.get("CB_CKPT_A", "")      # binding-trained (bind_weight 0.42)
CKPT_B = os.environ.get("CB_CKPT_B", "")      # no-binding control (ckpt_1)
CORPUS = os.environ.get("CB_CORPUS", "")
SOURCES = os.environ.get("CB_SOURCES", "")
SPM = os.environ.get("CUBBY_SPM", "")
CTX = _env("CB_CTX", 32); N_SLOTS = _env("CB_SLOTS", 8)
BATCH = _env("CB_B", 4); SEQ = _env("CB_S", 512)
N_BATCH = _env("CB_NBATCH", 24)               # eval batches (identical for both)
EVAL_SEED = _env("CB_EVAL_SEED", 12345)       # NOT the training seed (0)
CPT = _env("CB_CPT", 4.8, float)              # chars/token, matches train_colab
N_BIND = _env("CB_BIND_N", 16)


def build(vocab, d, layers, dev):
    torch.manual_seed(0)
    cfg = CubbyConfig(d_model=d, n_layers=layers, ctx_dim=CTX, vocab_core=vocab)
    m = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=d, n_slots=N_SLOTS, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(d, layers, grad_checkpoint=False),
        memory=MemoryLayer(HyperGenerator(ctx_dim=CTX, n_out=d * d), SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(vocab, d),
        head=TopKRetrievalHead(torch.randn(vocab, d) * 0.02, learnable=True),
        retrieval_k=vocab)
    move_model(m, dev)
    return m


@torch.no_grad()
def evaluate(path, batches, dev, meta):
    """Held-out CE + the binding retrieval diagnostic, on one checkpoint."""
    d, layers, vocab = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    model = build(vocab, d, layers, dev)
    ck = torch.load(path, map_location=dev)
    ps = list(model.parameters())
    with torch.no_grad():
        for p, s in zip(ps, ck["params"]):
            p.copy_(s.to(p.device))

    tot, ntok, hs = 0.0, 0, []
    for x, y in batches:
        x, y = x.to(dev), y.to(dev)
        h = model.features(x)
        logits = model.logits_from(h)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             y.reshape(-1), reduction="sum")
        tot += float(ce); ntok += y.numel()
        if len(hs) < 8:
            hs.append(h.float().cpu())
    ce = tot / ntok

    # retrieval: the metric the training loss never had
    h = torch.cat(hs, dim=0)
    n_seq, S, _ = h.shape
    roles = make_roles(N_BIND, d, h.device)
    g = torch.Generator().manual_seed(EVAL_SEED)
    idx = torch.randint(0, S, (n_seq, N_BIND), generator=g)
    f = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d))
    r = roles.unsqueeze(0)
    rec = (r * f).sum(dim=1, keepdim=True) * r
    rn, fn = F.normalize(rec, dim=-1), F.normalize(f, dim=-1)
    cos = float((rn * fn).sum(-1).mean())
    sim = torch.bmm(rn, fn.transpose(1, 2))
    acc = float((sim.argmax(-1) == torch.arange(N_BIND)).float().mean())
    ff = float(torch.bmm(fn, fn.transpose(1, 2))[:, ~torch.eye(N_BIND, dtype=bool)].mean())

    step = int(ck.get("step", -1))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return dict(ce=ce, ppl=math.exp(min(ce, 80)), bpc=ce / math.log(2) / CPT,
                cos=cos, acc=acc, ff=ff, step=step)


def main():
    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for nm, p in (("CB_CKPT_A", CKPT_A), ("CB_CKPT_B", CKPT_B)):
        if not p or not os.path.exists(p):
            raise SystemExit(f"{nm} not found: {p!r}")

    mA = torch.load(CKPT_A, map_location="cpu")["meta"]
    mB = torch.load(CKPT_B, map_location="cpu")["meta"]
    if mA != mB:
        raise SystemExit(f"checkpoints have different shapes: {mA} vs {mB} — not comparable")

    print("H-B6 follow-up — binding-trained vs no-binding (ckpt_1), held-out")
    print(f"  A (binding) {CKPT_A}")
    print(f"  B (control) {CKPT_B}")
    print(f"  model {mA} | device {dev}")
    print(f"  eval  {BATCH}x{SEQ} x{N_BATCH} batches, seed {EVAL_SEED} (training used 0)")

    from cubbyllm.training import WeightedCorpusPipeline
    if SOURCES:
        srcs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
    else:
        srcs = [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))]
    pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=EVAL_SEED, cache_only=True).prepare()
    it = pipe.batches(BATCH, SEQ)
    batches = [next(it) for _ in range(N_BATCH)]           # SAME batches for both
    print(f"  {N_BATCH * BATCH * SEQ:,} held-out tokens\n")

    print("  scoring A (binding-trained) ...", flush=True)
    A = evaluate(CKPT_A, batches, dev, mA)
    print("  scoring B (no-binding control) ...", flush=True)
    B = evaluate(CKPT_B, batches, dev, mB)

    print(f"\n  {'':<22} {'A binding':>12} {'B control':>12}   delta")
    print(f"  {'-'*22} {'-'*12} {'-'*12}   -----")
    print(f"  {'checkpoint step':<22} {A['step']:>12} {B['step']:>12}")
    print(f"  {'held-out CE':<22} {A['ce']:>12.4f} {B['ce']:>12.4f}   {A['ce']-B['ce']:+.4f}")
    print(f"  {'held-out ppl':<22} {A['ppl']:>12.2f} {B['ppl']:>12.2f}")
    print(f"  {'held-out bpc':<22} {A['bpc']:>12.4f} {B['bpc']:>12.4f}   {A['bpc']-B['bpc']:+.4f}")
    print(f"  {'binding cos (n=%d)' % N_BIND:<22} {A['cos']:>12.3f} {B['cos']:>12.3f}")
    print(f"  {'RETRIEVAL acc':<22} {A['acc']:>11.1%} {B['acc']:>11.1%}   {A['acc']-B['acc']:+.1%}")
    print(f"  {'filler-filler cos':<22} {A['ff']:>12.3f} {B['ff']:>12.3f}"
          f"   (high = collapsed)")

    same_step = A["step"] == B["step"]
    print()
    if not same_step:
        print(f"  !! CONFOUNDED on language modelling: the checkpoints are from"
              f" DIFFERENT steps ({A['step']} vs {B['step']}). The bpc delta above"
              f" mixes the binding effect with {abs(A['step']-B['step'])} steps of"
              f" extra training and CANNOT be attributed to the binding loss.")
        print(f"     The RETRIEVAL comparison is still informative — collapse is a"
              f" representational property, not a function of training length"
              f" (an UNTRAINED trunk already scores ~97%).")
    else:
        d_bpc = A["bpc"] - B["bpc"]
        verdict = ("binding was FREE on LM" if abs(d_bpc) < 0.01 else
                   "binding COST language modelling" if d_bpc > 0 else
                   "binding HELPED language modelling")
        print(f"  matched steps -> clean A/B. bpc delta {d_bpc:+.4f}: {verdict}.")
    print(f"\n  Retrieval reading: chance is {1/N_BIND:.1%}; an untrained trunk"
          f" scores ~97%. If B is high and A is near chance, the aux loss is the"
          f" cause and H-B6's falsification is confirmed on a trained control.")
    print(f"\n  wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
