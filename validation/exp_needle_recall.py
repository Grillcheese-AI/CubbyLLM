"""Needle-in-a-haystack: can a fixed recurrent state actually RETRIEVE?

THE QUESTION THIS DECIDES. The memory claim is arithmetic and safe: state is
n_layers x d, constant, ~128 KB at the 2B shape, against ~28 KB PER TOKEN for a
KV cache. Nothing can take that away. The *capability* claim — "ask it anything
about a long document" — is a different claim and it runs against the grain of
the architecture: a transformer's cache IS the document and can be attended to
exactly, while a recurrent state is a learned lossy summary optimised for
next-token prediction. Nothing in training rewarded keeping the date on page 312.

External evidence says this matters. Kimi K3 reaches 1M context with THREE
KDA (linear-attention) layers per ONE full-attention layer — a 3:1 hybrid, not
pure linear. cubby-lm's own 500M run used attention every 3rd layer and its
ablation recorded the largest single component gain in the table (+0.55
consistency). CubbyLLM is 0% attention, and `exp_d1b_backbone_bakeoff.py` chose
MinGRU on **bpc** — a prediction metric — against pure alternatives. A hybrid was
never on the ballot, and recall was never scored.

METHOD. Plant a verbatim fact at a controlled depth in filler drawn from the real
corpus, then score whether the model assigns higher likelihood to the true
continuation than to distractors. Scoring by likelihood rather than by generating
free text is deliberate: at this training scale free generation confabulates, and
we would be measuring the sampler. Likelihood asks the narrower question the
architecture is actually on trial for — did the information survive the state.

CONTROLS, because a number without one means nothing:
  * depth 0 with a SHORT context  -> near-ceiling, or the probe itself is broken
  * shuffled-needle control       -> the score the model gets with NO real signal
  * an untrained model            -> what chance looks like on this exact setup

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt_v21.pt \
  CB_CORPUS=/content/token_cache CUBBY_SPM=.../grillcheese_bbpe128k.json \
  python validation/exp_needle_recall.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CKPT = os.environ.get("CB_CKPT", "")
CORPUS = os.environ.get("CB_CORPUS", "")
SOURCES = os.environ.get("CB_SOURCES", "")
SPM = os.environ.get("CUBBY_SPM", "")
CTX_DIM = int(os.environ.get("CB_CTX_DIM", "32"))
LENGTHS = [int(x) for x in os.environ.get("CB_LENGTHS", "256,1024,4096").split(",")]
DEPTHS = [float(x) for x in os.environ.get("CB_DEPTHS", "0.0,0.25,0.5,0.75,1.0").split(",")]
TRIALS = int(os.environ.get("CB_TRIALS", "12"))
N_DISTRACT = int(os.environ.get("CB_DISTRACT", "7"))

# Needle/probe pairs. Deliberately simple and verbatim — we are testing whether
# a token sequence survives the state, not whether the model can reason.
FACTS = [
    ("The secret access code for the north vault is {}.", "The secret access code for the north vault is"),
    ("Dr. Halloran's laboratory notebook is stored in room {}.", "Dr. Halloran's laboratory notebook is stored in room"),
    ("The shipment from Trieste weighed exactly {} kilograms.", "The shipment from Trieste weighed exactly"),
    ("Agent Petrov's callsign during the operation was {}.", "Agent Petrov's callsign during the operation was"),
]
ANSWERS = ["47213", "3092", "8814", "60517", "27384", "91650", "5573", "10428"]


def build(meta, dev):
    from cubbyllm.core.config import CubbyConfig
    from cubbyllm.core.context import FrozenSlotRouter
    from cubbyllm.core.device import move_model
    from cubbyllm.core.generation import (BasisHyperGenerator, HyperGenerator,
                                          SnapshotHardener)
    from cubbyllm.model.assembly import CubbyModel
    from cubbyllm.model.backbone import MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    d, L, V = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    kind = meta.get("gen", "flat")
    gen = (HyperGenerator(ctx_dim=CTX_DIM, n_out=d * d) if kind == "flat" else
           BasisHyperGenerator(ctx_dim=CTX_DIM, d_model=d, n_layers=L))
    torch.manual_seed(0)
    m = CubbyModel(
        config=CubbyConfig(d_model=d, n_layers=L, ctx_dim=CTX_DIM, vocab_core=V),
        context_source=FrozenSlotRouter(input_dim=d, n_slots=8, ctx_dim=CTX_DIM).freeze(),
        backbone=MinGRUBackbone(d, L),
        memory=MemoryLayer(gen, SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(V, d),
        head=TopKRetrievalHead(torch.randn(V, d) * 0.02, learnable=True),
        retrieval_k=V)
    move_model(m, dev)
    return m, d, L, V, kind


@torch.no_grad()
def score_continuation(model, prefix_ids, cont_ids, dev):
    """Mean log-prob the model assigns to ``cont_ids`` following ``prefix_ids``.

    Uses the incremental decode path, so cost is O(1) per token and a 4k-token
    haystack costs 4k cheap steps rather than a 4k-wide forward.
    """
    state = None
    for t in range(prefix_ids.shape[0]):
        logits, state = model.step(prefix_ids[t].view(1), state)
    total = 0.0
    for t in range(cont_ids.shape[0]):
        lp = F.log_softmax(logits.float(), dim=-1)[0, int(cont_ids[t])]
        total += float(lp)
        logits, state = model.step(cont_ids[t].view(1), state)
    return total / max(1, cont_ids.shape[0])


def main():
    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT")
    ck = torch.load(CKPT, map_location="cpu")
    model, d, L, V, kind = build(ck["meta"], dev)
    trained = [p.detach().clone() for p in ck["params"]]
    with torch.no_grad():
        for p, s in zip(model.parameters(), trained):
            p.copy_(s.to(p.device))
    del ck

    from cubbyllm.training.data import _load_tokenizer
    encode, decode, eos, _ = _load_tokenizer(SPM)

    # filler from the REAL corpus — synthetic filler would make the task easier
    # than reality and the number would not transfer
    from cubbyllm.training import WeightedCorpusPipeline
    srcs = (json.load(open(SOURCES, encoding="utf-8"))["sources"] if SOURCES else
            [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
             for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))])
    pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=99, cache_only=True).prepare()

    print("needle-in-a-haystack — does the recurrent state retain retrievable detail?")
    print(f"  ckpt {CKPT}")
    print(f"  d={d} L={L} vocab={V} gen={kind} | {dev}")
    print(f"  state = {L*d*2/1024:.0f} KB fixed. A KV cache at 4k tokens would be ~112,000 KB.")
    print(f"  {TRIALS} trials/cell, 1 true answer vs {N_DISTRACT} distractors "
          f"(chance = {1/(N_DISTRACT+1):.1%})\n")

    g = torch.Generator().manual_seed(7)

    def run(tag, shuffled=False):
        print(f"  --- {tag} ---")
        print("   length |" + "".join(f"  d={int(dd*100):>3}% " for dd in DEPTHS))
        print("  --------+" + "-" * (9 * len(DEPTHS)))
        for ln in LENGTHS:
            row = []
            for dep in DEPTHS:
                hits = 0
                for tr in range(TRIALS):
                    fact, probe = FACTS[tr % len(FACTS)]
                    ans = ANSWERS[(tr * 3) % len(ANSWERS)]
                    distract = [a for a in ANSWERS if a != ans][:N_DISTRACT]
                    x, _ = next(pipe.batches(1, ln))
                    filler = x[0].tolist()
                    n_ids = encode(fact.format(ans))
                    if shuffled:                       # CONTROL: needle present but
                        n_ids = encode(fact.format(   # answering a DIFFERENT question
                            [a for a in ANSWERS if a != ans][tr % (len(ANSWERS) - 1)]))
                    pos = int(dep * max(0, len(filler) - len(n_ids)))
                    ctx = filler[:pos] + n_ids + filler[pos:]
                    ctx = ctx + encode(" " + probe)
                    pre = torch.tensor(ctx, dtype=torch.long, device=dev)
                    cands = [ans] + distract
                    scores = [score_continuation(model, pre,
                              torch.tensor(encode(" " + c), dtype=torch.long, device=dev), dev)
                              for c in cands]
                    hits += int(max(range(len(scores)), key=lambda i: scores[i]) == 0)
                row.append(hits / TRIALS)
            print(f"  {ln:>7} |" + "".join(f"  {v:>5.1%} " for v in row))
        print()

    run("TRAINED MODEL")
    run("CONTROL: shuffled needle (no true signal present)", shuffled=True)

    with torch.no_grad():                              # CONTROL: untrained = chance
        torch.manual_seed(0)
        fresh, *_ = build({"D": d, "L": L, "vocab": V, "gen": kind}, dev)
        for p, q in zip(model.parameters(), fresh.parameters()):
            p.copy_(q)
    run("CONTROL: untrained model (this is what chance looks like here)")

    print(f"  Read the TRAINED table against the two controls, not against 100%.")
    print(f"  Chance is {1/(N_DISTRACT+1):.1%}. If trained ~= controls at depth, the")
    print("  fixed state is not retaining retrievable detail and the long-document")
    print("  CAPABILITY claim does not hold — the MEMORY claim is unaffected either way.")
    print(f"\n  wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
