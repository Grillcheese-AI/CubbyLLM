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
pure linear. cubby-lm's production 1.7B run was also a hybrid: 22 MinGRU layers
with sliding-window attention on every 3rd (`trunk_torch/blocks.py:133`,
`--attn-every-n 3` in `runpod_launch.sh:104`) = 8 of 22.

RETRACTION (2026-08-03). An earlier version of this docstring claimed cubby-lm's
ablation recorded "+0.55 consistency" for that attention. THAT NUMBER DOES NOT
EXIST. A repo-wide search of cubby-lm found no such figure and NO with-attention
vs without-attention ablation at all; its own `TASKS.md:360` still queues the
2x2 as unrun, and `docs/ARCHITECTURE.md:288` labels the entry "IN VALIDATION —
not a proven result." What cubby-lm did measure points the other way: at step
110k the learned residual scale on its 8 attention layers had fallen to 0.109,
attenuated ~9x (`docs/MEMORY_PROBE.md:24,145`) — the model withdrawing from the
attention stream, though that is one telemetry reading, not an ablation either.

So: the hybrid is the inherited DESIGN, not an inherited RESULT. Nobody in
either repo has measured whether attention helps here. Meanwhile
`exp_d1b_backbone_bakeoff.py` chose MinGRU on **bpc** — a prediction metric —
and its `Trunk(d, n_layers, mixer_cls)` applies ONE mixer class to every layer,
so a hybrid was not merely unchosen, it was unrepresentable. Recall was never
scored. That is the gap this file exists to close.

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
    from cubbyllm.model.backbone import HybridBackbone, MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    d, L, V = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    kind = meta.get("gen", "flat")
    gen = (HyperGenerator(ctx_dim=CTX_DIM, n_out=d * d) if kind == "flat" else
           BasisHyperGenerator(ctx_dim=CTX_DIM, d_model=d, n_layers=L))
    # backbone must match what was TRAINED — reading it from meta is what makes the
    # A/B honest: a hybrid checkpoint reconstructed as pure MinGRU would score its
    # attention weights as noise. Defaults to mingru for pre-H-D3 checkpoints.
    if meta.get("backbone") == "hybrid":
        backbone = HybridBackbone(d, L, attn_every=int(meta.get("attn_every", 3)),
                                  window=int(meta.get("window", 512)),
                                  heads=int(meta.get("heads", 8)))
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


def _clone_bb(s):
    """One backbone layer's state. MinGRU layers carry a (1, d) tensor; hybrid
    attention layers carry a (k, v, pos) tuple — so recurse rather than assume
    every element is a tensor (which broke on the first hybrid checkpoint)."""
    if torch.is_tensor(s):
        return s.clone()
    return tuple(_clone_bb(e) for e in s)


def _clone(state):
    """Snapshot decode state — cheap next to a prefix. Handles both pure-MinGRU
    (all-tensor) and hybrid (tuple KV-cache) per-layer states."""
    return {"bb": [_clone_bb(s) for s in state["bb"]],
            "ctx_sum": state["ctx_sum"].clone(), "ctx_n": state["ctx_n"]}


@torch.no_grad()
def prefix_state(model, ids, dev, amp):
    """Consume the haystack ONCE. Returns (last_logits, state) to branch from.

    The whole reason a recurrent model can do this cheaply: state is a fixed-size
    snapshot, so N candidates cost one prefix pass plus N short continuations —
    not N prefix passes. (The first version of this file re-ran the prefix per
    candidate, an 8x waste that made a 4k row take hours.)
    """
    import contextlib
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if amp else contextlib.nullcontext())
    state, logits = None, None
    with ctx:
        for t in range(ids.shape[0]):
            logits, state = model.step(ids[t].view(1), state)
    return logits, state


@torch.no_grad()
def neutral_baselines(model, encode, dev, amp):
    """logP(candidate | probe alone) for every (probe, candidate) pair.

    THE FIX FOR A PROBE THAT WAS STRUCTURALLY PINNED TO 1/8. Until 2026-08-03
    candidates were ranked on RAW mean log-prob, which compares " 47213" against
    " 3092" on absolute likelihood — and those differ by several nats before any
    needle is planted, because one is a short common number and the other is a
    long rare one. The ranking was therefore decided by unconditional frequency,
    not by context: the same candidate won every trial, and since each answer is
    the true one exactly once over eight trials, accuracy came out to EXACTLY
    1/8 = 12.5%. That is what the whole sweep reported, in every cell, at every
    depth and length. It was not chance. It was a constant argmax.

    Subtracting the no-context score cancels the base rate, so a candidate can
    only win by being RAISED by the planted fact — standard PMI / contrastive
    scoring. 4 probes x 8 candidates = 32 short forwards, computed once.

    Caveat worth stating: the neutral context has no filler, while the real one
    has thousands of corpus tokens, so the two states differ by more than the
    needle. That shift hits all candidates within a cell about equally, so it
    cannot manufacture a constant argmax — but it does mean the PMI value is not
    a clean effect size. The properly matched comparison is TRAINED vs the
    shuffled-needle arm, which sees identical filler and now gets the identical
    correction. Read those two tables against each other, not the absolute PMI.
    """
    base = {}
    for _, probe in FACTS:
        pre = torch.tensor(encode(" " + probe), dtype=torch.long, device=dev)
        lg, st = prefix_state(model, pre, dev, amp)
        for c in ANSWERS:
            cid = torch.tensor(encode(" " + c), dtype=torch.long, device=dev)
            base[(probe, c)] = score_from(model, lg, st, cid, amp)
    return base


@torch.no_grad()
def score_from(model, logits0, state, cont_ids, amp):
    """Mean log-prob of ``cont_ids`` branching from a snapshotted state."""
    import contextlib
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if amp else contextlib.nullcontext())
    s, logits, total = _clone(state), logits0, 0.0
    with ctx:
        for t in range(cont_ids.shape[0]):
            total += float(F.log_softmax(logits.float(), dim=-1)[0, int(cont_ids[t])])
            logits, s = model.step(cont_ids[t].view(1), s)
    return total / max(1, cont_ids.shape[0])


def main():
    t0 = time.perf_counter()
    # Heartbeat through every silent setup phase, so "nothing printed yet" tells
    # you WHICH phase is slow (usually the corpus prepare) instead of leaving you
    # guessing whether it hung. Each line flushes immediately.
    def beat(msg):
        print(f"  [{time.perf_counter()-t0:5.1f}s] {msg}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beat(f"start | device {dev}")
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT")
    ck = torch.load(CKPT, map_location="cpu")
    beat(f"checkpoint loaded ({os.path.getsize(CKPT)/1e6:.0f} MB)")
    meta = dict(ck["meta"])            # keep for the untrained-control rebuild below
    model, d, L, V, kind = build(meta, dev)
    beat(f"model built (d={d} L={L} gen={kind})")
    trained = [p.detach().clone() for p in ck["params"]]
    with torch.no_grad():
        for p, s in zip(model.parameters(), trained):
            p.copy_(s.to(p.device))
    del ck
    beat("weights loaded")

    from cubbyllm.training.data import _load_tokenizer
    encode, decode, eos, _ = _load_tokenizer(SPM)
    beat("tokenizer loaded")

    # filler from the REAL corpus — synthetic filler would make the task easier
    # than reality and the number would not transfer
    from cubbyllm.training import WeightedCorpusPipeline
    srcs = (json.load(open(SOURCES, encoding="utf-8"))["sources"] if SOURCES else
            [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
             for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))])
    beat(f"preparing corpus pipeline ({len(srcs)} sources)...")
    pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=99, cache_only=True).prepare()
    beat("corpus ready — starting probes")

    print("needle-in-a-haystack — does the recurrent state retain retrievable detail?")
    print(f"  ckpt {CKPT}")
    print(f"  d={d} L={L} vocab={V} gen={kind} | {dev}")
    print(f"  state = {L*d*2/1024:.0f} KB fixed. A KV cache at 4k tokens would be ~112,000 KB.")
    print(f"  {TRIALS} trials/cell, 1 true answer vs {N_DISTRACT} distractors "
          f"(chance = {1/(N_DISTRACT+1):.1%})\n")

    g = torch.Generator().manual_seed(7)

    amp = dev.type == "cuda"
    base = neutral_baselines(model, encode, dev, amp)   # base rates, cancelled below
    batches = {ln: pipe.batches(1, ln) for ln in LENGTHS}   # one generator per length
    beat("baselines done — first table below (each cell = TRIALS prefills)")

    def run(tag, shuffled=False):
        print(f"  --- {tag} ---", flush=True)
        print("   length |" + "".join(f"  d={int(dd*100):>3}% " for dd in DEPTHS), flush=True)
        print("  --------+" + "-" * (9 * len(DEPTHS)), flush=True)
        winners = {}                    # which candidate STRING won, over all cells
        for ln in LENGTHS:
            row, t_row = [], time.perf_counter()
            for dep in DEPTHS:
                hits = 0
                for tr in range(TRIALS):
                    fact, probe = FACTS[tr % len(FACTS)]
                    ans = ANSWERS[(tr * 3) % len(ANSWERS)]
                    distract = [a for a in ANSWERS if a != ans][:N_DISTRACT]
                    filler = next(batches[ln])[0][0].tolist()
                    shown = (ans if not shuffled else
                             [a for a in ANSWERS if a != ans][tr % (len(ANSWERS) - 1)])
                    n_ids = encode(fact.format(shown))
                    pos = int(dep * max(0, len(filler) - len(n_ids)))
                    ctx = filler[:pos] + n_ids + filler[pos:] + encode(" " + probe)
                    pre = torch.tensor(ctx, dtype=torch.long, device=dev)
                    lg0, st0 = prefix_state(model, pre, dev, amp)     # haystack ONCE
                    cands = [ans] + distract
                    # PMI, not raw likelihood: subtract each candidate's score with
                    # no context at all, so a rare long number is not beaten by a
                    # common short one before the needle is even read. See
                    # neutral_baselines() for what this was hiding.
                    scores = [score_from(model, lg0, st0,
                              torch.tensor(encode(" " + c), dtype=torch.long, device=dev), amp)
                              - base[(probe, c)] for c in cands]      # branch per candidate
                    win = max(range(len(scores)), key=lambda i: scores[i])
                    winners[cands[win]] = winners.get(cands[win], 0) + 1
                    hits += int(win == 0)
                row.append(hits / TRIALS)
                print(f"      [len {ln} depth {dep:.2f}] {row[-1]:.1%}", flush=True)
            print(f"  {ln:>7} |" + "".join(f"  {v:>5.1%} " for v in row)
                  + f"   ({time.perf_counter()-t_row:.0f}s)", flush=True)
        # THE CHECK THAT WOULD HAVE CAUGHT THE ORIGINAL BUG. A probe whose winner
        # never changes with context is broken no matter what accuracy it prints
        # — and a constant winner yields exactly 1/n_candidates by construction,
        # which is indistinguishable from chance if you only look at the number.
        top, n = max(winners.items(), key=lambda kv: kv[1]), sum(winners.values())
        print(f"  argmax spread: {len(winners)} distinct winners over {n} trials; "
              f"most frequent {top[0]!r} took {top[1]}/{n} ({top[1]/n:.0%})"
              + ("   <<< CONSTANT ARGMAX — probe is not reading context"
                 if len(winners) == 1 else ""), flush=True)
        print(flush=True)

    run("TRAINED MODEL")
    run("CONTROL: shuffled needle (no true signal present)", shuffled=True)

    with torch.no_grad():                              # CONTROL: untrained = chance
        torch.manual_seed(0)
        # SAME meta (incl. backbone) — a partial dict silently built a MinGRU
        # against a hybrid model and the params mismatched (1536 qkv vs 512).
        fresh, *_ = build(meta, dev)
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
