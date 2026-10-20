"""exp_k10_hypertoken -- a fact shown ONCE, as a hypertoken, read back beyond the attention
window by the trunk's own algebraic read (H-K10, gate 1).

The channel is H-S12's, unchanged: a placed token's embedding is bound with the unit vector of
its place, the scene is the causal sum of the bound placed tokens, and mid-trunk a gated unitary
query reads the scene (`cubbyllm/model/spatial.py`). Here a place is an *address*: an entity's
cell is a hash into a 1,024^3 lattice, a relation is a lattice step the kernel separates (6 cells),
and a fact "s r o" puts the object's token at cell(s) + step(r). A document is 8 such facts, a
stretch of filler language (a fixed Markov chain over 496 words, so there is a language to learn
beside the facts), then 8 questions "Q s r HERE o" with HERE placed at the same address -- past
the attention window of 64, with facts that are random per document and never repeat across
documents, so the answer can only come from the scene. The control is the same documents with
nothing placed: the plain path, bit for bit, where the answer is beyond attention's reach.

Gate 1 (pre-registered): held-out questions >= 90% through the channel where the control sits at
chance beyond the window, and the control >= 80% when the questions are inside the window
(``--facts-per-doc 4 --filler 0``: attention can do it only within reach). Gate 2: the filler
language's held-out loss within 0.02 of the control's. Kill: the channel below 60%.

First read (2026-10-20, D = 256, 1,000 steps x 16 documents, ~25 min per arm on the CPU): before
training the scene decodes the object at its address at 0.84 by nearest entity (cos 0.17-0.35).
``--query placed --gate-bias 0`` (the hypertoken carries its address; gate and decode learned):
held-out answers 0.00 / 0.21 / 0.53 / 0.81 / 0.86 / 0.92 / 0.94 at steps 0 / 200 / 300 / 400 /
500 / 700 / 1,000 -- gate 1's first clause passes; the text control 0.00 throughout, the language
held-out loss 2.075 in both arms -- gate 2 passes; the in-window control learns the copy slowly,
0.22 at 1,000 steps (the second clause not shown at this budget). ``--query learned`` (H-S12's
route, the trunk recovering the address from its state): 0.00 for all 1,000 steps, the gate
drifting shut from -4 and from 0 alike -- the route is the hard part; the read is not.

Second read, the route (2026-10-20, same budget): ``--route-loss 1.0`` -- the learned query
taught its address at placed tokens, 1 - cos(unitary(q), S(p)) -- takes the learned route from
0.00 to **0.82** (route cosine 0.54); ``--gate-warmup 250`` alone (the gate pinned open, Gemini's
freeze-and-release) 0.00, with the route loss 0.77; ``--query spectral`` (a per-frequency
complex gain, d parameters) with the route loss 0.69, and with ``--read matched`` (the query's
conjugate at its own magnitudes instead of unitary()) 0.19; ``--q-init identity`` with the route
loss 0.66. Controls: ``--window 256`` (attention over the whole document) 0.03 at 1,000 steps;
the mismatched-address check (HERE carries another fact's valid address) 0.00 on every arm --
the channel relies on the address alone. Teaching the route beats re-parametrizing it.

    python validation/exp_k10_hypertoken.py [--query placed|learned|spectral] [--route-loss 1.0] [--gate-bias 0] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402

EOS, Q, HERE, ENT0, N_ENT = 0, 1, 2, 3, 509          # 509 entity names, one token each
REL0, N_REL = ENT0 + N_ENT, 16
FILL0, N_FILL, STEP = REL0 + N_REL, 496, 6           # 3 + 509 + 16 + 496 = 1,024; 6 cells between steps (kernel ~0 at 4)


def meta(d: int, spatial: bool, query: str = "learned", window: int = 64) -> dict:
    m = {"kind": "cubby_base", "D": d, "L": 4, "heads": 4, "window": window, "attn_every": 2, "ctx": 32,
         "slots": 8, "gen_basis": 16, "gen_rank": 8, "vocab": 1024, "vocab_real": 1024, "mem_every": 0,
         "tied": False, "causal_ctx": True}
    if spatial:
        m["spatial"] = {"scale": 2.0, "seed": 0, "at": 2}
        if query != "learned":
            m["spatial"]["query"] = query                      # "placed": the link's own flag (model/spatial.py)
    return m


def cell(entity: int, rng_seed: int = 7) -> np.ndarray:
    h = np.random.default_rng(entity * 7919 + rng_seed).integers(0, 1024, 3)
    return h.astype(np.int64)


def step(rel: int) -> np.ndarray:
    return np.array([STEP * (rel % 4), STEP * (rel // 4), 0], dtype=np.int64)


class Language:
    def __init__(self, rng) -> None:
        self.next = rng.integers(0, N_FILL, (N_FILL, 8))
        self.rng = rng

    def tokens(self, n: int) -> list:
        out, w = [], int(self.rng.integers(N_FILL))
        while len(out) < n:
            out.append(FILL0 + w)
            w = int(self.next[w][self.rng.integers(8)])
            if self.rng.random() < 0.1:
                out.append(EOS)
        return out[:n]


def document(rng, lang: Language, n_facts: int, filler: int, seq: int):
    """ids (seq,), places (seq, 3) int16 with NO_PLACE where unplaced, answer positions, answers."""
    ents = rng.choice(N_ENT, 2 * n_facts, replace=False)
    ids, pos = [], []
    facts = []
    for i in range(n_facts):
        s, o, r = ENT0 + int(ents[i]), ENT0 + int(ents[n_facts + i]), int(rng.integers(N_REL))
        addr = cell(s) + step(r)
        facts.append((s, r, o, addr))
        ids += [s, REL0 + r, o, EOS]
        pos += [None, None, addr, None]
    fill = lang.tokens(filler)
    ids += fill
    pos += [None] * len(fill)
    ask_at, answers = [], []
    for s, r, o, addr in [facts[i] for i in rng.permutation(n_facts)]:
        ask_at.append(len(ids) + 3)                      # the HERE position predicts o
        answers.append(o)
        ids += [Q, s, REL0 + r, HERE, o, EOS]
        pos += [None, None, None, addr, None, None]
    rest = lang.tokens(seq + 1 - len(ids))
    ids += rest
    pos += [None] * len(rest)
    ids, pos = ids[:seq + 1], pos[:seq + 1]
    P = np.full((seq + 1, 3), NO_PLACE, dtype=np.int16)
    for i, p in enumerate(pos):
        if p is not None:
            P[i] = p
    return np.array(ids, dtype=np.int64), P, ask_at, answers


def batch(rng, lang, n: int, n_facts: int, filler: int, seq: int):
    docs = [document(rng, lang, n_facts, filler, seq) for _ in range(n)]
    ids = torch.from_numpy(np.stack([d[0] for d in docs]))
    P = torch.from_numpy(np.stack([d[1] for d in docs]))
    return ids, P, [d[2] for d in docs], [d[3] for d in docs]


class SpectralQuery(torch.nn.Module):
    """``--query spectral``: the query head as a per-frequency complex gain on the normalized
    stream's spectrum -- the shape of a deconvolution by a fixed embedding, d parameters not
    d^2, starting at the identity (the stream's own phases, a fixed rotation ``o`` can undo)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.complex(torch.ones(d // 2 + 1), torch.zeros(d // 2 + 1)))

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(torch.fft.rfft(n.float(), dim=-1) * self.gain, n=n.shape[-1], dim=-1).to(n.dtype)


def forward(model, ids, P, spatial: bool):
    if not spatial:
        return model.forward(ids)
    placed = P[..., 0] != NO_PLACE
    positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
    return model.forward(ids, positions=positions, placed=placed)


@torch.no_grad()
def accuracy(model, ids, P, ask_at, answers, spatial: bool, mismatch: bool = False) -> float:
    """``mismatch``: every HERE carries a valid address of ANOTHER fact in its document (a cyclic
    shift): a channel that relies on the address alone must fall to ~0 (Gemini's control)."""
    if mismatch:
        P = P.clone()
        for b in range(ids.shape[0]):
            at = ask_at[b]
            P[b, at] = P[b, at[1:] + at[:1]]
    pred = forward(model, ids[:, :-1], P[:, :-1], spatial).argmax(-1)
    hit, n = 0, 0
    for b in range(ids.shape[0]):
        for at, o in zip(ask_at[b], answers[b]):
            hit += int(pred[b, at] == o)
            n += 1
    return hit / max(n, 1)


@torch.no_grad()
def language_loss(model, ids, P, spatial: bool) -> float:
    """Loss on the filler tokens only (the unplaced, non-question stretch)."""
    logits = forward(model, ids[:, :-1], P[:, :-1], spatial)
    tgt = ids[:, 1:]
    mask = (tgt >= FILL0) | (tgt == EOS)
    l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
    return float((l * mask.reshape(-1).float()).sum() / mask.float().sum())


def run(spatial: bool, args, log) -> dict:
    from train_base import build_model
    rng = np.random.default_rng(args.seed)
    lang = Language(np.random.default_rng(args.seed + 1))
    m = meta(args.d, spatial, "placed" if args.query == "placed" else "learned", args.window)
    model = build_model(m, torch.device("cpu"), seed=args.seed)
    if spatial and args.read == "matched":                         # the read by the query's conjugate at its own
        link = model.spatial                                       # magnitudes (unit power), not unitary()'s stretch
        model.spatial.lookup = lambda scene, q: link.unbind(scene, q / (q.norm(dim=-1, keepdim=True) + 1e-6))
    if spatial and args.gate_bias is not None:
        with torch.no_grad():
            model.spatial.gate.bias.fill_(args.gate_bias)         # the design's -4 unless asked
    if spatial and args.query == "spectral":
        model.spatial.q = SpectralQuery(args.d)                   # the link calls self.q(n): a drop-in
    elif spatial and args.q_init == "identity":
        with torch.no_grad():
            model.spatial.q.weight.copy_(torch.eye(args.d))       # the query starts as the stream's own phases
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))
    held = batch(np.random.default_rng(args.seed + 99), Language(np.random.default_rng(args.seed + 1)), 64, args.facts_per_doc, args.filler, args.seq)
    name = f"hypertoken ({args.query} query)" if spatial else "text control"
    extras = ((f", gate warmup {args.gate_warmup}" if args.gate_warmup else "") + (f", route loss {args.route_loss}" if args.route_loss else "")
              + (f", q init {args.q_init}" if args.q_init != "default" else "") + (f", {args.read} read" if args.read != "unitary" else ""))
    log(f"--- {name}: {n_par / 1e6:.2f}M params, D={args.d}, window {args.window}, {args.facts_per_doc} facts/doc, filler {args.filler}{extras}")
    taps: list = []
    if spatial and args.route_loss:
        model.spatial.q.register_forward_hook(lambda _m, _i, out: taps.append(out))   # the learned query, once per forward
    if spatial and args.gate_warmup:
        with torch.no_grad():
            model.spatial.gate.bias.fill_(2.0)                     # forced open while the read is learned
    t0, curve = time.time(), []
    for s in range(args.steps):
        ids, P, _, _ = batch(rng, lang, args.batch, args.facts_per_doc, args.filler, args.seq)
        taps.clear()
        logits = forward(model, ids[:, :-1], P[:, :-1], spatial)
        loss = F.cross_entropy(logits.reshape(-1, m["vocab"]), ids[:, 1:].reshape(-1))
        route = torch.zeros(())
        if spatial and args.route_loss and taps:
            placed = P[:, :-1, 0] != NO_PLACE                        # the route loss: the learned query at a placed token
            S = model.spatial.encode(torch.where(placed.unsqueeze(-1), P[:, :-1], torch.zeros_like(P[:, :-1])).float())
            route = 1.0 - F.cosine_similarity(model.spatial.unitary(taps[0][placed]), S[placed], dim=-1).mean()
            loss = loss + args.route_loss * route                    # must land on S(p), the address it carries
        opt.zero_grad()
        loss.backward()
        if spatial and args.gate_warmup and s < args.gate_warmup:
            model.spatial.gate.bias.grad = None                    # pinned open
        elif spatial and args.gate_warmup and s == args.gate_warmup:
            with torch.no_grad():
                model.spatial.gate.bias.fill_(0.0)                 # released: the read must now earn its keep
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == args.steps - 1:
            acc = accuracy(model, *held, spatial)
            ll = language_loss(model, held[0], held[1], spatial)
            g = f"  gate bias {float(model.spatial.gate.bias):+.2f}" if spatial else ""
            r = f"  route {float(route):.3f}" if args.route_loss else ""
            curve.append({"step": s, "loss": float(loss), "answers": acc, "language": ll, "route": float(route)})
            log(f"  step {s:5d}  loss {float(loss):.3f}  held-out answers {acc:.2f}  language {ll:.3f}{g}{r}   [{time.time() - t0:.0f}s]")
    out = {"name": name, "params": n_par, "final_answers": curve[-1]["answers"], "final_language": curve[-1]["language"], "curve": curve}
    if spatial:
        out["mismatched_address"] = accuracy(model, *held, spatial, mismatch=True)
        log(f"  mismatched-address control (HERE carries another fact's address): {out['mismatched_address']:.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--facts-per-doc", type=int, default=8)
    ap.add_argument("--filler", type=int, default=120, help="filler tokens between the facts and the questions")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gate-bias", type=float, default=None, help="override the link's starting gate bias (-4)")
    ap.add_argument("--gate-warmup", type=int, default=0, help="pin the gate open (+2) for this many steps, then release at 0")
    ap.add_argument("--route-loss", type=float, default=0.0,
                    help="weight of 1 - cos(unitary(q), S(p)) at placed tokens: the learned query taught its address")
    ap.add_argument("--query", choices=["learned", "placed", "spectral"], default="learned",
                    help="learned: H-S12's route, the trunk makes the query from its state; placed: a placed "
                         "token's query is its own address (the hypertoken carries it) -- the link's flag; "
                         "spectral: the learned route with a per-frequency query head (SpectralQuery)")
    ap.add_argument("--q-init", choices=["default", "identity"], default="default",
                    help="learned route: start the query head at the identity (the stream's own phases)")
    ap.add_argument("--read", choices=["unitary", "matched"], default="unitary",
                    help="matched: read by the query's conjugate at its own magnitudes (unit power), not unitary()")
    ap.add_argument("--window", type=int, default=64, help="the attention window (256 sees the whole document)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=["hypertoken", "control"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE_DIR, "logs", "exp_k10_hypertoken.json"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    report = {"args": vars(args)}
    for arm in args.only:
        report[arm] = run(arm == "hypertoken", args, log)
    report["log"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
