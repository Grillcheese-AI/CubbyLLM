"""exp_k7_exposure -- the exposure curve on our own trunk (H-K7, gate 1): how many times must a
fact be seen in training before the trunk recalls it, against a store that is written once.

The knowledge budget split says a trunk's tokens should buy language, not facts: a fact in the
store costs one write; a fact in the weights costs exposures ("Physics of Language Models 3.3":
~1,000, and diversity of wording is what makes it hard). Here is the law measured on the trunk we
actually train (``train_base.build_model``, tiny), on a synthetic world with a 1,024-symbol
vocabulary: 1,000 facts  ``s1 s2 REL o1 o2``  in four groups of 250 seen 1 / 10 / 100 / 1,000 times
in ONE pass over a corpus where they are mixed into a Markov "language" (a fixed sparse bigram
chain over 496 filler words, so the trunk has a language to learn beside the facts), each
exposure in one of four wordings (a filler word before, between, after, or none) when
``--paraphrase`` is on. Recall = both object tokens right, teacher-forced, from the bare wording.

The store's number is not measured because it is a construction: one write, recall 1.00 (the exact
store), or H-K1's Hits@1 for the vector map.

Gate 1 (pre-registered): recall < 5% at 1 exposure and >= 80% at 1,000 (the law holds on our
trunk); the curve in between is the price list.

First read (2026-10-20, 1.06M-parameter trunk, 3.75M tokens, 10 min on the CPU): with paraphrase,
exact recall 0.00 / 0.00 / 0.50 / 0.99 at 1 / 10 / 100 / 1,000 exposures (first object token
0.00 / 0.01 / 0.74 / 1.00), language held-out loss 6.96 -> 2.67 -- gate 1 passes; verbatim
wording, 0.00 / 0.00 / 0.30 / 0.86 (fewer tokens, 406 steps against 457, and the 1,000x group
wanders 0.83-0.93 over the last steps): wording diversity helps extraction here as in the law's
paper. The trunk's price for a fact is somewhere between 100 and 1,000 sightings; the store's is one.

    python validation/exp_k7_exposure.py [--paraphrase] [--facts 1000] [--out <json>]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

EOS, NAMES, RELS, FILL = 0, 511, 16, 496           # 1 + 511 + 16 + 496 = 1,024 symbols
REL0, FILL0 = 1 + NAMES, 1 + NAMES + RELS
META = {"kind": "cubby_base", "D": 128, "L": 4, "heads": 4, "window": 64, "attn_every": 2, "ctx": 32,
        "slots": 8, "gen_basis": 16, "gen_rank": 8, "vocab": 1024, "vocab_real": 1024, "mem_every": 0,
        "tied": False, "causal_ctx": True}
EXPOSURES = (1, 10, 100, 1000)


def world(rng, n_facts: int):
    """Entities are pairs of name pieces; (subject, relation) -> object, all unique."""
    ents = set()
    while len(ents) < int(n_facts * 1.5):
        ents.add((int(rng.integers(1, 1 + NAMES)), int(rng.integers(1, 1 + NAMES))))
    ents = list(ents)
    facts, seen = [], set()
    while len(facts) < n_facts:
        s, o = ents[rng.integers(len(ents))], ents[rng.integers(len(ents))]
        r = REL0 + int(rng.integers(RELS))
        if s != o and (s, r) not in seen:
            seen.add((s, r))
            facts.append((s, r, o))
    return facts


def wording(rng, fact, paraphrase: bool) -> list:
    (s1, s2), r, (o1, o2) = fact
    if not paraphrase:
        return [s1, s2, r, o1, o2, EOS]
    w = FILL0 + int(rng.integers(FILL))
    return [[s1, s2, r, o1, o2, EOS], [w, s1, s2, r, o1, o2, EOS], [s1, s2, w, r, o1, o2, EOS],
            [s1, s2, r, o1, o2, w, EOS]][int(rng.integers(4))]


class Language:
    """A fixed sparse bigram chain over the filler words: 8 successors per word."""

    def __init__(self, rng) -> None:
        self.next = rng.integers(0, FILL, (FILL, 8))
        self.rng = rng

    def sentence(self) -> list:
        n = int(self.rng.integers(5, 16))
        w = int(self.rng.integers(FILL))
        out = [FILL0 + w]
        for _ in range(n - 1):
            w = int(self.next[w][self.rng.integers(8)])
            out.append(FILL0 + w)
        return out + [EOS]


def corpus(rng, facts: list, paraphrase: bool, seq: int):
    """One pass: every fact of group g appears EXPOSURES[g] times, in as many filler tokens."""
    per = len(facts) // len(EXPOSURES)
    sents = []
    for g, k in enumerate(EXPOSURES):
        for f in facts[g * per:(g + 1) * per]:
            sents += [wording(rng, f, paraphrase) for _ in range(k)]
    n_fact = sum(len(s) for s in sents)
    lang = Language(rng)
    n = 0
    while n < n_fact:
        s = lang.sentence()
        sents.append(s)
        n += len(s)
    order = rng.permutation(len(sents))
    flat = np.array([t for i in order for t in sents[i]], dtype=np.int64)
    n_win = len(flat) // (seq + 1)
    return torch.from_numpy(flat[:n_win * (seq + 1)].reshape(n_win, seq + 1)), lang, n_fact


@torch.no_grad()
def recall(model, facts: list, per: int) -> list:
    """Teacher-forced exact recall of (o1, o2) after ``s1 s2 r``, per exposure group."""
    x = torch.tensor([[s1, s2, r, o1, o2] for (s1, s2), r, (o1, o2) in facts])
    pred = model.forward(x).argmax(-1)                         # (N, 5): pred[:, 2] -> o1, pred[:, 3] -> o2
    ok = (pred[:, 2] == x[:, 3]) & (pred[:, 3] == x[:, 4])
    first = pred[:, 2] == x[:, 3]
    out = []
    for g, k in enumerate(EXPOSURES):
        sl = slice(g * per, (g + 1) * per)
        out.append({"exposures": k, "n": per, "exact": float(ok[sl].float().mean()),
                    "first_token": float(first[sl].float().mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=1000)
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "exp_k7_exposure.json"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    from train_base import build_model
    t0 = time.time()
    facts = world(rng, args.facts)
    per = args.facts // len(EXPOSURES)
    data, lang, n_fact = corpus(rng, facts, args.paraphrase, args.seq)
    held = torch.tensor([t for _ in range(2000) for t in lang.sentence()][:40 * (args.seq + 1)]).reshape(40, args.seq + 1)
    model = build_model(META, torch.device("cpu"), seed=args.seed)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"exposure curve on the trunk: {n_par / 1e6:.2f}M params (D={META['D']} L={META['L']}), {len(facts)} facts in "
          f"groups of {per} seen {EXPOSURES} times, paraphrase={args.paraphrase}; corpus {data.numel() / 1e6:.2f}M tokens "
          f"({n_fact / 1e6:.2f}M of facts), one pass", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps = math.ceil(len(data) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    perm = torch.randperm(len(data))
    log, curve = [], []
    for step in range(steps):
        b = data[perm[step * args.batch:(step + 1) * args.batch]]
        logits = model.forward(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, META["vocab"]), b[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                hl = F.cross_entropy(model.forward(held[:, :-1]).reshape(-1, META["vocab"]), held[:, 1:].reshape(-1))
            r = recall(model, facts, per)
            curve.append({"step": step, "loss": float(loss), "held_language": float(hl), "recall": r})
            line = (f"  step {step:4d}/{steps}  loss {float(loss):.3f}  language held-out {float(hl):.3f}  exact recall "
                    + "  ".join(f"{x['exposures']}x:{x['exact']:.2f}" for x in r) + f"   [{time.time() - t0:.0f}s]")
            print(line, flush=True)
            log.append(line)
    final = recall(model, facts, per)
    report = {"args": vars(args), "meta": META, "params": n_par, "tokens": int(data.numel()), "fact_tokens": n_fact,
              "final": final, "curve": curve, "log": log, "wall_s": round(time.time() - t0, 1),
              "store": {"writes_per_fact": 1, "exact_store_recall": 1.0}}
    print("  final: " + "  ".join(f"{x['exposures']}x: exact {x['exact']:.2f} (first token {x['first_token']:.2f})" for x in final))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"  wrote {args.out}  ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
