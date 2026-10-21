"""exp_k10_far_facts -- the far-facts failure on a CPU toy: one difference at a time from the toy
that reads its facts (``exp_k10_hypertoken.py``, 0.94) toward the 450M that does not (0.021), and
the candidate fixes on the arms that break (order 0001, H-K10).

The documents are the toy's, generalized (``k10_far_docs.py``); at its settings they are its
documents exactly (``selfcheck``, run at every start). The ladder's knobs:

  --vocab V         answers from a large vocabulary: ids 0-1,023 are the toy's, entities fill 1,024..V-1
  --pieces k        an object is 1..k entity tokens, piece i at the address + 6 i z (``make_fact_shard
                    --lead``'s layout: HERE, right before the answer, asks for piece 0, piece j for j + 1,
                    the last asks for one that is not there and predicts EOS); implies --roles
  --roles           the 450M's roles: a fact's object pieces WRITE, a question's tokens only ASK, and only
                    writers are bound at the input (``bind="writes"``) -- the toy's HERE also writes
  --pretrain S      S steps text-only (nothing placed: the link gets no gradient), then the link as
                    ``CB_INIT_FROM`` adds it (o = 0, the gate at --gate-bias) and a fresh optimizer
  --questions q     q of the document's facts asked (--sparse: 1 of 8, ~0.4% answer tokens)
  --layers --at     the depth after the read; --gap N the filler between the facts and the questions
Fixes: --copy ring|table (B1, ``meta["spatial"]["copy"]``: the document's written ids, or the whole table), --copy-aux λ (B5: a CE of the copy logits
alone at ASK tokens), --answer-weight w (B3: the CE at answer tokens times w), --at L-2 (B2).

Scored on held-out documents like ``diag_fact_read.py``: the first piece, the later pieces, the
whole answer (``exact``), the EOS after it (``end``), the filler language's loss, and the read
itself -- the scene unbound at each asking token's address and cleaned up against the embedding
table (top-1, cos to the target), A0's measure, as it trains.

    python validation/exp_k10_far_facts.py --arm <name> [knobs] [--threads 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_k10_hypertoken as kh  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402

from k10_far_docs import ASK, WRITE, Layout, batch, selfcheck  # noqa: E402


def fwd(model, ids, P, R, places: bool = True):
    if not places:
        return model.forward(ids)
    placed = P[..., 0] != NO_PLACE
    positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
    if R is None:
        return model.forward(ids, positions=positions, placed=placed)
    return model.forward(ids, positions=positions, placed=placed, writes=placed & (R == WRITE), asks=placed & (R == ASK))


@torch.no_grad()
def evaluate(model, held, a) -> dict:
    ids, P, R, spans = held
    first = later = n_later = exact = end = n = 0
    lang_sum = lang_n = rd_hit = rd_cos = n_rd = 0.0
    link, E, gaps, lams = model.spatial, model.embedding.core.weight.float(), [], []
    hook = link.copy_path.register_forward_hook(lambda _m, _i, out: lams.append(out[2])) if link.copy_path else None
    for b in range(0, ids.shape[0], 8):
        x, p, r = ids[b:b + 8], P[b:b + 8], (R[b:b + 8] if R is not None else None)
        logits = fwd(model, x[:, :-1], p[:, :-1], None if r is None else r[:, :-1])
        pred, tgt = logits.argmax(-1), x[:, 1:]
        fill = ((tgt >= kh.FILL0) & (tgt < kh.FILL0 + kh.N_FILL)) | (tgt == kh.EOS)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
        lang_sum += float((ce * fill.reshape(-1)).sum()); lang_n += float(fill.sum())
        placed = p[:, :-1, 0] != NO_PLACE
        w = placed if r is None else placed & (r[:, :-1] == WRITE)
        pos = torch.where(placed.unsqueeze(-1), p[:, :-1], torch.zeros_like(p[:, :-1])).float()
        scene = link.scene(link.bind_input(E[x[:, :-1]], pos, w), w)
        for i in range(x.shape[0]):
            for h, o in spans[b + i]:
                ok = [int(pred[i, h + j]) == o[j] for j in range(len(o))]
                first += ok[0]; later += sum(ok[1:]); n_later += len(o) - 1; exact += all(ok); n += 1
                end += int(pred[i, h + len(o)]) == kh.EOS
                rd = link.lookup(scene[i, h], link.encode(pos[i, h]))           # the read at HERE, A0's cleanup
                cs = F.cosine_similarity(rd[None], E, dim=-1)
                rd_hit += int(cs.argmax()) == o[0]; rd_cos += float(cs[o[0]]); n_rd += 1
                wrong = [v for v in set(x[i, :h][w[i, :h]].tolist()) if v != o[0]]   # the other writers
                gaps.append(float(cs[o[0]] - cs[wrong].max()) if wrong else float("nan"))
    if hook is not None:
        hook.remove()
    n = max(n, 1)
    return {"exact": exact / n, "first": first / n, "later": later / max(n_later, 1), "end": end / n,
            "language": lang_sum / max(lang_n, 1), "read_top1": rd_hit / max(n_rd, 1), "read_cos": rd_cos / max(n_rd, 1),
            "questions": n, "writer_gap": float(np.nanmedian(gaps)) if gaps else 0.0,
            "lam": float(torch.cat(lams).mean()) if lams else None}


def build(a):
    from train_base import build_model
    m = kh.meta(a.d, True, a.query, a.window)
    m.update(vocab=a.vocab, vocab_real=a.vocab, L=a.layers)
    m["spatial"]["at"] = a.at if a.at is not None else a.layers // 2
    if a.roles:
        m["spatial"]["bind"] = "writes"
    if a.copy:
        m["spatial"]["copy"] = a.copy
    model = build_model(m, torch.device("cpu"), seed=a.seed)
    with torch.no_grad():
        model.spatial.gate.bias.fill_(a.gate_bias)
    return model, m


def optimizer(model, a, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))


def run(a, log) -> dict:
    rng = np.random.default_rng(a.seed)
    lang = kh.Language(np.random.default_rng(a.seed + 1))
    lay = Layout(a.vocab, a.entities)
    model, m = build(a)
    n_held = max(64, -(-512 // max(a.questions, 1)))                      # >= 512 held-out questions
    held = batch(np.random.default_rng(a.seed + 99), kh.Language(np.random.default_rng(a.seed + 1)), lay, n_held, a)
    log(f"--- {a.arm}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, meta {m}, {lay.n_ent} entities, "
        f"{n_held} held-out documents ({sum(len(s) for s in held[3])} questions)")
    held_w = held[1][..., 0] != NO_PLACE if held[2] is None else held[2] == WRITE
    items = float(held_w.sum(1).float().mean())
    distinct = float(np.mean([len(set(held[0][i][held_w[i]].tolist())) for i in range(n_held)]))
    import sim_hrr_ceiling as hc                                          # the read's best case (Addendum 2)
    cr = np.random.default_rng(0)
    ceil = {"items": items, "distinct_written": distinct, "vocab": hc.ceiling(round(items), a.vocab, a.d, 300, cr),
            "written": hc.ceiling(round(items), max(round(distinct), round(items)), a.d, 1500, cr)}
    log(f"  ceiling per piece (random codes, ideal decoder): {ceil['vocab']:.3f} over the vocabulary, {ceil['written']:.3f}"
        f" over the ~{distinct:.0f} written tokens ({items:.1f} items a scene)")
    if a.copy and a.copy_tau is None:                                      # tau: A0's median writer gap -> ~2 nats
        a.copy_tau = math.log(2.0 / max(evaluate(model, held, a)["writer_gap"], 1e-3))
    if a.copy:
        with torch.no_grad():
            model.spatial.copy_path.tau.fill_(a.copy_tau)
        log(f"  copy {a.copy}: tau {a.copy_tau:.3f} (exp {math.exp(a.copy_tau):.1f}), lam 0.1")
    taps, copy_taps = [], []
    if a.route_loss:
        model.spatial.q.register_forward_hook(lambda _m, _i, out: taps.append(out) if torch.is_grad_enabled() else None)
    if a.copy_aux:
        model.spatial.copy_path.register_forward_hook(
            lambda _m, inp, out: copy_taps.append((inp[4], out[1])) if torch.is_grad_enabled() else None)
    t0, curve = time.time(), []
    phases = ([("pretrain", a.pretrain)] if a.pretrain else []) + [("link", a.steps)]
    for phase, steps in phases:
        opt, sched = optimizer(model, a, steps)                              # CB_INIT_FROM: a fresh optimizer
        places = phase == "link"
        for s in range(steps):
            ids, P, R, spans = batch(rng, lang, lay, a.batch, a)
            taps.clear(); copy_taps.clear()
            logits = fwd(model, ids[:, :-1], P[:, :-1], None if R is None else R[:, :-1], places)
            ce = F.cross_entropy(logits.reshape(-1, a.vocab), ids[:, 1:].reshape(-1), reduction="none")
            if a.answer_weight != 1.0:                                       # B3: the answer tokens' CE times w
                wt = torch.ones_like(ids[:, 1:], dtype=torch.float32)
                for i, sp in enumerate(spans):
                    for h, o in sp:
                        wt[i, h:h + len(o)] = a.answer_weight
                ce = ce * wt.reshape(-1)
            loss = ce.mean()
            if places and taps:                                              # the route loss, as the toy has it
                placed = P[:, :-1, 0] != NO_PLACE
                S = model.spatial.encode(torch.where(placed.unsqueeze(-1), P[:, :-1], torch.zeros_like(P[:, :-1])).float())
                loss = loss + a.route_loss * (1.0 - F.cosine_similarity(model.spatial.unitary(taps[0][placed]), S[placed], dim=-1).mean())
            if places and copy_taps:                                         # B5: the copy logits alone, at ASK tokens
                idx, cl = copy_taps[0]                                       # log p_copy: an NLL, not a CE over it
                loss = loss + a.copy_aux * F.nll_loss(cl, ids[:, 1:].reshape(-1)[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if places and (s % 100 == 0 or s == steps - 1):
                ev = evaluate(model, held, a)
                curve.append(dict(step=s, loss=float(loss), **ev))
                log(f"  step {s:5d}  loss {float(loss):.3f}  exact {ev['exact']:.3f}  first {ev['first']:.3f}  later {ev['later']:.3f}"
                    f"  end {ev['end']:.3f}  language {ev['language']:.3f}  read top-1 {ev['read_top1']:.3f} cos {ev['read_cos']:.3f}"
                    f"  gap {ev['writer_gap']:.3f}" + (f"  lam {ev['lam']:.3f}" if ev["lam"] is not None else "")
                    + f"  gate {float(model.spatial.gate.bias):+.2f}   [{time.time() - t0:.0f}s]")
            elif not places and (s % 250 == 0 or s == steps - 1):
                log(f"  pretrain {s:5d}  loss {float(loss):.3f}   [{time.time() - t0:.0f}s]")
    return {"meta": m, "entities": lay.n_ent, "ceiling": ceil, "final": curve[-1], "curve": curve, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--facts-per-doc", type=int, default=8)
    ap.add_argument("--questions", type=int, default=0, help="facts asked per document (0: all)")
    ap.add_argument("--sparse", action="store_true", help="= --questions 1")
    ap.add_argument("--gap", type=int, default=120, help="filler tokens between the facts and the questions")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--entities", type=int, default=0, help="0: every id the vocabulary has for them")
    ap.add_argument("--pieces", type=int, default=1)
    ap.add_argument("--roles", action="store_true")
    ap.add_argument("--pretrain", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--at", type=int, default=None, help="the read's layer (default L / 2)")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--query", choices=["placed", "learned"], default="placed")
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--route-loss", type=float, default=1.0)
    ap.add_argument("--copy", choices=["ring", "table"], default=None, help="B1: the copy path over the ring or the vocabulary")
    ap.add_argument("--copy-tau", type=float, default=None, help="its starting log temperature (default: ~2 nats over A0's gap)")
    ap.add_argument("--copy-aux", type=float, default=0.0, help="B5: weight of the copy logits' own CE at ASK tokens")
    ap.add_argument("--answer-weight", type=float, default=1.0, help="B3: the CE at answer tokens times w")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    a.questions = 1 if a.sparse else (a.questions or a.facts_per_doc)
    a.roles = a.roles or a.pieces > 1
    if a.copy_aux and not a.copy:
        raise SystemExit("--copy-aux needs --copy")
    if a.threads:
        torch.set_num_threads(a.threads)
    selfcheck()
    out = os.path.join(HERE_DIR, "logs", f"exp_k10_far_facts_{a.arm}_{a.date}")
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"env: {time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} torch {torch.__version__} numpy {np.__version__} "
        f"{platform.processor() or platform.machine()} (CPU, {torch.get_num_threads()} threads)")
    report = {"args": vars(a), **run(a, log)}
    report["log"] = lines
    json.dump(report, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {out}.json")


if __name__ == "__main__":
    main()
