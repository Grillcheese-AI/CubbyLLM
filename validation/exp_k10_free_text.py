"""exp_k10_free_text -- free-text questions on the far-facts toy (order 0006, H-K10): can a question in
its own words, about a subject the model never saw, find the fact the host wrote?

The base is the configuration that passed order 0001's B line (``exp_k10_far_facts`` combo + the copy
over the ring with the read-aware gate: vocabulary 8,192, 4 facts a document with 1-5-piece objects,
one question, 8 layers, the read at 4). The documents are ``k10_free_docs``'s: 8 question templates
(6 train, 2 held out), 20% of the entities never in training, 20% of the questions about a fact the
document does not hold (answer NONE). The arms, at the same budget and seeds:

  U  text keys, the address given (``query="placed"``): the keys work as addresses
  H  today's hashed places, the learned route and its route loss: the control (the hash wall)
  R  text keys, the learned query and the route loss: can the trunk regenerate a key?
  P  text keys, the pointer query and its pointer loss (``model/spatial_key.py``)

Scored on held-out documents: exact per cell of {seen, unseen subject} x {seen, held-out template};
an absent fact answered with anything but NONE; a present one refused (NONE); the pointers' accuracy
(P); the filler language's loss; the keys' |cos| on the table at the start and the end.

    python validation/exp_k10_free_text.py --arm U|H|R|P [--seed 0] [--threads 2]
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
from exp_k10_far_facts import fwd, optimizer  # noqa: E402
from k10_far_docs import WRITE  # noqa: E402
from k10_free_docs import HELD_T, FreeLayout, batch, held_cells  # noqa: E402

ARMS = {"U": dict(key="text", query="placed", route=1.0), "H": dict(key=None, query="learned", route=1.0),
        "R": dict(key="text", query="learned", route=1.0), "P": dict(key="text", query="pointer", route=0.0)}


def build(a):
    from train_base import build_model
    m = kh.meta(a.d, True, "learned", a.window)
    m.update(vocab=a.vocab, vocab_real=a.vocab, L=a.layers)
    m["spatial"].update(at=a.at, bind="writes", copy="ring", copy_beta=0.0)          # beta set once held-out data is in
    if ARMS[a.arm]["query"] != "learned":
        m["spatial"]["query"] = ARMS[a.arm]["query"]
    if ARMS[a.arm]["key"]:
        m["spatial"].update(key="text", pointer_window=a.window_ptr)
    model = build_model(m, torch.device("cpu"), seed=a.seed)
    with torch.no_grad():
        model.spatial.gate.bias.fill_(a.gate_bias)
    return model, m


def inputs(a, key, plc, R):
    """What the arm's model is given: the text keys or the places; a pointer's askers only their piece."""
    P = plc.clone() if not ARMS[a.arm]["key"] else key.clone()
    if ARMS[a.arm]["query"] == "pointer":
        ask = R == 2
        P[..., 0] = torch.where(ask, torch.zeros_like(P[..., 0]), P[..., 0])
        P[..., 1] = torch.where(ask, torch.zeros_like(P[..., 1]), P[..., 1])
    return P


def truth(a, key, plc):
    return plc if not ARMS[a.arm]["key"] else key


@torch.no_grad()
def calibrate(model, a, held):
    """tau and beta from the read at init, at the true addresses: the median gap between the target and the
    best wrong writer (tau ~ 2 nats over it), and the midpoint between the present reads' target cosine
    and the absent reads' best (beta), both among the document's writers."""
    ids, key, plc, R, gold, spans = held
    link, E, T = model.spatial, model.embedding.core.weight.float(), truth(a, key, plc)
    gaps, present, absent = [], [], []
    for i in range(ids.shape[0]):
        w = R[i] == WRITE
        pos = torch.where((T[i, :, 0] != NO_PLACE).unsqueeze(-1), T[i], torch.zeros_like(T[i])).float()
        scene = link.scene(link.bind_input(E[ids[i]][None], pos[None], w[None], E), w[None])[0]
        here, o, info = spans[i]
        cand = sorted(set(ids[i, :here][w[:here]].tolist()))
        for j in range(len(o) + 1):
            cs = F.cosine_similarity(link.lookup(scene[here + j], link.encode(pos[here + j], E))[None], E[cand], dim=-1)
            if j < len(o) and not info["absent"]:
                t = cand.index(o[j])
                present.append(float(cs[t]))
                others = torch.cat([cs[:t], cs[t + 1:]])
                if len(others):
                    gaps.append(float(cs[t] - others.max()))
            else:
                absent.append(float(cs.max()))
    gap, pm, am = float(np.median(gaps)), float(np.median(present)), float(np.median(absent))
    return math.log(2.0 / max(gap, 1e-3)), (pm + am) / 2, gap, pm, am


@torch.no_grad()
def key_cos(model, lay, n: int = 400) -> dict:
    """|cos| between the text keys of different (subject, relation) pairs on the current table."""
    link, E = model.spatial, model.embedding.core.weight
    if link.keys is None:
        return {}
    g = np.random.default_rng(0)
    s = torch.tensor([lay.ent(e) for e in g.choice(lay.n_ent, n)])
    r = torch.tensor(kh.REL0 + g.integers(kh.N_REL, size=n))
    k = F.normalize(link.encode(torch.stack([s, r, torch.zeros_like(s)], -1).float(), E), dim=-1)
    c = (k @ k.T)[~torch.eye(n, dtype=torch.bool) & ~((s[:, None] == s[None]) & (r[:, None] == r[None]))].abs()
    return {"mean": float(c.mean()), "p90": float(c.quantile(0.9)), "p99": float(c.quantile(0.99)),
            "max": float(c.max()), "frac_below_0.1": float((c < 0.1).float().mean())}


@torch.no_grad()
def evaluate(model, a, held, lay) -> dict:
    ids, key, plc, R, gold, spans = held
    link, ptr = model.spatial, model.spatial.pointer
    cells = {f"{'unseen' if u else 'seen'}_{'held' if h else 'train'}_t": [] for u in (0, 1) for h in (0, 1)}
    absent_wrong, refused, lam, ptr_hit = [], [], {"present": [], "empty": []}, []
    lang_sum = lang_n = 0.0
    calls = []
    hook = link.copy_path.register_forward_hook(lambda _m, inp, out: calls.append((inp[4], out[2][:, 0])))
    try:
        for b in range(0, ids.shape[0], 16):
            sl = slice(b, b + 16)
            calls.clear()
            P = inputs(a, key[sl], plc[sl], R[sl])
            logits = fwd(model, ids[sl, :-1], P[:, :-1], R[sl, :-1])
            pred, tgt = logits.argmax(-1), ids[sl, 1:]
            fill = ((tgt >= kh.FILL0) & (tgt < kh.FILL0 + kh.N_FILL)) | (tgt == kh.EOS)
            ce = F.cross_entropy(logits.reshape(-1, a.vocab), tgt.reshape(-1), reduction="none")
            lang_sum += float((ce * fill.reshape(-1)).sum()); lang_n += float(fill.sum())
            S = ids.shape[1] - 1
            for ix, lv in calls:                                             # lam where the asked fact holds a piece
                for j, v in zip(ix.tolist(), lv.tolist()):
                    i, t = divmod(j, S)
                    here, o, info = spans[b + i]
                    lam["present" if not info["absent"] and here <= t < here + len(o) else "empty"].append(v)
            if ptr is not None and getattr(ptr, "last", None) is not None:  # the pointers: both on the gold pair
                ix, lg = ptr.last
                g = gold[sl, :-1].reshape(-1, 2)[ix]
                got = lg.argmax(-1) + 1
                ptr_hit += ((got == g).all(-1) | (got == g.flip(-1)).all(-1)).tolist()   # either way round: binding commutes
            for i in range(pred.shape[0]):
                here, o, info = spans[b + i]
                ans = [int(pred[i, here + j]) for j in range(len(o))]
                if info["absent"]:
                    absent_wrong.append(ans[0] != lay.none)
                    continue
                cells[f"{'unseen' if info['unseen'] else 'seen'}_{'held' if info['template'] in HELD_T else 'train'}_t"].append(ans == o)
                refused.append(ans[0] == lay.none)
    finally:
        hook.remove()
    out = {k: float(np.mean(v)) if v else None for k, v in cells.items()}
    out.update(n={k: len(v) for k, v in cells.items()}, exact=float(np.mean(sum(cells.values(), []))),
               absent_not_none=float(np.mean(absent_wrong)), n_absent=len(absent_wrong), present_refused=float(np.mean(refused)),
               language=lang_sum / max(lang_n, 1), pointer_acc=float(np.mean(ptr_hit)) if ptr_hit else None,
               lam_present=float(np.mean(lam["present"])) if lam["present"] else None,
               lam_empty=float(np.mean(lam["empty"])) if lam["empty"] else None, tau=float(link.copy_path.tau),
               beta=float(link.copy_path.beta))
    return out


def run(a, log) -> dict:
    rng = np.random.default_rng(a.seed)
    lang = kh.Language(np.random.default_rng(a.seed + 1))
    lay = FreeLayout(a.vocab)
    model, m = build(a)
    hr = np.random.default_rng(a.seed + 99)
    held = batch(hr, kh.Language(np.random.default_rng(a.seed + 1)), lay, a.held, a, held_cells(hr, a.held, a.absent))
    log(f"--- {a.arm}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, meta {m}, {lay.n_ent} entities "
        f"({len(lay.unseen)} never in training), {a.held} held-out documents")
    tau, beta, gap, pm, am = calibrate(model, a, held)
    with torch.no_grad():
        model.spatial.copy_path.tau.fill_(tau)
        model.spatial.copy_path.beta.fill_(beta)
    k0 = key_cos(model, lay)
    log(f"  copy ring: tau {tau:.3f} (writer gap {gap:.3f}), read-aware gate beta {beta:.3f} (present {pm:.3f} / absent {am:.3f})"
        + (f"; text keys |cos| at init: mean {k0['mean']:.3f} p90 {k0['p90']:.3f} p99 {k0['p99']:.3f} max {k0['max']:.3f},"
           f" {100 * k0['frac_below_0.1']:.1f}% below 0.1" if k0 else ""))
    taps = []
    if ARMS[a.arm]["route"]:
        model.spatial.q.register_forward_hook(lambda _m, _i, out: taps.append(out) if torch.is_grad_enabled() else None)
    E = model.embedding.core.weight
    opt, sched = optimizer(model, a, a.steps)
    t0, curve = time.time(), []
    for s in range(a.steps):
        ids, key, plc, R, gold, spans = batch(rng, lang, lay, a.batch, a)
        taps.clear()
        P = inputs(a, key, plc, R)
        logits = fwd(model, ids[:, :-1], P[:, :-1], R[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, a.vocab), ids[:, 1:].reshape(-1))
        parts = {"lm": float(loss)}
        if taps:                                                             # the route loss at placed tokens
            T = truth(a, key, plc)[:, :-1]
            placed = T[..., 0] != NO_PLACE
            tgt = model.spatial.encode(torch.where(placed.unsqueeze(-1), T, torch.zeros_like(T)).float(), E)
            rl = 1.0 - F.cosine_similarity(model.spatial.unitary(taps[0][placed]), tgt[placed], dim=-1).mean()
            loss = loss + ARMS[a.arm]["route"] * rl
            parts["route"] = float(rl)
        if model.spatial.pointer is not None:                                # the pointer loss at the askers
            ix, lg = model.spatial.pointer.last
            g = gold[:, :-1].reshape(-1, 2)[ix] - 1                          # offset k -> index k - 1
            g = torch.where((g >= 0) & (g < lg.shape[-1]), g, torch.full_like(g, -100))
            pl = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), g.reshape(-1), ignore_index=-100)
            loss = loss + a.pointer_loss * pl
            parts["pointer"] = float(pl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % a.eval_every == 0 or s == a.steps - 1:
            ev = evaluate(model, a, held, lay)
            curve.append(dict(step=s, loss=float(loss), **parts, **ev))
            log(f"  step {s:5d}  loss {float(loss):.3f} ({', '.join(f'{k} {v:.3f}' for k, v in parts.items())})  exact {ev['exact']:.3f}"
                f"  seen/train {ev['seen_train_t']:.3f}  seen/held {ev['seen_held_t']:.3f}  unseen/train {ev['unseen_train_t']:.3f}"
                f"  unseen/held {ev['unseen_held_t']:.3f}  absent not NONE {ev['absent_not_none']:.3f}  refused {ev['present_refused']:.3f}"
                f"  language {ev['language']:.3f}" + (f"  pointer {ev['pointer_acc']:.3f}" if ev["pointer_acc"] is not None else "")
                + f"  lam here {ev['lam_present']:.3f} empty {ev['lam_empty']:.3f}  [{time.time() - t0:.0f}s]")
    k1 = key_cos(model, lay)
    if k1:
        log(f"  text keys |cos| at the end: mean {k1['mean']:.3f} p90 {k1['p90']:.3f} p99 {k1['p99']:.3f} max {k1['max']:.3f},"
            f" {100 * k1['frac_below_0.1']:.1f}% below 0.1")
    return {"meta": m, "calibration": {"tau": tau, "beta": beta, "writer_gap": gap, "present": pm, "absent": am},
            "key_cos": {"init": k0, "end": k1}, "final": curve[-1], "curve": curve, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--held", type=int, default=1024, help="held-out documents (one question each)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--facts-per-doc", type=int, default=4)
    ap.add_argument("--gap", type=int, default=120)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--pieces", type=int, default=5)
    ap.add_argument("--absent", type=float, default=0.2, help="questions about a fact the document does not hold")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--at", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--window-ptr", type=int, default=16, help="the pointer's reach back")
    ap.add_argument("--pointer-loss", type=float, default=1.0)
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="", help="appended to the log name")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    name = f"exp_k10_free_text_{a.arm}_s{a.seed}{a.tag}_{a.date}"
    out, lines = os.path.join(HERE_DIR, "logs", name), []

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
