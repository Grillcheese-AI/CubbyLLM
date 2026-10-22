"""exp_k10_switch -- order 0021 A (H-K10): the switch alone on order 0006's toy, order 0005's recipe.

0006's copy path alone picks the right answer on ~0.96 of present facts and the mix on ~0.85: the trained language
model outvotes the copy where the gate is under 0.5 on a piece, and absent facts get an answer where it is open.
Here the joint model (``exp_k10_free_text.py --save``) is loaded and everything is frozen but the copy path's gate,
``tau`` and ``beta`` -- the trunk, the embeddings, the pointer -- then trained ``--copy-only`` steps more on the same
documents (absent facts included), with the language loss alone. The frozen parameters' hash is taken before and
after (it must not move), and the held-out set is the joint run's own.

Reported as 0006 reports, plus present answered wrong (``1 - exact - refused``) as its own column; at the end, for
the kill line, the read's best-match cosine on present and on empty pieces (can the gate's inputs tell them apart?).

    python validation/exp_k10_free_text.py --arm U --seed 0 --save <ckpt>        # the joint phase
    python validation/exp_k10_switch.py --resume <ckpt> --copy-only 500 [--threads 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_k10_free_text as ft  # noqa: E402
import exp_k10_hypertoken as kh  # noqa: E402
from exp_k10_far_facts import fwd  # noqa: E402
from k10_free_docs import FreeLayout, batch, held_cells  # noqa: E402

SWITCH = ("spatial.copy_path.gate.weight", "spatial.copy_path.gate.bias", "spatial.copy_path.tau", "spatial.copy_path.beta")


PARTS = ("embedding", "backbone", "memory", "head", "spatial", "context_source")


def tensors(model) -> list:
    """Every tensor of every part, named, listed at the first call (a fresh or loaded model: all trainable) (``CubbyModel`` is not an
    ``nn.Module``, and the embedding and head list only trainable tensors): what ``--save`` writes and a resume loads."""
    if getattr(model, "_tensors", None) is None:
        out = []
        for k in PARTS:
            c = getattr(model, k, None)
            if hasattr(c, "named_parameters"):
                out += [(f"{k}.{n}", t) for n, t in list(c.named_parameters()) + list(c.named_buffers())]
            elif hasattr(c, "parameters"):
                out += [(f"{k}.{i}", t) for i, t in enumerate(c.parameters())]
        model._tensors = out
    return model._tensors


def state_of(model) -> dict:
    return {n: t.detach().clone() for n, t in tensors(model)}


@torch.no_grad()
def load_state(model, sd: dict) -> None:
    for n, t in tensors(model):
        t.copy_(sd[n])


def named_params(model):
    return [(n, t) for n, t in tensors(model) if isinstance(t, torch.nn.Parameter)]


def frozen_hash(model) -> str:
    """A hash of every parameter and buffer the switch-alone phase must not move."""
    h = hashlib.sha256()
    for n, t in sorted(state_of(model).items()):
        if n not in SWITCH:
            h.update(n.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def load(path):
    """The joint model, its args and the held-out set it was scored on (the same seeds, so the same documents)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**ck["args"])
    model, _m = ft.build(a)
    load_state(model, ck["state"])
    lay = FreeLayout(a.vocab)
    hr = np.random.default_rng(a.seed + 99)
    held = batch(hr, kh.Language(np.random.default_rng(a.seed + 1)), lay, a.held, a, held_cells(hr, a.held, a.absent))
    return model, a, lay, held


@torch.no_grad()
def best_match(model, a, held, lay) -> dict:
    """The read's cosine to its best candidate (the gate's own `best`) and its margin over the second best, at the first
    answer token of present facts and of absent ones, and at every other empty asking position; and how well one
    threshold on the best match alone tells present from absent there (0021's kill line: if they overlap, no gate on
    these inputs can tell them apart)."""
    ids, key, plc, R, _gold, spans = held
    out, calls = {"present": [], "absent": [], "empty": []}, []

    def best(_m, inp, _o):
        r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
        e = F.normalize(table.float()[cid], dim=-1)
        rn = F.normalize(r.float(), dim=-1)
        sim = (rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)).masked_fill(~ok, float("-inf"))
        top = sim.topk(min(2, sim.shape[-1]), -1).values
        calls.append((idx, top[:, 0], top[:, 0] - (top[:, 1] if top.shape[-1] > 1 else 0)))
    hook = model.spatial.copy_path.register_forward_hook(best)
    try:
        for b in range(0, ids.shape[0], 16):
            sl, S = slice(b, b + 16), ids.shape[1] - 1
            calls.clear()
            fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
            for ix, bst, mg in calls:
                for j, v, m in zip(ix.tolist(), bst.tolist(), mg.tolist()):
                    i, t = divmod(j, S)
                    here, o, info = spans[b + i]
                    k = ("absent" if info["absent"] else "present") if t == here else "empty"
                    if k != "empty" or not (here <= t < here + len(o)) or info["absent"]:
                        out[k].append((v, m if np.isfinite(m) else 0.0))
    finally:
        hook.remove()
    q = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 4) for p in (5, 25, 50, 75, 95)}   # noqa: E731
    rep = {k: {"n": len(v), "best": q([x[0] for x in v]), "margin": q([x[1] for x in v])} for k, v in out.items() if v}
    if out["present"] and out["absent"]:
        pb, ab = np.array([x[0] for x in out["present"]]), np.array([x[0] for x in out["absent"]])
        score, thr = max(((float((pb >= t).mean() + (ab < t).mean()) / 2, float(t)) for t in np.linspace(0, 0.6, 241)))
        rep["one_threshold"] = {"at": round(thr, 4), "balanced_accuracy": round(score, 4)}
    return rep


def gate_targets(idx, spans, S):
    """1 where the asked fact holds a piece at that asking position, 0 elsewhere (an absent fact, the closing EOS)."""
    out = []
    for j in idx.tolist():
        i, t = divmod(j, S)
        here, o, info = spans[i]
        out.append(float(not info["absent"] and here <= t < here + len(o)))
    return torch.tensor(out)


def stream(a):
    """The switch-alone phase's documents: new draws (seed + 7) in the joint run's own filler language -- the language's
    grammar is drawn from its seed (seed + 1, as the joint run and the held-out set), and another seed is a language
    the trunk never learned (the first runs used seed + 8: the training loss sat at ~10.5 nats, 2026-10-22)."""
    return np.random.default_rng(a.seed + 7), kh.Language(np.random.default_rng(a.seed + 1))


def copy_only(model, a, held, lay, steps: int, lr: float, log, eval_every: int = 100, gate_loss: float = 0.0) -> dict:
    """`gate_loss` (off by default: the order's recipe is the language loss alone): adds the gate's binary cross-entropy
    against `gate_targets`, one term an asking position -- a present fact's five pieces and an absent fact's one NONE
    weigh alike there, where in the language loss the pieces outvote it."""
    for n, p in named_params(model):
        p.requires_grad_(n in SWITCH)
    params = [p for n, p in named_params(model) if n in SWITCH]
    assert len(params) == len(SWITCH), [n for n, _p in named_params(model) if "copy_path" in n]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng, lang = stream(a)
    lams = []
    hook = model.spatial.copy_path.register_forward_hook(lambda _m, inp, o: lams.append((inp[4], o[2][:, 0])) if gate_loss else None)
    before, t0, curve = frozen_hash(model), time.time(), []
    ev = ft.evaluate(model, a, held, lay)
    curve.append(dict(step=0, **ev))
    log(line("joint", ev, t0))
    for s in range(1, steps + 1):
        ids, key, plc, R, _gold, spans = batch(rng, lang, lay, a.batch, a)
        lams.clear()
        logits = fwd(model, ids[:, :-1], ft.inputs(a, key, plc, R)[:, :-1], R[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, a.vocab), ids[:, 1:].reshape(-1))
        for ix, lam in lams:                                             # the switch's own target, with --gate-loss
            loss = loss + gate_loss * F.binary_cross_entropy(lam.clamp(1e-6, 1 - 1e-6), gate_targets(ix, spans, ids.shape[1] - 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % eval_every == 0 or s == steps:
            ev = ft.evaluate(model, a, held, lay)
            curve.append(dict(step=s, loss=float(loss), **ev))
            log(line(f"step {s:4d}", ev, t0))
    hook.remove()
    after = frozen_hash(model)
    return {"frozen_hash": {"before": before, "after": after, "unchanged": before == after}, "final": curve[-1], "curve": curve}


def line(tag, ev, t0) -> str:
    return (f"  {tag}  exact {ev['exact']:.3f}  seen/train {ev['seen_train_t']:.3f}  seen/held {ev['seen_held_t']:.3f}"
            f"  unseen/train {ev['unseen_train_t']:.3f}  unseen/held {ev['unseen_held_t']:.3f}  absent answered {ev['absent_not_none']:.3f}"
            f"  present wrong {ev['present_wrong']:.3f}  refused {ev['present_refused']:.3f}  language {ev['language']:.4f}"
            f"  copy alone {ev['copy_alone']:.3f}  lam here {ev['lam_present']:.3f} empty {ev['lam_empty']:.3f}  [{time.time() - t0:.0f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", required=True, help="the joint model (exp_k10_free_text.py --save)")
    ap.add_argument("--copy-only", type=int, default=500, help="steps of the switch alone")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--gate-loss", type=float, default=0.0, help="the gate's BCE on present/empty asking positions (off)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    c = ap.parse_args()
    if c.threads:
        torch.set_num_threads(c.threads)
    model, a, lay, held = load(c.resume)
    name = (f"exp_k10_switch_{a.arm}_s{a.seed}{'_t12' if getattr(a, 'templates', 6) == 12 else ''}"
            f"{f'_gl{c.gate_loss:g}' if c.gate_loss else ''}{c.tag}_{c.date}")
    out, lines = os.path.join(HERE_DIR, "logs", name), []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"env: {time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} torch {torch.__version__} "
        f"{platform.processor() or platform.machine()} (CPU, {torch.get_num_threads()} threads)")
    log(f"--- {a.arm} seed {a.seed}: the switch alone ({', '.join(SWITCH)}), {c.copy_only} steps, lr {c.lr}")
    rep = {"args": {k: v for k, v in vars(a).items() if k != "save"}, "copy_only": {k: v for k, v in vars(c).items() if k != "resume"}}
    rep.update(copy_only(model, a, held, lay, c.copy_only, c.lr, log, c.eval_every, c.gate_loss))
    rep["best_match"] = best_match(model, a, held, lay)
    bm = rep["best_match"]
    log(f"  frozen parameters unchanged: {rep['frozen_hash']['unchanged']}; best match (p5/p50/p95) present "
        f"{[bm['present']['best'][k] for k in ('p5', 'p50', 'p95')]}, absent {[bm['absent']['best'][k] for k in ('p5', 'p50', 'p95')]};"
        f" one threshold {bm.get('one_threshold')}")
    rep["log"] = lines
    json.dump(rep, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {out}.json")


if __name__ == "__main__":
    main()
