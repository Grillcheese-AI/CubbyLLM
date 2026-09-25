"""Grow a trained CubbyModel into a bigger one that computes the same function.

Wired: STANDALONE — an offline tool (``validation/grow_base.py``), never in the
forward path. Plan and evidence: ``docs/research/2026-09-25-grow-450m.md``.

Three operations, composable in one call:

* **width** ×k (HyperCloning, arXiv 2409.12903). Every hidden vector h becomes
  [h, …, h] (k copies). A weight that maps hidden to hidden becomes the k×k
  block matrix of W/k, so it sends [h, h] to [Wh, Wh]. Why each part stays exact:
  - RMSNorm: the RMS of [h, h] equals the RMS of h, so its gain is just tiled.
  - MinGRU: its gate, value and decay are per-channel, and so is the recurrence,
    so the state becomes [s, s].
  - Attention: heads are duplicated and the head size is kept, so the 1/√dh
    scale and RoPE are unchanged.
  - Embedding and head: the embedding is tiled; the untied head becomes
    [H/k, H/k], so the logits are unchanged.
  - Router and generated adapter: their input sides are split k ways like any
    hidden-to-anything weight; the adapter's output side is tiled.
* **depth** ×n. Each block of ``attn_every`` layers is followed by n-1 copies of
  itself, so the grown model keeps the attention-every-3rd pattern that
  ``HybridBackbone`` derives from the layer index. The tail, when L is not a
  multiple of ``attn_every``, is copied by kind. With ``exit="zero"`` each copy
  starts with its exit zeroed, so it adds exactly nothing to the residual
  stream until trained:
  - attention: o_proj = 0;
  - MinGRU (it has no output projection): value weight and bias = 0, which
    keeps tanh(v) = 0 and the state at 0;
  - FFN: down_proj = 0.
  ``exit="copy"`` keeps the copies whole, as G_stack does (arXiv 2405.15319);
  that is not exact.
* **ffn** ×m. Each SwiGLU gains (m-1)× its hidden units: copies of existing
  units with a little noise on gate and up, to break the symmetry, and zero
  columns in down_proj, so the new units add nothing until trained.

``grow_into`` fills an already-built destination model (built by the caller
from ``grown_meta``), so this module never needs to know how a model is
assembled.
"""
from __future__ import annotations

import torch

from ..core.protocols import Wiring


def grown_meta(meta: dict, width: int = 1, depth: int = 1, ffn: int = 1) -> dict:
    """The destination architecture's meta (train_base's ``arch_meta`` shape).

    Architecture only, so a grown checkpoint's meta equals what ``arch_meta``
    gives for the same shape. ``ffn_mult`` is written only when it is not the
    default 2, which keeps every earlier checkpoint's meta unchanged."""
    for name, f in (("width", width), ("depth", depth), ("ffn", ffn)):
        if int(f) != f or f < 1:
            raise ValueError(f"{name} must be a positive integer, got {f}")
    out = dict(meta)
    out["D"] = meta["D"] * width
    out["heads"] = meta["heads"] * width          # head size is kept
    out["L"] = meta["L"] * depth
    mult = meta.get("ffn_mult", 2) * ffn
    out.pop("ffn_mult", None)
    if mult != 2:
        out["ffn_mult"] = mult
    return out


def layer_map(n_layers: int, attn_every: int, depth: int) -> list:
    """Destination layer j -> (source layer, is_copy), for ``depth`` × the layers.

    Blocks of ``attn_every`` layers are repeated in place, so every destination
    layer has the kind ``HybridBackbone`` gives index j (attention iff
    j % attn_every == 0). The tail left over when n_layers is not a multiple of
    attn_every goes in as it is, and its copies are placed by kind; when the
    tail has no copy of the kind a position needs (L=5, depth 3), the last
    layer of that kind is copied instead. (L=32, depth 2 needs no such borrow.)"""
    P, L = int(attn_every), int(n_layers)
    out = []
    for b in range(L // P):
        block = list(range(b * P, b * P + P))
        out += [(i, False) for i in block]
        for _ in range(depth - 1):
            out += [(i, True) for i in block]
    tail = list(range((L // P) * P, L))
    out += [(i, False) for i in tail]
    copies = [i for _ in range(depth - 1) for i in tail]
    for j in range(len(out), depth * L):
        want = j % P == 0
        pick = next((i for i in copies if (i % P == 0) == want), None)
        if pick is None:
            pick = next(i for i in range(L - 1, -1, -1) if (i % P == 0) == want)
        else:
            copies.remove(pick)
        out.append((pick, True))
    for j, (i, _) in enumerate(out):
        assert (j % P == 0) == (i % P == 0), (j, i)
    return out


# ── tensor maps for width ×k ─────────────────────────────────────────────────
def _in(w, k):
    """Input side split k ways: [W/k, …, W/k] along the last dim."""
    return torch.cat([w / k] * k, dim=-1) if k > 1 else w.clone()


def _out(w, k, dim=0):
    """Output side tiled: [W; …; W] along ``dim``."""
    return torch.cat([w] * k, dim=dim) if k > 1 else w.clone()


def _both(w, k):
    return _out(_in(w, k), k, 0)


def _qkv(w, k):
    """qkv is q, k and v stacked on dim 0; each is widened on its own, so the
    new heads are copies of the old heads, in order."""
    return torch.cat([_both(p, k) for p in w.chunk(3, dim=0)], dim=0)


@torch.no_grad()
def grow_into(src, dst, width: int = 1, depth: int = 1, ffn: int = 1,
              exit: str = "zero", noise: float = 1e-3, seed: int = 0) -> dict:
    """Fill ``dst`` (built from ``grown_meta(src meta, width, depth, ffn)``)
    from ``src``. Returns a report: the layer map and the parameter counts."""
    if exit not in ("zero", "copy"):
        raise ValueError(f"exit must be 'zero' or 'copy', got {exit!r}")
    k, m = int(width), int(ffn)
    sb, db = src.backbone, dst.backbone
    L = len(sb.mix)
    if len(db.mix) != L * depth or db.d_model != sb.d_model * k:
        raise ValueError("dst is not the grown shape of src")
    lmap = layer_map(L, sb.attn_every, depth)
    g = torch.Generator().manual_seed(int(seed))

    # embedding, head, router, generated adapter
    dst.embedding.core.weight.copy_(_out(src.embedding.core.weight, k, 1))
    dst.head._param.copy_(_in(src.head._param, k))
    sr, dr = src.context_source.router, dst.context_source.router
    dr[0].weight.copy_(_in(sr[0].weight, k))
    dr[0].bias.copy_(sr[0].bias)
    dr[2].weight.copy_(sr[2].weight)
    dr[2].bias.copy_(sr[2].bias)
    dst.context_source.slots.weight.copy_(src.context_source.slots.weight)
    sg, dg = src.memory.generator, dst.memory.generator
    for a, b in ((sg.ctx_proj, dg.ctx_proj), (sg.ctx_norm, dg.ctx_norm),
                 (sg.mix[0], dg.mix[0]), (sg.mix[2], dg.mix[2])):
        b.weight.copy_(a.weight)
        b.bias.copy_(a.bias)
    dg.A_basis.copy_(_in(sg.A_basis, k))
    dg.B_basis.copy_(_out(sg.B_basis, k, 1))
    rows = [lmap[j][0] if j < len(lmap) else 0 for j in range(dg.layer_emb.shape[0])]
    dg.layer_emb.copy_(sg.layer_emb[[r % sg.layer_emb.shape[0] for r in rows]])

    # the trunk
    for j, (i, is_copy) in enumerate(lmap):
        zero = is_copy and exit == "zero"
        db.n1[j].w.copy_(_out(sb.n1[i].w, k))
        db.n2[j].w.copy_(_out(sb.n2[i].w, k))
        sm, dm = sb.mix[i], db.mix[j]
        if sb.is_attn[i]:
            dm.qkv.weight.copy_(_qkv(sm.qkv.weight, k))
            dm.o.weight.copy_(torch.zeros_like(dm.o.weight) if zero else _both(sm.o.weight, k))
        else:
            for name in ("proj_g", "proj_v", "proj_d"):
                a, b = getattr(sm, name), getattr(dm, name)
                if zero and name == "proj_v":
                    b.weight.zero_()
                    b.bias.zero_()
                else:
                    b.weight.copy_(_both(a.weight, k))
                    b.bias.copy_(_out(a.bias, k))
        sf, df = sb.ffn[i], db.ffn[j]
        gw, uw, ow = _both(sf.g.weight, k), _both(sf.u.weight, k), _both(sf.o.weight, k)
        if m > 1:
            def jitter(w):
                return w + noise * w.std() * torch.randn(w.shape, generator=g)
            gw = torch.cat([gw] + [jitter(gw) for _ in range(m - 1)], dim=0)
            uw = torch.cat([uw] + [jitter(uw) for _ in range(m - 1)], dim=0)
            ow = torch.cat([ow] + [torch.zeros_like(ow)] * (m - 1), dim=1)
        df.g.weight.copy_(gw)
        df.u.weight.copy_(uw)
        df.o.weight.copy_(torch.zeros_like(ow) if zero else ow)

    n_src = sum(p.numel() for p in src.parameters()) + sum(
        p.numel() for p in src.context_source.parameters())
    n_dst = sum(p.numel() for p in dst.parameters()) + sum(
        p.numel() for p in dst.context_source.parameters())
    return {"width": k, "depth": depth, "ffn": m, "exit": exit,
            "layers": [[j, i, bool(c)] for j, (i, c) in enumerate(lmap)],
            "params_src": n_src, "params_dst": n_dst}


__wiring__ = Wiring.STANDALONE
