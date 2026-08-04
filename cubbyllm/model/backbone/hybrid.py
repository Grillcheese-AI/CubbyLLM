"""HybridBackbone — MinGRU trunk with a sliding-window attention layer every Nth.

Wired: WIRED — the chosen default backbone. It won the H-D4 A/B against pure
``MinGRUBackbone`` on real needle recall (2026-08-04): at d=512/L=8/~0.98B tokens,
matched, the hybrid hit ~98% recall inside its window vs ~32% for pure MinGRU,
both at bounded decode state. ``MinGRUBackbone`` remains a tested, pluggable
baseline — the interface-only ``base.py`` keeps both selectable.

WHY IT EXISTS. The bake-off that chose MinGRU (H-D1) scored on bpc — a prediction
metric that is blind to content-based lookup, which is the one thing attention
buys. The dense induction probe (``validation/exp_d3_induction``) measured what
bpc could not: at induction distance T, pure MinGRU sits at chance while a
windowed-attention hybrid solves 100%, at BOUNDED state — then the 413M-config
needle A/B confirmed it at scale (recall is a clean step function of needle
distance vs window; see H-D4). This is cubby-lm's own production shape
(``trunk_torch/blocks.py``: attention every 3rd layer, ``runpod_launch.sh``: 8 of
22), which CubbyLLM had dropped on the strength of a bpc-only bake-off.

WHY WINDOWED, NOT FULL. A full-attention layer keeps a KV cache that grows with
context, which would break the O(1)-decode guarantee that
``tests/model/test_mingru_decode.py::test_state_size_is_independent_of_context_
length`` enforces and that the whole inference-cost thesis rests on. A sliding
window of ``window`` tokens keeps the per-attention-layer state at O(window) =
constant, so decode stays context-independent. The honest cost: this layer can
only match within its window, so it extends distance-robust recall to ``window``
tokens at fixed state — beyond that, recall falls back to what the recurrent
state retains. No bounded-state model does arbitrary-distance recall; that is a
property of the information, not of this design.

POSITIONS. The attention layers carry **RoPE** (rotary position encoding). The
package has no other positional signal — MinGRU gets order from its recurrence —
so without this the windowed attention would be position-blind (permutation-
invariant over its window), which is exactly what the induction prev-token head
cannot be. RoPE also extrapolates past the training length, so a model trained at
S=1024 is scored fairly on a 4096-token needle. Applied by absolute position, so a
cached key keeps its rotation as the window slides and decode still matches the
parallel forward.

Trunk: N layers of [RMSNorm -> mixer -> +res, RMSNorm -> SwiGLU -> +res], where
the mixer is windowed attention on layers where ``i % attn_every == 0`` and MinGRU
elsewhere. Both mixers share the (x -> same-shape) forward and the
``step(x_t, state) -> (out, new_state)`` decode contract, so the trunk loops over
them uniformly. Conforms to ``cubbyllm.model.backbone.Backbone``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...core.protocols import Wiring
from .mingru import _MinGRUMixer, _RMSNorm, _SwiGLU


def _rope_tables(dh: int, positions, device):
    """cos/sin for rotary position encoding at the given absolute positions.
    positions: (P,) long -> returns (P, dh) cos and sin."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.outer(positions.float(), inv)            # (P, dh/2)
    emb = torch.cat([ang, ang], dim=-1)                  # (P, dh)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x, cos, sin):
    """x: (B, h, S, dh); cos/sin broadcastable to it. Rotates by absolute
    position, so a q·k dot depends only on the RELATIVE offset — which is what
    lets the windowed cache reuse keys rotated once, at their own position, and
    still match the parallel forward."""
    return x * cos + _rotate_half(x) * sin


class _WindowedAttnMixer(nn.Module):
    """Causal multi-head attention restricted to the last ``window`` tokens.

    cubby-lm's ``LocalCausalAttention`` (attention.py). ``forward`` uses a
    sliding-window causal mask; ``step`` carries a KV cache trimmed to the last
    ``window`` entries, so its state is O(window) and independent of context
    length. The two must agree to fp tolerance — the same requirement MinGRU's
    step has, for the same reason (a throughput benchmark must measure the trained
    model, not a faster different one).
    """

    def __init__(self, d: int, heads: int = 4, window: int = 512):
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} not divisible by heads {heads}")
        if (d // heads) % 2:
            raise ValueError(f"head dim {d // heads} must be even for RoPE")
        self.h = int(heads)
        self.dh = d // heads
        self.window = int(window)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, S, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        cos, sin = _rope_tables(self.dh, torch.arange(S, device=x.device), x.device)
        cos, sin = cos.view(1, 1, S, self.dh), sin.view(1, 1, S, self.dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)      # rotary position
        i = torch.arange(S, device=x.device)
        keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)   # causal+window
        return self.o(o.transpose(1, 2).reshape(B, S, d))

    def step(self, x_t, cache=None):
        """One token. x_t: (B, d). cache: (k, v, pos) with k,v each
        (B, h, <=window, dh) and pos a 0-dim tensor (the next absolute position),
        or None to start.

        RoPE is applied at the token's ABSOLUTE position before caching, so a
        retained key keeps the rotation it had in ``forward`` even after the window
        slides — which is why the query (rotated at the current position) dotted
        against the cached keys reproduces the masked parallel attention exactly.
        State is bounded by ``window`` (plus one scalar); it never grows with
        context length.
        """
        B, d = x_t.shape
        if cache is None:
            kc = vc = None
            pos = 0
        else:
            kc, vc, pos_t = cache
            pos = int(pos_t)
        q, k, v = self.qkv(x_t).chunk(3, -1)
        q = q.view(B, self.h, 1, self.dh)
        k = k.view(B, self.h, 1, self.dh)
        v = v.view(B, self.h, 1, self.dh)
        cos, sin = _rope_tables(self.dh, torch.tensor([pos], device=x_t.device),
                                x_t.device)
        cos, sin = cos.view(1, 1, 1, self.dh), sin.view(1, 1, 1, self.dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if kc is not None:
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)
        if k.shape[2] > self.window:                     # keep only the last window
            k, v = k[:, :, -self.window:], v[:, :, -self.window:]
        # every cached key is a valid (past-or-current, within-window) attendee,
        # so no mask is needed — the trim already enforces the window.
        o = F.scaled_dot_product_attention(q, k, v)
        new_pos = torch.tensor(pos + 1, device=x_t.device)
        return self.o(o.reshape(B, d)), (k, v, new_pos)


class HybridBackbone(nn.Module):
    """MinGRU + SwiGLU trunk with windowed attention every ``attn_every``-th layer.

    forward: (B, S, d_model) -> (B, S, d_model). Conforms to ``Backbone``.
    ``attn_every=3`` reproduces cubby-lm's interleave (layers 0, 3, 6, ...).
    """

    def __init__(self, d_model: int, n_layers: int = 2, attn_every: int = 3,
                 window: int = 512, heads: int = 4, grad_checkpoint: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.attn_every = int(attn_every)
        self.window = int(window)
        self.is_attn = [i % self.attn_every == 0 for i in range(n_layers)]
        self.n1 = nn.ModuleList([_RMSNorm(d_model) for _ in range(n_layers)])
        self.mix = nn.ModuleList([
            _WindowedAttnMixer(d_model, heads, window) if a else _MinGRUMixer(d_model)
            for a in self.is_attn
        ])
        self.n2 = nn.ModuleList([_RMSNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([_SwiGLU(d_model) for _ in range(n_layers)])
        self.grad_checkpoint = bool(grad_checkpoint)

    @property
    def n_attn_layers(self) -> int:
        return sum(self.is_attn)

    @staticmethod
    def _layer(x, n1, m, n2, f):
        x = x + m(n1(x))
        x = x + f(n2(x))
        return x

    def step(self, x_t, states=None):
        """Decode one token. x_t: (B, d_model). states: per-layer state list or
        None. Returns (y_t, new_states).

        State per recurrent layer is (B, d); per attention layer it is a KV cache
        (B, h, <=window, dh) — bounded by ``window``, so the TOTAL carried state
        stops growing once context exceeds the window, unlike a full KV cache.
        Both mixers expose ``step(x_t, state) -> (out, new_state)``, so the loop is
        uniform. Inference-only: never grad-checkpointed, runs under no_grad.
        """
        n = len(self.mix)
        states = [None] * n if states is None else states
        new_states = []
        for i in range(n):
            h, s = self.mix[i].step(self.n1[i](x_t), states[i])
            x_t = x_t + h
            x_t = x_t + self.ffn[i](self.n2[i](x_t))
            new_states.append(s)
        return x_t, new_states

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        for n1, m, n2, f in zip(self.n1, self.mix, self.n2, self.ffn):
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer, x, n1, m, n2, f, use_reentrant=False)
            else:
                x = self._layer(x, n1, m, n2, f)
        return x


__wiring__ = Wiring.WIRED
