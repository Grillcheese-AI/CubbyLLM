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

EPISODIC MEMORY (rung 0.0.5, opt-in via ``mem_every``). The honest limit above —
recall degrades past ``window`` — is exactly what ``cubbyllm.model.recall``
(``MemoryRead`` + ``EpisodicStore``) targets: a per-position kNN read over the
BEYOND-window causal past, interleaved after the mixer residual on layers where
``i % mem_every == 0``. ``mem_every=0`` (the default) constructs no ``MemoryRead``
modules at all, so the default path is unchanged until a real run opts in. In
``forward`` the read is masked and dense (attends over the whole beyond-window
past each call, training-time cost); in ``step`` it is backed by a persisted,
per-layer ``EpisodicStore`` plus a small FIFO buffer that ages a token into the
store only once it falls off the back of the last ``window`` steps — which is
what makes the store's causal contents agree with ``forward``'s ``j < i - window``
mask at every step, and is what the incremental-decode equivalence test checks.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...core.protocols import Wiring
from ..recall import EpisodicStore, MemoryRead
from ..recall.store import DEFAULT_CAPACITY
from .mingru import _MinGRUMixer, _RMSNorm, _SwiGLU

try:  # torch >= 2.5; used on CUDA where it does only the banded work
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - older torch (e.g. the .venv-dml 2.4.1)
    create_block_mask = flex_attention = None


@functools.lru_cache(maxsize=8)
def _window_block_mask(S: int, window: int, device: str):
    """Sliding-window causal BlockMask, cached per (S, window, device).

    Why this exists: handing SDPA a dense (S, S) bool mask forces the
    arbitrary-mask fallback, which computes FULL quadratic attention and then
    masks — measured at 2B scale as MFU 40.3% (S=1024) collapsing to 21.8%
    (S=4096) at identical B*S (exp_t1_mfu_pilot_a100_2b_s4096.log). FlexAttention
    skips the masked-out blocks entirely, restoring the linear-in-S cost the
    windowed design promises. The cache matters: building a BlockMask traces the
    mask_mod, which is far too slow to redo every forward.
    """
    def keep(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return create_block_mask(keep, B=None, H=None, Q_LEN=S, KV_LEN=S,
                             device=device)


@functools.lru_cache(maxsize=8)
def _rope_inv_freq(dh: int, device: str):
    """RoPE's inverse frequencies, per (head_dim, device). A constant: cached
    because ``step`` calls it once per attention layer per token, and building
    it there is two kernels and a host upload for a table that never changes."""
    return 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=device).float() / dh))


def _rope_tables(dh: int, positions, device):
    """cos/sin for rotary position encoding at the given absolute positions.
    positions: (P,) long -> returns (P, dh) cos and sin."""
    inv = _rope_inv_freq(dh, str(device))
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
        if self._use_flex(x):
            bm = _window_block_mask(S, self.window, str(x.device))
            o = flex_attention(q, k, v, block_mask=bm)   # banded work only
        else:
            i = torch.arange(S, device=x.device)
            keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)  # causal+window
        return self.o(o.transpose(1, 2).reshape(B, S, d))

    @staticmethod
    def _use_flex(x) -> bool:
        """flex on CUDA when available; the dense-mask SDPA path stays the
        reference (CPU tests, DirectML, old torch). CB_NO_FLEX=1 forces the
        reference path everywhere (A/B and escape hatch)."""
        import os
        if flex_attention is None or os.environ.get("CB_NO_FLEX") == "1":
            return False
        return x.is_cuda

    def step(self, x_t, cache=None):
        """One token. x_t: (B, d). cache: what ``init_state`` returns —
        two preallocated ``(B, h, window, dh)`` rings, a float32 position,
        an int64 ring slot, an additive liveness mask and a preallocated
        one — or None to start.

        RoPE is applied at the token's ABSOLUTE position before caching, so a
        retained key keeps the rotation it had in ``forward`` even after the window
        slides — which is why the query (rotated at the current position) dotted
        against the cached keys reproduces the masked parallel attention exactly.

        **The cache is a ring, not a list that grows into a window.** The
        old version concatenated the new key onto the cache and trimmed to
        the last ``window``, so the state's *shape* changed on every one of
        the first ``window`` tokens. Bounded state is still bounded that
        way, but a step whose shapes change every token can never be
        recorded and replayed — each token is a new graph signature, so a
        capture is taken and thrown away ``window`` times and the decode
        loop pays the Python every token anyway. Here the rings are full
        size from token 0, the new key is written in place at
        ``pos % window``, and a validity mask hides the slots not written
        yet. State size is constant **from token 1**, which is the property
        capture needs and the one the old trim did not have.

        The ring attends over its slots in ring order rather than in time
        order. Attention is a softmax over the keys and a sum over the
        values weighted by it, so permuting the (key, value) pairs together
        permutes nothing about the result but the order the sum is
        accumulated in — equal to the old path within float32
        reassociation, which is what ``test_hybrid.py`` asserts.

        The position stays a **tensor** the whole way through: it is rotated
        into RoPE by device ops and advanced in place by ``add_(1.0)``,
        never read back with ``int()``. The ring slot is a second, integer
        counter advanced beside it rather than re-derived from the float
        position (see ``init_state``).
        Reading it would make the step's work depend on a host value, which
        is what stops a step being recorded once and replayed (grilly's
        ``graphed``, torch's CUDA graphs). It is float32, exact to 2**24
        positions, because an int64 scalar add has to build its operand on
        the host.
        """
        B, d = x_t.shape
        W = self.window
        state = self.init_state(B, x_t.device, x_t.dtype) if cache is None else cache
        k_ring, v_ring, pos, slot, live, one = state
        q, k, v = self.qkv(x_t).chunk(3, -1)
        q = q.view(B, self.h, 1, self.dh)
        k = k.view(B, self.h, 1, self.dh)
        v = v.view(B, self.h, 1, self.dh)
        cos, sin = _rope_tables(self.dh, pos.reshape(1), x_t.device)
        cos, sin = cos.view(1, 1, 1, self.dh), sin.view(1, 1, 1, self.dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k_ring.index_copy_(2, slot, k)
        v_ring.index_copy_(2, slot, v)
        live.index_fill_(0, slot, 0.0)          # this slot now holds a real key
        o = F.scaled_dot_product_attention(q, k_ring, v_ring,
                                           attn_mask=live.view(1, 1, 1, W))
        slot.copy_(torch.remainder(slot + one, W))
        pos.add_(1.0)
        return self.o(o.reshape(B, d)), state

    def init_state(self, batch: int, device=None, dtype=None):
        """Every buffer one decode sequence needs, allocated once.

        ``step`` allocates nothing after this, which is what lets a single
        recorded step be replayed across calls: an allocation inside a
        capture pins a buffer the next capture would have to make again,
        and `torch.randn` inside one is a host write the replay cannot
        reproduce at all.

        The slot is an **int64 counter advanced by a device add**, not
        ``pos % window`` recomputed from the float position every token.
        Both give the same number; the float route cost 20 dispatches
        (a divide, a floor, a multiply, a subtract, two sign repairs and a
        cast) against three. ``one`` is preallocated for the same reason a
        capture refuses ``torch.tensor(1)``: an int64 scalar is built on
        the host.

        Liveness is an additive mask — ``0`` for a slot written, ``-inf``
        for one not — rather than ``arange(window) < pos + 1`` rebuilt per
        token."""
        z = dict(device=device)
        k_ring = torch.zeros(batch, self.h, self.window, self.dh, dtype=dtype, **z)
        return (k_ring, torch.zeros_like(k_ring),
                torch.zeros((), **z),
                torch.zeros(1, dtype=torch.int64, **z),
                torch.full((self.window,), float("-inf"), **z),
                torch.ones(1, dtype=torch.int64, **z))


class HybridBackbone(nn.Module):
    """MinGRU + SwiGLU trunk with windowed attention every ``attn_every``-th layer.

    forward: (B, S, d_model) -> (B, S, d_model). Conforms to ``Backbone``.
    ``attn_every=3`` reproduces cubby-lm's interleave (layers 0, 3, 6, ...).
    ``mem_every>0`` additionally interleaves an episodic ``MemoryRead`` on
    layers where ``i % mem_every == 0`` (default 0: memory off, see module
    docstring).
    """

    def __init__(self, d_model: int, n_layers: int = 2, attn_every: int = 3,
                 window: int = 512, heads: int = 4, grad_checkpoint: bool = False,
                 mem_every: int = 0, mem_topk: int = 8, mem_key: int = 64,
                 mem_capacity: int = DEFAULT_CAPACITY):
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

        # Episodic memory: reads the BEYOND-window past (rung 0.0.5). mem_every=0
        # disables it entirely (no MemoryRead modules constructed) so the default
        # path stays behaviour-identical until a real run turns this on.
        self.mem_every = int(mem_every)
        self.mem_topk = int(mem_topk)
        self.mem_key = int(mem_key)
        self.mem_capacity = int(mem_capacity)
        self.is_mem = [bool(self.mem_every) and (i % self.mem_every == 0)
                       for i in range(n_layers)]
        self.mem_n = nn.ModuleList([_RMSNorm(d_model) if a else nn.Identity()
                                    for a in self.is_mem])
        self.mem = nn.ModuleList([MemoryRead(d_model, d_key=mem_key, topk=mem_topk)
                                  if a else nn.Identity() for a in self.is_mem])

    @property
    def n_attn_layers(self) -> int:
        return sum(self.is_attn)

    def _layer(self, idx: int, x):
        """One layer: mixer residual, then (memory read if this is a memory
        layer), then the FFN residual. An instance method (not the old
        staticmethod) because the memory branch needs ``self.is_mem``/``self.mem``
        — passed whole to ``torch.utils.checkpoint.checkpoint`` in ``forward`` so
        the memory branch's params also get gradients under grad_checkpoint."""
        n1, m, n2, f = self.n1[idx], self.mix[idx], self.n2[idx], self.ffn[idx]
        x = x + m(n1(x))
        if self.is_mem[idx]:
            x = x + self.mem[idx](self.mem_n[idx](x), self.window)
        x = x + f(n2(x))
        return x

    def step(self, x_t, states=None):
        """Decode one token. x_t: (B, d_model). states: per-layer state, or None.
        Returns (y_t, new_states).

        With ``mem_every=0`` (the default) this is byte-for-byte the original
        behaviour: ``states`` is a plain per-layer list. With ``mem_every>0``,
        ``states`` becomes ``{"mix": [...], "mem": {layer_idx: {"store":
        EpisodicStore, "k_ring": ..., "v_ring": ..., "pos": ...}}}`` — the
        mixer states plus, per memory layer, a fixed-capacity store and a
        ``window``-slot ring of the (k, v) pairs not yet written into it.

        Causal contract (must match ``forward``'s ``j < i - window`` mask
        exactly): the ring holds the ``window`` most recent tokens' (k, v) —
        exactly the ones ``forward`` excludes as "inside the window" — and only
        the token falling OFF the back of it (now older than ``window``
        steps) gets written into the store. So at step t, the store the read
        queries holds exactly tokens ``0 .. t-window-1``: the same beyond-window
        candidate set ``forward`` computes for query i=t. Retrieval is by
        ``EpisodicStore.retrieve_cosine`` — the same selection metric
        ``MemoryRead.forward`` now uses — so the two paths pick the same top-K.

        **The ring is the FIFO.** The old version kept a Python list and
        popped its head once ``len(buf) > window``, which is a host branch
        on a host length — unrecordable, and a new graph signature on every
        one of the first ``window`` tokens besides. A ring of exactly
        ``window`` slots needs neither: the entry sitting at ``pos %
        window`` *is* the token ``window`` steps old, so reading that slot
        before overwriting it with the current token ages exactly one
        token per step, with no length, no branch and no growth.

        State per recurrent layer is (B, d); per attention layer it is a KV
        ring (B, h, window, dh); per memory layer a (window, d_key) and a
        (window, d_model) ring plus a 4-byte position. All of it is
        constant in size **from the first token**, and the store is
        constant too — which is what lets the whole step be recorded once
        and replayed (grilly2's ``docs/capture.md``). Both mixers expose
        ``step(x_t, state) -> (out, new_state)``, so the mixer loop is
        uniform. Inference-only: never grad-checkpointed, runs under
        no_grad.

        What the fixed store costs, stated rather than buried: beyond
        ``mem_capacity`` tokens of aged-out history the oldest entry is
        overwritten, where the old store kept everything. H-A8 in
        CUBBYLLM_HYPOTHESES.md carries the claim and the kill criterion.

        v1 scope: memory-enabled decode (``mem_every>0``) is single-sequence
        (B=1). Each memory layer keeps ONE ``EpisodicStore``, not one per batch
        row — passing B>1 would interleave different sequences' (k, v) into the
        same store. Matches eval decode (``exp_needle_recall``), which is B=1;
        a batched store is future work, not silently handled here.
        """
        n = len(self.mix)
        if self.mem_every <= 0:
            states = self.init_state(x_t.shape[0], x_t.device, x_t.dtype) if states is None else states
            new_states = []
            for i in range(n):
                h, s = self.mix[i].step(self.n1[i](x_t), states[i])
                x_t = x_t + h
                x_t = x_t + self.ffn[i](self.n2[i](x_t))
                new_states.append(s)
            return x_t, new_states

        assert x_t.shape[0] == 1, "memory-enabled decode (mem_every>0) supports batch size 1 only"
        if states is None:
            states = self.init_state(x_t.shape[0], x_t.device, x_t.dtype)
        mixs, mems = states["mix"], states["mem"]
        new_mix = []
        for i in range(n):
            h, s = self.mix[i].step(self.n1[i](x_t), mixs[i])
            x_t = x_t + h
            if self.is_mem[i]:
                x_t = self._memory_step(i, x_t, mems[i])
            x_t = x_t + self.ffn[i](self.n2[i](x_t))
            new_mix.append(s)
        return x_t, {"mix": new_mix, "mem": mems}

    def init_state(self, batch: int = 1, device=None, dtype=None):
        """Every decode buffer this backbone needs, allocated once.

        ``step`` allocates nothing once it has one. That is what makes a
        recorded step reusable across calls: the first step used to build
        the episodic store — `torch.randn` for the SimHash projection and
        all — and a capture cannot contain a host write, so a graphed step
        held over from an earlier sequence failed at the capture rather
        than at the call that caused it."""
        mix = [m.init_state(batch, device, dtype) if hasattr(m, "init_state") else None
               for m in self.mix]
        if self.mem_every <= 0:
            return mix
        return {"mix": mix,
                "mem": {i: self._new_mem_state(device) for i in range(len(self.mix))
                        if self.is_mem[i]}}

    def _new_mem_state(self, device) -> dict:
        """One memory layer's decode state.

        The store, the window-length ring that ages into it, and three
        counters: the float position (RoPE and the causal rule read it),
        the int64 ring slot, and the int64 store row. The last two are
        advanced by device adds rather than recomputed from the position —
        ``pos % window`` and ``clamp(pos - window, 0) % capacity`` in float
        cost 42 dispatches a token between them, against about eight."""
        return {
            "store": EpisodicStore(self.mem_key, self.d_model,
                                   capacity=self.mem_capacity, hamming=False,
                                   device=device),
            "k_ring": torch.zeros(self.window, self.mem_key, device=device),
            "v_ring": torch.zeros(self.window, self.d_model, device=device),
            "pos": torch.zeros((), device=device),
            "slot": torch.zeros(1, dtype=torch.int64, device=device),
            "aged": torch.zeros(1, dtype=torch.int64, device=device),
            "one": torch.ones(1, dtype=torch.int64, device=device),
            "zero": torch.zeros(1, device=device),
            "dead": torch.full((1,), float("-inf"), device=device),
        }

    def _memory_step(self, i: int, x_t, st: dict):
        """Read the beyond-window past, then age one token into the store.

        Order is the causal contract: the read must not see the token this
        step is about to write, because ``forward``'s mask does not
        (``j < i - window``). Every index here is a device tensor, so the
        whole thing records and replays."""
        store, k_ring, v_ring = st["store"], st["k_ring"], st["v_ring"]
        pos, slot, aged = st["pos"], st["slot"], st["aged"]
        one, zero, dead, W = st["one"], st["zero"], st["dead"], self.window
        q, k, v = self.mem[i].qkv(self.mem_n[i](x_t))            # (B,dk),(B,dk),(B,d)

        k_top, v_top, ok = store.retrieve_cosine(q[0], self.mem_topk, return_valid=True)
        r = self.mem[i].read(q.unsqueeze(1),                     # (B,1,dk)
                             k_top.unsqueeze(0).unsqueeze(0),    # (1,1,K,dk)
                             v_top.unsqueeze(0).unsqueeze(0),    # (1,1,K,d)
                             valid=ok.view(1, 1, -1))
        x_t = x_t + r[:, 0]        # exactly zero until the store has an entry

        # One predicate decides both things this step does differently
        # before the window has filled: whether the entry it ages into the
        # store is real, and whether the store row advances. Below the
        # window every step writes the ring's zeros into row 0 and marks
        # that row dead, so the command stream is the same shape from
        # token 1 and nothing downstream ever sees the row.
        aged_is_real = pos >= float(W)
        store.write_at(aged,
                       k_ring.index_select(0, slot), v_ring.index_select(0, slot),
                       torch.where(aged_is_real, zero, dead))
        k_ring.index_copy_(0, slot, k)
        v_ring.index_copy_(0, slot, v)
        slot.copy_(torch.remainder(slot + one, W))
        aged.copy_(torch.remainder(aged + aged_is_real.long(), store.capacity))
        pos.add_(1.0)
        return x_t

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        for idx in range(len(self.mix)):
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer, idx, x, use_reentrant=False)
            else:
                x = self._layer(idx, x)
        return x


__wiring__ = Wiring.WIRED
