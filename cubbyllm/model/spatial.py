"""SpatialLink -- the trunk's route to space: places bound into its input, the scene as the
running sum of what it has been shown, and a gated read of that scene from inside the trunk,
so the trunk itself decides when to use it.

Wired: STANDALONE until a base run trains with ``CB_SPATIAL=1`` (2026-10-17, Nick: "spatial
understanding provably at the input level"; "it also needs to have a link route from the trunk
so it knows when to use it").

Three parts, one phase table (``cubbyllm.spatial.ssp._phases``, so the host's ``Scene`` and the
trunk agree on what ``S(p)`` is):

**bind** (the input level). A token that has a place ``p`` enters the trunk as
``e (*) S(p)`` -- its embedding circularly convolved with the place's unit vector. Provable,
and ``tests/model/test_spatial.py`` holds each to float precision: a token with no place is
unchanged (``S(0)`` is the identity; a batch with no places at all is returned as is, so
every text golden is bit-identical with the link on); moving a token is binding
(``bind(e, p + d) = bind(e, p) (*) S(d)``); and the place can be read back from the bound
embedding by unbinding with ``e``. RoPE is this same encoding in one dimension applied inside
attention; this is its 3-D sibling at the input, applied to every mixer the trunk has.

**scene**. ``V_t = sum_{i <= t, placed_i} e_i (*) S(p_i)`` -- the running sum of the placed
tokens' bound embeddings, so the scene the trunk may read at position ``t`` is exactly what
it has been shown so far (causal, like the context's running mean; one accumulator per stream
at decode). No codes but the model's own embeddings: ``V (*) S(p)^-1`` is the embedding of
the token at ``p`` plus the others' crosstalk, so what the read brings back is already in the
trunk's vocabulary.

**read** (the route from the trunk). Mid-trunk (layer ``at``, L/2 by default, so the layers
after it can decode what comes back), from the stream ``h``: a gate ``g = sigmoid(w . rms(h)
+ b)`` says *whether* to consult the scene, a query ``q = W rms(h)`` made unitary (its
spectrum reduced to its phases, so it *is* a place, ``S(p)`` for the ``p`` the trunk means)
says *what to ask*, and the answer ``V (*) q^-1`` comes back through ``o``. ``o`` starts at
zero and the gate's bias at -4, so a model with the link added computes what it computed
before (growth's rule: exact at step 0); with nothing placed the scene is zero and so is the
read. What the gate and the query learn is the route; the lookup they route to is algebra.

**query="placed"** (2026-10-20, H-K10): a placed token asks about its own place -- its query is
``S(p)`` from the input, exact, no projection in the way; unplaced tokens keep the learned
query. A loci token carries its address, so ``here`` means *here* by construction (loci tokens
were called hypertokens until 2026-10-22; see H-K10's Name in CUBBYLLM_HYPOTHESES.md). Measured
on a tiny trunk (``validation/exp_k10_hypertoken.py``): a fact shown once, asked 120 tokens past
the window, answered at 0.94 this way and at 0.00 through the learned query alone -- the read
is easy to learn, recovering an address from the hidden state is the hard part.

**bind="writes"** (2026-10-20, the fact link's second 450M run): with roles given, only the
tokens that WRITE are bound at the input; a token that only ASKS keeps its own embedding and
asks with ``S(p)``. Binding every placed token scrambled the far facts' answer pieces -- each
one, asking for the next piece, entered the trunk as ``e (*) S(p_next)``, so the trunk lost
which token it had just read: next-piece accuracy 0.096 with the places against 0.213 without
(``validation/diag_fact_read.py``). The default ``"placed"`` keeps every earlier run exact.

**copy** (2026-10-21, order 0001, B1; ``meta["spatial"]["copy"]``, absent = None, nothing built):
at an asking token the unbound read is also cleaned up against the tokens this document wrote
(``"ring"``, the last 256) or the whole table (``"table"``) and mixed into the next-token
distribution (``model/spatial_copy.py``) -- the cleanup the far facts' read already supports
(0.992 a piece over the document's writers, 0.758 over the vocabulary) handed to the output
directly instead of through 16 layers. ``add_copy`` is the call; off, nothing changes.

What it brings CubbyLLM: a trunk that can be told where every block is without a word of
text, and ask the scene "what is here" at a place it chose, with the answer's correctness
guaranteed by the encoding rather than hoped for from training. A conventional model has no
place to put a coordinate but its tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.protocols import Wiring
from ..spatial.ssp import SSP, _phases

__all__ = ["SpatialLink", "NO_PLACE"]

#: the coordinate a token without a place carries in a shard's ``.pos.i16`` side file
NO_PLACE = -32768


class SpatialLink(nn.Module):
    def __init__(self, d_model: int, scale: float = 2.0, seed: int = 0, dims: int = 3,
                 at: int = 0, gate_bias: float = -4.0, query: str = "learned", bind: str = "placed",
                 copy: str | None = None, copy_tau: float = 3.0, copy_ring: int = 256,
                 copy_beta: float | None = None) -> None:
        super().__init__()
        if query not in ("learned", "placed"):
            raise ValueError(f"query must be 'learned' or 'placed', not {query!r}")
        if bind not in ("placed", "writes"):
            raise ValueError(f"bind must be 'placed' or 'writes', not {bind!r}")
        self.bind_mode = bind
        self.d, self.dims, self.scale, self.seed, self.at = int(d_model), int(dims), float(scale), int(seed), int(at)
        self.query = query
        self.register_buffer("phases", torch.tensor(_phases(self.d, self.dims, self.seed), dtype=torch.float32))
        self.norm_w = nn.Parameter(torch.ones(self.d))
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.gate = nn.Linear(self.d, 1)
        self.o = nn.Linear(self.d, self.d, bias=False)
        with torch.no_grad():
            self.o.weight.zero_()
            self.gate.bias.fill_(float(gate_bias))
        self.copy_path = None                    # registered last: its tensors follow the link's in parameters()
        if copy is not None:
            from .spatial_copy import SpatialCopy
            self.copy_path = SpatialCopy(self.d, copy, tau=copy_tau, ring=copy_ring, beta=copy_beta)

    def space(self) -> SSP:
        """The host's ``SSP`` in this link's space: same phases, so ``Scene`` vectors read here."""
        return SSP(self.d, self.dims, self.scale, self.seed)

    # -- the encoding, in torch ------------------------------------------------------------
    def spectrum(self, p: torch.Tensor) -> torch.Tensor:
        """``exp(i Theta p / scale)`` for places ``p`` (..., dims) -> (..., d/2+1) complex.
        Always float32, autocast or not: a cell coordinate in the thousands times a phase is an
        angle of ~1e4 radians, which bf16 (8 bits of mantissa) would get wrong by tens of radians --
        and ``polar`` refuses bf16 anyway (the first CUDA run, 2026-10-20)."""
        dt = p.device.type
        if dt in ("cuda", "cpu"):
            with torch.autocast(dt, enabled=False):
                ang = p.float() @ self.phases.T.float() / self.scale
        else:
            ang = p.float() @ self.phases.T.float() / self.scale
        return torch.polar(torch.ones_like(ang), ang)

    def encode(self, p: torch.Tensor) -> torch.Tensor:
        """``S(p)``, (..., d) float32 unit vectors."""
        return torch.fft.irfft(self.spectrum(p), n=self.d, dim=-1)

    # -- bind: the input level ------------------------------------------------------------
    def bind(self, e: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """``e (*) S(positions)``; ``positions`` (..., dims) matching ``e``'s leading shape,
        or None for a batch without places, returned untouched."""
        if positions is None:
            return e
        spec = torch.fft.rfft(e.float(), dim=-1) * self.spectrum(positions)
        return torch.fft.irfft(spec, n=self.d, dim=-1).to(e.dtype)

    def bind_input(self, e: torch.Tensor, positions: torch.Tensor | None, writes: torch.Tensor | None = None) -> torch.Tensor:
        """The input level: every placed token bound (``bind="placed"``), or with ``bind="writes"``
        and roles given only the writing ones -- an asking token's place goes to zero, ``S(0)`` the
        identity, so it enters as itself."""
        if positions is not None and writes is not None and self.bind_mode == "writes":
            positions = positions * writes.unsqueeze(-1).to(positions.dtype)
        return self.bind(e, positions)

    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``a (*) b^-1`` with ``b``'s exact inverse (its conjugate spectrum; exact for a
        unitary ``b`` such as ``S(p)``)."""
        spec = torch.fft.rfft(a.float(), dim=-1) * torch.conj(torch.fft.rfft(b.float(), dim=-1))
        return torch.fft.irfft(spec, n=self.d, dim=-1).to(a.dtype)

    # -- scene: what the trunk has been shown ---------------------------------------------
    @staticmethod
    def scene(x: torch.Tensor, placed: torch.Tensor | None, starts: torch.Tensor | None = None) -> torch.Tensor | None:
        """The running sum of the writing tokens' bound embeddings ``x`` (B, S, d): ``V_t``
        (B, S, d), causal; None when nothing is placed. ``starts`` (B, S) bool marks the tokens
        where a document begins: the sum restarts there, so a document reads only itself (one
        scene per document -- two documents in a window would otherwise share one vector, past
        its capacity; the fact link's first run, 2026-10-20)."""
        if placed is None:
            return None
        c = (x.float() * placed.unsqueeze(-1).float()).cumsum(1)
        if starts is None or not bool(starts.any()):
            return c
        S = x.shape[1]
        i = torch.arange(S, device=x.device).expand_as(starts)
        seg = torch.where(starts, i, torch.zeros_like(i)).cummax(1).values          # this token's document start
        cpad = torch.cat([torch.zeros_like(c[:, :1]), c], 1)                         # cpad[s] = sum before s
        return c - cpad.gather(1, seg.unsqueeze(-1).expand(-1, -1, c.shape[-1]))

    # -- read: the route from the trunk ---------------------------------------------------
    @staticmethod
    def unitary(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """``q`` with every spectral component brought to unit modulus: a place, whatever
        the trunk's magnitudes were."""
        spec = torch.fft.rfft(q.float(), dim=-1)
        return torch.fft.irfft(spec / (spec.abs() + eps), n=q.shape[-1], dim=-1)

    def lookup(self, scene: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """The algebra alone: ``scene (*) unitary(q)^-1`` -- what the scene holds at the place
        ``q`` names, (..., d)."""
        return self.unbind(scene, self.unitary(q))

    def read(self, h: torch.Tensor, scene: torch.Tensor | None, own=None) -> torch.Tensor:
        """The gated read of ``scene`` -- (B, S, d) running, or (B, d) at a decode step --
        from the trunk stream ``h`` of the same leading shape; zero when there is no scene.
        ``own`` is ``(S(p), placed)`` for ``query="placed"``: a placed token's query is its
        own place, the others' the learned one."""
        if scene is None:
            return torch.zeros_like(h)
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm_w
        g = torch.sigmoid(self.gate(n))                                  # (..., 1): whether
        q = self.q(n)                                                    # (..., d): what place
        if own is not None:                                              # S(p) stays float32: no bf16 round trip
            q = torch.where(own[1].unsqueeze(-1), own[0].float(), q.float())
        return g * self.o(self.lookup(scene.float(), q).to(h.dtype))

    def hook(self, scene: torch.Tensor | None, positions: torch.Tensor | None = None,
             placed: torch.Tensor | None = None):
        """``(layer, fn)`` for the backbone: the read at layer ``at`` over this scene;
        ``positions`` / ``placed`` (as ``bind`` / ``scene`` take them) feed ``query="placed"``."""
        own = None
        if self.query == "placed" and positions is not None and placed is not None:
            own = (self.encode(positions), placed)
        return (self.at, lambda h: self.read(h, scene, own))

    # -- copy: the read, cleaned up into the output (model/spatial_copy.py) -----------------
    def add_copy(self, logits: torch.Tensor, h: torch.Tensor, scene: torch.Tensor, positions: torch.Tensor,
                 asks: torch.Tensor, table: torch.Tensor, cands=None) -> torch.Tensor:
        """``logits`` (B, S, V) with the copy mixed in at the asking tokens (their rows become log
        p), every other row untouched: ``h`` (B, S, d) the final stream, ``scene`` (B, S, d) as the
        read saw it, ``positions`` (B, S, dims), ``asks`` (B, S), ``table`` (V, d) the embedding table
        the scene was written from. Mode "ring" also takes ``cands``: ``(tokens, writes, starts)`` in
        a forward, the decode state (its ring) in a step."""
        cp, V, d = self.copy_path, logits.shape[-1], h.shape[-1]
        idx, ids, ok = asks.reshape(-1).nonzero().squeeze(1), None, None
        if cp.mode == "ring" and idx.numel():
            from .spatial_copy import ring_candidates
            ids, ok = ring_candidates(*cands, idx, cp.ring) if isinstance(cands, tuple) else cp.candidates(cands, idx)
            keep = ok.any(1)                     # nothing written yet: the trunk's logits alone
            idx, ok, ids = idx[keep], ok[keep], ids if ids.dim() == 1 else ids[keep]
        if idx.numel() == 0:
            return logits
        flat = logits.reshape(-1, V)
        r = self.lookup(scene.reshape(-1, d)[idx].float(), self.encode(positions.reshape(-1, self.dims)[idx]))
        mix = cp(flat[idx], h.reshape(-1, d)[idx], r, table, idx, ids, ok)[0]
        return flat.index_copy(0, idx, mix.to(logits.dtype)).reshape(logits.shape)

    def extra_repr(self) -> str:
        return f"d={self.d}, dims={self.dims}, scale={self.scale}, seed={self.seed}, at={self.at}, query={self.query}"


__wiring__ = Wiring.STANDALONE
