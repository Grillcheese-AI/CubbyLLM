"""SpatialCopy -- the spatial link's copy path: what the scene holds at an asked address,
cleaned up against the tokens it could be and mixed into the next-token distribution (order 0001,
B1 over the ring, Addendum 3; H-K10).

Wired: STANDALONE until a base run trains with ``meta["spatial"]["copy"]`` (``SpatialLink(copy=...)``
owns one, ``CubbyModel.forward`` / ``step`` call it through ``SpatialLink.add_copy``).

Why. At an asking token the link's read ``r = V (*) S(p)^-1`` is the embedding of the token
written at ``p`` plus crosstalk; the trunk must carry it through ``o`` and the layers after the
read into an untied head. On the 450M's far facts it does not (0.021 exact), although the read
holds the answer: cleaned up against the table, the stored piece is the argmax over the 128k
vocabulary for 0.758 of the pieces and over the document's own writers for 0.992, 0.965 whole
answers (``validation/diag_fact_cleanup.py``, ``logs/diag_fact_cleanup_run3_2026-10-21.json``). So
the cleanup goes to the output directly, as the copy paths that work do (pointer-sentinel, Merity et
al. 2016; pointer-generator, See et al. 2017; kNN-LM): at an asking token

    p = (1 - lam) p_LM + lam p_copy,     lam = sigmoid(g . rms(h) + b),
    p_copy = softmax over the candidates c of exp(tau) cos(r, e_c), scattered onto their ids

and the logits there become ``log p`` (normalized, so a cross-entropy over them is ``-log p(y)``);
every other position keeps the trunk's logits bit for bit. The candidates:

  "ring"  -- the last ``ring`` (256) token ids this document WROTE into the scene, a ring like the
             attention window's, so the decode state stays fixed-size (Nick's call, B9); in a
             forward, the writers at or before the position in the same document (reset at
             ``starts``), capped at the last ``ring``. Duplicate ids sum.
  "table" -- the whole vocabulary (B1 as first written; its ceiling on run 3 is 0.457 whole answers).

``e_c`` is the table row the scene was written from: ``HybridEmbedding.embed`` returns the core row
when there is no tail generator, as ``train_base.build_model`` builds it (with a tail the stored
vector also carries the context shift, and the cosine is to the core row). ``lam`` starts at 0.1
(``g = 0``; kNN-Adapter's open start) -- it acts only at asking tokens, and its gradient is
``(p_copy(y) - p_LM(y)) / p(y)``, so it opens as soon as the copy is right more often than the
trunk. ``tau`` starts where the caller puts it: so that A0's median gap between the target and
the best wrong writer is ~2 nats. An asking token with no candidate yet keeps the trunk's logits.

**The read-aware gate** (``beta``, ``meta["spatial"]["copy_beta"]``; absent = the gate above). The
gate above sees only the stream ``h``, which cannot tell whether the asked address holds anything:
the last piece of a name asks for the piece after it, an empty address, and the copy still offers
a written token. On the toy (``exp_k10_far_facts``, 5-piece names) the gate learned HERE (lam 0.996)
but parted a real next piece (0.72) from an empty address (0.54) only weakly, and the missed later
pieces were the trunk's EOS (198 of 261). With ``beta`` the gate also sees the match itself,

    lam = sigmoid(g . rms(h) + b + exp(tau) (m - beta)),   m = the best candidate's cosine

a nat of match margin a logit of gate, ``beta`` a learned threshold started between the present
and the absent reads' best cosines -- the pointer-sentinel's point (Merity et al. 2016: the gate is
decided with the pointer's scores), compared by the best match, not a softmax sum, so twenty weak
crosstalk candidates do not add up to one strong one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.protocols import Wiring

__all__ = ["SpatialCopy", "MODES", "ring_candidates"]

MODES = ("ring", "table")


class SpatialCopy(nn.Module):
    def __init__(self, d_model: int, mode: str = "ring", tau: float = 3.0, ring: int = 256, lam: float = 0.1,
                 beta: float | None = None, refuse: dict | None = None) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"copy must be one of {MODES}, not {mode!r}")
        self.mode, self.ring = mode, int(ring)
        self.gate = nn.Linear(d_model, 1)                         # lam = sigmoid(gate(rms(h)))
        self.tau = nn.Parameter(torch.tensor(float(tau)))         # the cleanup's log temperature
        self.beta = None if beta is None else nn.Parameter(torch.tensor(float(beta)))   # the read-aware gate's threshold
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(math.log(lam / (1.0 - lam)))
        self.refuse = None                                         # copy or refuse (model/spatial_refuse.py)
        if refuse is not None:
            self.set_refuse(**refuse)

    def set_refuse(self, theta: float, theta_end: float, refuse_id: int, step=(0, 0, 6), lookahead: int = 16) -> None:
        """Copy or refuse (order 0022; ``meta["spatial"]["copy_refuse"]``, absent = the mix above): a piece clears
        ``theta``, an end reads below ``theta_end``, anything between refuses the answer, said as ``refuse_id``."""
        if not theta_end <= theta:
            raise ValueError(f"theta_end ({theta_end}) must not exceed theta ({theta})")
        self.refuse = (float(theta), float(theta_end), tuple(step), int(refuse_id), int(lookahead))

    def forward(self, logits: torch.Tensor, h: torch.Tensor, r: torch.Tensor, table: torch.Tensor,
                idx: torch.Tensor, ids: torch.Tensor | None = None, ok: torch.Tensor | None = None):
        """At n asking positions: ``logits`` (n, V) the trunk's, ``h`` (n, d) the final stream, ``r``
        (n, d) the reads, ``table`` (V, d); ``idx`` (n,) their flat positions (unused here: a forward
        hook reads it to find the targets); mode "ring": ``ids`` (C,) or (n, C) candidate ids and
        ``ok`` (n, C) which of them count. -> ``(log p, log p_copy, lam)``: (n, V), (n, V), (n, 1)."""
        n = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        g = self.gate(n).float()
        rn, scale = F.normalize(r.float(), dim=-1), self.tau.exp()
        if self.mode == "table":
            s = rn @ F.normalize(table.float(), dim=-1).T
            log_copy, best = F.log_softmax(scale * s, -1), s.max(-1, keepdim=True).values
        else:
            e = F.normalize(table.float()[ids], dim=-1)            # (C, d) or (n, C, d)
            s = rn @ e.T if ids.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)
            best = s.masked_fill(~ok, float("-inf")).max(-1, keepdim=True).values
            pc = F.softmax((scale * s).masked_fill(~ok, float("-inf")), -1)   # masked after the scale: no 0 * inf
            c = torch.zeros(rn.shape[0], logits.shape[-1], device=rn.device).scatter_add(
                1, ids.expand_as(pc) if ids.dim() == 1 else ids, pc)
            log_copy = torch.log(c.clamp_min(1e-30))               # 0 off the candidates: no mass there
        if self.beta is not None:                                  # the read-aware gate: does anything match here?
            g = g + scale * (best - self.beta)
        mix = torch.logaddexp(F.logsigmoid(-g) + F.log_softmax(logits.float(), -1), F.logsigmoid(g) + log_copy)
        return mix, log_copy, torch.sigmoid(g)

    # -- the ring at decode: the ids this stream wrote, fixed-size ----------------------------
    def init_state(self, batch: int, device=None) -> dict:
        st = {}
        if self.refuse is not None:                                # the answer's decision, held across its run
            st.update(cr_prev=torch.zeros(batch, dtype=torch.bool, device=device),
                      cr_L=torch.zeros(batch, dtype=torch.long, device=device),
                      cr_off=torch.zeros(batch, dtype=torch.long, device=device))
        if self.mode != "ring":
            return st
        return {**st, "ring": torch.zeros(batch, self.ring, dtype=torch.long, device=device),
                "ring_n": torch.zeros(batch, dtype=torch.long, device=device)}

    def remember(self, state: dict, token: torch.Tensor, write: torch.Tensor | None, start: torch.Tensor | None) -> None:
        """One decode step's writes into the ring, in place (a start empties it first)."""
        if "ring" not in state:
            return
        ring, n = state["ring"], state["ring_n"]
        if start is not None:
            n.mul_((~start).long())
        if write is None:
            return
        slot = (n % self.ring).unsqueeze(1)
        ring.scatter_(1, slot, torch.where(write, token, ring.gather(1, slot)[:, 0]).unsqueeze(1))
        n.add_(write.long())

    def candidates(self, state: dict, rows: torch.Tensor):
        """The decode state's ring at batch rows ``rows``: ids (n, ring), ok (n, ring)."""
        ok = torch.arange(self.ring, device=rows.device)[None] < state["ring_n"][rows].clamp(max=self.ring)[:, None]
        return state["ring"][rows], ok

    def extra_repr(self) -> str:
        return f"mode={self.mode}, ring={self.ring}, read_aware={self.beta is not None}, refuse={self.refuse}"


def ring_candidates(tokens: torch.Tensor, writes: torch.Tensor, starts: torch.Tensor | None,
                    idx: torch.Tensor, ring: int):
    """The ring in a forward: ids (C,) -- every writing token in the batch -- and ok (n, C): for
    each flat position in ``idx``, the writers at or before it in its document, the last ``ring``
    of them (what the decode ring holds there). ``tokens`` / ``writes`` / ``starts`` (B, S)."""
    S = tokens.shape[1]
    seg = starts.long().cumsum(1) if starts is not None else torch.zeros_like(tokens)
    wb, wt = writes.nonzero(as_tuple=True)                          # row-major: each row's writers in order
    ab, at = idx // S, idx % S
    ok = (wb[None] == ab[:, None]) & (wt[None] <= at[:, None]) & (seg[wb, wt][None] == seg[ab, at][:, None])
    ok = ok & (ok.flip(1).cumsum(1).flip(1) <= ring)                # the last `ring` of them
    return tokens[wb, wt], ok


__wiring__ = Wiring.STANDALONE
