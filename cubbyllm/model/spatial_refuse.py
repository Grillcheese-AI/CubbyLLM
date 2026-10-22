"""Copy or refuse -- the copy path's contract at an asked answer (order 0022, H-K10).

Why. Order 0021 found where the wrong answers come from: when the copy's gate closes, the trunk speaks, and at an
asking position the trunk's pick is never the fact. So under this contract the trunk never answers a fact. At the
first asking position of an answer (a run of consecutive asking tokens: ``here`` asks piece 0 and each answer piece
asks the next) the read of every piece is taken at once, ahead -- the asking tokens write nothing, so the scene is the
same at each of them and the reads ahead are the reads each piece will make -- and the answer is decided whole, from
each piece's best match ``m`` among the candidates:

    m >= theta              a piece: it comes from the copy alone (lambda = 1);
    m <  theta_end          the answer has ended here: the trunk speaks (the end of the answer, EOS);
    theta_end <= m < theta  unsure: the whole answer is refused.

An answer with no piece (the first read already ended: the document holds nothing there), or with no end within
``lookahead`` pieces, is refused too. A refused answer emits ``refuse_id`` at every one of its asking positions (the
toy's NONE; on the 450M a token the host speaks as "I don't know") and no fact token. Decode keeps the decision made at
the run's first position in its state, so decode = forward.

The reads ahead, by how an address names its piece:
- **hashed places** (``key=None``): piece ``j + k`` sits ``k`` steps up (``step``, e.g. ``(0, 0, 6)``: 6 cells up z,
  ``make_fact_shard.address``);
- **text keys, placed**: the piece slot ``positions[..., 2]`` plus ``k``;
- **the pointer**: its query ``key(s, r) (*) P_j`` rebound to ``P_{j+k}`` -- a product of unit spectra, so
  ``q (*) P_{j+k} (*) P_j^-1``; nothing of the pointer is recomputed.

Wired: STANDALONE until a run's meta sets ``spatial.copy_refuse`` (``SpatialLink.add_copy`` calls ``apply``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.protocols import Wiring

__all__ = ["runs", "queries", "decide", "apply", "STATE"]

#: the decode state: was the last token an asker, the decided answer length (0 = refused), the offset in the answer
STATE = ("cr_prev", "cr_L", "cr_off")


def runs(idx: torch.Tensor, S: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For sorted flat asking positions ``idx``: the index (into ``idx``) of each one's run start, and its offset."""
    n = idx.numel()
    ar = torch.arange(n, device=idx.device)
    cont = torch.zeros(n, dtype=torch.bool, device=idx.device)
    if n > 1:
        cont[1:] = (idx[1:] == idx[:-1] + 1) & (idx[1:] // S == idx[:-1] // S)
    start = torch.where(cont, torch.zeros_like(ar), ar).cummax(0).values
    return start, ar - start


def queries(link, pos: torch.Tensor, q: torch.Tensor | None, table: torch.Tensor, K: int, step) -> torch.Tensor:
    """(n, K, d): the query of piece ``+k`` (k = 0..K-1) from each start, ``pos`` (n, dims), ``q`` (n, d) the
    read's own query (the pointer's)."""
    k = torch.arange(K, device=pos.device)
    if link.keys is not None and link.query == "pointer":
        j = pos[:, 2].long()
        spec = torch.fft.rfft(q.float(), dim=-1)[:, None] * link.keys.piece(j[:, None] + k) \
            * link.keys.piece(j)[:, None].conj()
        return torch.fft.irfft(spec, n=link.d, dim=-1)
    p = pos[:, None].repeat(1, K, 1)
    if link.keys is not None:
        p[..., 2] = p[..., 2] + k
    else:
        p = p + k[:, None].to(p.dtype) * torch.as_tensor(step, dtype=p.dtype, device=p.device)
    return link.encode(p, table)


def decide(bests: torch.Tensor, theta: float, theta_end: float) -> torch.Tensor:
    """``bests`` (n, K), piece 0 first -> the answer's length (n,), 0 = refused."""
    K = bests.shape[1]
    end = bests < theta_end
    has_end = end.any(1)
    L = torch.where(has_end, end.float().argmax(1), torch.full_like(has_end, K, dtype=torch.long))
    before = torch.arange(K, device=bests.device)[None] < L[:, None]
    unsure = (before & (bests < theta)).any(1)
    return torch.where((L == 0) | ~has_end | unsure, torch.zeros_like(L), L)


def _bests(cp, r: torch.Tensor, table: torch.Tensor, ids, ok) -> torch.Tensor:
    """(n, K): each read's best cosine among its candidates (the ring's, or the table's)."""
    rn = F.normalize(r.float(), dim=-1)
    if cp.mode == "table":
        return (rn @ F.normalize(table.float(), dim=-1).T).max(-1).values
    e = F.normalize(table.float()[ids], dim=-1)
    s = torch.einsum("nkd,cd->nkc", rn, e) if ids.dim() == 1 else torch.einsum("nkd,ncd->nkc", rn, e)
    return s.masked_fill(~ok[:, None], float("-inf")).max(-1).values


def apply(link, logits, log_copy, idx, S, scene, pos, q, table, ids, ok, state=None) -> torch.Tensor:
    """The rows (n, V) at asking positions ``idx`` under the contract. Forward: runs from ``idx``; decode (``state``,
    S = 1, one position a row): the run continues while the row keeps asking."""
    cp = link.copy_path
    theta, theta_end, step, rid, K = cp.refuse
    starts, off = runs(idx, S) if state is None else (torch.arange(idx.numel(), device=idx.device), None)
    first = starts.unique() if state is None else (~state["cr_prev"][idx]).nonzero().squeeze(1)
    L = torch.zeros(idx.numel(), dtype=torch.long, device=idx.device)
    if first.numel():
        qk = queries(link, pos[first], None if q is None else q[first], table, K, step)
        r = link.lookup(scene[first].float()[:, None].expand(-1, K, -1).contiguous(), qk)   # MKL: no stride-0 FFT
        sub_ids = ids if ids is None or ids.dim() == 1 else ids[first]
        L[first] = decide(_bests(cp, r, table, sub_ids, None if ok is None else ok[first]), theta, theta_end)
    if state is None:
        L = L[starts]
    else:                                                 # decode: a new run decides, a continuing one remembers
        new = torch.zeros(idx.numel(), dtype=torch.bool, device=idx.device)
        new[first] = True
        state["cr_L"][idx] = torch.where(new, L, state["cr_L"][idx])
        state["cr_off"][idx] = torch.where(new, torch.zeros_like(L), state["cr_off"][idx] + 1)
        L, off = state["cr_L"][idx], state["cr_off"][idx]
    refused = torch.full_like(logits, -1e4, dtype=torch.float32)
    refused[:, rid] = 0.0
    out = torch.where((off < L)[:, None], log_copy, F.log_softmax(logits.float(), -1))
    return torch.where((L == 0)[:, None], F.log_softmax(refused, -1), out)


__wiring__ = Wiring.STANDALONE
