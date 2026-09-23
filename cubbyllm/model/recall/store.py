"""EpisodicStore — the inference-time, persisted per-token memory store.

Wired: STANDALONE — the decode-side counterpart to MemoryRead's training path.

Holds one sequence's (key, value) pairs plus binary key-codes (SimHash via random
projection for O(N) Hamming retrieval). An explicit state_dict object so it can be
checkpointed — the fix for MEMORY_PROBE.md trap 2 (the last store lived in __dict__
and every resume wiped it). retrieve_* return the top-K set; the read over that set
is MemoryRead.read, identical to training.

**Fixed capacity, device cursor.** The store used to be three tensors grown
by ``torch.cat`` and a host ``len()``. Both are fatal to a recorded decode
step: a growing tensor is a new graph signature every token, and a host
``len()`` is a value the replay cannot reproduce because it never runs the
Python that computed it (grilly2's ``docs/capture.md``; CUDA graphs impose
the same contract). So the buffers are allocated once at ``capacity`` rows
and written in place at a slot chosen on the device.

**Three things are precomputed at write time, because a decode step pays
for them once per token and a read pays for them once per token per
layer.** They are the difference between 35 dispatches for a retrieval and
about 6:

- **normalised keys.** Selection is cosine, so ``keys_n`` holds the L2-
  normalised key and the whole selection is one matmul against a normalised
  query. ``F.cosine_similarity`` over the store was 17 dispatches for the
  same number. The read still gathers the *unnormalised* key, as
  ``MemoryRead.forward`` does — only the selection is on the unit sphere.
- **liveness as an additive bias.** ``bias`` is ``-inf`` for a row never
  written and ``0`` for a live one, so masking is one add instead of
  ``arange(capacity) < count`` and a ``masked_fill`` every token.
- **the SimHash codes, only if something asks for them.** ``hamming=False``
  skips them; the decode path retrieves by cosine and paying a projection
  and a sign per token for a path it never takes is pure loss.

The cost is a real behaviour change and is stated rather than hidden: past
``capacity`` tokens of beyond-window history the oldest entry is
overwritten, where the old store kept everything. See H-A8 in
CUBBYLLM_HYPOTHESES.md for the claim and its kill criterion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring

#: Beyond-window tokens retained before the oldest is overwritten. 16384 is
#: ~16x the training context and ~53 MB at d_key=64 / d_model=512 /
#: n_bits=256; raise it with ``EpisodicStore(..., capacity=)``.
DEFAULT_CAPACITY = 16384

_DEAD = float("-inf")


class EpisodicStore:
    def __init__(self, d_key: int, d_model: int, n_bits: int = 256,
                 capacity: int = DEFAULT_CAPACITY, hamming: bool = True,
                 device=None):
        self.d_key, self.d_model, self.n_bits = int(d_key), int(d_model), int(n_bits)
        self.capacity, self.hamming = int(capacity), bool(hamming)
        # The SimHash projection is only built when something will read it:
        # `torch.randn` is a host write, so a store constructed inside a
        # capture would be refused, and a decode store retrieves by cosine.
        g = torch.Generator().manual_seed(0xC0DEB00C)          # fixed -> identical R everywhere
        z = dict(device=device) if device is not None else {}
        self.R = (torch.randn(d_key, n_bits, generator=g).to(**z) if self.hamming
                  else torch.zeros(d_key, 0, **z))
        self.keys = torch.zeros(self.capacity, d_key, **z)
        self.keys_n = torch.zeros(self.capacity, d_key, **z)   # L2-normalised, for selection
        self.values = torch.zeros(self.capacity, d_model, **z)
        self.codes = torch.zeros(self.capacity if self.hamming else 0, n_bits, **z)
        self.bias = torch.full((self.capacity,), _DEAD, **z)   # 0 live, -inf never written

    def __len__(self):
        """Entries written, capped at ``capacity``.

        **This reads the device back**, so it ends the command batch and
        cannot be called from a step that is being recorded. Tests and
        checkpointing use it; the decode path does not — liveness is the
        bias row, which never leaves the device."""
        return int((self.bias == 0.0).sum().item())

    def _to(self, device):
        if self.keys.device != device:
            for name in ("keys", "keys_n", "values", "codes", "bias", "R"):
                setattr(self, name, getattr(self, name).to(device))
        elif self.R.device != device:
            self.R = self.R.to(device)                         # cache: avoid re-copy every call

    def write(self, k, v):
        """Append ``k``'s rows at the current fill level (the host path).

        Prefill and the tests use this; it reads the device and so is not
        capture-safe. ``write_at`` is the one a recorded step calls."""
        self._to(k.device)
        n, start = k.shape[0], len(self)
        rows = torch.remainder(torch.arange(n, device=k.device) + start, self.capacity)
        self.write_at(rows, k, v, torch.zeros(n, device=k.device))

    def write_at(self, slot, k, v, live):
        """Write ``k``/``v`` into the rows ``slot`` names, in place.

        ``slot`` is a ``(M,)`` int64 **tensor**, which is the whole point:
        a recorded step computes it from its device position counter, so
        the replay writes the slot that replay is for rather than the one
        the capture saw.

        ``live`` is the bias to write — ``0`` for a row that now holds a
        real entry, ``-inf`` for one that does not. The caller owns it
        because what makes an entry live is the caller's causal rule
        (``pos - window``), not the fact that bytes landed, and because a
        step that has not reached the window yet still has to write
        *something* to keep its command stream the same shape."""
        self._to(k.device)
        self.keys.index_copy_(0, slot, k)
        self.keys_n.index_copy_(0, slot, F.normalize(k, dim=-1))
        self.values.index_copy_(0, slot, v)
        self.bias.index_copy_(0, slot, live)
        if self.hamming:
            self.codes.index_copy_(0, slot, torch.sign(k @ self.R))

    def _select(self, scores, topk, return_idx, return_valid):
        """Top-``topk`` over ``scores``, with rows never written out.

        The mask is the bias row, added rather than applied, so the shapes
        here do not depend on how full the store is — which is what lets
        the whole read be recorded on the first token and replayed on every
        later one. When fewer than ``topk`` entries are live the extra
        slots come back flagged invalid rather than absent, and
        ``MemoryRead.read`` reads zero through them."""
        top = (scores + self.bias).topk(min(int(topk), self.capacity))
        idx = top.indices
        out = (self.keys[idx], self.values[idx])
        if return_valid:
            out = out + (torch.isfinite(top.values),)
        return (idx,) + out if return_idx else out

    def retrieve_cosine(self, q, topk, return_idx=False, return_valid=False):
        """Cosine top-K. One matmul: the keys were normalised at write."""
        self._to(q.device)
        scores = F.linear(F.normalize(q, dim=-1).unsqueeze(0), self.keys_n)[0]
        return self._select(scores, topk, return_idx, return_valid)

    def retrieve_hamming(self, q, topk, return_idx=False, return_valid=False):
        if not self.hamming:
            raise RuntimeError(
                "this store was built with hamming=False, so its SimHash codes are "
                "not maintained and retrieve_hamming would read stale rows. Build it "
                "with hamming=True to use the Hamming index."
            )
        self._to(q.device)
        qc = torch.sign(q @ self.R)                             # (n_bits,) SimHash of query
        ham = (qc.unsqueeze(0) != self.codes).sum(-1)          # (N,) Hamming over n_bits bits
        return self._select(-ham.float(), topk, return_idx, return_valid)

    def state_dict(self):
        return {"keys": self.keys, "keys_n": self.keys_n, "values": self.values,
                "codes": self.codes, "R": self.R, "bias": self.bias,
                "capacity": self.capacity, "hamming": self.hamming,
                "d_key": self.d_key, "d_model": self.d_model, "n_bits": self.n_bits}

    def load_state_dict(self, sd):
        self.keys, self.values, self.R = sd["keys"], sd["values"], sd["R"]
        self.codes = sd["codes"]
        self.d_key, self.d_model, self.n_bits = sd["d_key"], sd["d_model"], sd["n_bits"]
        self.capacity = int(sd.get("capacity", self.keys.shape[0]))
        self.hamming = bool(sd.get("hamming", self.codes.shape[0] > 0))
        # a store saved before the precomputed columns existed has neither;
        # both are functions of the keys, so rebuild rather than refuse
        self.keys_n = sd.get("keys_n")
        if self.keys_n is None:
            self.keys_n = F.normalize(self.keys, dim=-1)
        self.bias = sd.get("bias")
        if self.bias is None:
            count = int(sd["count"].item()) if "count" in sd else self.keys.shape[0]
            live = torch.arange(self.capacity, device=self.keys.device) < count
            self.bias = torch.where(live, 0.0, torch.tensor(_DEAD, device=self.keys.device))


__wiring__ = Wiring.STANDALONE
