"""EpisodicStore — the inference-time, persisted per-token memory store.

Wired: STANDALONE — the decode-side counterpart to MemoryRead's training path.

Holds one sequence's (key, value) pairs plus binary key-codes (SimHash via random
projection for O(N) Hamming retrieval). An explicit state_dict object so it can be
checkpointed — the fix for MEMORY_PROBE.md trap 2 (the last store lived in __dict__
and every resume wiped it). retrieve_* return the top-K set; the read over that set
is MemoryRead.read, identical to training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring


class EpisodicStore:
    def __init__(self, d_key: int, d_model: int, n_bits: int = 256):
        self.d_key, self.d_model, self.n_bits = int(d_key), int(d_model), int(n_bits)
        g = torch.Generator().manual_seed(0xC0DEB00C)          # fixed -> identical R everywhere
        self.R = torch.randn(d_key, n_bits, generator=g)       # (d_key, n_bits) SimHash projection
        self.keys = torch.empty(0, d_key)
        self.values = torch.empty(0, d_model)
        self.codes = torch.empty(0, n_bits)

    def __len__(self):
        return self.keys.shape[0]

    def write(self, k, v):
        self.R = self.R.to(k.device)                           # cache: avoid re-copy every call
        self.keys = torch.cat([self.keys.to(k.device), k], 0)
        self.values = torch.cat([self.values.to(v.device), v], 0)
        code = torch.sign(k @ self.R)                          # (M, n_bits) SimHash
        self.codes = torch.cat([self.codes.to(k.device), code], 0)

    def retrieve_cosine(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        sims = F.cosine_similarity(q.unsqueeze(0), self.keys, dim=-1)
        idx = sims.topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])

    def retrieve_hamming(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        self.R = self.R.to(q.device)                           # cache: avoid re-copy every call
        qc = torch.sign(q @ self.R)                             # (n_bits,) SimHash of query
        ham = (qc.unsqueeze(0) != self.codes).sum(-1)          # (N,) Hamming over n_bits bits
        idx = (-ham).topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])

    def state_dict(self):
        return {"keys": self.keys, "values": self.values, "codes": self.codes,
                "R": self.R, "d_key": self.d_key, "d_model": self.d_model, "n_bits": self.n_bits}

    def load_state_dict(self, sd):
        self.keys, self.values, self.codes, self.R = sd["keys"], sd["values"], sd["codes"], sd["R"]
        self.d_key, self.d_model, self.n_bits = sd["d_key"], sd["d_model"], sd["n_bits"]


__wiring__ = Wiring.STANDALONE
