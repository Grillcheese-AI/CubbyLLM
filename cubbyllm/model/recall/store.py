"""EpisodicStore — the inference-time, persisted per-token memory store.

Wired: STANDALONE — the decode-side counterpart to MemoryRead's training path.

Holds one sequence's (key, value) pairs plus binary key-codes (sign(key)) for
O(N) Hamming retrieval. An explicit state_dict object so it can be checkpointed —
the fix for MEMORY_PROBE.md trap 2 (the last store lived in __dict__ and every
resume wiped it). retrieve_* return the top-K set; the read over that set is
MemoryRead.read, identical to training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.protocols import Wiring


class EpisodicStore:
    def __init__(self, d_key: int, d_model: int):
        self.d_key, self.d_model = int(d_key), int(d_model)
        self.keys = torch.empty(0, d_key)
        self.values = torch.empty(0, d_model)
        self.codes = torch.empty(0, d_key)

    def __len__(self):
        return self.keys.shape[0]

    def write(self, k, v):
        self.keys = torch.cat([self.keys.to(k.device), k], 0)
        self.values = torch.cat([self.values.to(v.device), v], 0)
        self.codes = torch.cat([self.codes.to(k.device), torch.sign(k)], 0)

    def retrieve_cosine(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        sims = F.cosine_similarity(q.unsqueeze(0), self.keys, dim=-1)
        idx = sims.topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])

    def retrieve_hamming(self, q, topk, return_idx=False):
        kk = min(topk, len(self))
        qc = torch.sign(q)
        ham = (qc.unsqueeze(0) != self.codes).sum(-1)          # (N,) Hamming distance
        idx = (-ham).topk(kk).indices
        return (idx, self.keys[idx], self.values[idx]) if return_idx else (self.keys[idx], self.values[idx])

    def state_dict(self):
        return {"keys": self.keys, "values": self.values, "codes": self.codes,
                "d_key": self.d_key, "d_model": self.d_model}

    def load_state_dict(self, sd):
        self.keys, self.values, self.codes = sd["keys"], sd["values"], sd["codes"]
        self.d_key, self.d_model = sd["d_key"], sd["d_model"]


__wiring__ = Wiring.STANDALONE
