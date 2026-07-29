"""DataPipeline — manifest-pinned corpus source (H-G3 reproducibility).

Wired: STANDALONE — data plumbing, not the forward path.

H-G3 found the candidate corpus (``unified/``) had silently drifted from its own
tokenizer report and mixes unlabeled sources. The lesson is encoded as a
contract: a pipeline must expose ``manifest_hash()`` so any training run pins
the exact mixture it consumed. ``InMemoryDataPipeline`` is the concrete,
testable reference — it holds a token array plus a manifest string and hashes
them together. A corpus-backed pipeline over ``D:\grillcheese_training_data`` is
a later cycle (the H-G3 cleanup gate); it implements the same protocol.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.protocols import Wiring

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from torch import Tensor


@runtime_checkable
class DataPipeline(Protocol):
    """A reproducible, manifest-pinned batch source."""

    def manifest_hash(self) -> str:
        """Stable hash of the exact corpus mixture (sources + versions + weights)."""
        ...

    def batches(self, batch_size: int, seq_len: int) -> "Iterator[Tensor]":
        """Yield token-id batches of shape (batch_size, seq_len)."""
        ...


class InMemoryDataPipeline:
    """A token array + a manifest string, hashed for reproducibility.

    tokens   — 1-D integer token-id array (numpy).
    manifest — a human-readable description of the exact mixture (sources,
               versions, weights). Pinned into the hash so a run records what it
               consumed. ``seed`` makes batch sampling deterministic.
    """

    def __init__(self, tokens: "np.ndarray", manifest: str, seed: int = 0) -> None:
        import numpy as np

        self.tokens = np.asarray(tokens).astype(np.int64).reshape(-1)
        self.manifest = str(manifest)
        self.seed = int(seed)

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.manifest.encode("utf-8"))
        h.update(self.tokens.tobytes())
        return h.hexdigest()

    def batches(self, batch_size: int, seq_len: int) -> "Iterator[Tensor]":
        import numpy as np
        import torch

        rng = np.random.default_rng(self.seed)
        n = len(self.tokens) - seq_len - 1
        if n <= 0:
            raise ValueError("corpus shorter than seq_len + 1")
        while True:
            ix = rng.integers(0, n, size=batch_size)
            x = np.stack([self.tokens[i:i + seq_len] for i in ix])
            y = np.stack([self.tokens[i + 1:i + seq_len + 1] for i in ix])
            yield torch.from_numpy(x), torch.from_numpy(y)


__wiring__ = Wiring.STANDALONE
