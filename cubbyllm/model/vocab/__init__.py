"""cubbyllm.model.vocab — hybrid embedding source + retrieval output head.

Wired: WIRED — package marker.
"""
from __future__ import annotations

from ...core.protocols import Wiring
from .embedding import EmbeddingSource, HybridEmbedding
from .output_head import RetrievalHead, TopKRetrievalHead

__all__ = ["EmbeddingSource", "HybridEmbedding", "RetrievalHead", "TopKRetrievalHead"]
__wiring__ = Wiring.WIRED
