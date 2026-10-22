"""cubbyllm/ops/packed.py -- grilly2's packed bit codes (``grilly.vsa.packed``) behind the ops seam (order 0031).

Bit codes are ``dim // 32`` int32 words a row. With grilly2 the Hamming top-k runs on the GPU; without it ``packed()``
is None and the caller computes the same with numpy. Only ``cubbyllm/ops/`` names grilly
(``test_ops_is_sole_grilly_importer``): the capsule store (``standin/capsules.py``, bound for ``cubbyllm/worlds/``)
reaches it through here.

Wired: WIRED (``capsules.CapsuleStore``).
"""
from __future__ import annotations

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED


def packed():
    """grilly2's ``vsa.packed`` module, or None (grilly v1, or no grilly)."""
    try:
        from grilly.vsa import packed as p
        return p
    except Exception:
        return None


def to_device(words):
    """A numpy array of packed words on grilly's device."""
    import grilly
    return grilly.from_numpy(words)


def topk(p, query, book, k: int, dim: int):
    """``(idx, dist)`` lists: the ``k`` rows of ``book`` (on the device) nearest ``query`` (numpy) in Hamming distance."""
    import grilly
    idx, dist, _ = p.topk(grilly.from_numpy(query), book, k=k, dim=dim)
    return idx.numpy().tolist(), dist.numpy().tolist()


__all__ = ["packed", "to_device", "topk"]
