"""BindingHead + unbind over BlockCodeOps.

Wired: WIRED — the binding head sits in the default forward path.

Two facts from the validation record shape this module:
  - The binding algebra is DECIDED: grilly BlockCodeOps, reached only through
    ``cubbyllm.ops.BlockCodeVSA`` (never importing grilly here — the seam rule).
  - A REAL unbind did not exist in the predecessor (H-B3: the dict-lookup
    ``UNBIND_ROLE`` was cubemind VM code, not the LM head). ``DirectUnbinder``
    is the from-scratch single-factor unbind; multi-factor decoding would chain
    F=2 hops (H-B4) or add a resonator — deferred, not needed for role/filler.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ...core.protocols import Wiring
from ...ops import BlockCodeVSA

if TYPE_CHECKING:
    pass


@runtime_checkable
class Unbinder(Protocol):
    """Recovers a bound factor — the from-scratch unbind (H-B3)."""

    def unbind(self, composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        ...


class DirectUnbinder:
    """Single-factor unbind via block-code correlation (exact for discrete codes).

    Recovers ``x`` from ``bind(x, key)`` given ``key``. For more than one unknown
    factor, chain F=2 hops (H-B4) — not implemented here by design.
    """

    def __init__(self, vsa: "BlockCodeVSA | None" = None) -> None:
        self.vsa = vsa if vsa is not None else BlockCodeVSA()

    def unbind(self, composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        return self.vsa.unbind(composite, known)


class BindingHead:
    """Binds role/filler (and superposes) via BlockCodeOps through the ops seam."""

    def __init__(self, vsa: "BlockCodeVSA | None" = None) -> None:
        self.vsa = vsa if vsa is not None else BlockCodeVSA()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two block-code hypervectors (association)."""
        return self.vsa.bind(a, b)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Superpose a set of bound pairs into one hypervector."""
        return self.vsa.bundle(vectors)

    def bind_pairs(self, roles: list[np.ndarray], fillers: list[np.ndarray]) -> np.ndarray:
        """Bind each role to its filler and bundle the result — the canonical
        role-filler structure grilly's SVC targets use (H-B5)."""
        if len(roles) != len(fillers):
            raise ValueError("roles and fillers must be the same length")
        bound = [self.vsa.bind(r, f) for r, f in zip(roles, fillers)]
        return self.vsa.bundle(bound)


__wiring__ = Wiring.WIRED
