"""cubbyllm/ops/grilly_spatial.py -- the spatial link on grilly2 (order 0052 A): ``SpatialDecode``'s numpy math at the
three points grilly2's ``CubbyForCausalLM.forward(..., spatial=)`` calls it.

  input   after the context (which reads the inputs unbound): the writing tokens' inputs bound to their places, the
          scene and the copy's ring written
  read    at layer ``at``, between the mixer and the FFN: what the scene holds where the stream asks; None (nothing added,
          the stream untouched) while no stream has written anything
  output  the copy mixed into the logits at asking tokens with a candidate
One ``GrillyLink`` is one batch of streams: its state (scene, ring) carries from a prompt to the steps after it, as the
trunk's ``CubbyState`` does. Before each forward the caller gives the call's tokens' roles (``roles``). Without a link,
or with nothing placed, grilly2's logits are its own, bit for bit.

Wired: STANDALONE (order 0052 A: the far-facts read at home; ``cubby_serve`` loads it behind ``--spatial``).
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import Wiring

__wiring__ = Wiring.STANDALONE


class GrillyLink:
    def __init__(self, dec, table, batch: int = 1) -> None:
        self.dec, self.at = dec, dec.at
        self.table = np.asarray(table, dtype=np.float32)       # the embedding table the scene is written from
        self.st, self._roles, self._call = dec.state(batch), None, None

    @classmethod
    def load(cls, export_dir: str, model, batch: int = 1) -> "GrillyLink | None":
        """The link beside a grilly2 export (``spatial.json``), or None; ``model`` is the loaded trunk (its table)."""
        from ..model.spatial_decode import SpatialDecode
        dec = SpatialDecode.load(export_dir)
        return None if dec is None else cls(dec, model.model.embed_tokens.weight.detach().numpy(), batch)

    def roles(self, token, position, write, ask=None, start=None) -> "GrillyLink":
        """This call's tokens (B, T), places (B, T, dims), writers, askers and document starts (B, T) bool."""
        self._roles = tuple(None if v is None else np.asarray(v) for v in (token, position, write, ask, start))
        return self

    def input(self, x):
        import grilly
        if self._roles is None:
            raise RuntimeError("GrillyLink.roles(...) first: the link needs this call's places")
        tok, pos, w, a, s = self._roles
        self._call = self.dec.call(self.st, x.detach().numpy(), tok, pos, w, a, s)
        self._roles = None
        return grilly.from_numpy(self._call.xb.astype(np.float32)) if self._call.wrote else x

    def read(self, h):
        import grilly
        r = self._call.read(h.detach().numpy())
        return None if r is None else grilly.from_numpy(r.astype(np.float32))

    def output(self, logits, h, keep: int = 0):
        import grilly
        if not self._call.rings:
            return logits
        return grilly.from_numpy(self._call.copy(logits.detach().numpy(), h.detach().numpy(), self.table, keep))


__all__ = ["GrillyLink"]
