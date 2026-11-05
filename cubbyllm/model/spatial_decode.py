"""cubbyllm/model/spatial_decode.py -- the spatial link at decode, in numpy: what a grilly2 trunk needs from the link
one token at a time, with no torch and no FFT (order 0052 A; H-K10, H-S12).

The link's algebra is circular convolution, done in torch with ``rfft`` / ``irfft``. grilly2 has no FFT kernel and no
complex type, and at d = 1,024 a real DFT is a (d/2+1) x d matrix product: ~1M multiply-adds a transform, against the
trunk's ~450M a token. So the transforms are two precomputed real matrices each way (the choice the order left open:
"precomputed on the host"), and everything else is elementwise or a small linear. Per stream it keeps, as
``CubbyModel.step`` does: the scene (d, the sum of the writing tokens' bound inputs since the document began) and the
copy's ring (the last ``ring`` written token ids).

At one step, for each row (the same calls and order as ``CubbyModel.step``):
  bind_input  the token's input x (embedding + context) bound to its place if it writes: x (*) S(p)
  write       a document start empties the scene and the ring; a writer adds its bound x and its id
  read        at layer ``at``, after the mixer, before the FFN: g * o(scene (*) unitary(q)^-1), q = S(p) at a placed
              (asking) row, else W_q rms(h); exactly zero while the scene is empty
  copy        at an asking row with a candidate: the logits become log((1 - lam) p_LM + lam p_copy) (spatial_copy.py)
Float64 on the host for the transforms (the trunk's own numbers are fp32); the place angles in float32, as torch
computes them (a coordinate in the thousands times a phase: float32's rounding is the encoding's own).

Wired: STANDALONE (order 0052 A: grilly2's trunk calls it at the hook; ``cubby_serve`` loads it behind a flag).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

__wiring__ = "STANDALONE"

FILES = ("spatial.json", "spatial.safetensors")


def dft(d: int) -> tuple:
    """(Fr, Fi, Ir, Ii): rfft(x) = x @ Fr + i x @ Fi; irfft(Xr + i Xi) = Xr @ Ir + Xi @ Ii (length d, d even)."""
    k = np.arange(d // 2 + 1)[:, None]
    n = np.arange(d)[None, :]
    ang = 2.0 * np.pi * ((k * n) % d) / d
    w = np.full((d // 2 + 1, 1), 2.0)
    w[0, 0] = w[-1, 0] = 1.0                       # the DC and Nyquist bins count once; their imaginary parts drop
    return np.cos(ang).T, -np.sin(ang).T, w * np.cos(ang) / d, -w * np.sin(ang) / d


class SpatialDecode:
    """The link's weights and its decode math; ``state(batch)`` is one batch of streams' scene and ring."""

    def __init__(self, cfg: dict, w: dict) -> None:
        self.cfg, self.d, self.dims = cfg, int(cfg["d"]), int(cfg["dims"])
        self.scale, self.at, self.query, self.bind_mode = float(cfg["scale"]), int(cfg["at"]), cfg["query"], cfg["bind"]
        if self.query not in ("placed", "learned") or cfg.get("key") or cfg.get("copy_refuse"):
            raise ValueError("this decode port has the placed / learned query and the plain copy only")
        f64 = lambda k: np.asarray(w[k], dtype=np.float64)                              # noqa: E731
        self.phases = np.asarray(w["phases"], dtype=np.float32)
        self.norm_w, self.q_w, self.o_w = f64("norm_w"), f64("q.weight"), f64("o.weight")
        self.gate_w, self.gate_b = f64("gate.weight")[0], float(f64("gate.bias")[0])
        self.copy = cfg.get("copy")
        if self.copy:
            if self.copy != "ring":
                raise ValueError("the copy's decode port is the ring")
            self.ring = int(cfg["copy_ring"])
            self.c_w, self.c_b = f64("copy.gate.weight")[0], float(f64("copy.gate.bias")[0])
            self.tau = float(f64("copy.tau").reshape(-1)[0])
            self.beta = float(f64("copy.beta").reshape(-1)[0]) if "copy.beta" in w else None
        self.Fr, self.Fi, self.Ir, self.Ii = dft(self.d)

    @classmethod
    def load(cls, export_dir: str) -> "SpatialDecode | None":
        """The link beside a grilly2 export, or None when the export has none."""
        if not os.path.exists(os.path.join(export_dir, FILES[0])):
            return None
        from safetensors.numpy import load_file
        cfg = json.load(open(os.path.join(export_dir, FILES[0]), encoding="utf-8"))
        return cls(cfg, load_file(os.path.join(export_dir, FILES[1])))

    # -- the algebra ----------------------------------------------------------------------------------------------------
    def rfft(self, x):
        return x @ self.Fr, x @ self.Fi

    def irfft(self, re, im):
        return re @ self.Ir + im @ self.Ii

    def spectrum(self, p):
        """exp(i Theta p / scale): (re, im), (..., d/2+1); the angle in float32, as torch's."""
        ang = (np.asarray(p, dtype=np.float32) @ self.phases.T) / np.float32(self.scale)
        return np.cos(ang).astype(np.float64), np.sin(ang).astype(np.float64)

    def encode(self, p):
        return self.irfft(*self.spectrum(p))

    def bind(self, e, p):
        er, ei = self.rfft(e)
        sr, si = self.spectrum(p)
        return self.irfft(er * sr - ei * si, er * si + ei * sr)

    def lookup(self, scene, q):
        """scene (*) unitary(q)^-1."""
        qr, qi = self.rfft(q)
        m = np.sqrt(qr * qr + qi * qi) + 1e-6
        ur, ui = qr / m, qi / m
        ur, ui = self.rfft(self.irfft(ur, ui))       # torch takes unitary(q) back to the signal and forward again
        ar, ai = self.rfft(scene)
        return self.irfft(ar * ur + ai * ui, ai * ur - ar * ui)

    # -- the decode state and the four calls ------------------------------------------------------------------------------
    def state(self, batch: int) -> dict:
        st = {"scene": np.zeros((batch, self.d))}
        if self.copy:
            st.update(ring=np.zeros((batch, self.ring), dtype=np.int64), ring_n=np.zeros(batch, dtype=np.int64))
        return st

    def bind_input(self, x, position, write):
        """x (B, d) the token's input; position (B, dims); write (B,) bool -> the input the trunk reads."""
        x = np.asarray(x, dtype=np.float64)
        if self.bind_mode != "writes":
            return self.bind(x, np.asarray(position, dtype=np.float32))
        w = np.asarray(write, dtype=bool)
        if not w.any():                              # S(0) is the identity: the input untouched, bit for bit
            return x
        return np.where(w[:, None], self.bind(x, np.asarray(position, dtype=np.float32)), x)

    def write(self, st: dict, x_bound, token, write, start=None) -> None:
        """A start empties the row's scene and ring; a writer adds its bound input and its token id."""
        write = np.asarray(write, dtype=bool)
        if start is not None:
            keep = ~np.asarray(start, dtype=bool)
            st["scene"] *= keep[:, None]
            if self.copy:
                st["ring_n"] *= keep
        st["scene"] += np.asarray(x_bound, dtype=np.float64) * write[:, None]
        if self.copy:
            rows = np.nonzero(write)[0]
            st["ring"][rows, st["ring_n"][rows] % self.ring] = np.asarray(token)[rows]
            st["ring_n"] += write

    def read(self, st: dict, h, position, ask):
        """What the read adds to the stream at layer ``at`` (B, d); zero rows where the scene is empty."""
        h = np.asarray(h, dtype=np.float64)
        if not st["scene"].any():                    # nothing written in any stream: the read adds exactly zero
            return np.zeros_like(h)
        n = h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6) * self.norm_w
        g = 1.0 / (1.0 + np.exp(-(n @ self.gate_w + self.gate_b)))
        q = n @ self.q_w.T
        if self.query == "placed" and position is not None and ask is not None:
            a = np.asarray(ask, dtype=bool)
            q[a] = self.encode(np.asarray(position, dtype=np.float32)[a])
        return g[:, None] * (self.lookup(st["scene"], q) @ self.o_w.T)

    def add_copy(self, st: dict, logits, h, position, ask, table):
        """The logits (B, V) with the copy mixed in at asking rows that have a candidate; ``table`` (V, d) the
        embedding table the scene was written from."""
        if not self.copy or ask is None:
            return logits
        a = np.asarray(ask, dtype=bool) & (st["ring_n"] > 0)
        out = np.array(logits, dtype=np.float32, copy=True)
        for b in np.nonzero(a)[0]:
            ids = st["ring"][b, :min(int(st["ring_n"][b]), self.ring)]
            r = self.lookup(st["scene"][b:b + 1], self.encode(np.asarray(position, dtype=np.float32)[b:b + 1]))[0]
            out[b] = self._mix(out[b], np.asarray(h[b], dtype=np.float64), r, table, ids)
        return out

    def _mix(self, logits, h, r, table, ids):
        n = h / math.sqrt(float((h * h).mean()) + 1e-6)
        g = float(n @ self.c_w + self.c_b)
        rn = r / max(float(np.linalg.norm(r)), 1e-12)
        e = np.asarray(table[ids], dtype=np.float64)
        e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
        s, scale = e @ rn, math.exp(self.tau)
        z = scale * s
        pc = np.exp(z - z.max())
        pc /= pc.sum()
        c = np.zeros(logits.shape[-1])
        np.add.at(c, ids, pc)
        if self.beta is not None:
            g += scale * (float(s.max()) - self.beta)
        lg = logits.astype(np.float64)
        lsm = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
        ls = lambda v: -np.logaddexp(0.0, -v)                                           # noqa: E731  log sigmoid
        with np.errstate(divide="ignore"):
            log_copy = np.log(np.maximum(c, 1e-30))
        return np.logaddexp(ls(-g) + lsm, ls(g) + log_copy).astype(np.float32)


__all__ = ["FILES", "SpatialDecode", "dft"]
