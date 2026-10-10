"""What — kinds of thing, learned without labels, and what each kind feels like (design §4.4–4.5).

KindCodebook. Every surface pixel's appearance — its chromaticity (shading multiplies all three
channels alike, so chromaticity is the same on a lit top and a shaded side) and its brightness —
is recalled against a small set of prototypes; the nearest within a radius is its kind (and moves
a little toward it), and a pixel nothing is near starts a new kind. Far pixels, washed by fog,
are not used. No label is ever given: a kind is "things that look like this".

Seen ⊗ felt. When the body touches a surface whose kind it has seen (the cell under its feet in
its own map), what it felt is bound to that kind (a running mean of the touch vector). Then
"looks like this" predicts "feels like this": a kind never touched borrows the feel of the
touched kinds that look most like it.

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

import numpy as np

__wiring__ = "STANDALONE"


def appearance(rgb: np.ndarray) -> np.ndarray:
    """(n, 3) uint8 -> (n, 3) features: chromaticity (2, scaled) and log brightness."""
    f = rgb.astype(np.float32)
    s = f.sum(axis=1, keepdims=True) + 1.0
    chroma = f[:, :2] / s
    return np.concatenate([chroma * 12.0, np.log1p(s / 60.0)], axis=1)


class KindCodebook:
    def __init__(self, radius: float = 0.55, rate: float = 0.02, max_kinds: int = 256) -> None:
        self.protos = np.zeros((0, 3), np.float32)
        self.counts: list[int] = []
        self.radius, self.rate, self.max_kinds = radius, rate, max_kinds
        self.feel: dict[int, np.ndarray] = {}          # kind -> mean touch vector
        self.felt: dict[int, int] = {}
        self.merged: dict[int, int] = {}               # kind -> the kind it was merged into

    def assign(self, feats: np.ndarray, learn: bool = True) -> np.ndarray:
        """Kind id per feature row (new kinds made as needed when learning)."""
        out = np.full(len(feats), -1, np.int64)
        if len(feats) == 0:
            return out
        if len(self.protos):
            d = np.linalg.norm(feats[:, None, :] - self.protos[None, :, :], axis=2)
            j = np.argmin(d, axis=1)
            near = d[np.arange(len(feats)), j] < self.radius
            out[near] = j[near]
        if learn:
            for i in np.nonzero(out < 0)[0]:                 # sequentially: each new kind can catch the next
                f = feats[i]
                if len(self.protos):
                    d = np.linalg.norm(self.protos - f, axis=1)
                    j = int(np.argmin(d))
                    if d[j] < self.radius:
                        out[i] = j
                        continue
                if len(self.protos) >= self.max_kinds:
                    out[i] = int(np.argmin(np.linalg.norm(self.protos - f, axis=1)))
                    continue
                self.protos = np.vstack([self.protos, f[None, :]])
                self.counts.append(0)
                out[i] = len(self.protos) - 1
            for k in np.unique(out[out >= 0]):
                m = feats[out == k].mean(axis=0)
                self.protos[k] += self.rate * (m - self.protos[k])
                self.counts[k] += int((out == k).sum())
        if self.merged:
            out = np.array([self.canon(int(k)) if k >= 0 else -1 for k in out], np.int64)
        return out

    def canon(self, kind: int) -> int:
        """The kind a kind was merged into (itself if never merged)."""
        seen = set()
        while kind in self.merged and kind not in seen:
            seen.add(kind)
            kind = self.merged[kind]
        return kind

    def consolidate(self, ratio: float = 0.6) -> dict:
        """Kinds that drifted together (one material first seen from two faces or distances) are
        one kind: the smaller is merged into the larger. -> {merged: into}."""
        out = {}
        live = [k for k in range(len(self.protos)) if k not in self.merged]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if a in self.merged or b in self.merged:
                    continue
                if np.linalg.norm(self.protos[a] - self.protos[b]) < ratio * self.radius:
                    big, small = (a, b) if self.counts[a] >= self.counts[b] else (b, a)
                    self.merged[small] = big
                    self.counts[big] += self.counts[small]
                    if small in self.feel and big not in self.feel:
                        self.feel[big], self.felt[big] = self.feel[small], self.felt[small]
                    out[small] = big
        return out

    # ------------------------------------------------------------------ seen ⊗ felt
    def touch(self, kind: int, vector) -> None:
        v = np.asarray(vector, np.float32)
        n = self.felt.get(kind, 0)
        self.feel[kind] = v if n == 0 else self.feel[kind] + (v - self.feel[kind]) / (n + 1)
        self.felt[kind] = n + 1

    def predict_feel(self, kind: int) -> np.ndarray | None:
        """What a kind should feel like: its own touch if felt, else the touched kinds that look
        most like it (weighted by likeness)."""
        if kind in self.feel:
            return self.feel[kind]
        if not self.feel or kind < 0 or kind >= len(self.protos):
            return None
        ks = list(self.feel)
        d = np.linalg.norm(self.protos[ks] - self.protos[kind], axis=1)
        w = np.exp(-d / max(self.radius, 1e-6))
        return (np.stack([self.feel[k] for k in ks]) * w[:, None]).sum(axis=0) / w.sum()
