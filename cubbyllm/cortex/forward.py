"""Forward model and curiosity (design §4.7) — what a place will look like, and how much there is
still to learn about it.

ViewModel. The forward model of the view: before a place comes into view (the move or the look
that brings it there is the action), it predicts how each of its faces will look — per face (the
cell behind the surface, which of its six sides faces the eye, and how far away it is seen, so a
lit top and a shaded side, or a face up close and the same face hazed by distance, are separate
predictions), a running estimate of its appearance; a face never seen is predicted from its
region's usual look at that distance. When the face is seen, its error is how far it looks from
what was predicted, and the prediction moves toward what was seen. Errors are filed per cell, one
sample per frame the cell is in view (its faces' mean error), so how much of a place happens to be
in view does not move its numbers.

surprise — a cell's recent mean error: prediction error as a curiosity signal.
progress — how much that error has dropped: the older half of its recent errors minus the newer
half (learning progress; Schmidhuber 1991, Oudeyer et al. 2007). A thing he is learning has
progress while he learns it and then none (boredom: move on); a flickering light keeps surprise
high and progress at zero — surprising forever, teaching nothing (Burda et al. 2018's noisy TV).
A region's (a column of R x R cells) surprise and progress are the means over its cells.

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

from collections import deque

import numpy as np

__wiring__ = "STANDALONE"


class ViewModel:
    def __init__(self, region: int = 4, window: int = 32, rate: float = 0.35, prior_rate: float = 0.05,
                 bins: int = 6) -> None:
        self.region, self.window, self.rate, self.prior_rate, self.bins = region, window, rate, prior_rate, bins
        self.pred: dict[tuple, np.ndarray] = {}       # face -> predicted appearance
        self.prior: dict[tuple, np.ndarray] = {}      # (region, distance bin) -> its usual look
        self.errors: dict[tuple, deque] = {}          # cell -> recent per-frame mean errors
        self.cells: dict[tuple, set] = {}             # region -> its cells with errors
        self.last: float | None = None                # this frame's mean error (the percept's surprise)
        self.in_view: list = []                       # regions with an error sample this frame

    def faces(self, eye, rays: np.ndarray, depth: np.ndarray, reach: float) -> np.ndarray:
        """(n, 6) face keys — cell x, y, z, axis, side, distance bin — for surface pixels seen
        within `reach`."""
        s = np.asarray(eye, np.float64)[None, :] + rays * depth[:, None]
        cell = np.floor(s + rays * 0.02).astype(np.int64)
        axis = np.argmin(np.abs(s - np.round(s)), axis=1)
        side = (rays[np.arange(len(rays)), axis] > 0).astype(np.int64)
        dbin = np.clip((depth / max(reach, 1e-6) * self.bins).astype(np.int64), 0, self.bins - 1)
        return np.concatenate([cell, axis[:, None], side[:, None], dbin[:, None]], axis=1)

    def observe(self, eye, rays: np.ndarray, depth: np.ndarray, feats: np.ndarray, reach: float) -> float | None:
        """One frame's surface pixels (rays, depth, appearance; all within `reach`) -> this
        frame's mean error."""
        if len(feats) == 0:
            self.last, self.in_view = None, []
            return None
        keys = self.faces(eye, rays, depth, reach)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        sums = np.zeros((len(uniq), feats.shape[1]), np.float64)
        np.add.at(sums, inv, feats)
        looks = sums / np.bincount(inv, minlength=len(uniq))[:, None]
        per: dict[tuple, list] = {}
        R = self.region
        for key, look in zip(map(tuple, uniq.tolist()), looks):
            usual = (key[0] // R, key[1] // R, key[5])
            p = self.pred.get(key)
            if p is None:
                p = self.prior.get(usual)
                self.pred[key] = look.copy()
            else:
                self.pred[key] = p + self.rate * (look - p)
            if p is not None:
                per.setdefault(key[:3], []).append(float(np.linalg.norm(look - p)))
            q = self.prior.get(usual)
            self.prior[usual] = look.copy() if q is None else q + self.prior_rate * (look - q)
        regs = set()
        for cell, es in per.items():
            self.errors.setdefault(cell, deque(maxlen=self.window)).append(float(np.mean(es)))
            reg = (cell[0] // R, cell[1] // R)
            self.cells.setdefault(reg, set()).add(cell)
            regs.add(reg)
        self.last = float(np.mean([e for es in per.values() for e in es])) if per else None
        self.in_view = list(regs)
        return self.last

    # ------------------------------------------------------------------ per cell
    def cell_surprise(self, cell) -> float | None:
        e = self.errors.get(tuple(cell))
        if not e:
            return None
        e = list(e)
        return float(np.mean(e[len(e) // 2:]))

    def cell_progress(self, cell, least: int = 4) -> float | None:
        e = self.errors.get(tuple(cell))
        if e is None or len(e) < least:
            return None
        e = list(e)
        h = len(e) // 2
        return float(np.mean(e[:h]) - np.mean(e[h:]))

    # ------------------------------------------------------------------ per region
    def surprise(self, region) -> float | None:
        v = [x for x in (self.cell_surprise(c) for c in self.cells.get(tuple(region), ())) if x is not None]
        return float(np.mean(v)) if v else None

    def progress(self, region) -> float | None:
        v = [x for x in (self.cell_progress(c) for c in self.cells.get(tuple(region), ())) if x is not None]
        return float(np.mean(v)) if v else None

    def regions(self) -> list:
        return list(self.cells)
