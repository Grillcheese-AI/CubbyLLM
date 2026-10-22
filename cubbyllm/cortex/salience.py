"""Salience and attention — which part of the view drives the cells hardest, and where to look.

Every tile drives a GIF neuron (generalised integrate-and-fire with multi-bit spikes and an
adaptive threshold, ported from cubemind/brain/gif_neuron.py); its spike rate is the tile's
salience. The drive is the retina's own response — brightness, colour, centre–surround contrast,
change and motion — minus a share of the whole view's mean drive (lateral inhibition: what stands
out, not what is merely there). Nothing says "glow matters": a glowing block is bright, saturated
and contrasts with its ground, so it fires more. Arousal (neurochemistry) raises the gain.

Attention picks the most salient tile not recently looked at (winner-take-all with inhibition of
return) and proposes it as a fixation: a small look, in degrees, relative to where the eyes point
— an action like any other, so where to look can be learned.

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

import numpy as np

__wiring__ = "STANDALONE"

WEIGHTS = {"lum": 0.6, "sat": 1.0, "contrast": 1.0, "transient": 1.5, "motion": 1.5}       # normalised maps
RAW_WEIGHTS = {"lum": 0.6, "sat": 2.0, "contrast": 2.5, "transient": 3.0, "motion": 3.0}   # raw responses


class Salience:
    def __init__(self, weights: dict | None = None, steps: int = 8, levels: int = 16, tau: float = 6.0,
                 theta: float = 0.25, alpha: float = 0.05, inhibit: float = 0.8, normalise: bool = False) -> None:
        self.w = dict(WEIGHTS if normalise else RAW_WEIGHTS, **(weights or {}))
        self.normalise = normalise
        self.steps, self.levels = steps, levels
        self.decay = float(np.exp(-1.0 / tau))
        self.theta0, self.alpha, self.inhibit = theta, alpha, inhibit

    def drive(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        """The retina's responses, weighted, then lateral inhibition. Optionally (normalise=True) each
        channel is first normalised the Itti–Koch way (Itti, Koch & Niebur 1998): scaled to its own
        peak and weighted by how much that peak stands above the channel's other local peaks. S0
        measures both; the raw responses rank glow, colour and motion higher, so they are the
        default (bench/s0_salience.py in the sandbox repo, results/s0_salience.json)."""
        norm = self._normalise if self.normalise else (lambda m: m)
        d = sum(self.w[k] * norm(np.abs(feats[k])) for k in self.w if k in feats)
        return np.maximum(0.0, d - self.inhibit * d.mean())

    @staticmethod
    def _normalise(m: np.ndarray, floor: float = 0.08) -> np.ndarray:
        top = float(m.max())
        if top <= 0:
            return m
        n = m / max(top, floor)
        p = np.pad(n, 1, mode="constant", constant_values=-1)
        neigh = np.max(np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]), axis=0)
        peaks = n[(n >= neigh) & (n > 0.05)]
        others = np.sort(peaks)[:-1] if peaks.size > 1 else np.zeros(0)
        mbar = float(others.mean()) if others.size else 0.0
        return n * (1.0 - mbar) ** 2

    def __call__(self, feats: dict[str, np.ndarray], arousal: float = 0.0) -> np.ndarray:
        """Spike rate per tile in [0, 1]: the salience map."""
        current = self.drive(feats) * (1.0 + 0.5 * float(np.clip(arousal, -1, 1)))
        v = np.zeros_like(current)
        theta = np.full_like(current, self.theta0)
        spikes = np.zeros_like(current)
        L = self.levels
        for _ in range(self.steps):
            v = v * self.decay + current
            v = np.clip(v, -L * theta * 2, L * theta * 2)
            s = np.clip(np.floor(v / theta), 0, L)
            v = v - s * theta
            theta = theta + self.alpha * s * self.theta0 - self.alpha * (theta - self.theta0)
            spikes += s
        return spikes / (self.steps * L)


class Attention:
    """Winner-take-all over salience with inhibition of return."""

    def __init__(self, shape: tuple[int, int], fov: float = 70.0, aspect: float = 0.75,
                 ior: float = 1.0, fade: float = 0.85) -> None:
        self.gh, self.gw = shape
        self.fov, self.vfov = fov, fov * aspect
        self.ior_map = np.zeros(shape, np.float32)
        self.ior, self.fade = ior, fade

    def __call__(self, sal: np.ndarray) -> tuple[tuple[int, int], tuple[float, float]]:
        """(tile, (yaw degrees, pitch degrees)) of the next look; the tile is then inhibited."""
        score = sal * (1.0 - np.clip(self.ior_map, 0, 1))
        i = int(np.argmax(score))
        ty, tx = divmod(i, self.gw)
        self.ior_map *= self.fade
        y0, y1, x0, x1 = max(0, ty - 1), min(self.gh, ty + 2), max(0, tx - 1), min(self.gw, tx + 2)
        self.ior_map[y0:y1, x0:x1] = np.maximum(self.ior_map[y0:y1, x0:x1], 0.5 * self.ior)
        self.ior_map[ty, tx] = self.ior
        u = (tx + 0.5) / self.gw * 2 - 1
        v = 1 - (ty + 0.5) / self.gh * 2
        return (ty, tx), (u * self.fov / 2, v * self.vfov / 2)
