"""cubbyllm/selfimprove/gif.py -- the generalized integrate-and-fire neuron, ported (order 0053; port, don't link).

From cubemind's ``brain/gif_neuron.py`` (itself from aura-hybrid-pre-model), the numpy path only (no grilly v1):
multi-bit spikes (up to L a step) and an adaptive threshold. Per step, for currents i_t:

    v = v * decay + i_t,  decay = exp(-dt / tau)
    v = clip(v, -2 L theta, 2 L theta)
    spike = clip(floor(v / (theta + 1e-6)), 0, L)
    v = v - spike * theta
    theta = theta + alpha * spike - alpha * (theta - threshold)        (alpha > 0)

Unlike cubemind's class, this one takes the currents directly (the projection is its caller's: 0053's encoder is a
fixed random up-projection), so ``run`` is the loop above alone; ``project`` is cubemind's Xavier-initialised linear,
kept so the recorded fixture (``tests/fixtures/gif_cubemind_spikes.json``) checks the whole neuron.

Wired: STANDALONE (order 0053's learner, ``cubbyllm/selfimprove/fastlearn.py``).
"""
from __future__ import annotations

import math

import numpy as np

__wiring__ = "STANDALONE"


class GIF:
    def __init__(self, L: int = 16, dt: float = 1.0, tau: float = 10.0, threshold: float = 1.0, alpha: float = 0.01):
        self.L, self.tau, self.threshold, self.alpha = int(L), float(tau), float(threshold), float(alpha)
        self.decay = math.exp(-dt / tau)

    def run(self, currents: np.ndarray, state=None):
        """currents (batch, T, n) -> (spikes (batch, T, n), (v, theta))."""
        currents = np.asarray(currents, dtype=np.float32)
        b, T, n = currents.shape
        if state is None:
            v = np.zeros((b, n), dtype=np.float32)
            theta = np.full((b, n), self.threshold, dtype=np.float32)
        else:
            v, theta = state[0].copy(), state[1].copy()
        out = np.empty_like(currents)
        for t in range(T):
            v = v * self.decay + currents[:, t]
            lim = self.L * theta * 2.0
            v = np.clip(v, -lim, lim)
            s = np.clip(np.floor(v / (theta + 1e-6)), 0, self.L).astype(np.float32)
            v = v - s * theta
            if self.alpha > 0:
                theta = theta + self.alpha * s - self.alpha * (theta - self.threshold)
            out[:, t] = s
        return out, (v, theta)


def project(input_dim: int, hidden_dim: int, seed: int = 42) -> np.ndarray:
    """cubemind's Xavier-uniform projection (hidden, input), float32, from ``default_rng(seed)``."""
    limit = math.sqrt(6.0 / (input_dim + hidden_dim))
    return np.random.default_rng(seed).uniform(-limit, limit, (hidden_dim, input_dim)).astype(np.float32)


__all__ = ["GIF", "project"]
