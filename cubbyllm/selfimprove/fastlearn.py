"""cubbyllm/selfimprove/fastlearn.py -- a fast learner beside the frozen trunk: STDP marks the synapses that produced
an answer, the VM's verdict decides (a three-factor rule; order 0053, H-A16). Numpy only; off unless
``CB_FASTLEARN=1`` (its harness, ``validation/selfimprove/fastlearn_day.py``); nothing on the serving path.

  encoder   a fixed random up-projection (``neurons`` = 8 x d) of the trunk's state, rms-normalised, presented for
            ``T`` steps to GIF neurons (``gif.py``); the activity is each neuron's spike count. A global current offset
            is calibrated so that ``rest`` (2%) of the neurons fire on the day's states (H-A5's sparsity). The encoder
            never learns. ``dense``: the same projection, every unit's value passed on (H-A5's size-matched control)
  readout   plastic weights W (neurons x properties); a candidate's bias is its property's ``act @ W[:, p]`` (a word
            naming several properties: the largest, and that property is the one it marks); its score the trunk's
            own plus ``gain`` x bias
  trace     at each choice the trace decays by exp(-1 / tau) and the firing neurons' synapses to the chosen
            property are marked by their counts
  rule      "three": the verdict through a striatum of its own (+1 right, 0 refused, -5 wrong; delta = reward -
            expected, per skill signature): dW = eta delta trace. "stdp": dW = eta trace at every choice, whatever
            the verdict (cubemind's rule). Each neuron's outgoing weights are kept within ``norm`` (H-A1's failure
            class is growth without limit)
With every weight zero the scores are the trunk's own, and the greedy choice is the base's, bit for bit.

Wired: STANDALONE (order 0053's harness; the brain's SNN slot is a later order's).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .gif import GIF

__wiring__ = "STANDALONE"

OUTCOME = {"right": "certified", "wrong": "wrong", "refused": "refused", "caught": "refused"}


def on() -> bool:
    """``CB_FASTLEARN=1``: the learner runs (its harness). Off by default."""
    return os.environ.get("CB_FASTLEARN") == "1"


@dataclass(frozen=True)
class Settings:
    neurons: int = 8192
    T: int = 8                   # steps each state is presented for
    rest: float = 0.02           # the share of neurons firing on the day's states
    L: int = 16
    gif_tau: float = 10.0
    alpha: float = 0.01
    gain: float = 1.0
    eta: float = 0.01
    tau: float = 1.0             # the trace's time constant, in choices
    norm: float = 1.0            # each neuron's outgoing weights' bound
    temperature: float = 1.0


class Encoder:
    def __init__(self, d: int, s: Settings, seed: int = 0, dense: bool = False) -> None:
        self.s, self.dense = s, dense
        self.W = (np.random.default_rng(seed).normal(size=(s.neurons, d)) / math.sqrt(d)).astype(np.float32)
        self.offset = 0.0
        self.gif = GIF(L=s.L, tau=s.gif_tau, threshold=1.0, alpha=s.alpha)

    def currents(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float32).reshape(-1, self.W.shape[1])
        h = h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6)
        return h @ self.W.T + np.float32(self.offset)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """(n, neurons) activity for n states (or (neurons,) for one)."""
        one = np.asarray(h).ndim == 1
        c = self.currents(h)
        if self.dense:
            act = c
        else:
            spikes, _ = self.gif.run(np.repeat(c[:, None], self.s.T, axis=1))
            act = spikes.sum(1)
        return act[0] if one else act

    def calibrate(self, H: np.ndarray, iters: int = 30) -> float:
        """The offset at which ``rest`` of the neurons fire (count > 0) over the states H; the share it reached."""
        if self.dense:
            return 1.0
        lo, hi = -20.0, 20.0
        for _ in range(iters):
            self.offset = (lo + hi) / 2
            share = float((self(H) > 0).mean())
            lo, hi = (lo, self.offset) if share > self.s.rest else (self.offset, hi)
        self.offset = (lo + hi) / 2
        return float((self(H) > 0).mean())


class FastLearner:
    def __init__(self, enc: Encoder, props: list[str], s: Settings, rule: str = "three", seed: int = 0) -> None:
        from ..reasoning.striatum import Striatum
        if rule not in ("three", "stdp"):
            raise ValueError(f"rule: 'three' or 'stdp', not {rule!r}")
        self.enc, self.s, self.rule = enc, s, rule
        self.col = {p: i for i, p in enumerate(dict.fromkeys(props))}
        self.W = np.zeros((s.neurons, len(self.col)), dtype=np.float64)
        self.trace = np.zeros_like(self.W)
        self.striatum = Striatum(seed=seed)                    # its own: never the serving one
        self.rng = np.random.default_rng(seed)
        self.deltas: list[float] = []

    def biases(self, act: np.ndarray, cand_props: list[list[str]]) -> tuple[np.ndarray, list[str | None]]:
        """Each candidate's bias and the property it marks (the largest of a word naming several)."""
        b, which = np.zeros(len(cand_props)), []
        for i, ps in enumerate(cand_props):
            cols = [(self.col[p], p) for p in ps if p in self.col]
            if not cols:
                which.append(None)
                continue
            vals = [float(act @ self.W[:, c]) for c, _ in cols]
            k = int(np.argmax(vals))
            b[i] = vals[k]
            which.append(cols[k][1])
        return b, which

    def choose(self, trunk_scores, act, cand_props, greedy: bool) -> tuple[int, str | None]:
        b, which = self.biases(act, cand_props)
        z = np.asarray(trunk_scores, dtype=np.float64) + self.s.gain * b
        if greedy:
            i = int(np.argmax(z))
        else:
            p = np.exp((z - z.max()) / self.s.temperature)
            i = int(self.rng.choice(len(z), p=p / p.sum()))
        return i, which[i]

    def learn(self, act: np.ndarray, prop: str | None, verdict: str, key: str) -> float:
        """One verdict on the last choice; returns the third factor used (delta; 1.0 for plain STDP)."""
        self.trace *= math.exp(-1.0 / self.s.tau)
        if prop is not None and prop in self.col:
            fire = np.where(act > 0, act, 0.0)
            self.trace[:, self.col[prop]] += fire
        if self.rule == "three":
            third = self.striatum.reward("", key, OUTCOME[verdict], source="vm")
        else:
            third = 1.0
        self.W += self.s.eta * third * self.trace
        n = np.linalg.norm(self.W, axis=1)
        over = n > self.s.norm
        if over.any():
            self.W[over] *= (self.s.norm / n[over])[:, None]
        self.deltas.append(float(third))
        return float(third)


__all__ = ["Encoder", "FastLearner", "OUTCOME", "Settings", "on"]
