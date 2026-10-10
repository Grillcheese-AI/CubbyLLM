"""The cortex contract (design §5): one step per frame, a percept out.

    cortex = SensoryCortex(config)
    p = cortex.step(frame=rgb|None, depth=None, touch=None, proprio=None, action=None, arousal=0.0)
    p.salience   (gh, gw) spike rates      p.fixation   (yaw, pitch) degrees: the next look
    p.features   the retina's channels     p.occupancy / p.objects / p.touch / p.surprise: later stages

Additive only, opt-in, its own state; with no frame it returns an empty percept.

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retina import Retina
from .salience import Attention, Salience

__wiring__ = "STANDALONE"


@dataclass
class Percept:
    salience: np.ndarray | None = None
    fixation: tuple[float, float] | None = None
    tile: tuple[int, int] | None = None
    features: dict = field(default_factory=dict)
    occupancy: object = None          # stage 3 (where)
    objects: list = field(default_factory=list)       # stage 4 (what)
    touch: list = field(default_factory=list)         # stage 4/5 (touch)
    surprise: float | None = None     # stage 6 (forward model)
    progress: float | None = None


class SensoryCortex:
    def __init__(self, grid: tuple[int, int] = (12, 16), fov: float = 70.0, normalise: bool = False,
                 weights: dict | None = None) -> None:
        self.retina = Retina(grid)
        self.salience = Salience(weights=weights, normalise=normalise)
        self.attention = Attention(grid, fov)

    def step(self, frame=None, depth=None, touch=None, proprio=None, action=None, arousal: float = 0.0) -> Percept:
        if frame is None:
            return Percept()
        feats = self.retina(frame)
        sal = self.salience(feats, arousal)
        tile, look = self.attention(sal)
        return Percept(salience=sal, fixation=look, tile=tile, features=feats,
                       touch=[(k, v) for k, v in (touch or {}).items()])
