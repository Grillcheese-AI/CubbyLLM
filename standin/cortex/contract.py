"""The cortex contract (design §5): one step per frame, a percept out.

    cortex = SensoryCortex(config)
    p = cortex.step(frame=rgb|None, depth=None, touch=None, proprio=None, action=None, arousal=0.0)
    p.salience   (gh, gw) spike rates      p.fixation   (yaw, pitch) degrees: the next look
    p.features   the retina's channels     p.occupancy  {"solid": cells, "free": cells} this frame
    (the cortex's own map: cortex.map)     p.objects / p.surprise: later stages

proprio, for "where": {"eye": (x, y, z)} when the position is given (sighted), or
{"velocity": (dx, dy, dz)} to integrate it (blind, from the start position); and always
{"yaw": radians, "pitch": radians}. depth: (h, w) in the frame's units, `sight` where nothing.

Additive only, opt-in, its own state; with no frame it returns an empty percept.

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retina import Retina
from .salience import Attention, Salience
from .where import Camera, OccupancyMap, PathIntegrator, SurfaceMap

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
                 weights: dict | None = None, start=(0.0, 0.0, 0.0), relocalise: bool = False) -> None:
        self.relocalise = relocalise          # point-to-plane matching against the surfaces seen so far
        self.fov = fov
        self.retina = Retina(grid)
        self.salience = Salience(weights=weights, normalise=normalise)
        self.attention = Attention(grid, fov)
        self.map = OccupancyMap()
        self.surfaces = SurfaceMap()
        self.path = PathIntegrator(start)
        self.corrections = 0
        self._cams: dict[tuple, Camera] = {}

    def step(self, frame=None, depth=None, touch=None, proprio=None, action=None, arousal: float = 0.0,
             sight: float | None = None) -> Percept:
        pro = proprio or {}
        if "eye" in pro:
            self.path.reset(pro["eye"])
        elif "velocity" in pro:
            self.path.step(pro["velocity"])
        if frame is None:
            return Percept()
        feats = self.retina(frame)
        sal = self.salience(feats, arousal)
        tile, look = self.attention(sal)
        occ = None
        if depth is not None and "yaw" in pro:
            h, w = depth.shape
            cam = self._cams.get((w, h))
            if cam is None:
                cam = self._cams[(w, h)] = Camera(w, h, self.fov)
            rays = cam.rays(pro["yaw"], pro["pitch"])
            far = sight if sight is not None else float(depth.max())
            pts, nrm = self.surfaces.surface(self.path.pos, rays, depth, far)
            if "eye" not in pro and self.relocalise:                 # integrated, not given: re-localise
                shift = self.surfaces.match(self.path.pos + pts, nrm)
                self.path.reset(self.path.pos + shift)
                self.corrections += float(np.abs(shift).sum() > 1e-3)
            self.surfaces.add(self.path.pos + pts, nrm)
            solid, free = self.map.integrate(self.path.pos, rays, depth, far)
            occ = {"solid": solid, "free": free}
        return Percept(salience=sal, fixation=look, tile=tile, features=feats, occupancy=occ,
                       touch=[(k, v) for k, v in (touch or {}).items()])
