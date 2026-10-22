"""The cortex contract (design §5): one step per frame, a percept out.

    cortex = SensoryCortex(config)
    p = cortex.step(frame=rgb|None, depth=None, touch=None, proprio=None, action=None, arousal=0.0)
    p.salience   (gh, gw) spike rates      p.fixation   (yaw, pitch) degrees: the next look
    p.features   the retina's channels     p.occupancy  {"solid": cells, "free": cells} this frame
    (the cortex's own map: cortex.map)     p.objects    [(row, col, kind)] for the surface pixels seen
    p.touch      [(site, kind, predicted feel, felt)] — what sight predicted, then what was felt
    p.surprise   the forward model's error this frame; p.progress its learning progress (cortex.view)

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
from .forward import ViewModel
from .what import KindCodebook, appearance
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
    surprise: float | None = None     # stage 6 (forward model: cortex.view)
    progress: float | None = None


class SensoryCortex:
    def __init__(self, grid: tuple[int, int] = (12, 16), fov: float = 70.0, normalise: bool = False,
                 weights: dict | None = None, start=(0.0, 0.0, 0.0), relocalise: bool = False,
                 eye_height: float = 1.62, view_near: float = 0.65) -> None:
        self.relocalise = relocalise          # point-to-plane matching against the surfaces seen so far
        self.fov = fov
        self.retina = Retina(grid)
        self.salience = Salience(weights=weights, normalise=normalise)
        self.attention = Attention(grid, fov)
        self.map = OccupancyMap()
        self.surfaces = SurfaceMap()
        self.kinds = KindCodebook()
        self.view = ViewModel()                     # the forward model of the view; its errors are curiosity
        self.view_near = view_near                  # (faces near enough not to be lost in the haze)
        self.cell_kind: dict[tuple, int] = {}       # the kind each cell in its map is seen as (by vote)
        self._votes: dict[tuple, dict] = {}
        self._frames = 0
        self.eye_height = eye_height
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
            kinds = self._see_kinds(frame, rays, depth, far)
        else:
            kinds = None
        felt = []
        if touch and "feet" in touch and self.cell_kind:          # what is underfoot, as seen, is what is felt
            feet = self.path.pos - np.array([0.0, 0.0, self.eye_height + 0.05])
            k = self.cell_kind.get(tuple(int(v) for v in np.floor(feet)))
            if k is not None:
                predicted = self.kinds.predict_feel(k)
                self.kinds.touch(k, touch["feet"])
                felt.append(("feet", k, predicted, list(touch["feet"])))
        surprise = progress = None
        if kinds is not None:
            surprise = self.view.last
            lp = [v for v in (self.view.progress(r) for r in self.view.in_view) if v is not None]
            progress = float(np.mean(lp)) if lp else None
        return Percept(salience=sal, fixation=look, tile=tile, features=feats, occupancy=occ,
                       objects=kinds or [], touch=felt or [(k, None, None, v) for k, v in (touch or {}).items()],
                       surprise=surprise, progress=progress)

    def _see_kinds(self, frame, rays, depth, far, every: int = 2):
        """Kinds per surface pixel (near enough not to be washed by fog), filed on their cells."""
        h, w = depth.shape
        sel = np.zeros((h, w), bool)
        sel[::every, ::every] = True
        near = sel & (depth < 0.65 * far)
        if not near.any():
            self.view.last, self.view.in_view = None, []
            return []
        looks = appearance(np.asarray(frame)[near])
        ks = self.kinds.assign(looks)
        clear = depth[near] < self.view_near * far
        self.view.observe(self.path.pos, rays[near][clear], depth[near][clear], looks[clear], self.view_near * far)
        self._frames += 1
        if self._frames % 25 == 0 and self.kinds.consolidate():       # kinds that drifted together are one
            for c, votes in self._votes.items():
                merged: dict = {}
                for k, n in votes.items():
                    merged[self.kinds.canon(k)] = merged.get(self.kinds.canon(k), 0) + n
                self._votes[c] = merged
                self.cell_kind[c] = max(merged, key=merged.get)
        pts = self.path.pos[None, :] + rays[near] * (depth[near][:, None] + 0.02)
        cells = np.floor(pts).astype(np.int64)
        for c, k in zip(map(tuple, cells.tolist()), ks.tolist()):      # a cell's kind: what most looks at it said
            votes = self._votes.setdefault(c, {})
            votes[k] = votes.get(k, 0) + 1
            self.cell_kind[c] = max(votes, key=votes.get)
        ys, xs = np.nonzero(near)
        return list(zip(ys.tolist(), xs.tolist(), ks.tolist()))
