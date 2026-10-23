"""cubbyllm/spatial/scene_file.py -- a scene as a file (order 0034): things (name, kind, cell) and named boxes, for any
world that wants Cubby to read it. Nothing in it is any one world's: a world's own exporter writes the file, and
the reader here knows only the format.

    {"format": "cubby.scene/1", "name": "...",
     "things":  [{"name": "forge", "kind": "building", "cell": [10, 9, 0]}, ...],
     "regions": [{"name": "the square", "lo": [8, 5, 0], "hi": [12, 9, 1]}, ...],
     "meta":    {...}}                       # free: the exporter's own record, kept and never read here

Cells are integers, x east, y north, z up (``scene.DIRECTIONS``); one thing per cell, one cell per name (after the
scene's normalization: case, spacing, a leading article). A file that breaks either is refused whole, never half
read.

``crop`` cuts a scene to a box round a centre, at the shard's size: the largest square, round the centre, of at most
``side`` cells on a side holding at most ``max_things`` things, within ``height`` cells of the centre's floor. The
cut is moved so its low corner is the origin, since the trunk learned its places in a box from 0 (the spatial
shard's 24 x 24 x 3). The things it leaves out are counted in its meta.

Wired: STANDALONE (``validation/exp_s12_scene_files.py`` reads scene files; ``Scene.from_json`` / ``to_json``).
"""
from __future__ import annotations

import json
import os

from ..core.protocols import Wiring

FORMAT = "cubby.scene/1"


def _cell(v, what: str) -> tuple[int, int, int]:
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(c, int) and not isinstance(c, bool) for c in v)):
        raise ValueError(f"{what}: a cell is three integers, got {v!r}")
    return int(v[0]), int(v[1]), int(v[2])


def load(obj):
    """A scene file's dict, from a dict, a JSON string or a path."""
    if isinstance(obj, dict):
        return obj
    text = str(obj)
    if not text.lstrip().startswith("{") and os.path.exists(text):
        with open(text, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(text)


def from_json(obj, ssp=None):
    """A ``Scene`` from a scene file (``load``'s forms). Refuses the file whole on a wrong format tag, a malformed
    cell, two things at one cell or one name twice."""
    from .scene import Scene, _norm
    d = load(obj)
    if d.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file: format {d.get('format')!r}")
    sc = Scene(ssp)
    seen: dict = {}
    for i, t in enumerate(d.get("things") or []):
        name, cell = str(t["name"]), _cell(t["cell"], f"things[{i}]")
        key = _norm(name)
        if key in seen:
            raise ValueError(f"things[{i}]: the name {name!r} is already things[{seen[key]}]")
        if sc.at(cell) is not None:
            raise ValueError(f"things[{i}]: cell {cell} already holds {sc.at(cell)!r}")
        seen[key] = i
        sc.place(name, cell, t.get("kind"))
    for i, r in enumerate(d.get("regions") or []):
        sc.region(str(r["name"]), _cell(r["lo"], f"regions[{i}].lo"), _cell(r["hi"], f"regions[{i}].hi"))
    sc.file_name, sc.meta = d.get("name"), dict(d.get("meta") or {})
    return sc


def to_json(sc) -> dict:
    """A scene file's dict (names as the scene keeps them, normalized; things along the cells' sort order)."""
    things = [{"name": n, "kind": sc.kinds.get(n), "cell": list(c)} for c, n in sorted((c, n) for n, c in sc.things.items())]
    regions = [{"name": r, "lo": list(lo), "hi": list(hi)} for r, (lo, hi) in sorted(sc.regions.items())]
    out = {"format": FORMAT, "name": getattr(sc, "file_name", None), "things": things, "regions": regions}
    meta = getattr(sc, "meta", None)
    if meta:
        out["meta"] = dict(meta)
    return out


def crop(sc, centre, max_things: int = 32, side: int = 24, height: int = 3):
    """``sc`` cut to the largest square round ``centre`` (at most ``side`` on a side) holding at most ``max_things``
    things within ``height`` cells up from the centre's floor, moved so its low corner is (0, 0, 0). Regions are cut
    to the box (a region outside it is dropped). The cut's meta records the box, the shift and what was left out."""
    from .scene import Scene
    cx, cy, cz = _cell(list(centre), "centre")

    def inside(c, r):
        return abs(c[0] - cx) <= r and abs(c[1] - cy) <= r and cz <= c[2] < cz + height

    best = 0
    for r in range(0, (side - 1) // 2 + 1):
        if sum(1 for c in sc.cells if inside(c, r)) > max_things:
            break
        best = r
    lo, hi = (cx - best, cy - best, cz), (cx + best, cy + best, cz + height - 1)
    out = Scene(sc.ssp)
    for c, n in sorted(sc.cells.items()):
        if inside(c, best):
            out.place(n, (c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]), sc.kinds.get(n))
    for name, (rlo, rhi) in sorted(sc.regions.items()):
        a = tuple(max(rlo[i], lo[i]) for i in range(3))
        b = tuple(min(rhi[i], hi[i]) for i in range(3))
        if all(a[i] <= b[i] for i in range(3)):
            out.region(name, tuple(a[i] - lo[i] for i in range(3)), tuple(b[i] - lo[i] for i in range(3)))
    out.file_name = getattr(sc, "file_name", None)
    out.meta = {**dict(getattr(sc, "meta", None) or {}),
                "crop": {"centre": [cx, cy, cz], "half_width": best, "lo": list(lo), "hi": list(hi), "shift": list(lo),
                         "kept": len(out.things), "left_out": len(sc.things) - len(out.things)}}
    return out


__wiring__ = Wiring.STANDALONE
