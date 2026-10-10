"""The sensory cortex, stage 1 (retina, salience, attention) on synthetic frames — no game.
Salience is physical: bright, saturated, contrasting, changing and moving things fire more,
and a uniform view fires nothing in particular.
Run: python -m pytest standin/tests/test_cortex.py -q"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cortex import SensoryCortex  # noqa: E402
from cortex.retina import Retina  # noqa: E402
from cortex.salience import Attention, Salience  # noqa: E402


def frame(bg=(40, 44, 52), h=48, w=64):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = bg
    return f


def tile_of(y, x, h=48, w=64, grid=(12, 16)):
    return y * grid[0] // h, x * grid[1] // w


def test_a_bright_saturated_patch_wins_and_a_uniform_view_does_not():
    f = frame()
    f[20:28, 40:48] = (120, 255, 140)                        # a glowing green block on dull ground
    c = SensoryCortex()
    p = c.step(f)
    assert 5 <= p.tile[0] <= 6 and 10 <= p.tile[1] <= 11       # one of the four tiles it covers
    assert p.salience[p.tile] > 3 * np.median(p.salience) + 1e-6
    flat = SensoryCortex().step(frame())
    assert flat.salience.max() < 0.05                         # nothing stands out


def test_brightness_colour_and_contrast_each_raise_salience():
    sal = Salience()
    r = Retina()
    def peak(patch):
        r.reset()
        f = frame()
        f[16:32, 24:40] = patch
        return sal(r(f))[tile_of(24, 32)]
    dark, grey, bright, colour = peak((30, 30, 30)), peak((90, 90, 90)), peak((230, 230, 230)), peak((230, 60, 60))
    assert bright > grey and colour > grey                   # brighter, and more colourful, drive harder


def test_motion_and_change_draw_the_eye():
    c = SensoryCortex()
    f = frame()
    f[10:16, 10:16] = (200, 200, 200)
    f[30:36, 50:56] = (200, 200, 200)
    c.step(f)
    g = frame()
    g[10:16, 10:16] = (200, 200, 200)
    g[30:36, 44:50] = (200, 200, 200)                         # the second square moved
    p = c.step(g)
    assert p.features["motion"][tile_of(33, 47)] > p.features["motion"][tile_of(13, 13)]


def test_attention_moves_on_from_where_it_just_looked():
    sal = np.zeros((12, 16), np.float32)
    sal[3, 4], sal[8, 12] = 0.9, 0.7
    att = Attention((12, 16))
    first, look1 = att(sal)
    second, look2 = att(sal)
    assert first == (3, 4) and second == (8, 12)               # inhibition of return
    assert look1[0] < 0 < look1[1] and look2[0] > 0 > look2[1]  # up-left, then down-right


def test_arousal_sharpens_the_response():
    f = frame()
    f[20:28, 40:48] = (150, 150, 150)
    calm = SensoryCortex().step(f, arousal=0.0).salience.max()
    alert = SensoryCortex().step(f, arousal=1.0).salience.max()
    assert alert > calm


def test_a_floor_seen_in_depth_becomes_solid_cells_under_free_ones():
    import math
    from cortex.where import Camera, OccupancyMap
    cam = Camera(32, 24, 70.0)
    eye = np.array([0.5, 0.5, 2.62])                          # eyes 1.62 above a floor whose top is z = 1
    rays = cam.rays(0.0, -math.radians(40))
    depth = np.full((24, 32), 6.0, np.float32)
    down = rays[..., 2] < -1e-3
    t = (1.0 - eye[2]) / rays[..., 2]
    depth[down] = np.minimum(t[down], 6.0)
    m = OccupancyMap()
    solid, free = m.integrate(eye, rays, depth, 6.0)
    assert solid and all(c[2] == 0 for c in solid)            # the cells whose tops are the floor
    assert free and all(c[2] >= 1 for c in free)              # everything between is air
    assert not (solid & free)


def test_path_integration_follows_velocity():
    c = SensoryCortex(start=(1.0, 2.0, 3.0))
    for _ in range(10):
        c.step(proprio={"velocity": (0.1, 0.0, 0.0)})
    assert np.allclose(c.path.pos, (2.0, 2.0, 3.0))


def _march(solid: set, eye, rays, sight=6.0, step=0.02):
    """A tiny reference ray caster for the tests: depth to the first solid cell."""
    d = rays.reshape(-1, 3)
    out = np.full(len(d), sight, np.float32)
    for i, r in enumerate(d):
        t = step
        while t < sight:
            p = np.floor(np.asarray(eye) + r * t).astype(int)
            if tuple(p) in solid:
                out[i] = t
                break
            t += step
    return out.reshape(rays.shape[:2])


def test_a_place_recognised_corrects_a_drifted_position():
    import math
    from cortex.where import Camera, OccupancyMap
    solid = {(x, y, 0) for x in range(-6, 7) for y in range(-6, 7)}
    solid |= {(3, y, z) for y in range(-2, 3) for z in (1, 2)} | {(x, 3, 1) for x in range(-2, 2)}
    cam = Camera(24, 18, 70.0)
    eye = np.array([0.5, 0.5, 2.62])
    m = OccupancyMap()
    for yaw in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        rays = cam.rays(yaw, -0.35)
        m.integrate(eye, rays, _march(solid, eye, rays), 6.0)
    rays = cam.rays(0.3, -0.35)
    depth = _march(solid, eye, rays)
    shift = m.align(eye + np.array([0.2, 0.0, 0.0]), rays, depth, 6.0)
    assert shift[0] < 0                                       # pushed back toward where it really is


def test_kinds_are_learned_from_looks_and_shading_does_not_split_them():
    from cortex.what import KindCodebook, appearance
    cb = KindCodebook()
    green_lit = np.array([[90, 180, 80]] * 20, np.uint8)
    green_shade = (green_lit * 0.65).astype(np.uint8)          # the same block's side, in shadow
    white = np.array([[235, 238, 245]] * 20, np.uint8)
    k1 = cb.assign(appearance(green_lit))
    k2 = cb.assign(appearance(green_shade))
    k3 = cb.assign(appearance(white))
    assert len(set(k1.tolist())) == 1 and set(k1.tolist()) == set(k2.tolist())
    assert set(k3.tolist()).isdisjoint(set(k1.tolist()))


def test_what_was_felt_is_predicted_for_a_kind_that_looks_alike():
    from cortex.what import KindCodebook, appearance
    cb = KindCodebook(radius=0.3)
    ice = cb.assign(appearance(np.array([[180, 225, 250]], np.uint8)))[0]
    rock = cb.assign(appearance(np.array([[120, 120, 125]], np.uint8)))[0]
    cb.touch(ice, [0.95, 0.7, 0.05, 0, -0.8, 0, 0.85])
    cb.touch(rock, [0.0, 0.75, 0.3, 0, 0.0, 0, 0.85])
    bluish = cb.assign(appearance(np.array([[170, 215, 245]], np.uint8)))[0]
    guess = cb.predict_feel(bluish)
    assert guess is not None and guess[0] > 0.5                 # looks like ice: expected to be slippery


def test_kinds_that_drift_together_become_one_and_keep_what_was_felt():
    from cortex.what import KindCodebook
    cb = KindCodebook(radius=0.5)
    a = cb.assign(np.array([[1.0, 1.0, 1.0]] * 30, np.float32))[0]
    b = cb.assign(np.array([[1.6, 1.0, 1.0]], np.float32))[0]      # first seen off to one side: its own kind
    cb.touch(b, [0.9, 0, 0, 0, 0, 0, 1])
    assert a != b and cb.consolidate() == {}
    cb.protos[b] = [1.2, 1.0, 1.0]                                  # ...then the two drift together
    assert cb.consolidate() == {b: a}
    assert cb.assign(np.array([[1.25, 1.0, 1.0]], np.float32))[0] == a
    assert cb.predict_feel(a)[0] > 0.5                              # the feel goes with the merge
