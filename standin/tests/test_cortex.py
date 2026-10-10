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
