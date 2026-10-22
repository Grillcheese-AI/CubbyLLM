"""Retina — what a frame does to the first layer of cells, per tile of a retinotopic grid.

Ported from cubemind/perception/bio_vision.py (opponent colour after Derrington et al. 1984,
Weber–Fechner luminance with a transient response, motion by frame difference), vectorised, on
RGB frames, plus a centre–surround (difference-of-Gaussians) contrast channel, which is new.

Channels, each a (gh, gw) map, all non-negative except the two opponents:
    lum        Weber–Fechner brightness, k·log(1 + I/I0)
    transient  how much that brightness changed since the last frame (lights on/off)
    rg, by     red–green and blue–yellow opponency (signed)
    sat        chroma: how far from grey (the opponents' magnitude)
    contrast   centre–surround: a bright thing on a dark ground, an edge
    motion     frame difference, at the pixel, pooled per tile

Wired: STANDALONE (part of the cortex plugin).
"""
from __future__ import annotations

import numpy as np

__wiring__ = "STANDALONE"

CHANNELS = ("lum", "transient", "rg", "by", "sat", "contrast", "motion")


def _box(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 box, edges clamped (separable, via cumulative sums)."""
    if r <= 0:
        return img
    p = np.pad(img, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    h, w = img.shape
    s = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return s / (k * k)


def _pool(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Mean per tile of a gh x gw grid (frames whose size is not a multiple are cropped evenly)."""
    h, w = img.shape[:2]
    th, tw = h // gh, w // gw
    y0, x0 = (h - th * gh) // 2, (w - tw * gw) // 2
    crop = img[y0:y0 + th * gh, x0:x0 + tw * gw]
    return crop.reshape(gh, th, gw, tw, *img.shape[2:]).mean(axis=(1, 3))


class Retina:
    def __init__(self, grid: tuple[int, int] = (12, 16), k: float = 2.0, i0: float = 50.0,
                 centre: int = 1, surround: int = 4) -> None:
        self.gh, self.gw = grid
        self.k, self.i0 = k, i0
        self.centre, self.surround = centre, surround
        self._prev_lum: np.ndarray | None = None       # tile brightness, last frame
        self._prev_px: np.ndarray | None = None        # pixel brightness, last frame

    def reset(self) -> None:
        self._prev_lum = self._prev_px = None

    def __call__(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        f = np.asarray(rgb, np.float32)
        if f.ndim == 2:
            f = np.repeat(f[..., None], 3, axis=2)
        r, g, b = f[..., 0] / 255, f[..., 1] / 255, f[..., 2] / 255
        L = 0.7 * r + 0.3 * g                              # cone responses from RGB
        M = 0.2 * r + 0.7 * g + 0.1 * b
        S = 0.02 * r + 0.1 * g + 0.9 * b
        rg_px, by_px = L - M, S - 0.5 * (L + M)
        intensity = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        lum_px = self.k * np.log1p(intensity / self.i0)
        dog = np.abs(_box(lum_px, self.centre) - _box(lum_px, self.surround))
        motion_px = np.zeros_like(lum_px) if self._prev_px is None or self._prev_px.shape != lum_px.shape \
            else np.abs(lum_px - self._prev_px)
        self._prev_px = lum_px
        gh, gw = self.gh, self.gw
        lum = _pool(lum_px, gh, gw)
        transient = np.zeros_like(lum) if self._prev_lum is None else np.abs(lum - self._prev_lum)
        self._prev_lum = lum
        rg, by = _pool(rg_px, gh, gw), _pool(by_px, gh, gw)
        sat = _pool(np.sqrt(rg_px ** 2 + by_px ** 2), gh, gw)
        return {"lum": lum, "transient": transient, "rg": rg, "by": by, "sat": sat,
                "contrast": _pool(dog, gh, gw), "motion": _pool(motion_px, gh, gw)}
