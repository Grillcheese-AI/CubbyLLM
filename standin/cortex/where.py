"""Where — depth and the body's own movement rebuilt into a map (design §4.3).

Three parts, none of them knowing what world the frames come from:

    Camera          a pinhole camera: one unit ray per pixel for a pose (yaw, pitch in radians,
                    z up, yaw 0 along +x, pitch up positive) and a horizontal field of view.
    PathIntegrator  where the eyes are, from the body's own velocity alone (dead reckoning), or
                    from a given position (the sighted tiers); the error between the two is
                    measured, not hidden.
    OccupancyMap    log-odds per cell: a ray that ends on a surface says the cell just behind
                    the surface is solid; every cell it passed through on the way is free.
                    What it holds is the cortex's own map — the allocentric map a planner can use.

Wired: STANDALONE (part of the cortex plugin; the blind agent reads it in a later stage).
"""
from __future__ import annotations

import math

import numpy as np

__wiring__ = "STANDALONE"

HIT, MISS, CLAMP = 0.9, -0.35, 4.0          # log-odds per observation, and its bound


class Camera:
    def __init__(self, w: int, h: int, fov: float) -> None:
        self.w, self.h, self.fov = w, h, fov
        tx = math.tan(math.radians(fov) / 2)
        ty = tx * h / w
        u = (np.arange(w) + 0.5) / w * 2 - 1
        v = 1 - (np.arange(h) + 0.5) / h * 2
        self._u = (u * tx).astype(np.float32)
        self._v = (v * ty).astype(np.float32)

    def rays(self, yaw: float, pitch: float) -> np.ndarray:
        fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        up = np.cross(right, fwd)
        d = fwd[None, None, :] + self._u[None, :, None] * right[None, None, :] + self._v[:, None, None] * up[None, None, :]
        return (d / np.linalg.norm(d, axis=2, keepdims=True)).astype(np.float32)


class PathIntegrator:
    """The eyes' position from velocity (blocks per step), corrected only when told the truth."""

    def __init__(self, start=(0.0, 0.0, 0.0)) -> None:
        self.pos = np.asarray(start, np.float64)

    def step(self, velocity) -> np.ndarray:
        self.pos = self.pos + np.asarray(velocity, np.float64)
        return self.pos

    def reset(self, pos) -> None:
        self.pos = np.asarray(pos, np.float64)


class OccupancyMap:
    def __init__(self, free_step: float = 0.5) -> None:
        self.odds: dict[tuple, float] = {}
        self.free_step = free_step

    def integrate(self, eye, rays: np.ndarray, depth: np.ndarray, sight: float) -> tuple[set, set]:
        """One frame. -> (cells seen solid, cells seen free) by this frame."""
        eye = np.asarray(eye, np.float64)
        d = rays.reshape(-1, 3).astype(np.float64)
        z = depth.reshape(-1).astype(np.float64)
        hit = z < sight - 1e-3
        pts = eye[None, :] + d[hit] * (z[hit][:, None] + 0.02)          # just behind the surface
        solid = {tuple(c) for c in np.floor(pts).astype(np.int64).tolist()}
        free: set = set()
        reach = np.where(hit, z - 0.05, sight)
        n = int(math.ceil(sight / self.free_step))
        for k in range(1, n + 1):
            t = k * self.free_step
            ok = reach > t
            if not ok.any():
                break
            cells = np.floor(eye[None, :] + d[ok] * t).astype(np.int64)
            free.update(map(tuple, np.unique(cells, axis=0).tolist()))
        free -= solid
        for c in solid:
            self.odds[c] = min(CLAMP, self.odds.get(c, 0.0) + HIT)
        for c in free:
            self.odds[c] = max(-CLAMP, self.odds.get(c, 0.0) + MISS)
        return solid, free

    def align(self, eye, rays: np.ndarray, depth: np.ndarray, sight: float, reach: float = 0.3,
              step: float = 0.1) -> np.ndarray:
        """Where the eyes must really be for this frame to agree best with the map so far: the
        small shift (within `reach`) under which the surface points land on cells the map holds
        solid AND the stretch of each ray just before its surface lands on cells it holds free
        (a shift that buries the view in a wall scores badly), with a small pull toward no shift
        (the odometer is not trusted less than it should be). How a place recognised corrects a
        drifting sense of position (the place cells resetting the path integrator)."""
        eye = np.asarray(eye, np.float64)
        d = rays.reshape(-1, 3).astype(np.float64)
        z = depth.reshape(-1).astype(np.float64)
        hit = z < sight - 1e-3
        if hit.sum() < 20 or len(self.odds) < 50:
            return np.zeros(3)
        dh, zh = d[hit], z[hit][:, None]
        surf = dh * (zh + 0.02)                                             # relative to the eye
        before = np.concatenate([dh * np.maximum(zh - 0.25, 0.05), dh * np.maximum(zh - 0.6, 0.05)])
        R = int(math.ceil(sight + reach)) + 1
        base = np.floor(eye).astype(np.int64) - R
        size = 2 * R + 1
        local = np.zeros((size, size, size), np.float32)
        for c, v in self.odds.items():
            i, j, k = c[0] - base[0], c[1] - base[1], c[2] - base[2]
            if 0 <= i < size and 0 <= j < size and 0 <= k < size:
                local[i, j, k] = v

        def look(points, off):
            cells = np.floor(eye + off + points).astype(np.int64) - base
            ok = np.all((cells >= 0) & (cells < size), axis=1)
            return local[cells[ok, 0], cells[ok, 1], cells[ok, 2]]

        steps = np.arange(-reach, reach + 1e-9, step)
        pull = 0.02 * len(surf)
        scores = {}
        for dx in steps:
            for dy in steps:
                for dz in steps:
                    off = np.array([dx, dy, dz])
                    sc = float(look(surf, off).sum() - np.maximum(look(before, off), 0).sum() - pull * np.abs(off).sum())
                    scores[(round(dx, 3), round(dy, 3), round(dz, 3))] = (sc, off)
        best_sc, best = max(scores.values(), key=lambda v: (v[0], -np.abs(v[1]).sum()))
        zero = scores[(0.0, 0.0, 0.0)][0] if (0.0, 0.0, 0.0) in scores else -np.inf
        return best if best_sc > zero + 0.05 * len(surf) else np.zeros(3)

    def occupied(self, p: float = 0.7) -> set:
        lo = math.log(p / (1 - p))
        return {c for c, v in self.odds.items() if v > lo}

    def known_free(self, p: float = 0.3) -> set:
        lo = math.log(p / (1 - p))
        return {c for c, v in self.odds.items() if v < lo}


class SurfaceMap:
    """Where surfaces are, finer than cells: a point and its normal per 0.25-block voxel, averaged
    over the frames that saw it — what point-to-plane matching (ICP, translation only: the body
    knows which way it faces) aligns a new frame against, to correct a drifting position."""

    def __init__(self, voxel: float = 0.25) -> None:
        self.voxel = voxel
        self.pts: dict[tuple, list] = {}            # voxel -> [sum of points, sum of normals, count]

    @staticmethod
    def surface(eye, rays: np.ndarray, depth: np.ndarray, sight: float, every: int = 2):
        """Surface points (relative to the eye) and their normals, from a depth image: the normal
        is the cross product of the neighbouring points, kept only where the surface is smooth."""
        h, w = depth.shape
        P = rays * depth[..., None]
        ok = depth < sight - 1e-3
        dx = P[1:-1, 2:] - P[1:-1, :-2]
        dy = P[2:, 1:-1] - P[:-2, 1:-1]
        n = np.cross(dx, dy)
        norm = np.linalg.norm(n, axis=2, keepdims=True)
        n = n / np.maximum(norm, 1e-9)
        inner = ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
        dd = depth[1:-1, 1:-1]
        smooth = (np.abs(depth[1:-1, 2:] - dd) < 0.3) & (np.abs(depth[1:-1, :-2] - dd) < 0.3) \
            & (np.abs(depth[2:, 1:-1] - dd) < 0.3) & (np.abs(depth[:-2, 1:-1] - dd) < 0.3)
        good = inner & smooth & (norm[..., 0] > 1e-6)
        sel = np.zeros_like(good)
        sel[::every, ::every] = True
        good &= sel
        pts = P[1:-1, 1:-1][good]
        nrm = n[good]
        r = rays[1:-1, 1:-1][good]
        flip = np.sum(nrm * r, axis=1) > 0                       # normals face the eye
        nrm[flip] *= -1
        return pts.astype(np.float64), nrm.astype(np.float64)

    def add(self, world_pts: np.ndarray, normals: np.ndarray) -> None:
        keys = np.floor(world_pts / self.voxel).astype(np.int64)
        for k, p, n in zip(map(tuple, keys.tolist()), world_pts, normals):
            e = self.pts.get(k)
            if e is None:
                self.pts[k] = [p.copy(), n.copy(), 1]
            else:
                e[0] += p
                e[1] += n
                e[2] += 1

    def match(self, world_pts: np.ndarray, normals: np.ndarray, iters: int = 4, prior: float = 0.2,
              reach: float = 0.35, constrain: float = 0.15, cap: float = 0.15) -> np.ndarray:
        """The translation that best lays these points on the surfaces already seen (point to
        plane, robust): only along directions the surfaces constrain, pulled toward no shift, and
        at most `cap` per frame (a world of cubes repeats every block: a large jump is more likely
        a wrong match than a true correction)."""
        if len(self.pts) < 100 or len(world_pts) < 30:
            return np.zeros(3)
        shift = np.zeros(3)
        v = self.voxel
        offsets = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for _ in range(iters):
            q = world_pts + shift
            keys = np.floor(q / v).astype(np.int64)
            A = np.zeros((3, 3))
            b = np.zeros(3)
            used = 0
            for k, p, n in zip(map(tuple, keys.tolist()), q, normals):
                best = None
                for o in offsets:
                    e = self.pts.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]))
                    if e is None:
                        continue
                    mp, mn = e[0] / e[2], e[1] / max(1e-9, np.linalg.norm(e[1]))
                    if float(mn @ n) < 0.8:
                        continue
                    dist = float(np.linalg.norm(mp - p))
                    if dist < reach and (best is None or dist < best[0]):
                        best = (dist, mp, mn)
                if best is None:
                    continue
                _, mp, mn = best
                r = float(mn @ (mp - p))
                if abs(r) > 0.25:
                    continue
                A += np.outer(mn, mn)
                b += mn * r
                used += 1
            if used < 30:
                break
            # only the directions the surfaces pin down: a floor fixes height, not where along it
            w, V = np.linalg.eigh(A)
            keep = w > constrain * used
            if not keep.any():
                break
            Vk = V[:, keep]
            step = Vk @ ((Vk.T @ b) / (w[keep] + prior * used))
            shift = shift + step
            if np.linalg.norm(step) < 1e-3:
                break
        n = np.linalg.norm(shift)
        return shift if n <= cap else shift * (cap / n)
