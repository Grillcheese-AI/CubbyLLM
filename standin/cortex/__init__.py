"""The sensory cortex (docs/superpowers/specs/2026-10-10-sensory-cortex-design.md): frames and
touch in, percepts out — salience, where to look next, and (later stages) occupancy, kinds and
surprise. Input-agnostic: it never knows where a frame came from (a game render, a screenshot,
a camera).

Wired: STANDALONE — a plugin, off unless mounted; nothing in the serve path imports it yet.
Built so far: the retina and salience/attention (build-order step 2), and "where": depth and
the body's movement into an occupancy map (step 3). Ported from cubemind's
perception (bio_vision's opponent colour, Weber–Fechner luminance and motion; the GIF neuron),
rewritten vectorised, never imported.
"""
from .contract import Percept, SensoryCortex
from .retina import Retina
from .salience import Attention, Salience
from .where import Camera, OccupancyMap, PathIntegrator

__wiring__ = "STANDALONE"
__all__ = ["SensoryCortex", "Percept", "Retina", "Salience", "Attention", "Camera", "OccupancyMap", "PathIntegrator"]
