"""Spatial understanding at the input: places as unit vectors (``ssp``), a line through a box
of cells that keeps neighbours near (``hilbert``), a scene read exactly for the walk and as one
vector for the trunk (``scene``), and the host's route to it (``route``). The trunk's own route
is ``cubbyllm.model.spatial``; the proofs are ``tests/spatial`` and ``tests/model/test_spatial.py``."""
from .hilbert import curve, order
from .route import route
from .scene import DIRECTIONS, Scene
from .ssp import SSP

__all__ = ["SSP", "Scene", "DIRECTIONS", "curve", "order", "route"]
