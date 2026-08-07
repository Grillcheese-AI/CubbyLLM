"""Chain-of-thought reasoning pipeline (spec 2026-08-07-cot-pipeline).

Planner walks a question's relation chain; retrieval (injected) finds the
facts; a CubeLang program holds the chain in a VSA frame and reads the
answer out with per-hop cosine confidence. Symbols only ever cross the VM
boundary.
"""
from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED
