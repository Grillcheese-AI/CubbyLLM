"""Chain-of-thought reasoning pipeline (spec 2026-08-07-cot-pipeline).

Planner walks a question's relation chain; retrieval (injected) finds the
facts; a CubeLang program holds the chain in a VSA frame and reads the
answer out with per-hop cosine confidence. Symbols only ever cross the VM
boundary.
"""
from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

from .index import TripleIndex                     # noqa: E402
from .pipeline import CoTResult, HopTrace, answer  # noqa: E402
from .planner import accepts, parse_fact, parse_question   # noqa: E402
from .programs import build_chain_program          # noqa: E402

__all__ = ["CoTResult", "HopTrace", "TripleIndex", "accepts", "answer", "parse_fact",
           "parse_question", "build_chain_program"]
