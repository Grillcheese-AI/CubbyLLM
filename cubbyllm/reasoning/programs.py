"""Emit the per-question CubeLang chain program (spec section 4.3).

One frame holds every hop's (role -> object) binding in superposition; each
public function re-binds the frame (the VM is stateless per call) and
recovers ONE role, because `run-proto` returns one (symbol, similarity) per
call — the client batches functions. `solve(mention)` is hop 1 (ISolve's
required entry; the arg is ignored, matching the shipped
reasoning_bridge.cube), `hop_i` are the rest, `control` recovers the
never-bound ABSENT_CTRL role and must come back empty/low.
"""
from __future__ import annotations

import re

from ..core.protocols import Wiring
from .planner import Triple

__wiring__ = Wiring.WIRED

CONTROL_ROLE = "ABSENT_CTRL"


def sanitize_role(hop: int, rel: str) -> str:
    core = re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_").upper()
    return f"H{hop}_{core}"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_chain_program(triples: list[Triple], relations: list[str],
                        chunk: int = 0) -> tuple[str, list[str]]:
    """`chunk`: how many hops share a frame. 0 (the default, and what the serving
    path passes) means one frame holds the WHOLE chain, which is the shipped
    shape and produces byte-identical output to before this parameter existed.

    Why the parameter exists (`validation/exp_r28_depth_capacity.py`,
    2026-09-15): every hop in a frame is one more vector in the bundle, so each
    hop's recovered similarity halves as the chain grows -- 1.000, 0.501, 0.246,
    0.125, 0.064, 0.032 at depths 1..6. `ABSENT_CTRL` is the noise floor and does
    not fall. They meet at depth 6 and cross at depth 7, past which NO threshold
    separates a true binding from noise. With `chunk=2` the bundle never exceeds
    two, and the curve goes flat: 8-18x separation from depth 2 to depth 12,
    unchanged median, and the whole-chain clear rate falling only 93% -> 72%.

    The depth limit was never the VSA; it was this function.

    The control function binds the FIRST group, so it measures the noise floor
    for a bundle of the size the hops are actually recovering from. With equal
    groups (all but possibly the last) that is the right comparison; the
    `fns[-1] is the control` contract the pipeline relies on is unchanged.
    """
    roles = [sanitize_role(i + 1, r) for i, r in enumerate(relations)]
    size = chunk if chunk and chunk > 0 else len(triples)
    groups = [list(range(i, min(i + size, len(triples))))
              for i in range(0, len(triples), size)] or [[]]
    group_of = {i: gi for gi, g in enumerate(groups) for i in g}

    def binds_for(idx: list[int]) -> str:
        return "\n".join(f'        bind frame, {roles[i]}, "{_escape(triples[i].obj)}";'
                         for i in idx)

    def fn(name: str, sig: str, role: str, idx: list[int]) -> str:
        return (f"    public function {name}({sig}): str {{\n"
                f"        create frame: number;\n{binds_for(idx)}\n"
                f"        return recover(frame, {role});\n    }}")

    parts = [fn("solve", "mention: str", roles[0], groups[group_of[0]])]
    fns = ["solve"]
    for i in range(1, len(roles)):
        parts.append(fn(f"hop_{i + 1}", "", roles[i], groups[group_of[i]]))
        fns.append(f"hop_{i + 1}")
    parts.append(fn("control", "", CONTROL_ROLE, groups[0]))
    fns.append("control")
    body = "\n\n".join(parts)
    return (f"use vsa;\n\nprogram CotChain implements ISolve {{\n{body}\n}}\n",
            fns)
