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


def build_chain_program(triples: list[Triple],
                        relations: list[str]) -> tuple[str, list[str]]:
    roles = [sanitize_role(i + 1, r) for i, r in enumerate(relations)]
    binds = "\n".join(
        f'        bind frame, {role}, "{_escape(t.obj)}";'
        for role, t in zip(roles, triples))

    def fn(name: str, sig: str, role: str) -> str:
        return (f"    public function {name}({sig}): str {{\n"
                f"        create frame: number;\n{binds}\n"
                f"        return recover(frame, {role});\n    }}")

    parts = [fn("solve", "mention: str", roles[0])]
    fns = ["solve"]
    for i in range(1, len(roles)):
        parts.append(fn(f"hop_{i + 1}", "", roles[i]))
        fns.append(f"hop_{i + 1}")
    parts.append(fn("control", "", CONTROL_ROLE))
    fns.append("control")
    body = "\n\n".join(parts)
    return (f"use vsa;\n\nprogram CotChain implements ISolve {{\n{body}\n}}\n",
            fns)
