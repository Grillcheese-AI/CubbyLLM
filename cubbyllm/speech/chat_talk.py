"""cubbyllm/speech/chat_talk.py -- the talk program the chat turn hands the VM: the resting state, the reply's
think block stripped, a string escaped for CubeLang, the self-grounding check. Split out of chat.py (order 0031),
verbatim; chat.py re-exports every name.

Wired: WIRED (the chat turn).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


RESTING = {"dopamine": 0.25, "serotonin": 0.20, "cortisol": 0.15, "oxytocin": 0.08, "noradrenaline": 0.07}


_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)


def _escape(s: str) -> str:
    """A CubeLang string literal: no raw newlines, escaped backslashes/quotes."""
    return " ".join(s.split()).replace("\\", "\\\\").replace('"', '\\"')


# What the host can back about Cubby without asking anyone: the identity facts
# themselves. Saying his own name is not a claim reaching past its grounding,
# and the first version of the filter treated "I'm Cubby" as one specific of
# unverified claim surface — which made an alarmed body refuse to introduce
# itself. The two "never say this" lists are excluded: those words are already
# a hard rejection upstream, and grounding them would be nonsense.
_NOT_GROUND = ("refuses_to_claim", "never_says_it_is", "forbidden_words", "note")


def _self_grounding(facts: dict) -> set[str]:
    """Every word the identity facts themselves assert, lowercased."""
    out: set[str] = set()

    def walk(v):
        if isinstance(v, str):
            out.update(w.lower() for w in re.findall(r"[\w']+", v))
        elif isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                walk(vv)

    for k, v in (facts or {}).items():
        if k not in _NOT_GROUND:
            walk(v)
    return out


def render_talk_program(candidates: list[str], question: str = "which reply") -> str:
    """`CubbyTalk implements IAgent` with the candidates embedded as literals.
    think() offers them via ASK; act() remembers the chosen reply; observe()
    counts feedback. Compiles under --strict; every op executes (validate)."""
    if not candidates:
        raise ValueError("at least one candidate (the don't-know line) is required")
    regs = [f"c{i}" for i in range(len(candidates))]
    lits = "\n".join(f'        create {r} : str;\n        assign {r} = "{_escape(c)}";' for r, c in zip(regs, candidates))
    return f'''# CubbyTalk -- one chat turn, mediated by the VM (stand-in, 2026-08-30)
program CubbyTalk implements IAgent {{
    storage {{
        turns: mutable u64 = 0;
        replies: mutable u64 = 0;
        good: mutable u64 = 0;
        bad: mutable u64 = 0;
    }}

    @system @once
    public function constructor() {{
        assign turns = 0;
        assign replies = 0;
        assign good = 0;
        assign bad = 0;
    }}

    @external
    public function think(input: str, ctx: str): str {{
        add turns, 1;
{lits}
        ask "{_escape(question)}", {", ".join(regs)};
        create chosen : str;
        pop chosen;
        return chosen;
    }}

    @external
    public function act(decision: str, ctx: str): str {{
        add replies, 1;
        create said : str;
        assign said = decision;
        remember said;
        return said;
    }}

    @external
    public function observe(result: str, ctx: str): str {{
        create seen : str;
        assign seen = result;
        remember seen;
        return seen;
    }}
}}
'''
