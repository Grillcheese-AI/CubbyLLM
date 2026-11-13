"""Order 0067, the second reader's step 3: the join. A long instruction's parts (atoms of `view_programs`, or their
negation) become one CubeLang program: a function per part, then a ``solve`` that calls the parts in the stated order
and joins them by the instruction's logic.

A part's function takes the view columns it reads as parameters (the VM build does not let a called function read the
caller's ``let`` values: step 3's probe; a list passed as an argument works). The host puts the reply's views at the
top of ``solve`` (`reply_views.literals`), so the program the reader writes is everything else (`join`).

**The logic** is a list of groups, all of which must hold: ``("one", i)``; ``("unless", i, j)`` part i unless part j
(i or j); ``("if", j, i)`` if part j then part i; ``("either", i, j)`` part i or part j. `spec(parts, groups)` is the
same logic as a `view_programs` spec, for the evaluator.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubbyllm.speech import reply_views as RV  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_compile as VC  # noqa: E402
import view_programs as VP  # noqa: E402

__wiring__ = "STANDALONE"

HEAD = "public function solve(mention: str): number {"


def spec(parts: list[dict], groups: list[tuple]) -> dict:
    def g(t):
        if t[0] == "one":
            return parts[t[1]]
        if t[0] == "if":
            return {"if": parts[t[1]], "then": parts[t[2]]}
        return {"unless" if t[0] == "unless" else "or": [parts[t[1]], parts[t[2]]]}
    gs = [g(t) for t in groups]
    return gs[0] if len(gs) == 1 else {"and": gs}


def _columns(body: str) -> list[str]:
    names = set(re.findall(r"\b(?:n_[a-z]+|(?:" + "|".join(RV.COLUMNS) + r")_[a-z_]+)\b", body))
    return sorted(names)


def join(parts: list[dict], groups: list[tuple]) -> str:
    """The program the reader writes: the parts' functions and ``solve`` without the views."""
    funcs, calls = [], []
    for i, p in enumerate(parts, start=1):
        body = VC.body(p)
        cols = _columns(body)
        params = ", ".join(f"{c}: number" for c in cols)
        inner = "\n        ".join(body.splitlines())
        funcs.append(f"    function part{i}({params}): number {{\n        {inner}\n    }}")
        calls.append(f"let p{i} = part{i}({', '.join(cols)});")
    w, out = 0, []

    def flag():
        nonlocal w
        w += 1
        out.append(f"create g{w}: number; assign g{w} = 0;")
        return f"g{w}"
    tops = []
    for t in groups:
        if t[0] == "one":
            tops.append(f"p{t[1] + 1}")
            continue
        a, b, c = f"p{t[1] + 1}", f"p{t[2] + 1}", flag()
        if t[0] == "if":                                 # if a then b: not a, or b
            out.append(f"if ({a} == 0) {{ assign {c} = 1; }} if ({b} == 1) {{ assign {c} = 1; }}")
        else:                                            # unless / either: a or b
            out.append(f"if ({a} == 1) {{ assign {c} = 1; }} if ({b} == 1) {{ assign {c} = 1; }}")
        tops.append(c)
    top = flag()
    out.append("".join(f"if ({x} == 1) {{ " for x in tops) + f"assign {top} = 1;" + " }" * len(tops))
    main = "\n        ".join(calls + out + [f"return {top};"])
    return "program Instruction implements ISolve {\n" + "\n".join(funcs) + f"\n    {HEAD}\n        {main}\n    }}\n}}\n"


def program(join_src: str, parts: list[dict], reply: str) -> str:
    """The program the host runs: `join_src` with the views its parts read bound at the top of ``solve``."""
    views = sorted({v for p in parts for v in VC.views_of(p)})
    lets = "\n        ".join(RV.literals(reply, views))
    return join_src.replace(HEAD, f"{HEAD}\n        {lets}", 1)


__all__ = ["HEAD", "join", "program", "spec"]
