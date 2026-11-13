"""Order 0067, the second reader's step 2: a spec (`view_programs.py`) compiled to CubeLang over the reply's views.

Only the forms step 0 found running on the VM build (`validation/logs/vm_ops_probe_2026-11-13.json`): a `while` over an
index bounded by the view's length ``n_<view>``, values read with ``let x = <view>_<col>[i];``, nested ``if``s and 0/1
flags (no ``&&``, ``||``, ``!``, ``else if``, no expression with a call or an index inside a condition), and no index
past a view's end (``at`` checks the index first): a missing value is never compared (cubelang's P6).

`body(spec)` -> the check's lines (what the second reader writes); `program(spec, reply)` -> the whole program the host
runs: the views the spec reads bound from the reply (`reply_views.literals`), then the body.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubbyllm.speech import reply_views as RV  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_programs as VP  # noqa: E402

__wiring__ = "STANDALONE"


class _W:
    def __init__(self):
        self.lines, self.n = [], 0

    def name(self, stem: str) -> str:
        self.n += 1
        return f"{stem}{self.n}"

    def flag(self) -> str:
        c = self.name("c")
        self.lines.append(f"create {c}: number; assign {c} = 0;")
        return c

    def atom(self, a: dict) -> str:
        v = a["view"]
        if a["q"] == "n":
            c = self.flag()
            self.lines.append(f"if (n_{v} {a['op']} {int(a['v'])}) {{ assign {c} = 1; }}")
            return c
        if a["q"] == "at":
            c, p, x = self.flag(), self.name("p"), self.name("x")
            if a["pos"] > 0:
                self.lines.append(f"create {p}: number; assign {p} = {a['pos'] - 1};")
            else:
                self.lines.append(f"create {p}: number; assign {p} = n_{v}; sub {p}, 1;")
            self.lines.append(f"if ({p} >= 0) {{ if ({p} < n_{v}) {{ let {x} = {v}_{a['col']}[{p}]; "
                              f"if ({x} {a['op']} {RV._lit(a['val'])}) {{ assign {c} = 1; }} }} }}")
            return c
        k, i = self.name("k"), self.name("i")
        xs = [self.name("x") for _ in a["where"]]
        lets = " ".join(f"let {x} = {v}_{cd['col']}[{i}];" for x, cd in zip(xs, a["where"]))
        test = "".join(f"if ({x} {cd['op']} {RV._lit(cd['v'])}) {{ " for x, cd in zip(xs, a["where"]))
        self.lines += [f"create {k}: number; assign {k} = 0;", f"create {i}: number; assign {i} = 0;",
                       f"while ({i} < n_{v}) {{ {lets} {test}add {k}, 1;{' }' * len(xs)} add {i}, 1; }}"]
        c = self.flag()
        cond = {"count": f"{k} {a.get('op')} {int(a.get('v', 0))}", "all": f"{k} == n_{v}", "any": f"{k} > 0",
                "none": f"{k} == 0"}[a["q"]]
        self.lines.append(f"if ({cond}) {{ assign {c} = 1; }}")
        return c

    def node(self, s: dict) -> str:
        if "q" in s:
            return self.atom(s)
        if "not" in s:
            a = self.node(s["not"])
            c = self.flag()
            self.lines.append(f"if ({a} == 0) {{ assign {c} = 1; }}")
            return c
        if "and" in s:
            a, b = (self.node(x) for x in s["and"])
            c = self.flag()
            self.lines.append(f"if ({a} == 1) {{ if ({b} == 1) {{ assign {c} = 1; }} }}")
            return c
        if "if" in s:
            a, b = self.node(s["if"]), self.node(s["then"])
            c = self.flag()
            self.lines.append(f"if ({a} == 0) {{ assign {c} = 1; }} if ({b} == 1) {{ assign {c} = 1; }}")
            return c
        a, b = (self.node(x) for x in (s.get("or") or s["unless"]))
        c = self.flag()
        self.lines.append(f"if ({a} == 1) {{ assign {c} = 1; }} if ({b} == 1) {{ assign {c} = 1; }}")
        return c


def body(spec: dict) -> str:
    w = _W()
    top = w.node(spec)
    return "\n".join(w.lines + [f"return {top};"])


def views_of(spec: dict) -> list[str]:
    return sorted({a["view"] for a in VP.atoms(spec)})


def program(spec: dict, reply: str, check: str | None = None) -> str:
    """The whole program for `reply`: the views `spec` reads, bound, then `check` (default: `body(spec)`)."""
    lets = "\n        ".join(RV.literals(reply, views_of(spec)))
    inner = "\n        ".join((check or body(spec)).splitlines())
    return (f"program ViewCheck implements ISolve {{\n    public function solve(mention: str): number {{\n        {lets}\n"
            f"        {inner}\n    }}\n}}\n")


__all__ = ["body", "program", "views_of"]
