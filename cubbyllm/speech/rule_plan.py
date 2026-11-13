"""cubbyllm/speech/rule_plan.py -- a rule as a plan over the reply world, compiled to a CubeLang program and run by
the VM (order 0067, step 3: the plan family; its runner).

A plan (the shape the program seat writes, the way it writes a plan over the store's relations for a question):
- a leaf ``{"m": <a reply_world relation>, "op": <op>, "v": <value>}``; ops: ``<=``, ``>=``, ``<``, ``>``, ``==``
  and ``!=`` (a count is compared as a number, any other value as a string), ``has`` / ``lacks`` (a relation with
  several values: the reply's words, its JSON keys, its openings); a relation with an argument
  (`reply_world.PARAMETRIC`: "count of the word", "starts with", ...) names it in ``"arg"``, a non-empty string;
- ``{"and": [...]}``, ``{"or": [...]}``, ``{"not": plan}``;
- optional ``"scope"`` (``session``, ``turn``, or a topic) on the outermost node: where the rule applies (the turn,
  step 4, reads it; the VM never does).

`program(plan, reply)` measures the reply (`reply_world`), binds each relation the plan reads as a value (a number,
a string, or a list), and writes the plan as nested comparisons that leave 1 (held) or 0 in ``held``. The VM runs it;
the host never decides whether a rule held. A plan that names a relation the reply world has not, or an op it does
not know, is refused before any program is written (`check`): the rule is then "asked", never "checked".

No kind of instruction has code here: "two sentences", "no word X" and "start with yes" are all plans.

Wired: STANDALONE (the turn, step 4, calls `run`; until then its tests).
"""
from __future__ import annotations

import re

from . import reply_world as rw

__wiring__ = "STANDALONE"

NUM_OPS = ("<=", ">=", "<", ">", "==", "!=")
SET_OPS = ("has", "lacks")
MULTI = ("word", "json key", "opening")                  # relations with several values (a list in the program)
FOLDED = ("word", "opening", "last word")                # the reply world gives these lowercased: a value is folded
#   to match; a JSON key is compared as written (keys are case-sensitive: "U1" is not "u1")


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _var(rel: str) -> str:
    return "m_" + re.sub(r"[^a-z0-9]+", "_", rel.lower()).strip("_")


def leaves(plan: dict) -> list[dict]:
    if "m" in plan:
        return [plan]
    if "not" in plan:
        return leaves(plan["not"])
    return [x for p in plan.get("and", plan.get("or", [])) for x in leaves(p)]


def check(plan: dict) -> str | None:
    """Why the plan cannot be run, or None."""
    if not isinstance(plan, dict) or not (set(plan) & {"m", "and", "or", "not"}):
        return "not a plan"
    for lf in leaves(plan):
        if lf.get("m") not in rw.RELATIONS and lf.get("m") not in rw.PARAMETRIC:
            return f"no measurement {lf.get('m')!r}"
        if (lf["m"] in rw.PARAMETRIC) != bool(isinstance(lf.get("arg"), str) and lf["arg"] != ""):
            return f"{lf['m']!r} takes an argument" if lf["m"] in rw.PARAMETRIC else f"{lf['m']!r} takes no argument"
        if lf.get("op") not in NUM_OPS + SET_OPS:
            return f"no op {lf.get('op')!r}"
        if (lf["op"] in SET_OPS) != (lf["m"] in MULTI):
            return f"{lf['op']} does not apply to {lf['m']!r}"
        if "v" not in lf:
            return "a leaf with no value"
    for key in ("and", "or"):
        if key in plan and not plan[key]:
            return f"an empty {key}"
    return None


def _is_num(rel: str) -> bool:
    return rel.endswith("count") or rel.startswith("count of")


def _key(lf: dict) -> tuple:
    return lf["m"], lf.get("arg")


def _lit(rel: str, v) -> str:
    return str(int(v)) if _is_num(rel) else f'"{_esc(str(v).lower() if rel in FOLDED else v)}"'


class _Writer:
    def __init__(self, lists: dict, names: dict) -> None:
        self.lines, self.n, self.lists = [], 0, lists       # a multi-valued relation: its list literal
        self.names = names                                  # (relation, argument) -> its variable

    def flag(self) -> str:
        self.n += 1
        c = f"c{self.n}"
        self.lines += [f"        create {c}: number;", f"        assign {c} = 0;"]
        return c

    def node(self, p: dict) -> str:
        c = None
        if "m" in p:
            c, var, op = self.flag(), self.names[_key(p)], p["op"]   # (`var` unused for a list)
            if op in SET_OPS:
                hit = self.flag()
                if self.lists.get(p["m"]):                   # (an empty list: nothing to find)
                    self.lines.append(f'        for (let w of [{self.lists[p["m"]]}]) {{ if (w == {_lit(p["m"], p["v"])}) '
                                      f'{{ assign {hit} = 1; }} }}')
                self.lines.append(f"        if ({hit} == {1 if op == 'has' else 0}) {{ assign {c} = 1; }}")
            elif op == "!=":
                self.lines.append(f"        if ({var} == {_lit(p['m'], p['v'])}) {{ }} else {{ assign {c} = 1; }}")
            else:
                self.lines.append(f"        if ({var} {op} {_lit(p['m'], p['v'])}) {{ assign {c} = 1; }}")
            return c
        if "not" in p:
            inner = self.node(p["not"])
            c = self.flag()
            self.lines.append(f"        if ({inner} == 0) {{ assign {c} = 1; }}")
            return c
        subs = [self.node(q) for q in p["and" if "and" in p else "or"]]
        c = self.flag()
        if "and" in p:
            self.lines.append("        " + "".join(f"if ({s} == 1) {{ " for s in subs) + f"assign {c} = 1;" + " }" * len(subs))
        else:
            self.lines += [f"        if ({s} == 1) {{ assign {c} = 1; }}" for s in subs]
        return c


def program(plan: dict, reply: str) -> tuple[str, list[str]]:
    """(the CubeLang source, the reply's statements it binds). Call `check` first."""
    keys = list(dict.fromkeys(_key(lf) for lf in leaves(plan)))
    plain = [r for r, a in keys if a is None]
    measured: dict = {}                                  # (relation, argument) -> its values
    for _s, r, v in rw.measure(reply, plain):
        measured.setdefault((r, None), []).append(v)
    for r, a in keys:
        if a is not None:
            measured[(r, a)] = [v for _s, _r, v in rw.measure_arg(reply, r, a)]
    names = {k: _var(k[0]) if k[1] is None else f"{_var(k[0])}_{i}" for i, k in enumerate(keys)}
    w = _Writer({r: ", ".join(_lit(r, v) for v in measured.get((r, None), [])) for r, a in keys if r in MULTI}, names)
    for k in keys:
        vals, r = measured.get(k, []), k[0]
        if r not in MULTI:
            w.lines.append(f"        create {names[k]}: {'number' if _is_num(r) else 'str'};")
            w.lines.append(f"        assign {names[k]} = {_lit(r, vals[0]) if vals else '0'};")
    top = w.node(plan)
    body = "\n".join(w.lines + [f"        return {top};"])
    src = f"program ReplyRule implements ISolve {{\n    public function solve(mention: str): number {{\n{body}\n    }}\n}}\n"
    said = lambda k: k[0] if k[1] is None else f'{k[0]} "{k[1]}"'          # noqa: E731
    return src, [f"{v} is the {said(k)} of {rw.SUBJ}" for k in keys for v in measured.get(k, [])]


def run(plan: dict, reply: str, run_fn) -> dict:
    """The rule's outcome on `reply`, by the VM (`run_fn(source, fn) -> {"result": ...}`): {"held": bool | None,
    "how": "checked" | "asked", "why", "statements", "program"}. A plan that cannot be run is "asked" (held None)."""
    why = check(plan)
    if why:
        return {"held": None, "how": "asked", "why": why, "statements": [], "program": None}
    src, sts = program(plan, reply)
    out = run_fn(src, "solve")
    res = out.get("result") if isinstance(out, dict) else None
    held = None if res is None else str(res).strip() in ("1", "1.0")
    return {"held": held, "how": "checked" if held is not None else "asked",
            "why": None if held is not None else "the VM returned nothing", "statements": sts, "program": src}


__all__ = ["MULTI", "NUM_OPS", "SET_OPS", "check", "leaves", "program", "run"]
