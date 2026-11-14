"""cubbyllm/speech/view_spec.py -- the spec language of a check over the reply's views (`reply_views.py`) and its
evaluator (orders 0067 and 0073). One implementation for the host and for the row builders: the core never imports
``validation/``; `validation/builders/view_programs.py` draws specs and re-exports these names.

**A spec.** An atom asks one thing of one view, the rows filtered by up to two conditions on their columns
(``{"col", "op", "v"}``, and-ed):
- ``count``: how many rows match, compared (``op``, ``v``); ``n``: how many rows the view has, compared;
- ``all`` / ``any`` / ``none``: every row / some row / no row matches (``all`` of no rows holds);
- ``at``: the row at ``pos`` (1, 2, 3, or -1 the last) has ``col`` ``op`` ``val`` (no such row: it fails).
A spec is an atom or ``{"and": [a, b]}``, ``{"or": [a, b]}``, ``{"not": a}``, ``{"if": a, "then": b}`` (b whenever a),
``{"unless": [a, b]}`` (a unless b: a or b).

`evaluate(spec, views)` -> bool, the truth the VM's run must agree with. `atoms(spec)` -> its atoms, in order.

Wired: STANDALONE (the row builders, the second reader's reads; the host compiles from it in order 0073's step 4).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

FILTERS = {                                               # view -> {column: "str" | "num"}
    "char": {"kind": "str", "case": "str", "char": "str"},
    "word": {"lower": "str", "first_lower": "str", "last_lower": "str", "case": "str", "len": "num"},
    "sentence": {"n_word": "num", "end_mark": "str", "first_word": "str", "last_word": "str"},
    "line": {"n_word": "num", "first_word": "str", "item": "str"},
    "paragraph": {"n_word": "num", "n_sentence": "num", "first_word": "str"},
    "item": {"marker": "str", "n_word": "num"},
    "json": {"key": "str", "type": "str", "value": "str"},
}
NUM_OPS = ("<", "<=", ">", ">=", "==", "!=")
STR_OPS = ("==", "!=")


def cmp(a, op, b) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]


def match(row: dict, conds: list[dict]) -> bool:
    return all(cmp(row[c["col"]], c["op"], c["v"]) for c in conds)


def evaluate(spec: dict, views: dict) -> bool:
    if "and" in spec:
        return all(evaluate(s, views) for s in spec["and"])
    if "or" in spec:
        return any(evaluate(s, views) for s in spec["or"])
    if "not" in spec:
        return not evaluate(spec["not"], views)
    if "if" in spec:
        return (not evaluate(spec["if"], views)) or evaluate(spec["then"], views)
    if "unless" in spec:
        return evaluate(spec["unless"][0], views) or evaluate(spec["unless"][1], views)
    rows = views[spec["view"]]
    q = spec["q"]
    if q == "n":
        return cmp(len(rows), spec["op"], spec["v"])
    if q == "at":
        idx = spec["pos"] - 1 if spec["pos"] > 0 else len(rows) - 1
        return 0 <= idx < len(rows) and cmp(rows[idx][spec["col"]], spec["op"], spec["val"])
    k = sum(1 for r in rows if match(r, spec["where"]))
    return {"count": lambda: cmp(k, spec["op"], spec["v"]), "all": lambda: k == len(rows),
            "any": lambda: k > 0, "none": lambda: k == 0}[q]()


def atoms(spec: dict) -> list[dict]:
    if "q" in spec:
        return [spec]
    if "not" in spec:
        return atoms(spec["not"])
    if "if" in spec:
        return atoms(spec["if"]) + atoms(spec["then"])
    return [a for s in spec.get("and") or spec.get("or") or spec.get("unless") for a in atoms(s)]


__all__ = ["FILTERS", "NUM_OPS", "STR_OPS", "atoms", "cmp", "evaluate", "match"]
