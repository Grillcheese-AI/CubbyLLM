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

**Said forms (order 0073).** A part can be written as its words say it: a condition, or an atom, carries
``"not": True`` (the thing it sits on is negated: a condition's comparison; an atom's quantifier or comparison; for
``at``, the comparison only), and ``at`` carries ``"absent": "pass"`` where the words say "if there is one" (no such
row: it holds). `fold(part)` is the same part with no ``not`` anywhere, the one spelling the compiler takes; a part
and its folded form have the same truth on every reply.

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
INV = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
TWO_VALUED = {("line", "item"): ("yes", "no")}            # one folded spelling: ``==`` the value that holds


def cmp(a, op, b) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]


def match(row: dict, conds: list[dict]) -> bool:
    return all(cmp(row[c["col"]], c["op"], c["v"]) != bool(c.get("not")) for c in conds)


def evaluate(spec: dict, views: dict) -> bool:
    if "and" in spec:
        return all(evaluate(s, views) for s in spec["and"])
    if "or" in spec:
        return any(evaluate(s, views) for s in spec["or"])
    if "not" in spec and "q" not in spec:                 # around a spec; on an atom it is the atom's own (below)
        return not evaluate(spec["not"], views)
    if "if" in spec:
        return (not evaluate(spec["if"], views)) or evaluate(spec["then"], views)
    if "unless" in spec:
        return evaluate(spec["unless"][0], views) or evaluate(spec["unless"][1], views)
    rows = views[spec["view"]]
    q, neg = spec["q"], bool(spec.get("not"))
    if q == "n":
        return cmp(len(rows), spec["op"], spec["v"]) != neg
    if q == "at":
        idx = spec["pos"] - 1 if spec["pos"] > 0 else len(rows) - 1
        if not 0 <= idx < len(rows):
            return spec.get("absent") == "pass"           # "if there is one": no such row, it holds
        return cmp(rows[idx][spec["col"]], spec["op"], spec["val"]) != neg
    k = sum(1 for r in rows if match(r, spec["where"]))
    return {"count": lambda: cmp(k, spec["op"], spec["v"]), "all": lambda: k == len(rows),
            "any": lambda: k > 0, "none": lambda: k == 0}[q]() != neg


def atoms(spec: dict) -> list[dict]:
    if "q" in spec:
        return [spec]
    if "not" in spec:
        return atoms(spec["not"])
    if "if" in spec:
        return atoms(spec["if"]) + atoms(spec["then"])
    return [a for s in spec.get("and") or spec.get("or") or spec.get("unless") for a in atoms(s)]


def _plain(view: str, c: dict) -> dict:
    """A condition with no ``not`` and, on a two-valued column, ``==`` the value that holds."""
    op, v = (INV[c["op"]] if c.get("not") else c["op"]), c["v"]
    pair = TWO_VALUED.get((view, c["col"]))
    if pair and op == "!=" and v in pair:
        op, v = "==", pair[1 - pair.index(v)]
    return {"col": c["col"], "op": op, "v": v}


def fold(part: dict) -> dict:
    """`part` (an atom, or ``{"not": atom}``, in its said form) as one atom with no ``not``: a comparison turned
    (`INV`), ``any`` and ``none`` exchanged, ``not all`` of one condition as ``any`` of its complement, ``not at`` as
    the turned comparison that holds when there is no such row. ``not all`` of two conditions raises ValueError: it
    is an or-filter, which the slots do not have."""
    neg = False
    while "q" not in part:                                 # {"not": ...} around the atom, any depth
        neg, part = not neg, part["not"]
    a = {k: v for k, v in part.items() if k != "not"}
    q, view, own = a["q"], a["view"], bool(part.get("not"))
    if "where" in a:
        a["where"] = [_plain(view, c) for c in a["where"]]
    if q == "at":
        c = _plain(view, {"col": a["col"], "op": a["op"], "v": a["val"], "not": own != neg})
        a.update(op=c["op"], val=c["v"])
        if neg:                                           # the whole atom negated: no such row now holds, or fails
            if a.pop("absent", None) != "pass":
                a["absent"] = "pass"
        return a
    if own == neg:
        return a
    if q in ("n", "count"):
        a["op"] = INV[a["op"]]
    elif q in ("any", "none"):
        a["q"] = "none" if q == "any" else "any"
    else:                                                 # not every row matches: some row fails the condition
        if len(a["where"]) != 1:
            raise ValueError("not all of two conditions has no folded form")
        a.update(q="any", where=[_plain(view, dict(a["where"][0], **{"not": True}))])
    return a


__all__ = ["FILTERS", "INV", "NUM_OPS", "STR_OPS", "TWO_VALUED", "atoms", "cmp", "evaluate", "fold", "match"]
