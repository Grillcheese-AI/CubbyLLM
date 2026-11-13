"""Order 0072, build step 5: a Python evaluator of every verb kind's contract, written from the order's words, apart
from the CubeLang (`cubbyllm/speech/verb_programs.py`), so the VM's verdict has something independent to agree with.

`evaluate(kind, variant, data)` -> 1 / 0 (the gate: 2 do, 1 ask first, 0 refuse). `data` is what the host binds:
groups ``<g>_<col>`` with ``n_<g>``, single values ``arg_<name>``.
"""
from __future__ import annotations

from collections import Counter

__wiring__ = "STANDALONE"

OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def _rows(d, g, *cols):
    return list(zip(*(d[f"{g}_{c}"] for c in cols))) if cols else []


def pick(v, d):
    if v == "read":                                       # the named field or passage, exactly
        return int(d["out_val"] == [d["arg_want"]])
    meets = OPS[d["arg_op"]]
    given_ok = all(meets(k, d["arg_t"]) and y in d["in_val"] for y, k in _rows(d, "out", "val", "key"))
    complete = not d.get("arg_all") or all(x in d["out_val"] for x, k in _rows(d, "in", "val", "key") if meets(k, d["arg_t"]))
    return int(given_ok and complete)


def reorder(v, d):
    if v == "sort":
        keys = d["out_key"]
        ordered = all((a <= b) if d["arg_dir"] == "asc" else (a >= b) for a, b in zip(keys, keys[1:]))
        return int(Counter(d["in_val"]) == Counter(d["out_val"]) and ordered)
    rest = list(d["in_val"])
    rest.remove(d["arg_item"]) if d["arg_item"] in rest else None
    out, p = d["out_val"], d["arg_place"]
    return int(len(out) == len(d["in_val"]) and p < len(out) and out[p] == d["arg_item"] and out[:p] + out[p + 1:] == rest)


def edit(v, d):
    x, y = d.get("in_val", []), d.get("out_val", [])
    if v == "replace":
        return int(y == [d["arg_new"] if a == d["arg_old"] else a for a in x])
    if v == "add":
        p = d["arg_place"]
        return int(y == x[:p] + [d["arg_item"]] + x[p:])
    if v == "delete":
        return int(y == [a for a in x if a != d["arg_item"]])
    if v == "set":
        want = [(k, d["arg_value"] if k == d["arg_name"] else a) for k, a in zip(d["in_key"], x)]
        return int(list(zip(d["out_key"], y)) == want and len(y) == len(d["out_key"]))
    if v == "clear":
        return int(y == [])
    if v == "copy":
        return int(y == x and d["orig_val"] == x)
    if v == "same":
        return int(y == x)
    if v == "merge":
        return int(Counter(y) == Counter(x) + Counter(d["in2_val"]))
    return int(y == [a + d["arg_k"] if OPS[d["arg_op"]](a, d["arg_t"]) else a for a in x])     # apply


def compute(v, d):
    if v == "aggregate":
        xs, fn = d["in_val"], d["arg_fn"]
        if fn in ("max", "min") and not xs:
            return 0
        want = {"sum": sum, "count": len, "max": max, "min": min}[fn](xs)
        return int(want == d["arg_got"])
    if v == "expression":
        a, b = d["arg_a"], d["arg_b"]
        return int({"add": a + b, "sub": a - b, "mul": a * b}[d["arg_fn"]] == d["arg_got"])
    if v == "larger":
        a, b = d["arg_a"], d["arg_b"]
        return int(("a" if a > b else "b" if b > a else "equal") == d["arg_got"])
    return int(d["exp_val"] == d["out_val"])            # mapped: the host function's result, item by item


def verdict(v, d):
    if v == "required":
        return int(set(d["in_key"]) <= set(d["out_key"]))
    if v == "tests":
        return int(d["arg_ran"] > 0 and d["arg_failed"] == 0)
    return int(d["a_val"] == d["b_val"])


def write(v, d):
    ok = 0 < d["arg_words"] and d["arg_min"] <= d["arg_words"] <= d["arg_max"] and d["arg_lang"] == d["arg_want_lang"]
    if "arg_subject" in d:
        ok = ok and d["arg_subject"] in d["out_lower"]
    if "arg_share" in d:
        ok = ok and d["arg_words"] * 100 <= d["arg_share"] * d["arg_src_words"]
    return int(ok)


def act(v, d):
    if v == "read_back":
        return int(d["arg_got"] == d["arg_want"])
    if not d["arg_tool"]:
        return 0
    return (2 if d["arg_yes"] else 1) if d["arg_class"] == "ask first" else 2


def evaluate(kind: str, variant: str, data: dict) -> int:
    return {"pick": pick, "reorder": reorder, "edit": edit, "compute": compute, "verdict": verdict, "write": write,
            "act": act}[kind](variant, data)


__all__ = ["evaluate"]
