"""cubbyllm/speech/verb_programs.py -- one CubeLang program per verb kind (order 0072, build step 2).

What a step of an instruction asks is checked by a program over values the host binds (never computed from a missing
value: cubelang's P6-1, P6-8, P6-9 are not on the build yet). Names follow the reply views' convention, so the static
guard (`verb_guard.py`) can check every read: a group of parallel lists ``<g>_<col>`` with its length ``n_<g>`` (``in``
the input, ``out`` what the reply or the host's tool gave, ``in2`` a second input, ``exp`` what the host's own
function gives), and single values ``arg_<name>``. A list is read only as ``let x = <g>_<col>[i];`` inside ``while
(i < n_<g>)`` or ``if (i < n_<g>)``.

`program(kind, variant, data)` -> the whole program; its ``solve`` returns 1 (holds) or 0, or for the gate 2 (do), 1
(ask the user first), 0 (refuse). `VARIANTS`: the variants of each kind and the data each reads.

Wiring: STANDALONE (0072's measurement and 0067's second reader read it; nothing in the serve path).
"""
from __future__ import annotations

from cubbyllm.speech.reply_views import _lit

__wiring__ = "STANDALONE"

VARIANTS = {
    "pick": ("filter", "read"), "reorder": ("sort", "move"),
    "edit": ("replace", "add", "delete", "set", "clear", "copy", "merge", "apply", "same"),
    "compute": ("aggregate", "expression", "larger", "mapped"), "verdict": ("required", "tests", "equal"),
    "write": ("form",), "act": ("gate", "read_back"),
}


def _bind(data: dict) -> list[str]:
    out = []
    for k, v in data.items():
        if isinstance(v, list):
            out.append(f"let {k} = [{', '.join(_lit(x) for x in v)}];")
        elif isinstance(v, str):
            out.append(f"let {k} = {_lit(v)};")
        elif int(v) < 0:                                  # the VM takes no unary minus as an operand
            out.append(f"create {k}: number; assign {k} = 0; sub {k}, {-int(v)};")
        else:
            out.append(f"create {k}: number; assign {k} = {int(v)};")
    return out


def _wrap(data: dict, body: list[str]) -> str:
    lines = _bind(data) + body
    return ("program VerbCheck implements ISolve {\n    public function solve(mention: str): number {\n        "
            + "\n        ".join(lines) + "\n    }\n}\n")


def _count(g: str, col: str, value: str, c: str, i: str) -> str:
    """``c`` = how many rows of ``g`` have ``col`` equal to ``value``."""
    return (f"assign {c} = 0; assign {i} = 0; while ({i} < n_{g}) {{ let q_{i} = {g}_{col}[{i}]; "
            f"if (q_{i} == {value}) {{ add {c}, 1; }} add {i}, 1; }}")


def _same_seq(a: str, b: str, ok: str) -> list[str]:
    """``a_val`` and ``b_val`` equal, item by item (``n_a == n_b`` first)."""
    return [f"if (n_{a} != n_{b}) {{ assign {ok} = 0; }}", "create i: number; assign i = 0;",
            f"while (i < n_{a}) {{ if (i < n_{b}) {{ let x = {a}_val[i]; let y = {b}_val[i]; "
            f"if (x != y) {{ assign {ok} = 0; }} }} add i, 1; }}"]


def _pick(v, d):
    op = d.get("arg_op")
    if v == "read":                                       # the named field or passage, exactly
        return ["create ok: number; assign ok = 0;",
                "if (n_out == 1) { if (0 < n_out) { let y = out_val[0]; if (y == arg_want) { assign ok = 1; } } }",
                "return ok;"]
    body = ["create ok: number; assign ok = 1; create c: number; create i: number; create j: number;",
            "assign i = 0; while (i < n_out) { let y = out_val[i]; let yk = out_key[i]; "
            f"if (yk {op} arg_t) {{ }} else {{ assign ok = 0; }} {_count('in', 'val', 'y', 'c', 'j')} "
            "if (c == 0) { assign ok = 0; } add i, 1; }"]
    if d.get("arg_all"):                                  # complete: every input item that meets it is given
        body.append("assign j = 0; while (j < n_in) { let x = in_val[j]; let xk = in_key[j]; "
                    f"if (xk {op} arg_t) {{ {_count('out', 'val', 'x', 'c', 'i')} if (c == 0) {{ assign ok = 0; }} }} "
                    "add j, 1; }")
    return body + ["return ok;"]


def _reorder(v, d):
    if v == "sort":                                       # the same items, in the host key's order
        cmp = "<" if d["arg_dir"] == "asc" else ">"
        return ["create ok: number; assign ok = 1; create a: number; create b: number; create i: number; create j: number;",
                "if (n_in != n_out) { assign ok = 0; }",
                "create p: number; assign i = 1; while (i < n_out) { assign p = i; sub p, 1; if (p >= 0) { if (p < n_out) { "
                f"let k1 = out_key[p]; let k2 = out_key[i]; if (k2 {cmp} k1) {{ assign ok = 0; }} }} }} add i, 1; }}",
                "create m: number; assign m = 0; while (m < n_in) { let x = in_val[m]; "
                f"{_count('in', 'val', 'x', 'a', 'i')} {_count('out', 'val', 'x', 'b', 'j')} "
                "if (a != b) { assign ok = 0; } add m, 1; }", "return ok;"]
    # move: the named item at the named place, the rest in their order
    return ["create ok: number; assign ok = 1; create i: number; create j: number;",
            "if (n_in != n_out) { assign ok = 0; }",
            "if (arg_place < n_out) { let y = out_val[arg_place]; if (y != arg_item) { assign ok = 0; } } "
            "else { assign ok = 0; }",
            "create skipped: number; assign skipped = 0; assign i = 0; assign j = 0;",
            "while (i < n_in) { let x = in_val[i]; create hit: number; assign hit = 0; "
            "if (skipped == 0) { if (x == arg_item) { assign skipped = 1; assign hit = 1; } } "
            "if (hit == 0) { if (j == arg_place) { add j, 1; } if (j < n_out) { let y = out_val[j]; "
            "if (y != x) { assign ok = 0; } } else { assign ok = 0; } add j, 1; } add i, 1; }", "return ok;"]


def _edit(v, d):
    ok = ["create ok: number; assign ok = 1; create i: number; create j: number; create c: number; create e: number;"]
    if v == "replace":                                    # every old becomes new, nothing else changes
        return ok + ["if (n_in != n_out) { assign ok = 0; }",
                     "assign i = 0; while (i < n_in) { if (i < n_out) { let x = in_val[i]; let y = out_val[i]; "
                     "if (x == arg_old) { if (y != arg_new) { assign ok = 0; } } else { if (y != x) { assign ok = 0; } } } "
                     "add i, 1; }", "return ok;"]
    if v in ("add", "delete"):                            # add: the item at its place; delete: every such item gone
        skip = "arg_place" if v == "add" else None
        lines = ["create want: number; assign want = n_in;"]
        if v == "add":
            lines += ["add want, 1; if (want != n_out) { assign ok = 0; }",
                      "if (arg_place < n_out) { let y = out_val[arg_place]; if (y != arg_item) { assign ok = 0; } } "
                      "else { assign ok = 0; }",
                      "assign i = 0; assign j = 0; while (i < n_in) { if (j == arg_place) { add j, 1; } "
                      "let x = in_val[i]; if (j < n_out) { let y = out_val[j]; if (y != x) { assign ok = 0; } } "
                      "add j, 1; add i, 1; }"]
        else:
            lines += ["assign i = 0; assign j = 0; while (i < n_in) { let x = in_val[i]; if (x != arg_item) { "
                      "if (j < n_out) { let y = out_val[j]; if (y != x) { assign ok = 0; } } else { assign ok = 0; } "
                      "add j, 1; } add i, 1; }", "if (j != n_out) { assign ok = 0; }"]
        assert skip or v == "delete"
        return ok + lines + ["return ok;"]
    if v == "set":                                        # the named value set, the rest unchanged
        return ok + ["if (n_in != n_out) { assign ok = 0; }",
                     "assign i = 0; while (i < n_in) { if (i < n_out) { let k = in_key[i]; let k2 = out_key[i]; "
                     "let x = in_val[i]; let y = out_val[i]; if (k != k2) { assign ok = 0; } "
                     "if (k == arg_name) { if (y != arg_value) { assign ok = 0; } } else { if (y != x) { assign ok = 0; } } } "
                     "add i, 1; }", "return ok;"]
    if v == "clear":
        return ["create ok: number; assign ok = 0; if (n_out == 0) { assign ok = 1; }", "return ok;"]
    if v in ("copy", "same", "export"):                    # copy: the original unchanged and an equal copy
        body = _same_seq("in", "out", "ok")
        if v == "copy":
            body += [ln.replace("n_out", "n_orig").replace("out_val", "orig_val") for ln in _same_seq("in", "out", "ok")]
        return ok + body + ["return ok;"]
    if v == "merge":                                      # every item of each input once, nothing added
        return ok + ["create want: number; assign want = n_in; add want, n_in2; if (want != n_out) { assign ok = 0; }",
                     "create m: number; assign m = 0; while (m < n_out) { let x = out_val[m]; "
                     f"{_count('in', 'val', 'x', 'c', 'i')} {_count('in2', 'val', 'x', 'e', 'j')} add c, e; "
                     f"{_count('out', 'val', 'x', 'e', 'j')} if (c != e) {{ assign ok = 0; }} add m, 1; }}", "return ok;"]
    # apply: the rule (add arg_k to every item arg_op arg_t) on the items it covers, the others unchanged
    return ok + ["if (n_in != n_out) { assign ok = 0; }",
                 "assign i = 0; while (i < n_in) { if (i < n_out) { let x = in_val[i]; let y = out_val[i]; "
                 f"create w: number; assign w = x; if (x {d['arg_op']} arg_t) {{ add w, arg_k; }} "
                 "if (y != w) { assign ok = 0; } } add i, 1; }", "return ok;"]


def program(kind: str, variant: str, data: dict) -> str:
    from cubbyllm.speech import verb_programs_more as more
    body = {"pick": _pick, "reorder": _reorder, "edit": _edit}.get(kind)
    body = body(variant, data) if body else more.body(kind, variant, data)
    shown = {k: v for k, v in data.items() if k not in ("arg_op", "arg_dir", "arg_fn")}   # spelled into the body
    return _wrap(shown, body)


__all__ = ["VARIANTS", "program"]
