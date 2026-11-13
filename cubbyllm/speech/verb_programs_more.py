"""cubbyllm/speech/verb_programs_more.py -- the compute, verdict, write and act kinds' programs (order 0072, step 2;
the naming and the reading rules are `verb_programs.py`'s).

- compute: ``aggregate`` (``arg_fn``: sum, max, min, count of ``in_val``) equals the reply's ``arg_got``; ``expression``
  (``arg_a`` ``arg_fn`` ``arg_b``: add, sub, mul); ``larger`` ("a", "b" or "equal"); ``mapped`` (each ``out_val`` equals
  the host function's ``exp_val``: process, parse, export).
- verdict: ``required`` (every ``in_key`` among ``out_key``); ``tests`` (``arg_ran`` > 0 and ``arg_failed`` == 0);
  ``equal`` (``a`` and ``b`` item by item).
- write, form only: not empty, word count within ``arg_min`` .. ``arg_max``, ``arg_lang`` == ``arg_want_lang``, the
  subject's word among ``out_lower`` (when ``arg_subject`` is given), at most ``arg_share`` percent of ``arg_src_words``.
- act: ``gate`` -> 2 do, 1 ask first, 0 refuse (no tool); ``read_back`` -> 1 when ``arg_got`` == ``arg_want``.

Wiring: STANDALONE (`verb_programs.program` reads it).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"


def _compute(v, d):
    if v == "aggregate":
        fn = d["arg_fn"]
        init = {"sum": "assign s = 0;", "count": "assign s = n_in;", "max": "assign s = 0; sub s, 1000000000;",
                "min": "assign s = 1000000000;"}[fn]
        step = {"sum": "add s, x;", "count": "", "max": "if (x > s) { assign s = x; }",
                "min": "if (x < s) { assign s = x; }"}[fn]
        loop = "" if fn == "count" else f"assign i = 0; while (i < n_in) {{ let x = in_val[i]; {step} add i, 1; }}"
        guard = "if (n_in == 0) { assign ok = 0; }" if fn in ("max", "min") else ""
        return ["create ok: number; assign ok = 0; create s: number; create i: number;", init, loop,
                "if (s == arg_got) { assign ok = 1; }", guard, "return ok;"]
    if v == "expression":
        op = {"add": "add", "sub": "sub", "mul": "mul"}[d["arg_fn"]]
        return ["create ok: number; assign ok = 0; create s: number; assign s = arg_a;", f"{op} s, arg_b;",
                "if (s == arg_got) { assign ok = 1; }", "return ok;"]
    if v == "larger":
        return ['let w = "equal";', 'if (arg_a > arg_b) { assign w = "a"; } if (arg_b > arg_a) { assign w = "b"; }',
                "create ok: number; assign ok = 0; if (w == arg_got) { assign ok = 1; }", "return ok;"]
    return ["create ok: number; assign ok = 1; create i: number;", "if (n_exp != n_out) { assign ok = 0; }",
            "assign i = 0; while (i < n_exp) { if (i < n_out) { let x = exp_val[i]; let y = out_val[i]; "
            "if (x != y) { assign ok = 0; } } add i, 1; }", "return ok;"]


def _verdict(v, d):
    if v == "required":
        return ["create ok: number; assign ok = 1; create c: number; create i: number; create j: number;",
                "assign i = 0; while (i < n_in) { let k = in_key[i]; assign c = 0; assign j = 0; "
                "while (j < n_out) { let h = out_key[j]; if (h == k) { add c, 1; } add j, 1; } "
                "if (c == 0) { assign ok = 0; } add i, 1; }", "return ok;"]
    if v == "tests":
        return ["create ok: number; assign ok = 0;", "if (arg_ran > 0) { if (arg_failed == 0) { assign ok = 1; } }",
                "return ok;"]
    return ["create ok: number; assign ok = 1; create i: number;", "if (n_a != n_b) { assign ok = 0; }",
            "assign i = 0; while (i < n_a) { if (i < n_b) { let x = a_val[i]; let y = b_val[i]; "
            "if (x != y) { assign ok = 0; } } add i, 1; }", "return ok;"]


def _write(v, d):
    body = ["create ok: number; assign ok = 1;", "if (arg_words == 0) { assign ok = 0; }",
            "if (arg_words < arg_min) { assign ok = 0; } if (arg_words > arg_max) { assign ok = 0; }",
            "if (arg_lang != arg_want_lang) { assign ok = 0; }"]
    if "arg_subject" in d:
        body += ["create c: number; assign c = 0; create i: number; assign i = 0;",
                 "while (i < n_out) { let w = out_lower[i]; if (w == arg_subject) { add c, 1; } add i, 1; }",
                 "if (c == 0) { assign ok = 0; }"]
    if "arg_share" in d:                                  # words * 100 <= share * source words
        body += ["create l: number; assign l = arg_words; mul l, 100; create r: number; assign r = arg_src_words;",
                 "mul r, arg_share; if (l > r) { assign ok = 0; }"]
    return body + ["return ok;"]


def _act(v, d):
    if v == "read_back":
        return ["create ok: number; assign ok = 0; if (arg_got == arg_want) { assign ok = 1; }", "return ok;"]
    # the gate: no tool -> refuse; read / change -> do; ask first -> do only with a yes given in this turn
    return ["create r: number; assign r = 0;",
            'if (arg_tool == 1) { if (arg_class == "ask first") { if (arg_yes == 1) { assign r = 2; } else { assign r = 1; } } '
            "else { assign r = 2; } }", "return r;"]


def body(kind: str, variant: str, data: dict) -> list[str]:
    return {"compute": _compute, "verdict": _verdict, "write": _write, "act": _act}[kind](variant, data)


__all__ = ["body"]
