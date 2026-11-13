"""Order 0067, the second reader's step 0 (Nick, 2026-11-13: "go ahead with step 0"): which operations the program
generator needs does the VM build actually run? Checked against a build, never the docs.

Each probe is a small `ISolve` program (the form `rule_plan.program` already runs) whose `solve` returns a number; a
probe passes only when it compiles, runs, and returns the expected value. A probe that compiles but returns another
value is reported as wrong (the operation exists in the parser but not in what runs). Several probes try other
spellings of one operation (`starts_with`, `startsWith`, ...): the operation counts as present if any spelling works.

The VM is `$CUBELANG_EXE` (the run records which build, `vm_identity`).

  python validation/vm_ops_probe.py [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402

__wiring__ = "STANDALONE"

N = "create r: number; assign r = 0;"
# (operation, spelling, body, expected): the body ends by returning r (or a value).
PROBES = [
    ("list literal + for-of", "numbers", N + " let xs = [3, 5, 7]; for (let x of xs) { add r, x; } return r;", 15),
    ("list literal + for-of", "strings", N + ' let xs = ["a", "b", "a"]; for (let x of xs) { add r, 1; } return r;', 3),
    ("list length", ".len()", 'let xs = [1, 2, 3, 4]; return xs.len();', 4),
    ("list index", "xs[i]", 'let xs = [4, 9, 2]; create i: number; assign i = 1; return xs[i];', 9),
    ("count with a filter", "if in loop", N + " let xs = [3, 5, 7, 1]; for (let x of xs) { if (x > 4) { add r, 1; } } return r;", 2),
    ("string equality", "==", N + ' let xs = ["a", "b", "a"]; for (let w of xs) { if (w == "a") { add r, 1; } } return r;', 2),
    ("string inequality", "!=", N + ' let xs = ["a", "b", "a"]; for (let w of xs) { if (w != "a") { add r, 1; } } return r;', 1),
    ("numeric comparison", "<", N + " if (2 < 3) { assign r = 1; } return r;", 1),
    ("numeric comparison", "<=", N + " if (3 <= 3) { assign r = 1; } return r;", 1),
    ("numeric comparison", ">", N + " if (3 > 2) { assign r = 1; } return r;", 1),
    ("numeric comparison", ">=", N + " if (2 >= 3) { assign r = 1; } return r;", 0),
    ("numeric comparison", "==", N + " if (3 == 3) { assign r = 1; } return r;", 1),
    ("numeric comparison", "!=", N + " if (3 != 3) { assign r = 1; } return r;", 0),
    ("&& as an operand", "a && b", N + " create a: number; assign a = 2; if (a > 1 && a < 5) { assign r = 1; } return r;", 1),
    ("&& as an operand", "false side", N + " create a: number; assign a = 7; if (a > 1 && a < 5) { assign r = 1; } return r;", 0),
    ("|| as an operand", "a || b", N + " create a: number; assign a = 7; if (a < 1 || a > 5) { assign r = 1; } return r;", 1),
    ("|| as an operand", "false side", N + " create a: number; assign a = 3; if (a < 1 || a > 5) { assign r = 1; } return r;", 0),
    ("not", "!(...)", N + " create a: number; assign a = 3; if (!(a > 5)) { assign r = 1; } return r;", 1),
    ("else if", "else if", N + " create a: number; assign a = 3; if (a > 5) { assign r = 1; } else if (a > 2) { assign r = 2; } else { assign r = 3; } return r;", 2),
    ("all over a list", "flag", "create ok: number; assign ok = 1; let xs = [3, 5, 7]; for (let x of xs) { if (x < 3) { assign ok = 0; } } return ok;", 1),
    ("any over a list", "flag", "create hit: number; assign hit = 0; let xs = [3, 5, 7]; for (let x of xs) { if (x == 5) { assign hit = 1; } } return hit;", 1),
    ("arithmetic expression", "a + b * 2", "create a: number; assign a = 3; create b: number; assign b = 4; let c = a + b * 2; return c;", 11),
    ("modulo", "a % 2", "create a: number; assign a = 7; let c = a % 2; return c;", 1),
    ("while + break", "while", N + " create i: number; assign i = 0; while (i < 10) { add i, 1; if (i == 4) { break; } } return i;", 4),
    ("nested lists", "[[..],[..]]", N + " let xs = [[1, 2], [3]]; for (let row of xs) { add r, row.len(); } return r;", 3),
    ("a function with parameters", "call", None, 6),
    ("string length", ".len()", 'let s = "hello"; return s.len();', 5),
    ("string length", "len(s)", 'let s = "hello"; return len(s);', 5),
    ("string index", "s[0]", N + ' let s = "hello"; if (s[0] == "h") { assign r = 1; } return r;', 1),
    ("prefix", ".starts_with", N + ' let s = "hello"; if (s.starts_with("he")) { assign r = 1; } return r;', 1),
    ("prefix", ".startsWith", N + ' let s = "hello"; if (s.startsWith("he")) { assign r = 1; } return r;', 1),
    ("prefix", "starts_with(s, p)", N + ' let s = "hello"; if (starts_with(s, "he")) { assign r = 1; } return r;', 1),
    ("suffix", ".ends_with", N + ' let s = "hello"; if (s.ends_with("lo")) { assign r = 1; } return r;', 1),
    ("suffix", ".endsWith", N + ' let s = "hello"; if (s.endsWith("lo")) { assign r = 1; } return r;', 1),
    ("suffix", "ends_with(s, p)", N + ' let s = "hello"; if (ends_with(s, "lo")) { assign r = 1; } return r;', 1),
    ("case", ".to_upper", N + ' let s = "hello"; if (s.to_upper() == "HELLO") { assign r = 1; } return r;', 1),
    ("case", ".toUpperCase", N + ' let s = "hello"; if (s.toUpperCase() == "HELLO") { assign r = 1; } return r;', 1),
    ("case", "upper(s)", N + ' let s = "hello"; if (upper(s) == "HELLO") { assign r = 1; } return r;', 1),
    ("case", ".to_lower", N + ' let s = "HeLLo"; if (s.to_lower() == "hello") { assign r = 1; } return r;', 1),
    ("substring / contains", ".contains", N + ' let s = "hello"; if (s.contains("ell")) { assign r = 1; } return r;', 1),
    ("string concatenation", "+", N + ' let s = "he" + "llo"; if (s == "hello") { assign r = 1; } return r;', 1),
    ("map access", 'm["k"]', 'let m = {"k": 5}; return m["k"];', 5),
    ("a string from the host", "the mention argument", N + ' if (mention == "probe") { assign r = 1; } return r;', 1),
    # the same operations in the forms this build encodes: a value bound first, opcode statements, nested ifs
    ("list length", "let n = xs.len()", "let xs = [1, 2, 3, 4]; let n = xs.len(); return n;", 4),
    ("list index", "let v = xs[i]", "let xs = [4, 9, 2]; create i: number; assign i = 1; let v = xs[i]; return v;", 9),
    ("&& as an operand", "nested ifs", N + " create a: number; assign a = 2; if (a > 1) { if (a < 5) { assign r = 1; } } return r;", 1),
    ("|| as an operand", "two ifs, one flag", N + " create a: number; assign a = 7; if (a < 1) { assign r = 1; } if (a > 5) { assign r = 1; } return r;", 1),
    ("not", "else branch", N + " create a: number; assign a = 3; if (a > 5) { } else { assign r = 1; } return r;", 1),
    ("else if", "else { if }", N + " create a: number; assign a = 3; if (a > 5) { assign r = 1; } else { if (a > 2) { assign r = 2; } else { assign r = 3; } } return r;", 2),
    ("arithmetic expression", "add / mul statements", "create c: number; assign c = 4; mul c, 2; add c, 3; return c;", 11),
    ("modulo", "div statement (whole?)", "create c: number; assign c = 7; div c, 2; return c;", 3),
    ("while + break", "while, no break", N + " create i: number; assign i = 0; while (i < 4) { add i, 1; } return i;", 4),
    ("nested lists", "let n = row.len()", N + " let xs = [[1, 2], [3]]; for (let row of xs) { let n = row.len(); add r, n; } return r;", 3),
    ("a function with parameters", "mul in the callee", None, 6),
    ("map access", 'let v = m["k"]', 'let m = {"k": 5}; let v = m["k"]; return v;', 5),
    ("string index", 'let c = s[0]', N + ' let s = "hello"; let c = s[0]; if (c == "h") { assign r = 1; } return r;', 1),
    ("string length", 'let n = s.len()', 'let s = "hello"; let n = s.len(); return n;', 5),
    # string helpers composed from characters and length (no built-in helper runs)
    ("prefix", "composed: char loop", 'create ok: number; assign ok = 1; let s = "hello"; let p = "he"; let n = p.len(); '
     'create i: number; assign i = 0; while (i < n) { let a = s[i]; let b = p[i]; if (a == b) { } else { assign ok = 0; } '
     'add i, 1; } return ok;', 1),
    ("prefix", "composed: char loop, a miss", 'create ok: number; assign ok = 1; let s = "hello"; let p = "hx"; let n = p.len(); '
     'create i: number; assign i = 0; while (i < n) { let a = s[i]; let b = p[i]; if (a == b) { } else { assign ok = 0; } '
     'add i, 1; } return ok;', 0),
    ("suffix", "composed: char loop from the end", 'create ok: number; assign ok = 1; let s = "hello"; let p = "lo"; '
     'let n = p.len(); let m = s.len(); create i: number; assign i = 0; create j: number; while (i < n) { assign j = m; '
     'sub j, n; add j, i; let a = s[j]; let b = p[i]; if (a == b) { } else { assign ok = 0; } add i, 1; } return ok;', 1),
    ("case", "composed: alphabet membership", 'create caps: number; assign caps = 0; let s = "HeLLo"; '
     'let up = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]; '
     'let n = s.len(); create i: number; assign i = 0; while (i < n) { let c = s[i]; for (let u of up) { '
     'if (c == u) { add caps, 1; } } add i, 1; } return caps;', 3),
    ("modulo", "composed: subtract in a loop", "create c: number; assign c = 7; while (c >= 2) { sub c, 2; } return c;", 1),
    # two string variables compared (the composed helpers above rest on it)
    ("string variable == string variable", "let, unequal", N + ' let a = "x"; let b = "y"; if (a == b) { assign r = 1; } return r;', 0),
    ("string variable == string variable", "create/assign, unequal", N + ' create a: str; assign a = "x"; create b: str; '
     'assign b = "y"; if (a == b) { assign r = 1; } return r;', 0),
    ("string variable == string variable", "let, equal", N + ' let a = "x"; let b = "x"; if (a == b) { assign r = 1; } return r;', 1),
    ("string variable == string variable", "chars of two strings, unequal", N + ' let s = "ab"; let a = s[0]; let b = s[1]; '
     'if (a == b) { assign r = 1; } return r;', 0),
    ("number variable == number variable", "unequal", N + " create a: number; assign a = 3; create b: number; assign b = 4; "
     "if (a == b) { assign r = 1; } return r;", 0),
    ("number variable < number variable", "4 < 3", N + " create a: number; assign a = 4; create b: number; assign b = 3; "
     "if (a < b) { assign r = 1; } return r;", 0),
    ("string variable == literal", "unequal", N + ' let a = "x"; if (a == "y") { assign r = 1; } return r;', 0),
    ("suffix", "composed, a miss", 'create ok: number; assign ok = 1; let s = "hello"; let p = "xo"; '
     'let n = p.len(); let m = s.len(); create i: number; assign i = 0; create j: number; while (i < n) { assign j = m; '
     'sub j, n; add j, i; let a = s[j]; let b = p[i]; if (a == b) { } else { assign ok = 0; } add i, 1; } return ok;', 0),
    ("substring / contains", "composed: window compare", 'create hit: number; assign hit = 0; let s = "hello"; let p = "ell"; '
     'let n = p.len(); let m = s.len(); create k: number; assign k = 0; create last: number; assign last = m; sub last, n; '
     'create i: number; create ok: number; create j: number; while (k <= last) { assign ok = 1; assign i = 0; '
     'while (i < n) { assign j = k; add j, i; let a = s[j]; let b = p[i]; if (a == b) { } else { assign ok = 0; } add i, 1; } '
     'if (ok == 1) { assign hit = 1; } add k, 1; } return hit;', 1),
]
CALL_SRC2 = """program P implements ISolve {
    function twice(x: number): number { create y: number; assign y = x; mul y, 2; return y; }
    public function solve(mention: str): number { let v = twice(3); return v; }
}
"""
CALL_SRC = """program P implements ISolve {
    function twice(x: number): number { let y = x * 2; return y; }
    public function solve(mention: str): number { create r: number; assign r = 0; let v = twice(3); return v; }
}
"""


def source(body: str) -> str:
    return f"program P implements ISolve {{\n    public function solve(mention: str): number {{\n        {body}\n    }}\n}}\n"


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vm = cc.CubelangSession()
    probes, ops = [], defaultdict(list)
    for op, spelling, body, want in PROBES:
        src = (CALL_SRC2 if spelling == "mul in the callee" else CALL_SRC) if body is None else source(body)
        try:
            out = vm.run(src, fn="solve", args=["probe"])
        except Exception as e:                           # a transport failure is reported, not hidden
            out = {"error": f"{type(e).__name__}: {e}"}
        got = out.get("result") if isinstance(out, dict) else None
        err = None if got is not None else str((out or {}).get("error") or (out or {}).get("errors") or out)[:300]
        status = "works" if num(got) == float(want) else ("wrong value" if got is not None else "fails")
        probes.append({"operation": op, "spelling": spelling, "status": status, "want": want, "got": got, "error": err,
                       "source": src})
        ops[op].append(status)
        print(f"  {status:12s} {op:30s} {spelling:22s} got {got!r}" + (f"  ({err[:90]})" if err else ""), flush=True)
    summary = {op: ("present" if "works" in st else "wrong value" if "wrong value" in st else "missing")
               for op, st in ops.items()}
    out = {"order": "0067", "what": "the second reader's step 0: the generator's operations, run on the VM build",
           "vm": vm_identity.identity(), "summary": summary, "probes": probes}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
