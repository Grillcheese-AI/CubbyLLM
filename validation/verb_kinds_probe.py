"""Order 0072 (Cowork, 2026-11-13): one CubeLang program per verb kind, each run on a pass and a fail case.

The kinds of `docs/orders/0072-verb-programs.md`: pick, reorder, edit, compute, verdict, write (form only) and act
(the gate and the read-back). Values are bound as literals by the host, lengths are checked before any index, and
nothing is compared that can be missing (cubelang P6-1 is not built yet). The VM is `$CUBELANG_EXE`.
Usage: python validation/verb_kinds_probe.py [out.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402


def src(body):
    return ("program P implements ISolve {\n    public function solve(mention: str): number {\n        "
            + body + "\n    }\n}\n")


def lit(xs):
    return "[" + ", ".join(json.dumps(x) for x in xs) + "]"


def count_body(name_in, name_list, counter):
    return (f"assign {counter} = 0; for (let q of {name_list}) {{ if (q == {name_in}) {{ add {counter}, 1; }} }} ")


# reorder: sort (ascending by the host's key). ok iff same length, non-decreasing, same multiset.
def sort_prog(xs, ys):
    return src(
        f"let xs = {lit(xs)}; let ys = {lit(ys)}; create ok: number; assign ok = 1; "
        "let n = xs.len(); let m = ys.len(); if (n != m) { assign ok = 0; } "
        "create first: number; assign first = 1; create prev: number; assign prev = 0; "
        "for (let y of ys) { if (first == 1) { assign first = 0; } else { if (y < prev) { assign ok = 0; } } "
        "assign prev = y; } "
        "create a: number; create b: number; "
        "for (let x of xs) { assign a = 0; for (let q of xs) { if (q == x) { add a, 1; } } "
        "assign b = 0; for (let q of ys) { if (q == x) { add b, 1; } } if (a != b) { assign ok = 0; } } "
        "return ok;")


# pick: filter (keep items above t). ok iff every kept item is in the input and above t, and every input item above
# t is kept (complete).
def filter_prog(xs, t, ys):
    return src(
        f"let xs = {lit(xs)}; let ys = {lit(ys)}; create t: number; assign t = {t}; create ok: number; assign ok = 1; "
        "create c: number; "
        "for (let y of ys) { if (y > t) { } else { assign ok = 0; } assign c = 0; "
        "for (let q of xs) { if (q == y) { add c, 1; } } if (c == 0) { assign ok = 0; } } "
        "for (let x of xs) { if (x > t) { assign c = 0; for (let q of ys) { if (q == x) { add c, 1; } } "
        "if (c == 0) { assign ok = 0; } } } "
        "return ok;")


# edit: rename (every old word becomes new, nothing else changes). Lengths checked before any index.
def rename_prog(xs, old, new, ys):
    return src(
        f"let xs = {lit(xs)}; let ys = {lit(ys)}; let old = {json.dumps(old)}; let new = {json.dumps(new)}; "
        "create ok: number; assign ok = 1; let n = xs.len(); let m = ys.len(); "
        "if (n != m) { assign ok = 0; } else { create i: number; assign i = 0; "
        "while (i < n) { let a = xs[i]; let b = ys[i]; "
        "if (a == old) { if (b != new) { assign ok = 0; } } else { if (b != a) { assign ok = 0; } } add i, 1; } } "
        "return ok;")


# compute: calculate (the total of the items); the reply's number is bound by the host.
def total_prog(xs, got):
    return src(
        f"let xs = {lit(xs)}; create s: number; assign s = 0; for (let x of xs) {{ add s, x; }} "
        f"create g: number; assign g = {got}; create ok: number; assign ok = 0; if (s == g) {{ assign ok = 1; }} "
        "return ok;")


# verdict: validate (every required field present).
def validate_prog(required, present):
    return src(
        f"let req = {lit(required)}; let have = {lit(present)}; create ok: number; assign ok = 1; create c: number; "
        "for (let k of req) { assign c = 0; for (let h of have) { if (h == k) { add c, 1; } } "
        "if (c == 0) { assign ok = 0; } } return ok;")


# write: summarize (form only): non-empty and at most 30% of the source's words; the meaning is the talk seat's.
def summary_prog(n_src, n_reply):
    return src(
        f"create ns: number; assign ns = {n_src}; create nr: number; assign nr = {n_reply}; "
        "create lim: number; assign lim = ns; mul lim, 0.3; create ok: number; assign ok = 1; "
        "if (nr == 0) { assign ok = 0; } if (nr > lim) { assign ok = 0; } return ok;")


# act: the gate for a change outside the reply. 2 = do, 1 = ask the user first, 0 = refuse (no tool for it).
def gate_prog(cls, confirmed, has_tool):
    return src(
        f"let cls = {json.dumps(cls)}; create yes: number; assign yes = {confirmed}; create tool: number; "
        f"assign tool = {has_tool}; create r: number; assign r = 0; "
        "if (tool == 1) { if (cls == \"read\") { assign r = 2; } else { if (cls == \"reversible\") { assign r = 2; } "
        "else { if (yes == 1) { assign r = 2; } else { assign r = 1; } } } } return r;")


# act: the read-back after the host ran it (restart -> the service reports running).
def readback_prog(want, status):
    return src(
        f"let want = {json.dumps(want)}; let got = {json.dumps(status)}; create ok: number; assign ok = 0; "
        "if (got == want) { assign ok = 1; } return ok;")


CASES = [
    ("reorder", "sort", sort_prog([3, 1, 2, 2], [1, 2, 2, 3]), 1),
    ("reorder", "sort", sort_prog([3, 1, 2, 2], [1, 2, 3, 2]), 0),
    ("reorder", "sort", sort_prog([3, 1, 2, 2], [1, 2, 3, 3]), 0),
    ("pick", "filter", filter_prog([5, 12, 7, 20], 10, [12, 20]), 1),
    ("pick", "filter", filter_prog([5, 12, 7, 20], 10, [12]), 0),
    ("pick", "filter", filter_prog([5, 12, 7, 20], 10, [12, 20, 7]), 0),
    ("edit", "rename", rename_prog(["the", "cat", "saw", "a", "cat"], "cat", "dog",
                                   ["the", "dog", "saw", "a", "dog"]), 1),
    ("edit", "rename", rename_prog(["the", "cat", "saw", "a", "cat"], "cat", "dog",
                                   ["the", "dog", "saw", "a", "cat"]), 0),
    ("edit", "rename", rename_prog(["the", "cat", "saw", "a", "cat"], "cat", "dog",
                                   ["the", "dog", "saw", "one", "dog"]), 0),
    ("compute", "calculate", total_prog([4, 9, 2], 15), 1),
    ("compute", "calculate", total_prog([4, 9, 2], 16), 0),
    ("verdict", "validate", validate_prog(["name", "date", "total"], ["date", "name", "total", "note"]), 1),
    ("verdict", "validate", validate_prog(["name", "date", "total"], ["date", "name"]), 0),
    ("write", "summarize", summary_prog(200, 48), 1),
    ("write", "summarize", summary_prog(200, 90), 0),
    ("act", "open (read)", gate_prog("read", 0, 1), 2),
    ("act", "restart (reversible)", gate_prog("reversible", 0, 1), 2),
    ("act", "delete (irreversible), no yes", gate_prog("irreversible", 0, 1), 1),
    ("act", "send (outward), yes this turn", gate_prog("outward", 1, 1), 2),
    ("act", "deploy, no tool", gate_prog("outward", 1, 0), 0),
    ("act", "restart: read-back", readback_prog("running", "running"), 1),
    ("act", "restart: read-back", readback_prog("running", "stopped"), 0),
]


def main():
    vm = cc.CubelangSession()
    rows, bad = [], 0
    for kind, verb, prog, want in CASES:
        try:
            out = vm.run(prog, fn="solve", args=["probe"])
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        got = out.get("result") if isinstance(out, dict) else None
        try:
            ok = float(str(got).strip()) == float(want)
        except (TypeError, ValueError):
            ok = False
        bad += 0 if ok else 1
        err = None if got is not None else str((out or {}).get("error") or out)[:200]
        rows.append({"kind": kind, "verb": verb, "want": want, "got": got, "agree": ok, "error": err})
        print(f"{'ok ' if ok else 'BAD'} {kind:8s} {verb:32s} want {want} got {got!r}" + (f"  {err}" if err else ""))
    print(f"{len(CASES) - bad} of {len(CASES)} agree; vm: {vm_identity.identity()}")
    if len(sys.argv) > 1:
        out = {"order": "0072", "what": "one program per verb kind, a pass and a fail case each, on the VM build",
               "rows": rows, "agree": len(CASES) - bad, "cases": len(CASES), "vm": vm_identity.identity()}
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
