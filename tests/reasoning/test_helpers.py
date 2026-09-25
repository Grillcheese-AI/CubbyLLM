"""CubeLang helpers (2026-09-24): the shapes that recur in the emitter's verified programs, as functions,
adopted only when every program they would rewrite still runs to its own result. A small interpreter
stands in for the VM (CALL binds arg0..; `let x = f(...)` takes the returned value)."""
import re

from cubbyllm.reasoning.helpers import Shape, candidates, gate, instances, parse, rewrite, with_helpers
from cubbyllm.reasoning.skills import Library

HEAD = '''program GSM{n} implements ISolver {{
    type Input = str;
    type Output = quantity;
    public function parse(raw: str): Input {{ return raw; }}
    public pure function verify(input: Input, output: Output): bool {{ return true; }}

    public function solve(input: Input): Output {{
'''
TAIL = '''        sum {last};
        query {last};
        return {last};
    }}
}}
'''


def prog(n, steps):
    """steps: [(name, init, [(op, operand), ...])]"""
    body = []
    for name, init, ops in steps:
        body.append(f"        create {name} : quantity;   # step")
        body.append(f"        assign {name} = {init};")
        body += [f"        {op} {name}, {v};" for op, v in ops]
    return HEAD.format(n=n) + "\n".join(body) + "\n" + TAIL.format(last=steps[-1][0])


def vm(source, buggy=()):
    """Runs the step programs this module writes; `buggy` helper names divide by an integer division."""
    funcs = {m.group(1): m.group(2) for m in re.finditer(r"function (\w+)\([^)]*\)[^{\n]*\{\n(.*?)\n    \}", source, re.S)}

    def val(reg, v):
        v = v.strip()
        return reg[v] if v in reg else float(v)

    def run(name, args):
        reg = {f"arg{i}": a for i, a in enumerate(args)}
        for line in funcs[name].split("\n"):
            s = line.split("#")[0].strip().rstrip(";")
            if m := re.match(r"assign (\w+) = (.+)", s):
                reg[m.group(1)] = val(reg, m.group(2))
            elif m := re.match(r"(add|sub|mul|div) (\w+), (.+)", s):
                a, b = reg[m.group(2)], val(reg, m.group(3))
                op = m.group(1)
                reg[m.group(2)] = (a + b if op == "add" else a - b if op == "sub" else a * b if op == "mul"
                                   else (a // b if name in buggy else a / b))
            elif m := re.match(r"let (\w+) = (\w+)\((.*)\)", s):
                reg[m.group(1)] = run(m.group(2), [val(reg, x) for x in m.group(3).split(",")])
            elif m := re.match(r"return (\w+)", s):
                return reg[m.group(1)]
    return run("solve", [])


P1 = prog(1, [("s0", "48", [("div", "2")]), ("s1", "0", [("add", "s0"), ("mul", "3")])])        # (48/2)*3


def test_two_steps_where_the_first_feeds_only_the_second_are_one_shape():
    (key, inst), = instances(parse(P1))
    assert key == ("div", "mul", True) and inst.inner == "s0" and inst.outer == "s1"
    assert inst.leaves == (("lit", "48"), ("lit", "2"), ("lit", "3"))


def test_a_rewrite_is_one_line_and_runs_to_the_same_result():
    new, used = rewrite(P1, [Shape("div", "mul", True)])
    assert used == ["div_mul"] and "let s1 = div_mul(48, 2, 3);" in new
    assert "assign s0" not in new and "function div_mul(" in new
    assert vm(new) == vm(P1) == 72.0


def test_a_step_read_twice_is_not_folded_away():
    p = prog(2, [("s0", "10", [("mul", "2")]), ("s1", "0", [("add", "s0"), ("add", "1")]),
                 ("s2", "0", [("add", "s0"), ("add", "s1")])])
    assert [k for k, _ in instances(parse(p))] == [] or all(i.inner != "s0" for _k, i in instances(parse(p)))


def test_a_leaf_that_is_mostly_one_number_is_baked_in():
    progs = [{"id": str(i), "program": prog(i, [("s0", str(10 + i), [("mul", str(3 + i))]),
                                                  ("s1", "0", [("add", "s0"), ("div", "100")])])} for i in range(25)]
    shapes = [s for s, _n in candidates(progs, min_support=20)]
    assert shapes[0] == Shape("mul", "div", True, ((2, "100"),)) and shapes[0].name == "mul_div_100"
    new, _ = rewrite(progs[0]["program"], shapes[:1])
    assert "let s1 = mul_div_100(10, 3);" in new and vm(new) == vm(progs[0]["program"])


def test_a_helper_that_changes_one_programs_result_stays_out_and_the_ledger_keeps_it(tmp_path):
    progs = [{"id": str(i), "program": prog(i, [("s0", str(7 + i), [("div", "2")]),
                                                  ("s1", "0", [("add", "s0"), ("mul", "3")])])} for i in range(22)]
    for p in progs:
        p["result"] = vm(p["program"])
    lib = Library(tmp_path / "skills.jsonl")
    rep = gate(progs, [(Shape("div", "mul", True), 22)], lambda s: vm(s, buggy={"div_mul"}), "n1", library=lib,
               min_support=20)
    assert rep["adopted"] == [] and rep["rejected"]["div_mul"]["why"] == "counterexample"   # 7/2 is not 3
    rep = gate(progs, [(Shape("div", "mul", True), 22)], vm, "n1", library=lib, min_support=20)
    assert rep["adopted"] == ["div_mul"] and "div_mul" in Library(tmp_path / "skills.jsonl").helpers


def test_a_program_calling_an_adopted_helper_is_given_its_definition():
    lib = Library()
    lib.adopt_helper(Shape("div", "mul", True), 30, "n1")
    written = P1.replace("        create s0 : quantity;   # step\n        assign s0 = 48;\n        div s0, 2;\n", "")
    written = re.sub(r"        create s1.*?mul s1, 3;", "        let s1 = div_mul(48, 2, 3);", written, flags=re.S)
    assert "function div_mul" not in written
    ran = with_helpers(written, lib)
    assert "function div_mul(" in ran and vm(ran) == 72.0
    assert with_helpers(P1, lib) == P1                          # a program that calls none is left alone
    lib.retire_helper("div_mul", "test", "n2")
    assert with_helpers(written, lib) == written and "div_mul" not in lib.helpers
