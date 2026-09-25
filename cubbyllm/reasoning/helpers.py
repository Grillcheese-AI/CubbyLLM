"""helpers -- the skill library's second kind: CubeLang sub-programs mined from verified emitter programs.

Wired: WIRED -- `standin/eval_emitter_vm.run_vm` injects every adopted helper a program calls
(`with_helpers`), so a program that calls one runs like any other: the emitter evals and the ES loop go
through it. Until the emitter is taught to call helpers nothing it writes does, and nothing changes.

What a helper is. The emitter's arithmetic programs are straight lines of binary steps:

    create s0 : quantity; assign s0 = 48; div s0, 2;                s0 = 48 / 2
    create s1 : quantity; assign s1 = 0; add s1, s0; mul s1, 3;     s1 = s0 * 3

Two steps where the first feeds only the second are one expression, (48 / 2) * 3. The same SHAPE -- divide,
then multiply -- recurs across thousands of verified programs with different numbers. A helper is that shape
as a CubeLang function of its leaves (CALL binds `arg0..`, never the declared names; docs/REFERENCE.md 3.3),
and the two steps become one line, ``let s1 = div_mul(48, 2, 3);``. A leaf that is the same number in most of
a shape's programs is baked in: ``mul_div_100(a, b)`` is (a * b) / 100.

The gate is the skill library's rule, every past episode re-verifies: a candidate is adopted only if EVERY
verified program on record that contains its shape, rewritten to call it, runs in the VM to exactly the
result it ran to before. One program that moves keeps the helper out. Then the whole record is rewritten
with every adopted helper at once and re-run, so the helpers are checked together too; a helper a joint
rewrite breaks is retired, never deleted.

What it buys, and when. Shorter programs -- fewer statements for the emitter to get wrong, each a verified
unit. Only after an adapter round teaches the emitter to call them: the rewritten, re-verified record
(`rewrite`) is that round's data. Advantage over a model that learns the idiom in its weights: the idiom is a
named, VM-verified function on a ledger, checked against every program it would have changed, and retired by
one program it breaks.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

OPS = ("add", "sub", "mul", "div")
_CREATE = re.compile(r"^\s*create\s+(s\d+)\s*:\s*(\w+)\s*;")
_ASSIGN = re.compile(r"^\s*assign\s+(s\d+)\s*=\s*([^;#]+?)\s*;")
_OP = re.compile(r"^\s*(add|sub|mul|div)\s+(s\d+)\s*,\s*([^;#]+?)\s*;")
_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_REF = re.compile(r"^s\d+$")


# ── reading a program ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    name: str
    lines: list[int]              # every line of the program that belongs to the step
    expr: tuple                   # ("lit", "48") | ("ref", "s0") | (op, left, right)


def _leaf(v: str):
    v = v.strip()
    if _REF.match(v):
        return ("ref", v)
    if _NUM.match(v):
        return ("lit", v)
    return None


def parse(program: str) -> tuple[list[str], dict[str, Step], list[str]] | None:
    """(lines, steps by name in order, the step names in order), or None when a step does something this
    reader does not model -- such a program is left exactly as it is."""
    lines = program.split("\n")
    if "function solve" not in program:
        return None
    start = next(i for i, l in enumerate(lines) if "function solve" in l)
    steps: dict[str, Step] = {}
    order: list[str] = []
    for i in range(start + 1, len(lines)):
        l = lines[i]
        m = _CREATE.match(l)
        if m:
            steps[m.group(1)] = Step(m.group(1), [i], None)
            order.append(m.group(1))
            continue
        m = _ASSIGN.match(l)
        if m and m.group(1) in steps:
            leaf = _leaf(m.group(2))
            if leaf is None or steps[m.group(1)].expr is not None:
                return None
            steps[m.group(1)].lines.append(i)
            steps[m.group(1)].expr = leaf
            continue
        m = _OP.match(l)
        if m and m.group(2) in steps:
            s, leaf = steps[m.group(2)], _leaf(m.group(3))
            if leaf is None or s.expr is None:
                return None
            s.lines.append(i)
            # 0 + x is x, exactly: the emitter's "assign s = 0; add s, sJ" is a copy
            s.expr = leaf if (s.expr == ("lit", "0") and m.group(1) == "add") else (m.group(1), s.expr, leaf)
            continue
        if re.match(r"^\s*(add|sub|mul|div|assign)\s+s\d+", l):
            return None                                   # a step touched outside the shapes read here
    if not order or any(steps[n].expr is None for n in order):
        return None
    return lines, steps, order


def _uses(steps: dict[str, Step], lines: list[str], order: list[str]) -> Counter:
    """How often each step is READ: by other steps' expressions and by the program's other lines."""
    n = Counter()

    def walk(e):
        if e[0] == "ref":
            n[e[1]] += 1
        elif e[0] in OPS:
            walk(e[1]); walk(e[2])
    for s in steps.values():
        walk(s.expr)
    owned = {i for s in steps.values() for i in s.lines}
    for i, l in enumerate(lines):
        if i not in owned:
            for name in re.findall(r"\bs\d+\b", l.split("#")[0]):
                if name in steps:
                    n[name] += 1
    return n


# ── shapes ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Shape:
    """Two binary steps as one: inner (op1 of two leaves) feeding the outer op2 on the left or the right,
    with optional leaves baked in as constants: {leaf position: literal}."""
    op1: str
    op2: str
    inner_left: bool
    consts: tuple = ()            # ((position, literal), ...)

    @property
    def name(self) -> str:
        base = f"{self.op1}_{self.op2}" if self.inner_left else f"{self.op2}_of_{self.op1}"
        return base + "".join(f"_{v.replace('.', 'p').replace('-', 'm')}" for _p, v in self.consts)

    @property
    def params(self) -> int:
        return 3 - len(self.consts)

    def source(self) -> str:
        """The helper as a CubeLang function of its free leaves (arg0.. in leaf order)."""
        fixed = dict(self.consts)
        args, k = [], 0
        for pos in range(3):
            if pos in fixed:
                args.append(fixed[pos])
            else:
                args.append(f"arg{k}")
                k += 1
        sig = ", ".join(f"p{i}: quantity" for i in range(self.params))
        if self.inner_left:                                  # (a op1 b) op2 c
            body = [f"create t : quantity;", f"assign t = {args[0]};", f"{self.op1} t, {args[1]};",
                    f"{self.op2} t, {args[2]};", "return t;"]
        else:                                                # a op2 (b op1 c)
            body = [f"create u : quantity;", f"assign u = {args[1]};", f"{self.op1} u, {args[2]};",
                    f"create t : quantity;", f"assign t = {args[0]};", f"{self.op2} t, u;", "return t;"]
        inner = "\n".join(f"        {b}" for b in body)
        return f"    function {self.name}({sig}): quantity {{\n{inner}\n    }}\n"


@dataclass
class Instance:
    inner: str
    outer: str
    leaves: tuple                 # three leaf expressions, in the shape's order


def instances(parsed) -> list[tuple[tuple, Instance]]:
    """Every (op1, op2, inner_left) instance in a program: an inner single-op step read exactly once, by an
    outer single-op step. The final (returned) step is never an inner one."""
    lines, steps, order = parsed
    uses = _uses(steps, lines, order)
    out = []
    for name in order:
        e = steps[name].expr
        if e[0] not in OPS:
            continue
        op2, left, right = e
        for side, child, other in ((True, left, right), (False, right, left)):
            if child[0] != "ref" or other[0] not in ("lit", "ref"):
                continue
            inner = steps.get(child[1])
            if inner is None or uses[child[1]] != 1 or inner.expr[0] not in OPS:
                continue
            op1, a, b = inner.expr
            if a[0] not in ("lit", "ref") or b[0] not in ("lit", "ref"):
                continue
            leaves = (a, b, other) if side else (other, a, b)
            out.append(((op1, op2, side), Instance(child[1], name, leaves)))
    return out


def fits(shape: Shape, inst: Instance) -> bool:
    return all(inst.leaves[p] == ("lit", v) for p, v in shape.consts)


# ── rewriting ──────────────────────────────────────────────────────────────────────────────────────────
def rewrite(program: str, shapes: list[Shape]) -> tuple[str, list[str]]:
    """The program with every instance of `shapes` replaced by one helper call (the first shape that fits
    wins; instances do not overlap) and the helpers it now calls defined before `solve`. -> (program,
    helper names used). A program this reader does not model comes back unchanged."""
    parsed = parse(program)
    if parsed is None:
        return program, []
    lines, steps, order = parsed
    by_key = defaultdict(list)
    for s in shapes:
        by_key[(s.op1, s.op2, s.inner_left)].append(s)
    taken: set[str] = set()
    edits: dict[int, str | None] = {}
    used: list[Shape] = []
    for key, inst in instances(parsed):
        if inst.inner in taken or inst.outer in taken:
            continue
        shape = next((s for s in by_key.get(key, []) if fits(s, inst)), None)
        if shape is None:
            continue
        taken.update((inst.inner, inst.outer))
        free = [v for p, (_k, v) in enumerate(inst.leaves) if p not in dict(shape.consts)]
        indent = re.match(r"^(\s*)", lines[steps[inst.outer].lines[0]]).group(1)
        for i in steps[inst.inner].lines + steps[inst.outer].lines:
            edits[i] = None
        edits[steps[inst.outer].lines[0]] = f"{indent}let {inst.outer} = {shape.name}({', '.join(free)});"
        if shape not in used:
            used.append(shape)
    if not used:
        return program, []
    out = [edits.get(i, l) for i, l in enumerate(lines)]
    out = [l for l in out if l is not None]
    return define(("\n".join(out)), used), [s.name for s in used]


def define(program: str, shapes: list[Shape]) -> str:
    """The helpers' definitions inserted before `solve`, once each (a helper already defined is skipped)."""
    lines = program.split("\n")
    at = next((i for i, l in enumerate(lines) if "function solve" in l), None)
    if at is None:
        return program
    new = [s.source() for s in shapes if not re.search(rf"function\s+{s.name}\s*\(", program)]
    return "\n".join(lines[:at] + [n.rstrip("\n") for n in new] + lines[at:])


def with_helpers(program: str, library) -> str:
    """Define every adopted helper the program calls and does not define itself: how a program written with
    helper calls runs. A program that calls none comes back unchanged."""
    helpers = getattr(library, "helpers", {}) or {}
    calls = [n for n in helpers if re.search(rf"\b{re.escape(n)}\s*\(", program)
             and not re.search(rf"function\s+{re.escape(n)}\s*\(", program)]
    if not calls:
        return program
    lines = program.split("\n")
    at = next((i for i, l in enumerate(lines) if "function solve" in l), None)
    if at is None:
        return program
    return "\n".join(lines[:at] + [helpers[n]["source"].rstrip("\n") for n in calls] + lines[at:])


# ── mining and the gate ────────────────────────────────────────────────────────────────────────────────
def candidates(programs: list[dict], min_support: int = 20, const_share: float = 0.6) -> list[tuple[Shape, int]]:
    """Shapes with at least `min_support` programs, most supported first; a leaf that is the same literal in
    at least `const_share` of a shape's programs also gives the shape with that constant baked in."""
    by_key: dict[tuple, set] = defaultdict(set)
    leaf_vals: dict[tuple, list[Counter]] = defaultdict(lambda: [Counter(), Counter(), Counter()])
    for p in programs:
        parsed = parse(p["program"])
        if parsed is None:
            continue
        seen = set()
        for key, inst in instances(parsed):
            by_key[key].add(p["id"])
            if key in seen:
                continue
            seen.add(key)
            for pos, (kind, v) in enumerate(inst.leaves):
                if kind == "lit":
                    leaf_vals[key][pos][v] += 1
    out = []
    for key, ids in by_key.items():
        if len(ids) < min_support:
            continue
        op1, op2, left = key
        out.append((Shape(op1, op2, left), len(ids)))
        for pos in range(3):
            if not leaf_vals[key][pos]:
                continue
            v, n = leaf_vals[key][pos].most_common(1)[0]
            if n >= min_support and n >= const_share * len(ids):
                out.append((Shape(op1, op2, left, ((pos, v),)), n))
    # the constant forms first: they are the more specific, and a rewrite prefers them
    return sorted(out, key=lambda sn: (-len(sn[0].consts), -sn[1], sn[0].name))


def gate(programs: list[dict], shapes: list[tuple[Shape, int]], run, night: str, library=None,
         min_support: int = 20) -> dict:
    """Adopt every shape under which EVERY program on record that contains it still runs to its own result;
    then rewrite the record with all of them at once and re-run it, retiring a helper the joint rewrite
    breaks. `run(source) -> result` is the VM. `programs`: {id, program, result}. -> the report."""
    adopted: list[Shape] = []
    rejected: dict[str, dict] = {}
    base = {p["id"]: p["result"] for p in programs}
    for shape, support in shapes:
        touched = 0
        bad = None
        for p in programs:
            new, used = rewrite(p["program"], [shape])
            if not used:
                continue
            touched += 1
            got = run(new)
            if str(got) != str(base[p["id"]]):
                bad = {"program": p["id"], "was": base[p["id"]], "now": got}
                break
        if bad is not None:
            rejected[shape.name] = {"why": "counterexample", **bad}
        elif touched < min_support:
            rejected[shape.name] = {"why": "support", "programs": touched}
        else:
            adopted.append(shape)
            if library is not None:
                library.adopt_helper(shape, touched, night)
    # jointly: every program with every adopted helper, re-run; a helper that breaks one is retired
    retired = []
    while adopted:
        broke = None
        for p in programs:
            new, used = rewrite(p["program"], adopted)
            if used and str(run(new)) != str(base[p["id"]]):
                broke = (p["id"], used)
                break
        if broke is None:
            break
        worst = next(s for s in reversed(adopted) if s.name in broke[1])
        adopted.remove(worst)
        retired.append({"helper": worst.name, "program": broke[0]})
        if library is not None:
            library.retire_helper(worst.name, f"program {broke[0]} changed its result under the joint rewrite", night)
    return {"adopted": [s.name for s in adopted], "shapes": adopted, "rejected": rejected, "retired": retired}
