"""tools/split_module.py -- one module split in place, behaviour unchanged (order 0031: a file over 300 lines is split
after its move).

A plan (JSON) names, per new module, the top-level names it takes and/or the methods (and ``attrs``, class-level
constants, which the mixin then holds as its own) of one class it takes as a mixin:

    {"source": "cubbyllm/reasoning/ask.py",
     "modules": {"cubbyllm/reasoning/ask_forms.py": {"doc": "...", "names": ["TAU_VM", "profile_ask"]},
                 "cubbyllm/reasoning/ask_memory.py": {"doc": "...", "mixin": {"class": "AskLoop", "name": "AskMemory",
                                                                               "methods": ["load_held"]}}}}

The tool cuts each piece verbatim (with the comment lines right above it), writes each new module with the imports its
code needs (the source's own import lines it uses, and what it uses from the other new modules), adds each mixin to the
class's bases, and makes the source re-export every moved top-level name, so importers do not change. A new module
never imports the source (no import cycle): a name it needs that stays behind is listed and nothing is written, for the
plan to move it too. A method that uses a double-underscore attribute (its name mangling would change) is refused.

    python tools/split_module.py <plan.json> [--dry-run]

Wired: STANDALONE (a maintenance command for order 0031).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from pathlib import Path, PurePosixPath

__wiring__ = "STANDALONE"

REPO = Path(__file__).resolve().parents[1]


def _span(lines: list[str], node) -> tuple[int, int]:
    """(first, last) 1-based lines of a statement with its decorators and the comment lines right above it."""
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    while first > 1 and lines[first - 2].strip().startswith("#"):
        first -= 1
    return first, node.end_lineno


def _bound(node) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _import_names(node) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names if a.name != "*"]
    return []


def _used(nodes) -> set[str]:
    out = set()
    for n in nodes:
        for x in ast.walk(n):
            if isinstance(x, ast.Name):
                out.add(x.id)
    return out


def _mod(path: str) -> str:
    return PurePosixPath(path).stem


def wrap(mod: str, names, noqa: str = "", width: int = 118) -> str:
    """`from <mod> import a, b` on one line when it fits under `width`, else parenthesized and wrapped; a `# noqa` (if
    any) on the first line."""
    names = sorted(set(names))
    tag = f"  # noqa: {noqa}" if noqa else ""
    one = f"from {mod} import {', '.join(names)}{tag}"
    if len(one) <= width:
        return one + "\n"
    head = f"from {mod} import ("
    lines, cur = [], head
    for k, n in enumerate(names):
        piece = n + ("," if k < len(names) - 1 else ")")
        limit = width - (len(tag) if not lines else 0)
        sep = "" if cur.endswith("(") or not cur.strip() else " "
        if len(cur) + len(sep) + len(piece) > limit:
            lines.append(cur.rstrip())
            cur, sep = " " * len(head), ""
        cur += sep + piece
    lines.append(cur)
    lines[0] += tag
    return "\n".join(lines) + "\n"


def split(plan: dict, dry: bool = False) -> dict:
    src_path = REPO / plan["source"]
    src = src_path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    top = {}                                                  # name -> top-level node defining it
    imports = {}                                              # bound name -> the import statement's text
    future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body)
    for n in tree.body:
        for b in _bound(n):
            top[b] = n
        if isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
            text = "".join(lines[n.lineno - 1:n.end_lineno]).rstrip("\n")
            for b in _import_names(n):
                imports[b] = (n.lineno, text)                 # copied in the source's order
    owner, cut, pieces, mixins, problems = {}, [], {}, [], []
    for new, spec in plan["modules"].items():
        pieces[new] = []
        for name in spec.get("names", []):
            if name not in top:
                problems.append(f"{new}: {name} is not a top-level name of {plan['source']}")
                continue
            node = top[name]
            if id(node) in {id(c[2]) for c in cut}:
                continue
            for b in _bound(node):
                owner[b] = new
            f, l = _span(lines, node)
            cut.append((f, l, node))
            pieces[new].append((f, l, node, None))
        mx = spec.get("mixin")
        if mx:
            cls = top.get(mx["class"])
            meths = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for m in mx["methods"]:
                node = meths.get(m)
                if node is None:
                    problems.append(f"{new}: {mx['class']}.{m} is not a method")
                    continue
                if any(isinstance(x, ast.Attribute) and x.attr.startswith("__") and not x.attr.endswith("__")
                       for x in ast.walk(node)):
                    problems.append(f"{new}: {mx['class']}.{m} uses a double-underscore attribute")
                f, l = _span(lines, node)
                cut.append((f, l, node))
                pieces[new].append((f, l, node, mx["name"]))
            for a_name in mx.get("attrs", []):                # class-level constants go with it, as the mixin's own
                node = next((m for m in cls.body if isinstance(m, (ast.Assign, ast.AnnAssign)) and a_name in _bound(m)), None)
                if node is None:
                    problems.append(f"{new}: {mx['class']}.{a_name} is not a class attribute")
                    continue
                if id(node) in {id(c[2]) for c in cut}:
                    continue
                f, l = _span(lines, node)
                cut.append((f, l, node))
                pieces[new].append((f, l, node, mx["name"]))
            mixins.append((mx["class"], mx["name"], new))
    builtin = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    out_files, deps = {}, {}
    for new, items in pieces.items():
        nodes = [p[2] for p in items]
        need = _used(nodes) - builtin
        own = {b for p in items if p[3] is None for b in _bound(p[2])}    # what this module itself defines
        imp_lines, from_new = [], {}
        for name in sorted(need):
            if name in own or (name in owner and owner[name] == new):
                continue
            if name in owner:
                from_new.setdefault(owner[name], []).append(name)
            elif name in imports:
                if imports[name] not in imp_lines:
                    imp_lines.append(imports[name])
            elif name in top:
                problems.append(f"{new}: uses {name}, which stays in {plan['source']} (move it too)")
        body = []
        mixin_name = next((p[3] for p in items if p[3]), None)
        for f, l, node, mx in sorted(items, key=lambda p: p[0]):
            if mx is None:
                body.append("".join(lines[f - 1:l]).rstrip("\n") + "\n")
        if mixin_name:
            meth = ["".join(lines[f - 1:l]).rstrip("\n") + "\n" for f, l, node, mx in sorted(items, key=lambda p: p[0]) if mx]
            cls_doc = plan["modules"][new]["mixin"].get("doc", "")
            body.append(f"class {mixin_name}:\n    \"\"\"{cls_doc}\"\"\"\n\n" + "\n".join(meth))
        head = '"""' + plan["modules"][new]["doc"].rstrip() + '\n"""\n'
        head += "from __future__ import annotations\n\n" if future else ""
        head += "\n".join(t for _l, t in sorted(imp_lines)) + ("\n" if imp_lines else "")
        for m, names in sorted(from_new.items()):
            head += wrap(f".{_mod(m)}", names)
        head += f"\n__wiring__ = \"{plan['modules'][new].get('wiring', 'WIRED')}\"\n\n\n"
        out_files[new] = head + "\n\n".join(body)
        deps[new] = set(from_new)
    # the new modules import each other at their top: a cycle would fail at import time
    for a in deps:
        seen, stack = set(), list(deps[a])
        while stack:
            b = stack.pop()
            if b == a:
                problems.append(f"{a}: an import cycle among the new modules (through {sorted(deps[a])})")
                break
            if b not in seen:
                seen.add(b)
                stack.extend(deps.get(b, ()))
    # the source: the pieces cut, the re-exports and the mixins in
    keep = list(lines)
    for f, l, _node in sorted(cut, key=lambda c: c[0], reverse=True):
        del keep[f - 1:l]
    rest = "".join(keep)
    reexp = "# split out in order 0031 (tools/split_module.py); re-exported so importers do not change\n"
    for new, items in pieces.items():
        names = sorted({b for p in items if p[3] is None for b in _bound(p[2])})
        if names:
            reexp += wrap(f".{_mod(new)}", names, "F401")
    for cls, mname, new in mixins:
        reexp += f"from .{_mod(new)} import {mname}\n"
    rtree = ast.parse(rest)
    first_def = next((n for n in rtree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Expr))), None)
    at = (first_def.lineno - 1) if first_def is not None else len(rest.splitlines())
    rl = rest.splitlines(keepends=True)
    rest = "".join(rl[:at]) + reexp + "\n" + "".join(rl[at:])
    for cls, mname, new in mixins:
        rtree = ast.parse(rest)
        node = next(n for n in rtree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        rl = rest.splitlines(keepends=True)
        line = rl[node.lineno - 1]
        if node.bases:
            line = line.replace("(", f"({mname}, ", 1)
        else:
            line = line.replace(f"class {cls}:", f"class {cls}({mname}):", 1)
        rl[node.lineno - 1] = line
        rest = "".join(rl)
    import re as _re
    rest = _re.sub(r"\n{4,}", "\n\n\n", rest)
    # a re-export block's mixin bases combine: class X(A): + B -> class X(B, A)
    out_files[plan["source"]] = rest
    for path, text in out_files.items():
        try:
            ast.parse(text)
        except SyntaxError as e:
            problems.append(f"{path}: {e}")
    report = {p: t.count("\n") for p, t in out_files.items()}
    if problems:
        return {"problems": problems, "lines": report}
    if not dry:
        for path, text in out_files.items():
            (REPO / path).write_text(text, encoding="utf-8", newline="\n")
    return {"problems": [], "lines": report}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = split(json.loads(Path(a.plan).read_text(encoding="utf-8")), a.dry_run)
    print(json.dumps(r, indent=1))
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
