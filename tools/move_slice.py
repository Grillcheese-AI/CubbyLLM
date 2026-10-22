"""tools/move_slice.py -- one slice of order 0031's moves: files out of standin/ into their homes, history kept.

For each ``old -> new`` of the slice (``--slice <name>`` from the move map, or pairs given as ``old=new``):

1. ``git mv old new`` (history follows the file);
2. a shim at ``old``: importing the old name gives the new module object itself (``sys.modules``), with a
   ``DeprecationWarning``; running the old file as a script runs the new home (``runpy``);
3. in every moved file: imports of moved modules become package imports (relative inside ``cubbyllm/``, absolute
   elsewhere); each ``__file__`` path expression is rewritten to the same place from the new location (the data stays
   where it was); a statement that only edits ``sys.path`` is dropped. A ``__file__`` use it cannot evaluate is listed,
   for a hand edit.

The moves done so far are kept in ``tools/moves_0031.json`` (old -> new), so later slices rewrite imports of earlier
ones. ``--dry-run`` prints what it would do.

Wired: STANDALONE (a maintenance command for order 0031).
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

__wiring__ = "STANDALONE"

REPO = Path(__file__).resolve().parents[1]
DONE = REPO / "tools" / "moves_0031.json"


def mod_of(path: str) -> str:
    """``cubbyllm/worlds/base.py`` -> ``cubbyllm.worlds.base``; a package ``__init__`` -> the package."""
    p = PurePosixPath(path).with_suffix("")
    parts = list(p.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def flat_names(old: str) -> list[str]:
    """The names an old module was imported by: ``worlds``, ``standin.worlds`` (and ``data.x``, ``standin.data.x``)."""
    p = PurePosixPath(old).with_suffix("")
    parts = list(p.parts)
    names = [parts[-1], ".".join(parts)]
    if len(parts) > 2:                                    # standin/data/x.py, standin/cortex/x.py
        names.append(".".join(parts[1:]))
    return names


# ---- imports ---------------------------------------------------------------------------------------------------
def _target(name: str, moves: dict) -> tuple[str, str] | None:
    """An imported dotted name -> (old path, new module) when it names a moved module."""
    for old, new in moves.items():
        if name in flat_names(old):
            return old, mod_of(new)
    return None


def _rel(from_file: str, to_mod: str) -> tuple[str, str]:
    """For a file inside cubbyllm/: the relative form of ``to_mod`` -> (dots + package path, last name)."""
    here = mod_of(from_file).split(".")[:-1] if not from_file.endswith("__init__.py") else mod_of(from_file).split(".")
    to = to_mod.split(".")
    common = 0
    while common < min(len(here), len(to) - 1) and here[common] == to[common]:
        common += 1
    dots = "." * (len(here) - common + 1)
    return dots + ".".join(to[common:-1]), to[-1]


def rewrite_import(node, src_lines, file_new: str, moves: dict) -> str | None:
    """The new text of an Import / ImportFrom node that names a moved module, else None."""
    in_pkg = file_new.startswith("cubbyllm/")
    out = []
    if isinstance(node, ast.Import):
        changed = False
        keep = []
        for a in node.names:
            t = _target(a.name, moves)
            if t is None:
                keep.append(a.name + (f" as {a.asname}" if a.asname else ""))
                continue
            changed = True
            bind = a.asname or a.name.split(".")[0]
            if "." in a.name and not a.asname:            # `import standin.worlds`: the name bound is `standin`
                return None                               # left for a hand edit (rare)
            if in_pkg:
                pkg, last = _rel(file_new, t[1])
                out.append(f"from {pkg} import {last}" + (f" as {bind}" if bind != last else ""))
            else:
                pkg, last = t[1].rsplit(".", 1)
                out.append(f"from {pkg} import {last}" + (f" as {bind}" if bind != last else ""))
        if not changed:
            return None
        if keep:
            out.insert(0, "import " + ", ".join(keep))
        return out
    if node.level or not node.module:
        return None
    t = _target(node.module, moves)
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
    if t is not None:
        if in_pkg:
            pkg, last = _rel(file_new, t[1])
            mod = pkg + last if set(pkg) == {"."} else pkg + "." + last
            return [f"from {mod} import {names}"]
        return [f"from {t[1]} import {names}"]
    if node.module in ("standin", "standin.data", "standin.cortex", "data", "cortex"):   # from standin import worlds
        parts = []
        for a in node.names:
            t = _target(f"{node.module}.{a.name}", moves) or _target(a.name, moves)
            if t is None:
                return None
            pkg, last = (_rel(file_new, t[1]) if in_pkg else tuple(t[1].rsplit(".", 1)))
            bind = a.asname or a.name
            parts.append(f"from {pkg} import {last}" + (f" as {bind}" if bind != last else ""))
        return parts
    return None


# ---- __file__ ---------------------------------------------------------------------------------------------------
def _chain(node):
    """A path expression rooted at ``__file__`` -> (kind 'path'|'str', ops, constructor text) or None."""
    ops = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute) and cur.attr in ("parent", "parents"):
            ops.append(("parent", 1)) if cur.attr == "parent" else None
            cur = cur.value
        elif isinstance(cur, ast.Subscript) and isinstance(cur.value, ast.Attribute) and cur.value.attr == "parents" \
                and isinstance(cur.slice, ast.Constant):
            ops.append(("parent", cur.slice.value + 1))
            cur = cur.value.value
        elif isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr in ("resolve", "absolute") \
                and not cur.args:
            ops.append(("resolve", 0))
            cur = cur.func.value
        elif isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr == "with_name" \
                and len(cur.args) == 1 and isinstance(cur.args[0], ast.Constant):
            ops.append(("with_name", cur.args[0].value))
            cur = cur.func.value
        elif isinstance(cur, ast.Call) and ast.unparse(cur.func) in ("Path", "pathlib.Path", "_pathlib.Path") \
                and len(cur.args) == 1 and isinstance(cur.args[0], ast.Name) and cur.args[0].id == "__file__":
            return "path", list(reversed(ops)), ast.unparse(cur.func)
        elif isinstance(cur, ast.Call) and ast.unparse(cur.func) in ("os.path.dirname",) and len(cur.args) == 1:
            ops.append(("parent", 1))
            cur = cur.args[0]
        elif isinstance(cur, ast.Call) and ast.unparse(cur.func) in ("os.path.abspath", "os.path.realpath") \
                and len(cur.args) == 1:
            ops.append(("resolve", 0))
            cur = cur.args[0]
        elif isinstance(cur, ast.Name) and cur.id == "__file__":
            return "str", list(reversed(ops)), None
        else:
            return None


def _resolve(old: str, ops) -> tuple[PurePosixPath, bool] | None:
    """Apply the chain to the old path (repo-relative): -> (target, resolved?) ; a target above the repo: None."""
    p = PurePosixPath(old)
    resolved = False
    for op, arg in ops:
        if op == "resolve":
            resolved = True
        elif op == "parent":
            for _ in range(arg):
                if str(p) in (".", ""):
                    return None
                p = p.parent
        elif op == "with_name":
            p = p.with_name(arg)
    return p, resolved


def _expr(new: str, target: PurePosixPath, kind: str, ctor: str | None, resolved: bool) -> str:
    """The same place, written from the new file."""
    up = len(PurePosixPath(new).parent.parts)               # cubbyllm/worlds/x.py -> 2 levels to the repo
    rel = [x for x in target.parts if x not in (".",)]
    if kind == "path":
        base = f"{ctor}(__file__){'.resolve()' if resolved else ''}.parents[{up}]" if up else f"{ctor}(__file__).parent"
        return base + "".join(f' / "{x}"' for x in rel)
    head = "os.path.dirname(os.path.abspath(__file__))"
    parts = ", ".join(['".."'] * up + [f'"{x}"' for x in rel])
    return f"os.path.normpath(os.path.join({head}, {parts}))" if (up or rel) else head


def _is_path_edit(stmt) -> bool:
    """A statement whose every effect is a sys.path edit (an insert, an append, a for/if around them)."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        f = ast.unparse(stmt.value.func)
        return f.endswith(("sys.path.insert", "sys.path.append", "sys.path.extend"))
    if isinstance(stmt, (ast.For, ast.If)):
        body = stmt.body + getattr(stmt, "orelse", [])
        return bool(body) and all(_is_path_edit(s) for s in body)
    if isinstance(stmt, ast.Assign) and any(ast.unparse(t).startswith(("sys.path", "_sys.path")) for t in stmt.targets):
        return True
    return False


def rewrite_file(src: str, old: str, new: str, moves: dict, paths_only: bool = False) -> tuple[str, list[str]]:
    """``paths_only`` (a moved test): only the __file__ paths; its imports and sys.path edits stay (it imports
    through the shims)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits, notes = [], []                                 # (start (line, col), end (line, col), text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not paths_only:
            new_txt = rewrite_import(node, lines, new, moves)
            if new_txt:
                indent = " " * node.col_offset
                edits.append(((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset),
                              ("\n" + indent).join(new_txt)))
    taken = set()
    for node in ast.walk(tree):
        c = _chain(node) if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)) else None
        if c is None or id(node) in taken:
            continue
        kind, ops, ctor = c
        if not any(op in ("parent", "with_name") for op, _ in ops):
            continue                                      # the file itself (its name, its stem): left as it is
        r = _resolve(old, ops)
        if r is None:
            continue
        for sub in ast.walk(node):
            taken.add(id(sub))
        target, resolved = r
        edits.append(((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset),
                      _expr(new, target, kind, ctor, resolved)))
    for stmt in ast.walk(tree):
        body = getattr(stmt, "body", None)
        if not isinstance(body, list):
            continue
        for s in body:
            if _is_path_edit(s) and not paths_only:
                edits.append(((s.lineno, 0), (s.end_lineno, len(lines[s.end_lineno - 1].rstrip("\r\n"))), None))
    # drop edits nested inside a dropped statement; apply the rest bottom up
    drops = sorted({(a, b) for a, b, t in edits if t is None})
    drops = [d for d in drops if not any(o != d and o[0] <= d[0] and d[1] <= o[1] for o in drops)]   # the outermost only
    keep = [e for e in edits if e[2] is not None and not any(d[0] <= e[0] and e[1] <= d[1] for d in drops)]
    keep += [(a, b, None) for a, b in drops]
    for (l0, c0), (l1, c1), txt in sorted(keep, key=lambda e: e[0], reverse=True):
        if txt is None:                                   # the whole statement's lines go
            del lines[l0 - 1:l1]
            continue
        head, tail = lines[l0 - 1][:c0], lines[l1 - 1][c1:]
        lines[l0 - 1:l1] = [head + txt + tail]
    out = "".join(lines)
    ast.parse(out)
    for n in ast.walk(ast.parse(out)):
        if isinstance(n, ast.Name) and n.id == "__file__":
            seg = ast.get_source_segment(out, n) or ""
            del seg
    for i, l in enumerate(out.splitlines(), 1):
        if "__file__" in l and "parents[" not in l and "os.path.dirname(os.path.abspath(__file__))" not in l:
            notes.append(f"{new}:{i}: {l.strip()[:110]}")
    return out, notes


SHIM = '''"""{old} moved to {new} (order 0031). This name works for one cycle: importing it gives the moved module
itself, and running it runs the new home."""
import importlib as _importlib
import os as _os
import sys as _sys
import warnings as _warnings

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), {ups}))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
_NEW = "{mod}"
if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module(_NEW, run_name="__main__", alter_sys=True)
else:
    _warnings.warn(f"{{__name__}} moved to {{_NEW}} (order 0031)", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module(_NEW)
'''


def shim(old: str, new: str) -> str:
    ups = ", ".join(['".."'] * len(PurePosixPath(old).parent.parts))
    return SHIM.format(old=old, new=new, ups=ups, mod=mod_of(new))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", nargs="*", help="old=new")
    ap.add_argument("--tests", nargs="*", default=[], help="test files to move as old=new (no rewrite, no shim)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pairs = dict(p.split("=", 1) for p in a.pairs)
    moves = json.loads(DONE.read_text(encoding="utf-8")) if DONE.exists() else {}
    moves.update(pairs)
    notes = []
    for old, new in pairs.items():
        src = (REPO / old).read_text(encoding="utf-8")
        out, n = rewrite_file(src, old, new, moves)
        notes += n
        if a.dry_run:
            print(f"would move {old} -> {new}")
            continue
        (REPO / new).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=REPO, check=True)
        (REPO / new).write_text(out, encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", new], cwd=REPO, check=True)       # the rename commit carries the rewrite
        (REPO / old).write_text(shim(old, new), encoding="utf-8", newline="\n")   # the shim: staged by the next commit
    for t in a.tests:
        old, new = t.split("=", 1)
        out, n = rewrite_file((REPO / old).read_text(encoding="utf-8"), old, new, moves, paths_only=True)
        notes += n
        if not a.dry_run:
            (REPO / new).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "mv", old, new], cwd=REPO, check=True)
            (REPO / new).write_text(out, encoding="utf-8", newline="\n")
            subprocess.run(["git", "add", new], cwd=REPO, check=True)
    if not a.dry_run:
        DONE.write_text(json.dumps(moves, indent=1) + "\n", encoding="utf-8")
    for n in notes:
        print("CHECK", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
