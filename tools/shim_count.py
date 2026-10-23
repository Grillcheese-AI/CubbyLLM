"""tools/shim_count.py -- in-repo imports of the names order 0031 left behind as shims (order 0033; Cowork's answer 7
in 0031's review: the shims go when this reads 0 and the Colab notebooks are updated; the target is 0032's report).

A shim is a module under ``standin/`` that says it "moved to" a new home and hands that out (``_NEW = "..."``). Its
old names are the flat one (``hdc_store``: ``standin/`` or ``standin/data/`` on ``sys.path``) and the dotted one
(``standin.hdc_store``); a shimmed package's names include its submodules (``cortex.vision``). An import counts when
it names one of them: absolute, anywhere in a ``.py`` file (a function's lazy import too), or in a notebook's code
cell. A flat name that a module of another folder also owns (``validation/x.py`` beside ``standin/x.py``) is left out,
since which one an import reaches depends on ``sys.path``.

    python tools/shim_count.py [--by file|name] [--json]

Wired: STANDALONE (a maintenance command; ``tests/test_shim_count.py`` reads it).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

__wiring__ = "STANDALONE"

REPO = Path(__file__).resolve().parents[1]
SKIP = (".venv", "_discard", "node_modules", "docs/archive", ".git", "__pycache__", ".claude")
_NEW = re.compile(r"""^_NEW\s*=\s*["']([\w.]+)["']""", re.M)
FLAT_ROOTS = ("standin", "standin/data")                 # the folders scripts put on sys.path to import a flat name


def _skipped(p: Path, root: Path = REPO) -> bool:
    rel = p.relative_to(root).as_posix()
    return any(rel.startswith(s) or f"/{s}" in f"/{rel}" for s in SKIP)


def shims(root: Path = REPO) -> dict[str, str]:
    """{shim file (repo-relative): its new home}, for every module under standin/ that says it moved."""
    out = {}
    for p in sorted((root / "standin").rglob("*.py")):
        if _skipped(p, root):
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        m = _NEW.search(src)
        if m and "moved to" in src[:600]:
            out[p.relative_to(root).as_posix()] = m.group(1)
    return out


def old_names(root: Path = REPO) -> set[str]:
    """Every name an import of a shim could use: dotted from the repo root, flat from a folder scripts put on sys.path."""
    names = set()
    for f in shims(root):
        mod = f[:-3].replace("/", ".")
        mod = mod[:-9] if mod.endswith(".__init__") else mod
        names.add(mod)
        for base in FLAT_ROOTS:
            prefix = base.replace("/", ".") + "."
            if mod.startswith(prefix):
                flat = mod[len(prefix):]
                if "." not in flat.split(".", 1)[0] and flat:
                    names.add(flat)
    owned = {p.stem for d in ("validation", "tests", "tools", "plugins") for p in (root / d).rglob("*.py") if not _skipped(p, root)}
    return {n for n in names if n.startswith("standin.") or n.split(".")[0] not in owned}


def _imports(src: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
    return out


def _cells(p: Path) -> list[str]:
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return ["".join(l for l in (c.get("source") or []) if not l.lstrip().startswith(("!", "%")))
            for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def count(root: Path = REPO) -> dict:
    """{"total", "notebooks", "by_name", "by_file", "hits"}: the in-repo imports of shimmed names."""
    names, own = old_names(root), set(shims(root))
    hits = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if _skipped(p, root) or rel in own:
            continue
        for line, mod in _imports(p.read_text(encoding="utf-8", errors="replace")):
            if mod in names or any(mod.startswith(n + ".") for n in names if "." not in n):
                hits.append({"file": rel, "line": line, "name": mod, "notebook": False})
    for p in sorted(root.rglob("*.ipynb")):
        if _skipped(p, root):
            continue
        for i, src in enumerate(_cells(p)):
            for _line, mod in _imports(src):
                if mod in names or any(mod.startswith(n + ".") for n in names if "." not in n):
                    hits.append({"file": p.relative_to(root).as_posix(), "line": f"cell {i}", "name": mod, "notebook": True})
    code = [h for h in hits if not h["notebook"]]
    return {"total": len(code), "notebooks": len(hits) - len(code), "shims": len(own),
            "by_name": dict(Counter(h["name"].split(".")[0] if not h["name"].startswith("standin.") else h["name"]
                                    for h in code).most_common()),
            "by_file": dict(Counter(h["file"] for h in code).most_common()), "hits": hits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--by", choices=("file", "name"), default="name")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    c = count()
    if a.json:
        print(json.dumps(c, indent=1))
        return 0
    print(f"{c['total']} in-repo imports of shimmed names in code, {c['notebooks']} in notebooks ({c['shims']} shims)")
    for k, n in c[f"by_{a.by}"].items():
        print(f"  {n:4d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
