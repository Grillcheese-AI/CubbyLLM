"""The three homes' rules (order 0031), each pinned by a scan of the source, lazy imports included.

- `cubbyllm/` (Apache-2.0, what makes Cubby think) imports nothing from `cubby_serve/`, `standin/` or `validation/`,
  opens no socket (Nick, 2026-10-22: no `socket`, `urllib.request`, `http.client` or `requests`), and edits no
  `sys.path`.
- `cubby_serve/` (BSL 1.1, the product) reaches the stand-in only through the three lazy imports the review named:
  `emitter_llama` (--gguf), `lfm_source` (--web-reader, --ask lfm), `pacman` (--pacman).
- `standin/` holds only what the move map keeps there (the GGUF seat and its facade, the LFM source, the proposer,
  the stand-in's eval, its probes, the game, three GGUF-only builders) and the one-cycle shims.
- `import cubbyllm` loads neither torch nor grilly (a fresh interpreter).
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NETWORK = ("socket", "urllib.request", "http.client", "requests")
STANDIN_SEAMS = {"emitter_llama", "lfm_source", "pacman"}
SHIM = ("(order 0031). This name works for one cycle", "moved (order 0031)", "moved to cubbyllm/cortex/ (order 0031)")
STAYS = set("""coach emitter emitter_llama eval_emitter_vm forge hypothesis knowledge lfm_source openrouter pac_affect
pac_forge pac_nav pac_sense pac_speech pacman pacpaths pacui_threejs pacworld programs verse world
data/build_game_sft data/build_profile_sft data/gap_families
scripts/forge_probe scripts/probe_v3 scripts/question_style_probe scripts/talk_probe scripts/tool_call_probe""".split())


def _py(folder: str):
    for p in sorted((ROOT / folder).rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def _imports(path: pathlib.Path):
    """(line, dotted name, inside a function) for every absolute import in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lazy = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lazy.update(id(n) for n in ast.walk(node) if n is not node)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name, id(node) in lazy
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module, id(node) in lazy
            for a in node.names:                              # from urllib import request
                yield node.lineno, f"{node.module}.{a.name}", id(node) in lazy


def _flat_names(folder: str) -> set[str]:
    """The names a script on `sys.path` could import a module of `folder` by (its stem; not its tests)."""
    return {p.stem for p in _py(folder) if "tests" not in p.relative_to(ROOT).parts and p.stem != "__init__"}


def test_cubbyllm_imports_nothing_from_the_other_homes():
    outside = {"cubby_serve", "standin", "validation"} | _flat_names("standin") | _flat_names("validation")
    bad = [f"{p.relative_to(ROOT)}:{line}: {name}"             # a flat name is a crossing: `import ask` is the shim
           for p in _py("cubbyllm") for line, name, _lazy in _imports(p) if name.split(".")[0] in outside]
    assert not bad, f"cubbyllm/ reaches outside the package: {bad}"


def test_cubbyllm_opens_no_socket():
    bad = [f"{p.relative_to(ROOT)}:{line}: {name}" for p in _py("cubbyllm") for line, name, _lazy in _imports(p)
           if any(name == n or name.startswith(n + ".") for n in NETWORK)]
    assert not bad, f"cubbyllm/ imports the network (hand a fetch in from cubby_serve/sources/): {bad}"


def test_cubbyllm_edits_no_sys_path():
    bad = []
    for p in _py("cubbyllm"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(("sys.path.insert", "sys.path.append",
                                                                               "sys.path.extend")):
                bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(ast.unparse(t).lstrip("_").startswith("sys.path") for t in targets):
                    bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not bad, f"cubbyllm/ edits sys.path: {bad}"


def test_cubby_serve_reaches_the_stand_in_only_through_the_named_lazy_imports():
    stand = _flat_names("standin")                             # the shims' names too: `import ask` is a crossing
    seen, bad = set(), []
    for p in _py("cubby_serve"):
        for line, name, lazy in _imports(p):
            top = name.split(".")
            mod = top[1] if top[0] == "standin" and len(top) > 1 else top[0]
            if top[0] != "standin" and mod not in stand:
                continue
            seen.add(mod)
            if mod not in STANDIN_SEAMS or not lazy:
                bad.append(f"{p.relative_to(ROOT)}:{line}: {name}" + ("" if lazy else " (at module level)"))
    assert not bad, f"cubby_serve/ imports the stand-in beyond {sorted(STANDIN_SEAMS)}, or eagerly: {bad}"
    assert seen <= STANDIN_SEAMS


def test_standin_holds_only_what_stays_and_the_shims():
    other = []
    for p in _py("standin"):
        rel = p.relative_to(ROOT / "standin").with_suffix("").as_posix()
        if rel.startswith(("tests/", "data/out/", "models/")) or rel in STAYS:
            continue
        if not any(m in p.read_text(encoding="utf-8", errors="replace") for m in SHIM):
            other.append(rel)
    assert not other, f"standin/ holds modules the move map does not keep there: {other}"


def test_import_cubbyllm_loads_neither_torch_nor_grilly():
    code = "import sys, cubbyllm; print(sorted(m for m in ('torch', 'grilly') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"import cubbyllm pulled in {out.stdout.strip()}"
