"""The three anti-sprawl CI guards — the sibling repos' failure patterns as
executable checks. These must stay green for the life of the package.
"""
import ast
import importlib
import pathlib
import pkgutil

import cubbyllm

ROOT = pathlib.Path(cubbyllm.__file__).parent


def _modules():
    for m in pkgutil.walk_packages([str(ROOT)], prefix="cubbyllm."):
        # Generated protobuf stubs (`*_pb2.py`) are protoc output, fully
        # overwritten on every regen — a hand-appended `__wiring__` tail
        # would just get silently dropped next regen. They're pure
        # generated data schemas, never a forward path, so the wiring
        # guard doesn't apply to them; skip by module-name suffix rather
        # than requiring each one to carry a manual tail.
        if m.name.endswith("_pb2"):
            continue
        yield m.name


def test_no_model_naming_collision():
    """No `model.py` file and no `models/` directory anywhere (the sibling
    `model.py`/`models/`/`model` collision)."""
    bad = [
        p for p in ROOT.rglob("*")
        if p.name == "model.py" or (p.is_dir() and p.name == "models")
    ]
    assert not bad, f"forbidden model.py/models present: {bad}"


def test_every_module_declares_wiring():
    """Every module declares its Wired/Standalone intent (the 'built but never
    wired in' trap)."""
    missing = []
    for name in _modules():
        mod = importlib.import_module(name)
        if not hasattr(mod, "__wiring__"):
            missing.append(name)
    assert not missing, f"modules missing __wiring__: {missing}"


def test_ops_is_sole_grilly_importer():
    """Only `cubbyllm/ops/**` may import grilly (single-seam discipline)."""
    offenders = []
    for py in ROOT.rglob("*.py"):
        if "ops" in py.relative_to(ROOT).parts:
            continue
        src = py.read_text(encoding="utf-8")
        if "grilly" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.startswith("grilly") for n in names):
                offenders.append(str(py.relative_to(ROOT)))
                break
    assert not offenders, f"non-ops modules importing grilly: {offenders}"
