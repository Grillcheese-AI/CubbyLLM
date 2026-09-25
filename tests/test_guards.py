"""The anti-sprawl CI guards — the sibling repos' failure patterns as executable
checks. These must stay green for the life of the package.
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


# ---------------------------------------------------------------------------
# 2026-09-15. `test_every_module_declares_wiring` above is named after "the
# 'built but never wired in' trap" and does not catch it: it asserts the MARKER
# EXISTS, never that `Wiring.WIRED` is true. `Wiring`'s own docstring is explicit
# that the marker is intent and that a component is done only when "its intent is
# WIRED, its body is real, assembly.py actually calls it, AND its test is
# un-skipped and green". Of those four clauses the suite enforced one.
#
# An AST sweep found three modules declaring WIRED with no production importer at
# all — hippocampus, striatum and retriever — each measured in its own experiment
# and never mounted on a forward path. That is not a design error; the intent was
# honestly declared. The error is that nothing measured the gap between intent
# and reality, so it accumulated silently to three.
#
# This guard closes the "actually calls it" clause, and it is deliberately
# BIDIRECTIONAL: a newly-unwired module fails, and a listed module that becomes
# wired ALSO fails, so the ledger cannot go stale in either direction.

#: module -> why it is not yet on a forward path. Retire an entry by WIRING the
#: module, never by adding to this list to silence the guard.
# 2026-09-24: hippocampus and striatum left the list -- the sleep cycle (reasoning/sleep.py) writes the
# day's certified chains into the one and delivers the day's outcomes to the other every night. The ask
# loop does not yet READ either at question time; that is the next wiring step, not a reason to list them.
KNOWN_UNWIRED = {
    "cubbyllm.reasoning.retriever":
        "the scorer seam exists in pipeline._walk; nothing builds one from this module",
}


def _production_importers() -> dict[str, set[str]]:
    """module fullname -> the set of PRODUCTION modules importing it.

    Production means: not under a tests/ directory, and not a test_/exp_ file.
    A re-export in a package __init__ counts — that is a real forward edge.
    """
    repo = ROOT.parent
    out: dict[str, set[str]] = {}
    for py in repo.rglob("*.py"):
        parts = py.relative_to(repo).parts
        if "__pycache__" in parts or any(p in (".venv-dml", "tests") for p in parts):
            continue
        if py.name.startswith(("test_", "exp_")):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = py.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            tails: list[str] = []
            if isinstance(node, ast.Import):
                tails += [a.name.rsplit(".", 1)[-1] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # `from .pipeline import answer` -> the edge is to `pipeline`.
                if node.module:
                    tails.append(node.module.rsplit(".", 1)[-1])
                # `from . import simlog` has module=None and carries the target in
                # names -- the first cut of this guard read only `node.module` and
                # so could not see a relative submodule import at all. It promptly
                # flagged `simlog` and `events`, both of which are called from
                # pipeline.py and learn.py. An instrument that fires on a case whose
                # answer you already know is how you find out it is broken.
                tails += [a.name for a in node.names]
            else:
                continue
            for tail in tails:
                if py.stem == tail:          # a module importing its own name
                    continue
                out.setdefault(tail, set()).add(rel)
    return out


def test_wired_modules_have_a_production_importer():
    """`Wiring.WIRED` must mean something calls it, not that someone intended to.

    The clause `test_every_module_declares_wiring` leaves unchecked.
    """
    importers = _production_importers()
    unwired = []
    for name in _modules():
        mod = importlib.import_module(name)
        if getattr(mod, "__wiring__", None) is None:
            continue
        if str(getattr(mod.__wiring__, "value", mod.__wiring__)) != "wired":
            continue
        if name.endswith("__init__") or name.count(".") < 2:
            continue                          # package roots are entry points
        if not importers.get(name.rsplit(".", 1)[-1]):
            unwired.append(name)

    surprising = sorted(set(unwired) - set(KNOWN_UNWIRED))
    assert not surprising, (
        "modules declare Wiring.WIRED but nothing in production imports them: "
        f"{surprising}. Wire them, or change the declaration to STANDALONE. "
        "Adding to KNOWN_UNWIRED to silence this is the trap the guard exists for."
    )

    # the other direction: a listed module that is now wired must leave the list,
    # or the ledger quietly becomes a lie in the reassuring direction.
    fixed = sorted(set(KNOWN_UNWIRED) - set(unwired))
    assert not fixed, (
        f"these are wired now and must be removed from KNOWN_UNWIRED: {fixed}"
    )
