"""Pytest bootstrap: make the repo root importable so `import cubbyllm` works
without an editable install.

Order 0031 moves the stand-in's tests here a slice at a time, and some of them
import helpers from other test files (`from test_ask import FakeEmitter`). Each
plain test folder (no `__init__.py`), and `standin/tests` while it exists, is put
on the path, so a helper resolves wherever its file lives now."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for d in [ROOT / "standin" / "tests", *sorted(p for p in (ROOT / "tests").iterdir() if p.is_dir())]:
    if d.is_dir() and not (d / "__init__.py").exists() and d.name != "__pycache__" and str(d) not in sys.path:
        sys.path.append(str(d))
