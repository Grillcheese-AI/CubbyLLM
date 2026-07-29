"""Pytest bootstrap: make the repo root importable so `import cubbyllm` works
without an editable install."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
