"""standin/history_graph.py moved to cubbyllm/worlds/history_graph.py (order 0031). This name works for one cycle: importing it gives the moved module
itself, and running it runs the new home."""
import importlib as _importlib
import os as _os
import sys as _sys
import warnings as _warnings

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
_NEW = "cubbyllm.worlds.history_graph"
if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module(_NEW, run_name="__main__", alter_sys=True)
else:
    _warnings.warn(f"{__name__} moved to {_NEW} (order 0031)", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module(_NEW)
