"""standin/serve.py moved (order 0031): the brain to cubbyllm/brain.py, its builder and command line to
cubby_serve/build.py. This name works for one cycle: importing it gives cubbyllm.brain itself (so a patch on it patches
the brain), and running it runs cubby_serve/build.py."""
import importlib as _importlib
import os as _os
import sys as _sys
import warnings as _warnings

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
_NEW = "cubbyllm.brain"
if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("cubby_serve.build", run_name="__main__", alter_sys=True)
else:
    _warnings.warn(f"{__name__} moved to {_NEW} (order 0031); build_serve to cubby_serve.build", DeprecationWarning,
                   stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module(_NEW)
