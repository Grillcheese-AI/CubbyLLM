"""standin/cortex/ moved to cubbyllm/cortex/ (order 0031). This name works for one cycle: importing it (``cortex`` or
``standin.cortex``) gives the moved package itself, and each submodule under the old name is the moved submodule
(``cortex.retina`` is ``cubbyllm.cortex.retina``), so no class exists twice."""
import importlib as _importlib
import os as _os
import sys as _sys
import warnings as _warnings

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
_NEW = "cubbyllm.cortex"
_warnings.warn(f"{__name__} moved to {_NEW} (order 0031)", DeprecationWarning, stacklevel=2)
_pkg = _importlib.import_module(_NEW)
for _sub in ("contract", "forward", "retina", "salience", "what", "where"):
    _sys.modules[f"{__name__}.{_sub}"] = _importlib.import_module(f"{_NEW}.{_sub}")
_sys.modules[__name__] = _pkg
