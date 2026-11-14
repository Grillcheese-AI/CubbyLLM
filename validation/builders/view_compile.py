"""Order 0067, the second reader's step 2: a spec compiled to CubeLang over the reply's views.

**Moved to the core** by order 0073's step 1: `cubbyllm/speech/view_compile.py` (`body`, `program`, `views_of`), so
the host and the row builders use one compiler. This name stays as a shim for one cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubbyllm.speech.view_compile import body, program, views_of  # noqa: E402

__wiring__ = "STANDALONE"

__all__ = ["body", "program", "views_of"]
