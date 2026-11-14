"""Order 0067, the second reader's step 3: the join of a long instruction's parts into one CubeLang program.

**Moved to the core** by order 0073's step 1: `cubbyllm/speech/view_join.py` (`HEAD`, `join`, `program`, `spec`), so
the host and the row builders use one join. This name stays as a shim for one cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cubbyllm.speech.view_join import HEAD, join, program, spec  # noqa: E402

__wiring__ = "STANDALONE"

__all__ = ["HEAD", "join", "program", "spec"]
