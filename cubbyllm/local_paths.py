"""cubbyllm/local_paths.py -- where this machine keeps things, by name, so no committed file carries a local path.

A committed file names a place (``$CUBBY_MODELS/base450m``, ``$CUBBY_DATASETS/hdc``); ``expand`` turns the names into
this machine's folders. A name is read from, in order:

1. the environment (``CUBBY_MODELS=...``);
2. ``paths.local.json`` at the repo root -- git-ignored, one machine's ``{"CUBBY_MODELS": "...", ...}``;
3. the repo's own places, which need no setting: ``CUBBY_REPO`` (this checkout) and ``CUBBY_GITHUB`` (the folder that
   holds it and its sibling repos).

Then the platform's own variables (``$ProgramFiles``, ``$WINDIR``, ``$TEMP``) and ``~``. A name nobody sets is left as
written, so the error a script then raises names what to set. The guard ``tests/test_local_paths.py`` keeps local
paths out of committed text; ``tools/scrub_local_paths.py`` rewrites them to these names.

Wired: WIRED (standin's loci reader, the validation scripts' defaults, the tests that read a model file; ``data_dir``:
every data place the packages keep, order 0033).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

__wiring__ = "WIRED"

REPO = Path(__file__).resolve().parents[1]
FILE = REPO / "paths.local.json"
DERIVED = {"CUBBY_REPO": str(REPO), "CUBBY_GITHUB": str(REPO.parent)}
DATA_DEFAULT = REPO / "standin" / "data" / "out"          # the data root when nothing names one (`data_dir`)
_VAR = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)\}?")
_cache: dict | None = None


def _file() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def value(name: str, default: str | None = None) -> str | None:
    """One name's folder: the environment, then ``paths.local.json``, then the repo's own; else ``default``."""
    return os.environ.get(name) or _file().get(name) or DERIVED.get(name) or default


def data_dir(*parts: str) -> Path:
    """The packages' one data root (order 0033): the caches, the worlds' files, the keys and tokens the server keeps.
    ``$CUBBY_DATA`` (the environment, then ``paths.local.json``), else today's place, ``standin/data/out``; a path under
    it is ``data_dir("web_cache")``. Modules read it when they load, so the server names it before it imports them
    (``python -m cubby_serve --data-root <dir>``). ``tools/move_data_root.py`` moves what is there to a new root."""
    return Path(value("CUBBY_DATA") or DATA_DEFAULT).joinpath(*parts)


def expand(text: str) -> str:
    """``text`` with its ``$NAME`` / ``${NAME}`` replaced (``value``), then the platform's variables and ``~``."""
    out = _VAR.sub(lambda m: value(m.group(1)) or m.group(0), str(text))
    return os.path.expanduser(os.path.expandvars(out))


__all__ = ["data_dir", "expand", "value", "REPO", "FILE"]
