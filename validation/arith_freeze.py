"""arith_freeze - order 0085, line 3: what a read is frozen on, hashed at its start and at its end.

``freeze(adapter, settings, vm_exe)`` -> {"adapter": {"name", "sha256"} (the weights file, `emitter_lora.safetensors`),
"speak rule": {"files": {path: sha256}, "sha256"} (`cubbyllm/reasoning/speak_vote.py` and the answer-kind rules
beside it, `answer_kind.py`; line endings read as LF so a checkout on another machine hashes the same), "settings":
{..., "sha256"} (k, m, temperature, top-k, the rules: JSON with sorted keys), "vm": {"exe sha256", "commit"} (the
VM binary, and the commit of the repository it was built in when git can tell)}. No time in it: two calls on an
unchanged read are equal, and `differences(start, end)` names what moved. A mismatch voids the read.

Wiring: STANDALONE (`exp_0078_dev_read.py --records`).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

__wiring__ = "STANDALONE"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEAK_RULE = ("cubbyllm/reasoning/speak_vote.py", "cubbyllm/reasoning/answer_kind.py")
WEIGHTS = "emitter_lora.safetensors"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _commit(exe: str) -> "str | None":
    try:
        out = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(exe)), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def freeze(adapter: str, settings: dict, vm_exe: "str | None") -> dict:
    w = os.path.join(adapter, WEIGHTS) if adapter and os.path.isdir(adapter) else adapter
    files = {p: _source_sha(os.path.join(ROOT, p)) for p in SPEAK_RULE}
    return {"adapter": {"name": os.path.basename(os.path.normpath(adapter)) if adapter else None,
                        "sha256": sha256_file(w) if w and os.path.isfile(w) else None},
            "speak rule": {"files": files, "sha256": hashlib.sha256("".join(files[p] for p in SPEAK_RULE).encode()).hexdigest()},
            "settings": {**settings, "sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()},
            "vm": {"exe sha256": sha256_file(vm_exe) if vm_exe and os.path.isfile(vm_exe) else None,
                   "commit": (_commit(vm_exe) if vm_exe else None) or os.environ.get("CUBELANG_COMMIT")}}


def differences(a: dict, b: dict, at: str = "") -> list[str]:
    """The paths where two freezes differ ([] when the read stayed frozen)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in differences(a.get(k), b.get(k), f"{at}.{k}" if at else k)]
    return [] if a == b else [at]


__all__ = ["SPEAK_RULE", "WEIGHTS", "differences", "freeze", "sha256_file"]
