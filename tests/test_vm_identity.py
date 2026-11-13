"""Every run names its VM (Nick, 2026-11-13): the identity a summary records carries no drive or user folder, and says
whether the binary is older than its sources."""
from __future__ import annotations

import re

import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.bridges import vm_identity


def test_the_vm_is_named_without_a_local_path():
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"no cubelang.exe ({e})")
    v = vm_identity.identity()
    assert set(v) == {"path", "modified", "sha256", "commit", "older_than_sources"}
    assert not re.match(r"^[A-Za-z]:|^/", v["path"]) and "Users" not in v["path"] and v["path"].endswith(("cubelang.exe", "cubelang"))
    assert len(v["sha256"]) == 64 and isinstance(v["older_than_sources"], bool)
