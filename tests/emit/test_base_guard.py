"""Order 0048: the base guard. An adapter whose json records its base's sha256 loads only on that base (a mismatch
refuses, naming both); one that records none loads with a warning; the one-off tool writes the hash only after the
pairing emits a program, keeping the json as it was."""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from cubbyllm.emit.seats_450m import check_base, short_name  # noqa: E402


def test_a_matched_base_loads_a_mismatched_one_refuses_a_missing_hash_warns():
    asked = []
    h = lambda: asked.append(1) or "a" * 64                                         # noqa: E731
    assert check_base({"base_sha256": "a" * 64}, h, "base450m_cont/grilly", "emitter/v12e") == "ok"
    with pytest.raises(ValueError, match="base450m/grilly.*refused"):
        check_base({"base_sha256": "b" * 64, "base_name": "base450m_cont"}, h, "base450m/grilly", "emitter/v12e")
    asked.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert check_base({}, h, "base450m/grilly", "talk/talk_v2_2") == "unrecorded"
    assert w and "records no base hash" in str(w[0].message) and not asked          # no hash asked, no export hashed
    assert short_name("/models/base450m_cont/grilly") == "base450m_cont/grilly"


def test_the_tool_records_only_after_a_program_and_keeps_the_json(tmp_path):
    import adapter_base_hash as T
    assert T.works('create frame: number;\n bind frame, SEED, "$E1";\n bind frame, HOP1, "capital";')
    assert not T.works("use strict;\nclass Tour {}") and not T.works('bind frame, SEED, "France";')
    meta = tmp_path / "emitter_lora.json"
    meta.write_text(json.dumps({"rank": 16}), encoding="utf-8")
    got = T.record(str(meta), "c" * 64, "base450m_cont")
    assert got == {"rank": 16, "base_sha256": "c" * 64, "base_name": "base450m_cont"}
    assert json.loads((tmp_path / "emitter_lora.json.pre_hash").read_text()) == {"rank": 16}
