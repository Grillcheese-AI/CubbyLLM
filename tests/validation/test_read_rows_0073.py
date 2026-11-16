"""Order 0073, step 6 (validation/builders/read_rows_0073.py): the file the Colab read is given holds the held-out
rows' ids, splits and instructions and nothing else: no train row, no segment, no description, no reply."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import read_rows_0073 as B  # noqa: E402


def test_only_ids_splits_and_instructions_of_the_held_out_rows_leave(tmp_path):
    rows = [{"id": f"d{k}", "split": s, "instruction": f"Écris {k} mots.", "segments": [{"kind": "check"}], "said": [{"m": {}}],
             "join": "program", "replies": [{"text": "un", "held": True}]}
            for k, s in enumerate(["train", "test_long", "dev", "test", "train", "dev"])]
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    man = B.build(src, tmp_path / "out")
    body = (tmp_path / "out" / "read_rows.jsonl").read_bytes()
    got = [json.loads(ln) for ln in body.decode("utf-8").splitlines()]
    assert [(r["id"], r["split"]) for r in got] == [("d2", "dev"), ("d5", "dev"), ("d3", "test"), ("d1", "test_long")]
    assert all(set(r) == {"id", "split", "instruction"} for r in got) and got[0]["instruction"] == "Écris 2 mots."
    assert man["rows"] == 4 and man["by_split"] == {"dev": 2, "test": 1, "test_long": 1} and man["keys"] == ["id", "split", "instruction"]
    assert man["sha256"] == hashlib.sha256(body).hexdigest() and man["rows_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "out" / "read_rows.manifest.json").read_text(encoding="utf-8")) == man
    for word in ("segments", "said", "join", "replies", "held", "program"):
        assert word.encode() not in body
