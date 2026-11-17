"""Order 0074, step 1 (validation/builders/build_rows_0074.py): a small build, run in the VM as it is built. The train
set holds rows of 5 to 8 parts, each length as often; dev holds both halves; test_longer's rows have 9 to 12 parts,
their replies that can tell and their joined program run in the VM; the set read back passes its check
(validation/exp_rows_0074_check.py) and a spoiled copy does not; the manifest carries no row text."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0074 as B  # noqa: E402
import exp_rows_0074_check as CK  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    out = tmp_path_factory.mktemp("rows0074")
    sys.argv = ["build_rows_0074.py", "--train", "8", "--train-long", "8", "--pairs", "4", "--held", "6",
                "--reserve", "80", "--band", "0", "1", "--out", str(out)]
    B.main()                                             # stops on any VM / evaluator disagreement
    read = lambda name: [json.loads(ln) for ln in open(out / name, encoding="utf-8")]   # noqa: E731
    return read("rows.jsonl"), read("pair_rows.jsonl"), json.loads((out / "manifest.json").read_text(encoding="utf-8")), out


def test_each_set_has_its_lengths_and_train_has_each_long_length_as_often(built):
    rows, pairs, man, _ = built
    lengths = {s: Counter(r["n_parts"] for r in rows if r["split"] == s) for s in B.SPLITS}
    assert set(lengths["test"]) <= {2, 3, 4} and set(lengths["test_long"]) <= {5, 6, 7, 8}
    assert set(lengths["test_longer"]) <= {9, 10, 11, 12} and sum(lengths["test_longer"].values()) == 6
    assert sum(n for k, n in lengths["dev"].items() if k <= 4) == 3 == sum(n for k, n in lengths["dev"].items() if k >= 5)
    long_train = Counter(r["n_parts"] for r in rows if r["id"].startswith("d1"))
    assert long_train == Counter({5: 2, 6: 2, 7: 2, 8: 2})
    assert all(2 <= r["n_parts"] <= 4 for r in pairs) and len(pairs) == 8
    assert man["by_split"] == {"train": 24, "dev": 6, "test": 6, "test_long": 6, "test_longer": 6}
    assert man["by_plan_part"]["train_long"] == 8 and man["pool"] == {"2 to 4": 80, "5 to 8": 160, "9 to 12": 240}


def test_rows_of_9_to_12_parts_carry_replies_that_can_tell_and_their_verdicts_hold(built):
    rows, _, _, _ = built
    longer = [r for r in rows if r["split"] == "test_longer"]
    assert any(x.get("extra") for r in longer for x in r["replies"])
    for r in longer:
        m, groups = [s["m"] for s in r["said"]], [tuple(g) for g in r["groups"]]
        assert r["join"] == VJ.join(m, groups) and len(r["said"]) == r["n_parts"]
        for x in r["replies"]:
            assert VS.evaluate(VJ.spec(m, groups), RV.views(x["text"])) == x["held"]
    assert not any(x.get("extra") for r in rows if r["split"] == "train" for x in r["replies"])


def test_the_set_read_back_passes_its_check_and_a_spoiled_copy_does_not(built, tmp_path):
    rows, pairs, _, out = built
    got = CK.check(out, [])
    assert not any(got["must be 0"].values()), got["must be 0"]
    assert got["counted"]["rows"] == len(rows) + len(pairs)
    bad, i = list(rows), next(j for j, r in enumerate(rows) if r["n_parts"] >= 5 and r["split"] == "train")
    bad[i] = dict(rows[i], split="test")                             # a train row of 5 to 8 parts called test
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bad),
                                         encoding="utf-8", newline="\n")
    for name in ("pair_rows.jsonl", "manifest.json"):
        shutil.copy(out / name, tmp_path / name)
    faults = CK.check(tmp_path, [])["must be 0"]
    assert faults["a row whose number of parts is not its set's"] == 1
    assert faults["a SHA-256 that is not the manifest's"] >= 1


def test_the_manifest_carries_counts_and_hashes_and_no_row_text(built):
    rows, pairs, man, _ = built
    dump = json.dumps(man, ensure_ascii=False)
    assert not any(r["instruction"] in dump or r["replies"][0]["text"] in dump for r in rows + pairs)
    assert set(man["sha256"]["the lines of rows.jsonl by split"]) == set(B.SPLITS)
    assert man["vm_disagreements"] == 0 and man["vm_runs_agreeing"] == sum(len(r["replies"]) for r in rows + pairs)
