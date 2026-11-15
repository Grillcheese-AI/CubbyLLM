"""Order 0073, step 5 (validation/sft/build_view_split_sft.py): the reader's two SFT sets from a small build. The
two sets have the same records in the same order and differ only in the answers that carry a ``not``; train rows
and pairs are train, dev is held, test and test_long are in neither; every answer reads back through the grammar's
rules; the key tokens are each check's tag and description."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation" / "sft"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0073 as B  # noqa: E402
import build_view_split_sft as S  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402


class Tok:
    """A character tokenizer: one token per character, offsets exact."""
    class Enc:
        def __init__(self, text):
            self.ids, self.offsets = [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def encode(self, text):
        return self.Enc(text)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    ds = tmp_path_factory.mktemp("rows0073")
    sys.argv = ["build_rows_0073.py", "--train", "12", "--pairs", "4", "--held", "4", "--reserve", "40",
                "--band", "0", "1", "--out", str(ds)]
    B.main()
    out = tmp_path_factory.mktemp("sft")
    mans = S.build(ds, out, None, 1280)
    sets = {arm: [json.loads(ln) for ln in open(out / S.NAME.format(arm=arm), encoding="utf-8")] for arm in V.ARMS}
    return ds, sets, mans


def test_the_two_sets_have_the_same_records_and_differ_only_in_the_nots(built):
    ds, sets, mans = built
    m, w = sets["m"], sets["w"]
    assert [(a["id"], a["split"], a["family"], a["prompt"]) for a in m] == [(b["id"], b["split"], b["family"], b["prompt"]) for b in w]
    differ = [(a["answer"], b["answer"]) for a, b in zip(m, w) if a["answer"] != b["answer"]]
    assert differ and all("not " in b and len(b) != len(a) for a, b in differ)
    assert mans["m"]["answers that differ from the other arm's"] == len(differ) == mans["w"]["answers that differ from the other arm's"]
    assert mans["m"]["sha256"] != mans["w"]["sha256"] and mans["m"]["rows_sha256"] == mans["w"]["rows_sha256"]


def test_train_and_pairs_are_train_dev_is_held_and_the_test_sets_are_out(built):
    ds, sets, mans = built
    rows = {r["id"]: r for name in ("rows.jsonl", "pair_rows.jsonl") for r in map(json.loads, open(ds / name, encoding="utf-8"))}
    for arm in V.ARMS:
        recs = sets[arm]
        assert {rows[x["id"]]["split"] for x in recs} == {"train", "dev"} and len(recs) == 12 + 8 + 4
        assert all((x["split"] == "held") == (rows[x["id"]]["split"] == "dev") for x in recs)
        assert mans[arm]["by_split"] == {"train": 20, "held": 4} and mans[arm]["records"] == 24
        assert {x["family"].split("_")[0] for x in recs} == {"long", "pair"}
        for x in recs:
            r = rows[x["id"]]
            assert x["prompt"] == V.prompt(r["instruction"]) and x["answer"] == V.target(r, arm)
            assert "lines" in V.parse(x["answer"], r["instruction"], arm, own_text=True)


def test_the_key_tokens_are_the_tags_and_descriptions(built):
    _, sets, _ = built
    got = S.lengths(sets["w"], Tok(), 1280)
    assert got["records_over_max_len"] == {} and 0.05 < got["share of answer tokens that are key"] < 0.6
    assert S.lengths(sets["w"], Tok(), 50)["records_over_max_len"]             # counted when a record is too long
