"""Order 0073, step 4 (validation/exp_view_split_gold.py, view_split_probe.py): the check of the reader's form on a
small build. Every row's target parses back, is taken whole by the machine in both variants and accounts to the
row's own program; arm M's machine refuses arm W's ``not``; on one-slot changes of the gold answer the machine and
`parse` agree, some taken and some refused; walked at random in ids, no state is left without an id or a way on."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0073 as B  # noqa: E402
import exp_view_split_gold as G  # noqa: E402
import view_split_probe as P  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    out = tmp_path_factory.mktemp("rows0073")
    sys.argv = ["build_rows_0073.py", "--train", "12", "--pairs", "4", "--held", "4", "--reserve", "40",
                "--band", "0", "1", "--out", str(out)]
    B.main()
    return [json.loads(ln) for name in ("rows.jsonl", "pair_rows.jsonl") for ln in open(out / name, encoding="utf-8")]


def test_every_target_parses_back_is_taken_whole_and_accounts_to_the_rows_program(rows):
    got, n = G.gold(rows), len(rows)
    for arm in V.ARMS:
        assert got[f"arm {arm}: targets"] == n == got[f"arm {arm}: the accounting gives the row's program"]
        assert got[f"arm {arm}: the accounting gives the row's groups"] == n and got[f"arm {arm}: something asked or flagged"] == 0
        for name in ("free", "own text"):
            assert got[f"arm {arm}, {name}: parsed back to the row's lines"] == n == got[f"arm {arm}, {name}: taken whole by the machine"]
    assert 0 < got["rows where the arms differ"] == got["arm W's target refused by arm M's machine"]


def test_on_one_slot_changes_the_machine_and_parse_agree(rows):
    got = G.agree(rows[:14], 3)
    for name in ("free", "own text"):
        assert got[f"{name}: the two differ"] == 0, {k: v for k, v in got.items() if "differ on" in k}
        assert got[f"{name}: taken by both"] > 100 and got[f"{name}: refused by both"] > 100
    assert got["own text: taken by both"] < got["free: taken by both"]                    # the variant refuses more
    kinds = {k.split(": ")[1] for k in got if k.startswith("kind of change")}
    assert {"a comparison", "a view", "a column", "a number", "a join tag", "a kind", "a line cut in two", "no change"} <= kinds
    lines = V.parse(V.target(rows[0], "w"), rows[0]["instruction"], "w")["lines"]
    changes = P.mutants(lines, rows[0]["instruction"], random.Random(1))
    assert ("no change", V.target(rows[0], "w")) in changes and len({a for _, a in changes}) > 20


def test_walked_in_ids_no_state_is_left_without_an_id_or_a_way_on(rows):
    letters = sorted({ch for r in rows for arm in V.ARMS for ch in V.target(r, arm) + r["instruction"]}
                     | set("abcdefghijklmnopqrstuvwxyz0123456789_<>=!|\"\\ \n"))
    pieces = [""] + letters + ["check ", " | ", "none ", " where ", "the ", "\ncheck n"]
    vocab, eos = T.Vocab(dict(enumerate(pieces))), len(pieces)
    got = G.walked(rows[:3], vocab, 5, walks=3, steps=250, sample=25)
    assert got["machines"] == 12 and got["states reached"] > 1000
    assert got["states with no id allowed that are not a whole answer"] == 0 == got["states with no way on to a whole answer"]
    ids = {p: i for i, p in enumerate(pieces)}
    tk = SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[ids[ch] for ch in text]))
    read = G.in_ids(rows[:4], vocab, tk, eos)
    assert read["targets"] == 16 == read["ended `finished`"] and read["ids allowed at a step, most"] < len(pieces)
