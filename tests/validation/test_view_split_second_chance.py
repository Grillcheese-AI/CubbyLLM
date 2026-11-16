"""Order 0073, step 6 (validation/exp_view_split_second_chance.py; `change` and `differs` of view_split_rowscore.py):
the second chance on descriptions. A reversal is named by its slot, before -> after; a description that is not the
gold's says in which slots; the table puts a fixed check under its cue words, how it was said and its ``not``s, and
an unfixed one under the slots that differ; gold answers leave nothing to fix; a reversed comparison is fixed by
the reversal back; the script refuses answers of test rows."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0073 as B  # noqa: E402
import exp_view_split_second_chance as SC  # noqa: E402
import view_split_rowscore as R  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402

WORDS = {"q": "count", "view": "word", "op": ">=", "v": 4, "where": [{"col": "len", "op": "<", "v": 10}]}
LAST = {"q": "at", "view": "sentence", "pos": -1, "col": "end_mark", "op": "==", "val": "."}


def test_a_reversal_is_named_by_its_slot_and_a_difference_by_its_slots():
    assert R.change(WORDS, dict(WORDS, op="<")) == "a count compared: >= -> <"
    assert R.change(WORDS, dict(WORDS, where=[{"col": "len", "op": ">=", "v": 10}])) == "a filter: < -> >="
    assert R.change({"q": "n", "view": "word", "op": "<", "v": 40}, {"q": "n", "view": "word", "op": ">", "v": 40}) == "a view's size: < -> >"
    assert R.change(dict(WORDS, q="any"), dict(WORDS, q="none")) == "the quantifier: any -> none"
    assert R.change(LAST, dict(LAST, op="!=")) == "a position's value: == -> !=" and R.change(LAST, dict(LAST, absent="pass")) == "absent: fail -> pass"
    assert R.differs(WORDS, WORDS) == [] and R.differs(dict(WORDS, v=5, view="sentence"), WORDS) == ["view", "v"]
    assert R.differs(dict(WORDS, where=[{"col": "lower", "op": "<", "v": 9}]), WORDS) == ["a condition's col", "a condition's v"]
    assert R.differs(dict(WORDS, where=[]), WORDS) == ["the number of conditions"] and R.differs(dict(LAST, absent="pass"), LAST) == ["absent"]


def part(**kw) -> dict:
    return dict({"part": 0, "line": True, "as": "check", "right": False, "same": False, "fixed": False, "fix": None, "differs": [], "nots": 0}, **kw)


def test_the_table_files_each_gold_check_once():
    text = "Use at most 40 words. Do not use the word \"so\". Stay kind. End with a period, if there is one."
    cuts = [(0, 21), (22, 47), (48, 58), (59, len(text))]
    assert [text[a:b] for a, b in cuts[:3]] == ["Use at most 40 words.", 'Do not use the word "so".', "Stay kind."]
    row = {"instruction": text, "lang": "en", "segments": [{"kind": "check", "part": i, "how": how, "start": a, "end": b}
                                                           for i, (how, (a, b)) in enumerate(zip(("imp", "imp", "imp", "cond"), cuts))]}
    scored = {"parts": [part(part=0, fixed=True, fix="a view's size: > -> <=", differs=["op"]),
                        part(part=1, differs=["view", "a condition's v"], nots=1),
                        part(part=2, line=False, **{"as": "none"}, differs=None),
                        part(part=3, right=True, differs=["absent"])]}
    more = {"parts": [part(part=0, fixed=True, fix="a view's size: > -> <=", differs=["op"]), part(part=1, right=True, same=True),
                      part(part=2, line=False, **{"as": None}, differs=None), part(part=3, differs=None)]}
    got = SC.table([row, row], [scored, more])
    assert got["counts"] == {"rows": 2, "gold checks": 8, "fixed by one reversal": 2, "wrong, and no one reversal fixes it": 1,
                             "no check line: written as `none`": 1, "no check line: no line with its span": 1,
                             "a description that does not compile": 1, "right as written": 2, "right as written, not the gold's form": 1}
    assert got["patterns"] == {"at most | said: imp | not: 0 | en": {"a view's size: > -> <=": 2}}
    assert got["not fixed, by the slots that differ"] == {"view + a condition's v": 1}
    assert got["right in another form, by the slots that differ"] == {"absent": 1}
    assert SC.cue("Ne mets pas plus de 3 phrases, sauf si") == "plus de + pas + ne + sauf si" and SC.cue("Write it.") == "(no cue word)"
    assert SC.cue("End on a small note, if there is one.") == "if there is"            # whole words: not "all", "no", "ne"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    out = tmp_path_factory.mktemp("rows0073")
    sys.argv = ["build_rows_0073.py", "--train", "16", "--pairs", "4", "--held", "6", "--reserve", "40",
                "--band", "0", "1", "--out", str(out)]
    B.main()
    return out, [json.loads(ln) for ln in open(out / "rows.jsonl", encoding="utf-8")], cc.CubelangSession()


def rec(row: dict, answer: str) -> dict:
    return {"id": row["id"], "split": row["split"], "arm": "m", "answer": answer, "end": "finished", "ids": 40, "own_text": False}


def test_gold_answers_leave_nothing_to_fix_and_a_reversed_comparison_is_fixed_by_the_reversal_back(built):
    _, rows, vm = built
    gold = SC.table(rows, [R.score(r, rec(r, V.target(r, "m")), "m", False, vm) for r in rows])
    assert gold["counts"] == {"rows": len(rows), "gold checks": sum(r["n_parts"] for r in rows), "right as written": sum(r["n_parts"] for r in rows)}
    assert gold["patterns"] == {} == gold["not fixed, by the slots that differ"]
    used, scored, want = [], [], []
    for r in rows:
        views = [RV.views(x["text"]) for x in r["replies"]]
        i = next((i for i, s in enumerate(r["said"]) if s["m"]["q"] in ("n", "count")
                  and [VS.evaluate(dict(s["m"], op=VS.INV[s["m"]["op"]]), v) for v in views] != [VS.evaluate(s["m"], v) for v in views]), None)
        if i is None or len(used) >= 5:
            continue
        m, new = r["said"][i]["m"], copy.deepcopy(r)
        new["said"][i]["m"] = dict(m, op=VS.INV[m["op"]])
        used.append(r)
        scored.append(R.score(r, rec(r, V.target(new, "m")), "m", False, vm))
        want.append(f"{R.SITE[m['q']]}: {VS.INV[m['op']]} -> {m['op']}")
    got = SC.table(used, scored)
    assert len(used) == 5 == got["counts"]["fixed by one reversal"] and "wrong, and no one reversal fixes it" not in got["counts"]
    fixes = [f for v in got["patterns"].values() for f, n in v.items() for _ in range(n)]
    assert sorted(fixes) == sorted(want) and all(" | said: " in k and k.rsplit(" | ", 1)[1] in ("en", "fr") for k in got["patterns"])


def test_the_script_refuses_answers_of_test_rows(built, tmp_path):
    out, rows, _ = built
    for split, refused in (("test", True), ("dev", False)):
        r = next(x for x in rows if x["split"] == split)
        answers = tmp_path / f"{split}.jsonl"
        answers.write_text(json.dumps(rec(r, V.target(r, "m"))) + "\n", encoding="utf-8")
        sys.argv = ["exp_view_split_second_chance.py", "--answers", str(answers), "--rows", str(out / "rows.jsonl"), "--out", str(tmp_path / f"{split}.json")]
        if refused:
            with pytest.raises(SystemExit, match="dev rows only"):
                SC.main()
        else:
            SC.main()
            assert json.loads((tmp_path / "dev.json").read_text(encoding="utf-8"))["counts"]["rows"] == 1
