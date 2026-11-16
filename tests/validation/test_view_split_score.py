"""Order 0073, step 6 (validation/view_split_rowscore.py, exp_view_split_score.py): the scoring, on a small build.
The gold answer of every row scores whole, checked and right on every count, in both arms; a reversed comparison is
not right and is fixed by one reversal; a gold check written as ``none`` leaves the row checked and costs false
passes; an answer cut at the limit is not whole, its rest is flagged and none of its replies is checked; a changed
or dropped join tag is counted at the pair; a description that does not compile is asked by the host and its row
is not checked."""
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
import exp_view_split_score as E  # noqa: E402
import view_split_rowscore as R  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402


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


def rec(row: dict, answer: str, end: str = "finished", arm: str = "m") -> dict:
    return {"id": row["id"], "split": row["split"], "arm": arm, "answer": " " + answer, "end": end, "ids": 40,
            "seconds": 0.5, "own_text": False}


def changed(row: dict, edit) -> dict:
    new = copy.deepcopy(row)
    edit(new)
    return new


def checks(row: dict) -> dict:
    return {s["part"]: s for s in row["segments"] if s["kind"] == "check"}


def lone(row: dict) -> list[int]:
    return [g[1] for g in row["groups"] if g[0] == "one"]


def test_the_gold_answer_scores_whole_checked_and_right_in_both_arms(built):
    _, rows, vm = built
    for arm, own in (("m", False), ("w", True)):
        scored = [R.score(r, rec(r, V.target(r, arm), arm=arm), arm, own, vm) for r in rows]
        for s in scored:
            assert s["whole"] and s["checked"] and s["unaccounted"] == 0 == s["flagged_chars"] and s["segments_right"] == s["segments"]
            assert all(p["line"] and p["right"] and p["same"] and not p["fixed"] and p["as"] == "check" for p in s["parts"])
            assert all(g["right"] and g["right_or"] for g in s["pairs"]) and s["pairs_not_gold"] == 0
            assert all(x["verdict"] == x["held"] for x in s["replies"])
        got = E.tally(scored)
        c, rates = got["counts"], E.rates(got)
        assert c["rows"] == len(rows) == c["rows whole"] == c["rows checked"] == c["rows that ended `finished`"]
        assert c["replies"] == c["replies checked"] == c["agree"] == sum(len(r["replies"]) for r in rows) and c["false passes"] == 0
        assert c["gold checks"] == c["parts right"] == c["parts right without a run (the gold form)"] == sum(r["n_parts"] for r in rows)
        assert c["gold pairs"] == c["gold pairs right"] == sum(1 for r in rows for g in r["groups"] if g[0] != "one") > 0
        assert rates["line 3: agreement"] == 1.0 == rates["line 5: parts right"] == rates["line 7: replies checked"]
        assert rates["line 3: false passes over replies built to fail"] == 0.0 == rates["line 7: gold checks asked"]
        assert sum(n for _, n in got["by"]["direction"].values()) == c["gold checks"] == sum(n for _, n in got["by"]["position"].values())
        assert sum(x["replies"] for x in got["by"]["rows by parts"].values()) == c["replies"]


def test_a_reversed_comparison_is_not_right_and_is_fixed_by_one_reversal(built):
    _, rows, vm = built
    told = wrong = 0
    for r in rows:
        views = [RV.views(x["text"]) for x in r["replies"]]
        for i, said in enumerate(r["said"]):
            m = said["m"]
            if m["q"] not in ("n", "count"):
                continue
            mut = dict(m, op=VS.INV[m["op"]])
            if [VS.evaluate(mut, v) for v in views] == [VS.evaluate(m, v) for v in views] or told >= 6:
                continue
            told += 1
            s = R.score(r, rec(r, V.target(changed(r, lambda n: n["said"][i].update(m=mut)), "m")), "m", False, vm)
            p = s["parts"][i]
            assert p["line"] and not p["same"] and not p["right"] and p["fixed"] and s["checked"]
            assert all(q["right"] for k, q in enumerate(s["parts"]) if k != i)
            wrong += sum(1 for x in s["replies"] if x["verdict"] != x["held"])
    assert told == 6 and wrong > 0                          # the row's verdict changes with it on some reply


def test_a_gold_check_written_as_none_leaves_the_row_checked_and_costs_false_passes(built):
    _, rows, vm = built
    r = next(x for x in rows if x["n_parts"] >= 2 and lone(x))
    i = lone(r)[0]
    s = R.score(r, rec(r, V.target(changed(r, lambda n: checks(n)[i].update(kind="none")), "m")), "m", False, vm)
    p = s["parts"][i]
    assert s["whole"] and s["checked"] and not p["line"] and p["as"] == "none" and not p["right"] and not p["asked_by_reader"]
    forms, groups = [x["m"] for x in r["said"]], [tuple(g) for g in r["groups"] if tuple(g) != ("one", i)]
    want = [VS.evaluate(VJ.spec(forms, groups), RV.views(x["text"])) for x in r["replies"]]   # the row without that check
    assert [x["verdict"] for x in s["replies"]] == want
    c = E.tally([s])["counts"]
    assert c["false passes"] == sum(1 for w, x in zip(want, r["replies"]) if w and not x["held"])
    assert c["gold checks written as `none`"] == 1 and c["gold checks with their line"] == r["n_parts"] - 1
    asked = R.score(r, rec(r, V.target(changed(r, lambda n: checks(n)[i].update(kind="asked")), "m")), "m", False, vm)
    assert asked["checked"] and asked["parts"][i]["asked_by_reader"] and [x["verdict"] for x in asked["replies"]] == want
    assert E.tally([asked])["counts"]["false passes on rows where the reader wrote a gold check as asked"] == c["false passes"]


def test_an_answer_cut_at_the_limit_is_not_whole_and_none_of_its_replies_is_checked(built):
    _, rows, vm = built
    r = next(x for x in rows if x["n_parts"] >= 3)
    answer = V.target(r, "m")
    s = R.score(r, rec(r, answer[:len(answer) - 5], end="limit"), "m", False, vm)
    last = max(checks(r))
    assert not s["whole"] and not s["checked"] and s["flagged_chars"] > 0 and s["unaccounted"] == 0
    assert all(x["verdict"] is None for x in s["replies"])
    assert all(p["line"] and p["right"] for p in s["parts"][:last]) and len(s["parts"]) == r["n_parts"]
    c = E.tally([s])["counts"]
    assert c["rows that ended `limit`"] == 1 and c["replies checked"] == 0 and c["rows with something flagged"] == 1
    assert c["replies not checked: their row is not checked"] == len(r["replies"])
    none = R.score(r, None, "m", False, vm)
    assert not none["read"] and none["unaccounted"] == 0 and none["flagged_chars"] == len(r["instruction"].strip())
    assert E.tally([none])["counts"]["rows not read"] == 1 and not any(p["line"] for p in none["parts"])


def test_a_changed_or_dropped_join_tag_is_counted_at_the_pair(built):
    _, rows, vm = built

    def retag(row, kind, first, second, tag):               # the pair's tags taken off, then `tag` put on `first`
        def edit(n):
            for k in (first, second):
                checks(n)[k].pop("join", None)
            if tag:
                checks(n)[first]["join"] = tag
        return V.target(changed(row, edit), "m")
    r, g = next((x, g) for x in rows for g in x["groups"] if g[0] == "unless")
    a, b = sorted(g[1:])
    s = R.score(r, rec(r, retag(r, "unless", a, b, ["either", "next"])), "m", False, vm)
    pair = next(x for x in s["pairs"] if x["connective"] == "unless")
    assert not pair["right"] and pair["right_or"] and s["pairs_not_gold"] == 1 and s["checked"]
    assert all(x["verdict"] == x["held"] for x in s["replies"])                # the two compile the same
    s = R.score(r, rec(r, retag(r, "unless", a, b, None)), "m", False, vm)
    pair = next(x for x in s["pairs"] if x["connective"] == "unless")
    assert not pair["right"] and not pair["right_or"] and s["pairs_not_gold"] == 0 and s["checked"]
    r, g = next((x, g) for x in rows for g in x["groups"] if g[0] == "if")
    cond, then = g[1], g[2]
    s = R.score(r, rec(r, retag(r, "if", then, cond, ["if", "next" if cond > then else "prev"])), "m", False, vm)   # turned round
    pair = next(x for x in s["pairs"] if x["connective"] == "if")
    assert not pair["right"] and not pair["right_or"] and s["pairs_not_gold"] == 1
    assert E.tally([s])["by"]["connective"]["if"][0] == sum(1 for x in s["pairs"] if x["connective"] == "if") - 1


def test_a_description_that_does_not_compile_is_asked_by_the_host_and_its_row_is_not_checked(built):
    _, rows, vm = built
    r = next(x for x in rows if lone(x))
    i = lone(r)[0]
    lines, k = V.target(r, "w").split("\n"), 1 + sorted(s["start"] for s in r["segments"]).index(checks(r)[i]["start"])
    lines[k] = "check not all word where len > 3 and len < 9" + V.BAR + lines[k].split(V.BAR, 1)[1]
    s = R.score(r, rec(r, "\n".join(lines), arm="w"), "w", False, vm)
    p = s["parts"][i]
    assert s["whole"] and not s["checked"] and p["line"] and p["asked_by_host"] and not p["right"] and not p["fixed"]
    assert all(x["verdict"] is None for x in s["replies"]) and s["asked_by_host"] == 1
    got = E.tally([s])
    assert got["counts"]["gold checks asked by the host"] == 1 == got["counts"]["rows with a part asked by the host"]
    assert E.rates(got)["line 7: replies checked"] == 0.0


def test_the_script_scores_a_file_of_answers_and_keeps_the_rows_counts(built, tmp_path):
    out, rows, _ = built
    dev = [r for r in rows if r["split"] == "dev"]
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(rec(r, V.target(r, "w"), arm="w")) + "\n" for r in dev[:-1]), encoding="utf-8")
    sys.argv = ["exp_view_split_score.py", "--answers", str(answers), "--rows", str(out / "rows.jsonl"), "--splits", "dev",
                "--out", str(tmp_path / "report.json"), "--keep", str(tmp_path / "kept.jsonl")]
    E.main()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    c = report["splits"]["dev"]["counts"]
    assert report["arm"] == "w" and report["own_text"] is False and len(report["answers_sha256"]) == 64 == len(report["rows_sha256"])
    assert c["rows"] == len(dev) and c["rows not read"] == 1 and c["rows checked"] == len(dev) - 1
    kept = [json.loads(ln) for ln in open(tmp_path / "kept.jsonl", encoding="utf-8")]
    assert [k["id"] for k in kept] == [r["id"] for r in dev] and kept[-1]["checked"] == 0 and kept[0]["parts right"] == dev[0]["n_parts"]
    assert set(kept[0]) == {"id", "split", "n_parts", "lang", "replies", "checked", "agree", "false passes", "built to fail",
                            "gold checks", "parts right"}
