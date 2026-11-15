"""Order 0073, step 3 (validation/builders/build_rows_0073.py): a small build, run. Each check carries how it was
worded and its said form in both spellings, arm M's the fold of arm W's with no ``not``; the row's program is the
join of the folded forms and the evaluator's verdict on them is each reply's ``held``; a tied pair has one tagged
check whose neighbour is its partner; train rows keep 8 replies and held-out rows mark the ones added; a pair's
twins differ in one lone part; no held-out row has a train row's composition; the manifest carries no row text;
the set read back passes its check (validation/exp_rows_0073_check.py) and a spoiled copy does not."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0073 as B  # noqa: E402
import exp_rows_0073_check as CK  # noqa: E402
import view_said as S  # noqa: E402

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
    out = tmp_path_factory.mktemp("rows0073")
    sys.argv = ["build_rows_0073.py", "--train", "20", "--pairs", "8", "--held", "6", "--reserve", "80",
                "--band", "0", "1", "--out", str(out)]
    B.main()                                             # stops on any VM / evaluator disagreement
    read = lambda name: [json.loads(ln) for ln in open(out / name, encoding="utf-8")]   # noqa: E731
    return read("rows.jsonl"), read("pair_rows.jsonl"), json.loads((out / "manifest.json").read_text(encoding="utf-8")), out


def test_each_check_carries_its_said_form_in_both_spellings(built):
    rows, pairs, _, _ = built
    for r in rows + pairs:
        checks = [s for s in r["segments"] if s["kind"] == "check"]
        assert [s["part"] for s in checks] == list(range(r["n_parts"])) == list(range(len(r["said"])))
        for seg, part, said in zip(checks, r["parts"], r["said"]):
            assert seg["how"] == said["how"] and said["w"] == S.said(part, r["lang"], said["how"])
            assert said["m"] == VS.fold(said["w"]) and S.marks(said["m"]) == 0


def test_the_program_is_the_join_of_the_folded_forms_and_gives_each_reply_its_verdict(built):
    rows, pairs, _, _ = built
    for r in rows + pairs:
        m, groups = [s["m"] for s in r["said"]], [tuple(g) for g in r["groups"]]
        assert r["join"] == VJ.join(m, groups)
        for x in r["replies"]:
            views = RV.views(x["text"])
            assert VS.evaluate(VJ.spec(m, groups), views) == x["held"] == VS.evaluate(VJ.spec(r["parts"], groups), views)
        assert sum(1 for x in r["replies"] if x["held"]) >= 2 and sum(1 for x in r["replies"] if not x["held"]) >= 2


def test_a_tied_pair_has_one_tagged_check_beside_its_partner(built):
    rows, pairs, _, _ = built
    tied = 0
    for r in rows + pairs:
        checks = [s for s in r["segments"] if s["kind"] == "check"]
        tags = {s["part"]: s["join"] for s in checks if s.get("join")}
        groups = [g for g in r["groups"] if g[0] != "one"]
        assert len(tags) == len(groups)
        for kind, a, b in groups:
            i = next(x for x in (a, b) if x in tags)
            assert tags[i][0] == kind and i + (1 if tags[i][1] == "next" else -1) == (b if i == a else a)
            tied += 1
    assert tied


def test_train_rows_keep_8_replies_and_held_out_rows_mark_the_ones_added(built):
    rows, pairs, _, _ = built
    for r in rows + pairs:
        added = [x for x in r["replies"] if x.get("extra")]
        assert len(r["replies"]) - len(added) <= 8 and r["replies"][:len(r["replies"]) - len(added)] == [
            x for x in r["replies"] if not x.get("extra")]
        if r["split"] == "train":
            assert not added
        assert set(r["mutants"]) == {"parts", "parts told", "parts told alone", "groups", "groups told", "groups told alone"}
        assert r["mutants"]["parts told"] <= r["mutants"]["parts told alone"] <= r["mutants"]["parts"]
        assert all(x in (0, 1, 2) for _, got in r["reach"]["numbers"] + r["reach"]["groups"] for x in got)
    assert any(x.get("extra") for r in rows if r["split"] != "train" for x in r["replies"])
    assert {r["split"] for r in rows} == {"train", "dev", "test", "test_long"} and {r["split"] for r in pairs} == {"train"}
    assert all((5 <= r["n_parts"] <= 8) == (r["split"] == "test_long") for r in rows)


def test_twins_differ_in_one_lone_part(built):
    _, pairs, _, _ = built
    assert len(pairs) == 16
    for a, b in zip(pairs[::2], pairs[1::2]):
        assert a["pair"] == b["pair"] and a["id"][:-1] == b["id"][:-1] and a["instruction"] != b["instruction"]
        assert a["groups"] == b["groups"] and a["lang"] == b["lang"]
        other = [i for i, (p, q) in enumerate(zip(a["parts"], b["parts"])) if p != q]
        assert len(other) == 1 and ["one", other[0]] in a["groups"]
        p, q = a["parts"][other[0]], b["parts"][other[0]]
        assert p == {"not": q} or q == {"not": p}
        said = [[(s["kind"], s.get("part"), r["instruction"][s["start"]:s["end"]]) for s in r["segments"]] for r in (a, b)]
        assert len(said[0]) == len(said[1])                    # the same sentences in the same order, but the flipped one
        assert [x == y for x, y in zip(*said)].count(False) == 1
        assert next(x for x, y in zip(*said) if x != y)[:2] == ("check", other[0])


def test_no_held_out_row_has_a_train_rows_composition_and_ids_are_unique(built):
    rows, pairs, _, _ = built
    train = {B.key(r) for r in rows + pairs if r["split"] == "train"}
    assert not any(B.key(r) in train for r in rows if r["split"] != "train")
    ids = [r["id"] for r in rows + pairs]
    texts = [r["instruction"] for r in rows + pairs]
    assert len(set(ids)) == len(ids) and len(set(texts)) == len(texts)


def test_the_set_read_back_passes_its_check_and_a_spoiled_copy_does_not(built, tmp_path):
    rows, pairs, _, out = built
    got = CK.check(out)
    assert not any(got["must be 0"].values()), got["must be 0"]
    assert got["counted"]["rows"] == len(rows) + len(pairs) and got["counted"]["pairs"] == 8
    first = rows[0]["replies"][0]
    bad = [dict(rows[0], replies=[dict(first, held=not first["held"])] + rows[0]["replies"][1:])] + rows[1:]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bad),
                                         encoding="utf-8", newline="\n")
    for name in ("pair_rows.jsonl", "manifest.json"):
        shutil.copy(out / name, tmp_path / name)
    faults = CK.check(tmp_path)["must be 0"]
    assert faults["a verdict that is not the evaluator's on the folded forms"] == 1
    assert faults["a SHA-256 that is not the manifest's"] == 2          # the file's and its train lines'


def test_the_manifest_carries_counts_and_hashes_and_no_row_text(built):
    rows, pairs, man, out = built
    dump = json.dumps(man, ensure_ascii=False)
    assert not any(r["instruction"] in dump or r["replies"][0]["text"] in dump for r in rows + pairs)
    assert man["by_split"] == {"train": 36, "dev": 6, "test": 6, "test_long": 6} and man["pairs"] == 8
    for name in ("rows.jsonl", "pair_rows.jsonl"):
        assert man["sha256"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert man["vm_disagreements"] == 0 and man["vm_runs_agreeing"] == sum(len(r["replies"]) for r in rows + pairs)
    share = man["sets"]["train with its pairs"]["bounds"]["all"]["lower share"]
    assert share == man["lower bounds' share of the bounds"]["train with its pairs"]
