"""Order 0077, builds 1 and 2 (validation/builders/real_prompts_0077.py, build_rows_0077.py).

Pinned without data: round 6's split is kept (held-out prompts only in the test sets, French held-out prompts in
none, no Dolly prompt in train or dev, no prompt twice, the 600-character bound); a real prompt put around composed
checks gives one task segment a prompt line, and the reader's answer for the row reads back strictly (the spans tile
the instruction). With the off-repo prompts and the VM: a small build reads back on both arms, its train rows hold
no Dolly text, and every set's rows come from its own prompts."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation"), str(ROOT)]
import build_rows_0077 as B  # noqa: E402
import real_prompts_0077 as RP  # noqa: E402

from cubbyllm.speech import view_split as V  # noqa: E402


def _fake():
    splits, pool, rng = {}, {}, random.Random(0)
    for i in range(400):
        src = "dolly" if i % 4 == 0 else "oasst1"
        split = ("train", "dev", "held")[i % 3]
        lang = "fr" if src == "oasst1" and i % 5 == 0 else "en"
        sid = f"{src}:{i}"
        splits[sid] = (split, lang)
        n = 700 if i % 17 == 0 else rng.randint(30, 590)
        pool[sid] = ("Line one of a prompt.\nAnd a second line. " * 40)[:n]
    return splits, pool


def test_round_6s_split_is_kept_and_no_prompt_is_used_twice():
    splits, pool = _fake()
    got, counts = RP.sets(77, 20, 10, 10, None, splits, pool)
    seen = [sid for v in got.values() for sid, _, _ in v]
    assert len(seen) == len(set(seen))
    for name, v in got.items():
        for sid, lang, p in v:
            assert len(p) <= RP.MAX_CHARS and p == pool[sid]
            split, l2 = splits[sid]
            assert lang == l2
            if name in ("train_real", "dev_real"):
                assert sid.startswith("oasst1:") and split == {"train_real": "train", "dev_real": "dev"}[name]
            else:
                assert split == "held" and lang == "en"
                assert sid.startswith("oasst1:" if name == "test_real" else "dolly:")
    assert counts["French held-out prompts of round 6 kept out"] > 0
    again, _ = RP.sets(77, 20, 10, 10, None, splits, pool)
    assert again == got                                                 # seeded


COMPOSED = 'Write fewer than 10 sentences. Do not use the word "blue".'
SEGS = [{"kind": "check", "part": 0, "how": "x", "start": 0, "end": 30},
        {"kind": "check", "part": 1, "how": "x", "start": 31, "end": len(COMPOSED)}]


@pytest.mark.parametrize("where", B.PLACES)
def test_a_prompt_put_around_the_checks_is_one_task_segment_a_line_and_tiles(where):
    prompt = "I need help with a letter. It is for my landlord.\n\nThe heating has been off for a week.\nWhat do I say?"
    text, segs = B.place(random.Random(3), prompt, COMPOSED, SEGS, where)
    tasks = [text[s["start"]:s["end"]] for s in segs if s["kind"] == "task"]
    lines = [x for x in prompt.split("\n") if x.strip()]
    if where == "around":
        assert tasks[0] == "I need help with a letter." and tasks[1:] == ["It is for my landlord."] + lines[1:]
    else:
        assert tasks == lines
    checks = [text[s["start"]:s["end"]] for s in segs if s["kind"] == "check"]
    assert checks == [COMPOSED[:30], COMPOSED[31:]]
    row = {"instruction": text, "segments": segs,
           "said": [{"m": {"what": "x"}, "w": {"what": "x"}}] * 2}
    lines_out = [ln for ln in V.target(dict(row, segments=[s for s in segs if s["kind"] == "task"]), "m").split("\n")[1:]]
    assert all(ln.startswith("task | ") and "\n" not in ln for ln in lines_out)
    ends = sorted((s["start"], s["end"]) for s in segs)
    assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))            # in order, no overlap
    covered = set().union(*(range(a, b) for a, b in ends))
    assert all(text[i].isspace() for i in range(len(text)) if i not in covered)   # nothing left out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from cubbyllm.bridges import cubelang_client as cc
    try:
        cc.find_cubelang_exe()
        RP.round6_splits()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"the VM or round 6's rows are not here ({e})")
    out = tmp_path_factory.mktemp("rows0077")
    sys.argv = ["build_rows_0077.py", "--train", "7", "--dev", "7", "--test", "7", "--reserve", "80",
                "--band", "0", "1", "--out", str(out)]
    B.main()
    rows = [json.loads(ln) for ln in open(out / "rows.jsonl", encoding="utf-8")]
    return rows, json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_a_small_build_reads_back_and_keeps_its_sources_apart(built):
    rows, man = built
    assert man["by_set"] == {s: 7 for s in B.SETS}
    splits = RP.round6_splits()
    for r in rows:
        assert r["prompt_chars"] <= RP.MAX_CHARS
        for arm in "mw":
            got = V.parse(V.target(r, arm), r["instruction"], arm)
            assert "error" not in got, (r["id"], got)
            assert [x["kind"] for x in got["lines"]] == [s["kind"] for s in r["segments"]]
        split, lang = splits[r["source_id"]]
        if r["split"] in ("train_real", "dev_real"):
            assert r["source"] == "oasst1" and split != "held"
        else:
            assert split == "held" and lang == "en"
    assert not any(r["source"] == "dolly" for r in rows if r["split"] == "train_real")
    ids = [r["source_id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert "instruction" not in json.dumps(man) or all(r["instruction"] not in json.dumps(man) for r in rows)
