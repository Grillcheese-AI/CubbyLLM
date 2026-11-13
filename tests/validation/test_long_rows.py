"""Order 0067, the second reader's step 3: long instructions built in reverse. A negated part says the same thing as
the atom it is rewritten into; French says its negations whole; the segments tile the instruction in order; parts are
numbered as the text states them; held-out phrases only in held-out rows; and the joined program's VM verdict is the
evaluator's (a small build, run)."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT)]
import build_long_rows as B  # noqa: E402
import real_rows  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VR  # noqa: E402
import view_wording as W  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402


def atoms(seed, n):
    rng, out = random.Random(seed), []
    while len(out) < n:
        reps = VR.pool(rng, 20)
        vs = [RV.views(t) for t in reps]
        a = VP.sample_atom(rng, vs)
        if a:
            out.append((a, vs))
    return out


def test_a_negated_part_equals_the_atom_it_is_said_as():
    n = 0
    for a, vs in atoms(1, 80):
        eq = B.positive({"not": a})
        if eq is None:
            continue
        for v in vs:
            assert VP.evaluate(eq, v) == (not VP.evaluate(a, v)), a
        n += 1
    assert n > 30


def test_french_negations_are_whole():
    rng = random.Random(2)
    for a, _ in atoms(2, 120):
        for s in (W.say(a, "fr", rng), W.say_not(a, "fr", rng), W.say_cond(a, "fr")):
            if re.search(r"\baucune?\b", s):
                assert re.search(r"(\bne\b|n')", s.split("aucun")[0], re.I), s


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    try:
        cc.find_cubelang_exe()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"cubelang.exe not found ({e})")
    out = tmp_path_factory.mktemp("long")
    sys.argv = ["build_long_rows.py", "--train", "12", "--held", "4", "--out", str(out)]
    B.main()                                             # stops on any VM / evaluator disagreement
    return [json.loads(ln) for ln in open(out / "rows.jsonl", encoding="utf-8")]


def test_segments_tile_the_instruction(built):
    for r in built:
        text, at = r["instruction"], 0
        for s in r["segments"]:
            assert s["start"] >= at and text[at:s["start"]].strip() == "", (text, s)
            assert text[s["start"]:s["end"]].strip()
            at = s["end"]
        assert text[at:].strip() == ""


def test_parts_are_numbered_as_the_text_states_them(built):
    for r in built:
        seen = []
        for s in r["segments"]:
            if s["kind"] == "check" and s["part"] not in seen:
                seen.append(s["part"])
        assert seen == list(range(r["n_parts"]))


def test_held_out_phrases_only_in_held_out_rows(built):
    train, held = real_rows._phrases(9)
    held_keys = {k for k, _ in held["en"] + held["fr"]}
    for r in built:
        keys = {s["asked"] for s in r["segments"] if s["kind"] == "asked"}
        if r["split"] in ("train", "dev"):
            assert not keys & held_keys
        else:
            assert keys <= held_keys


def test_parts_and_kinds(built):
    for r in built:
        lo, hi = (5, 8) if r["split"] == "test_long" else (2, 4)
        assert lo <= r["n_parts"] <= hi
        assert sum(1 for x in r["replies"] if x["held"]) >= 2 and sum(1 for x in r["replies"] if not x["held"]) >= 2
