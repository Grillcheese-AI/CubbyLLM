"""Order 0034's runner (validation/exp_s12_scene_files.py, s12_questions.py) on a small village.

Pinned:
- The 30 questions follow the plan, all different, the same for the same seed, each answered from the exact reading;
  a placed question carries the cell it is about.
- The host, AskLoop(scene=), speaks the scene's relations (position, a neighbour, the nearest of a kind, the region)
  only verified and right. It refuses what its grammar does not route (n steps away, a cell, "where is").
- The trunk's documents hold every question once, each placed question's `here` on its cell (the real tokenizer;
  skipped without it).
"""
from __future__ import annotations

import collections
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation", ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.spatial.scene import Scene  # noqa: E402
from s12_questions import PLACED, PLAN, questions  # noqa: E402


def village() -> Scene:
    sc = Scene()
    for name, cell, kind in (("well", (5, 5, 0), "water"), ("oak", (6, 5, 0), "tree"), ("forge", (5, 7, 0), "building"),
                             ("hut", (9, 5, 0), "building"), ("lantern", (5, 5, 1), "light"), ("ada brook", (7, 5, 0), "villager"),
                             ("tom reed", (2, 3, 0), "villager"), ("lina moss", (5, 6, 0), "villager"),
                             ("hugo vale", (11, 12, 0), "villager"), ("barn", (12, 12, 0), "building")):
        sc.place(name, cell, kind)
    sc.region("the square", (3, 3, 0), (8, 8, 1))
    return sc


def test_the_questions_follow_the_plan_and_each_is_answered_from_the_exact_reading():
    sc = village()
    qs = questions(sc, 7)
    assert qs == questions(sc, 7) and len(qs) == 30 and len({q["question"] for q in qs}) == 30
    assert collections.Counter(q["shape"] for q in qs) == dict(PLAN)
    for q in qs:
        if q["shape"] in PLACED:
            cell = tuple(q["address"])
            assert (sc.at(cell) or "nothing") == q["answer"]
            assert q["cls"] == ("nothing" if sc.at(cell) is None else "name" if sc.kinds[sc.at(cell)] == "villager" else "kind")
        elif q["shape"] in ("position", "where"):
            assert q["answer"] == " ".join(map(str, sc.where(q["subject"]))) and q["address"] is None
    assert all(sc.kinds[q["subject"]] == "villager" for q in qs if q["shape"] == "where")


def test_the_host_routes_every_shape_speaks_every_answer_right_and_refuses_every_empty_one():
    """Since order 0035 the three shapes 0034 left unrouted (n steps away, at a cell, where is X) reach the scene too."""
    from exp_s12_scene_files import host
    from test_books import exact_vm
    sc = village()
    qs = questions(sc, 7)
    rows = host(sc, qs, exact_vm())
    assert sum(r["wrong"] for r in rows) == 0
    for q, r in zip(qs, rows):
        assert r["routed"], q["question"]
        assert (r["spoken"] and r["right"]) if q["answer"] != "nothing" else not r["spoken"], q["question"]
    assert {q["shape"] for q, r in zip(qs, rows) if r["spoken"]} == {s for s, _k in PLAN}


def test_the_trunks_documents_hold_every_question_once_with_each_here_on_its_cell():
    from exp_s12_scene_files import SPM, trunk_docs
    if not pathlib.Path(SPM).exists():
        pytest.skip("no tokenizer on this machine")
    from tokenizers import Tokenizer
    from cubbyllm.model.spatial import NO_PLACE
    tk = Tokenizer.from_file(SPM)
    sc = village()
    placed = [q for q in questions(sc, 7) if q["shape"] in PLACED]
    docs = trunk_docs(tk, sc, placed, True, tk.token_to_id("</s>"))
    assert [q["question"] for d in docs for q in d[3]] == [q["question"] for q in placed]
    here = tk.encode("here", add_special_tokens=False).ids
    for ids, places, roles, chunk in docs:
        assert len(ids) == 512
        asked = [tuple(int(v) for v in places[i]) for i in range(len(ids)) if roles[i] & 2]
        assert asked == [tuple(q["address"]) for q in chunk for _ in here]
        assert all(places[i][0] != NO_PLACE for i in range(len(ids)) if roles[i])


def test_the_act_counts_never_take_a_wrong_cell_for_a_placement_or_a_refusal(tmp_path):
    import json
    from s12_act import counts
    rows = [{"seed": 1, "drawn": "placeable", "cell": [1, 0, 0], "expected": [1, 0, 0], "host_cell_is_expected": True,
             "verified": True, "sent": True, "ok": True, "stands": True, "elsewhere": 0},
            {"seed": 1, "drawn": "placeable", "cell": [2, 0, 0], "expected": [3, 0, 0], "host_cell_is_expected": False,
             "verified": True, "sent": True, "ok": True, "stands": True, "elsewhere": 0},          # a lamp, but not where meant
            {"seed": 2, "drawn": "unplaceable", "cell": [5, 5, 5], "expected": [5, 5, 5], "host_cell_is_expected": True,
             "verified": True, "sent": False, "refused": "nothing under it"}]
    p = tmp_path / "act_log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    c = counts(str(p))
    assert c["placeable"] == 2 and c["placed_on_the_expected_cell"] == 1 and c["host_cell_not_expected"] == 1
    assert c["unplaceable_refused_before_the_order"] == 1 and c["refused_why"] == {"nothing under it": 1}
    assert c["villages"] == 2 and c["placed_elsewhere"] == 0
