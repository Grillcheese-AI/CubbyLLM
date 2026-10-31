"""Order 0046, R2's choosing arms on canned profiles (``validation/retrieval_bench/r2_arms.py``, ``r2_vm.py``): only an
item that bears the name is a candidate; a kind that states no value of the relation drops out; the label beats an
alias; the question's own words decide only past a rank-normalized margin; B1's table over the branches that verify
(refuse, pick, once, each, ask), and B1 through the VM itself when a cubelang binary is here."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from retrieval_bench import r2_arms as A  # noqa: E402
from retrieval_bench import r2_vm as V  # noqa: E402


def prof(q, label, desc="", aliases=(), **values):
    return {"id": q, "labels": {"en": label}, "aliases": {"en": list(aliases)}, "desc": {"en": desc},
            "values": {k: v for k, v in values.items()}}


P = {"Q1": prof("Q1", "A1 motorway", "motorway in Switzerland", P17=["Switzerland"], P131=["canton of Bern"]),
     "Q2": prof("Q2", "A1 motorway", "motorway in Austria", P17=["Austria"], P131=["Upper Austria"]),
     "Q3": prof("Q3", "A1 (band)", "a band", aliases=["A1 motorway"]),
     "Q4": prof("Q4", "Motorway A1 Serbia", "road")}


def test_only_items_bearing_the_name_and_carrying_the_relation_compete():
    assert sorted(A.bearing("A1 Motorway!", ["Q1", "Q2", "Q3", "Q4"], P)) == ["Q1", "Q2", "Q3"]
    assert A.tiers("A1 motorway", ["Q1", "Q2", "Q3", "Q4"], P, ["P17"]) == {"outcome": "several", "items": ["Q1", "Q2"]}
    assert A.tiers("A1 motorway", ["Q1", "Q3"], P, ["P17"])["item"] == "Q1"                       # the band carries no P17
    assert A.tiers("A1 motorway", ["Q1", "Q3"], P)["item"] == "Q1"                                # the label over an alias
    assert A.tiers("nothing", ["Q1"], P)["outcome"] == "refuse"


def test_a_wikimedia_meta_page_never_competes():
    page = dict(P, Q5=prof("Q5", "A1 motorway", "Wikimedia disambiguation page"))
    assert "Q5" not in A.bearing("A1 motorway", ["Q1", "Q5"], page)
    assert A.tiers("A1 motorway", ["Q1", "Q5"], page) == {"outcome": "pick", "item": "Q1", "by": "the only one"}
    assert A.tiers("A1 motorway", ["Q5"], page)["outcome"] == "refuse"


def test_the_questions_own_words_decide_only_past_the_margin():
    q = "Which country is A1 motorway in canton of Bern in?"
    got = A.bm25("A1 motorway", q, "en", ["Q1", "Q2"], P, ["P17"], tau=0.5)
    assert got["outcome"] == "pick" and got["item"] == "Q1" and got["margin"] == 1.0
    bare = A.bm25("A1 motorway", "Which country is A1 motorway in?", "en", ["Q1", "Q2"], P, ["P17"], tau=0.5)
    assert bare["outcome"] == "several"                                                            # no word separates them
    assert "motorway" not in A.cues(q, "A1 motorway") and "bern" in A.cues(q, "A1 motorway")


def test_b1_decides_over_the_branches_that_verify():
    br = lambda q, *v: {"item": q, "values": list(v), "wall_ms": 1}                               # noqa: E731
    assert V.decide([br("Q1"), br("Q2")])["outcome"] == "refuse"
    assert V.decide([br("Q1", "Switzerland"), br("Q2")]) == {"outcome": "pick", "item": "Q1", "by": "the only branch that verifies"}
    assert V.decide([br("Q1", "USA"), br("Q2", "USA")])["outcome"] == "once"
    assert V.decide([br("Q1", "a"), br("Q2", "b")])["outcome"] == "each"
    assert V.decide([br(f"Q{i}", str(i)) for i in range(4)])["outcome"] == "ask"


def test_the_gates_read_off_a_summary():
    from retrieval_bench.r2_gates import budget, gates
    s = {"fame/bare": {"rows": 4, "x": {"held": 3, "single": 1}},
         "fame/qualified": {"rows": 10, "x": {"right": 6, "wrong": 1, "undecided": 3}},
         "shadowlink/Top": {"rows": 4, "x": {"right": 3, "wrong": 1}},
         "shadowlink/Shadow": {"rows": 4, "x": {"right": 1, "wrong": 1, "refused": 2}},
         "mintaka": {"rows": 50, "x": {"right": 9, "wrong": 1, "undecided": 40}}}
    g = gates(s)["x"]
    assert g["bare_single"] == 1 and g["qualified"] == {"wrong": 1, "right": 6, "resolved": 0.7}
    assert g["shadowlink"] == {"top": 0.75, "shadow": 0.5, "gap_points": 25.0}
    assert g["mintaka"]["wrong_per_row"] == 0.02 and g["mintaka"]["decided"] == 10
    rows = [{"arms": {"b1": {"k": k, "wall_ms": w}}} for k, w in [(1, 5), (2, 9), (2, 11), (7, 40)]]
    assert budget(rows) == {"1": {"n": 1, "p50_ms": 5, "p95_ms": 5}, "2": {"n": 2, "p50_ms": 11, "p95_ms": 11},
                            "6+": {"n": 1, "p50_ms": 40, "p95_ms": 40}}


def test_b1_through_the_vm(tmp_path):
    from cubbyllm.bridges.cubelang_client import CubelangNotFound, CubelangSession, find_cubelang_exe
    try:
        find_cubelang_exe()
    except CubelangNotFound:
        pytest.skip("no cubelang binary here")
    seg = tmp_path / "seg.jsonl"
    assert V.write_segment(str(seg), P, ["P17"]) == 2
    with CubelangSession() as s:
        assert V.b1(s, str(seg), ["Q1", "Q2"], "P17")["outcome"] == "each"
        assert V.b1(s, str(seg), ["Q1", "Q3"], "P17")["item"] == "Q1"
