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


def test_qfhrr_decides_past_the_margin_on_grilly2(tmp_path):
    import shutil
    from retrieval_bench import r2_qfhrr as Q
    assert Q.decide(["Q1", "Q2"], [0.8, 0.2], 0.5)["item"] == "Q1"
    assert Q.decide(["Q1", "Q2"], [0.8, 0.7], 0.5)["outcome"] == "several"
    assert Q.decide(["Q1", "Q2"], [0.0, 0.0], 0.0)["outcome"] == "several"                     # no context, no pick
    q = "Which country is A1 motorway in canton of Bern in?"
    rows = [{"id": "r1", "mention": "A1 motorway", "question": q, "lang": "en", "cands": ["Q1", "Q2"], "pids": ["P17"]},
            {"id": "r2", "mention": "A1 motorway", "question": q, "lang": "en", "cands": ["Q1", "Q3"], "pids": ["P17"]}]
    jb, done = Q.job(rows, P, "")
    assert list(done) == ["r2"] and done["r2"]["item"] == "Q1"                                   # settled before any score
    assert jb["rows"][0]["items"] == ["Q1|en", "Q2|en"] and "bern" in jb["rows"][0]["cues"].split()
    try:
        Q.grilly2_dir()
    except SystemExit:
        pytest.skip("no grilly2 checkout here")
    if not shutil.which("uv"):
        pytest.skip("no uv here")
    from cubbyllm.worlds.fast_words import FastWordEncoder, signal_code
    table = tmp_path / "table.npz"                       # one word: every other word falls back to its signal code
    FastWordEncoder(["x"], signal_code("x", 80, 128)[None], {"x": 1.0}, 1.0, 80, 128).save_npz(table)
    jb["table"] = str(table)
    rep = Q.score(jb)
    s1, s2 = rep["scores"]["r1"]
    assert s1 > 0.5 > 0.1 > s2                   # "canton", "bern" are Q1's; Q2 shares a block by chance (80 of 128)
    assert rep["interpreter"] == "grilly2" and rep["check"]["max_abs_diff_vs_numpy"] < 1e-5


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


def test_b1q_s_cues_are_what_is_neither_the_name_nor_the_relation_nor_a_function_word():
    from retrieval_bench import r2_b1q as Q
    lx = {"by_wording": {"en": {"born": ["P19", "P569"], "place of birth": ["P19"], "country": ["P17"]}, "fr": {"né": ["P19"]}}}
    assert Q.cue_runs("Where was Paz the singer born?", "Paz", ["P19"], lx) == ["singer"]
    assert Q.cue_runs("Where was Paz born?", "Paz", ["P19"], lx) == []                  # a bare question: no qualifier
    assert "the" not in Q.cue_runs("Which country is A1 motorway in canton of Bern in?", "A1 motorway", ["P17"], lx)
    assert {"canton", "bern", "canton of bern"} - set(Q.cue_runs("Which country is A1 motorway in canton of Bern in?",
                                                                   "A1 motorway", ["P17"], lx)) == set()


def test_b1q_holds_a_qualifier_either_way():
    from retrieval_bench import r2_b1q as Q
    prof = {"value_ids": {"P106": ["Q177220"], "P19": ["Q90"]}}
    assert Q.holds("Q1", prof, {"Q177220"}, {})                                      # the candidate states it
    assert Q.holds("Q1", {"value_ids": {}}, {"Q5"}, {"Q5": {"P50": ["Q1"]}})          # the value item states the candidate
    assert not Q.holds("Q1", prof, {"Q33999"}, {"Q33999": {"P50": ["Q2"]}})


def test_b1q_takes_every_name_of_the_items_out_of_the_cues():
    from retrieval_bench import r2_b1q as Q
    lx = {"by_wording": {"en": {"country": ["P17"]}, "fr": {"pays": ["P17"]}}}
    q = "Dans quel pays se trouve autoroute A1 ?"
    assert "autoroute" in Q.cue_runs(q, "A1 motorway", ["P17"], lx)                   # the English name alone misses it
    assert Q.cue_runs(q, ["A1 motorway", "autoroute A1"], ["P17"], lx) == ["trouve"]  # every name the items go by


def test_the_name_is_one_span_and_a_qualifier_is_the_longest_name():
    from retrieval_bench import r2_b1q as Q
    lx = {"by_wording": {"en": {"country": ["P17"]}, "fr": {}}}
    assert "dhaka district" in Q.cue_runs("Which country is Dhaka in Dhaka District in?", "Dhaka", ["P17"], lx)
    assert Q.longest(["terre", "terre neuve", "terre neuve et labrador", "cap"]) == ["terre neuve et labrador", "cap"]


def test_b1q_s_rules_before_seed_1_short_names_as_written_and_the_template_s_words():
    from retrieval_bench.r2_b1q import cue_runs, short_ok, written
    from retrieval_bench.r2_choosing import template_words
    names = {"Q1553": {"NE", "Nebraska"}, "Q30": {"US", "United States"}}
    assert written("Où est né Tim Brown ?", "ne") == "né"
    assert short_ok("Où est né Tim Brown ?", "ne", ["Q1553"], names) == []                    # "né" never names "NE"
    assert short_ok("Is Rye in the US?", "us", ["Q30"], names) == ["Q30"]                     # written as named
    assert short_ok("x", "nebraska", ["Q1553"], {}) == ["Q1553"]                              # longer cues untouched
    lx = {"by_wording": {"en": {}, "fr": {}}}
    tw = template_words({"lang": "fr", "relation": "P19", "form": "bare"})
    assert tw == ["ou", "est", "ne"] and cue_runs("Où est né Tim Brown ?", "Tim Brown", ["P19"], lx, tw) == []
    q = {"lang": "en", "relation": "P17", "form": "qualified", "discriminator": {"relation": "P131"}}
    assert cue_runs("Which country is Rosebank in Western Cape in?", "Rosebank", ["P17"], lx, template_words(q)) == \
        ["cape", "western", "western cape"]


def test_b1q_walks_located_in_one_vm_hop_at_a_time(monkeypatch):
    from retrieval_bench import r2_b1q as Q
    facts = {("Q1", "~ western cape"): [], ("Q2", "~ western cape"): [], ("Q1", "P19"): ["Q9"], ("Q2", "P19"): ["Q9"],
             ("Q1", "P131"): ["Q5465"], ("Q5465", "P131"): ["Q127167"], ("Q2", "P131"): ["Q8"], ("Q8", "P131"): []}
    monkeypatch.setattr(V, "branch", lambda s, k, q, p: {"item": q, "values": facts.get((q, p), []), "wall_ms": 0.1})
    out = Q.b1q(None, "", ["Q1", "Q2"], "P19", ["western cape"], {"western cape": ["Q127167"]})
    assert out["outcome"] != "ask" and out.get("item") == "Q1"            # Cape Town is in the Western Cape; Q2 is not


def test_b1q_s_compared_rule_two_names_compared_or_chosen_are_no_cue_for_each_other():
    """Cowork's check of the seed-0 rerun (2026-11-04 night): "Does California have more electoral votes than Rhode
    Island?" -- California is the other side, not a qualifier of Rhode Island. A same-side qualifier stays; "où" is no
    choice; "ou" / "or" only between two names."""
    from retrieval_bench.r2_b1q import cue_runs
    lx = {"by_wording": {"en": {}, "fr": {}}}
    ri = cue_runs("Does California have more electoral votes than Rhode Island?", "Rhode Island", [], lx)
    assert "california" not in ri and "electoral votes" in ri
    assert "rhode island" not in cue_runs("Does California have more electoral votes than Rhode Island?", "California", [], lx)
    pa = cue_runs("Is Paris in Texas bigger than London?", "Paris", [], lx)
    assert "texas" in pa and "london" not in pa
    assert "nile" not in cue_runs("Which river is longer, the Nile or the Amazon?", "Amazon", [], lx)
    assert "paris" not in cue_runs("Lyon est-elle plus grande que Paris ?", "Lyon", [], lx)
    assert "lyon" not in cue_runs("Qui est né à Paris ou à Lyon ?", "Paris", [], lx)
    assert "france" in cue_runs("Où est la tour Eiffel en France ?", "tour Eiffel", [], lx)        # "où": no choice
    assert "france" in cue_runs("Was Tim born or raised in France?", "Tim", [], lx)              # "or" joins no names


def test_the_profile_cache_skips_a_line_an_interrupted_write_left_torn(tmp_path):
    """A run stopped mid-write left 53,802 NUL bytes as one line of r2_profiles.jsonl (2026-11-05); the next run must
    load the rest and refetch that one, not stop."""
    from retrieval_bench.r2_items import Profiles
    p = tmp_path / "profiles.jsonl"
    p.write_text('{"id": "Q1"}\n' + "\x00" * 50 + "\n" + '{"id": "Q2"}\n', encoding="utf-8")
    prof = Profiles(None, str(p))
    assert set(prof.p) == {"Q1", "Q2"} and prof.torn == 1
