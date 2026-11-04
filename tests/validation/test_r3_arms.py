"""Order 0046, R3's reading arms (``validation/retrieval_bench/r3_*.py``): the gold property rules; the names taken out
before the lexicon reads; the list kept within 5 with a run of equal scores whole; the qFHRR arm read only past its
margin; the 450M's prompt carries the name as a slot and its relation is the program's HOP1."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_bench import r3_450m as M  # noqa: E402
from retrieval_bench import r3_arms as A  # noqa: E402
from retrieval_bench import r3_gold as G  # noqa: E402
from retrieval_bench import r3_qfhrr as Q  # noqa: E402


def test_gold_properties_follow_the_written_rules():
    assert G.gold({"set": "simplequestions_wd", "prop": "P19"}) == (["P19"], "gold")
    q = lambda *ps: {"set": "qald9plus", "extra": {"props": list(ps)}}                                  # noqa: E731
    assert G.gold(q("P31", "P36")) == (["P36"], "gold")                                                # a class constraint is not the relation
    assert G.gold(q("P31"))[1] == "no relation" and G.gold(q("P26", "P19"))[1] == "several relations"
    claims = {"Q1": {"P36": ["Q2"]}, "Q3": {"P50": ["Q1"]}}
    assert G.links(["Q1"], ["Q2", "Q3"], claims) == ["P36", "P50"]                                     # either way
    row = {"set": "mintaka", "mentions": [{"qid": "Q1"}], "answer_qids": ["Q9"]}
    assert G.gold(row, claims) == ([], "no link")


def test_the_list_keeps_a_run_of_equal_scores_whole_within_five():
    assert A.top({"P1": 2, "P2": 1, "P3": 1}) == {"top": ["P1", "P2", "P3"], "undecided": False}
    assert A.top({"P1": 3, **{f"P{i}": 1 for i in range(2, 8)}}) == {"top": ["P1"], "undecided": False}
    assert A.top({f"P{i}": 1 for i in range(1, 7)}) == {"top": [], "undecided": True}
    assert A.judge({"read": True, "top": [], "undecided": True}, ["P1"]) == "undecided"
    assert A.judge({"read": True, "top": ["P2"]}, ["P1"]) == "wrong" and A.judge(None, ["P1"]) == "unread"


def test_the_lexicon_reads_what_is_left_once_the_name_is_out():
    lx = {"by_wording": {"en": {"capital": ["P36"], "capital of": ["P1376"], "in": ["P131"], "country": ["P17", "P27"]}}}
    q = "What is the capital of France in?"
    text = A.leftover(q, [[23, 29]])
    assert "France" not in text
    got = A.lexicon(text, "en", lx)
    assert got["wordings"] == ["capital of", "in"] and got["top"] == ["P1376", "P131"]               # longest first, scored by length
    assert A.lexicon(text, "en", lx, content_only=True)["top"] == ["P1376"]                          # "in" is a function word
    assert A.lexicon("Which country?", "fr", lx)["read"] is False                                    # the row's language only


def test_qfhrr_reads_only_past_the_margin():
    assert Q.decide([["P1", 0.8], ["P2", 0.2], ["P3", 0.1]], 0.5)["top"] == ["P1", "P2", "P3"]
    assert Q.decide([["P1", 0.8], ["P2", 0.7]], 0.5)["read"] is False
    assert Q.decide([["P1", 0.0], ["P2", 0.0]], 0.0)["read"] is False
    lx = {"datatype": {"P17": "wikibase-item", "P9": "string"},
          "by_wording": {"en": {"country": ["P17"], "of the": ["P9"]}, "fr": {"pays": ["P17"]}}}
    assert Q.codebook(lx) == {"P17|en": "country", "P17|fr": "pays"}                                 # function words out


def test_the_450m_prompt_slots_the_names_and_its_relation_is_hop1():
    ms = [{"qid": "Q90", "span": [23, 28]}, {"qid": "Q90", "span": [33, 38]}]
    assert M.annotate("What is the capital of Paris and Paris?", ms) == \
        "What is the capital of [E1: Paris] and [E1: Paris]?"
    prog = 'bind frame, SEED, "$E1";\n        bind frame, HOP1, "official language";\n bind frame, HOP2, "x";'
    assert M.read(prog) == {"hop1": "official language", "hops": 2}
    assert M.read("http://www.example.com") == {"hop1": None, "hops": 0}


def test_r3o_hands_over_only_properties_the_item_carries():
    from retrieval_bench import r3o as O
    from retrieval_bench.r3o_props import candidates
    props = {"Q1": {"out": ["P19", "P27"], "in": ["P50"]}, "Q2": {"out": ["P17"], "in": None}}
    c = candidates(["Q1", "Q2"], props)
    assert c["all"] == {"P17", "P19", "P27", "P50"} and c["in_unknown"] is True         # an unanswered query is unknown, not none
    found = [("born", 1, ["P19", "P569"]), ("country", 1, ["P17", "P27", "P495"])]
    got = O.within(found, {"P19", "P27"})
    assert got["top"] == ["P19", "P27"] and "P569" not in got["top"]                   # date of birth is not this item's
    assert O.within(found, {"P40"})["read"] is False
    assert O.within([], {"P19"}, {"P19": O.WHOLE})["top"] == ["P19"]                   # the 450M's whole wording
