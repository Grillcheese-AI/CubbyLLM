"""Order 0070, the missing-fact guard (``validation/retrieval_bench/r3g_*.py``): a hidden-fact row never carries its gold
(nor the gold's inverse); each guard refuses when its condition holds and passes the arm's list unchanged otherwise, on
made-up items; the split is the committed one, pinned by its hash and re-derived from its strata by seed 0."""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_bench import r3g_guards as G  # noqa: E402
from retrieval_bench import r3g_set as S  # noqa: E402

SPLIT_SHA256 = "4bef4d41d3258d14a1e4e4b8222a4367687fe0a74f775ea4b40d5be179cecb5a"
LIST = {"read": True, "top": ["P569", "P27"], "undecided": False}


def test_a_hidden_row_never_carries_its_gold_nor_the_gold_s_inverse():
    inv = {"P40": ["P22", "P25"], "P26": ["P26"]}
    assert S.hide({"P40", "P22", "P569", "P27"}, ["P40"], inv) == {"P569", "P27"}
    assert S.hide({"P26", "P19"}, ["P26"], inv) == {"P19"}
    rng = random.Random(0)
    pool = [f"P{i}" for i in range(1, 40)]
    for _ in range(500):
        cands, golds = set(rng.sample(pool, 12)), rng.sample(pool, rng.randint(1, 3))
        inv = {g: rng.sample(pool, 2) for g in golds}
        out = S.hide(cands, golds, inv)
        assert not out & set(golds) and not out & {i for g in golds for i in inv[g]} and out <= cands


def test_truthy_reads_the_best_rank_and_never_a_deprecated_value():
    def st(q, rank):
        return {"rank": rank, "mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": q}}}}
    assert S.truthy({"P1696": [st("P22", "normal"), st("P25", "preferred"), st("P9", "deprecated")]}, "P1696") == ["P25"]
    assert S.truthy({"P1696": [st("P22", "normal"), st("P9", "deprecated")]}, "P1696") == ["P22"]
    assert S.truthy({}, "P1696") == []


def test_g1_refuses_when_an_absent_property_beats_the_items_best_by_the_margin():
    scores = {"P20": 0.8, "P19": 0.5, "P569": 0.4}
    absent, own = G.bests(scores, {"P569"})
    assert (absent, own) == (0.8, 0.4)
    assert G.g1(absent, own, 0.5) and not G.g1(absent, own, 0.51)
    assert not G.g1(*G.bests(scores, {"P20", "P569"}), 0.0)            # the items carry the best: through
    assert not G.g1(*G.bests({"P20": 0.5, "P569": 0.5}, {"P569"}), 0.0)  # a tie does not beat
    assert G.apply(LIST, False) is LIST and G.apply(LIST, True) == G.REFUSED


def test_g2_refuses_when_the_best_run_holds_an_expected_absent_property_and_never_when_unavailable():
    assert G.g2({"P19": 1, "P569": 1}, {"P19"})                         # "born": place or date, the place absent
    assert not G.g2({"P19": 1, "P569": 2}, {"P19"})
    assert not G.g2({"P19": 1, "P569": 1}, None) and not G.g2({"P19": 1}, set())
    assert not G.g2({"P19": 0.0}, {"P19"})                             # nothing scores: nothing to refuse


def test_g3_keeps_what_the_lexicon_or_grammar_names_and_refuses_an_empty_list():
    assert G.g3(LIST, {"P27", "P31"}) == {"read": True, "top": ["P27"], "undecided": False}
    assert G.g3(LIST, {"P569", "P27"}) == LIST
    assert G.g3(LIST, {"P31"}) == G.REFUSED
    unread = {"read": False, "top": [], "undecided": False}
    assert G.g3(unread, set()) is unread


def test_choose_follows_the_preregistered_rule():
    pts = [{"t": {"m": 0.0}, "handovers": 5, "loss": 0.05, "sq_ok": True},
           {"t": {"m": 0.2}, "handovers": 9, "loss": 0.01, "sq_ok": True},
           {"t": {"m": 0.3}, "handovers": 9, "loss": 0.01, "sq_ok": True},
           {"t": {"m": 0.4}, "handovers": 8, "loss": 0.0, "sq_ok": False}]
    got = G.choose(pts)
    assert got["t"] == {"m": 0.3} and got["flag"] is None
    worse = [{"t": {"m": 0.0}, "handovers": 5, "loss": 0.05, "sq_ok": True},
             {"t": {"m": 0.5}, "handovers": 9, "loss": 0.04, "sq_ok": True}]
    assert G.choose(worse)["t"] == {"m": 0.5} and G.choose(worse)["flag"]


def test_the_split_is_the_committed_one():
    d = S.load_split()
    assert d["sha256"] == SPLIT_SHA256 and d["seed"] == 0 and d["rows"] == 870
    strata = {k: v["guard-dev"] + v["guard-test"] for k, v in d["strata"].items()}
    assert S.split(strata, 0) == d["strata"]
    for v in d["strata"].values():
        assert len(v["guard-test"]) - len(v["guard-dev"]) in (0, 1) and not set(v["guard-dev"]) & set(v["guard-test"])
