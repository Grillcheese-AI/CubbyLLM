"""Order 0037 -- a shared name decided by the question's relation and its cue words, on a synthetic hdc world:
four sovereign states (each with a capital), ten towns (one with a capital), six people (citizenships, awards: two
Nobels), five films (cast, awards), a constituency. Namesakes: "Fredonia" (a state, a constituency, an item whose only
fact is its country), "Ann Lee" (a person, and an item with only a citizenship), "Mara Vale" (a person and a film, both
awarded), "Elm" (a state and a town).

Pinned:
- a kind carries a relation by its share against the typed items' base, not "any member" (one town in ten with a
  capital does not make towns carry one);
- an item that records no kind is read from its own specific relations (a citizenship reads as a person; "country",
  which many kinds carry, reads as nothing, and such an item does not compete);
- one item that fits is read, the others are named by their kind (never the one of no kind); two of a kind are both
  noted, none read; a cue word rules out a kind it never goes with, a word the world never uses rules out nothing;
- off (no ``CB_NAMESAKE_RELATION``), strict as order 0036 left it.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from cubbyllm.reasoning.planner import QuestionPlan  # noqa: E402
from cubbyllm.worlds import hdc_cues, hdc_kinds  # noqa: E402
from cubbyllm.worlds.hdc_relation import others  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld  # noqa: E402
from cubbyllm.worlds.hdc_world import write_world  # noqa: E402

KIND, CIT, OCC, AWARD, CAST, LOC = ("instance of", "country of citizenship", "occupation", "award received", "cast member",
                                    "located in the administrative territorial entity")
PIDS = {KIND: "P31", "capital": "P36", "country": "P17", CIT: "P27", OCC: "P106", AWARD: "P166", CAST: "P161", LOC: "P131"}


def _world():
    labels, facts, n = {}, {}, [1000]

    def item(label):
        n[0] += 1
        q = f"Q{n[0]}"
        labels[q] = (label, "")
        return q

    def fact(s, rel, o):
        facts[(s, rel, o)] = PIDS[rel]

    kinds = {k: item(k) for k in ("sovereign state", "town", "human", "film", "constituency", "rock")}
    for i in range(30):                                  # typed items of a kind that carries nothing asked here: in the
        fact(item(f"Rock {i}"), KIND, kinds["rock"])     #   real world each kind is a small share of the typed items
    land, cap, nobel, medal, bear, actor = (item(x) for x in ("Ruritania Land", "Strelsau", "Nobel Prize in Physics",
                                                              "Elliott Medal", "Golden Bear", "Some Actor"))
    for i in range(4):
        s = item(f"State {i}")
        fact(s, KIND, kinds["sovereign state"]); fact(s, "capital", item(f"Capital {i}")); fact(s, "country", land)
    for i in range(10):
        t = item(f"Town {i}")
        fact(t, KIND, kinds["town"]); fact(t, LOC, land); fact(t, "country", land)
        if i == 0:
            fact(t, "capital", cap)                                  # one town in ten: towns do not carry a capital
    for i in range(6):
        p = item(f"Person {i}")
        fact(p, KIND, kinds["human"]); fact(p, CIT, land)
        if i < 4:
            fact(p, AWARD, nobel if i < 2 else medal)
    for i in range(5):
        f = item(f"Film {i}")
        fact(f, KIND, kinds["film"]); fact(f, CAST, actor); fact(f, "country", land)
        if i < 3:
            fact(f, AWARD, bear)
    fr_state, fr_const, fr_bare = item("Fredonia"), item("Fredonia"), item("Fredonia")
    fact(fr_state, KIND, kinds["sovereign state"]); fact(fr_state, "capital", item("Freedonia City")); fact(fr_state, "country", land)
    fact(fr_const, KIND, kinds["constituency"]); fact(fr_bare, "country", land)
    ann1, ann2 = item("Ann Lee"), item("Ann Lee")
    fact(ann1, KIND, kinds["human"]); fact(ann1, CIT, land); fact(ann2, CIT, land)
    mara_p, mara_f = item("Mara Vale"), item("Mara Vale")
    fact(mara_p, KIND, kinds["human"]); fact(mara_p, AWARD, medal)
    fact(mara_f, KIND, kinds["film"]); fact(mara_f, AWARD, bear)
    elm_s, elm_t = item("Elm"), item("Elm")
    fact(elm_s, KIND, kinds["sovereign state"]); fact(elm_s, "capital", item("Elm City")); fact(elm_t, KIND, kinds["town"])
    max_b, max_p = item("Max Roe"), item("Max Roe")                  # two of a kind a fact tells apart
    for q, job, home in ((max_b, "baseball player", "Ruritania Land"), (max_p, "politician", "Grand Fenwick")):
        fact(q, KIND, kinds["human"]); fact(q, OCC, item(job)); fact(q, CIT, item(home) if home != "Ruritania Land" else land)
    return facts, labels, {"fr": fr_state, "const": fr_const, "bare": fr_bare, "ann1": ann1, "ann2": ann2,
                           "mara_p": mara_p, "mara_f": mara_f, "elm_s": elm_s, "elm_t": elm_t, "max_b": max_b, "max_p": max_p}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(hdc_kinds, "SPREAD", 2)        # (a small world: "country", three kinds carry it, is generic here)
    facts, labels, ids = _world()
    db = tmp_path / "hdc_world.sqlite"
    write_world(db, facts, labels, {}, PIDS)
    hdc_kinds.build(str(db))
    hdc_cues.build(str(db), hdc_kinds.for_world(str(db)))
    w = HdcWorld(db)
    assert w._kinds is not None and w._cues is not None
    return w, ids


def hop(w, rel, name, cues=()):
    w.clear_withheld()
    with w.cued(cues):
        got = w.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)
    return [t.obj for _f, t in got], w.decision(name), w.withheld(name)


def test_a_kind_carries_a_relation_by_its_share_not_by_any_member(world):
    w, _ids = world
    assert w._kinds.carries("sovereign state", "capital") and not w._kinds.carries("town", "capital")
    assert w._kinds.applies("town", "capital")                    # (0036's "any member" said yes)


def test_an_item_of_no_recorded_kind_is_read_from_its_specific_relations_only(world):
    w, _ids = world
    assert w._kinds.read({CIT}) == {"human"}
    assert w._kinds.read({"country"}) == set()                    # many kinds carry it: it says only "somewhere"


def test_one_item_fits_its_value_is_read_and_the_others_are_named_by_their_kind(world, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    w, ids = world
    reads, d, held = hop(w, "capital", "Fredonia")
    assert reads == ["Freedonia City"] and d["fit"] == [ids["fr"]] and held == []
    named = others(d, w.namesakes("Fredonia"), w._kinds)
    assert [o["item"] for o in named] == [ids["const"]] and named[0]["kind"] == "constituency"   # (not the bare one)
    reads, d, _ = hop(w, "capital", "Elm")                         # a town's kind does not carry a capital
    assert reads == ["Elm City"] and d["fit"] == [ids["elm_s"]]


def test_two_of_a_kind_both_fit_and_neither_is_read(world, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    w, ids = world
    reads, d, held = hop(w, CIT, "Ann Lee")
    assert reads == [] and d["fit"] == [ids["ann1"], ids["ann2"]] and held == [ids["ann1"], ids["ann2"]]
    assert d["kinds"][ids["ann2"]] == [["human"], "read"]


def test_a_cue_word_rules_out_a_kind_it_never_goes_with_and_an_unknown_word_nothing(world, monkeypatch):
    monkeypatch.setenv("CB_NAMESAKE_RELATION", "1")
    w, ids = world
    reads, d, _ = hop(w, AWARD, "Mara Vale")
    assert reads == [] and d["fit"] == [ids["mara_p"], ids["mara_f"]]          # a person and a film: both awarded
    reads, d, _ = hop(w, AWARD, "Mara Vale", cues=["nobel"])
    assert d["fit"] == [ids["mara_p"]] and d["ruled_out"] == {ids["mara_f"]: "nobel"} and reads == ["Elliott Medal"]
    reads, d, _ = hop(w, AWARD, "Mara Vale", cues=["zyzzyva"])
    assert d["fit"] == [ids["mara_p"], ids["mara_f"]] and d["cues"] == []


def test_off_the_hop_is_strict_as_order_0036_left_it(world, monkeypatch):
    monkeypatch.delenv("CB_NAMESAKE_RELATION", raising=False)
    w, ids = world
    reads, d, held = hop(w, "capital", "Fredonia")
    assert reads == [] and d is None and held == [ids["fr"]]
