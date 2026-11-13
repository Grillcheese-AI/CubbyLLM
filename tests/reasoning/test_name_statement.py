"""Order 0054, the certificate's name statement (Cowork's shape, 2026-11-11, and her read of the first build): the item
is the one the world handed facts for, by id; the check can fail (one tampered statement per check); a pick says
what it compared and what chose last; the branch with no store; the refusals; the live source. The store is
test_wd_store_names's fixture world."""
from __future__ import annotations

import copy
import importlib.util
import pathlib
from types import SimpleNamespace

from cubbyllm.reasoning import name_statement as ns
from cubbyllm.reasoning.planner import QuestionPlan

_spec = importlib.util.spec_from_file_location("_wd_names2", pathlib.Path(__file__).parents[1] / "worlds" / "test_wd_store_names.py")
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
world = N.world                                         # the fixture
N.T.ITEMS.update({"Q8090": N.item("Echo Tale", *N.of("Q8002"), N.st(50, {"entity": "Q8020"})),
                  "Q8091": N.item("Echo Tale", *N.of("Q8002"), N.st(50, {"entity": "Q8020"}))})
N.T.NAMES["echo tale"] = [("Q8090", 0, "exact"), ("Q8091", 0, "exact")]


def loop_over(w):
    return SimpleNamespace(world=SimpleNamespace(held_by=lambda f: w, others=[w]), source=None)


def walk(w, rel, name):
    """A first hop on `name`, as the loop makes it; its first fact."""
    N.fresh(w)
    got = w.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)
    return got[0][0] if got else None


def steps(st):
    return [s["rule"] for s in st["steps"]]


def test_an_alias_and_a_fold_carry_the_worlds_own_string(world):
    fact = walk(world, "place of death", "emperor constantine")
    st = ns.build(loop_over(world), {"seed": "Emperor Constantine", "trace": [fact]})
    assert st["item"]["id"] == "Q8040" and st["item"]["label"] == "Constantine II" and st["first_fact"]["subject_ids"] == ["Q8040"]
    assert st["steps"][0] == {"rule": "alias", "alias": "Emperor Constantine", "lang": "en"}
    assert ns.ground(st) == "re-derived" and ns.klass(st) == "store statement"
    fact = walk(world, "author", "gospel of john")
    st = ns.build(loop_over(world), {"seed": "gospel of john", "trace": [fact]})
    assert st["item"]["id"] == "Q8010" and steps(st) == ["fold"] and st["steps"][0]["equal"] == ["gospel of john", "Gospel of John"]
    assert ns.ground(st) == "re-derived" and ns.klass(st) == "checked"


def test_the_item_is_the_one_the_rule_kept_never_matched_by_label(world):
    fact = walk(world, "author", "hymn of the land")          # two items labelled alike; the rule keeps the score
    st = ns.build(loop_over(world), {"seed": "hymn of the land", "trace": [fact]})
    assert st["item"]["id"] == "Q8031" and st["item"]["how"] == "kept one"
    pick = next(s for s in st["steps"] if s["rule"] == "namesake")
    assert pick["kept"] == ["Q8031"] and "author" in pick["compared"]["relations"]
    assert ns.ground(st) == "stated only" and ns.klass(st) == "taken as stated"
    assert {"id": "Q8030", "label": "Hymn of the Land", "dropped_by": "namesake"} in st["others"]["listed"]


def test_several_kept_the_last_choice_is_a_step_of_its_own(world):
    fact = walk(world, "author", "echo tale")                 # both kept, both say the same: the first in id order
    st = ns.build(loop_over(world), {"seed": "echo tale", "trace": [fact]})
    last = next(s for s in st["steps"] if s["rule"] == "first_by_id")
    assert last == {"rule": "first_by_id", "kept": ["Q8090", "Q8091"], "chose": "Q8090", "how": "first_by_id"}
    assert st["item"]["id"] == "Q8090" and ns.ground(st) == "stated only"


def test_each_check_catches_its_tampered_statement(world):
    fact = walk(world, "author", "hymn of the land")
    st = ns.build(loop_over(world), {"seed": "hymn of the land", "trace": [fact]})
    bad = copy.deepcopy(st)
    bad["item"]["id"] = "Q8030"                                # an item outside the kept list
    assert ns.ground(bad) == "not the same"
    fact = walk(world, "author", "gospel of john")
    st = ns.build(loop_over(world), {"seed": "gospel of john", "trace": [fact]})
    bad = copy.deepcopy(st)
    bad["first_fact"]["subject_ids"] = ["Q8011"]               # the item is not the first fact's subject (by id)
    assert ns.ground(bad) == "not the same"
    bad = copy.deepcopy(st)
    bad["first_fact"]["subject_ids"], bad["item"]["label"] = [], "The Gospel According to John"   # (by label)
    assert ns.ground(bad) == "not the same"
    bad = copy.deepcopy(st)
    bad["steps"] = [{"rule": "label", "label": "Gospel of Luke"}]   # a label that is not the asked name
    assert ns.ground(bad) == "not the same"
    fact = walk(world, "place of death", "emperor constantine")
    st = ns.build(loop_over(world), {"seed": "Emperor Constantine", "trace": [fact]})
    bad = copy.deepcopy(st)
    bad["steps"][0]["alias"] = "Constantine the Great"         # an alias string that is not the asked name
    assert ns.ground(bad) == "not the same"
    bad["steps"][0]["alias"] = None                            # composed from the asked name: nothing compared
    assert ns.ground(bad) == "stated only"


def test_a_set_aside_original_and_a_role_are_picks_stated_only(world, monkeypatch):
    monkeypatch.setenv("CB_ORIGINAL_FIRST", "1")
    fact = walk(world, "author", "hymn of the land")
    st = ns.build(loop_over(world), {"seed": "Hymn of the Land", "trace": [fact or "Ada is the author of Hymn of the Land"]})
    aside = [s for s in st["steps"] if s["rule"] == "original_first"]
    assert aside and aside[0]["set_aside"] == "Q8031" and "based on" in aside[0]["statement"]
    assert {"id": "Q8031", "label": "Hymn of the Land", "dropped_by": "original_first"} in st["others"]["listed"]
    assert ns.ground(st) == "stated only"
    monkeypatch.setenv("CB_ALIAS_OPEN", "1")
    fact = walk(world, "place of death", "emperor constantine")
    world._tiers("emperor constantine")
    st = ns.build(loop_over(world), {"seed": "Emperor Constantine", "trace": [fact or "Lyon is the place of death of Constantine II"]})
    assert st["others"]["count"] >= 2 and ns.ground(st) == "stated only"


def test_a_place_qualifier_names_its_fact():
    q = ns._qualifier("nashville tn", "Nashville", [("Nashville", "located in the administrative territorial entity", "Tennessee"),
                                                    ("Nashville", "country", "United States")])
    assert q["rule"] == "place_qualifier" and q["statement"] == "unstated"        # "tn" names no fact by itself
    q = ns._qualifier("greensburg indiana", "Greensburg", [("Greensburg", "located in the administrative territorial entity", "Indiana")])
    assert q["statement"] == "Indiana is the located in the administrative territorial entity of Greensburg"


def test_the_branch_with_no_store_names_its_world_and_counts_nothing():
    from cubbyllm.worlds.base import FactStore
    wiki = FactStore(["Henry Selick is the director of The Nightmare Before Christmas"], name="wiki")
    loop = SimpleNamespace(world=SimpleNamespace(held_by=lambda f: None, primary=wiki, others=[]), source=None)
    st = ns.build(loop, {"seed": "nightmare before christmas", "trace": ["Henry Selick is the director of The Nightmare Before Christmas"]})
    assert st["store"] == "wiki" and st["item"]["id"] is None and st["item"]["reason"] == "this world's facts name no item"
    assert st["others"]["count"] is None and ns.ground(st) == "stated only" and ns.klass(st) == "taken as stated"


def test_the_refusals_carry_a_statement(world):
    loop = SimpleNamespace(world=SimpleNamespace(primary=None, others=[world]), source=None)
    st = ns.build(loop, {"seed": "twin tale", "trace": [], "reason": "ambiguous_entity",
                         "candidates": [["Twin Tale", "literary work", "Q8070"], ["Twin Tale", "literary work", "Q8071"]]})
    assert st["store"] == world.name and st["others"]["count"] == 2
    assert [o["id"] for o in st["others"]["listed"]] == ["Q8070", "Q8071"]
    st = ns.build(loop, {"seed": "nowhere town", "trace": [], "reason": "entity_unresolved", "entities": ["nowhere town"]})
    assert st["item"]["id"] is None and st["worlds"][0] == {"store": world.name, "items": 0} and st["others"]["count"] == 0


def test_the_live_source_states_the_items_own_label_and_its_read():
    class Src:
        last: dict = {}

        def resolve(self, name, rels=None):
            self.last.update(how="exact", exact=["Q1", "Q2"])
            return ("Q1", "Republic of Ireland")

        def read_of(self, qid):
            return {"read_at": "2026-11-11T10:00:00Z"}
    loop = SimpleNamespace(world=None, source=Src())
    st = ns.build(loop, {"seed": "ireland", "trace": ["Dublin is the capital of ireland"]}, live=True)
    assert st["item"] == {"id": "Q1", "label": "Republic of Ireland", "how": "exact"} and st["read"]["read_at"]
    assert st["steps"] == [{"rule": "alias", "alias": None, "how": "exact"}] and st["others"]["count"] == 1
    assert ns.ground(st) == "stated only"


def test_the_world_that_handed_the_fact_is_named_not_the_first_that_holds_it():
    fact = "Carson is the author of The Gospel According to John"

    class Built:                                       # holds the fact: built for two items, handed for neither
        name = "wdstore"

        def __contains__(self, f):
            return f == fact

        def handed(self, key):
            return None

        def fact_items(self, f):
            return ["Q7", "Q9"]

    class Below:                                       # read it from its one item
        name = "hdc"
        snapshot = "2019"

        def __contains__(self, f):
            return f == fact

        def handed(self, key):
            return None

        def fact_items(self, f):
            return ["Q5"]

        def _named(self, key):
            return "label", {"Q5": [("The Gospel According to John", "author", "Carson")]}

        def decision(self, key):
            return None
    built, below = Built(), Below()
    loop = SimpleNamespace(world=SimpleNamespace(held_by=lambda f: built, others=[built, below]), source=None)
    st = ns.build(loop, {"seed": "the gospel according to john", "trace": [fact]})
    assert st["store"] == "hdc" and st["item"]["id"] == "Q5" and ns.ground(st) == "re-derived"
