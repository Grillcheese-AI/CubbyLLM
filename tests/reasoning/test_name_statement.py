"""Order 0054, the certificate's name statement (Cowork's shape, 2026-11-11): one test for each kind of name step among
the replay's 17 (an alias, a fold, a place qualifier, a role, a set-aside original) and a refusal that carries its
options. The store is test_wd_store_names's fixture world."""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

from cubbyllm.reasoning import name_statement as ns

_spec = importlib.util.spec_from_file_location("_wd_names2", pathlib.Path(__file__).parents[1] / "worlds" / "test_wd_store_names.py")
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
world = N.world                                         # the fixture


def loop_over(w):
    return SimpleNamespace(world=SimpleNamespace(held_by=lambda f: w, others=[w]), source=None)


def steps(st):
    return [s["rule"] for s in st["steps"]]


def test_an_alias_and_a_fold(world):
    N.fresh(world)
    st = ns.build(loop_over(world), {"seed": "Emperor Constantine", "trace": ["Lyon is the place of death of Constantine II"]})
    assert st["item"] == {"id": "Q8040", "label": "Constantine II"} and steps(st) == ["alias"]
    assert ns.ground(st, []) == "re-derived"
    st = ns.build(loop_over(world), {"seed": "gospel of john", "trace": ["John the Apostle is the author of Gospel of John"]})
    assert steps(st) == ["fold"] and ns.ground(st, []) == "re-derived"


def test_a_set_aside_original_and_a_role_are_picks_stated_only(world, monkeypatch):
    monkeypatch.setenv("CB_ORIGINAL_FIRST", "1")
    N.fresh(world)
    world._tiers("hymn of the land")
    st = ns.build(loop_over(world), {"seed": "Hymn of the Land", "trace": ["Ada is the author of Hymn of the Land"]})
    aside = [s for s in st["steps"] if s["rule"] == "original_first"]
    assert aside and aside[0]["set_aside"] == "Q8031" and "based on" in aside[0]["statement"]
    assert {"id": "Q8031", "label": "Hymn of the Land", "dropped_by": "original_first"} in st["others"]["listed"]
    assert ns.ground(st, []) == "stated only"
    monkeypatch.setenv("CB_ALIAS_OPEN", "1")
    N.fresh(world)
    world._tiers("emperor constantine")
    st = ns.build(loop_over(world), {"seed": "Emperor Constantine", "trace": ["Lyon is the place of death of Constantine II"]})
    assert "role" in steps(st) and st["others"]["count"] >= 2


def test_a_place_qualifier_names_its_fact():
    q = ns._qualifier("nashville tn", "Nashville", [("Nashville", "located in the administrative territorial entity", "Tennessee"),
                                                    ("Nashville", "country", "United States")])
    assert q["rule"] == "place_qualifier" and q["statement"] == "unstated"        # "tn" names no fact by itself
    q = ns._qualifier("greensburg indiana", "Greensburg", [("Greensburg", "located in the administrative territorial entity", "Indiana")])
    assert q["statement"] == "Indiana is the located in the administrative territorial entity of Greensburg"


def test_a_refusal_over_a_name_carries_its_options(world, monkeypatch):
    monkeypatch.setenv("CB_ALIAS_OPEN", "1")
    N.fresh(world)
    st = ns.build(loop_over(world), {"seed": "emperor constantine", "trace": [], "reason": "name_unresolved"})
    assert st["item"]["id"] is None and st["others"]["count"] >= 3
