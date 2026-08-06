"""Guards the committed mowm axiom corpus (the M2 screen's only data input)."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "mowm_axioms.json"


def _corpus():
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_corpus_has_280_axioms_with_all_fields():
    d = _corpus()
    ax = d["axioms"]
    assert len(ax) == 280
    assert d["_provenance"]["n"] == len(ax)
    for a in ax:
        assert a["name"] and a["domain"] and a["formula_str"]


def test_corpus_covers_fifteen_domains_with_physics_largest():
    top = collections.Counter(
        a["domain"].split(".")[0] for a in _corpus()["axioms"]
    )
    assert len(top) == 15
    assert top["physics"] == 84
    assert sum(top.values()) == 280


def test_name_and_formula_are_distinct_views():
    # The screen poses `name` against worlds seeded from `formula_str`; if the
    # two views were ever identical the cross-view metric would be trivial.
    ax = _corpus()["axioms"]
    assert sum(a["name"] == a["formula_str"] for a in ax) == 0
