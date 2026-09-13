"""standin/sources.py -- PropertyAliases: Wikidata's property vocabulary as a LOCAL resolver.

Pinned (2026-09-12, after exp_r14's 875 API calls per 100 questions):
  * a wording that is a property's label or alias, in English or French, names that
    property's English label; a wording that is neither names nothing;
  * WikidataSource.relations() reads the table and makes NO call; the property-search API
    is used only when `online_relations=True`.
Run: python -m pytest standin/tests/test_property_aliases.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sources import PropertyAliases, WikidataSource  # noqa: E402

TABLE = {"source": "test", "langs": ["en", "fr"], "n_properties": 5, "properties": {
    "P569": {"label": {"en": "date of birth", "fr": "date de naissance"}, "aliases": {"en": ["born", "DOB", "birthday"], "fr": ["né le", "naissance"]}, "datatype": "time"},
    "P19": {"label": {"en": "place of birth", "fr": "lieu de naissance"}, "aliases": {"en": ["birthplace", "born in"], "fr": ["né à"]}, "datatype": "wikibase-item"},
    "P571": {"label": {"en": "inception", "fr": "date de fondation ou de création"}, "aliases": {"en": ["founded", "established", "created"], "fr": ["fondé en", "création"]}, "datatype": "time"},
    "P1082": {"label": {"en": "population", "fr": "population"}, "aliases": {"en": ["inhabitants"], "fr": ["habitants"]}, "datatype": "quantity"},
    "P214": {"label": {"en": "VIAF ID", "fr": "identifiant VIAF"}, "aliases": {"en": ["VIAF"], "fr": []}, "datatype": "external-id"},
}}


def table(tmp_path) -> pathlib.Path:
    p = tmp_path / "props.json"; p.write_text(json.dumps(TABLE), encoding="utf-8"); return p


def test_a_wording_names_the_property_whose_label_or_alias_it_is(tmp_path):
    pa = PropertyAliases(table(tmp_path))
    assert pa.relations("born") == ["date of birth"]
    assert pa.relations("Born in") == ["place of birth"]
    assert pa.relations("founded") == ["inception"] and pa.relations("established") == ["inception"]
    assert pa.relations("date de naissance") == ["date of birth"] and pa.relations("naissance") == ["date of birth"]
    assert pa.relations("lieu de naissance") == ["place of birth"]
    assert pa.relations("year") == [] and pa.relations("banana") == []


def test_the_source_resolves_relations_locally_and_makes_no_call(tmp_path):
    src = WikidataSource(cache_dir=None, aliases=PropertyAliases(table(tmp_path)), offline=True)
    assert src.relations("born") == ["date of birth"] and src.calls == 0
    assert src.relations("xyzzy") == [] and src.calls == 0            # no fallback to the API unless asked


def test_a_label_has_the_value_kind_its_datatype_says_and_nothing_else_has_one(tmp_path):
    """2026-09-13: the value kind the typed answer class can narrow an ambiguous wording
    by -- time -> date, quantity -> number, item -> name; an external id is no kind a
    question asks for, and a label the table has no datatype for is None (never narrows)."""
    pa = PropertyAliases(table(tmp_path))
    assert pa.kind("date of birth") == "date" and pa.kind("inception") == "date"
    assert pa.kind("place of birth") == "name" and pa.kind("population") == "number"
    assert pa.kind("VIAF ID") is None and pa.kind("banana") is None
    untyped = json.loads(json.dumps(TABLE)); untyped["properties"]["P569"].pop("datatype")
    p = tmp_path / "untyped.json"; p.write_text(json.dumps(untyped), encoding="utf-8")
    assert PropertyAliases(p).kind("date of birth") is None
    src = WikidataSource(cache_dir=None, aliases=pa, offline=True)
    assert src.kind("date of birth") == "date" and src.calls == 0
    assert WikidataSource(cache_dir=None, aliases=None, offline=True).kind("date of birth") is None
