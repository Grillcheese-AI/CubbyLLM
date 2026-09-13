"""standin/lfm_source.py -- the local model as a fact PROPOSER (2026-09-13).

Pinned:
  * a line in the store's form whose relation the property table names (a label offered in the
    prompt, or a wording naming exactly one property) becomes a Triple; a relation the table
    cannot name is dropped, never invented;
  * a line about another subject is dropped (the model drifted); a non-answer object
    ('unknown', a '?') is dropped; duplicates collapse; the source is `latent`.
No model: the raw completion is stubbed. Run: python -m pytest standin/tests/test_lfm_source.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sources import PropertyAliases  # noqa: E402

TABLE = {"source": "test", "langs": ["en", "fr"], "n_properties": 4, "properties": {
    "P569": {"label": {"en": "date of birth", "fr": "date de naissance"}, "aliases": {"en": ["born", "DOB"], "fr": []}, "datatype": "time"},
    "P19": {"label": {"en": "place of birth", "fr": "lieu de naissance"}, "aliases": {"en": ["birthplace"], "fr": []}, "datatype": "wikibase-item"},
    "P106": {"label": {"en": "occupation", "fr": "occupation"}, "aliases": {"en": ["profession", "job"], "fr": ["mAtier"]}, "datatype": "wikibase-item"},
    "P27": {"label": {"en": "country of citizenship", "fr": "pays de citoyennetAc"}, "aliases": {"en": ["citizenship", "nationality"], "fr": []}, "datatype": "wikibase-item"},
}}


def _source(tmp_path, text: str):
    import lfm_source as L
    (tmp_path / "table.json").write_text(json.dumps(TABLE), encoding="utf-8")
    src = L.LfmSource.__new__(L.LfmSource)          # no GGUF load: the completion is stubbed
    src.gguf = "stub.gguf"; src.em = None; src.aliases = PropertyAliases(tmp_path / "table.json")
    src.relations_offered = L.RELATIONS; src.max_new = 100; src.max_facts = 40; src.cache = None
    src.calls = 0; src.model_calls = 0; src.last = {}
    src._raw = lambda entity: text                  # what the model would have completed after '- '
    return src


def test_lines_in_the_stores_form_become_triples_and_the_rest_is_dropped(tmp_path):
    src = _source(tmp_path, "1932-03-23 is the date of birth of Masaki Tsuji\n"
                            "- Nagoya is the birthplace of Masaki Tsuji\n"              # a wording: one property -> its label
                            "- screenwriter is the profession of Masaki Tsuji\n"
                            "- Japan is the country of citizenship of Masaki Tsuji\n"
                            "- Tokyo is the favourite city of Masaki Tsuji\n"          # no such relation in the table: dropped
                            "- 1970 is the date of birth of Someone Else\n"            # another subject: dropped
                            "- unknown is the date of birth of Masaki Tsuji\n"         # a non-answer: dropped
                            "- Nagoya is the birthplace of Masaki Tsuji\n")            # a duplicate: collapsed
    facts = src.facts("Masaki Tsuji")
    assert [(t.obj, t.rel, t.subj) for t in facts] == [
        ("1932-03-23", "date of birth", "Masaki Tsuji"), ("Nagoya", "place of birth", "Masaki Tsuji"),
        ("screenwriter", "occupation", "Masaki Tsuji"), ("Japan", "country of citizenship", "Masaki Tsuji")]
    assert src.last["dropped_relation"] == 1 and src.last["dropped_subject"] == 1 and src.last["dropped_object"] == 1
    assert src.latent is True and src.name == "lfm"
    assert src.relations("born") == ["date of birth"] and src.kind("date of birth") == "date"


def test_a_completion_with_nothing_usable_yields_no_facts(tmp_path):
    src = _source(tmp_path, "I don't know.\n- ? is the date of birth of Masaki Tsuji\n")
    assert src.facts("Masaki Tsuji") == [] and src.last["parsed"] == 0
