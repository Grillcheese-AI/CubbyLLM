"""Pins for the offline article Source (standin/wikitext.py): the frames read what they say and
nothing else, in English and in French, with the sentence as provenance; the jsonl corpus is
indexed once and read by offset; an entity without an article yields nothing.
Run: python -m pytest standin/tests/test_wikitext.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from wikitext import WikiTextSource, read_frames  # noqa: E402

EN = ("Doug Tewell", "Douglas Fred Tewell (born August 27, 1949) is an American professional golfer who was born in Pittsburg, Kansas. He founded a golf academy in 1998.")
EN2 = ("Charles River Shire", "Charles River Shire was one of eight shires of Virginia created in the Virginia Colony in 1634.")
EN3 = ("Alvin Toffler", "Alvin Eugene Toffler (October 4, 1928 – June 27, 2016) was an American writer, futurist, and businessman.")
FR = ("Marie Curie", "Marie Curie, née le 7 novembre 1867 à Varsovie et morte le 4 juillet 1934 à Passy, est une physicienne et chimiste polonaise.")
FR2 = ("Lévis", "Lévis est une ville du Québec située dans la région de la Chaudière-Appalaches.")


def test_english_frames_read_dates_places_and_inception_with_the_sentence():
    facts = {(t.rel, t.obj): s for t, s in read_frames(*EN)}
    assert facts[("date of birth", "1949-08-27")].startswith("Douglas Fred Tewell (born August 27, 1949)")
    assert ("place of birth", "Pittsburg, Kansas") in facts
    assert not any(r == "inception" for r, _o in facts)               # 'founded a golf academy in 1998' is the academy's, not his
    assert all(t.subj == "Doug Tewell" for t, _s in read_frames(*EN))
    assert {(t.rel, t.obj) for t, _s in read_frames("Acme Corp", "Acme Corp is an American company that was founded in 1998 in Ohio.")} == {("inception", "1998")}
    assert read_frames(*EN2) == []                                   # 'created in the Virginia Colony in 1634' is not an inception frame
    facts3 = {(t.rel, t.obj) for t, _s in read_frames(*EN3)}
    assert ("date of birth", "1928-10-04") in facts3 and ("date of death", "2016-06-27") in facts3


def test_french_frames():
    facts = {(t.rel, t.obj) for t, _s in read_frames(*FR, lang="fr")}
    assert ("date of birth", "1867-11-07") in facts and ("date of death", "1934-07-04") in facts
    assert ("place of birth", "Varsovie") in facts
    assert read_frames(*FR2, lang="fr") == []                        # 'ville du Québec située dans la région' names a region, not a commune's department


def test_jsonl_corpus_is_indexed_once_and_read_by_offset(tmp_path):
    p = tmp_path / "wiki.jsonl"
    rows = [{"title": EN[0], "text": EN[1], "lang": "latest.en"}, {"title": FR[0], "text": FR[1], "lang": "latest.fr"}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    src = WikiTextSource(jsonl=[p])
    assert (tmp_path / "wiki.jsonl.titles2.json").exists()
    fs = src.facts("doug tewell")
    assert {(t.rel, t.obj) for t in fs} == {("date of birth", "1949-08-27"), ("place of birth", "Pittsburg, Kansas")}
    assert src.last["corpus"] == "wiki" and src.last["title"] == "Doug Tewell" and src.last["sentences"]["date of birth|1949-08-27"]
    fr = src.facts("Marie Curie")
    assert ("date of birth", "1867-11-07") in {(t.rel, t.obj) for t in fr} and src.last["lang"] == "fr"
    assert src.facts("nobody at all") == []
    # the cached index is what a second instance loads
    src2 = WikiTextSource(jsonl=[p])
    assert src2.facts("Marie Curie") and src2.corpora[0][3] == src.corpora[0][3]


def test_run_1_hazards_a_nationality_is_not_a_person_and_an_article_keeps_its_the(tmp_path):
    """exp_r11 wikitext run 1 admitted 'British is the author of The Roar': the author
    frame took the first capitalised word of 'by British author Jane Doe', and the seed
    'roar' matched the article 'The Roar' because the title key dropped the article."""
    assert read_frames("The Roar", "The Roar is a 2004 novel by British author Emma Clayton about a flooded world.") == []
    assert {(t.rel, t.obj) for t, _s in read_frames("The Roar", "The Roar is a 2004 novel by Emma Clayton.")} == {("author", "Emma Clayton")}
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps({"title": "The Roar", "text": "The Roar is a 2004 novel by Emma Clayton.", "lang": "latest.en"}) + "\n", encoding="utf-8")
    src = WikiTextSource(jsonl=[p])
    assert src.facts("roar") == [] and src.facts("Roar") == []
    assert [t.obj for t in src.facts("the roar")] == ["Emma Clayton"]
    # the source names the labels its frames read, for the words a question uses
    assert src.relations("born") == ["date of birth", "place of birth"]
    assert src.relations("naissance") == ["date of birth", "place of birth"] and src.relations("fondée") == ["inception"]
    assert src.relations("population") == []
