"""standin/encyclopedia.py -- entries, names and frames over OCR'd encyclopedia text.

Pinned (2026-09-13, from volume 13's own lines, exp_r16 probe):
  * a headword line names a person as given names + surname, the pronunciation skipped, the
    running head of the page and cross-reference initials dropped; a caption that ran into the
    given names ('Ulysses Simpson Historical Collection') leaves the entry nameless and factless;
  * the headword's years are the birth and death years; a full date read from the opening must
    agree with them or neither is stated; 'bom' reads as 'born'; line-break hyphens are removed;
  * a person frame counts only in a sentence about the subject ('His father was born in Paris'
    states nothing); a place with an initial in it ('Justus J. Schifferes Coauthor') is a byline;
  * a place entry states its immediate container; the source names its relations and their kinds.
Run: python -m pytest standin/tests/test_encyclopedia.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from encyclopedia import EncyclopediaSource, entries, given_names, headword, read_entry, surname  # noqa: E402

GOETHE = ("gu'ta, Johann Wolfgang von (1749- 1832), the greatest of all German poets. His father, Johann Caspar Goethe, "
          "a lawyer, was born in Frankfurt. Goethe was bom on Aug. 28, 1749, in the free city of Frankfurt. He died in "
          "Weimar on March 22, 1832.")


def test_a_headword_line_names_the_person_and_the_years_are_the_birth_and_death_years():
    name, person, facts = read_entry("GOETHE", GOETHE)
    assert name == "Johann Wolfgang von Goethe" and person
    got = {(t.rel, t.obj) for t, _s in facts}
    assert got == {("date of birth", "1749-08-28"), ("place of birth", "Frankfurt"),
                   ("date of death", "1832-03-22"), ("place of death", "Weimar")}
    sentences = {t.rel: s for t, s in facts}
    assert sentences["date of birth"].startswith("Goethe was bom on Aug. 28, 1749")   # the sentence, the father's skipped
    assert all(t.subj == "Johann Wolfgang von Goethe" for t, _s in facts)


def test_a_sentence_about_someone_else_states_nothing_about_the_subject():
    body = "Alec, gin'is (1914-2000), British actor. His father was born in Paris on May 1, 1880. He worked in London."
    name, _p, facts = read_entry("GUINNESS", body)
    assert name == "Alec Guinness"
    assert {(t.rel, t.obj) for t, _s in facts} == {("date of birth", "1914"), ("date of death", "2000")}


def test_a_full_date_that_disagrees_with_the_headword_year_drops_both():
    """A birth sentence dated to another year than the headword's is either an OCR misread or
    another entry's sentence: neither year is stated, and nor is the place that sentence names."""
    body = "hak, Edward (1705-1781), British admiral. He was born in London on Feb. 21, 1710. He died on Oct. 17, 1781."
    _n, _p, facts = read_entry("HAWKE", body)
    got = {(t.rel, t.obj) for t, _s in facts}
    assert got == {("date of death", "1781-10-17")}
    body = "hak, Edward (1705-1781), British admiral. He was born in London on Feb. 21, 1705."
    got = {(t.rel, t.obj) for t, _s in read_entry("HAWKE", body)[2]}
    assert got == {("date of birth", "1705-02-21"), ("place of birth", "London"), ("date of death", "1781")}


def test_line_break_hyphens_and_the_ocr_reading_of_born():
    body = "Jane, good'al (1934- ), British ethologist. Goodall was born in Lon- don on April 3, 1934."
    _n, _p, facts = read_entry("GOODALL", body)
    assert {(t.rel, t.obj) for t, _s in facts} == {("date of birth", "1934-04-03"), ("place of birth", "London")}
    body = "gwon'dar, a city in northwestern Ethi- opia, is the capital of Begemdir prov- ince."
    name, person, facts = read_entry("GONDAR", body)
    assert name == "Gondar" and not person
    assert [(t.rel, t.obj) for t, _s in facts] == [("located in the administrative territorial entity", "Begemdir Province")]


def test_a_place_entry_states_the_container_it_names_as_one_never_the_opener():
    """'a city in Japan' is a country as often as a region: the frame reads 'in Fulton county',
    'is in Shizuoka prefecture', 'is in Hampshire,' -- a container named as one -- in Wikidata's
    label form; a thing that is not a place ('a large bird ... in Fulton county') is located nowhere."""
    rel = "located in the administrative territorial entity"
    assert [(t.rel, t.obj) for t, _s in read_entry("HAPEVILLE", "hap'vil, is a city in north central Georgia, in Fulton county, 7 miles southwest of Atlanta.")[2]] == [(rel, "Fulton County")]
    assert [t.obj for t, _s in read_entry("HAMAMATSU", "ha-ma-ma-tsoo, a city in Japan, is in Shizuoka prefecture, midway between Tokyo and Osaka.")[2]] == ["Shizuoka Prefecture"]
    assert [t.obj for t, _s in read_entry("GOSPORT", "gos'port, a borough in south central England, is in Hampshire, on the west shore of Portsmouth Harbour.")[2]] == ["Hampshire"]
    assert read_entry("GRODNO", "grbd'ns, a city in Belarus, and the capital of Hrodna region.")[2][0][0].obj == "Hrodna Region"
    assert read_entry("GOLDEN EAGLE", "a large bird of prey, is found in Fulton county and beyond.")[2] == []


def test_a_sentence_dated_to_another_year_is_another_entrys_run_in():
    """volume 13: the headword of the entry after Nordahl Grieg (1902-1943) was missed, so 'He
    was born in Cumberland county, Pa., on March 5, 1794' followed his entry -- the year says
    it is not his, and the source now disagrees with itself about his birth: neither the
    place nor either year is stated. The next entry's headword run into a place ('New York
    GRISWOLD') is refused."""
    body = "greg, Nordahl (1902-1943), Norwegian writer. He was born in Cumberland county, Pa., on March 5, 1794. He died in 1943."
    _n, _p, facts = read_entry("GRIEG", body)
    assert {(t.rel, t.obj) for t, _s in facts} == {("date of death", "1943")}
    body = "griz'wold, Rufus Wilmot (1815-1857), American editor. Griswold died in New York GRISWOLD v. CONNECTICUT, a case."
    _n, _p, facts = read_entry("GRISWOLD", body)
    assert all(t.rel != "place of death" for t, _s in facts)


def test_captions_and_bylines_the_ocr_ran_into_the_text_are_not_names_or_places():
    assert given_names("Ulysses Simpson Historical Collection") is None
    assert given_names("gu'ta, Johann Wolfgang von") == "Johann Wolfgang von"
    assert given_names("gild, Curtis, Jr.") == "Curtis"
    name, person, facts = read_entry("GRANT", "Ulysses Simpson Historical Collection (1822-1885), 18th president.")
    assert name is None and person and facts == []
    body = "han, Julius Ferdinand von (1839-1921), Austrian meteorologist. He died in Justus J. Schifferes Coauthor of Science."
    _n, _p, facts = read_entry("HANN", body)
    assert all(t.rel != "place of death" for t, _s in facts)


def test_running_heads_initials_and_surnames():
    assert headword("GOMBERT-GOMEZ GOMBERT") == "GOMBERT"
    assert headword("GOSHO HEINOSUKE-GOSNOLD GOSHO HEINOSUKE") == "GOSHO HEINOSUKE"
    assert headword("GORDON GORDON") == "GORDON" and headword("HAGUE COURT HAGUE COURT") == "HAGUE COURT"
    assert headword("N. Y. GOTTSCHED") == "GOTTSCHED"                     # a byline's 'Dobbs Ferry, N. Y.' ran in
    assert headword("PEREZ GALDOS- PEREZ GALDOS") == "PEREZ GALDOS"
    assert given_names("Ibr'ing, Ellis Gray") == "Ellis Gray"              # the OCR's capital I for l in a pronunciation
    assert given_names("O'Brien, Conor") == "O'Brien" and given_names("D'Annunzio") == "D'Annunzio"
    head_line = "boist, Count Friedrich Ferdinand von (1809-1886), Saxon statesman. "
    tail = "Beust died at Castle Altenberg on Oct. 24, 1886."
    body = head_line + "x" * (1500 - len(head_line) - len("Beust died at Castle Altenb")) + " " + tail   # the opening ends 'Castle Altenb'
    _n, _p, facts = read_entry("BEUST", body)
    assert all(t.rel != "place of death" for t, _s in facts)             # cut off at the opening's end: not read
    assert ("place of death", "Castle Altenberg") in {(t.rel, t.obj) for t, _s in read_entry("BEUST", head_line + tail)[2]}
    assert headword("H. M.") is None and headword("N.Y.") is None
    assert surname("GONGORA Y ARGOTE") == "Gongora y Argote" and surname("O'BRIEN") == "O'Brien"
    _n, _p, facts = read_entry("GRAU SAN MARTIN", "Ramon, grou (1882-1969), Cuban. He was born in Pinar del Rio, Cuba, on Sept. 13, 1882.")
    assert ("place of birth", "Pinar del Rio") in {(t.rel, t.obj) for t, _s in facts}


def test_the_source_indexes_a_volume_and_names_its_relations_and_kinds(tmp_path):
    vol = tmp_path / "Volume 13. Goethe to Hearst - 2005.txt"
    vol.write_text("Front matter. GOETHE, " + GOETHE + " GOSPORT, gos'port, a borough in south central England, "
                   "is in Hampshire. GOULD, goold, Jay (1836-1892), American financier.", encoding="utf-8")
    src = EncyclopediaSource(tmp_path)
    assert [h for h, _s, _b, _e in entries(vol.read_text(encoding="utf-8"))] == ["GOETHE", "GOSPORT", "GOULD"]
    assert {t.obj for t in src.facts("Johann Wolfgang von Goethe")} == {"1749-08-28", "Frankfurt", "1832-03-22", "Weimar"}
    assert src.last["volume"].startswith("Volume 13") and src.last["headword"] == "GOETHE"
    assert [t.obj for t in src.facts("gosport")] == ["Hampshire"]
    assert {(t.rel, t.obj) for t in src.facts("Jay Gould")} == {("date of birth", "1836"), ("date of death", "1892")}
    assert src.facts("Gould") == [] and src.facts("nobody") == []          # a person by full name only
    assert src.relations("born") == ["date of birth", "place of birth"] and src.kind("date of birth") == "date"
    assert src.kind("place of birth") == "name" and src.calls == 0
    assert (tmp_path / "Volume 13. Goethe to Hearst - 2005.entries.v2.json").exists()
