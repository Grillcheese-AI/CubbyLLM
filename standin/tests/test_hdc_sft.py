"""standin/data/build_hdc_sft.py -- template sentences read back into the VM's facts-block lines.

Pinned: a sentence is read through the template it was written from, the longest one where two share a start;
two relations written to one sentence read as the relation the sentence names ('capital', not 'capital of',
whose reading is the reverse); a noun that ends on a verb is said the other way round; a chain answer states
every hop in order and ends on the answer.
Run: python -m pytest standin/tests/test_hdc_sft.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_hdc_sft import Templates, clause, load_names, parse_rows, rename, sentence  # noqa: E402

CSV = """relation,template,edited_relation,noun_template
country,Which {relation} is {subject} in?,,the {relation} {subject} is in
country of citizenship,What is the {relation} of {subject}?,,the {relation} of {subject}
capital of,What is the {relation} of {subject}?,capital,the {relation} of {subject}
capital,What is the {relation} of {subject}?,,the {relation} of {subject}
educated at,Where is the {subject} {relation}?,,the institution where {subject} was educated
instance of,What is the {relation} of {subject}?,instance,the {relation} of {subject}
official language,What is the {relation} of {subject}?,,the {relation} of {subject}
"""


def tpl(tmp_path):
    p = tmp_path / "relation_template_mapping.csv"
    p.write_text(CSV, encoding="utf-8")
    return Templates(str(p))


def test_a_sentence_reads_back_through_its_template(tmp_path):
    t = tpl(tmp_path)
    assert t.parse("united stated is the country Philadelphia Racquet Club is in")[:3] == \
        ("Philadelphia Racquet Club", "country", "united stated")
    assert t.parse("Canada is the country of citizenship of Bank of Montreal Smith")[:3] == \
        ("Bank of Montreal Smith", "country of citizenship", "Canada")          # the longer template, 'of' in the name
    assert t.parse("mt holyoke coll is the institution where lynne barrett was educated")[:3] == \
        ("lynne barrett", "educated at", "mt holyoke coll")
    assert t.parse("This is the End is the official language of Nowhere")[:3] == ("Nowhere", "official language", "This is the End")
    assert t.parse("nothing here matches a template") is None


def test_one_sentence_two_relations_reads_as_the_one_it_names(tmp_path):
    assert tpl(tmp_path).parse("Balatonfured is the capital of balatonfured district")[1] == "capital"


def test_clauses_and_a_chain():
    assert clause("the {subject} is in", "X", "V", "other") == "V is the X is in"
    assert clause("the country {subject} is in", "Burrages End", "united stated") == "Burrages End is in united stated"
    assert clause("the capital of {subject}", "Peru", "Lima") == "the capital of Peru is Lima"
    assert clause("the instance of {subject}", "Funera", "Taxxon", "instance of") == "Funera is an instance of Taxxon"
    hops = [clause("the country {subject} is in", "Burrages End", "united stated"),
            clause("the official language of {subject}", "united stated", "English")]
    assert sentence(hops) == ("Burrages End is in united stated, and the official language of united stated is "
                              "English.")
    assert sentence(["a", "b", "c"]) == "A, b, and c."


def test_every_name_is_the_entity_s_proper_name(tmp_path):
    """The set writes an entity by one random alias ('Huamn', 'united stated', 'p:ca'); the rows carry the label."""
    names_file = tmp_path / "names.tsv"
    names_file.write_text("alias\tqid\tname\thow\n"
                          "united stated\tQ30\tUnited States of America\tlabel\n"
                          "Philadelphia Racquet Club\tQ7182853\tPhiladelphia Racquet Club\tlabel\n"
                          "Number of words in English\tQ1860\tEnglish\tlabel\n"
                          "lists\tQ13406463\t\twikimedia\n", encoding="utf-8")
    names = load_names(str(names_file))
    assert names["united stated"] == "United States of America" and names["lists"] == ""
    assert rename("Which country is laird hill, texas in?", "laird hill, texas", "Laird Hill") == "Which country is Laird Hill in?"
    assert rename("Where is LAIRD HILL?", "laird hill", "Laird Hill") == "Where is Laird Hill?"
    assert rename("Where is it?", "laird hill", "Laird Hill") is None
    t = tpl(tmp_path)
    rows = [("train", "What is the official language of the country of Philadelphia Racquet Club?",
             "Number of words in English", 2,
             ["united stated is the country Philadelphia Racquet Club is in",
              "Number of words in English is the official language of united stated"]),
            ("train", "Which country is Philadelphia Racquet Club in?", "lists", 1,
             ["lists is the country Philadelphia Racquet Club is in"])]
    parsed, _, funnel = parse_rows(rows, t, names)
    assert len(parsed) == 1 and funnel["no_proper_name"] == 1               # a name with no usable label: row dropped
    _, q, a, hops = parsed[0]
    assert a == "English" and [(h[0], h[2]) for h in hops] == [("Philadelphia Racquet Club", "United States of America"),
                                                              ("United States of America", "English")]
