"""wikikg: triples -> template facts both ways, hop phrases per direction, the world, functional-hop questions."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import wikikg as wk  # noqa: E402
from cubbyllm.reasoning.planner import parse_fact, parse_question  # noqa: E402

TRIPLES = [
    ("GirardDesargues", "BORN_ON", "1591-02-21"),
    ("GirardDesargues", "CREATOR_OF", "DesarguesianPlane"),
    ("KeikoMatsuzaka", "WORKS_FOR", "Daiei"),
    ("Paris", "CAPITAL_OF", "France"),
    ("AlphabetInc", "PARENT_COMPANY_OF", "Google"),
    ("SundarPichai", "POSITION_HELD", "AlphabetInc"),
    ("SundarPichai", "SPOUSE_OF", "AnjaliPichai"),
    ("Dairy", "HAS_PART", "Milk"),
    ("Dairy", "HAS_PART", "Cheese"),
    ("Lyon", "FLOWS_INTO_SEA", "Mediterranean"),          # unmapped relation -> generic phrase
    ("Loop", "PART_OF", "Loop"),                          # self loop -> dropped
]


def test_decamel_and_dates():
    assert wk.decamel("KeikoMatsuzaka") == "Keiko Matsuzaka"
    assert wk.decamel("AlphabetInc_Creation") == "Alphabet Inc Creation"
    assert wk.decamel("USAPresident") == "USA President"
    assert wk.decamel("1591-02-21") == "1591-02-21"


def test_facts_read_both_ways_and_every_fact_parses():
    assert wk.facts_for("GirardDesargues", "BORN_ON", "1591-02-21") == ["1591-02-21 is the birth date of Girard Desargues"]
    assert wk.facts_for("GirardDesargues", "CREATOR_OF", "DesarguesianPlane") == [
        "Girard Desargues is the creator of Desarguesian Plane", "Desarguesian Plane is the creation of Girard Desargues"]
    assert wk.facts_for("KeikoMatsuzaka", "WORKS_FOR", "Daiei") == ["Daiei is the employer of Keiko Matsuzaka", "Keiko Matsuzaka is the employee of Daiei"]
    assert wk.facts_for("SundarPichai", "SPOUSE_OF", "AnjaliPichai") == ["Sundar Pichai is the spouse of Anjali Pichai", "Anjali Pichai is the spouse of Sundar Pichai"]
    assert wk.facts_for("Lyon", "FLOWS_INTO_SEA", "Mediterranean") == ["Mediterranean is the flows into sea of Lyon"]
    assert wk.facts_for("X", "PRODUCT_OF", "Y") == ["X is the product of Y"], "an unmapped X_OF name keeps the S-is-the-X-of-O reading"
    assert wk.facts_for("Loop", "PART_OF", "Loop") == []
    facts = wk.facts_from_triples(TRIPLES)
    assert all(parse_fact(f) is not None for f in facts) and len(facts) == len(set(facts))
    assert "Paris is the capital of France" in facts


def test_hop_phrase_per_direction():
    assert wk.hop_phrase("CREATOR_OF", "backward") == "creator" and wk.hop_phrase("CREATOR_OF", "forward") == "creation"
    assert wk.hop_phrase("BORN_ON", "forward") == "birth date" and wk.hop_phrase("BORN_ON", "backward") is None
    assert wk.hop_phrase("PART_OF", "forward") == "whole" and wk.hop_phrase("HAS_PART", "forward") == "part"


def test_relation_map_covers_the_heavy_relations():
    for r in ("TIMELINE_EVENT", "HAS_PART", "MEMBER_OF", "IS_A", "PART_OF", "VARIANT_OF", "INSTANCE_OF", "LOCATED_IN",
              "PERFORMED_BY", "CREATED_BY", "CAUSED_BY", "LOCATION_OF", "POSITION_HELD", "BORN_ON", "AUTHORED_BY"):
        assert r in wk.RELATIONS, r
    for r, (noun, inv, form) in wk.RELATIONS.items():
        assert form in ("so", "os") and noun and noun == noun.lower() and (inv is None or inv == inv.lower()), r


def test_wiki_world_looks_up_both_directions_and_falls_back_lexically():
    w = wk.wiki_world(triples=TRIPLES)
    assert len(w.index) == len(w) > 0
    plan = parse_question("What is the capital of France?")
    assert [f for f, _ in w.lookup(plan, 0, None)] == ["Paris is the capital of France"]
    plan = parse_question("What is the creation of Girard Desargues?")
    assert [t.obj for _, t in w.lookup(plan, 0, None)] == ["Desarguesian Plane"]
    plan = parse_question("What is the employer of the holder of Alphabet Inc?")      # POSITION_HELD read back: holder
    c0 = w.lookup(plan, 0, None)
    assert [t.obj for _, t in c0] == ["Sundar Pichai"]
    assert w("Girard Desargues birth", 2)[0][1].endswith("birth date of Girard Desargues")   # no encoder: IDF fallback


def test_chain_questions_take_functional_hops_only():
    w = wk.wiki_world(triples=TRIPLES)
    paths = [
        (["AlphabetInc", "SundarPichai", "AnjaliPichai"], ["POSITION_HELD", "SPOUSE_OF"], ["backward", "forward"]),   # holder -> spouse: unique
        (["Milk", "Dairy", "Cheese"], ["HAS_PART", "HAS_PART"], ["backward", "forward"]),                              # 'part of Dairy' has two answers
        (["GirardDesargues", "1591-02-21", "X"], ["BORN_ON", "BORN_ON"], ["forward", "backward"]),                    # backward BORN_ON has no phrase
    ]
    qs = wk.chain_questions(paths, w.index, n=10)
    assert len(qs) == 1
    q = qs[0]
    assert q["question"] == "What is the spouse of the holder of Alphabet Inc?" and q["answer"] == "Anjali Pichai"
    assert q["facts"] == ["Sundar Pichai is the holder of Alphabet Inc", "Anjali Pichai is the spouse of Sundar Pichai"]
