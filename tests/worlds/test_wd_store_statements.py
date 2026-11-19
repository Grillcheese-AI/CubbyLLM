"""Order 0076, step 2 (cubbyllm/worlds/wd_store_statements.py): the store world keeps, beside each fact it hands out,
the statements that said it, as a record: the item, property, statement id, rank, the item's revision and its time
qualifiers with their precision; one sentence said by two statements keeps both, and a statement is kept once."""
from __future__ import annotations

from cubbyllm.worlds.wd_store_statements import Statements

ENTITY = {"revision": 7, "modified": "2025-12-18T18:39:28Z"}


def _st(guid, value, rank=1, quals=(), qnodes=()):
    return {"pid": 421, "guid": guid, "rank": rank, "best": True, "value": {"entity": value}, "references": [{}],
            "hot_qualifiers": list(quals), "hot_qualifier_nodes": list(qnodes)}


def test_a_fact_keeps_each_statement_that_said_it_once():
    w = Statements()
    s1, s2 = _st("g1", "Q1"), _st("g2", "Q2", rank=2)
    w._note_statement("Q829", ENTITY, "Utah", "located in time zone", "UTC-07:00", s1, {})
    w._note_statement("Q829", ENTITY, "Utah", "located in time zone", "UTC-07:00", s1, {})
    w._note_statement("Q829", ENTITY, "Utah", "located in time zone", "UTC-07:00", s2, {})
    got = w.statements_of("utah", "Located in time zone", "utc-07:00")        # read back normalized
    assert [(g["guid"], g["rank"], g["pid"], g["revision"]) for g in got] == [("g1", 1, "P421", 7), ("g2", 2, "P421", 7)]
    assert w.statements_of("Utah", "currency", "dollar") == []


def test_time_qualifiers_keep_their_raw_literal_and_precision():
    w = Statements()
    s = _st("g3", "Q3", quals=[(580, {"typed": ["+2020-01-01T00:00:00Z"]}), (582, {"typed": ["+2021-06-01T00:00:00Z"]})],
            qnodes=[(580, "n1")])
    w._note_statement("Q1", ENTITY, "A", "member of", "B", s, {"n1": {"wikibase:timePrecision": "11"}})
    times = w.statements_of("A", "member of", "B")[0]["times"]
    assert times == {"start": {"time": "+2020-01-01T00:00:00Z", "precision": 11},
                     "end": {"time": "+2021-06-01T00:00:00Z", "precision": None}}
