"""The query-log projection (order 0055's addendum): properties by role, counted once per query; the hot-size table."""
import json

from validation.fact_store import query_logs as Q

W = "http://www.wikidata.org/"
QS = [
    f"SELECT ?x WHERE {{ ?x <{W}prop/direct/P31> <{W}entity/Q5> ; <{W}prop/direct/P31> ?y . }}",
    f"SELECT ?x WHERE {{ ?x <{W}prop/P39> ?s . ?s <{W}prop/statement/P39> ?o ; <{W}prop/qualifier/P580> ?t . }}",
    f"SELECT ?x WHERE {{ ?x <{W}prop/direct/P31>/<{W}prop/direct/P279>* <{W}entity/Q515> . }} LIMIT 5",
    "SELECT ?l WHERE { wd:Q42 rdfs:label ?l . }",
    "SELECT ?x WHERE { ?x wdt:P2860 wd:Q1 ; pr:P248 ?r . }",
]


def _count():
    return Q.count(json.dumps(q) for q in QS)


def test_each_property_counts_once_per_query_by_role():
    res, per = _count()
    assert res["queries"] == 5
    assert res["props"]["main"] == {"P31": 2, "P39": 1, "P279": 1, "P2860": 1}
    assert res["props"]["qualifier"] == {"P580": 1} and res["props"]["reference"] == {"P248": 1}
    assert per[1] == {"P39", "P580"} and per[3] == set()


def test_shapes_see_paths_qualifiers_and_limits():
    res, _ = _count()
    shapes = dict(res["shapes_top"])
    assert shapes["limit,path,props:2"] == 1 and shapes["props:2,qualifiers"] == 1 and shapes["props:0"] == 1


def test_the_projection_counts_a_query_once_all_its_properties_are_hot():
    res, per = _count()
    st = {"P31": 100, "P39": 10, "P279": 5, "P2860": 1000, "P580": 0}
    rows = {r["k"]: r for r in Q.project(res, per, st, ks=(1, 2, 5))}
    assert rows[1]["statements"] == 100                         # P31 ranks first (2 queries)
    assert rows[1]["queries_covered"] == 2                      # the P31 query and the one naming none
    assert rows[5]["queries_covered"] == 5 and rows[5]["statements"] == 1115
    assert rows[1]["queries_with_props_share"] == 0.25
