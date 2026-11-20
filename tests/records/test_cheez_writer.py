"""Order 0076, step 2 (cubbyllm/records): the certificate and refusal writer. Pinned: a spoken answer resting on one
store statement a fact gets a certificate whose fields are the record's (answer, program hash, statement, rank,
revision, snapshot, checks, time); writing twice gives the same record; every case the record cannot state is no
record, with its reason (a year asked, two programs, a fact no world handed, a fact said by several statements, a
refusal word with no code); a valid-time beginning only from a start qualifier at day precision, derived from it. With
CheezML's shapes at the pin, the records conform (skipped while the pin lacks the host's check codes)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cubbyllm.records import cheez_writer as W

ROOT = Path(__file__).resolve().parents[2]
RUN = {"run": "t", "snapshot": "a" * 64, "as_of": "2025-12-29T23:00:38Z", "store": "wdstore",
       "host": "urn:uuid:00000000-0000-4000-8000-000000000001", "vm": "urn:uuid:00000000-0000-4000-8000-000000000002"}
ST = {"item": "Q79", "pid": "P38", "guid": "Q79-g1", "rank": 1, "best": True, "value": {"entity": "Q199462"},
      "revision": 2440000000, "modified": "2025-12-01T10:00:00Z", "references": 1, "times": {}}
CHECKS = [{"check": c, "satisfied": True} for c in ("answer class", "tense", "time rule", "conflict")]
NAME = {"asked": "egypt", "item": {"id": "Q79", "label": "Egypt", "how": "first_by_id"},
        "steps": [{"rule": "fold", "equal": ["egypt", "Egypt"]}, {"rule": "namesake", "kept": ["Q79", "Q2334225"]},
                  {"rule": "first_by_id", "kept": ["Q79", "Q2334225"], "chose": "Q79"}]}
REC = {"id": "webq:275", "spoken": True, "answer": "Egyptian pound", "proposer": "everyday", "reason": None, "name": NAME,
       "cert": {"facts": [{"fact": "Egyptian pound is the currency of Egypt", "triple": ["Egypt", "currency", "Egyptian pound"],
                           "worlds": [{"world": "wdstore", "items": ["Q79"], "statements": [ST]}]}],
                "programs": ["b" * 64, "b" * 64], "as_of": None, "checks": CHECKS, "at": "2026-11-19T20:50:33Z"}}


def _node(g, kind):
    return [n for n in g["@graph"] if kind in n["@type"]]


def test_a_spoken_answer_gets_a_certificate_with_the_records_own_fields():
    g, why = W.write(REC, RUN)
    assert why is None
    fv, = _node(g, "cheez:FactVersion")
    assert (fv["rdf:subject"], fv["rdf:predicate"], fv["rdf:object"], fv["wikibase:rank"], fv["schema:identifier"]) == \
        ("wd:Q79", "wdt:P38", {"@id": "wd:Q199462"}, "wikibase:NormalRank", "Q79-g1")
    assert "time:hasBeginning" not in fv["cheez:validTime"]                 # the statement states no start
    src = next(n for n in g["@graph"] if n["@id"] == fv["prov:hadPrimarySource"])
    assert src["schema:version"] == "2440000000"
    ans, = _node(g, "cubby:Answer")
    prog, = _node(g, "cube:Program")
    run, = _node(g, "cube:ProgramRun")
    cert, = _node(g, "cheez:Certificate")
    assert ans["prov:value"] == "Egyptian pound" and prog["cheez:hash"] == "b" * 64
    assert run["cheez:asOf"] == RUN["as_of"] and len(cert["cheez:check"]) == 4
    assert W.write(copy.deepcopy(REC), RUN) == (g, None)                    # written again: the same record


@pytest.mark.parametrize("change, why", [
    (lambda r: r["cert"].update(as_of="2012"), "names a year"),
    (lambda r: r["cert"].update(programs=["b" * 64, "c" * 64]), "2 programs"),
    (lambda r: r["cert"]["facts"][0].update(worlds=[]), "no world says it handed"),
    (lambda r: r["cert"]["facts"][0]["worlds"][0].update(statements=[ST, dict(ST, guid="g2")]), "2 statements"),
    (lambda r: r["cert"].update(checks=[dict(CHECKS[0], satisfied=False)]), "not satisfied"),
    (lambda r: r.update(spoken=False, reason="entity_unresolved"), "no code"),
    (lambda r: r.update(name=None), "no item id"),
    (lambda r: r["name"]["steps"].append({"rule": "guess"}), "no code"),
])
def test_what_the_record_cannot_state_gets_no_record_and_its_reason(change, why):
    r = copy.deepcopy(REC)
    change(r)
    g, got = W.write(r, RUN)
    assert g is None and why in got


def test_the_name_statement_is_a_chain_of_decisions_ending_on_the_item():
    g, _ = W.write(REC, RUN)
    steps = [n for n in _node(g, "cheez:Decision") if str(n.get("dcterms:type")).endswith("NameStep")]
    assert [s["dcterms:type"] for s in steps] == ["cubby:FoldNameStep", "cubby:NamesakeNameStep", "cubby:FirstByIdNameStep"]
    opts = {n["@id"]: n["schema:about"] for n in _node(g, "cheez:Option") if "schema:about" in n}
    assert opts[steps[-1]["cheez:chosenOption"]] == "wd:Q79" and steps[-1]["prov:wasDerivedFrom"] == steps[1]["@id"]
    run, = _node(g, "cube:ProgramRun")
    assert steps[-1]["@id"] in run["prov:used"]
    q1 = next(n for n in _node(g, "cheez:Question") if n.get("cheez:hasResult") == steps[1]["@id"])
    assert sorted(opts[o] for o in q1["cheez:option"]) == ["wd:Q2334225", "wd:Q79"]


@pytest.mark.parametrize("precision, written", [(11, True), (9, False)])
def test_a_beginning_only_from_a_day_precise_start_qualifier(precision, written):
    r = copy.deepcopy(REC)
    r["cert"]["facts"][0]["worlds"][0]["statements"][0]["times"] = {"start": {"time": "+2020-03-01T00:00:00Z", "precision": precision}}
    fv, = _node(W.write(r, RUN)[0], "cheez:FactVersion")
    b = fv["cheez:validTime"].get("time:hasBeginning")
    assert bool(b) == written
    if written:
        assert b["time:inXSDDateTimeStamp"] == "2020-03-01T00:00:00Z" and b["prov:wasDerivedFrom"]


def test_a_refusal_carries_its_coded_reason_and_the_checks_that_ran():
    r = copy.deepcopy(REC)
    r.update(spoken=False, reason="contested", answer=None)
    r["cert"]["checks"] = CHECKS[:3] + [{"check": "conflict", "satisfied": False}]
    g, why = W.write(r, RUN)
    ref, = _node(g, "cubby:Refusal")
    assert why is None and ref["cubby:refusalReason"] == "cubby:Conflict" and ref["cubby:conflictKind"] == "cubby:ValueConflict"
    outcomes = sorted(n["cheez:outcome"] for n in _node(g, "cheez:CheckRun"))
    assert outcomes == ["cheez:Satisfied"] * 3 + ["cheez:Violated"]


def test_the_records_conform_to_cheezml_at_the_pin():
    rdflib, pyshacl = pytest.importorskip("rdflib"), pytest.importorskip("pyshacl")
    src = ROOT / "vendor" / "cheezml" / "src"
    if not (src / "shapes.ttl").exists() or "TenseCheck" not in (src / "shapes.ttl").read_text(encoding="utf-8"):
        pytest.skip("the pinned CheezML has no code for the host's checks yet")
    ctx = json.loads((src / "context.jsonld").read_text(encoding="utf-8"))["@context"]
    shapes = rdflib.Graph().parse(str(src / "shapes.ttl"), format="turtle")
    refused = dict(copy.deepcopy(REC), spoken=False, reason="ambiguous_hop", answer=None)
    for rec in (REC, refused):
        g, _ = W.write(rec, RUN)
        data = rdflib.Graph().parse(data=json.dumps({"@context": ctx, **g}), format="json-ld")
        ok, _, text = pyshacl.validate(data, shacl_graph=shapes, inference="none", advanced=True)
        assert ok, text[:1500]


def test_the_read_back_finds_a_record_that_differs_from_the_run():
    import sys
    sys.path.insert(0, str(ROOT / "validation" / "system_eval"))
    import write_cheez as X
    g, _ = W.write(REC, RUN)
    assert X.differences(g, REC, RUN) == []
    for spoil, field in ((lambda n: n.update({"prov:value": "Euro"}) if "cubby:Answer" in n["@type"] else None, "answer"),
                         (lambda n: n.update({"wikibase:rank": "wikibase:PreferredRank"}) if "cheez:FactVersion" in n["@type"] else None,
                          "fact versions"),
                         (lambda n: n.update({"cheez:hash": "c" * 64}) if "cube:Program" in n["@type"] else None, "program"),
                         (lambda n: n.update({"schema:about": "wd:Q2334225"}) if n.get("schema:about") == "wd:Q79" else None,
                          "name statement")):
        bad = copy.deepcopy(g)
        for n in bad["@graph"]:
            spoil(n)
        assert field in X.differences(bad, REC, RUN)
