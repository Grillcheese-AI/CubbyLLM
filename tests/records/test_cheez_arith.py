"""Order 0085, lines 1 and 2's CheezML side (cubbyllm/records/cheez_arith.py, validation/cheez_shapes.py,
validation/cheez_proposal.py): the shape check accepts the pin's 8 examples and rejects its 15 broken ones before it
rules on a record; an arithmetic refusal conforms at the pin; a certificate does not (v0.2 has no terms for it) and
conforms under the proposal, which keeps every example's verdict. The shape tests skip without rdflib and pyshacl
(`uv run --no-project --with rdflib --with pyshacl python -m pytest tests/records/test_cheez_arith.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "validation", ROOT / "tests" / "reasoning", ROOT / "tests" / "validation", ROOT / "tests" / "records"):
    sys.path.insert(0, str(p))
import arith_freeze as F  # noqa: E402
from test_arith_cert import _chains  # noqa: E402
from test_step_loop import QUESTION  # noqa: E402

from cubbyllm.reasoning.answer_kind import RULES  # noqa: E402
from cubbyllm.records import cheez_arith as C  # noqa: E402
from cubbyllm.records.arith_trace import trace  # noqa: E402

RUN = {"run": "test", "host": "urn:uuid:00000000-0000-0000-0000-000000000001",
       "vm": "urn:uuid:00000000-0000-0000-0000-000000000002", "vm version": "efa2559", "at": "2026-11-25T12:00:00Z",
       "deployment": "urn:uuid:00000000-0000-0000-0000-000000000003",
       "calibration": {"data sha256": "ab" * 32, "screen sha256": None, "at": "2026-11-24T18:00:00Z"}}


def _traces(tmp_path):
    ad = tmp_path / "s1"
    ad.mkdir()
    (ad / F.WEIGHTS).write_bytes(b"weights")
    fz = F.freeze(str(ad), {"k": 4, "m": 2, "temperature": 0.7, "top-k": 40, "rules": list(RULES)}, None)
    keep, _ = _chains()
    return trace(QUESTION, keep, 4, 2, RULES, fz), trace(QUESTION, keep, 4, 3, RULES, fz)


def test_the_writer_is_deterministic_and_codes_the_reason(tmp_path):
    cert, ref = _traces(tmp_path)
    g1, why = C.write(cert, RUN, "q1")
    assert why is None and g1 == C.write(cert, RUN, "q1")[0]
    types = [t for n in g1["@graph"] for t in (n["@type"] if isinstance(n["@type"], list) else [n["@type"]])]
    assert types.count("cube:ProgramRun") == 3 and types.count("cubby:FieldSuggestion") == 5   # 308, 2, 6, 185, "one"
    ans = next(n for n in g1["@graph"] if "cubby:Answer" in n["@type"])
    assert ans["prov:value"] == "1109" and ans["prov:wasGeneratedBy"] == C.uid(RUN, "q1", "run 2")
    runs = {n["@id"]: n for n in g1["@graph"] if "cube:ProgramRun" in n["@type"]}
    assert runs[C.uid(RUN, "q1", "run 2")]["prov:wasInformedBy"] == [C.uid(RUN, "q1", "run 1")]
    g2, _ = C.write(ref, RUN, "q2")
    r = next(n for n in g2["@graph"] if "cubby:Refusal" in n["@type"])
    assert r["cubby:refusalReason"] == "cubby:NotCertified"
    assert C.write({**ref, "why": "no try reached a value"}, RUN, "q2")[0]["@graph"][-1]["cubby:refusalReason"] == "cubby:NoPlan"
    assert C.write({**ref, "why": "?"}, RUN, "q2") == (None, "a refusal reason with no code (?)")
    assert C.write({**cert, "stop": "$N1"}, RUN, "q1")[1].startswith("the answer is a slot the VM did not compute")


def _shapes():
    pytest.importorskip("rdflib")
    pytest.importorskip("pyshacl")
    if not (ROOT / "vendor" / "cheezml" / "src" / "shapes.ttl").exists():
        pytest.skip("vendor/cheezml is not checked out")
    import cheez_proposal as P
    import cheez_shapes as S
    return S, P


def test_the_shape_check_is_tested_on_the_pins_examples_first():
    S, P = _shapes()
    ex = S.examples()
    assert ex["line"] and len(ex["accepted"]) == 8 and len(ex["rejected"]) == 15


def test_a_refusal_conforms_at_the_pin_and_a_certificate_under_the_proposal(tmp_path):
    S, P = _shapes()
    cert, ref = _traces(tmp_path)
    assert S.check(C.write(ref, RUN, "q2")[0])["ok"]
    g = C.write(cert, RUN, "q1")[0]
    at_pin = S.check(g, context=P.CONTEXT)
    assert not at_pin["ok"] and any("statement version" in m for m in at_pin["messages"])
    assert set(at_pin["undefined"]) == P.TERMS - {P.NS_CUBBY + "WholeCountCheck"}   # "how much": that rule does not apply
    prop = S.check(g, P.shapes(), P.TERMS, P.ALLOWED, P.CONTEXT)
    assert prop["ok"], prop
    import copy
    blind = copy.deepcopy(g)                                        # runs that used no number of the question: refused
    for n in blind["@graph"]:
        if "cube:ProgramRun" in n["@type"]:
            n["prov:used"] = [x for x in n["prov:used"] if "program" in x or x == n["prov:used"][0]]
    nb = S.check(blind, P.shapes(), P.TERMS, P.ALLOWED, P.CONTEXT)
    assert not nb["ok"] and any("a number of the question" in m for m in nb["messages"]), nb
    bad = {**cert, "steps": [dict(cert["steps"][0], value="155")] + cert["steps"][1:]}
    assert C.write(bad, RUN, "q1")[0] != g                          # a changed value is a changed record


def test_the_proposal_keeps_every_examples_verdict():
    S, P = _shapes()
    import glob
    import json
    sh = P.shapes()
    ex = ROOT / "vendor" / "cheezml" / "docs" / "v0.2" / "examples"
    for f in sorted(glob.glob(str(ex / "*.jsonld"))):
        assert S.check(json.load(open(f, encoding="utf-8")), sh, P.TERMS, P.ALLOWED)["ok"], f
    for f in sorted(glob.glob(str(ex / "negative" / "*.jsonld"))):
        assert not S.check(json.load(open(f, encoding="utf-8")), sh, P.TERMS, P.ALLOWED)["ok"], f
