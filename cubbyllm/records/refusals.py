"""cubbyllm/records/refusals.py -- a CheezML refusal (Maasdam) from a record that was not spoken, and the codes the
writer maps the host's words to (order 0076, step 2).

A refusal's reason is coded from the host's own reason word (the record's ``reason``, before its first colon). A
word with no code here gets no record, and the report counts it: no reason is guessed. Checks that ran on the
answer before it was refused are kept on the refusal, each with its outcome.

Wiring: STANDALONE (`cheez_writer` calls it).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

PROPOSER = {"everyday": "cubby:EverydayTable", "450m": "cubby:LMProgramEmitter", "host": "cubby:HostGrammar",
            "rules": "cubby:OvernightRuleLibrary"}
CHECK = {"answer class": "cubby:AnswerClassCheck", "tense": "cubby:TenseCheck", "snapshot time": "cubby:SnapshotTimeCheck",
         "time rule": "cubby:TimeRuleCheck", "conflict": "cubby:ConflictCheck"}
CODE = {"no_plan": "cubby:NoPlan", "plan_does_not_cover_question": "cubby:NoPlan", "emit_error": "cubby:NoPlan",
        "unknown_relation": "cubby:RelationMissing", "relation_missing": "cubby:RelationMissing",
        "ambiguous_relation": "cubby:Ambiguous", "ambiguous_entity": "cubby:Ambiguous", "ambiguous_hop": "cubby:Ambiguous",
        "retrieval_exhausted": "cubby:FactMissing", "walk_failed": "cubby:WalkFailed",
        "vm_verify_failed": "cubby:NotCertified", "contested": "cubby:Conflict"}
CONFLICT = {"contested": "cubby:ValueConflict"}


def head(reason) -> str:
    return str(reason or "").split(":")[0].strip()


def refusal(rec: dict, run: dict, u) -> tuple[dict | None, str | None]:
    cert = rec.get("cert") or {}
    if not cert or "error" in cert:
        return None, "no cert record"
    word = head(rec.get("reason"))
    code = CODE.get(word)
    if code is None:
        return None, f"a refusal reason with no code ({word or 'none'})"
    checks = [c for c in cert.get("checks") or [] if c["check"] in CHECK]
    ref = {"@id": u("refusal"), "@type": ["cubby:Refusal", "cheez:Decision", "prov:Entity"], "cubby:refusalReason": code,
           "prov:generatedAtTime": cert["at"]}
    if word in CONFLICT:
        ref["cubby:conflictKind"] = CONFLICT[word]
    if PROPOSER.get(rec.get("proposer")):
        ref["cubby:proposer"] = PROPOSER[rec["proposer"]]
    graph = [{"@id": run["host"], "@type": ["cheez:Host", "prov:SoftwareAgent"]},
             {"@id": u("record"), "@type": ["cheez:Record", "prov:Entity"], "cheez:writtenBy": run["host"],
              "prov:wasDerivedFrom": ref["@id"], "prov:generatedAtTime": cert["at"]},
             {"@id": u("question"), "@type": ["cheez:Question", "prov:Entity"], "cheez:hasResult": ref["@id"]}, ref]
    if checks:
        ref["cheez:check"] = []
        for i, c in enumerate(checks):
            ref["cheez:check"].append(u(f"check{i}"))
            graph.append({"@id": u(f"check{i}"), "@type": ["cheez:CheckRun", "prov:Activity"], "dcterms:type": CHECK[c["check"]],
                          "cheez:outcome": "cheez:Satisfied" if c["satisfied"] else "cheez:Violated",
                          "prov:endedAtTime": cert["at"]})
    return {"@graph": graph}, None


__all__ = ["CHECK", "CODE", "CONFLICT", "PROPOSER", "head", "refusal"]
