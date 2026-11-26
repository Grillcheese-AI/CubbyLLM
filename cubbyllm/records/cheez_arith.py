"""cubbyllm/records/cheez_arith.py -- an arithmetic trace (`arith_trace.trace`) as a CheezML record (order 0085, step 2),
written by the host from the trace alone.

- **A refusal** (Maasdam): v0.2 terms only; it conforms at the pin. The reason is coded from the trace's ``why``:
  "no try reached a value" -> `cubby:NoPlan` (no plan ran to a value: a slot copied or unknown, a malformed step, the
  VM, no stop); the vote short ("too few agree", "tie") or a rule taking out the value ("rule: ...") ->
  `cubby:NotCertified` (v0.2's AnswerShape: an answer that cannot be certified is a refusal with that reason).
- **A certificate** (Comte): v0.2 cannot say it (its certificate shapes need a stored statement read, a store
  snapshot and a listed check kind). It is written with the terms proposed in
  `docs/research/2026-11-25-cheezml-arith/PROPOSAL.md` (``PROPOSED`` below, Nick's to add); each `$N` a
  `cubby:FieldSuggestion` read from a `cubby:DocumentRegion` of the hashed question, its offsets as a W3C Web
  Annotation `oa:TextPositionSelector`; each `$K` a `prov:Entity` derived from the closed list (a `cheez:Rule` by
  hash); each step a `cube:ProgramRun` of its own `cube:Program` (the SHA-256 of what the VM ran), informed by the
  runs whose `$S` it used; the vote a check with m as a calibrated `cheez:Threshold`; the answer-kind rules that
  apply, one check each. What stays in the trace only: every try's final value, and the rules that do not apply.

``write(trace, run)`` -> ``(graph, None)`` or ``(None, why)``. ``run``: {"run", "host", "vm", "vm version", "at",
"deployment", "calibration": {"data sha256", "screen sha256", "at"}} (IRIs, an xsd:dateTime). Ids are UUIDv5 of the
run, the question id and the node's role, so writing twice gives the same record.

Wiring: STANDALONE (`validation/arith_records.py`; nothing in the serve path).
"""
from __future__ import annotations

import hashlib
import json
import uuid

from ..reasoning.slots import SLOT_RX

__wiring__ = "STANDALONE"

NS = uuid.UUID("0c5e4a1e-85a7-4c1e-9b0f-0085a7c0de85")
OA = "http://www.w3.org/ns/oa#"
PROPOSED = {"slot": "cubby:SlotCheck", "vote": "cubby:VoteCheck", "negative": "cubby:NonNegativeCheck",
            "whole": "cubby:WholeCountCheck", "money": "cubby:MoneyDecimalsCheck", "m": "cubby:SpeakVotes",
            "weights": "cheez:AdapterWeights", "settings": "cheez:Settings"}
REASON = {"no try reached a value": "cubby:NoPlan", "too few agree": "cubby:NotCertified", "tie": "cubby:NotCertified"}
SHA = "SHA-256"


def uid(run: dict, rid: str, role: str) -> str:
    return f"urn:uuid:{uuid.uuid5(NS, f'{run['run']}|{rid}|{role}')}"


def _agents(run: dict) -> list:
    vm = {"@id": run["vm"], "@type": ["cube:VM", "prov:SoftwareAgent"]}
    if run.get("vm version"):
        vm["schema:version"] = run["vm version"]
    return [{"@id": run["host"], "@type": ["cheez:Host", "prov:SoftwareAgent"]}, vm]


def refusal(tr: dict, run: dict, rid: str) -> tuple:
    why = str(tr.get("why") or "")
    code = REASON.get(why) or ("cubby:NotCertified" if why.startswith("rule:") else None)
    if code is None:
        return None, f"a refusal reason with no code ({why or 'none'})"
    u = lambda role: uid(run, rid, role)                  # noqa: E731
    ref = {"@id": u("refusal"), "@type": ["cubby:Refusal", "cheez:Decision", "prov:Entity"], "cubby:refusalReason": code,
           "cubby:proposer": "cubby:LMProgramEmitter", "prov:generatedAtTime": run["at"]}
    return {"@graph": _agents(run)[:1] + [
        {"@id": u("record"), "@type": ["cheez:Record", "prov:Entity"], "cheez:writtenBy": run["host"],
         "prov:wasDerivedFrom": ref["@id"], "prov:generatedAtTime": run["at"]},
        {"@id": u("question"), "@type": ["cheez:Question", "prov:Entity"], "cheez:hasResult": ref["@id"]}, ref]}, None


def _artifact(iri: str, sha: str, kind: "str | None" = None, name: "str | None" = None) -> dict:
    a = {"@id": iri, "@type": ["cheez:Artifact", "prov:Entity"], "cheez:hash": sha, "cheez:hashAlgorithm": SHA}
    if kind:
        a["dcterms:type"] = kind
    if name:
        a["schema:name"] = name
    return a


def _check(iri: str, kind: str, satisfied_by, used, at: str, **extra) -> dict:
    return {"@id": iri, "@type": ["cheez:CheckRun", "prov:Activity"], "dcterms:type": kind, "cheez:outcome": "cheez:Satisfied",
            "cheez:isSatisfiedBy": satisfied_by, "prov:used": used, "prov:endedAtTime": at, **extra}


def certificate(tr: dict, run: dict, rid: str, reader: str = "cubbyllm.reasoning.slots.extract") -> tuple:
    from ..reasoning.arith_world import CONVERSIONS
    stop = tr.get("stop") or ""
    if not stop.startswith("$S"):
        return None, f"the answer is a slot the VM did not compute ({stop or 'none'})"
    fz, cal = tr.get("freeze") or {}, run.get("calibration") or {}
    if not (fz.get("adapter", {}).get("sha256") and cal.get("data sha256")):
        return None, "no adapter hash or no calibration in the record (the freeze)"
    u = lambda role: uid(run, rid, role)                  # noqa: E731
    at = run["at"]
    q, adapter = u("question text"), u("adapter")
    graph = _agents(run) + [_artifact(q, tr["question_sha256"], name="the question"),
                            _artifact(adapter, fz["adapter"]["sha256"], PROPOSED["weights"], fz["adapter"].get("name"))]
    ent: dict = {}
    for s in tr["slots"]:
        if s["kind"] == "N" and s["id"] not in ent:
            reg = u(f"region {s['id']} {s['start']}")
            graph.append({"@id": reg, "@type": ["cubby:DocumentRegion", "prov:Entity"], "schema:text": s["text"],
                          "cubby:readerVersion": reader, "schema:isPartOf": q,
                          "oa:hasSelector": {"@type": "oa:TextPositionSelector", "oa:start": s["start"], "oa:end": s["end"]}})
            ent[s["id"]] = u(f"slot {s['id']}")
            graph.append({"@id": ent[s["id"]], "@type": ["cubby:FieldSuggestion", "prov:Entity"], "prov:value": s["value"],
                          "cubby:region": reg, "schema:identifier": s["id"]})
        elif s["kind"] == "K":
            lst = u("conversions")
            if lst not in [n["@id"] for n in graph]:
                text = json.dumps([[f.text, f.value] for f in CONVERSIONS])
                graph.append({"@id": lst, "@type": ["cheez:Rule", "prov:Entity"], "schema:name": "the arithmetic world's conversions",
                              "cheez:hash": hashlib.sha256(text.encode()).hexdigest(), "cheez:hashAlgorithm": SHA})
            ent[s["id"]] = u(f"slot {s['id']}")
            graph.append({"@id": ent[s["id"]], "@type": "prov:Entity", "prov:value": s["value"], "schema:name": s["fact"],
                          "schema:identifier": s["id"], "prov:wasDerivedFrom": lst})
    progs, made_by = [], {}
    for st in tr["steps"]:
        j = st["k"]
        prog, run_j, val = u(f"program {j}"), u(f"run {j}"), u(f"slot $S{j + 1}")
        names = list(dict.fromkeys(m.group(0) for m in SLOT_RX.finditer(st["program"])))
        if any(n not in ent for n in names):
            return None, f"step {j} names a slot the record does not have"
        informed = [made_by[n] for n in names if n in made_by]
        graph += [{"@id": prog, "@type": ["cube:Program", "cheez:Option", "prov:Entity"], "cheez:hash": st["ran_sha256"],
                   "cheez:hashAlgorithm": SHA, "cubby:proposer": "cubby:LMProgramEmitter", "prov:wasDerivedFrom": adapter},
                  {"@id": run_j, "@type": ["cube:ProgramRun", "prov:Activity"], "prov:wasAssociatedWith": run["vm"],
                   "prov:used": [prog] + [ent[n] for n in names], "cheez:outcome": "cube:Return", "prov:endedAtTime": at,
                   **({"prov:wasInformedBy": informed} if informed else {})},
                  {"@id": val, "@type": "prov:Entity", "prov:value": st["value"], "schema:identifier": f"$S{j + 1}",
                   "prov:wasGeneratedBy": run_j}]
        progs.append(prog)
        ent[f"$S{j + 1}"], made_by[f"$S{j + 1}"] = val, run_j
    if stop not in made_by:
        return None, f"the stop names {stop}, which no step made"
    last = made_by[stop]
    chosen = progs[[u(f"run {j}") for j in range(len(progs))].index(last)]
    rule, settings, thr, calib = u("speak rule"), u("settings"), u("threshold m"), u("calibration m")
    graph += [{"@id": rule, "@type": ["cheez:Rule", "prov:Entity"], "schema:name": "the speak rule (speak_vote, answer_kind)",
               "cheez:hash": fz["speak rule"]["sha256"], "cheez:hashAlgorithm": SHA},
              _artifact(settings, fz["settings"]["sha256"], PROPOSED["settings"]),
              _artifact(u("screen m"), cal["screen sha256"] or fz["speak rule"]["sha256"]),
              _artifact(u("calibration data"), cal["data sha256"]),
              {"@id": calib, "@type": ["cheez:Calibration", "prov:Activity"], "cheez:screen": u("screen m"),
               "cheez:calibrationData": [u("calibration data")], "prov:endedAtTime": cal.get("at") or at,
               "cheez:encoder": adapter, "cheez:deployment": run["deployment"]},
              {"@id": thr, "@type": ["cheez:Threshold", "prov:Entity"], "dcterms:type": PROPOSED["m"],
               "cheez:thresholdValue": {"@value": str(tr["m"]), "@type": "xsd:decimal"}, "prov:wasGeneratedBy": calib}]
    checks = [_check(u("check slots"), PROPOSED["slot"], progs, [ent[s["id"]] for s in tr["slots"] if s["kind"] in "NK"], at),
              _check(u("check vote"), PROPOSED["vote"], chosen, [rule, settings], at, **{
                  "cheez:observedValue": {"@value": str(tr["vote"]["votes"]), "@type": "xsd:decimal"}, "cheez:threshold": thr})]
    for c in tr.get("checks") or []:
        if c["outcome"] == "violated":
            return None, f"a spoken answer with a violated rule ({c['rule']})"
        if c["outcome"] == "satisfied":
            checks.append(_check(u(f"check {c['rule']}"), PROPOSED[c["rule"]], chosen, [rule], at))
    ans, cert = u("answer"), u("certificate")
    graph += checks + [
        {"@id": u("question"), "@type": ["cheez:Question", "prov:Entity"], "cheez:option": chosen, "cheez:hasResult": ans},
        {"@id": ans, "@type": ["cubby:Answer", "cheez:Decision", "prov:Entity"], "prov:value": tr["answer"],
         "cubby:proposer": "cubby:LMProgramEmitter", "cheez:chosenOption": chosen, "prov:wasGeneratedBy": last},
        {"@id": cert, "@type": ["cheez:Record", "cheez:Certificate", "prov:Entity"], "cheez:writtenBy": run["host"],
         "cheez:certifies": ans, "prov:wasGeneratedBy": last, "cheez:check": [c["@id"] for c in checks],
         "prov:generatedAtTime": at}]
    return {"@graph": graph}, None


def write(tr: dict, run: dict, rid: str) -> tuple:
    return certificate(tr, run, rid) if tr["kind"] == "certificate" else refusal(tr, run, rid)


__all__ = ["OA", "PROPOSED", "REASON", "certificate", "refusal", "uid", "write"]
