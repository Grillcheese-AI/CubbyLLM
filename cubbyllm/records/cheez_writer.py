"""cubbyllm/records/cheez_writer.py -- a CheezML certificate (Comté) or refusal (Maasdam) for one answered question,
written by the host from the run's record (order 0076, step 2; CheezML at the pin in `vendor/cheezml`). Never by the
program or a model: every field is copied from the record or its run's facts, nothing is looked up or guessed.

``write(rec, run)`` -> ``(graph, None)`` or ``(None, why)``:
- ``rec``: one arm C record with its ``cert`` (`validation/system_eval/cert_facts.py`, ``CB_CERT_FACTS=1``);
- ``run``: {"run": its name, "snapshot": the store snapshot's hash, "as_of": the store's own date (an
  xsd:dateTimeStamp), "store": the world name of the statement store, "host": host id, "vm": vm id}.

A certificate carries the name statement (`cheez_name`: how the asked name reached the item, step by step). A
spoken answer gets a certificate only when its name statement names its item by id, every fact of its trace was handed by the statement store with exactly
one statement each, its program is one hash, and the question names no year (an as-of year has no instant of its own
to write: counted, never filled in). A refusal gets a record when its reason has a code (`refusals.CODE`). Every
other case is returned with its reason, for the report to count (v0.2's rule: no record rather than a wrong one).
Ids are UUIDs derived from the run, the question and the node's role, so writing twice gives the same record.

Wiring: STANDALONE (see the package docstring).
"""
from __future__ import annotations

import uuid

from .cheez_facts import RANK, fact_version, program_node, snapshot_node
from .cheez_name import name_nodes
from .refusals import CHECK, PROPOSER, refusal

__wiring__ = "STANDALONE"

NS = uuid.UUID("6f1c2b52-0b5e-4f0e-9f31-4c75c0de0076")


def uid(run: dict, rec: dict, role: str) -> str:
    return f"urn:uuid:{uuid.uuid5(NS, f'{run['run']}|{rec['id']}|{role}')}"


def _store_statements(rec: dict, run: dict) -> tuple[list | None, str | None]:
    """[(fact, statement)] for the trace, or (None, why)."""
    out = []
    for f in rec["cert"].get("facts") or []:
        here = [w for w in f["worlds"] if w["items"]]          # the worlds that handed the fact
        if not here:
            return None, "a fact no world says it handed (the live source, or a world that keeps no item)"
        store = [w for w in here if w["world"] == run["store"]]
        if not store:
            return None, f"a fact handed by a world with no statements ({here[0]['world']})"
        sts = store[0]["statements"]
        if len(sts) != 1:
            return None, f"a fact said by {len(sts)} statements of the store"
        out.append((f, sts[0]))
    return (out, None) if out else (None, "no trace")


def certificate(rec: dict, run: dict) -> tuple[dict | None, str | None]:
    cert = rec.get("cert") or {}
    if "error" in cert or not cert:
        return None, "no cert record"
    if cert.get("as_of"):
        return None, "the question names a year (an as-of year is not an instant)"
    progs = sorted(set(cert.get("programs") or []))
    if len(progs) != 1:
        return None, f"{len(progs)} programs ran"
    if PROPOSER.get(rec.get("proposer")) is None:
        return None, f"a proposer with no code ({rec.get('proposer')})"
    checks = [c for c in cert.get("checks") or [] if CHECK.get(c["check"])]
    if not checks or not all(c["satisfied"] for c in checks):
        return None, "no recorded check, or one not satisfied"
    facts, why = _store_statements(rec, run)
    if facts is None:
        return None, why
    u = lambda role: uid(run, rec, role)                  # noqa: E731
    named, last_step, why = name_nodes(rec.get("name"), u)
    if named is None:
        return None, why
    at, prog, run_id, ans = cert["at"], u("program"), u("run"), u("answer")
    graph = [program_node(prog, progs[0], PROPOSER[rec["proposer"]]), snapshot_node(u("snapshot"), run["snapshot"]),
             {"@id": run["host"], "@type": ["cheez:Host", "prov:SoftwareAgent"]},
             {"@id": run["vm"], "@type": ["cube:VM", "prov:SoftwareAgent"]}]
    graph += named
    used = [prog, last_step]
    for i, (f, st) in enumerate(facts):
        nodes = fact_version(lambda role, i=i: u(f"fact{i}:{role}"), st)
        graph += nodes
        used.append(nodes[0]["@id"])
    graph.append({"@id": run_id, "@type": ["cube:ProgramRun", "prov:Activity"], "prov:wasAssociatedWith": run["vm"],
                  "prov:used": used, "cheez:outcome": "cube:Return", "prov:endedAtTime": at,
                  "cheez:storeSnapshot": u("snapshot"), "cheez:asOf": run["as_of"]})
    check_ids = []
    for i, c in enumerate(checks):
        check_ids.append(u(f"check{i}"))
        graph.append({"@id": check_ids[-1], "@type": ["cheez:CheckRun", "prov:Activity"], "dcterms:type": CHECK[c["check"]],
                      "cheez:outcome": "cheez:Satisfied", "cheez:isSatisfiedBy": prog, "prov:used": used[2:],
                      "prov:endedAtTime": at})
    graph += [{"@id": u("question"), "@type": ["cheez:Question", "prov:Entity"], "cheez:option": prog, "cheez:hasResult": ans},
              {"@id": ans, "@type": ["cubby:Answer", "cheez:Decision", "prov:Entity"], "prov:value": rec["answer"],
               "cubby:proposer": PROPOSER[rec["proposer"]], "cheez:chosenOption": prog, "prov:wasGeneratedBy": run_id},
              {"@id": u("certificate"), "@type": ["cheez:Record", "cheez:Certificate", "prov:Entity"],
               "cheez:writtenBy": run["host"], "cheez:certifies": ans, "prov:wasGeneratedBy": run_id,
               "cheez:check": check_ids, "prov:generatedAtTime": at}]
    return {"@graph": graph}, None


def write(rec: dict, run: dict) -> tuple[dict | None, str | None]:
    """The record for one question: a certificate when it was spoken, a refusal when it was not."""
    if rec.get("spoken"):
        return certificate(rec, run)
    return refusal(rec, run, lambda role: uid(run, rec, role))


__all__ = ["NS", "RANK", "certificate", "uid", "write"]
