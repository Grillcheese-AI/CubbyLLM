"""cubbyllm/records/cheez_facts.py -- the nodes of a CheezML record that come from the store (order 0076, step 2):
a stored statement as a `cheez:FactVersion` (CheezML a99a774: `rdf:subject` / `rdf:predicate` / `rdf:object` with
Wikidata IRIs, `wikibase:rank`, the statement id as `schema:identifier`, the item's revision as `schema:version` on
its source), the store snapshot and the program, each by hash.

A valid time's beginning is written only from the statement's own start qualifier, at day precision or finer (an
instant states a day; a year-precision start has no instant of its own), with the qualifier as what it was derived
from. Otherwise the interval has no beginning (Nick, 2026-11-19: never a date the source did not give).

Wiring: STANDALONE (`cheez_writer` calls it).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

RANK = {0: "wikibase:DeprecatedRank", 1: "wikibase:NormalRank", 2: "wikibase:PreferredRank"}
SNAPSHOT_HASH = "BLAKE3"                                  # wdstore's snapshot hash (its src/snapshot.rs)
PROGRAM_HASH = "SHA-256"                                  # `cert_facts.note_program`
DAY = 11                                                  # Wikibase time precision: 11 = a day


def value(v) -> dict | None:
    """A store value as JSON-LD: an item by its IRI, a typed literal, a language-tagged string, a string."""
    if not isinstance(v, dict) or not v:
        return {"@value": str(v)} if v is not None else None
    kind, x = next(iter(v.items()))
    if kind == "entity":
        return {"@id": f"wd:{x}"}
    if kind == "typed":
        return {"@value": str(x[0])}
    if kind == "mono":
        return {"@value": x[0], "@language": x[1]}
    return {"@value": str(x)}


def _instant(raw: str) -> str | None:
    """"+2020-01-01T00:00:00Z" -> "2020-01-01T00:00:00Z" (a year in four digits, AD), else None."""
    s = raw[1:] if raw[:1] in "+-" else raw
    return s if raw[:1] != "-" and len(s.split("-")[0]) == 4 else None


def fact_version(u, st: dict) -> list:
    """[the fact version, its source, its start qualifier when written]; ``u(role)`` gives each node's id."""
    fv = {"@id": u("fact"), "@type": ["cheez:FactVersion", "prov:Entity"], "rdf:subject": f"wd:{st['item']}",
          "rdf:predicate": f"wdt:{st['pid']}", "rdf:object": value(st["value"]), "wikibase:rank": RANK[st["rank"]],
          "prov:hadPrimarySource": u("source"), "cheez:validTime": {"@type": "time:Interval"},
          "prov:generatedAtTime": st["modified"]}
    if st.get("guid"):
        fv["schema:identifier"] = st["guid"]
    out = [fv, {"@id": u("source"), "@type": "prov:Entity", "schema:identifier": st["item"],
                "schema:version": str(st["revision"])}]
    start = (st.get("times") or {}).get("start")
    at = _instant(start["time"]) if start and (start.get("precision") or 0) >= DAY else None
    if at:
        fv["cheez:validTime"]["time:hasBeginning"] = {"@type": "time:Instant", "time:inXSDDateTimeStamp": at,
                                                      "prov:wasDerivedFrom": u("start")}
        out.append({"@id": u("start"), "@type": "prov:Entity", "rdf:predicate": "wdt:P580",
                    "schema:identifier": f"{st['guid']} P580"})
    return out


def program_node(nid: str, sha: str, proposer: str) -> dict:
    return {"@id": nid, "@type": ["cube:Program", "cheez:Option", "prov:Entity"], "cheez:hash": sha,
            "cheez:hashAlgorithm": PROGRAM_HASH, "cubby:proposer": proposer}


def snapshot_node(nid: str, h: str) -> dict:
    return {"@id": nid, "@type": ["cheez:Artifact", "prov:Entity"], "dcterms:type": "cheez:StoreSnapshot",
            "cheez:hash": h, "cheez:hashAlgorithm": SNAPSHOT_HASH}


__all__ = ["DAY", "RANK", "fact_version", "program_node", "snapshot_node", "value"]
