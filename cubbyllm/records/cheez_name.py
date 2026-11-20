"""cubbyllm/records/cheez_name.py -- the name statement in a certificate (order 0076, step 2; Cowork's ruling of
2026-11-19): how the question's name reached the item the answer is about, in CheezML v0.2's decision terms.

Each step of the record's name statement (`reasoning.name_statement`, kept by run_c as ``name``) is one
`cheez:Question` over the items the step kept (each a `cheez:Option` that is `schema:about` its Wikidata item) and
its `cheez:Decision`, the rule as `dcterms:type` (`cubby:NameSteps`), choosing one. A step that matched the asked name to an item's own label
or alias (fold, label, alias) has the one item it named; a step that compared namesakes (namesake, first by id,
place qualifier) has every item it kept and the one it chose. The steps are chained (`prov:wasDerivedFrom` the step
before), and the first step's question holds the asked name (`schema:name`). The run
that certified the answer used the last step. Only items named by id are written; a step that names none is no
record (counted by the writer).

Wiring: STANDALONE (`cheez_writer` calls it).
"""
from __future__ import annotations

__wiring__ = "STANDALONE"

STEP = {"fold": "cubby:FoldNameStep", "label": "cubby:LabelNameStep", "alias": "cubby:AliasNameStep",
        "namesake": "cubby:NamesakeNameStep", "first_by_id": "cubby:FirstByIdNameStep",
        "place_qualifier": "cubby:PlaceQualifierNameStep"}
MATCH = ("fold", "label", "alias")


def name_nodes(name: dict | None, u) -> tuple[list | None, str | None, str | None]:
    """(nodes, the last step's id, why not): ``u(role)`` gives each node's id."""
    if not name or not (name.get("item") or {}).get("id"):
        return None, None, "a name statement with no item id"
    item, steps = name["item"]["id"], name.get("steps") or []
    if not steps or any(s.get("rule") not in STEP for s in steps):
        return None, None, f"a name step with no code ({[s.get('rule') for s in steps if s.get('rule') not in STEP]})"
    nodes, options, prev = [], {}, None

    def option(q: str) -> str:
        if q not in options:
            options[q] = u(f"name:option:{q}")
            nodes.append({"@id": options[q], "@type": ["cheez:Option", "prov:Entity"], "schema:about": f"wd:{q}"})
        return options[q]
    for i, s in enumerate(steps):
        kept = [item] if s["rule"] in MATCH else list(s.get("kept") or [item])
        chose = s.get("chose") or (item if item in kept else None)
        if chose is None:
            return None, None, f"a {s['rule']} step that chose no item"
        q = {"@id": u(f"name:question{i}"), "@type": ["cheez:Question", "prov:Entity"],
             "cheez:option": [option(x) for x in kept], "cheez:hasResult": u(f"name:step{i}")}
        if i == 0:
            q["schema:name"] = name.get("asked") or ""
        d = {"@id": u(f"name:step{i}"), "@type": ["cheez:Decision", "prov:Entity"], "dcterms:type": STEP[s["rule"]],
             "cheez:chosenOption": option(chose)}
        if prev:
            d["prov:wasDerivedFrom"] = prev
        nodes += [q, d]
        prev = d["@id"]
    if nodes[-1]["cheez:chosenOption"] != options.get(item):
        return None, None, "the name statement's last step did not choose its item"
    return nodes, prev, None


__all__ = ["MATCH", "STEP", "name_nodes"]
