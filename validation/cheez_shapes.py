"""cheez_shapes - order 0085, line 1: the shape check that rules on a CheezML record, at the pin (`vendor/cheezml`).

``check(doc)`` -> {"conforms": bool, "messages": [...], "outside": [...], "undefined": [...], "ok": bool}:
- the record (JSON-LD; the pin's `src/context.jsonld` is put in when it has no context) against the pin's
  `src/shapes.ttl` (pySHACL, no inference, as the pin's own `tools/validate.py` runs it);
- every IRI inside the namespaces the pin allows (its `validate.ALLOWED`: nothing minted elsewhere);
- every CheezML term used (`https://w3id.org/cheezml/...`) defined in the pin's vocabulary (`docs/v0.2/*.ttl`).
``ok`` is all three. `examples()` runs it on the pin's 8 examples and 15 negative ones: it must accept all 8 and
reject all 15 before it rules on any record (Gemini's review, 2026-11-25, on Nick's yes).

Needs rdflib and pyshacl (not in the repository's venv: `uv run --no-project --with rdflib --with pyshacl`).

    python validation/cheez_shapes.py --examples
    python validation/cheez_shapes.py <records.jsonl> [--proposal]   (the writer's records: their "cheez" field)

Wiring: STANDALONE (order 0085).
"""
from __future__ import annotations

import glob
import json
import os
import sys

__wiring__ = "STANDALONE"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = os.path.join(ROOT, "vendor", "cheezml")
NS = "https://w3id.org/cheezml/"
ALLOWED = ("https://w3id.org/cheezml/", "urn:uuid:", "http://www.w3.org/ns/prov#", "https://schema.org/",
           "http://www.w3.org/2006/time#", "http://www.w3.org/2004/02/skos/core#", "http://purl.org/dc/terms/",
           "http://www.w3.org/2001/XMLSchema#", "http://www.w3.org/ns/shacl#", "http://creativecommons.org/",
           "https://creativecommons.org/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://wikiba.se/ontology#",
           "http://www.wikidata.org/entity/", "http://www.wikidata.org/prop/direct/")   # the pin's validate.ALLOWED
_cache: dict = {}


def _pin():
    if not _cache:
        import rdflib
        _cache["shapes"] = rdflib.Graph().parse(os.path.join(PIN, "src", "shapes.ttl"), format="turtle")
        _cache["context"] = json.load(open(os.path.join(PIN, "src", "context.jsonld"), encoding="utf-8"))["@context"]
        terms = set()
        for f in glob.glob(os.path.join(PIN, "docs", "v0.2", "*.ttl")):
            if os.path.basename(f) == "shapes.ttl":
                continue
            g = rdflib.Graph().parse(f, format="turtle")
            terms |= {str(s) for s in g.subjects() if str(s).startswith(NS)}
        _cache["terms"] = terms
    return _cache


def check(doc: dict, shapes=None, terms=(), allowed=(), context: "dict | None" = None) -> dict:
    """``shapes``, ``terms``, ``allowed``, ``context``: a proposal's shapes graph, and the terms, namespaces and
    context entries it adds (`cheez_proposal`); by default the pin's alone."""
    import pyshacl
    import rdflib
    from rdflib import RDF, Literal, URIRef
    from rdflib.namespace import SH
    pin = _pin()
    if "@context" not in doc:
        doc = {"@context": {**pin["context"], **(context or {})}, **doc}
    ok_ns = ALLOWED + tuple(allowed)
    g = rdflib.Graph().parse(data=json.dumps(doc), format="json-ld")
    conforms, rg, _ = pyshacl.validate(g, shacl_graph=shapes if shapes is not None else pin["shapes"], inference="none",
                                       advanced=False, allow_warnings=False)
    top = set(rg.objects(None, SH.result))
    msgs = sorted({str(m) for r in top for m in rg.objects(r, SH.resultMessage)})
    outside, used = set(), set()
    for t in g:
        for x in t:
            if isinstance(x, URIRef) and str(x) != str(RDF.type):
                if not str(x).startswith(ok_ns):
                    outside.add(str(x))
                if str(x).startswith(NS) and "/shapes" not in str(x):
                    used.add(str(x))
            elif isinstance(x, Literal) and x.datatype is not None and not str(x.datatype).startswith(ok_ns):
                outside.add("datatype " + str(x.datatype))
    undefined = sorted(u for u in used if u not in pin["terms"] and u not in set(terms))
    return {"conforms": bool(conforms), "messages": msgs, "outside": sorted(outside), "undefined": undefined,
            "ok": bool(conforms) and not outside and not undefined}


def examples() -> dict:
    """The pin's examples through `check`: {"accepted": [...], "rejected": [...], "line": all 8 in, all 15 out}."""
    ex = os.path.join(PIN, "docs", "v0.2", "examples")
    pos = sorted(glob.glob(os.path.join(ex, "*.jsonld")))
    neg = sorted(glob.glob(os.path.join(ex, "negative", "*.jsonld")))
    res = {os.path.basename(p): check(json.load(open(p, encoding="utf-8")))["ok"] for p in pos + neg}
    acc = [os.path.basename(p) for p in pos if res[os.path.basename(p)]]
    rej = [os.path.basename(p) for p in neg if not res[os.path.basename(p)]]
    return {"positive": len(pos), "negative": len(neg), "accepted": acc, "rejected": rej,
            "line": len(pos) == 8 and len(neg) == 15 and len(acc) == 8 and len(rej) == 15}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "--examples":
        print(json.dumps(examples(), indent=1))
        return
    over = {}
    if "--proposal" in argv:                               # the terms order 0085 proposes, applied in memory
        import cheez_proposal as P
        over = {"shapes": P.shapes(), "terms": P.TERMS, "allowed": P.ALLOWED, "context": P.CONTEXT}
    n, none, bad, kinds = 0, 0, [], {}
    for ln in open(argv[0], encoding="utf-8"):
        d = json.loads(ln)
        doc = d["cheez"] if "cheez" in d else d.get("record", d)
        if doc is None:                                    # a trace the writer gave no CheezML record (its why kept)
            none += 1
            continue
        r = check(doc, **over)
        n += 1
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        if not r["ok"]:
            bad.append({"id": d.get("id"), "kind": d.get("kind"), **{k: r[k] for k in ("messages", "outside", "undefined")}})
    print(json.dumps({"against": "the proposal" if over else "the pin", "records": n, "by kind": kinds,
                      "conform": n - len(bad), "no CheezML record": none, "not": bad[:20]}, indent=1))


__all__ = ["ALLOWED", "check", "examples"]

if __name__ == "__main__":
    main()
