"""Validate the CheezML v0.1 synthesis: shapes parse, positives conform, negatives fail for their stated reason,
IRIs stay inside the allowed namespaces, a second JSON-LD processor agrees, and every CheezML term used is defined."""
import glob
import io
import json
import os
import re
import sys

import pyshacl
import rdflib
from rdflib import RDF, Literal, URIRef
from rdflib.namespace import SH

SYNTH = sys.argv[1]
out = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=out)


ALLOWED = (
    "https://w3id.org/cheezml/",
    "urn:uuid:",
    "http://www.w3.org/ns/prov#",
    "https://schema.org/",
    "http://www.w3.org/2006/time#",
    "http://www.w3.org/2004/02/skos/core#",
    "http://purl.org/dc/terms/",
    "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/ns/shacl#",
)
RDF_TYPE = str(RDF.type)  # produced by the JSON-LD "@type" keyword itself

EXPECT = {
    "n01-certificate-by-agreement": "Only the VM certifies",
    "n01b-certificate-truth-score": "no truth score",
    "n02-refusal-without-reason": "exactly one reason code",
    "n02b-conflict-without-kind": "names its conflict kind",
    "n03-threshold-without-calibration": "every threshold carries its calibration",
    "n03b-route-against-margin": "The outcome must follow from the values",
    "n04-split-beyond-zmax": "No split at or beyond z_max",
    "n05-rejection-without-breaking-case": "kept with the case that broke it",
    "n06-commit-without-source": "names its source",
    "n06b-number-not-in-region": "must appear in its region's text",
}

failures = []

# ---------- shapes ----------
sg = rdflib.Graph().parse(os.path.join(SYNTH, "shapes.ttl"), format="turtle")
node_shapes = set(sg.subjects(RDF.type, SH.NodeShape))
targeted = {s for s in node_shapes if (s, SH.targetClass, None) in sg}
prop_shapes = list(sg.objects(None, SH.property))
sparql = list(sg.objects(None, SH.sparql))
all_prop = set(sg.subjects(SH.path, None))  # every property shape, nested ones included
no_msg = [ps for ps in all_prop if (ps, SH.message, None) not in sg]
no_msg_sparql = [c for c in sparql if (c, SH.message, None) not in sg]
CUSTOM = {str(m) for m in sg.objects(None, SH.message)}
p(f"shapes.ttl: parsed, {len(sg)} triples, {len(node_shapes)} named node shapes "
  f"({len(targeted)} with targets), {len(all_prop)} property constraints (nested included), {len(sparql)} SPARQL constraints")
p(f"  property constraints without sh:message: {len(no_msg)}; SPARQL constraints without sh:message: {len(no_msg_sparql)}")
if no_msg or no_msg_sparql:
    failures.append("shape constraints without sh:message")
logical_no_msg = [s for s in node_shapes
                  if any((s, k, None) in sg for k in (SH["or"], SH.xone, SH.closed, SH["not"], SH["class"]))
                  and (s, SH.message, None) not in sg]
p(f"  named shapes with node-level constraints but no sh:message: {len(logical_no_msg)}")
if logical_no_msg:
    failures.append("node-level constraints without sh:message")


def iri_check(g):
    bad = set()
    for t in g:
        for x in t:
            if isinstance(x, URIRef):
                s = str(x)
                if s == RDF_TYPE:
                    continue
                if not s.startswith(ALLOWED):
                    bad.add(s)
            elif isinstance(x, Literal) and x.datatype is not None:
                if not str(x.datatype).startswith(ALLOWED):
                    bad.add("datatype " + str(x.datatype))
    return bad


def ill_typed(g):
    return [o for o in g.objects() if isinstance(o, Literal) and getattr(o, "ill_typed", False)]


def placeholder_in_iri(g):
    return {str(x) for t in g for x in t if isinstance(x, URIRef) and "PLACEHOLDER" in str(x).upper()}


def validate(path):
    raw = open(path).read()
    doc = json.loads(raw)
    ctx_inline = isinstance(doc.get("@context"), dict)
    g = rdflib.Graph().parse(data=raw, format="json-ld")
    conforms, rg, text = pyshacl.validate(g, shacl_graph=sg, inference="none", advanced=False,
                                          allow_warnings=False)
    top = set(rg.objects(None, SH.result))  # top-level results only (sh:detail results are nested evidence)
    msgs = sorted({str(m) for r in top for m in rg.objects(r, SH.resultMessage)})
    defaults = [r for r in top if not any(str(m) in CUSTOM for m in rg.objects(r, SH.resultMessage))]
    return g, conforms, msgs, len(top), ctx_inline, len(defaults)


# ---------- second processor (pyld) ----------
try:
    from pyld import jsonld

    def pyld_count(path):
        doc = json.load(open(path))
        nq = jsonld.to_rdf(doc, {"format": "application/n-quads"})
        return len([ln for ln in nq.splitlines() if ln.strip()])
except Exception as e:  # pragma: no cover
    jsonld = None
    p("pyld unavailable:", e)

# ---------- positives ----------
p("\nPOSITIVE EXAMPLES (must conform)")
pos_files = sorted(glob.glob(os.path.join(SYNTH, "examples", "*.jsonld")))
used_terms = set()
for f in pos_files:
    g, conforms, msgs, n, inline, nd = validate(f)
    bad = iri_check(g)
    ph = placeholder_in_iri(g)
    it = ill_typed(g)
    pc = pyld_count(f) if jsonld else None
    ok = conforms and not bad and not ph and not it and inline and (pc is None or pc == len(g))
    p(f"  {os.path.basename(f)}: triples={len(g)} (pyld {pc}), context inline={inline}, conforms={conforms}, "
      f"results={n}, IRIs outside allowed={len(bad)}, PLACEHOLDER in IRI={len(ph)}, ill-typed literals={len(it)}"
      f"  -> {'PASS' if ok else 'FAIL'}")
    for m in msgs:
        p("      message:", m)
    for b in sorted(bad):
        p("      bad IRI:", b)
    if not ok:
        failures.append(os.path.basename(f))
    for t in g:
        for x in t:
            if isinstance(x, URIRef) and str(x).startswith("https://w3id.org/cheezml/"):
                used_terms.add(str(x))

# ---------- target coverage: every targeted shape has focus nodes in the positives (no vacuous pass) ----------
allpos = rdflib.Graph()
for f in pos_files:
    allpos.parse(f, format="json-ld")
p("\nTARGET COVERAGE (focus nodes per targeted shape across the six positives)")
MAIN = ["cubby:TurnShape", "cubby:CertifiedAnswerTurnShape", "cubby:AnswerShape", "cheez:CertificateShape",
        "cheez:CertificateConsistencyShape", "cubby:RefusalShape", "cubby:RefusalConflictShape",
        "mowm:RoutingDecisionShape", "mowm:RoutingBestShape", "mowm:RoutingRunnerUpShape", "mowm:RoutingOutcomeShape",
        "mowm:WorldSplitShape", "cubby:HelperRejectionShape", "cubby:FieldCommitShape"]
zero = []
for s in sorted(targeted, key=lambda x: sg.namespace_manager.qname(x)):
    qn = sg.namespace_manager.qname(s)
    nodes = set()
    for c in sg.objects(s, SH.targetClass):
        nodes |= set(allpos.subjects(RDF.type, c))
    if not nodes:
        zero.append(qn)
    p(f"  {qn}: {len(nodes)}")
missing_main = [m for m in MAIN if m in zero]
p(f"  targeted shapes with no focus node: {zero if zero else 'none'}")
if missing_main:
    failures.append("main shapes without focus nodes: " + ", ".join(missing_main))

# ---------- negatives ----------
p("\nNEGATIVE EXAMPLES (must not conform, for the stated reason)")
for f in sorted(glob.glob(os.path.join(SYNTH, "examples", "negative", "*.jsonld"))):
    name = os.path.basename(f)[:-7]
    g, conforms, msgs, n, inline, nd = validate(f)
    bad = iri_check(g)
    exp = EXPECT.get(name)
    hit = exp is not None and any(exp in m for m in msgs)
    ok = (not conforms) and hit and inline and nd == 0
    p(f"  {name}: conforms={conforms}, top-level results={n}, expected message found={hit}, "
      f"results with default message={nd}, IRIs outside allowed={len(bad)}"
      f"  -> {'PASS' if ok else 'FAIL'}")
    for m in msgs:
        p("      message:", m)
    if not ok:
        failures.append(name)

# ---------- vocabulary coverage ----------
vocab_path = os.path.join(SYNTH, "vocabulary.md")
if os.path.exists(vocab_path):
    vtext = open(vocab_path).read()
    pref = {"cheez": "https://w3id.org/cheezml/core#", "cubby": "https://w3id.org/cheezml/cubby#",
            "mowm": "https://w3id.org/cheezml/mowm#", "cube": "https://w3id.org/cheezml/cubelang#"}
    defined = set()
    for pf, ns in pref.items():
        for m in re.finditer(r"`%s:([A-Za-z][A-Za-z0-9]*)`" % pf, vtext):
            defined.add(ns + m.group(1))
    # codes are defined in the Turtle block
    blocks = re.findall(r"```turtle\n(.*?)```", vtext, re.S)
    code_graph = rdflib.Graph()
    for b in blocks:
        code_graph.parse(data=b, format="turtle")
    for s in code_graph.subjects():
        if isinstance(s, URIRef):
            defined.add(str(s))
    shape_terms = {str(x) for t in sg for x in t if isinstance(x, URIRef)
                   and str(x).startswith(tuple(pref.values()))
                   and not str(x).endswith("Shape")}
    missing_ex = sorted(used_terms - defined)
    missing_sh = sorted(shape_terms - defined)
    p(f"\nVOCABULARY COVERAGE: vocabulary.md Turtle blocks parsed ({len(code_graph)} triples); "
      f"CheezML IRIs used in examples={len(used_terms)}, in shapes={len(shape_terms)}; "
      f"undefined in vocabulary.md: examples={len(missing_ex)}, shapes={len(missing_sh)}")
    for m in missing_ex:
        p("    undefined (examples):", m)
    for m in missing_sh:
        p("    undefined (shapes):", m)
    if missing_ex or missing_sh:
        failures.append("vocabulary coverage")
    # alignment links point at the published Decision Ontology namespace, never used as main terms
    dec = [str(o) for o in code_graph.objects(None, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#closeMatch"))]
    p(f"  skos:closeMatch alignment targets: {sorted(dec)}")
else:
    p("\nvocabulary.md not present yet: coverage check skipped")

p("\nRESULT:", "ALL CHECKS PASS" if not failures else "FAILURES: " + ", ".join(failures))
open(os.path.join(SYNTH, "validation.txt"), "w").write(out.getvalue())
sys.exit(1 if failures else 0)
