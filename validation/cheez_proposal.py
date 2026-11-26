"""cheez_proposal - order 0085: the CheezML terms an arithmetic certificate needs, as a change to the pin's shapes,
applied in memory (never written into `vendor/cheezml`: the vocabulary is Nick's; the proposal is
`docs/research/2026-11-25-cheezml-arith/PROPOSAL.md`, and this module is its executable form).

- **Check kinds** (cheez:CheckKinds, and CheckRunShape's list): ``cubby:SlotCheck``, ``cubby:VoteCheck``,
  ``cubby:NonNegativeCheck``, ``cubby:WholeCountCheck``, ``cubby:MoneyDecimalsCheck``.
- **A threshold parameter** (ThresholdShape's list): ``cubby:SpeakVotes`` (m).
- **Artifact kinds** (cheez:ArtifactKinds; no shape lists them): ``cheez:AdapterWeights``, ``cheez:Settings``.
- **Offsets, nothing minted:** the W3C Web Annotation namespace allowed (``oa:hasSelector`` a
  ``oa:TextPositionSelector`` with ``oa:start`` / ``oa:end`` on a ``cubby:DocumentRegion``).
- **The certifying run** (CertifyingRunShape): it, or a run it was informed by (``prov:wasInformedBy``, any number of
  steps), used at least one statement version whose source records its revision **or** one field read from a region
  of a hashed question (a ``cubby:FieldSuggestion`` whose region ``schema:isPartOf`` a ``cheez:Artifact``). A run that
  read no stored fact needs no store snapshot already (ProgramRunShape's rule is about runs that read one).

``shapes()`` (an rdflib graph), ``TERMS``, ``ALLOWED``, ``CONTEXT``: what `cheez_shapes.check` takes to check against
the proposal instead of the pin.

Wiring: STANDALONE (order 0085's tests and report).
"""
from __future__ import annotations

import os

from cheez_shapes import NS, PIN

__wiring__ = "STANDALONE"

OA = "http://www.w3.org/ns/oa#"
NS_CUBBY, NS_CORE = NS + "cubby#", NS + "core#"
KINDS = ("cubby:SlotCheck", "cubby:VoteCheck", "cubby:NonNegativeCheck", "cubby:WholeCountCheck", "cubby:MoneyDecimalsCheck")
TERMS = {NS_CUBBY + k.split(":")[1] for k in KINDS + ("cubby:SpeakVotes",)} | \
        {NS_CORE + "AdapterWeights", NS_CORE + "Settings"}
ALLOWED = (OA,)
CONTEXT = {"oa": OA, "oa:start": {"@type": "xsd:nonNegativeInteger"}, "oa:end": {"@type": "xsd:nonNegativeInteger"}}

_OLD_RUN = """  sh:property [
    sh:path prov:used ;
    sh:qualifiedValueShape [
      sh:class cheez:FactVersion ;"""
_NEW_RUN = """  sh:or (
    [ sh:path ( [ sh:zeroOrMorePath prov:wasInformedBy ] prov:used ) ;
      sh:qualifiedValueShape [ sh:class cheez:FactVersion ;
        sh:property [ sh:path ( prov:hadPrimarySource schema:version ) ; sh:minCount 1 ;
                      sh:message "The statement's source records its revision." ] ] ;
      sh:qualifiedMinCount 1 ;
      sh:message "The certifying run, or a run it was informed by, read a statement version with its source's revision." ]
    [ sh:path ( [ sh:zeroOrMorePath prov:wasInformedBy ] prov:used ) ;
      sh:qualifiedValueShape [ sh:class cubby:FieldSuggestion ;
        sh:property [ sh:path ( cubby:region schema:isPartOf ) ; sh:class cheez:Artifact ; sh:minCount 1 ;
                      sh:message "The field was read from a region of a hashed question." ] ] ;
      sh:qualifiedMinCount 1 ;
      sh:message "The certifying run, or a run it was informed by, used a number read from a region of the question." ]
  ) ;
  sh:message "The certifying run, or a run it was informed by, read a statement version or a number of the question" """


def shapes_text() -> str:
    t = open(os.path.join(PIN, "src", "shapes.ttl"), encoding="utf-8").read()
    kinds = "mowm:ZMaxCheck mowm:AnswerSupportCheck mowm:ResidualCheck"
    params = "mowm:Tau mowm:TauPruneSteps"
    for old, new in ((kinds, kinds + " " + " ".join(KINDS)), (params, params + " cubby:SpeakVotes")):
        if t.count(old) != 1:
            raise ValueError(f"the pin's shapes changed: {old!r} is not there once")
        t = t.replace(old, new)
    old_msg = "and read at least one statement version with its source and revision."
    if t.count(old_msg) != 1:
        raise ValueError("the pin's certificate message changed")
    t = t.replace(old_msg, "and read, itself or through the runs it was informed by, a statement version with its "
                           "source's revision or a number of the question.")
    j = t.index(_OLD_RUN, t.index("shapes:CertifyingRunShape"))   # the FactVersion property: the shape's last
    k = t.index("\n  ] .", j) + len("\n  ] .")
    return t[:j] + _NEW_RUN + ' .' + t[k:]


def shapes():
    import rdflib
    return rdflib.Graph().parse(data=shapes_text(), format="turtle")


__all__ = ["ALLOWED", "CONTEXT", "KINDS", "NS_CORE", "NS_CUBBY", "OA", "TERMS", "shapes", "shapes_text"]
