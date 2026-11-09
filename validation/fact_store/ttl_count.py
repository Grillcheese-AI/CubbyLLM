"""validation/fact_store/ttl_count.py -- order 0055 step 1 on its source dump: the WikiKGQA 2026 challenge's Wikidata
(Turtle, ``wikikgqa-wikidata-all-dump.ttl.bz2``, CC0), counted in one pass, never unpacked to disk.

The same pass as ``dump_count.py`` (the parts as one stream, lbzip2, the hash on the way, the 1% sample by the same
hash of the id, so it draws the same ids as the JSON dump's sample), with a Turtle worker. Wikidata's dump writes each
entity as a block that starts with its ``data:<id> a schema:Dataset`` line (its revision, ``schema:version``); a
subject at column 0, its predicates on lines with one tab, further objects of a predicate on lines with two tabs.
Blocks are cut before ``\\ndata:``, so a worker always holds whole entities. Counted, as in the JSON pass:
  entities by type, revisions; statements by rank, novalue and somevalue; statements per property, and by datatype
  (each property's ``wikibase:propertyType``); qualifier snaks (time apart); references attached (distinct by hash,
  estimated: the hashes are the JSON dump's); EN, FR and ``mul`` labels, descriptions and aliases; enwiki and frwiki
  sitelinks; scholarly articles; value and reference nodes written out.
Turtle only:
  ``side.tsv.gz``  the dump's date (``wikibase:Dump``'s ``schema:dateModified``) and every redirect (``owl:sameAs``)
  sample_dangling  in the sampled entities, value and reference nodes used but written elsewhere in the dump (the
                   dump writes a shared node once): what a store built from the sample alone would miss

    python validation/fact_store/ttl_count.py --parts <dir> --out <dir> [--workers 8] [--stage <local dir>]

Resumable: a rerun after a dropped runtime goes on from the last checkpoint (``dump_resume.py``).

Wired: STANDALONE (order 0055's build; Colab, Nick's run).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re

try:
    from .dump_count import HLL_BITS, LANGS, SCHOLARLY, TIME_Q, Fmt, _hll_add, in_sample, run
except ImportError:                                           # run as a script from its folder (Colab)
    from dump_count import HLL_BITS, LANGS, SCHOLARLY, TIME_Q, Fmt, _hll_add, in_sample, run

__wiring__ = "STANDALONE"

LABELS = {b"rdfs:label": "labels", b"schema:description": "descriptions", b"skos:altLabel": "aliases"}
RANKS = {b"wikibase:NormalRank": "normal", b"wikibase:PreferredRank": "preferred", b"wikibase:DeprecatedRank": "deprecated"}
KINDS = {b"wikibase:Item": "item", b"wikibase:Property": "property", b"ontolex:LexicalEntry": "lexeme"}
SITES = {b"<https://en.wikipedia.org/>": "enwiki", b"<https://fr.wikipedia.org/>": "frwiki"}
SOMEVALUE = (b"_:", b"<http://www.wikidata.org/.well-known/genid/")
SCHOL = b"wd:" + SCHOLARLY.encode()
NODE = re.compile(rb"(?<=[ \t])(wdv:[0-9a-f]+|wdref:[0-9a-f]+)")
NODE_DEF = re.compile(rb"^(wdv:[0-9a-f]+|wdref:[0-9a-f]+)", re.M)
REF = re.compile(rb"wdref:([0-9a-f]+)")
NOVALUE = re.compile(rb"wdno:(P[0-9]+)")
SAMPLE_CAP = 16 << 20                                         # the largest sampled entity so far: 0.8 MB


def triples(text: bytes):
    """(subject, predicate, object text) for each line of a block in the dump's layout."""
    subj = pred = None
    for line in text.split(b"\n"):
        if not line or line.startswith(b"@prefix"):
            continue
        if line.startswith(b"\t\t"):
            yield subj, pred, line[2:]
        elif line.startswith(b"\t"):
            pred, _, obj = line[1:].partition(b" ")
            yield subj, pred, obj
        else:
            subj, _, rest = line.partition(b" ")
            pred, _, obj = rest.partition(b" ")
            yield subj, pred, obj


def _first(obj: bytes) -> bytes:
    return obj.split(b" ", 1)[0].rstrip(b",;.")


def _lang(obj: bytes) -> str | None:
    o = obj.rstrip(b" ;.,")
    i = o.rfind(b'"@')
    return o[i + 2:].decode("ascii", "replace") if i >= 0 else None


def count_block(text: bytes, c: collections.Counter, props: collections.Counter, reg: bytearray, side: list) -> str | None:
    """One entity's block (or the dump's header); returns the entity's id."""
    eid = text[5:text.find(b" ")].decode() if text.startswith(b"data:") else None
    me = b"wd:" + eid.encode() if eid else None
    kind, scholarly, stmts, prop_of = None, False, 0, {}
    for s, p, o in triples(text):
        if s == me:
            if p == b"a":
                kind = KINDS.get(_first(o), kind)
            elif p in LABELS:
                lg = _lang(o)
                if lg in LANGS:
                    c[f"{LABELS[p]}:{lg}"] += 1
            elif p.startswith(b"p:P"):
                props[p[2:].decode()] += 1
                prop_of[_first(o)] = p[2:]
                stmts += 1
            elif p == b"wdt:P31" and _first(o) == SCHOL:
                scholarly = True
            elif p == b"wikibase:propertyType":
                c[f"ptype:{eid}:{_first(o).decode()}"] = 1
        elif s.startswith(b"wds:"):
            if p == b"wikibase:rank":
                c[f"rank:{RANKS.get(_first(o), '?')}"] += 1
            elif p == b"a" and b"wdno:" in o:                   # no value: the main snak's, or a qualifier's
                for pid in NOVALUE.findall(o):
                    if prop_of.get(s) == pid:
                        c["snak:novalue"] += 1
                    else:
                        c["qualifier_snaks"] += 1
                        if pid.decode() in TIME_Q:
                            c["qualifier_snaks:time"] += 1
            elif p.startswith(b"ps:"):
                c["snak:somevalue" if o.startswith(SOMEVALUE) else "snak:value"] += 1
            elif p.startswith(b"pq:"):
                c["qualifier_snaks"] += 1
                if p[3:].decode() in TIME_Q:
                    c["qualifier_snaks:time"] += 1
            elif p == b"prov:wasDerivedFrom":
                for h in REF.findall(o):
                    c["reference_blocks"] += 1
                    _hll_add(reg, h.decode())
        elif s.startswith(b"wdref:"):
            if p == b"a":
                c["reference_nodes"] += 1
            elif p.startswith(b"pr:"):
                c["reference_snaks:written"] += 1
        elif s.startswith(b"wdv:"):
            if p == b"a":
                c["value_nodes"] += 1
        elif s == b"data:" + (eid or "").encode() and p == b"schema:version":
            c["revisions"] += 1
        elif p == b"schema:isPartOf" and _first(o) in SITES:
            c[f"sitelinks:{SITES[_first(o)]}"] += 1
        elif p == b"owl:sameAs":
            c["redirects"] += 1
            side.append(b"redirect\t" + s + b"\t" + _first(o))
        elif s == b"wikibase:Dump" and p == b"schema:dateModified":
            side.append(b"dump_date\t" + o.split(b'"')[1])
    if kind:
        c[f"entities:{kind}"] += 1
        c[f"statements:{kind}"] += stmts
    if scholarly:
        c["items:scholarly"] += 1
        c["statements:scholarly"] += stmts
    return eid if kind else None


def work(block: bytes, cap: int | None = None):
    """One block's counts. A sampled entity is kept whole up to ``cap`` bytes (``SAMPLE_CAP``); past it (an entity
    followed by a stretch with no ``data:`` header, cut by the fallback) its text is kept to the last blank line
    before ``cap``, and counted as ``sample_capped``."""
    cap = SAMPLE_CAP if cap is None else cap
    c, props, reg, sample, side = collections.Counter(), collections.Counter(), bytearray(1 << HLL_BITS), [], []
    for k, piece in enumerate(block.split(b"\ndata:")):
        text = piece if k == 0 else b"data:" + piece
        c["bytes"] += len(text) + 1
        if k == 0 and not text.startswith(b"data:"):
            c["headerless_bytes"] += len(text) + 1             # a fallback block's text before its first header
        eid = count_block(text, c, props, reg, side)
        if eid and in_sample(eid):
            if len(text) > cap:
                text = text[:max(text.rfind(b"\n\n", 0, cap), 0)]
                c["sample_capped"] += 1
            sample.append(text)
            c["sample_entities"] += 1
            c["sample_bytes"] += len(text)
            used, made = set(NODE.findall(text)), set(NODE_DEF.findall(text))
            for kind in (b"wdv:", b"wdref:"):
                u = {x for x in used if x.startswith(kind)}
                c[f"sample_used:{kind[:-1].decode()}"] += len(u)
                c[f"sample_dangling:{kind[:-1].decode()}"] += len(u - made)
    return dict(c), dict(props), bytes(reg), sample, side


class Ttl(Fmt):
    work, sep, sample, side, end, sections = staticmethod(work), b"\ndata:", "sample_1pct.ttl.gz", "side.tsv.gz", b".", ()
    fallback = b"\n\n"                                         # a blank line: the end of a subject's triples

    @staticmethod
    def finish(res: dict, props: collections.Counter, out: str) -> None:
        """The property types out of the counts, and the statements by datatype; the dump's date from the side file."""
        counts, types = res["counts"], {}
        for key in [k for k in counts if k.startswith("ptype:")]:
            _, pid, t = key.split(":", 2)
            types[pid] = t
            del counts[key]
        by = collections.Counter()
        for pid, n in props.items():
            by[types.get(pid, "?")] += n
        res["statements_by_datatype"] = dict(by.most_common())
        res["property_types"] = dict(collections.Counter(types.values()).most_common())
        with gzip.open(os.path.join(out, Ttl.side), "rt", encoding="utf-8") as f:
            res["dump_date"] = next((ln.split("\t")[1].strip() for ln in f if ln.startswith("dump_date\t")), None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--stage", default=None, help="a local disk: each segment is written there, then copied whole")
    ap.add_argument("--every", type=float, default=None, help="seconds between checkpoints (default 900)")
    a = ap.parse_args(argv)
    res = run(a.parts, a.out, a.workers, fmt=Ttl, stage=a.stage, every=a.every)
    print(json.dumps({k: v for k, v in res.items() if k != "props_top"}, indent=1))


if __name__ == "__main__":
    main()
