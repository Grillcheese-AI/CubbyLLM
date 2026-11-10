"""validation/fact_store/roundtrip_check.py -- order 0055's acceptance: entities round-trip with 0 differences
against a second, independent parser.

The second parser is rdflib's Turtle reader (BSD-3), which shares no code with the store's own (Rust, token by
token). Each entity block of the dump is read with rdflib, under the dump's own prefixes, into what the store
claims to keep (revision, timestamp, datatype, EN/FR/mul names, enwiki/frwiki titles, every statement: its id,
property, rank, best-rank flag, main value, main value node, qualifiers hot and cold with their value nodes,
references), and compared with the store's own record of it (`wdstore dump`). rdflib's literal normalization is
off: a value is compared as the dump wrote it ("+1234.50" stays "+1234.50").

    uv run --with rdflib python validation/fact_store/roundtrip_check.py --blocks <segment.gz> --prefixes <ttl>
        --store <wdstore dump's jsonl> [--limit 100000] --out <json>

Wired: STANDALONE (order 0055's acceptance; read once, offline).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re

import rdflib
from rdflib import BNode, Literal, URIRef

__wiring__ = "STANDALONE"

rdflib.NORMALIZE_LITERALS = False
HOT = {580, 582, 585, 1319, 1326}
LANGS = {"en", "fr", "mul"}
WIKIS = {"https://en.wikipedia.org/", "https://fr.wikipedia.org/"}
ENT = "http://www.wikidata.org/entity/"
GENID = "http://www.wikidata.org/.well-known/genid/"
ID = re.compile(r"^(Q|P|L)[1-9][0-9]*(-[FS][0-9]+)?$")


def prefixes(path: str) -> tuple[str, dict[str, str]]:
    text = open(path, encoding="utf-8").read()
    return text, dict(re.findall(r"@prefix (\w*): <([^>]+)>", text))


def expand(s: str, pm: dict[str, str]) -> str:
    """A prefixed name, as the store keeps Wikidata's own, written out; anything else as it is."""
    m = re.match(r"^(\w+):(.*)$", s)
    return pm[m.group(1)] + m.group(2) if m and m.group(1) in pm and not s.startswith(("http:", "https:")) else s


def norm_store(v, pm):
    if isinstance(v, str):
        return v
    k, x = next(iter(v.items()))
    if k == "iri":
        return {"iri": expand(x, pm)}
    if k == "typed":
        return {"typed": [x[0], expand(x[1], pm)]}
    return v


def norm_rdf(o):
    if isinstance(o, BNode) or (isinstance(o, URIRef) and str(o).startswith(GENID)):
        return "somevalue"
    if isinstance(o, URIRef):
        s = str(o)
        if s.startswith(ENT) and ID.match(s[len(ENT):]):
            return {"entity": s[len(ENT):]}
        return {"iri": s}
    if isinstance(o, Literal):
        if o.language:
            return {"mono": [str(o), o.language]}
        if o.datatype is None:
            return {"str": str(o)}
        return {"typed": [str(o), str(o.datatype)]}
    raise ValueError(o)


def key(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def local(u, ns: str) -> str:
    return str(u)[len(ns):] if str(u).startswith(ns) else ""


def prop(u, ns: str) -> int | None:
    """The property number when `u` is `ns` + P<number> exactly (ps: is a prefix of psv: and psn: IRIs)."""
    m = re.fullmatch(r"P([0-9]+)", local(u, ns))
    return int(m.group(1)) if m else None


def read(block: str, head: str, pm: dict[str, str]) -> dict | None:
    """The block, read by rdflib, as the store's export writes an entity (lists in a canonical order)."""
    g = rdflib.Graph()
    g.parse(data=head + block, format="turtle")
    N = {k: rdflib.Namespace(v) for k, v in pm.items()}
    about = [(s, o) for s, o in g.subject_objects(N["schema"].about) if str(s).startswith(pm["data"])]
    if not about:
        return None
    data, me = about[0]
    kinds = {N["wikibase"].Item, N["wikibase"].Property, N["ontolex"].LexicalEntry}
    if not any(t in kinds for t in g.objects(me, rdflib.RDF.type)):
        return None
    out = {"id": local(me, ENT), "revision": int(str(g.value(data, N["schema"].version))),
           "modified": str(g.value(data, N["schema"].dateModified) or ""),
           "datatype": None, "names": [], "sitelinks": [], "statements": []}
    t = g.value(me, N["wikibase"].propertyType)
    if t is not None:
        out["datatype"] = "wikibase:" + local(t, pm["wikibase"])
    for kind, pred in ((0, rdflib.RDFS.label), (1, N["skos"].altLabel), (2, N["schema"].description)):
        for o in g.objects(me, pred):
            if isinstance(o, Literal) and o.language in LANGS:
                out["names"].append([o.language, kind, str(o)])
    out["names"].sort(key=lambda n: (n[0], n[1], n[2]))
    for art in g.subjects(N["schema"].about, me):
        site, name = g.value(art, N["schema"].isPartOf), g.value(art, N["schema"].name)
        if site is not None and name is not None and str(site) in WIKIS:
            out["sitelinks"].append([str(site), str(name)])
    out["sitelinks"].sort()
    for p, s in g.predicate_objects(me):
        n = prop(p, pm["p"])
        if n is None:
            continue
        st = {"guid": local(s, pm["wds"]).split("-", 1)[1], "pid": n, "rank": None, "best": False, "value": None,
              "value_node": None, "hot_qualifiers": [], "hot_qualifier_nodes": [], "cold_qualifiers": [],
              "cold_qualifier_nodes": [], "references": []}
        for pp, o in g.predicate_objects(s):
            if pp == rdflib.RDF.type:
                if o == N["wikibase"].BestRank:
                    st["best"] = True
                elif (q := prop(o, pm["wdno"])) is not None:
                    if q == n and st["value"] is None:
                        st["value"] = "novalue"
                    else:
                        st["hot_qualifiers" if q in HOT else "cold_qualifiers"].append([q, "novalue"])
            elif pp == N["wikibase"].rank:
                st["rank"] = {"NormalRank": 1, "PreferredRank": 2, "DeprecatedRank": 0}[local(o, pm["wikibase"])]
            elif prop(pp, pm["ps"]) is not None:
                st["value"] = norm_rdf(o)
            elif prop(pp, pm["psv"]) is not None:
                st["value_node"] = local(o, pm["wdv"])
            elif (q := prop(pp, pm["pqv"])) is not None:
                st["hot_qualifier_nodes" if q in HOT else "cold_qualifier_nodes"].append([q, local(o, pm["wdv"])])
            elif (q := prop(pp, pm["pq"])) is not None:
                st["hot_qualifiers" if q in HOT else "cold_qualifiers"].append([q, norm_rdf(o)])
            elif pp == N["prov"].wasDerivedFrom:
                st["references"].append(local(o, pm["wdref"]))
        for k in ("hot_qualifiers", "hot_qualifier_nodes", "cold_qualifiers", "cold_qualifier_nodes", "references"):
            st[k].sort(key=key)
        out["statements"].append(st)
    out["statements"].sort(key=lambda s: (s["pid"], s["guid"]))
    return out


def canon_store(e: dict, pm) -> dict:
    e = dict(e)
    e.pop("cold", None)
    e["names"] = sorted(e["names"], key=lambda n: (n[0], n[1], n[2]))
    for s in e["statements"]:
        s["value"] = norm_store(s["value"], pm)
        for k in ("hot_qualifiers", "cold_qualifiers"):
            s[k] = sorted(([q, norm_store(v, pm)] for q, v in s[k]), key=key)
        for k in ("hot_qualifier_nodes", "cold_qualifier_nodes", "references"):
            s[k] = sorted(s[k], key=key)
    e["statements"].sort(key=lambda s: (s["pid"], s["guid"]))
    return e


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    head, pm = prefixes(a.prefixes)
    stored = {}
    with open(a.store, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            stored[e["id"]] = e
    text = gzip.open(a.blocks, "rt", encoding="utf-8").read() if a.blocks.endswith((".gz",)) or ".gz." in a.blocks else open(a.blocks, encoding="utf-8").read()
    blocks = ["data:" + b if k else b for k, b in enumerate(text.split("\ndata:"))]
    n = same = statements = 0
    diffs, missing = [], []
    import functools
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        got = pool.imap(functools.partial(read, head=head, pm=pm), blocks[:int(a.limit * 1.05) + 10], chunksize=64)
        results = [r for r in got if r is not None][:a.limit]
    for r in results:
        n += 1
        statements += len(r["statements"])
        s = stored.get(r["id"])
        if s is None:
            missing.append(r["id"])
            continue
        c = canon_store(s, pm)
        if c == r:
            same += 1
        elif len(diffs) < 20:
            d = {k: {"rdflib": r[k], "store": c[k]} for k in r if k != "statements" and r[k] != c.get(k)}
            rs, cs = {x["guid"]: x for x in r["statements"]}, {x["guid"]: x for x in c["statements"]}
            for g in sorted(set(rs) | set(cs)):
                x, y = rs.get(g), cs.get(g)
                if x != y:
                    d[f"statement {g}"] = ({f: {"rdflib": x[f], "store": y[f]} for f in x if x[f] != y.get(f)}
                                           if x and y else {"rdflib": x, "store": y})
            diffs.append({"id": r["id"], "differs": key(d)[:3000]})
    res = {"entities": n, "statements": statements, "same": same, "different": n - same - len(missing),
           "missing_from_store": len(missing), "first_differences": diffs, "first_missing": missing[:20],
           "parser": f"rdflib {rdflib.__version__}"}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("first_differences",)}, ensure_ascii=False))
    if res["first_differences"]:
        print(res["first_differences"][0]["differs"][:1500])


if __name__ == "__main__":
    main()
