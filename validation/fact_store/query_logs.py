"""validation/fact_store/query_logs.py -- order 0055 (Cowork's addendum, 2026-11-06): the hot store chosen by real use.

Reads the organic split of the Wikidata Query Logs (Walter and Bast, SIGIR 2026; ``ad-freiburg/wikidata-query-logs``,
code Apache-2.0): 859,305 distinct SPARQL queries people ran against Wikidata from 2017-06-12 to 2018-03-25, from
Malyshev et al.'s logs (TU Dresden, CC0). One query per line, a JSON string, IRIs written out. Only the queries are
read; the dataset's model-written questions are not.

Counted, by distinct queries (a query counts once per property, however often it names it):
  props       each property by its role: ``main`` (wdt, p, ps, psv, psn, wdtn), ``qualifier`` (pq, pqv, pqn),
              ``reference`` (pr, prv, prn), ``entity`` (the property named as a thing, wd:P31)
  shapes      each query's features, as one sorted key (paths, qualifiers, references, OPTIONAL, FILTER, aggregates,
              GROUP BY, ORDER BY, LIMIT, VALUES, MINUS / NOT EXISTS, a subquery, the label and geo services)
With ``--statements`` (a count's ``resume.json`` / ``resume.done.json``, whose ``props`` hold the dump's statements per
property), the projection: properties ranked by the queries using them (main or qualifier role); for each hot size K,
the statements the top K hold and the share of queries whose properties (main and qualifier) all sit in the top K.
Qualifier statements are not counted by the dump pass, so the hot store's size is its main statements.

    python validation/fact_store/query_logs.py --logs organic.jsonl --out <json> [--statements resume.json]

Wired: STANDALONE (order 0055's projection; read once, offline).
"""
from __future__ import annotations

import argparse
import collections
import json
import re

__wiring__ = "STANDALONE"

WD = "http://www.wikidata.org/"
ROLES = {"prop/direct/": "main", "prop/direct-normalized/": "main", "prop/statement/": "main",
         "prop/statement/value/": "main", "prop/statement/value-normalized/": "main", "prop/": "main",
         "prop/qualifier/": "qualifier", "prop/qualifier/value/": "qualifier",
         "prop/qualifier/value-normalized/": "qualifier", "prop/reference/": "reference",
         "prop/reference/value/": "reference", "prop/reference/value-normalized/": "reference", "entity/": "entity"}
PREFIXED = {"wdt": "main", "wdtn": "main", "p": "main", "ps": "main", "psv": "main", "psn": "main",
            "pq": "qualifier", "pqv": "qualifier", "pqn": "qualifier", "pr": "reference", "prv": "reference",
            "prn": "reference", "wd": "entity"}
IRI = re.compile(r"<http://www\.wikidata\.org/((?:prop/(?:[a-z-]+/)*)?|entity/)(P\d+)>")
PFX = re.compile(r"(?<![\w:/])(wdtn|wdt|psv|psn|ps|pqv|pqn|pq|prv|prn|pr|p|wd):(P\d+)\b")
FEATURES = {
    "path": re.compile(r"(?:>|P\d+)\s*[*+]|(?:>|P\d+)\s*/\s*(?:<|\w+:)|\^\s*(?:<|\w+:)"),
    "optional": re.compile(r"\bOPTIONAL\b", re.I), "filter": re.compile(r"\bFILTER\b", re.I),
    "aggregate": re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX|SAMPLE|GROUP_CONCAT)\s*\(", re.I),
    "group_by": re.compile(r"\bGROUP\s+BY\b", re.I), "order_by": re.compile(r"\bORDER\s+BY\b", re.I),
    "limit": re.compile(r"\bLIMIT\b", re.I), "values": re.compile(r"\bVALUES\b", re.I),
    "negation": re.compile(r"\bMINUS\b|\bNOT\s+EXISTS\b", re.I),
    "subquery": re.compile(r"\{\s*SELECT\b", re.I), "ask": re.compile(r"^\s*ASK\b", re.I),
    "label_service": re.compile(r"wikiba\.se/ontology#label>|wikibase:label\b"),
    "geo_service": re.compile(r"wikiba\.se/ontology#(?:box|around)>|wikibase:(?:box|around)\b"),
}


def uses(q: str) -> dict[str, set[str]]:
    """The properties a query names, by role (each once)."""
    out = collections.defaultdict(set)
    for path, pid in IRI.findall(q):
        role = ROLES.get(path)
        if role:
            out[role].add(pid)
    for pfx, pid in PFX.findall(q):
        out[PREFIXED[pfx]].add(pid)
    return out


def shape(q: str, u: dict[str, set[str]]) -> str:
    f = [k for k, rx in FEATURES.items() if rx.search(q)]
    if u.get("qualifier"):
        f.append("qualifiers")
    if u.get("reference"):
        f.append("references")
    n = len(u.get("main", set()) | u.get("qualifier", set()))
    f.append(f"props:{min(n, 5)}{'+' if n >= 5 else ''}")
    return ",".join(sorted(f))


def count(lines) -> tuple[dict, list[set[str]]]:
    props = {r: collections.Counter() for r in ("main", "qualifier", "reference", "entity")}
    shapes, per_query, n = collections.Counter(), [], 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        q = json.loads(ln)
        q = q if isinstance(q, str) else q.get("sparql") or q.get("info", {}).get("raw_sparql", "")
        n += 1
        u = uses(q)
        for role, ps in u.items():
            props[role].update(ps)
        shapes[shape(q, u)] += 1
        per_query.append(u.get("main", set()) | u.get("qualifier", set()))
    return {"queries": n, "props": {r: dict(c.most_common()) for r, c in props.items()},
            "shapes_top": shapes.most_common(60), "shapes_n": len(shapes)}, per_query


KS = (25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000)


def project(res: dict, per_query: list[set[str]], statements: dict[str, int], ks=KS) -> list[dict]:
    """Hot size K -> the statements the top K properties hold, and the queries whose properties all sit in them."""
    use = collections.Counter(res["props"]["main"]) + collections.Counter(res["props"]["qualifier"])
    rank = [p for p, _ in sorted(use.items(), key=lambda kv: (-kv[1], int(kv[0][1:])))]
    pos = {p: i for i, p in enumerate(rank)}
    worst = collections.Counter(max((pos[p] for p in ps), default=-1) for ps in per_query)
    total, rows, covered, upto = sum(statements.values()) or 1, [], 0, -1
    for k in ks:
        for r in range(upto + 1, k):
            covered += worst.get(r, 0)
        upto = k - 1
        hot = sum(statements.get(p, 0) for p in rank[:k])
        q_ok = covered + worst.get(-1, 0)
        with_props = max(len(per_query) - worst.get(-1, 0), 1)
        rows.append({"k": k, "statements": hot, "statements_share": round(hot / total, 4),
                     "queries_covered": q_ok, "queries_share": round(q_ok / max(len(per_query), 1), 4),
                     "queries_with_props_share": round(covered / with_props, 4)})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--statements", default=None, help="a dump count's resume.json / resume.done.json")
    a = ap.parse_args(argv)
    with open(a.logs, encoding="utf-8") as f:
        res, per_query = count(f)
    res["queries_without_props"] = sum(1 for ps in per_query if not ps)
    if a.statements:
        st = json.load(open(a.statements))
        res["statements_source"] = {"offset": st.get("offset"), "statements": sum(st["props"].values()),
                                    "props": len(st["props"])}
        res["projection"] = project(res, per_query, st["props"])
        use = collections.Counter(res["props"]["main"]) + collections.Counter(res["props"]["qualifier"])
        res["largest"] = [{"prop": p, "statements": n, "queries": use.get(p, 0)}
                          for p, n in sorted(st["props"].items(), key=lambda kv: -kv[1])[:30]]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("props", "shapes_top")}, indent=1))


if __name__ == "__main__":
    main()
