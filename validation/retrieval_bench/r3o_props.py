"""r3o_props -- order 0046, R3o's candidates: the properties an item carries, outgoing and incoming, fetched by id and
dated (Cowork, 2026-11-04: "the check step applied to relations: a property the item does not carry cannot verify").

  out  every property the item has a statement of (``wbgetentities`` claims, through the source's cache)
  in   every property some item states with this one as its value (one SPARQL query at query.wikidata.org:
       ``SELECT DISTINCT ?p WHERE { ?s ?p wd:Q }`` over direct properties). A query the service cannot answer in its
       time (a very common value: a country, a sex) is recorded as unavailable, never as "none".
Kept off-repo in ``$CUBBY_DATA_ROOT/retrieval/r3o_props.jsonl``, one line an item, with the fetch's date.

Wired: STANDALONE (R3o's bench).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

__wiring__ = "STANDALONE"

SPARQL = "https://query.wikidata.org/sparql"
DIRECT = "http://www.wikidata.org/prop/direct/"
QUERY = "SELECT DISTINCT ?p WHERE {{ ?s ?p wd:{q} . FILTER(STRSTARTS(STR(?p), \"" + DIRECT + "\")) }}"


def incoming(q: str, tries: int = 3, timeout: float = 70) -> tuple[list[str] | None, str]:
    """(the properties pointing at `q`, "ok") or (None, why) when the service cannot say."""
    from cubbyllm.sources.wikidata import UA
    why = "not asked"
    for k in range(tries):
        try:
            req = urllib.request.Request(SPARQL, data=urllib.parse.urlencode({"query": QUERY.format(q=q)}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.2)
            return sorted({b["p"]["value"][len(DIRECT):] for b in rows}, key=lambda p: int(p[1:])), "ok"
        except urllib.error.HTTPError as e:
            why = f"http {e.code}"
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After") or 30) + 1)
                continue
            if e.code in (500, 502, 503, 504):              # the service's own timeout on a very common value
                time.sleep(5 * (k + 1))
                continue
            break
        except Exception as e:                              # noqa: BLE001 -- recorded, never "no properties"
            why = type(e).__name__
            time.sleep(5 * (k + 1))
    return None, why


class ItemProps:
    """{item: {"out": [...], "in": [...] | None, "in_status": str, "fetched": date}}, one fetch per item, ever."""

    def __init__(self, src, path: str | None = None) -> None:
        from retrieval_sets.common import data_root
        self.src, self.path = src, path or os.path.join(data_root(), "r3o_props.jsonl")
        self.p: dict[str, dict] = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    self.p[d["id"]] = d

    def get(self, qids) -> dict[str, dict]:
        from retrieval_sets.common import entities
        need = [q for q in dict.fromkeys(qids) if q and q not in self.p]
        for i in range(0, len(need), 50):
            chunk = need[i:i + 50]
            ents = entities(self.src, chunk, props="claims", langs="en")
            with open(self.path, "a", encoding="utf-8") as f:
                for q in chunk:
                    out = sorted((ents.get(q) or {}).get("claims") or {}, key=lambda p: int(p[1:]))
                    inc, status = incoming(q)
                    d = {"id": q, "out": out, "in": inc, "in_status": status, "fetched": _dt.date.today().isoformat()}
                    self.p[q] = d
                    f.write(json.dumps(d) + "\n")
            print(f"item properties: {min(i + 50, len(need))}/{len(need)} fetched", flush=True)
        return {q: self.p[q] for q in qids if q in self.p}


def candidates(items: list[str], props: dict[str, dict]) -> dict:
    """The row's candidate properties: {"all": set, "out": set, "in": set, "in_unknown": bool}."""
    out = set().union(*[set(props.get(q, {}).get("out") or ()) for q in items]) if items else set()
    inc = set().union(*[set(props.get(q, {}).get("in") or ()) for q in items]) if items else set()
    unknown = any(props.get(q, {}).get("in") is None for q in items)
    return {"all": out | inc, "out": out, "in": inc, "in_unknown": unknown}


__all__ = ["ItemProps", "candidates", "incoming"]
