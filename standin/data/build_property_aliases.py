"""build_property_aliases -- Wikidata's property labels and aliases (EN + FR), once, as a local table.

Wired: STANDALONE (offline builder; run once; output off-repo under standin/data/out/).

Lever 4 and lever 6 ask "which relation label does this wording name?"; until 2026-09-12 they
asked Wikidata's property search per wording, per question -- 875 API calls for 100 SimpleQA
questions in exp_r14. Serving must not depend on a live external call per question (Nick,
2026-09-12): the whole property vocabulary is ~12k properties and one SPARQL query, so it is
fetched ONCE here and read locally forever after (`sources.PropertyAliases`). The API stays for
entity FACTS, in the learning phase, cached.

  python standin/data/build_property_aliases.py [--out standin/data/out/wikidata_properties_en_fr.json]
"""
from __future__ import annotations

import argparse, json, pathlib, time, urllib.error, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "standin" / "data" / "out" / "wikidata_properties_en_fr.json"
SPARQL = "https://query.wikidata.org/sparql"
UA = "cubbyllm-standin/0.1 (research; property alias table)"
QUERY = """
SELECT ?p ?lang ?text ?kind WHERE {
  ?p a wikibase:Property .
  { ?p rdfs:label ?text . BIND("label" AS ?kind) }
  UNION
  { ?p skos:altLabel ?text . BIND("alias" AS ?kind) }
  BIND(LANG(?text) AS ?lang)
  FILTER(?lang IN ("en", "fr"))
}
"""
# the value kind of every property, one request (~12.7k rows): what a typed ask can narrow by
DATATYPES = """
SELECT ?p ?dt WHERE { ?p a wikibase:Property ; wikibase:propertyType ?dt . }
"""
_ONTOLOGY_TO_DATATYPE = {"Time": "time", "Quantity": "quantity", "WikibaseItem": "wikibase-item", "String": "string",
                         "Monolingualtext": "monolingualtext", "ExternalId": "external-id", "Url": "url",
                         "CommonsMedia": "commonsMedia", "GlobeCoordinate": "globe-coordinate", "Math": "math",
                         "GeoShape": "geo-shape", "TabularData": "tabular-data", "MusicalNotation": "musical-notation",
                         "WikibaseProperty": "wikibase-property", "WikibaseLexeme": "wikibase-lexeme",
                         "WikibaseForm": "wikibase-form", "WikibaseSense": "wikibase-sense", "EntitySchema": "entity-schema"}


def fetch_datatypes() -> dict[str, str]:
    """{pid: wikidata datatype name} from one SPARQL request; the ontology's local names
    are mapped onto the Action API's datatype names so both build paths agree."""
    out: dict[str, str] = {}
    for b in fetch(DATATYPES):
        pid = b["p"]["value"].rsplit("/", 1)[-1]
        local = b["dt"]["value"].rsplit("#", 1)[-1]
        out[pid] = _ONTOLOGY_TO_DATATYPE.get(local, local.lower())
    return out


def fetch(query: str) -> list[dict]:
    url = SPARQL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))["results"]["bindings"]
        except urllib.error.HTTPError as e:              # the endpoint was at 1 req/min on 2026-09-12/13
            if e.code != 429 or attempt == 5:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or 65.0
            print(f"  429 from SPARQL: waiting {wait:.0f}s", flush=True); time.sleep(wait)
    raise RuntimeError("unreachable")


API = "https://www.wikidata.org/w/api.php"


def fetch_entities(max_pid: int, sleep_s: float = 1.0, props_wanted: str = "labels|aliases|datatype") -> dict[str, dict]:
    """The Action API, 50 ids a call (the SPARQL endpoint was rate-limited to 1 req/min
    during an outage on 2026-09-12, and still on 2026-09-13): ~270 calls for P1..P13500, once."""
    props: dict[str, dict] = {}
    ids = [f"P{i}" for i in range(1, max_pid + 1)]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        url = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk), "props": props_wanted,
                                                  "languages": "en|fr", "format": "json"})
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                wait = float(e.headers.get("Retry-After") or 0) or 30.0 * (attempt + 1)
                print(f"  429 at {i}: waiting {wait:.0f}s", flush=True); time.sleep(wait)
            except Exception:                            # noqa: BLE001
                if attempt == 7:
                    raise
                time.sleep(10)
        for pid, ent in (data.get("entities") or {}).items():
            if "missing" in ent:
                continue
            d = {"label": {}, "aliases": {"en": [], "fr": []}}
            if ent.get("datatype"):
                d["datatype"] = ent["datatype"]
            for lang, lab in (ent.get("labels") or {}).items():
                d["label"][lang] = lab["value"]
            for lang, al in (ent.get("aliases") or {}).items():
                d["aliases"][lang] = [x["value"] for x in al]
            props[pid] = d
        if (i // 50) % 40 == 0:
            print(f"  {i + len(chunk)}/{len(ids)} ids, {len(props)} properties so far", flush=True)
        time.sleep(sleep_s)
    return props


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-pid", type=int, default=13600)
    ap.add_argument("--sparql", action="store_true", help="one SPARQL query instead of the Action API")
    ap.add_argument("--datatypes", action="store_true",
                    help="add each property's datatype to an EXISTING table, nothing else (one SPARQL request; "
                         "--via-api reads it from the Action API in ~270 calls when the SPARQL endpoint is rate-limited)")
    ap.add_argument("--via-api", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.datatypes:
        out = pathlib.Path(a.out)
        d = json.loads(out.read_text(encoding="utf-8"))
        if a.via_api:
            dts = {pid: p["datatype"] for pid, p in fetch_entities(a.max_pid, props_wanted="datatype").items() if p.get("datatype")}
        else:
            dts = fetch_datatypes()
        n = 0
        for pid, p in d["properties"].items():
            if pid in dts:
                p["datatype"] = dts[pid]; n += 1
        out.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        from collections import Counter
        print(f"datatypes: {len(dts)} fetched, {n}/{len(d['properties'])} properties typed, {time.perf_counter() - t0:.0f}s")
        print("  " + ", ".join(f"{k} {v}" for k, v in Counter(p.get("datatype") for p in d["properties"].values()).most_common(8)))
        return
    if a.sparql:
        rows = fetch(QUERY)
        dts = fetch_datatypes()
        props: dict[str, dict] = {}
        for b in rows:
            pid = b["p"]["value"].rsplit("/", 1)[-1]
            d = props.setdefault(pid, {"label": {}, "aliases": {"en": [], "fr": []}})
            lang, text, kind = b["lang"]["value"], b["text"]["value"], b["kind"]["value"]
            if kind == "label":
                d["label"][lang] = text
            else:
                d["aliases"][lang].append(text)
            if pid in dts:
                d["datatype"] = dts[pid]
    else:
        props = fetch_entities(a.max_pid)
    for d in props.values():
        for lang in d["aliases"]:
            d["aliases"][lang] = sorted(set(d["aliases"][lang]))
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "query.wikidata.org", "fetched": time.strftime("%Y-%m-%d"), "langs": ["en", "fr"],
                               "n_properties": len(props), "properties": props}, ensure_ascii=False), encoding="utf-8")
    n_alias = sum(len(d["aliases"]["en"]) + len(d["aliases"]["fr"]) for d in props.values())
    print(f"wrote {out.name}: {len(props)} properties, {n_alias} aliases (en+fr), {out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f}s")
    for pid in ("P569", "P19", "P571", "P20", "P26", "P131"):
        d = props.get(pid, {})
        print(f"  {pid}: {d.get('label', {}).get('en')!r} / {d.get('label', {}).get('fr')!r}; en aliases {d.get('aliases', {}).get('en', [])[:6]}")


if __name__ == "__main__":
    main()
