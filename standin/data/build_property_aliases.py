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


def fetch(query: str) -> list[dict]:
    url = SPARQL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["results"]["bindings"]


API = "https://www.wikidata.org/w/api.php"


def fetch_entities(max_pid: int, sleep_s: float = 1.0) -> dict[str, dict]:
    """The Action API, 50 ids a call (the SPARQL endpoint was rate-limited to 1 req/min
    during an outage on 2026-09-12): ~270 calls for P1..P13500, once."""
    props: dict[str, dict] = {}
    ids = [f"P{i}" for i in range(1, max_pid + 1)]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        url = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels|aliases",
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
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.sparql:
        rows = fetch(QUERY)
        props: dict[str, dict] = {}
        for b in rows:
            pid = b["p"]["value"].rsplit("/", 1)[-1]
            d = props.setdefault(pid, {"label": {}, "aliases": {"en": [], "fr": []}})
            lang, text, kind = b["lang"]["value"], b["text"]["value"], b["kind"]["value"]
            if kind == "label":
                d["label"][lang] = text
            else:
                d["aliases"][lang].append(text)
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
