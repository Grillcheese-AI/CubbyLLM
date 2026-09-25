"""wikidata_kin -- real families from Wikidata, for the skill library's real-world episodes. WO-2.6 follow-up.

Wired: STANDALONE (validation helper; never imported by cubbyllm/).

WHAT IT BUILDS
--------------
Episodes stated by Wikidata itself: a ``relative`` statement (P1038) whose qualifier ``kinship to subject``
(P1039) names how the relative is related -- "Queen X: relative Y, kinship to subject: grandson". Around
each pair, the family graph Wikidata holds: father (P22), mother (P25), child (P40), sibling (P3373), spouse
(P26), and every person's sex or gender (P21). The chains between the two people come from that graph, the
relation they compose to from the qualifier -- two different editors' work, which is what makes it an
episode rather than a restatement.

And the kinship TERMS' own hierarchy (P279, subclass of): "paternal grandfather" is a "grandfather", "father's
brother" an "uncle". A term entails its superclasses, so an episode stating the finer term does not
contradict a rule concluding the coarser one.

Everything is cached (one file per SPARQL query) under ``standin/data/out/wikidata_kin/``, which is
gitignored; a rebuild with the cache in place makes no network call.

    python validation/wikidata_kin.py --per-kind 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "standin" / "data" / "out" / "wikidata_kin"
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FAMILY = {"P22": "father", "P25": "mother", "P40": "child", "P3373": "sibling", "P26": "spouse"}
GENDER = {"Q6581097": "male", "Q6581072": "female"}           # everything else stays ungendered
def _ua() -> str:
    from sources import UA
    return UA


def sparql(query: str, retries: int = 12) -> list[dict]:
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
    path = OUT / "cache" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(SPARQL, data=urllib.parse.urlencode({"query": query}).encode(),
                                         headers={"User-Agent": _ua(), "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows), encoding="utf-8")
            print(f"    sparql {len(rows)} rows in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
            time.sleep(0.3)
            return rows
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 429:                                     # the service's rate limit: wait as it says
                wait = int(e.headers.get("Retry-After") or 30)
                print(f"    sparql rate-limited, waiting {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait + 1)
                continue
            print(f"    sparql failed after {time.time() - t0:.1f}s: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
        except Exception as e:                                    # noqa: BLE001 -- retried, then raised
            err = e
            print(f"    sparql failed after {time.time() - t0:.1f}s: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"SPARQL failed after {retries} tries: {err}")


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def labels(ids: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        key = hashlib.sha256(("labels:" + "|".join(chunk)).encode()).hexdigest()[:24]
        path = OUT / "cache" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            u = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels",
                                                    "languages": "en", "format": "json"})
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": _ua()}), timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.5)
        for q, e in (data.get("entities") or {}).items():
            lab = ((e.get("labels") or {}).get("en") or {}).get("value")
            if lab:
                out[q] = lab
    return out


def kin_values(top: int = 70) -> list[tuple[str, int]]:
    rows = sparql(f"SELECT ?k (COUNT(?st) AS ?n) WHERE {{ ?st pq:P1039 ?k }} GROUP BY ?k ORDER BY DESC(?n) LIMIT {top}")
    return [(qid(b["k"]["value"]), int(b["n"]["value"])) for b in rows]


def taxonomy(terms: list[str], depth: int = 6) -> dict[str, list[str]]:
    """term qid -> every class it is a subclass of (P279, followed up to `depth` levels), itself first.
    Read through the API a level at a time: the query service's transitive P279+ times out."""
    parents: dict[str, list[str]] = {}
    frontier = list(dict.fromkeys(terms))
    for _ in range(depth):
        todo = [t for t in frontier if t not in parents]
        if not todo:
            break
        nxt = []
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            key = hashlib.sha256(("subclass:" + "|".join(chunk)).encode()).hexdigest()[:24]
            path = OUT / "cache" / f"{key}.json"
            if path.exists():
                slim = json.loads(path.read_text(encoding="utf-8"))
            else:
                u = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk),
                                                        "props": "claims", "format": "json"})
                with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": _ua()}), timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                slim = {q: [(((st.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}).get("id")
                            for st in (e.get("claims") or {}).get("P279", []) if st.get("rank") != "deprecated"]
                        for q, e in (data.get("entities") or {}).items()}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(slim), encoding="utf-8")
                time.sleep(0.3)
            for q in chunk:
                parents[q] = [p for p in slim.get(q) or [] if p]
                nxt += parents[q]
        frontier = nxt
    out: dict[str, list[str]] = {}
    for t in terms:
        seen, stack = [t], [t]
        while stack:
            for p in parents.get(stack.pop(), []):
                if p not in seen:
                    seen.append(p)
                    stack.append(p)
        out[t] = seen
    return out


def pairs(kind: str, limit: int) -> list[tuple[str, str]]:
    rows = sparql(f"SELECT ?s ?r WHERE {{ ?s p:P1038 ?st . ?st ps:P1038 ?r ; pq:P1039 wd:{kind} . }} LIMIT {limit}")
    return [(qid(b["s"]["value"]), qid(b["r"]["value"])) for b in rows if "entity/Q" in b["r"]["value"]]


def neighbours(ids: list[str], edges: dict, gender: dict) -> set[str]:
    """Fetch the family edges and sex of `ids` (not fetched before) through the Wikidata API -- 50 people a
    call, the claims read for P22 P25 P40 P3373 P26 P21 (deprecated ones skipped) -- and return the people
    they point at. The API, not SPARQL: the query service throttles a client this chatty."""
    todo = [i for i in ids if i not in edges]
    found: set[str] = set()
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        key = hashlib.sha256(("claims:" + "|".join(chunk)).encode()).hexdigest()[:24]
        path = OUT / "cache" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = None
            for attempt in range(8):
                try:
                    u = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk),
                                                            "props": "claims", "format": "json"})
                    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": _ua()}), timeout=60) as r:
                        data = json.loads(r.read().decode("utf-8"))
                    break
                except Exception as e:                            # noqa: BLE001 -- retried
                    print(f"    api failed: {str(e)[:100]}", file=sys.stderr, flush=True)
                    time.sleep(10 * (attempt + 1))
            if data is None:
                raise RuntimeError("the Wikidata API did not answer")
            slim = {}
            for q, e in (data.get("entities") or {}).items():
                claims = e.get("claims") or {}
                slim[q] = {p: [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
                                for st in claims.get(p, []) if st.get("rank") != "deprecated"]
                           for p in list(FAMILY) + ["P21"] if p in claims}
            data = {"slim": slim}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.3)
        for x in chunk:
            edges.setdefault(x, [])
            for p, vals in (data["slim"].get(x) or {}).items():
                for y in vals:
                    if not y:
                        continue
                    if p == "P21":
                        if y in GENDER:
                            gender[x] = GENDER[y]
                    else:
                        edges[x].append((FAMILY[p], y))
                        found.add(y)
        if (i // 50) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(todo)} people this ring", file=sys.stderr, flush=True)
    return found


def relative_statements(ids: list[str]) -> dict[str, list]:
    """Every "relative" statement (P1038) with a "kinship to subject" qualifier (P1039) that `ids` carry:
    person -> [(relative, [kinship qids])]. The reverse of a sampled pair is among them when Wikidata states
    the relation both ways -- the episodes an inverse rule is learned from."""
    out: dict[str, list] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        key = hashlib.sha256(("relatives:" + "|".join(chunk)).encode()).hexdigest()[:24]
        path = OUT / "cache" / f"{key}.json"
        if path.exists():
            slim = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = None
            for attempt in range(8):
                try:
                    u = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(chunk),
                                                            "props": "claims", "format": "json"})
                    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": _ua()}), timeout=60) as r:
                        data = json.loads(r.read().decode("utf-8"))
                    break
                except Exception as e:                            # noqa: BLE001 -- retried
                    print(f"    api failed: {str(e)[:100]}", file=sys.stderr, flush=True)
                    time.sleep(10 * (attempt + 1))
            if data is None:
                raise RuntimeError("the Wikidata API did not answer")
            slim = {}
            for q, e in (data.get("entities") or {}).items():
                rows = []
                for st in (e.get("claims") or {}).get("P1038", []):
                    if st.get("rank") == "deprecated":
                        continue
                    y = (((st.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}).get("id")
                    ks = [((qq.get("datavalue") or {}).get("value") or {}).get("id")
                          for qq in (st.get("qualifiers") or {}).get("P1039", [])]
                    ks = [k for k in ks if k]
                    if y and ks:
                        rows.append([y, ks])
                slim[q] = rows
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(slim), encoding="utf-8")
            time.sleep(0.3)
        for x in chunk:
            out[x] = slim.get(x) or []
    return out


def build(per_kind: int, top: int, s_depth: int, r_depth: int) -> dict:
    kinds = kin_values(top)
    names = labels([k for k, _ in kinds])
    tax = taxonomy([k for k, _ in kinds])
    names.update(labels(sorted({s for sups in tax.values() for s in sups} - set(names))))
    stated: dict[tuple[str, str], set[str]] = {}
    for k, _n in kinds:
        for s, r in pairs(k, per_kind):
            if s != r:
                stated.setdefault((s, r), set()).add(k)
    edges: dict[str, list] = {}
    gender: dict[str, str] = {}
    for depth, seeds in ((s_depth, {s for s, _ in stated}), (r_depth, {r for _, r in stated})):
        frontier = set(seeds)
        for _ in range(depth + 1):                  # the frontier's own edges, then the next ring's
            frontier = neighbours(sorted(frontier), edges, gender) - set(edges)
            print(f"  fetched {len(edges)} people, next ring {len(frontier)}", flush=True)
    data = {"kinds": [{"qid": k, "label": names.get(k), "n": n} for k, n in kinds],
            "taxonomy": {k: v for k, v in tax.items()}, "labels": names,
            "pairs": [{"s": s, "r": r, "kinds": sorted(ks)} for (s, r), ks in sorted(stated.items())],
            "edges": edges, "gender": gender}
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=60)
    ap.add_argument("--top", type=int, default=70)
    # one ring each side reaches four hops: Wikidata declares father/mother the inverse of child and sibling and
    # spouse symmetric (P1696), so an edge fetched from the relative's side also runs the other way
    ap.add_argument("--s-depth", type=int, default=1, help="rings of the subject's family fetched past the subject")
    ap.add_argument("--r-depth", type=int, default=1, help="rings of the relative's family fetched past the relative")
    ap.add_argument("--relatives-only", action="store_true",
                    help="add every pair's relative statements (both ways) to an existing kin.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.relatives_only:
        data = json.loads((OUT / "kin.json").read_text(encoding="utf-8"))
        people = sorted({p["s"] for p in data["pairs"]} | {p["r"] for p in data["pairs"]})
        data["relatives"] = relative_statements(people)
        extra = sorted({k for rows in data["relatives"].values() for _y, ks in rows for k in ks} - set(data["labels"]))
        data["labels"].update(labels(extra))
        for k, sups in taxonomy(extra).items():
            data["taxonomy"].setdefault(k, sups)
        data["labels"].update(labels(sorted({s for v in data["taxonomy"].values() for s in v} - set(data["labels"]))))
        (OUT / "kin.json").write_text(json.dumps(data), encoding="utf-8")
        print(f"relative statements for {len(people)} people: {sum(len(v) for v in data['relatives'].values())} "
              f"({time.time() - t0:.0f}s)")
        return
    data = build(a.per_kind, a.top, a.s_depth, a.r_depth)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kin.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{len(data['pairs'])} pairs, {len(data['edges'])} people, "
          f"{sum(len(v) for v in data['edges'].values())} family edges, {len(data['gender'])} genders "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
