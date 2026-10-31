"""Scratch (untracked): order 0043, step 1 -- why the source refused each name. Live source (cached calls reused),
no ledger. Prints the search hits, which are exact, each candidate's kinds (P31), and which carry the relation."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cubbyllm.sources.wikidata import WikidataSource
from cubbyllm.sources.wikidata_resolve import _exact, name_key
from cubbyllm.sources.wikidata_aliases import _META
from cubby_serve.sources.http import wikidata as wikidata_fetch

CASES = [("uk", "located in time zone"), ("uk", "currency"), ("uk", "official language"), ("georgia", "capital"),
         ("cyprus", "currency"), ("obama", "place of birth"), ("king henry viii", "religion"),
         ("queen isabella", "mother"), ("ben franklin", "place of birth"), ("anne hathaway", "place of birth"),
         ("steve nash", "country of citizenship"), ("london", "located in time zone"), ("chicago", "located in time zone"),
         ("toronto", "located in time zone"), ("oregon", "capital"), ("iceland", "country"), ("george wilson", "occupation"),
         ("jack london", "occupation"), ("petershausen", "country"), ("notojima", "country")]
src = WikidataSource(offline=False, use_choices=False, fetch=wikidata_fetch())
out = []
for name, rel in CASES:
    got = src.resolve(name, relations=[rel])
    last = dict(src.last)
    s = src._get({"action": "wbsearchentities", "search": name, "language": "en", "limit": 5}) or {}
    hits = s.get("search") or []
    key = name_key(name)
    pids = src.aliases.pids(rel)
    rows = []
    for h in hits:
        claims = src._claims(h["id"])
        kinds = [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id") for st in claims.get("P31", [])]
        kinds = [k for k in kinds if k]
        labels = src._labels_for(kinds)
        rows.append({"id": h["id"], "label": h.get("label"), "desc": h.get("description"), "match": (h.get("match") or {}).get("text"),
                     "exact": _exact(h, key), "meta": bool(_META.search(str(h.get("description") or ""))),
                     "kinds": [labels.get(k, k) for k in kinds], "carries": any(p in claims for p in pids), "n_claims": len(claims)})
    rec = {"name": name, "relation": rel, "pids": pids, "resolved": got, "how": last.get("how"), "ambiguous": last.get("ambiguous"),
           "unresolved": last.get("unresolved"), "hits": rows}
    out.append(rec)
    print(json.dumps(rec, ensure_ascii=False)[:1800], flush=True)
json.dump(out, open("validation/logs/_diag_0043_names.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("calls", src.calls)
