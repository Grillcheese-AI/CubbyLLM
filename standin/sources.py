"""sources — where a fact the store lacks can be fetched from (search-and-learn, lever 3).

Wired: WIRED (stand-in serve stack; implements `cubbyllm.reasoning.learn.Source`, which is
`facts(entity) -> list[str]` of template facts). The network lives HERE, not in cubbyllm/:
the package chooses and gates, the serve stack fetches under policy.

`WikidataSource`: one entity label -> the item's claims, rendered as `OBJ is the <property
label> of <entity>` — the property labels are the vocabulary the eval store's relations came
from ('country of citizenship', 'award received', 'place of birth'), which is the vocabulary the
emitter already names. Item values render as their English label, times as ISO dates (year when
that is all the precision there is), quantities as the amount, strings as themselves. Every value
of a multi-valued property becomes its own fact — the walk's tie-break, not this source, decides
between them, and a question that needs a qualifier the store cannot hold is the disposer's to
refuse. Responses are cached as JSON under `data/out/wikidata_cache/` so a rerun is offline and
byte-identical, and `last` records the QID, the label and whether the hit came through an alias.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

__wiring__ = "WIRED"

API = "https://www.wikidata.org/w/api.php"
UA = "cubbyllm-standin/0.1 (research; licensing@grillcheese.ai)"
CACHE = pathlib.Path(__file__).resolve().parent / "data" / "out" / "wikidata_cache"
SKIP_PROPS = {"P31"}       # 'instance of' floods every entity with class facts; the store calls it 'instance' when it wants it


from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402


PROPERTY_TABLE = CACHE.parent / "wikidata_properties_en_fr.json"        # standin/data/out/, built once by build_property_aliases.py


class PropertyAliases:
    """Wikidata's property labels and aliases (EN + FR) as a LOCAL relation resolver -- the
    same answer `WikidataSource.relations()` used to fetch per wording (875 API calls for
    100 questions, exp_r14 2026-09-12), read from a table built once. `relations(text)`:
    the labels (English) of every property whose label or alias, in either language, IS
    the wording. Exact tier only, like the API version; the host still intersects with what
    the store holds and refuses more than one."""
    name = "property_aliases"

    def __init__(self, path: pathlib.Path | str = PROPERTY_TABLE) -> None:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.meta = {k: v for k, v in d.items() if k != "properties"}
        self._by_text: dict[str, set[str]] = {}
        for pid, p in d["properties"].items():
            label_en = p["label"].get("en")
            if not label_en:
                continue
            for lang in ("en", "fr"):
                for text in [p["label"].get(lang)] + list(p["aliases"].get(lang, [])):
                    if text:
                        self._by_text.setdefault(_normalize(text), set()).add(label_en)

    def __len__(self) -> int:
        return len(self._by_text)

    def relations(self, text: str) -> list[str]:
        return sorted(self._by_text.get(_normalize(text), ()))


def property_aliases(path: pathlib.Path | str = PROPERTY_TABLE) -> "PropertyAliases | None":
    """The local table if it has been built, else None (the caller may fall back to the API)."""
    return PropertyAliases(path) if pathlib.Path(path).is_file() else None


class WikidataSource:
    name = "wikidata"

    def __init__(self, cache_dir: pathlib.Path | str | None = CACHE, sleep_s: float = 0.2,
                 max_facts: int = 300, offline: bool = False, aliases: "PropertyAliases | None" = None,
                 online_relations: bool = False) -> None:
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.sleep_s, self.max_facts, self.offline = sleep_s, max_facts, offline
        self.calls = 0
        self.last: dict = {}
        self._labels: dict[str, str] = {}
        # relations: the local table first (zero calls); the property-search API only when
        # asked for explicitly -- serving never depends on a live call per wording
        self.aliases = aliases if aliases is not None else property_aliases()
        self.online_relations = online_relations

    # -- transport, cached ---------------------------------------------------------------
    def _get(self, params: dict) -> dict | None:
        params = dict(params, format="json")
        key = hashlib.sha256(urllib.parse.urlencode(sorted(params.items())).encode()).hexdigest()[:24]
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            return None
        url = f"{API}?{urllib.parse.urlencode(params)}"
        self.calls += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception:                                  # noqa: BLE001 -- a source that fails returns nothing
            return None
        if path:
            path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(self.sleep_s)
        return data

    def _labels_for(self, ids: list[str]) -> dict[str, str]:
        need = [i for i in dict.fromkeys(ids) if i not in self._labels]
        for i in range(0, len(need), 50):
            chunk = need[i:i + 50]
            data = self._get({"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels", "languages": "en"})
            for qid, ent in ((data or {}).get("entities") or {}).items():
                lab = ((ent.get("labels") or {}).get("en") or {}).get("value")
                if lab:
                    self._labels[qid] = lab
        return {i: self._labels[i] for i in ids if i in self._labels}

    # -- lever 4: wording -> the property labels it names -----------------------------------
    def relations(self, text: str) -> list[str]:
        """Property labels whose label or alias IS this wording ('born' -> ['date of
        birth'], 'citizenship' -> ['country of citizenship']); a prefix hit ('born' vs
        'born in') is not an alias and is left out. The host intersects with what the
        store holds and refuses more than one. Local table first; the API only with
        `online_relations=True` (or when no table has been built and we are not offline)."""
        if self.aliases is not None:
            local = self.aliases.relations(text)
            if local or not self.online_relations:
                return local
        elif self.offline:
            return []
        s = self._get({"action": "wbsearchentities", "search": text, "language": "en", "type": "property", "limit": 8})
        out = []
        for h in (s or {}).get("search") or []:
            m = ((h.get("match") or {}).get("text") or "")
            if _normalize(m) == _normalize(text) and h.get("label"):
                out.append(h["label"])
        return out

    # -- the Source call -----------------------------------------------------------------
    def facts(self, entity: str) -> list[str]:
        self.last = {"entity": entity, "qid": None, "label": None, "alias": False, "n_claims": 0}
        s = self._get({"action": "wbsearchentities", "search": entity, "language": "en", "limit": 5})
        hits = (s or {}).get("search") or []
        if not hits:
            return []
        # exp_r11 Gemini run 2 (2026-09-12): the proposer's seed 'james young' (the question
        # said 'James Young (Missouri politician)') matched the first of several items
        # labelled James Young, and a wrong birth year was verified and SPOKEN. An entity
        # label shared by two or more items is an ambiguity, and the host refuses those; it
        # never picks. A search whose hits carry neither the label nor an alias equal to the
        # query is no resolution either (the old fallback to hits[0] was a guess).
        exact = [h for h in hits if _normalize(h.get("label", "")) == _normalize(entity)
                 or _normalize(((h.get("match") or {}).get("text") or "")) == _normalize(entity)]
        if len(exact) != 1:
            self.last["ambiguous"] = [(h["id"], h.get("label"), h.get("description")) for h in exact] if exact else []
            self.last["unresolved"] = not exact
            return []
        hit = exact[0]
        qid, label = hit["id"], hit.get("label") or entity
        alias = _normalize(label) != _normalize(entity)
        self.last.update(qid=qid, label=label, alias=alias)
        subj = entity if alias else label            # the user's words when the hit came through an alias
        e = self._get({"action": "wbgetentities", "ids": qid, "props": "claims", "languages": "en"})
        claims = (((e or {}).get("entities") or {}).get(qid) or {}).get("claims") or {}
        self.last["n_claims"] = sum(len(v) for v in claims.values())
        # collect item targets and property ids, one label round-trip for all of them
        props = [p for p in claims if p not in SKIP_PROPS]
        targets = []
        for p in props:
            for st in claims[p]:
                dv = ((st.get("mainsnak") or {}).get("datavalue") or {})
                if dv.get("type") == "wikibase-entityid":
                    targets.append(dv["value"]["id"])
        labels = self._labels_for(props + targets)
        out: list[str] = []
        for p in props:
            rel = labels.get(p)
            if not rel:
                continue
            for st in claims[p]:
                snak = st.get("mainsnak") or {}
                if st.get("rank") == "deprecated" or snak.get("datatype") in ("external-id", "url", "commonsMedia", "globe-coordinate", "math", "musical-notation", "tabular-data", "geo-shape"):
                    continue                       # identifiers and media are not facts the walk can use
                dv = snak.get("datavalue") or {}
                v = dv.get("value"); t = dv.get("type")
                if t == "wikibase-entityid":
                    obj = labels.get(v["id"])
                elif t == "time":
                    ts, prec = v.get("time", ""), v.get("precision", 11)
                    obj = ts[1:5] if prec <= 9 else ts[1:11]
                elif t == "quantity":
                    obj = v.get("amount", "").lstrip("+")
                elif t == "string":
                    obj = v
                elif t == "monolingualtext":
                    obj = v.get("text")
                else:
                    obj = None
                if obj and " is the " not in obj:
                    out.append(Triple(obj=obj, rel=rel, subj=subj))     # structured: the store is told where REL ends
                if len(out) >= self.max_facts:
                    return out
        return out
