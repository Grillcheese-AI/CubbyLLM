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

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .wikidata_aliases import (CACHE, PROPERTY_TABLE, PropertyAliases, _META, _inflections,  # noqa: F401
                               property_aliases)
from .wikidata_resolve import WikiResolve

__wiring__ = "WIRED"

API = "https://www.wikidata.org/w/api.php"
UA = "cubbyllm-standin/0.1 (research; licensing@grillcheese.ai)"
SKIP_PROPS = {"P31"}       # 'instance of' floods every entity with class facts; the store calls it 'instance' when it wants it
CLASS_PROPS = ("P31",)     # ... and a profile ask WANTS it: "what is X" is a question about the kind
MAX_CLASS_FACTS = 3        # the kind, not the whole taxonomy


from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402


class WikidataSource(WikiResolve):
    name = "wikidata"

    def __init__(self, cache_dir: pathlib.Path | str | None = CACHE, sleep_s: float = 0.2,
                 max_facts: int = 300, offline: bool = False, aliases: "PropertyAliases | None" = None,
                 online_relations: bool = False, use_choices: bool = False, fetch=None) -> None:
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.sleep_s, self.max_facts, self.offline = sleep_s, max_facts, offline
        # url -> the response body: the server hands it in (cubby_serve.sources.http.wikidata()); this package opens
        # no socket (order 0031). Offline, or a cached request, never needs it
        self.fetch = fetch
        self.calls = 0
        self.last: dict = {}
        self._labels: dict[str, str] = {}
        # relations: the local table first (zero calls); the property-search API only when
        # asked for explicitly -- serving never depends on a live call per wording
        self.aliases = aliases if aliases is not None else property_aliases()
        self.online_relations = online_relations
        self.use_choices = bool(use_choices)             # serving learns what askers mean; the benches do not
        self.choices: dict[str, dict[str, int]] = {}
        cp = self._choices_path()
        if self.use_choices and cp is not None and cp.exists():
            try:
                self.choices = json.loads(cp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.choices = {}
        # label -> the item(s) it named in claims this source served; persisted beside the cache
        self.links: dict[str, list[str]] = {}
        self._links_dirty = False
        p = self._links_path()
        if p is not None and p.exists():
            try:
                self.links = {k: list(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            except (OSError, ValueError):
                self.links = {}

    # -- transport, cached ---------------------------------------------------------------
    def _get(self, params: dict) -> dict | None:
        params = dict(params, format="json")
        key = hashlib.sha256(urllib.parse.urlencode(sorted(params.items())).encode()).hexdigest()[:24]
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            return None
        if self.fetch is None:
            raise RuntimeError("an online WikidataSource needs fetch= (cubby_serve.sources.http.wikidata()): "
                               "cubbyllm/ opens no socket")
        url = f"{API}?{urllib.parse.urlencode(params)}"
        self.calls += 1
        try:
            data = json.loads(self.fetch(url).decode("utf-8"))
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

    def kind(self, label: str) -> str | None:
        """The value kind of a property label (date / number / name), from the local table
        only; None without one. The host narrows an ambiguous wording by it, never by rank."""
        return self.aliases.kind(label) if self.aliases is not None else None

    def wordings(self, label: str) -> list[str]:
        """Every English wording of the property this label names (local table only)."""
        return self.aliases.wordings(label) if self.aliases is not None else []

    # -- the Source call -----------------------------------------------------------------
    def _claims(self, qid: str) -> dict:
        e = self._get({"action": "wbgetentities", "ids": qid, "props": "claims", "languages": "en"})
        return (((e or {}).get("entities") or {}).get(qid) or {}).get("claims") or {}

    def _links_path(self):
        return self.cache / "_links.json" if self.cache else None

    # -- the choice ledger: what ASKERS meant by a name, and the VM none the wiser ------------
    # Nick, 2026-09-14: "an algorithm that allows the vm to still be neutral while scoring most
    # requested answers, this way we will eventually be able to guess the context if none".
    # The discipline: a tally of the items askers PICKED in a clarify turn, kept host-side beside the
    # cache. It never enters a program, a similarity or a verdict -- the VM verifies exactly what it
    # verified before, and a fact reached this way is gated like any other. It only chooses WHICH item
    # to ask the source about, it ranks below every context tier (the hop the question needs, label over
    # alias), and it is on the record (`how`) whenever it decided. Off by default: the benches must stay
    # a function of the code and the data, so only serving (`use_choices=True`) consults or writes it.
    def _choices_path(self):
        return self.cache / "_choices.json" if self.cache else None

    @staticmethod
    def _key(entity: str, among: list[str]) -> str:
        """The ambiguity, not the name: (normalized surface form, the candidate set that survived the
        deterministic tiers). A tally keyed on the name alone silently carries onto a DIFFERENT ambiguity
        when the source gains a namesake -- the field's Q5, and the `james young` defect wearing a new
        coat. A changed candidate set is a new key with no history, so the asker is asked again."""
        return _normalize(entity) + "|" + ",".join(sorted(among))


    # a sense's pull halves in a quarter: what a name meant last year is not what it means now.
    HALF_LIFE_S = 90 * 86400
    MIN_ASKERS, MIN_MASS, MIN_MARGIN, MIN_SHARE = 3, 3.0, 2.0, 0.7


    # a statement's TIME qualifiers, dropped until 2026-09-14. 9-18% of statements carry one, and they
    # are the ones that matter: `position held` 2011-2019, `spouse` 1895-1906, `educated at` 1891-1893.
    # Without them the store cannot say WHEN a fact was true, so "who was governor in 2015" has two true
    # answers and no way to choose (the 'as of' residue of exp_r11 and the hdc emitter arm), and no fact
    # is datable, so "what happened in <year>" has nothing to read. The fact TEXT is untouched -- the
    # template, the gate, the walk and the VM see exactly what they saw -- and the time rides beside it
    # in `self.times`, the way provenance does.
    TIME_QUALIFIERS = {"P580": "start", "P582": "end", "P585": "point"}

    @staticmethod
    def _qual_times(st: dict) -> dict:
        out = {}
        for pid, slot in WikidataSource.TIME_QUALIFIERS.items():
            for q in (st.get("qualifiers") or {}).get(pid, []):
                v = (q.get("datavalue") or {}).get("value") or {}
                t = str(v.get("time") or "")
                if t[:1] in "+-" and len(t) >= 5:
                    # the SIGN is the era. Dropping it turns 2560 BC into 2560 AD -- 4,586 years of
                    # silent error (Nick, 2026-09-14, on the Great Pyramid). A BCE time keeps its minus.
                    out[slot] = ("-" if t.startswith("-") else "") + t[1:11]
        return out

    def facts(self, entity: str, relations: list[str] | None = None, via: str | None = None,
              qid: str | None = None, asker: str | None = None, classes: bool = False) -> list["Triple"]:
        """Returns `Triple(obj, rel, subj)`, NOT fact sentences. (The annotation
        said `list[str]` until 2026-09-14; it had been returning structured
        triples since the store was changed to be told where REL ends, and the
        stale annotation cost a caller a debugging round.)

        `relations`: the wordings the walk needs from this entity (the stalled hop's), used
        only to decide among several items that share the name; `via`: the walked fact whose
        object this entity is, which names the item outright. Neither keeps the strict rule
        from applying when they are absent.

        `classes`: also bring back P31, capped. It is skipped by default because a class fact
        answers no hop and floods the store -- but for "WHAT is the Bank of England" the class IS
        the answer, and without it nothing licenses the word "bank". 2026-09-14, live: the guard
        refused the whole paraphrase over "institution" while the store was not allowed to know
        the kind. An unlicensed kind is still refused; this just lets a fact license it."""
        self.last = {"entity": entity, "qid": None, "label": None, "alias": False, "n_claims": 0, "how": None}
        self.times: dict[str, dict] = {}                     # fact text -> {start, end, point}, beside the facts
        r = self.resolve(entity, relations, via, qid, asker)
        if r is None:
            return []
        qid, label = r
        alias = _normalize(label) != _normalize(entity)
        self.last.update(qid=qid, label=label, alias=alias)
        subj = entity if alias else label            # the user's words when the hit came through an alias
        claims = self._claims(qid)
        self.last["n_claims"] = sum(len(v) for v in claims.values())
        # collect item targets and property ids, one label round-trip for all of them
        props = [p for p in claims if p not in SKIP_PROPS or (classes and p in CLASS_PROPS)]
        if relations and self.aliases is not None:
            # the walk's own relations FIRST (2026-10-16, live: 'who is the prime minister of canada?' fetched
            # Canada's 300 facts, every one a duplicate, and `head of government` was not among them -- an
            # item with hundreds of statements is cut at `max_facts` in claim order, and the one property
            # the question asked for sat past the cut). The order of the rest is the item's own.
            wanted = {pid for r in relations for pid in self.aliases.pids(r)}
            props = [p for p in props if p in wanted] + [p for p in props if p not in wanted]
        if classes:
            claims = dict(claims)
            for cp in CLASS_PROPS:                           # the kind, not a taxonomy: the first few
                if cp in claims:
                    claims[cp] = claims[cp][:MAX_CLASS_FACTS]
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
                    if obj:                          # the claim points at an ITEM: a walk through this fact asks about it, not about its name
                        self._link(f"{obj} is the {rel} of {subj}", v["id"])
                elif t == "time":
                    ts, prec = v.get("time", ""), v.get("precision", 11)
                    obj = ts[1:5] if prec <= 9 else ts[1:11]
                    if ts.startswith("-"):
                        # A negative year is BEFORE the era. The fact TEXT is what gets spoken, so it
                        # says so: "2560 BC", not "2560". Day precision on a BCE date is noise -- the
                        # year is the claim. (self.times keeps the signed ISO for the machine side.)
                        obj = (ts[1:5].lstrip("0") or "0") + " BC"
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
                    when = self._qual_times(st)
                    if when:
                        self.times[_normalize(f"{obj} is the {rel} of {subj}")] = when
                if len(out) >= self.max_facts:
                    self._save_links()
                    return out
        self._save_links()
        return out
