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
# Wikimedia's own pages ABOUT pages: never the subject of a question (2026-09-14)
_META = __import__("re").compile(r"\bWikimedia\b.*\b(disambiguation|list|category|template|module|portal|project)\b"
                                 r"|\b(disambiguation page|list article|Wikinews article)\b", __import__("re").I)


from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402


PROPERTY_TABLE = CACHE.parent / "wikidata_properties_en_fr.json"        # standin/data/out/, built once by build_property_aliases.py


def _inflections(key: str) -> list[str]:
    """The regular inflections of a normalized wording's FIRST word (the verb of 'work for',
    'die', 'attend'): -s/-es/-d/-ed/-ing forms and the stems they come from. Exact strings
    to look up, no guessing beyond them."""
    words = key.split()
    if not words:
        return []
    w, rest = words[0], words[1:]
    forms = {w + "s", w + "es", w + "d", w + "ed", w + "ing"}
    if w.endswith("e"):
        forms.add(w[:-1] + "ing")
    if w.endswith("y"):
        forms.add(w[:-1] + "ied"); forms.add(w[:-1] + "ies")
    stems: set[str] = set()
    for suf in ("ies", "ied", "ing", "es", "ed", "s", "d"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            stem = w[:-len(suf)]
            stems.add(stem)
            if suf in ("ies", "ied"):
                stems.add(stem + "y")
            if suf in ("ing", "ed", "d"):
                stems.add(stem + "e")
    for s in stems:                                              # 'establishing' -> 'establish' -> 'established'
        forms.update({s, s + "s", s + "es", s + "d", s + "ed", s + "ing"})
    forms.discard(w)
    return sorted(" ".join([f] + rest) for f in forms)


class PropertyAliases:
    """Wikidata's property labels and aliases (EN + FR) as a LOCAL relation resolver -- the
    same answer `WikidataSource.relations()` used to fetch per wording (875 API calls for
    100 questions, exp_r14 2026-09-12), read from a table built once. `relations(text)`:
    the labels (English) of every property whose label or alias, in either language, IS
    the wording. Exact tier only, like the API version; the host still intersects with what
    the store holds and refuses more than one."""
    name = "property_aliases"
    # a property's datatype -> the value kind `plan_verify.ask_type` names ('when' -> date,
    # 'how many' -> number, 'who' -> name); the rest (external ids, urls, media) are no
    # kind a question asks for and never narrow anything
    KIND = {"time": "date", "quantity": "number", "wikibase-item": "name", "string": "name", "monolingualtext": "name"}

    def __init__(self, path: pathlib.Path | str = PROPERTY_TABLE) -> None:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.meta = {k: v for k, v in d.items() if k != "properties"}
        self._by_text: dict[str, set[str]] = {}
        self._texts_of: dict[str, set[str]] = {}         # English label -> every English wording of its property
        self._kind: dict[str, str | None] = {}           # any English wording -> its property's kind (None when they disagree)
        self._pid_of: dict[str, str] = {}                # English label -> the property id (the claim key an item carries)
        for pid, p in d["properties"].items():
            label_en = p["label"].get("en")
            if not label_en:
                continue
            self._pid_of[_normalize(label_en)] = pid
            k = self.KIND.get(p.get("datatype") or "")
            en_texts = [t for t in [label_en] + list(p["aliases"].get("en", [])) if t]
            self._texts_of.setdefault(_normalize(label_en), set()).update(en_texts)
            for lang in ("en", "fr"):
                for text in [p["label"].get(lang)] + list(p["aliases"].get(lang, [])):
                    if text:
                        self._by_text.setdefault(_normalize(text), set()).add(label_en)
            for text in en_texts:
                key = _normalize(text)
                if key in self._kind and self._kind[key] != k:
                    self._kind[key] = None               # two properties share this wording and disagree on kind: no kind
                else:
                    self._kind[key] = k

    def __len__(self) -> int:
        return len(self._by_text)

    def relations(self, text: str) -> list[str]:
        """Exact tier first; a wording the table lacks is tried as its inflections ('die' ->
        'died', 'attend' -> 'attended', 'work for' -> 'works for'), each an exact lookup."""
        key = _normalize(text)
        hit = self._by_text.get(key)
        if hit:
            return sorted(hit)
        out: set[str] = set()
        for alt in _inflections(key):
            out.update(self._by_text.get(alt, ()))
        return sorted(out)

    def wordings(self, label: str) -> list[str]:
        """Every English wording (label and aliases) of the property whose English label this
        is: 'date of birth' -> ['DOB', 'birth date', 'born', ...]. A store that holds the
        property under one of these ('birth date') holds the label's relation."""
        return sorted(self._texts_of.get(_normalize(label), ()))

    def pids(self, wording: str) -> list[str]:
        """The property ids a wording names, through its labels: 'given name' -> ['P735'],
        'born' -> ['P569', 'P19']. What an item's claims are keyed by."""
        out = []
        for label in self.relations(wording) or [wording]:
            pid = self._pid_of.get(_normalize(label))
            if pid and pid not in out:
                out.append(pid)
        return out

    def kind(self, label: str) -> str | None:
        """'date' / 'number' / 'name' for a wording whose property datatype says so; None
        when the table has no datatype for it (built before 2026-09-13), when properties
        sharing the wording disagree ('born': date of birth / place of birth), or for a
        kind no question asks for. None never narrows: the host keeps the ambiguity."""
        return self._kind.get(_normalize(label))


def property_aliases(path: pathlib.Path | str = PROPERTY_TABLE) -> "PropertyAliases | None":
    """The local table if it has been built, else None (the caller may fall back to the API)."""
    return PropertyAliases(path) if pathlib.Path(path).is_file() else None


class WikidataSource:
    name = "wikidata"

    def __init__(self, cache_dir: pathlib.Path | str | None = CACHE, sleep_s: float = 0.2,
                 max_facts: int = 300, offline: bool = False, aliases: "PropertyAliases | None" = None,
                 online_relations: bool = False, use_choices: bool = False) -> None:
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

    def choose(self, entity: str, qid: str, among: list[str] | None = None, asker: str | None = None) -> None:
        """Record that an asker meant THIS item by that name. One vote per asker per ambiguity, last
        write wins -- repeating a question is not a second vote, and a person may change their mind.
        Only an EXPLICIT clarify answer writes here: never an auto-bound decision (which would freeze
        the prior into its own past), never 'the VM verified it', never the absence of a correction."""
        if not self.use_choices or not among:
            return
        key = self._key(entity, among)
        self.choices.setdefault(key, {})[str(asker or "anon")] = {"item": qid, "t": int(time.time())}
        p = self._choices_path()
        if p is not None:
            try:
                p.write_text(json.dumps(self.choices, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass

    # a sense's pull halves in a quarter: what a name meant last year is not what it means now.
    HALF_LIFE_S = 90 * 86400
    MIN_ASKERS, MIN_MASS, MIN_MARGIN, MIN_SHARE = 3, 3.0, 2.0, 0.7

    def preferred(self, entity: str, among: list[str], asker: str | None = None) -> tuple[str, str] | None:
        """(qid, how) when the ledger may bind this name to one item, else None -- and None is a refusal
        to guess, which is always available. Two tiers, both about which item was MEANT and neither about
        what is true:

          * THE SAME ASKER already said, for this very ambiguity, which item they meant. One observation
            is enough because it is not a vote about the world, it is this person's own stated referent.
          * OTHER ASKERS, and then only well past a coin flip: at least MIN_ASKERS distinct askers with a
            live vote, a decayed margin of MIN_MARGIN over the runner-up and MIN_SHARE of the mass. One
            observation, a 51/49 split, or a sense that was popular last year and is not now all fail it.
        """
        if not self.use_choices or not among:
            return None
        votes = self.choices.get(self._key(entity, among)) or {}
        now = time.time()
        mine = votes.get(str(asker or "anon"))
        if mine and mine.get("item") in among:
            return mine["item"], "your earlier choice"
        mass: dict[str, float] = {}
        for who, v in votes.items():
            if who == str(asker or "anon") or v.get("item") not in among:
                continue
            w = 0.5 ** ((now - float(v.get("t", now))) / self.HALF_LIFE_S)
            if w >= 1 / 16:                                  # a year old at this half-life: dead, not faint
                mass[v["item"]] = mass.get(v["item"], 0.0) + w
        if len(mass) < 1:
            return None
        total = sum(mass.values())
        ranked = sorted(mass.items(), key=lambda kv: -kv[1])
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else ("", 0.0))
        n_askers = sum(1 for who, v in votes.items() if v.get("item") in among and who != str(asker or "anon")
                       and 0.5 ** ((now - float(v.get("t", now))) / self.HALF_LIFE_S) >= 1 / 16)
        if (n_askers >= self.MIN_ASKERS and total >= self.MIN_MASS
                and top[1] >= second[1] + self.MIN_MARGIN and top[1] / total >= self.MIN_SHARE):
            return top[0], f"asker history ({top[1]:.1f} of {total:.1f}, {n_askers} askers)"
        return None

    def _link(self, fact: str, qid: str) -> None:
        """Remember which ITEM the object of a served claim IS, keyed by the FACT ('jim haslam is
        the father of bill haslam' -> Q6195484), so the walk's next hop, reached through that
        fact, asks about that item and not about a string several items are labelled with.
        Keyed by the fact, never by the name alone (2026-09-14: a name-keyed link sent 'Marie
        Curie' to a film named after her, the object of one of her own claims)."""
        key = _normalize(fact)
        if not key or qid in self.links.get(key, ()):
            return
        self.links.setdefault(key, []).append(qid)  # two items behind one fact text: the link tier stands aside (see resolve)
        self._links_dirty = True

    def _save_links(self) -> None:
        p = self._links_path()
        if p is not None and self._links_dirty:
            try:
                p.write_text(json.dumps(self.links, ensure_ascii=False), encoding="utf-8")
                self._links_dirty = False
            except OSError:
                pass

    def resolve(self, entity: str, relations: list[str] | None = None, via: str | None = None,
                qid: str | None = None, asker: str | None = None) -> tuple[str, str] | None:
        """(qid, label) for the entity, or None. Three tiers, each deterministic, none a pick
        by rank: (1) LINKED -- the entity is the object of a fact this source served (`via`:
        the walked fact that reached it), so the item is the one that claim pointed at; (2) the
        search's exact hits (label or alias equal to the query) when there is exactly one; (3)
        the QUESTION decides among several exact hits: those whose claims carry the property the
        walk needs next (`relations`: the plan's wording for the stalled hop -- a book edition
        and a ferry named 'Marie Curie' have no date of birth), then, among those, the item
        whose LABEL is the name the question used over items that only carry it as an alias
        ('Jim Haslam' over 'Jimmy Haslam'). Still more than one -> ambiguous, refused."""
        key = _normalize(entity)
        if via and not qid and len(self.links.get(_normalize(via), ())) == 1:
            qid = self.links[_normalize(via)][0]
            self.last.update(how="linked", via=via)
            return qid, self._labels_for([qid]).get(qid) or entity
        s = self._get({"action": "wbsearchentities", "search": entity, "language": "en", "limit": 5})
        hits = (s or {}).get("search") or []
        if not hits:
            self.last["unresolved"] = True
            return None
        # exp_r11 Gemini run 2 (2026-09-12): the proposer's seed 'james young' (the question
        # said 'James Young (Missouri politician)') matched the first of several items
        # labelled James Young, and a wrong birth year was verified and SPOKEN. An entity
        # label shared by two or more items is an ambiguity the host never picks among by
        # rank. A search whose hits carry neither the label nor an alias equal to the query
        # is no resolution either (the old fallback to hits[0] was a guess).
        exact = [h for h in hits if _normalize(h.get("label", "")) == key
                 or _normalize(((h.get("match") or {}).get("text") or "")) == key]
        # a Wikimedia meta-page (disambiguation, list article, category, template) is never the thing a
        # question is about: dropping them is a filter on what an item IS, not a pick among candidates
        # (2026-09-14, live: 'the last roman emperor' was two hits -- the legendary figure and a
        # disambiguation page -- and the pair was refused as ambiguous)
        real = [h for h in exact if not _META.search(str(h.get("description") or ""))]
        cands = real if real else exact
        how = "exact" + ("" if len(real) == len(exact) else " (meta-pages dropped)")
        if len(cands) > 1 and relations and self.aliases is not None:
            # the relations the question needs, IN ORDER, narrowing at each step -- specific evidence
            # before generic. Taking their union was too weak (2026-09-14, live: 'where is quebec city'
            # stayed ambiguous between the city and an 1841 electoral district because BOTH carry
            # `country`; the city alone carries `located in the administrative territorial entity`,
            # which the where-ask asks for first). Lexicographic, not a score: the evidence is the
            # question's own relation set, so a famous item never wins for being famous.
            claims = {h["id"]: self._claims(h["id"]) for h in cands}
            # ...but only among candidates that are DIFFERENT KINDS of thing. Two items of one kind are
            # separated by this only through an accident of which is more completely described, and that
            # is the James Young incident again (2026-09-12: two men of one name, a wrong birth year
            # spoken). If every candidate shares a class, no attribute decides -- the asker does.
            classes = [{(st.get("mainsnak") or {}).get("datavalue", {}).get("value", {}).get("id")
                        for st in claims[h["id"]].get("P31", [])} for h in cands]
            of_a_kind = bool(classes) and bool(set.intersection(*classes))
            if not of_a_kind:
                for r in relations:
                    if len(cands) == 1:
                        break
                    pids = self.aliases.pids(r)
                    if not pids:
                        continue
                    having = [h for h in cands if any(p in claims[h["id"]] for p in pids)]
                    if having and len(having) < len(cands):
                        cands, how = having, f"relation: {r}"
        if len(cands) > 1:
            labelled = [h for h in cands if _normalize(h.get("label", "")) == key]
            if len(labelled) == 1:
                cands, how = labelled, how + " + label over alias"
        if qid:                                              # the USER named the item (the panel's clarify choice):
            # recorded against the candidate set they were OFFERED, which is why the search ran first --
            # a vote keyed on the bare name would carry onto a different ambiguity later (the field's Q5)
            self.choose(entity, qid, [h["id"] for h in cands], asker)
            self.last.update(how="chosen by the asker")
            return qid, self._labels_for([qid]).get(qid) or entity
        if len(cands) > 1:                                   # no context left: what askers have meant by this name
            offered = [h["id"] for h in cands]
            self.last["offered"] = offered                   # the ambiguity the ledger is keyed on
            pref = self.preferred(entity, offered, asker)
            if pref:
                cands = [h for h in cands if h["id"] == pref[0]]
                how = pref[1]
        if len(cands) != 1:
            self.last["ambiguous"] = [(h["id"], h.get("label"), h.get("description")) for h in cands]
            self.last["unresolved"] = not exact
            return None
        hit = cands[0]
        self.last.update(how=how)
        return hit["id"], hit.get("label") or entity

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
                if t.startswith("+") and len(t) >= 5:
                    out[slot] = t[1:11]                      # YYYY-MM-DD, the store's own date shape
        return out

    def facts(self, entity: str, relations: list[str] | None = None, via: str | None = None,
              qid: str | None = None, asker: str | None = None) -> list[str]:
        """`relations`: the wordings the walk needs from this entity (the stalled hop's), used
        only to decide among several items that share the name; `via`: the walked fact whose
        object this entity is, which names the item outright. Neither keeps the strict rule
        from applying when they are absent."""
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
                    if obj:                          # the claim points at an ITEM: a walk through this fact asks about it, not about its name
                        self._link(f"{obj} is the {rel} of {subj}", v["id"])
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
                    when = self._qual_times(st)
                    if when:
                        self.times[_normalize(f"{obj} is the {rel} of {subj}")] = when
                if len(out) >= self.max_facts:
                    self._save_links()
                    return out
        self._save_links()
        return out
