"""cubbyllm/sources/wikidata_resolve.py -- which item a name means: the askers' choices learned, the preferred item,
the links a served claim made, the resolution tiers (the item a claim pointed at, one exact hit, the next hop, label
over alias). Split out of wikidata.py (order 0031) as a mixin of WikidataSource, verbatim.

Wired: WIRED (WikidataSource).
"""
from __future__ import annotations

import json
import os
import time
import unicodedata
from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402
from .wikidata_aliases import _META

__wiring__ = "WIRED"

_HONORIFIC = ("saint", "dr", "doctor")


def name_key(s: str) -> str:
    """A name as it is compared with a search hit's label or alias: normalized; with ``CB_NAME_MATCH=1`` (off by
    default), its accents folded and "st" read as "saint" (Cowork's sorting of the 76 refusals, 2026-10-24: "diego
    velazquez" is Diego Velázquez, "st louis" is St. Louis)."""
    k = _normalize(s or "")
    if os.environ.get("CB_NAME_MATCH") != "1":
        return k
    k = "".join(c for c in unicodedata.normalize("NFKD", k) if not unicodedata.combining(c))
    return " ".join("saint" if w == "st" else w for w in k.split())


def bare_key(s: str) -> str:
    """`name_key` without a leading honorific (saint, dr, doctor); the name itself off ``CB_NAME_MATCH=1``."""
    w = name_key(s).split()
    while os.environ.get("CB_NAME_MATCH") == "1" and len(w) > 1 and w[0] in _HONORIFIC:
        w = w[1:]
    return " ".join(w)


def _exact(hit: dict, key: str) -> bool:
    """A search hit whose label or matched alias is the name."""
    return name_key(hit.get("label", "")) == key or name_key(((hit.get("match") or {}).get("text") or "")) == key


class WikiResolve:
    """WikidataSource's entity resolution (a mixin; order 0031)."""

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

    def _one_kind(self, cands) -> bool:
        """Every candidate shares a class (P31): two of a kind, which no attribute and no label separates."""
        classes = [{(st.get("mainsnak") or {}).get("datavalue", {}).get("value", {}).get("id")
                    for st in self._claims(h["id"]).get("P31", [])} - {None} for h in cands]
        return bool(classes) and all(classes) and bool(set.intersection(*classes))

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
        key = name_key(entity)
        if via and not qid and len(self.links.get(_normalize(via), ())) == 1:
            qid = self.links[_normalize(via)][0]
            self.last.update(how="linked", via=via)
            return qid, self._labels_for([qid]).get(qid) or entity
        s = self._get({"action": "wbsearchentities", "search": entity, "language": "en", "limit": 5})
        hits = (s or {}).get("search") or []
        if not any(_exact(h, key) for h in hits) and bare_key(entity) != key:
            # nothing carries the name: once more without its leading honorific ("st elizabeth ann seton" is
            # Elizabeth Ann Seton), and only then, so "saint louis" stays St. Louis (CB_NAME_MATCH=1, off by default)
            b = bare_key(entity)
            s = self._get({"action": "wbsearchentities", "search": b, "language": "en", "limit": 5})
            hits2 = (s or {}).get("search") or []
            if any(_exact(h, b) for h in hits2):
                hits, key = hits2, b
        if not any(_exact(h, key) for h in hits) and os.environ.get("CB_PLACE_QUALIFIER") == "1":
            from .wikidata_place import resolve_place     # no item of the whole name: "cleveland ohio" read as a
            got = resolve_place(self, entity)              #   name and where it is (off by default)
            if got:
                self.last.update(how=got[2])
                return got[0], got[1]
            if self.last.get("ambiguous"):                 # several in that place: the asker decides
                return None
        if not hits:
            self.last["unresolved"] = True
            return None
        # exp_r11 Gemini run 2 (2026-09-12): the proposer's seed 'james young' (the question
        # said 'James Young (Missouri politician)') matched the first of several items
        # labelled James Young, and a wrong birth year was verified and SPOKEN. An entity
        # label shared by two or more items is an ambiguity the host never picks among by
        # rank. A search whose hits carry neither the label nor an alias equal to the query
        # is no resolution either (the old fallback to hits[0] was a guess).
        exact = [h for h in hits if _exact(h, key)]
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
            labelled = [h for h in cands if name_key(h.get("label", "")) == key]
            # ...and the label decides only among DIFFERENT kinds of thing, by the James Young rule above
            # (2026-10-24, live: "where was ben franklin born?" said Carthage. A minor item is LABELLED "Ben
            # Franklin"; Benjamin Franklin carries it as an alias; two people, and which one Wikidata labels
            # with the nickname is an accident of naming). Two of a kind: the asker decides. CB_NAMESAKE_TIER=1
            # keeps the label's old reach, as it does for the hdc world's tier (order 0036).
            if len(labelled) == 1 and (os.environ.get("CB_NAMESAKE_TIER") == "1" or not self._one_kind(cands)):
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
