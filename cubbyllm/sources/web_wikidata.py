"""cubbyllm/sources/web_wikidata.py -- the web source's hand-off to Wikidata: a Wikipedia hit's item, the places that
contain a place (so a finer value is not a disagreement), the item's claims, the article's lead. Split out of web.py
(order 0031) as a mixin of WebSource, verbatim.

Wired: WIRED (WebSource).
"""
from __future__ import annotations

import json
import urllib.robotparser
from cubbyllm.reasoning.planner import Triple, normalize
from .web_frames import _fold, _key

__wiring__ = "WIRED"


class WebWikidata:
    """WebSource's Wikidata hand-off and containment (a mixin; order 0031)."""

    def _qid(self, url: str) -> str | None:
        sp = urllib.parse.urlsplit(url)
        if not sp.path.startswith("/wiki/"):
            return None
        title = urllib.parse.unquote(sp.path[6:]).replace("_", " ")
        api = (f"https://{sp.netloc}/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "redirects": 1,
             "titles": title, "format": "json"}))

        def make():
            self.calls["wikipedia"] += 1
            got = self.http(api, {"Accept": "application/json"})
            if got is None:
                return None
            try:
                pages = json.loads(got[1].decode("utf-8", "replace"))["query"]["pages"]
            except (ValueError, KeyError):
                return None
            for pg in pages.values():
                q = (pg.get("pageprops") or {}).get("wikibase_item")
                if q:
                    return q
            return ""
        return self._cached("qid", api, make) or None

    def ancestors(self, place: str) -> set[str]:
        """The places that contain `place` per Wikidata (P131 / P17, six levels), for EVERY item whose label is
        exactly the name, intersected: a Paris in Texas and one in France vouch for nothing in common."""
        sets = self.items_up(place)
        return set.intersection(*sets) if sets else set()

    def items_up(self, place: str) -> list[set[str]]:
        """Per item labelled exactly `place`: the places that contain it."""
        key = _fold(place)
        if key in self._up:
            return self._up[key]
        wd, got = self.containment, []
        if wd is not None and key:
            try:
                s = wd._get({"action": "wbsearchentities", "search": place, "language": "en", "limit": 7}) or {}
                items = [h["id"] for h in s.get("search") or [] if _fold(h.get("label", "")) == key][:4]
                sets = []
                for qid in items:
                    seen, frontier = set(), {qid}
                    for _ in range(6):
                        nxt = {v["id"] for q in frontier for p in self.CONTAINED_BY for st in wd._claims(q).get(p, [])
                               for v in [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")]
                               if isinstance(v, dict) and v.get("id") and v["id"] not in seen}
                        seen |= nxt; frontier = nxt
                        if not frontier or len(seen) > 80:
                            break
                    sets.append({_fold(l) for l in (wd._labels_for(sorted(seen)) if seen else {}).values() if l})
                got = sets
            except Exception:                                # noqa: BLE001 -- no oracle is no containment, never a guess
                got = []
        self._up[key] = got
        return got

    def _finer(self, kind: str | None, a: str, b: str) -> bool:
        """Does value `a` say everything `b` says, and more? A date to the day says its year; 'Syracuse, New
        York' says New York; Warsaw says Poland when Wikidata places Warsaw in Poland."""
        if kind == "date":
            return a != b and a.startswith(b + "-")
        if kind != "name":
            return False
        pa = [_fold(x) for x in a.split(",") if _fold(x)]
        pb = [_fold(x) for x in b.split(",") if _fold(x)]
        if not pa or not pb or pa == pb:
            return False
        return pb[0] in pa[1:] or pb[0] in self.ancestors(a.split(",")[0])

    def _claims(self, evidence: dict) -> list[tuple[Triple, set[str], list[str]]]:
        """(the claim, the sites that vouch for it, the tally keys it merges), per relation today's evidence
        touched -- ONE value per relation when the web agrees, all of them when it does not.

        1. One spelling: values whose comma parts are prefixes of one another ('London, England, UK' /
           'London, England') are one claim, stated as the part all of them said; accents fold. Values that
           disagree anywhere stay apart.
        2. Entailment: a finer value vouches for a coarser one -- a site saying 'Syracuse, New York' says
           New York, a date to the day says its year. Never the other way: 'Poland' does not say Warsaw.
        3. Attested = vouched for by `min_sites` sites. If the attested values form one chain (each finer
           or coarser than the next), the finest is THE claim and nothing else is returned: the store gets
           one value and the walk can speak it. If they do not (two dates; London, England / London,
           Ontario), every attested value is returned and the walk refuses on the split. What is not
           attested stays in the tally, unreturned, when something else is -- one site's dissent is not a
           claim -- and is returned held when nothing is."""
        wanted = {(normalize(t.rel), normalize(t.subj)) for t, _s in evidence.values()}
        cands: dict[tuple[str, str], dict[str, tuple[str, str, str, set[str]]]] = {}
        for k, by_site in self.tally.items():
            e0 = next(iter(by_site.values()), None) if by_site else None
            if not isinstance(e0, dict) or "rel" not in e0:
                continue
            g = (normalize(e0["rel"]), normalize(e0["subj"]))
            if g in wanted:
                cands.setdefault(g, {})[k] = (e0["obj"], e0["rel"], e0["subj"], set(by_site))
        out = []
        for _g, members in cands.items():
            keys = list(members)
            kind = self.aliases.kind(members[keys[0]][1])
            parts = {k: [_fold(x) for x in members[k][0].split(",") if _fold(x)] for k in keys}

            def consistent(a, b):
                x, y = (parts[a], parts[b]) if len(parts[a]) <= len(parts[b]) else (parts[b], parts[a])
                return bool(x) and y[: len(x)] == x
            # 1. spellings
            claims: list[tuple[Triple, set[str], list[str]]] = []
            done: set[str] = set()
            for k in keys:
                if k in done:
                    continue
                comp = [k]
                if kind == "name":
                    stack = [k]
                    while stack:
                        c = stack.pop()
                        for o in keys:
                            if o not in comp and consistent(c, o):
                                comp.append(o); stack.append(o)
                    if len(comp) > 1 and not all(consistent(x, y) for x in comp for y in comp):
                        comp = [k]
                done.update(comp)
                shortest = min(comp, key=lambda c: len(parts[c]))
                n = len(parts[shortest])
                obj = ", ".join([x.strip() for x in members[shortest][0].split(",") if x.strip()][:n])
                _o, rel, subj, _s = members[shortest]
                claims.append((Triple(obj=obj, rel=rel, subj=subj), set().union(*(members[c][3] for c in comp)), sorted(comp)))
            # 1b. one place, two qualifiers: 'Matale, Sri Lanka' and 'Matale, Central Province' are one claim
            #     when ONE item named Matale lies in both (Wikidata); the claim is then the name alone
            if kind == "name" and self.containment is not None:
                claims = self._same_head(claims)
            # 2. entailment
            sup = [set(c[1]) for c in claims]
            supk = [set(c[2]) for c in claims]
            for i, ci in enumerate(claims):
                for j, cj in enumerate(claims):
                    if i != j and self._finer(kind, cj[0].obj, ci[0].obj):
                        sup[i] |= cj[1]; supk[i] |= set(cj[2])
            wit = [self.witnesses(sup[i], supk[i], claims[i][0]) for i in range(len(claims))]
            att = [i for i in range(len(claims)) if wit[i] >= self.min_sites]
            if not att:
                out.extend((c[0], c[1], c[2], wit[i]) for i, c in enumerate(claims))   # nothing attested: every value, held
                continue
            # 3. one chain -> the finest; otherwise the split, for the walk to refuse
            chain = all(self._finer(kind, claims[x][0].obj, claims[y][0].obj) or self._finer(kind, claims[y][0].obj, claims[x][0].obj)
                        for x in att for y in att if x != y)
            if chain:
                i = next(i for i in att if not any(j != i and self._finer(kind, claims[j][0].obj, claims[i][0].obj) for j in att))
                others = [c[0].obj for j, c in enumerate(claims) if j != i and not self._finer(kind, claims[i][0].obj, c[0].obj)
                          and not self._finer(kind, c[0].obj, claims[i][0].obj)]
                self._dissent[_key(claims[i][0])] = others
                if others and self.dissent_blocks:
                    out.extend((c[0], sup[j], c[2], wit[j]) for j, c in enumerate(claims))   # a veto: the walk meets them all and refuses
                else:
                    out.append((claims[i][0], sup[i], claims[i][2], wit[i]))
            else:
                out.extend((claims[i][0], sup[i], claims[i][2], wit[i]) for i in att)
        return out

    def _wikipedia_lead(self, entity: str) -> dict | None:
        """The live English Wikipedia lead for the entity (the REST summary, redirects followed), or None for a
        missing page or a disambiguation page -- then the family's mirrors are witnesses as before."""
        title = urllib.parse.quote(entity.strip().replace(" ", "_"), safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

        def make():
            self.calls["wikipedia"] += 1
            got = self.http(url, {"Accept": "application/json"})
            if got is None:
                return {}
            try:
                d = json.loads(got[1].decode("utf-8", "replace"))
            except ValueError:
                return {}
            if d.get("type") != "standard" or not d.get("extract"):
                return {}
            return {"extract": d["extract"], "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page") or url}
        got = self._cached("wplead", url, make)
        return got or None
