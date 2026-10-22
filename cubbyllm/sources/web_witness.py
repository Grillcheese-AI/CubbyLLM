"""cubbyllm/sources/web_witness.py -- the web source's witnesses: independent sites stating the same value, and the
queries that find them. Split out of web.py (order 0031) as a mixin of WebSource, verbatim.

Wired: WIRED (WebSource).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import Triple, normalize
from .web_frames import QUERY_WORDS, _fold

__wiring__ = "WIRED"


class WebWitness:
    """WebSource's witnesses and queries (a mixin; order 0031)."""

    def _same_head(self, claims: list) -> list:
        by_head: dict[str, list[int]] = {}
        for i, (t, _s, _m) in enumerate(claims):
            by_head.setdefault(_fold(t.obj.split(",")[0]), []).append(i)
        merged, gone = [], set()
        for head, idx in by_head.items():
            if len(idx) < 2:
                continue
            quals = {_fold(q) for i in idx for q in claims[i][0].obj.split(",")[1:] if _fold(q)}
            if quals and any(quals <= anc for anc in self.items_up(claims[idx[0]][0].obj.split(",")[0])):
                t0 = claims[idx[0]][0]
                merged.append((Triple(obj=t0.obj.split(",")[0].strip(), rel=t0.rel, subj=t0.subj),
                               set().union(*(claims[i][1] for i in idx)), sorted({k for i in idx for k in claims[i][2]})))
                gone.update(idx)
        return [c for i, c in enumerate(claims) if i not in gone] + merged

    def witnesses(self, sites: set[str], keys: set[str], t: Triple, n: int = 5) -> int:
        """How many INDEPENDENT witnesses `sites` are: sites whose evidence sentences share an n-word run
        (the fact's own words aside) copied one text, and count once. Live, 2026-09-24: four sites gave Agha
        Hashar Kashmiri's death as 28 April 1935 in the same words -- an old Wikipedia lead, since corrected."""
        fact_words = set(normalize(t.subj).split()) | set(normalize(t.obj).split()) | set(normalize(t.rel).split())
        fact_words |= {w for k in keys for e in self.tally.get(k, {}).values() for w in normalize(e.get("obj", "")).split()}
        shingles: dict[str, set[tuple]] = {}
        for site in sites:
            sh = set()
            for k in keys:
                sent = (self.tally.get(k, {}).get(site) or {}).get("sentence") or ""
                words = [w for w in normalize(sent).split() if w not in fact_words and not w.isdigit()]
                sh |= {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
            shingles[site] = sh
        parent = {x: x for x in sites}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ss = sorted(sites)
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if shingles[a] & shingles[b]:
                    parent[find(a)] = find(b)
        return len({find(x) for x in sites})

    # -- the facts -----------------------------------------------------------------------------------
    def queries(self, entity: str, rels: list[str]) -> list[str]:
        """The entity, quoted, with the relation's label; then (with `second_query`) the way a person would
        type it -- 'Marie Curie born'. Two phrasings bring more sites, which is what a second witness needs."""
        qs = [f'"{entity}" {rels[0]}' if rels else f'"{entity}"']
        if rels and self.second_query:
            w = QUERY_WORDS.get(normalize(rels[0]))
            if w is None:
                w = next((x for x in self.aliases.wordings(rels[0]) if x.islower() and len(x) > 3
                          and normalize(x) != normalize(rels[0])), None)
            if w:
                qs.append(f"{entity} {w}")
        return qs
