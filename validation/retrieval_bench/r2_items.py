"""r2_items -- order 0046, R2's inputs: each mention's candidates (R1's, unordered) and each candidate's profile.

Candidates: the union of Wikidata's search shortlist (its first 20 hits, in the row's language) and the local name
index's arms (exact, normalized, then the trigram arm's groups of equal score, each group whole, while the set holds at
most 20 more) -- a set, never ranked (Cowork's check: the two sources are complementary, 140 items found only locally
and 124 only by the API in English).

A profile is what the choosing arms may read of an item, fetched through the source's cache (live Wikidata by id):
  labels / aliases   EN, FR and mul names (the "name as written" tier)
  kinds              P31, and one ``subclass of`` step above each (as ``sources.wikidata_namesakes`` reads kinds)
  props              the properties it carries a statement of
  values             {property: [value labels]} in EN and FR, item values by label, literals as written
  desc               its EN and FR descriptions
Profiles are kept in ``$CUBBY_DATA_ROOT/retrieval/r2_profiles.jsonl`` (one fetch per item, ever).

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json
import os

__wiring__ = "STANDALONE"

API_K = 20
LOCAL_K = 20


def candidates(src, ix, text: str, lang: str) -> dict:
    """{"api": [...], "local": [...], "all": sorted set} for one mention."""
    from retrieval_bench.r1_finding import by_score
    from retrieval_sets.common import get
    d = get(src, {"action": "wbsearchentities", "search": text, "language": lang, "uselang": lang, "type": "item",
                  "limit": API_K}) or {}
    api = [h.get("id") for h in d.get("search") or [] if h.get("id")]
    local = sorted(ix.exact(text) | ix.normalized(text))
    for g in by_score(ix.near(text, 100)):
        if len(set(local) | set(g)) - len(ix.exact(text) | ix.normalized(text)) > LOCAL_K:
            break
        local = sorted(set(local) | set(g))
    return {"api": api, "local": local, "all": sorted(set(api) | set(local))}


def _ids(claims: dict, pid: str) -> list[str]:
    return [((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
            for st in claims.get(pid, []) if ((st.get("mainsnak") or {}).get("datavalue") or {}).get("type") == "wikibase-entityid"]


def _literal(dv: dict) -> str | None:
    v, t = dv.get("value"), dv.get("type")
    if t == "time":
        return str(v.get("time", ""))[1:5]
    if t == "quantity":
        return str(v.get("amount", "")).lstrip("+")
    if t == "monolingualtext":
        return v.get("text")
    return v if isinstance(v, str) else None


class Profiles:
    """Item profiles, fetched once and kept on disk."""

    def __init__(self, src, path: str | None = None):
        from retrieval_sets.common import data_root
        self.src = src
        self.path = path or os.path.join(data_root(), "retrieval", "r2_profiles.jsonl")
        self.p: dict[str, dict] = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.p[r["id"]] = r

    def get(self, qids) -> dict[str, dict]:
        want = [q for q in dict.fromkeys(qids) if q and q not in self.p]
        for i in range(0, len(want), 50):
            self._fetch(want[i:i + 50])
        return {q: self.p[q] for q in qids if q in self.p}

    def _fetch(self, qids: list[str]) -> None:
        from retrieval_sets.common import entities
        ents = entities(self.src, qids, props="labels|aliases|descriptions|claims", langs="en|fr|mul")
        targets, kinds = set(), {}
        for q, e in ents.items():
            cl = e.get("claims") or {}
            kinds[q] = _ids(cl, "P31")
            targets |= {v for pid in cl for v in _ids(cl, pid)}
        up = entities(self.src, sorted({k for ks in kinds.values() for k in ks}), props="claims", langs="en")
        names = entities(self.src, sorted(targets), props="labels", langs="en|fr")
        lab = lambda q, lg: (((names.get(q) or {}).get("labels") or {}).get(lg) or {}).get("value")   # noqa: E731
        rows = []
        for q in qids:
            e = ents.get(q) or {}
            cl = e.get("claims") or {}
            ks = set(kinds.get(q, []))
            ks |= {u for k in kinds.get(q, []) for u in _ids((up.get(k) or {}).get("claims") or {}, "P279")}
            values = {}
            for pid, sts in cl.items():
                vs = []
                for st in sts:
                    dv = (st.get("mainsnak") or {}).get("datavalue") or {}
                    if dv.get("type") == "wikibase-entityid":
                        vs += [x for x in (lab(dv["value"]["id"], "en"), lab(dv["value"]["id"], "fr")) if x]
                    else:
                        x = _literal(dv)
                        if x:
                            vs.append(x)
                values[pid] = vs
            r = {"id": q, "labels": {lg: (e.get("labels") or {}).get(lg, {}).get("value") for lg in ("en", "fr", "mul")},
                 "aliases": {lg: [a.get("value") for a in (e.get("aliases") or {}).get(lg, [])] for lg in ("en", "fr", "mul")},
                 "desc": {lg: (e.get("descriptions") or {}).get(lg, {}).get("value") for lg in ("en", "fr")},
                 "kinds": sorted(ks), "props": sorted(cl), "values": values,
                 "value_ids": {pid: _ids(cl, pid) for pid in cl}}
            self.p[q] = r
            rows.append(r)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = ["Profiles", "candidates"]
