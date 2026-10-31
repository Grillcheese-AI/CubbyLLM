"""r2_arms -- order 0046, R2's choosing arms: given a mention's candidates and their profiles (``r2_items``), which item
does the question mean? Pure functions, no network.

Each arm returns an outcome:
  {"outcome": "pick", "item": q}            one item decided
  {"outcome": "several", "items": [...]}    namesakes left, nothing decided (asked back, or answered each by B1)
  {"outcome": "refuse", "why": ...}         no candidate bears the name, or none carries the relation

Every arm starts from the same two steps (``docs/RETRIEVAL.md``, "Choose"): the candidates that BEAR THE NAME (a label or
alias, EN / FR / mul, equal to the mention once case, accents and punctuation are folded); then, when the question's
relation is known, a kind that cannot carry it drops out (here: the item states no value of it). Within what is left:
  tiers   today's last tier: the item whose LABEL is the name over items carrying it only as an alias;
  bm25    the question's own words (the mention's and stopwords taken out) against each item's description and values,
          BM25 over the remaining candidates; the best is picked only when its rank-normalized margin over the next,
          (s1 - s2) / s1, is at least ``tau`` (set on dev, never a raw score threshold).
Never a rank by fame, sitelinks, links or order.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import math
import re

__wiring__ = "STANDALONE"

STOP = set("""a an the of in on at to for from by with and or is are was were be been who what when where which why how
does do did has have had this that these those it its into about as his her their whose whom
le la les l un une des du de d en au aux à et ou est sont était qui que quoi quand où quel quelle quels quelles comment
dans sur par pour avec son sa ses leur leurs se ce cette ces il elle ils elles a ont""".split())


def fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def names(p: dict) -> tuple[set, set]:
    """(its labels, its aliases), folded, over EN, FR and mul."""
    labs = {fold(v) for v in (p.get("labels") or {}).values() if v}
    als = {fold(a) for v in (p.get("aliases") or {}).values() for a in v or () if a}
    return labs, als - labs


def bearing(mention: str, cands, profiles: dict) -> list[str]:
    m = fold(mention)
    return [q for q in cands if q in profiles and m in set().union(*names(profiles[q]))]


def carrying(cands, profiles: dict, pids) -> list[str]:
    return [q for q in cands if any(profiles[q].get("values", {}).get(p) for p in pids)] if pids else list(cands)


def _start(mention, cands, profiles, pids):
    named = bearing(mention, cands, profiles)
    if not named:
        return None, {"outcome": "refuse", "why": "no candidate bears the name"}
    left = carrying(named, profiles, pids)
    if not left:
        return None, {"outcome": "refuse", "why": "none carries the relation", "items": named}
    if len(left) == 1:
        return None, {"outcome": "pick", "item": left[0], "by": "the only one" if len(named) == 1 else "carries the relation"}
    return left, None


def tiers(mention: str, cands, profiles: dict, pids=()) -> dict:
    left, done = _start(mention, cands, profiles, pids)
    if done:
        return done
    m = fold(mention)
    by_label = [q for q in left if m in names(profiles[q])[0]]
    if len(by_label) == 1:
        return {"outcome": "pick", "item": by_label[0], "by": "the label over an alias"}
    return {"outcome": "several", "items": sorted(left)}


def doc(p: dict, lang: str) -> list[str]:
    parts = [p.get("desc", {}).get(lang) or "", p.get("desc", {}).get("en") or ""]
    parts += [v for vs in (p.get("values") or {}).values() for v in vs]
    return [w for w in fold(" ".join(parts)).split() if w not in STOP]


def cues(question: str, mention: str) -> list[str]:
    q = fold(question)
    m = fold(mention)
    if m:
        q = (" " + q + " ").replace(" " + m + " ", " ")
    return [w for w in q.split() if w not in STOP and (not w.isdigit() or len(w) == 4)]      # a year is a cue


def bm25_scores(words: list[str], docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75) -> dict[str, float]:
    n = len(docs)
    avg = sum(len(d) for d in docs.values()) / max(n, 1) or 1.0
    df = {w: sum(1 for d in docs.values() if w in d) for w in set(words)}
    out = {}
    for q, d in docs.items():
        s = 0.0
        for w in set(words):
            tf = d.count(w)
            if not tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        out[q] = s
    return out


def bm25(mention: str, question: str, lang: str, cands, profiles: dict, pids=(), tau: float = 0.5) -> dict:
    left, done = _start(mention, cands, profiles, pids)
    if done:
        return done
    sc = bm25_scores(cues(question, mention), {q: doc(profiles[q], lang) for q in left})
    ranked = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))
    s1, s2 = ranked[0][1], ranked[1][1]
    margin = (s1 - s2) / s1 if s1 > 0 else 0.0
    if s1 > 0 and margin >= tau:
        return {"outcome": "pick", "item": ranked[0][0], "by": "the question's words", "margin": round(margin, 4)}
    return {"outcome": "several", "items": sorted(left), "margin": round(margin, 4)}


__all__ = ["bearing", "bm25", "bm25_scores", "carrying", "cues", "fold", "tiers"]
