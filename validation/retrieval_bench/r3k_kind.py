"""r3k_kind -- order 0046, R3k (Cowork, 2026-11-04): the kind the question asks bounds the item's properties further.

The asked kind, from a closed list of question words per language, written in Cowork's check before any row was read:
  who / qui -> person; where / où -> place; when / quand / what year / en quelle année -> time; how many / combien ->
  quantity; anything else -> no kind. The listed word opening the question (after an optional preposition) decides.
The properties kept, when a kind is asked: the item's outgoing properties whose values on this item are of that kind.
  time, quantity   the property's datatype (the frozen lexicon's): ``time`` / ``quantity``
  person, place    a value is of the kind when its ``instance of`` reaches one of the kind's classes: the items
                   Wikidata labels with the kind's word ("person" / "personne", "place" / "lieu"), having a ``subclass
                   of`` of their own (0047's ``ask_class.classes`` reading). Two readings, both reported:
                     one    ``instance of`` or one ``subclass of`` step above it (0047's ``is_a``, as Cowork wrote)
                     chain  ``instance of`` then any number of ``subclass of`` steps (one SPARQL query a row), because
                            "place" sits several steps above "city" or "village"
No kind: the row is scored as in R3o. Incoming properties (asked from the value's side) are kept only when no kind is
asked: a kind is read off the values the item states.

The re-read (mode ``gaps``, Cowork's check of 2026-11-04 afternoon, three gaps in the rule, written before it is run):
  incoming    an incoming property stays when its subjects are of the kind ("who was born here?" asks for persons):
              some item stating it with this one as its value reaches the kind through instance of / subclass of*
              (one SPARQL query a row, FILTER EXISTS per property); unanswered: every incoming property stays
  the chain   the kind along subclass of, as ``chain``
  how many    a quantity-valued property, or a property with several values on this item (their count)

Wired: STANDALONE (R3k's bench).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

__wiring__ = "STANDALONE"

WORDS = [  # (pattern, kind): the closed list; the question word opens the question, after an optional preposition, so a
    # relative "qui" ("le film qui...") or the conjunction "ou" is never read as one
    (r"what year|quelle ann[ée]e", "time"), (r"how many|how much|combien", "quantity"),
    (r"who|whom|qui", "person"), (r"where|o[ù]", "place"), (r"when|quand", "time")]
LEAD = r"^\s*(?:in|at|from|to|on|by|for|of|en|dans|de|à|a|par|pour|sur|depuis)?\s*(?:"
KIND_WORDS = {"person": ("person", "personne"), "place": ("place", "lieu")}
DATATYPE = {"time": "time", "quantity": "quantity"}


def asked_kind(question: str) -> str | None:
    q = (question or "").lower()
    for rx, kind in WORDS:
        if re.match(LEAD + rx + r")(?!\w)", q):
            return kind
    return None


def kind_classes(src, kind: str) -> set[str]:
    from cubbyllm.reasoning.ask_class import classes
    return {c for word, lang in zip(KIND_WORDS[kind], ("en", "fr")) for c in classes(src, word, lang)}


def one_step(src, qid: str, cls: set) -> bool:
    from cubbyllm.reasoning.ask_class import is_a
    return is_a(src, qid, cls)


def chain(values: list[str], cls: set) -> set[str] | None:
    """The values whose instance of reaches a class through any number of subclass of steps; None if unanswered."""
    from cubbyllm.sources.wikidata import UA
    if not values or not cls:
        return set()
    q = ("SELECT DISTINCT ?v WHERE { VALUES ?v { " + " ".join(f"wd:{v}" for v in values) + " } VALUES ?c { "
         + " ".join(f"wd:{c}" for c in cls) + " } ?v wdt:P31/wdt:P279* ?c . }")
    for k in range(3):
        try:
            req = urllib.request.Request("https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": q}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.2)
            return {b["v"]["value"].rsplit("/", 1)[1] for b in rows}
        except Exception:                                  # noqa: BLE001 -- recorded as unknown, never as "none"
            time.sleep(5 * (k + 1))
    return None


def kept(kind: str | None, out_props: set, in_props: set, claims: dict, lx: dict, of_kind,
         in_of_kind: set | None = None, counts: bool = False) -> set:
    """The candidate properties left once the kind bounds them. `claims`: the item's {property: [value items]};
    `of_kind`: a value item -> bool for person / place. The re-read: `in_of_kind`, the incoming properties whose
    subjects are of the kind (kept); `counts`, "how many" also keeps a property with several values here."""
    if kind is None:
        return out_props | in_props
    if kind in DATATYPE:
        out = {p for p in out_props if lx["datatype"].get(p) == DATATYPE[kind]}
        if counts and kind == "quantity":
            out |= {p for p in out_props if len(set(claims.get(p) or ())) >= 2}
        return out
    return {p for p in out_props if any(of_kind(v) for v in claims.get(p) or ())} | (in_of_kind or set())


def incoming_of_kind(item: str, props: list[str], cls: set) -> set[str] | None:
    """The incoming properties of `item` some subject of which reaches a class of `cls`; None if unanswered."""
    from cubbyllm.sources.wikidata import UA
    if not props or not cls:
        return set()
    q = ("SELECT ?p WHERE { VALUES ?p { " + " ".join(f"wdt:{p}" for p in props) + " } FILTER EXISTS { ?s ?p wd:" + item
         + " . ?s wdt:P31/wdt:P279* ?c . VALUES ?c { " + " ".join(f"wd:{c}" for c in cls) + " } } }")
    for k in range(3):
        try:
            req = urllib.request.Request("https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": q}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.2)
            return {b["p"]["value"].rsplit("/", 1)[1] for b in rows}
        except Exception:                                  # noqa: BLE001 -- recorded as unknown, never as "none"
            time.sleep(5 * (k + 1))
    return None


__all__ = ["asked_kind", "chain", "incoming_of_kind", "kept", "kind_classes", "one_step"]


def bound(rows: list[dict], items: dict, props: dict, src, lx: dict, mode: str) -> tuple[dict, dict]:
    """({row id: kept properties} for rows that ask a kind, info): R3k's candidates, `mode` "one" or "chain"."""
    from retrieval_bench.r3_gold import ClaimIds
    cls = {k: kind_classes(src, k) for k in KIND_WORDS}
    claims = ClaimIds(src).get(sorted({q for r in rows for q in items[r["id"]]}))
    memo: dict[str, bool] = {}
    out, unanswered, in_unanswered = {}, 0, 0
    for r in rows:
        kind = r["kind_asked"] = asked_kind(r["question"])
        if kind is None:
            continue
        its = items[r["id"]]
        out_p = set().union(*[set(props.get(q, {}).get("out") or ()) for q in its]) if its else set()
        in_p = set().union(*[set(props.get(q, {}).get("in") or ()) for q in its]) if its else set()
        cl: dict[str, list] = {}
        for q in its:
            for p, vs in (claims.get(q) or {}).items():
                cl.setdefault(p, []).extend(vs)
        of_kind = lambda v: False                                                      # noqa: E731
        in_k: set | None = None
        if kind in KIND_WORDS and mode == "gaps" and in_p:
            in_k = set()
            for q in its:
                got = incoming_of_kind(q, sorted(set(props.get(q, {}).get("in") or ()), key=lambda p: int(p[1:])), cls[kind])
                if got is None:                         # the service could not say: every incoming property stays
                    in_unanswered += 1
                    got = set(props.get(q, {}).get("in") or ())
                in_k |= got
        if kind in KIND_WORDS:
            vals = sorted({v for p in out_p for v in cl.get(p) or ()})
            if mode in ("chain", "gaps"):
                ok = chain(vals, cls[kind])
                if ok is None:                          # the service could not say: the kind bounds nothing here
                    unanswered += 1
                    out[r["id"]] = out_p | in_p
                    continue
                of_kind = lambda v, ok=ok: v in ok                                     # noqa: E731
            else:
                def of_kind(v, kind=kind):
                    if v not in memo:
                        memo[v] = one_step(src, v, cls[kind])
                    return memo[v]
        out[r["id"]] = kept(kind, out_p, in_p, cl, lx, of_kind, in_k, counts=mode == "gaps")
    return out, {"classes": {k: sorted(v) for k, v in cls.items()}, "chain_unanswered": unanswered,
                 "incoming_unanswered": in_unanswered, "rows_with_kind": len(out)}


def pick_margin(summary: dict, taus: list[float]) -> float | None:
    """R3k's margin rule (Cowork, written before any row was read): the smallest margin at which both the pooled wrong
    rate and the missing-fact handovers are at most lexicon_content's, on the same rows and candidates."""
    arms = summary["all"]["arms"]
    base = arms.get("lexicon_content") or {}
    for t in sorted(taus):
        a = arms.get(f"qfhrr@{t}") or {}
        if a and a["all"]["wrong_rate"] <= base["all"]["wrong_rate"] and \
                a["missing_fact"]["handed_over"] <= base["missing_fact"]["handed_over"]:
            return t
    return None
