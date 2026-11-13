"""r2_misses_report -- order 0069, A's table: B1q's misses by cause, set and language, from ``r2_misses``' repeated
decisions (each checked against R2's logs first). Ids and short codes only (the sets' licences): no question text.

  1. wrong picks: whether the gold bears the name among the candidates; how the pick stood (its own statement, a
     statement pointing at it, the "located in" walk, or the relation alone with no qualifier); whether the cue that
     held names one item or several
  2. rows the tiers decide and B1q does not (test, Mintaka and WebQSP_EL): no candidate bears the name; no qualifier
     (no span of the question names an item; a span that would name one was taken by the comparison rule; the short-
     name rule left none); the qualifier held by none; held by several; the relation not verified
  3. seed 1's qualified rows left unresolved (undecided or refused), by the same causes, English and French apart, and
     how many of them carry a demonym ("French", "allemand"): a country's demonym (Wikidata P1549) that names no item

    python validation/retrieval_bench/r2_misses_report.py
-> validation/logs/retrieval_R2_misses_<date>.json

Wired: STANDALONE (order 0069's bench).
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"


def wrong_cause(r: dict, got: dict) -> dict:
    pick = got.get("item")
    held = (got.get("cands") or {}).get(pick, {}).get("holds")
    gold = r.get("gold")
    return {"id": r["id"], "set": r["set"], "lang": r["lang"], "gold": gold, "pick": pick,
            "gold_in_candidates": gold in (r.get("cands") or ()), "gold_bears_name": gold in (r.get("named") or ()),
            "stood_by": (held or {}).get("by") or ("relation alone" if got.get("qualifier") == "none" else "?"),
            "hops": (held or {}).get("hops"), "cue_items": (held or {}).get("cue_items"),
            "cue_names_several": len((held or {}).get("cue_items") or ()) > 1,
            "gold_holds": bool(((got.get("cands") or {}).get(gold) or {}).get("holds"))}


def undecided_cause(r: dict, got: dict, extra_names: set) -> str:
    """Why B1q decided nothing on this row (one code)."""
    if got.get("why") == "no candidate bears the name":
        return "no candidate bears the name"
    if got.get("qualifier") == "none" or got.get("why") == "no relation and no qualifier":
        if any(c in (r.get("_items") or {}) for c in r.get("cues") or ()):
            return "no qualifier: the short-name rule left none"
        if r["id"] in extra_names:
            return "no qualifier: the comparison rule took the name"
        return "no qualifier: no span names an item" if r.get("cues") else "no qualifier: no cue span"
    stand = [q for q, c in (got.get("cands") or {}).items() if c.get("holds")]
    if not stand:
        return "the qualifier held by none"
    if len(stand) > 1:
        return "the qualifier held by several"
    return "the relation not verified"


def compared_names(rows: list[dict]) -> set:
    """Rows where turning the comparison rule off would give a span that names an item (looked up exactly)."""
    from retrieval_bench import r2_b1q as Q
    from retrieval_sets.lexicon import check
    from retrieval_sets.names_segment import exact_items
    lx, keep, extra = check(), Q.compared, {}
    try:
        Q.compared = lambda *a, **k: None
        for r in rows:
            names = [r["mention"]]
            more = set(Q.cue_runs(r["question"], names, r["pids"] or r.get("pids_given") or [], lx,
                                  r.get("relation_words") or ())) - set(r.get("cues") or ())
            if more:
                extra[r["id"]] = sorted(more)
    finally:
        Q.compared = keep
    found = exact_items(sorted({c for v in extra.values() for c in v})) if extra else {}
    return {i for i, cs in extra.items() if any(found.get(c) for c in cs)}


def demonyms(src, rows: list[dict], profiles_values: dict) -> set[str]:
    """The English and French demonyms (P1549) of the countries the rows' candidates name (P17, P27), folded."""
    from retrieval_bench.r2_arms import fold
    from retrieval_sets.common import entities
    countries = sorted({c for r in rows for q in r.get("cands") or () for p in ("P17", "P27")
                        for c in (profiles_values.get(q) or {}).get(p) or ()})
    out = set()
    ents = entities(src, countries, props="claims", langs="en|fr") if countries else {}
    for e in ents.values():
        for st in (e.get("claims") or {}).get("P1549") or ():
            v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
            if v.get("language") in ("en", "fr") and v.get("text"):
                out.add(fold(v["text"]))
    return out


def held_by_none(r: dict, got: dict) -> str:
    """A.2's finer split of "the qualifier held by none": whether the gold bears the name and what the cues name."""
    from retrieval_bench.r2_b1q import longest
    items_of = r.get("cue_items") or {}
    checked = longest([x for x in r.get("cues") or () if items_of.get(x)])
    return ("gold bears the name" if r["gold"] in (r.get("named") or ()) else "gold does not bear the name") + "; " +         ("a cue names the gold itself" if any(r["gold"] in items_of[x] for x in checked) else "the cues name other items")


def qualifier_named(r: dict, got: dict, want: str | None, items: dict, names: set) -> str:
    """A.3's finer split: whether a cue names the qualifier's own item, and if not, why (the item's names in the row's
    language, folded, against the cue spans)."""
    from retrieval_bench.r2_arms import STOP
    held = [q for q, x in (got.get("cands") or {}).items() if x.get("holds")]
    kind = "held by several" if len(held) > 1 else "held by none" if not held else "held by one"
    if any(want in (items.get(x) or []) for x in r.get("cues") or ()):
        return f"{kind}; a cue names the qualifier"
    cues = set(r.get("cues") or ())
    if names & cues:
        return f"{kind}; the qualifier's name is a cue span, the name index lacks it"
    if any(n.split()[0] in STOP and " ".join(n.split()[1:]) in cues for n in names if len(n.split()) > 1):
        return f"{kind}; the qualifier's name starts with an article the cue rule strips"
    return f"{kind}; the qualifier's name is not a span of the question"


def table(items: list, key) -> dict:
    out = collections.defaultdict(collections.Counter)
    for x in items:
        out[f"{x['set']}/{x['lang']}"][key(x)] += 1
        out["all"][key(x)] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def main() -> None:
    from retrieval_bench import r2_misses as M
    from retrieval_sets.common import source
    rep = {"what": "order 0069 A: B1q's misses by cause (a diagnostic: nothing is scored anew)",
           "date": _dt.date.today().isoformat(), "logs": M.LOGS, "check": {}}
    sets = {}
    for name in ("dev", "test", "seed1"):
        d = M.load(name)
        c = M.check(name, d)
        rep["check"][name] = {"agree": len(c["agree"]), "differ": len(c["differ"]), "differ_first": c["differ"][:10]}
        for r in d["rows"]:
            r["_items"] = d["items"]
        sets[name] = (d, c["agree"])
    wrong = []
    for name in ("test", "dev"):
        d, agree = sets[name]
        wrong += [{**wrong_cause(r, agree[r["id"]]), "split": name} for r in d["rows"] if (agree.get(r["id"]) or {}).get("judge") == "wrong"]
    rep["1_wrong"] = {s: table([w for w in wrong if w["split"] == s], lambda w: f"{w['stood_by']}"
                               + ("; gold does not bear the name" if not w["gold_bears_name"] else "")
                               + ("; the cue names several items" if w["cue_names_several"] else "")) for s in ("test", "dev")}
    rep["1_rows"] = wrong
    d, agree = sets["test"]
    rows2 = [r for r in d["rows"] if r["set"] in ("mintaka", "webqsp_el") and r["id"] in agree
             and agree[r["id"]]["tiers"] in ("right", "wrong") and agree[r["id"]]["judge"] not in ("right", "wrong")]
    extra = compared_names(rows2)
    rep["2_tiers_decide_b1q_not"] = table([{**r, "c": undecided_cause(r, agree[r["id"]], extra)} for r in rows2], lambda x: x["c"])
    rep["2_held_by_none"] = table([{**r, "c": held_by_none(r, agree[r["id"]])} for r in rows2
                                   if undecided_cause(r, agree[r["id"]], extra) == "the qualifier held by none"], lambda x: x["c"])
    d, agree = sets["seed1"]
    rows3 = [r for r in d["rows"] if r["kind"] == "qualified" and r["id"] in agree and agree[r["id"]]["judge"] in ("undecided", "refused")]
    extra = compared_names(rows3)
    rep["3_seed1_unresolved"] = table([{**r, "c": undecided_cause(r, agree[r["id"]], extra)} for r in rows3], lambda x: x["c"])
    src = source()
    from retrieval_bench.r2_arms import fold
    from retrieval_sets.common import entities, names_of
    disc = {}
    for line in open(os.path.join(os.path.dirname(HERE), "retrieval_sets", "fame_set_seed1.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        if x.get("form") == "qualified":
            disc[x["id"]] = (x.get("discriminator") or {}).get("value")
    ents = entities(src, sorted({disc.get(r["id"]) for r in rows3 if disc.get(r["id"])}), props="labels|aliases",
                    langs="fr|en|mul")
    rep["3_qualifier_named"] = table([{**r, "c": qualifier_named(r, agree[r["id"]], disc.get(r["id"]), d["items"],
                                                                {fold(n) for n in names_of(ents.get(disc.get(r["id"])) or {}, r["lang"])})}
                                      for r in rows3], lambda x: x["c"])
    dem = demonyms(src, rows3, {q: p.get("value_ids") or {} for q, p in d["profiles"].items()})
    rep["3_demonym_rows"] = table([r for r in rows3 if set(fold(r["question"]).split()) & dem
                                   and not any(c in d["items"] for c in fold(r["question"]).split() if c in dem)],
                                  lambda r: "a demonym that names no item")
    rep["3_demonyms_known"] = len(dem)
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R2_misses_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("1_rows",)}, indent=1)[:5000], "->", path)


if __name__ == "__main__":
    main()
