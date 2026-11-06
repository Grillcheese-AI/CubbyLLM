"""validation/fact_store/audit.py -- order 0055 step 0: the coverage audit, on live Wikidata, no download.

For each of the 1,000 questions (``audit_sets``): is the gold a value of a statement on the asked item, elsewhere in
Wikidata, or not found? The classes, first that holds:
  not_a_value   the gold is yes or no (Mintaka's yes/no questions): no statement holds it
  on_item       a statement of an asked item, at any rank, holds the gold as its value or a time qualifier's; or an
                item value's label states a text gold no item bears as its exact name (``by_label``)
  on_item_rev   the gold is an item with a statement whose value is an asked item (the reverse index's case)
  elsewhere     the gold is an item in Wikidata, not reached from the asked item
  not_found     none of the above: for a date, number or string gold, "not on the asked item" (the order's three
                classes; two hops were tried and cost about three minutes a question on the Query Service)
The asked items: Mintaka's own ids. SimpleQA and MKQA have none, so the runs of 1 to 6 words that may be a name
(``name_spans``: content words at the edges; in English, capitalised) are looked up by their exact name
(``Wd.search_exact``), longest first, a run inside a name already found skipped (longest match), and every item so
named is an asked item: no fame ordering, no top-k. Ambiguity is counted: the name that reached the gold, and how
many items it names. A SimpleQA gold is also looked up by its exact name, so an item gold is compared by Q-id. Mintaka's asked items are looked for by name in its
English and its French question (step 1's name recall, measured on them).

Rows go off the repo (``$CUBBY_DATA_ROOT/retrieval/audit_0055/rows.jsonl``: MKQA's stay there, 0048); resumable.

    python validation/fact_store/audit.py [--n 20] [--offline]

Wired: STANDALONE (order 0055's audit).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

MAX_ASKED, MAX_SEARCH = 60, 40
TIME_QUALIFIERS = {"P580", "P582", "P585"}                # start, end, point in time: kept hot (the synthesis)
KIND_OR_PLACE = {"P31", "P279", "P17", "P131", "P27", "P495", "P641", "P407", "P364"}
CONNECTORS = {"of", "the", "de", "du", "des", "la", "le", "and", "for", "von", "van", "der", "da", "di", "del", "y", "s"}
_LEAD = re.compile(r"^(in|at|the|a|an|on|by)\s+", re.I)


def gold_names(text: str) -> list[str]:
    """A text gold's forms looked up by exact name: as given, without a lead word, before a parenthesis."""
    t = text.strip().rstrip(".")
    out = [t, _LEAD.sub("", t), t.split(" (")[0]]
    return [x for x in dict.fromkeys(o.strip() for o in out) if x]


def name_spans(question: str, lang: str) -> list[tuple[int, int, str]]:
    """The runs that may be a name, longest first: each edge word a content word, or, in English, a capitalised word
    past the first (so "The Book" mid-question, never "Who received")."""
    from emitter_wd_mkqa import WORD, spans_of
    from retrieval_bench.r2_arms import STOP
    first = next(WORD.finditer(question), None)
    first = first.start() if first else -1

    def edge(w: str, at: int) -> bool:
        if lang == "en":
            return (w[:1].isupper() or w[:1].isdigit()) and (w.lower() not in STOP or at != first)
        return w.lower() not in STOP

    out = []
    for s, e, text in spans_of(question):
        ws = list(WORD.finditer(text))
        inner_ok = lang != "en" or all(w.group(0)[:1].isupper() or w.group(0)[:1].isdigit() or
                                         w.group(0).lower() in CONNECTORS for w in ws[1:-1])
        if ws and inner_ok and edge(ws[0].group(0), s) and edge(ws[-1].group(0), s + ws[-1].start()):
            out.append((s, e, text))
    return sorted(out, key=lambda x: (-(x[1] - x[0]), x[0]))


def asked_by_name(wd, question: str, lang: str) -> dict:
    """{asked item: (span, how many items that span names)}: the longest names first, a name inside one already found
    not looked up (longest match), every item a name names kept, at most MAX_ASKED; plus whether a search was cut
    (``truncated``) or the asked set capped."""
    asked, truncated, capped, taken, searched = {}, False, False, [], 0
    for s, e, text in name_spans(question, lang):
        if any(s < b and a < e for a, b in taken):
            continue
        if searched >= MAX_SEARCH:
            capped = True
            break
        searched += 1
        ids, cut = wd.search_exact(text, lang)
        truncated |= cut
        if ids:
            taken.append((s, e))
        for q in ids:
            if q in asked:
                continue
            if len(asked) >= MAX_ASKED:
                capped = True
                break
            asked[q] = (text, len(ids))
    return {"asked": asked, "truncated": truncated, "capped": capped}


def hits_on(ents: dict, items, golds: list[dict]) -> tuple[list[dict], int]:
    """Statements of ``items`` holding a gold: [{item, pid, where, rank}] (exact), and how many held it coarser."""
    from fact_store.audit_values import match, snak_values
    hits, coarser = [], 0
    for q in items:
        for pid, sts in ((ents.get(q) or {}).get("claims") or {}).items():
            for st in sts:
                for where, dv in snak_values(st):
                    if where != "main" and where[5:] not in TIME_QUALIFIERS:
                        continue
                    for g in golds:
                        m = match(dv, g)
                        if m == "exact":
                            hits.append({"item": q, "pid": pid, "where": where, "rank": st.get("rank")})
                        elif m == "coarser":
                            coarser += 1
    return hits, coarser


def pointing_at(ents: dict, asked) -> list[dict]:
    """Statements of the gold items whose value is an asked item: [{item, pid, asked}]. Kind and place links (instance
    of, subclass of, country, located in, citizenship, origin, sport, language) are left out: a gold that is merely of
    the asked kind, or in the asked place, is not a fact about the asked item."""
    out = []
    for g, e in ents.items():
        for pid, sts in (e.get("claims") or {}).items():
            if pid in KIND_OR_PLACE:
                continue
            for st in sts:
                v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(v, dict) and v.get("id") in asked:
                    out.append({"item": g, "pid": pid, "asked": v["id"]})
    return out


def by_label(wd, ents: dict, items, golds: list[dict], cap: int = 500) -> list[dict]:
    """Item values of ``items`` whose label or alias (English, French, ``mul``) states a text gold: for a gold that no
    item bears as its exact name. At most ``cap`` values are named."""
    from fact_store.audit_values import MONTHS, fold, match
    texts = [g for g in golds if g.get("text") and not g.get("qids") and                 # a date or a number is
             any(w not in MONTHS and not w.isdigit() for w in fold(g["text"]).split())]  # never an item's name
    vals = {}
    for q in items:
        for pid, sts in ((ents.get(q) or {}).get("claims") or {}).items():
            for st in sts:
                v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(v, dict) and v.get("id", "").startswith("Q"):
                    vals.setdefault(v["id"], []).append((q, pid, st.get("rank")))
    if not texts or not vals:
        return []
    names = wd.entities(list(vals)[:cap], props="labels|aliases")
    hits = []
    for vid, e in names.items():
        ns = [l.get("value") for l in (e.get("labels") or {}).values()]
        ns += [a.get("value") for al in (e.get("aliases") or {}).values() for a in al]
        if any(match({"type": "string", "value": n}, g) == "exact" for n in ns if n for g in texts):
            hits += [{"item": q, "pid": pid, "where": "main", "rank": r, "value": vid} for q, pid, r in vals.get(vid, [])]
    return hits


def names_found(ent: dict, question: str, lang: str) -> bool:
    from fact_store.audit_values import fold
    fq = f" {fold(question)} "
    names = []
    for lg in (lang, "mul"):
        names += [((ent.get("labels") or {}).get(lg) or {}).get("value")]
        names += [a.get("value") for a in (ent.get("aliases") or {}).get(lg) or []]
    return any(n and fold(n) and f" {fold(n)} " in fq for n in names)


def classify(wd, row: dict) -> dict:
    out = {"id": row["id"], "set": row["set"], "extra": row["extra"]}
    if row["set"] == "mintaka" and (row["extra"].get("complexity") == "yesno" or
                                    all(g["text"].lower() in ("true", "false", "yes", "no") and not g["qids"]
                                        for g in row["golds"])):
        return {**out, "class": "not_a_value"}
    golds = [dict(g) for g in row["golds"]]
    if row["asked"]:
        asked = {q: (None, 1) for q in row["asked"]}
        meta = {"truncated": False, "capped": False}
        names = wd.entities(row["asked"], props="labels|aliases")
        out["name_en"] = any(names_found(names.get(q) or {}, row["question"], "en") for q in row["asked"])
        if row.get("question_fr"):
            out["name_fr"] = any(names_found(names.get(q) or {}, row["question_fr"], "fr") for q in row["asked"])
    else:
        meta = asked_by_name(wd, row["question"], row["lang"])
        asked = meta["asked"]
    if row["set"] == "simpleqa":
        for g in golds:
            g["qids"] = list(dict.fromkeys(q for n in gold_names(g["text"]) for q in wd.search_exact(n, "en")[0]))
    gold_qids = [q for g in golds for q in g["qids"]]
    out.update({"asked_n": len(asked), "truncated": meta["truncated"], "capped": meta["capped"],
                "gold_items": len(gold_qids)})
    ents = wd.entities(list(asked))
    hits, coarser = hits_on(ents, list(asked), golds)
    out["coarser"] = coarser
    if not hits:
        hits = by_label(wd, ents, list(asked), golds)
        out["by_label"] = bool(hits)
    if hits:
        spans = {asked[h["item"]][0] for h in hits}
        subject = next(iter(asked.values()))[0] if asked else None
        out.update({"class": "on_item", "hits": hits[:5], "rank": sorted({h["rank"] for h in hits}),
                    "qualifier_only": all(h["where"] != "main" for h in hits),
                    "ambiguous": any(asked[h["item"]][1] > 1 for h in hits), "names_hit": len(spans),
                    "items_hit": len({h["item"] for h in hits}), "on_longest_name": subject in spans})
        return out
    rev = pointing_at(wd.entities(gold_qids), asked)
    if rev:
        subject = next(iter(asked.values()))[0]
        return {**out, "class": "on_item_rev", "hits": rev[:5], "ambiguous": any(asked[h["asked"]][1] > 1 for h in rev),
                "on_longest_name": any(asked[h["asked"]][0] == subject for h in rev)}
    return {**out, "class": "elsewhere" if gold_qids else "not_found"}


def main(argv=None) -> None:
    from fact_store.audit_sets import questions
    from fact_store.audit_wd import Wd
    from retrieval_sets.common import data_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="the first n of each set (a smoke)")
    ap.add_argument("--offline", action="store_true", help="the audit's cache only")
    ap.add_argument("--out", default=os.path.join(data_root(), "audit_0055"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    counts: dict = {}
    rows = questions(counts)
    if a.n:
        rows = [r for s in ("simpleqa", "mintaka", "mkqa") for r in [x for x in rows if x["set"] == s][:a.n]]
    path = os.path.join(a.out, "rows.jsonl")
    done = {json.loads(x)["id"] for x in open(path, encoding="utf-8") if x.strip()} if os.path.exists(path) else set()
    wd, t0 = Wd(offline=a.offline), time.time()
    todo = [r for r in rows if r["id"] not in done]
    print(f"audit: {len(rows)} questions, {len(done)} done, {len(todo)} to go; mkqa left out {counts}", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            t = time.time()
            rec = classify(wd, r)
            rec["s"] = round(time.time() - t, 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s; api {wd.src.calls})",
                      flush=True)
    json.dump({"mkqa_kept_left_out": counts, "api_calls": wd.src.calls, "wall_s": round(time.time() - t0, 1)},
              open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
