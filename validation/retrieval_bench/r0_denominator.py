"""r0_denominator -- order 0046, R0: today's pipeline on every dev set, each refusal labelled by its stage and route.

Today's pipeline is the ask loop as the stack mounts it (``exp_everyday_webq.py``): the wiki world with the hdc world
behind it, the Wikidata source live, the lexicon, the everyday table and the tense check on, today's defaults for every
flag (0037's rule on, the source over the snapshot on, 0043's source rule off). The programs seat holds no model
(``--emitter none``): the host's own reader only -- stated in the log.

Each question is labelled once, by the first stage that stopped it:
  reading          no plan, or a plan that does not cover the question
  no candidate     the source found no item for the seed
  several          the seed is a shared name nothing decided (ambiguous_entity)
  wrong candidate  the source resolved the seed to an item that is not a gold item
  relation         the relation is unknown or ambiguous
  several values   the hop held several values (ambiguous_hop)
  value missing    the right item was reached and the fact was not there
  time             the question's time and the fact's disagree
  other            any other refusal, by its reason
  spoken           answered: right, wrong (against the gold answers), or unscored (the set gives no answer to compare)
Every gold item is also checked against the local stores (``stores``); the dump line (decision 4 in
``docs/RETRIEVAL.md``, the recommended one, written here before anything is counted): the full dump is worth it only if
at least 20% of the route-labelled seed misses (no candidate, several, wrong candidate) are items absent from every
local store, over a sample of at least 1,000 mentions.

    python validation/retrieval_bench/r0_denominator.py [--n 300] [--workers 3]
-> validation/logs/retrieval_R0_<date>.json (ids and counts only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import fold, read_rows, sample  # noqa: E402

__wiring__ = "STANDALONE"

SETS = ("mintaka", "qald9plus", "webqsp_el", "graphq_el", "simplequestions_wd", "amber", "cronquestions")
DUMP_LINE = {"share": 0.20, "min_mentions": 1000, "stages": ("no candidate", "several", "wrong candidate"),
             "status": "the recommended line of docs/RETRIEVAL.md decision 4, written before counting; Nick's decision is open"}
READING = {"no_plan", "plan_does_not_cover_question", "profile_empty"}
RELATION = {"unknown_relation", "ambiguous_relation"}
TIME = {"time_mismatch", "time_undated"}
WIKI_TEXT = {"links_kept": False, "redirects_kept": False,
             "evidence": "wikipedia_dedup (74,567 routed articles): title, lang, domain and plain text, no link markup; "
                         "wiki_full (14 GB, EN and FR): plain text with no titles, no links, no redirects"}
_LOOP = None


def _loop():
    """The ask loop as the stack mounts it, with the source's calls recorded per question."""
    global _LOOP
    if _LOOP is None:
        from cubbyllm.worlds import wikikg as wk
        from cubbyllm.worlds.hdc_store import HdcWorld, Stores
        from cubbyllm.reasoning.ask import AskLoop
        from cubbyllm.sources.wikidata import WikidataSource
        from cubby_serve.sources.http import wikidata as fetch
        from validation.exp_everyday_webq import NoModel
        wk.ensure_data(("triplets",))
        src = WikidataSource(offline=False, use_choices=False, fetch=fetch())
        calls: list = []
        orig = src.facts

        @functools.wraps(orig)
        def facts(entity, relations=None, via=None, qid=None, asker=None, classes=False):
            out = orig(entity, relations=relations, via=via, qid=qid, asker=asker, classes=classes)
            last = src.last or {}
            calls.append({"qid": last.get("qid"), "ambiguous": [x[0] for x in last.get("ambiguous") or []],
                          "unresolved": bool(last.get("unresolved")), "n": len(out)})
            return out
        src.facts = facts
        loop = AskLoop(NoModel(), world=Stores(wk.wiki_world(), HdcWorld()), source=src, lexicon=True, history_path=None)
        _LOOP = (loop, calls)
    return _LOOP


def label(rec: dict, calls: list, gold_items: set, gold_answers: set) -> tuple[str, str | None]:
    """(stage, detail) for one question's record."""
    reason = rec.get("reason")
    resolved = {c["qid"] for c in calls if c["qid"]}
    if rec.get("verified"):
        if not gold_answers:
            return "spoken", "unscored"
        said = fold(str(rec.get("answer") or ""))
        return "spoken", ("right" if said in gold_answers or any(said and (said in g or g in said) for g in gold_answers if len(g) > 3)
                          else "wrong")
    if reason in READING or str(reason or "").startswith("emit_error"):
        return "reading", reason
    if reason == "ambiguous_entity" or (calls and any(c["ambiguous"] for c in calls) and not resolved & gold_items):
        return "several", reason
    if reason in RELATION:
        return "relation", reason
    if reason == "ambiguous_hop":
        return "several values", reason
    if reason in TIME:
        return "time", reason
    if resolved and gold_items and not resolved & gold_items:
        return "wrong candidate", reason
    if reason == "entity_unresolved" or (calls and not resolved):
        return "no candidate", reason
    if reason == "retrieval_exhausted":
        return "value missing", reason
    return "other", reason


def run_chunk(job: tuple) -> list[dict]:
    rows, gold_answers = job
    loop, calls = _loop()
    out = []
    for r in rows:
        calls.clear()
        t = time.perf_counter()
        try:
            rec = loop.ask(r["question"])
        except Exception as e:                           # noqa: BLE001 -- a crash is a row
            rec = {"verified": False, "reason": f"error: {type(e).__name__}"}
        items = {m["qid"] for m in r["mentions"] if m.get("qid")}
        stage, detail = label(rec, calls, items, gold_answers.get(r["id"], set()))
        out.append({"id": r["id"], "set": r["set"], "lang": r["lang"], "stage": stage, "detail": detail,
                    "gold_items": sorted(items), "resolved": sorted({c["qid"] for c in calls if c["qid"]}),
                    "source_calls": len(calls), "plan": rec.get("plan"), "wall_s": round(time.perf_counter() - t, 2)})
    return out


def answers_of(rows: list[dict], gold) -> dict:
    """{row id: folded gold answers}: the set's literals and its answer items' names."""
    gold.fetch(q for r in rows for q in r.get("answer_qids") or [])
    out = {}
    for r in rows:
        names = set(fold(a) for a in r.get("answers") or [] if a)
        for q in r.get("answer_qids") or []:
            names |= {fold(n) for n in gold.names(q, r["lang"])}
        out[r["id"]] = {n for n in names if n}
    return out


def main() -> None:
    from retrieval_bench.mentions import Gold
    from retrieval_bench.name_index import NameIndex
    from retrieval_bench.stores import Stores
    from retrieval_sets.common import source
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    a = ap.parse_args()
    t0 = time.time()
    gold = Gold(source())
    chunks, all_rows = [], []
    for name in a.sets:
        for lang in ("en", "fr"):
            rows = sample(read_rows(name, split="dev", lang=lang), a.n)
            if rows:
                all_rows += rows
    gold.fetch(m["qid"] for r in all_rows for m in r["mentions"] if m.get("qid"))
    ga = answers_of(all_rows, gold)
    per = max(1, len(all_rows) // (a.workers * 6))
    for i in range(0, len(all_rows), per):
        part = all_rows[i:i + per]
        chunks.append((part, {r["id"]: ga[r["id"]] for r in part}))
    with ProcessPoolExecutor(a.workers) as pool:
        out = [x for part in pool.map(run_chunk, chunks) for x in part]
    stores = Stores(NameIndex(), gold)
    held = {}
    for r in out:
        for q in r["gold_items"]:
            if q not in held:
                held[q] = stores.held(q)
        r["absent"] = [q for q in r["gold_items"] if not any(held[q].values())]
    summary = {}
    for key in sorted({(r["set"], r["lang"]) for r in out}):
        sub = [r for r in out if (r["set"], r["lang"]) == key]
        st = collections.Counter(r["stage"] if r["stage"] != "spoken" else f"spoken: {r['detail']}" for r in sub)
        summary[f"{key[0]}/{key[1]}"] = {"questions": len(sub), "stages": dict(st.most_common())}
    seed = [r for r in out if r["stage"] in DUMP_LINE["stages"]]
    seed_m = [q for r in seed for q in r["gold_items"]]
    absent_m = [q for r in seed for q in r["absent"]]
    all_m = [q for r in out for q in r["gold_items"]]
    dump = {"line": DUMP_LINE, "mentions_in_sample": len(all_m), "seed_miss_mentions": len(seed_m), "absent_among_them": len(absent_m),
            "share": round(len(absent_m) / len(seed_m), 4) if seed_m else None}
    dump["verdict"] = ("the sample is under the line's minimum" if len(all_m) < DUMP_LINE["min_mentions"] else
                       "over the line: the full dump would find items no local store holds" if (dump["share"] or 0) >= DUMP_LINE["share"]
                       else "under the line: the misses are not items the local stores lack")
    by_store = collections.Counter(k for q in set(all_m) for k, v in held[q].items() if v)
    rep = {"what": "order 0046 R0: the denominator", "date": _dt.date.today().isoformat(), "n_per_set_lang": a.n,
           "pipeline": "ask loop, wiki + hdc worlds, live Wikidata source, lexicon, everyday table, tense on; no model in the programs seat",
           "summary": summary, "dump": dump, "items": {"distinct": len(set(all_m)), "held_by": dict(by_store)},
           "wikipedia_text": WIKI_TEXT, "wall_s": round(time.time() - t0), "rows": out}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R0_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": summary, "dump": dump, "items": rep["items"]}, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
