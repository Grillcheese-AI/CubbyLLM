"""exp_books_bench -- order 0012's acceptance on the real books (`standin/books.py`, `book_frames.py`,
`book_source.py`, the ask loop with `books=`): CPU, no model, no network.

1. The passage finder: (entity, relation) pairs the wiki world holds, the entity named in the books -- is
   the gold object in one of the top-5 passages? Per era (the top passage's book's).
2. The frames: what they read about wiki-world subjects, against the world's value for the same relation:
   agreement / (agreement + contradiction), and the yield per frame. Subjects split by a hash into a dev
   half (where the frames were looked at) and a test half (the number reported).
3. The bench: questions the store cannot answer but a book can -- (entity, relation) pairs the books read a
   value for, their wiki facts taken out of the world -- asked as people ask (`question_forms`), through the
   ask loop with the books before an offline source and the CubeLang VM. Wrong answers spoken flat (the kill
   line: 0) and hedged, against the world's value; every attributed answer's citation (subject, value and
   cue in one sentence); no reply naming a source; a flat answer from the books only with two independent
   witnesses. Then the same facts asked in the wrong tense: none may be answered flat.

Dates agree by year; names when one's words contain the other's ("Ajaccio" / "Ajaccio Corsica"). The log
keeps facts and counts only: no passage, sentence, title or author (the books stay off-repo).

    CUBBY_BOOKS_INDEX=<index.sqlite> python validation/exp_books_bench.py [--pairs 100] [--frames 200] [--bench 100]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import book_frames as bf  # noqa: E402
import wikikg as wk  # noqa: E402
from book_source import BookSource, independent  # noqa: E402
from books import BookIndex, norm  # noqa: E402
from question_forms import Forms, TemplateEmitter  # noqa: E402
from books_scoring import DATES, MANY, agree, gazetteer, year  # noqa: E402,F401 -- the scoring, here as it was

REL = {"BORN_ON": "date of birth", "BORN_IN": "place of birth", "DIED_ON": "date of death", "DIED_IN": "place of death",
       "FOUNDED_BY": "founded by", "LOCATED_IN": "location", "PART_OF": "part of", "BURIED_AT": "place of burial"}
TENSE = {"location": "present", "part of": "present", "place of burial": "present", "capital": "present"}
WRONG = {"past": "present", "present": "future"}


# Verdicts a reviewer gave where the offline gazetteer cannot see the place (order 0012's review): (subject, relation,
# answer) -> (verdict, by and why). The bench reports them apart, with the counts as scored without them.
REVIEWED = {("Gaiseric", "place of death", "Carthage"): ("finer", "Cowork, 2026-10-22: Carthage, the Vandal Kingdom's "
                                                                   "capital from 439, is a finer, true place")}


def gold_pairs():
    """{(subject name, relation): {values}} from the wiki world's triples, names de-camelled."""
    out = collections.defaultdict(set)
    for s, r, o in wk.load_triples():
        if r in REL and s != o:
            out[(wk.decamel(s), REL[r])].add(wk.decamel(o) if REL[r] not in DATES else o)
        elif r == "CAPITAL_OF":
            out[(wk.decamel(o), "capital")].add(wk.decamel(s))
    return out


def half(name: str) -> str:
    return "dev" if hashlib.blake2b(name.encode(), digest_size=2).digest()[0] % 2 == 0 else "test"


COVERED = {"date of birth", "date of death", "place of birth", "place of death", "founded by", "capital", "place of burial"}


def finder(idx, pairs, n, rng, covered: bool = False) -> dict:
    """Is the gold object in one of the top-5 passages? Pairs whose subject the books mention; `covered`: only the
    relations a history book states (births, deaths, founders, capitals, burials) and subjects mentioned in 10 or
    more paragraphs. `ceiling`: the gold anywhere in the top 200 -- what any ranking of these passages could reach."""
    def mentions(s):
        r = idx.db.execute("SELECT id FROM names WHERE name = ?", (norm(s),)).fetchone()
        return idx.db.execute("SELECT COUNT(*) FROM mentions WHERE name = ?", (r[0],)).fetchone()[0] if r else 0
    named = [k for k in pairs if (k[1] in COVERED) >= covered and mentions(k[0]) >= (10 if covered else 1)]
    rng.shuffle(named)
    rows, by_era = [], collections.defaultdict(lambda: [0, 0, 0])
    for s, rel in named:
        if len(rows) >= n:
            break
        gold = pairs[(s, rel)]
        ps = idx.passages(s, words=bf.CUES.get(rel), k=200)
        if not ps:
            continue
        key = lambda g: (str(year(g)) if year(g) is not None else None) if rel in DATES else norm(g)    # noqa: E731
        has = lambda t: any(key(g) and key(g) in norm(t) for g in gold)                                 # noqa: E731
        hit, ceil = any(has(p.text) for p in ps[:5]), any(has(p.text) for p in ps)
        era = ps[0].era or "unknown"
        by_era[era][0] += hit; by_era[era][1] += ceil; by_era[era][2] += 1
        rows.append({"subject": s, "relation": rel, "hit": bool(hit), "ceiling": bool(ceil), "era": era})
    k, c = sum(r["hit"] for r in rows), sum(r["ceiling"] for r in rows)
    return {"pairs": len(rows), "hits": k, "rate": round(k / max(1, len(rows)), 3), "ceiling": c,
            "per_era": {e: {"hits": a, "ceiling": b, "pairs": m, "rate": round(a / m, 3)} for e, (a, b, m) in sorted(by_era.items())},
            "rows": rows}


def frames(src, pairs, need, rng, within=None, cap: int = 30000) -> dict:
    subjects = collections.defaultdict(list)
    for (s, rel) in pairs:
        subjects[s].append(rel)
    named = {r[0] for r in src.index.db.execute("SELECT name FROM names")}
    order = sorted(s for s in subjects if norm(s) in named)                     # the subjects the books mention
    rng.shuffle(order)
    stats = {h: collections.Counter() for h in ("dev", "test")}
    per = collections.defaultdict(collections.Counter)
    read, asked = [], 0
    for s in order[:cap]:
        if sum(stats[h][x] for h in stats for x in ("agree", "coarser", "contradict")) >= need:
            break
        rels = subjects[s]
        asked += 1
        for t in (t for rel in rels for t in src.facts(s, relations=[rel])):     # one relation a call, as the loop asks
            a = agree(t.rel, t.obj, pairs.get((s, t.rel), set()), within)
            per[t.rel]["read"] += 1
            if a is None:
                continue
            h = half(s)
            stats[h][a] += 1
            per[t.rel][a] += 1
            read.append({"subject": s, "relation": t.rel, "value": t.obj, "verdict": a, "half": h})
        for rel in rels:
            per[rel]["asked"] += 1
    tot = sum(stats.values(), collections.Counter())
    ratio = lambda c, k: round((c["agree"] + k * c["coarser"]) / max(1, c["agree"] + c["coarser"] + c["contradict"]), 3)   # noqa: E731
    return {"subjects_asked": asked, "dev": dict(stats["dev"]), "test": dict(stats["test"]), "all": dict(tot),
            "agreement": {"with_coarser": {h: ratio(c, 1) for h, c in list(stats.items()) + [("all", tot)]},
                          "strict": {h: ratio(c, 0) for h, c in list(stats.items()) + [("all", tot)]}},
            "per_frame": {r: {**c, "yield": round(c["read"] / max(1, c["asked"]), 3)} for r, c in sorted(per.items())},
            "reads": read}


HDC_GOLD = {"capital": "capital", "part of": "part of", "located in the administrative territorial entity": "location",
            "country": "location"}


def frames_bulk(worlds: str, pairs, hdc_path=None, within=None) -> dict:
    """The frames over every paragraph of every book (order 0013's bulk worlds, after the same-name guard), against
    the wiki world's values and the hdc world's capitals, parts and locations: agreement per half and per relation."""
    import sqlite3
    gold = collections.defaultdict(set)
    for (s, rel), vs in pairs.items():
        gold[(norm(s), rel)] |= set(vs)
    if hdc_path and os.path.exists(hdc_path):
        db = sqlite3.connect(f"file:{hdc_path}?mode=ro", uri=True)
        q = f"SELECT skey, rel, obj FROM facts WHERE rel IN ({','.join('?' * len(HDC_GOLD))})"
        for sk, rel, o in db.execute(q, list(HDC_GOLD)):
            gold[(sk, HDC_GOLD[rel])].add(o)
    stats, per, rows = {h: collections.Counter() for h in ("dev", "test")}, collections.defaultdict(collections.Counter), []
    for f in sorted(glob.glob(os.path.join(worlds, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("inverse") or not r.get("rel") or " is the " not in r["fact"]:
                continue
            value, rel, subj = r["fact"].split(" is the ", 1)[0], r["rel"], r["subject"]
            a = agree(rel, value, gold.get((norm(subj), rel), set()), within)
            per[rel]["facts"] += 1
            if a is None:
                continue
            h = half(subj)
            stats[h][a] += 1; per[rel][a] += 1
            rows.append({"subject": subj, "relation": rel, "value": value, "verdict": a, "half": h, "tier": r["tier"]})
    tot = sum(stats.values(), collections.Counter())
    ok = lambda c: round((c["agree"] + c["finer"] + c["coarser"]) / max(1, sum(c[k] for k in ("agree", "finer", "coarser", "contradict"))), 3)   # noqa: E731
    return {"compared": sum(tot[k] for k in ("agree", "finer", "coarser", "contradict")), "all": dict(tot),
            "agreement": {h: ok(c) for h, c in list(stats.items()) + [("all", tot)]},
            "per_relation": {k: dict(v) for k, v in sorted(per.items())}, "rows": rows}


class Offline:
    """The source below the books: nothing (no network); the books' facts stand or fall on their own."""
    name = "offline"

    def facts(self, entity):
        return []


def bench(idx, pairs, reads, n, rng, within=None, bulk=None) -> dict:
    from ask import AskLoop
    cands = list({(r["subject"], r["relation"]): r for r in reads}.values())
    rng.shuffle(cands)
    cands = cands[:n]
    out_keys = {(s, rel) for s, rel in ((c["subject"], c["relation"]) for c in cands)}
    triples = [(s, r, o) for s, r, o in wk.load_triples()
               if not ((wk.decamel(s), REL.get(r)) in out_keys or (r == "CAPITAL_OF" and (wk.decamel(o), "capital") in out_keys))]
    world = wk.wiki_world(triples=triples)
    src = BookSource(idx)
    loop = AskLoop(TemplateEmitter(), world=world, source=Offline(), lexicon=True, books=src)
    loaded = sum(loop.load_learned(f) for f in sorted(glob.glob(os.path.join(bulk, "*.jsonl")))   # order 0013's worlds
                 if not os.path.basename(f).startswith("_")) if bulk else None                   # (never the held)
    held = loop.load_held(os.path.join(bulk, "_held.jsonl")) if bulk else None     # the held: a veto, never an answer
    forms = Forms()
    rows, tense_rows = [], []
    for c in cands:
        s, rel = c["subject"], c["relation"]
        tense = TENSE.get(rel, "past")
        q = forms.render(rel, s, tense)[0]
        rec = loop.ask(q)
        ans, tier = rec.get("answer"), rec.get("tier")
        v = agree(rel, ans, pairs[(s, rel)], within) if rec["verified"] and ans else None
        if v == "contradict" and tier == "at_one_point":                     # one value of many over time: unconfirmed
            v = "other"
        raw = v
        v = REVIEWED.get((s, rel, ans), (v,))[0] if v == "contradict" else v  # a reviewer's verdict, reported apart
        ok = None if v in (None, "other") else True if v in ("agree", "finer") else bool(tier) if v == "coarser" else False
        cite_ok = names_ok = two_ok = None
        if rec["verified"]:
            fact = rec["trace"][-1] if rec["trace"] else ""
            ws = (rec.get("evidence") or {}).get(fact, [])
            last = s.split()[-1].lower()
            cue = bf.CUES.get(rel, [])
            cite_ok = any(last in w["sentence"].lower() and (str(year(ans)) if rel in DATES else ans.lower()) in w["sentence"].lower()
                          and any(x in w["sentence"].lower() for x in cue + [rel.split()[0]]) for w in ws) if tier else None
            names_ok = not any((w["author"] and w["author"] in (rec.get("reply") or "")) or (w["title"] and w["title"] in (rec.get("reply") or ""))
                               for w in ws)
            if not tier:
                prov = loop.world.provenance.get(" ".join(fact.split()), "")
                two_ok = (len(prov.split("+")) >= 2 and (independent(ws) or any(not x.startswith("book:") for x in prov.split("+"))))
        rows.append({"question": q, "subject": s, "relation": rel, "verified": rec["verified"], "answer": ans, "tier": tier,
                     "verdict": v, "reviewed": v != raw,
                     "reason": rec.get("reason"), "correct": ok, "citation_ok": cite_ok, "reply_names_no_source": names_ok,
                     "flat_on_two_witnesses": two_ok})
        wrong_q = forms.render(rel, s, WRONG[tense])[0]
        wr = loop.ask(wrong_q)
        tense_rows.append({"question": wrong_q, "verified": wr["verified"], "tier": wr.get("tier"), "reason": wr.get("reason")})
    spoken = [r for r in rows if r["verified"]]
    flat, hedged = [r for r in spoken if not r["tier"]], [r for r in spoken if r["tier"]]
    return {"bulk_facts_loaded": loaded, "bulk_held_loaded": held, "questions": len(rows), "spoken": len(spoken), "flat": len(flat), "hedged": len(hedged),
            "wrong_flat": sum(r["correct"] is False for r in flat), "wrong_hedged": sum(r["correct"] is False for r in hedged),
            "reviewed": [{"question": r["question"], "answer": r["answer"], "verdict": r["verdict"],
                          "by": REVIEWED[(r["subject"], r["relation"], r["answer"])][1]} for r in rows if r["reviewed"]],
            "wrong_flat_as_scored": sum(r["correct"] is False or (r["reviewed"]) for r in flat),
            "unconfirmed": sum(r["correct"] is None for r in spoken),
            "correct_flat": sum(r["correct"] is True for r in flat), "correct_hedged": sum(r["correct"] is True for r in hedged),
            "citations_ok": sum(bool(r["citation_ok"]) for r in hedged), "replies_naming_a_source": sum(r["reply_names_no_source"] is False for r in spoken),
            "flat_without_two_witnesses": sum(r["flat_on_two_witnesses"] is False for r in flat),
            "refused": dict(collections.Counter(r["reason"] for r in rows if not r["verified"])),
            "wrong_tense": {"asked": len(tense_rows), "answered_flat": sum(r["verified"] and not r["tier"] for r in tense_rows),
                            "answered_hedged": sum(bool(r["verified"] and r["tier"]) for r in tense_rows),
                            "refused": dict(collections.Counter(r["reason"] for r in tense_rows if not r["verified"]))},
            "rows": rows, "tense_rows": tense_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=os.environ.get("CUBBY_BOOKS_INDEX"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--bench", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hdc", default=os.environ.get("CUBBY_HDC_WORLD"), help="the hdc world: a gazetteer for 'coarser'")
    ap.add_argument("--frames-bulk", default=None, help="order 0013's worlds dir: frames over every book, against the gold")
    ap.add_argument("--bulk", default=None, help="order 0013: the bulk pass's worlds dir, loaded into the bench's store")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", f"exp_books_bench_{_dt.date.today().isoformat()}.json"))
    a = ap.parse_args()
    t0 = time.time()
    idx = BookIndex(a.index)
    pairs = gold_pairs()
    rep = {"args": {k: v for k, v in vars(a).items() if k not in ("index", "out", "hdc")}, "index": idx.counts()}
    rep["finder"] = finder(idx, pairs, a.pairs, random.Random(a.seed))
    rep["finder_covered"] = finder(idx, pairs, a.pairs, random.Random(a.seed), covered=True)
    for k in ("finder", "finder_covered"):
        print(f"{k}: {rep[k]['hits']}/{rep[k]['pairs']} gold in the top 5, {rep[k]['ceiling']} in the top 200 "
              f"[{time.time() - t0:.0f}s]", flush=True)
    within = gazetteer(a.hdc)
    rep["frames"] = frames(BookSource(idx), pairs, a.frames, random.Random(a.seed + 1), within)
    print(f"frames: agreement {rep['frames']['agreement']} (dev {rep['frames']['dev']}, test {rep['frames']['test']}) "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if a.frames_bulk:                                                    # 0012's frames line on >= 200 triples
        rep["frames_bulk"] = frames_bulk(a.frames_bulk, pairs, a.hdc, within)
        print(f"frames over every book: {rep['frames_bulk']['compared']} compared, agreement {rep['frames_bulk']['agreement']}",
              flush=True)
    rep["bench"] = bench(idx, pairs, rep["frames"]["reads"], a.bench, random.Random(a.seed + 2), within, a.bulk)
    b = rep["bench"]
    print(f"bench: {b['spoken']}/{b['questions']} spoken ({b['flat']} flat, {b['hedged']} hedged), wrong flat {b['wrong_flat']} (as scored before review {b['wrong_flat_as_scored']}), "
          f"wrong hedged {b['wrong_hedged']}; wrong tense answered flat {b['wrong_tense']['answered_flat']}/{b['wrong_tense']['asked']} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
