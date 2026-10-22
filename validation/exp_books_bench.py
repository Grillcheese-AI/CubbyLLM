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
import hashlib
import json
import os
import random
import re
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

REL = {"BORN_ON": "date of birth", "BORN_IN": "place of birth", "DIED_ON": "date of death", "DIED_IN": "place of death",
       "FOUNDED_BY": "founded by", "LOCATED_IN": "location", "PART_OF": "part of", "BURIED_AT": "place of burial"}
DATES = {"date of birth", "date of death", "inception", "point in time"}
TENSE = {"location": "present", "part of": "present", "place of burial": "present", "capital": "present"}
WRONG = {"past": "present", "present": "future"}


def year(v: str):
    m = re.search(r"(-?\d{3,4})(?:-\d\d)?(?:-\d\d)?\s*(BC|BCE)?", str(v))
    return None if not m else (-int(m.group(1)) if m.group(2) else int(m.group(1)))


def agree(rel: str, value: str, gold: set[str]) -> bool | None:
    """True / False against the world's values; None when none of them is comparable."""
    if rel in DATES:
        ys = {year(g) for g in gold} - {None}
        return None if not ys or year(value) is None else year(value) in ys
    v = set(norm(value).split()) - {"the"}
    gs = [set(norm(g).split()) - {"the"} for g in gold]
    gs = [g for g in gs if g]
    return None if not gs or not v else any(v <= g or g <= v for g in gs)


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


def finder(idx, pairs, n, rng) -> dict:
    named = [k for k in pairs if idx.db.execute("SELECT 1 FROM names WHERE name = ?", (norm(k[0]),)).fetchone()]
    rng.shuffle(named)
    rows, by_era = [], collections.defaultdict(lambda: [0, 0])
    for s, rel in named:
        if len(rows) >= n:
            break
        gold = pairs[(s, rel)]
        ps = idx.passages(s, words=bf.CUES.get(rel), k=5)
        if not ps:
            continue
        txt = [norm(p.text) for p in ps]
        hit = any((str(year(g)) in t) if rel in DATES else (norm(g) in t) for g in gold for t in txt if (year(g) if rel in DATES else norm(g)))
        era = ps[0].era or "unknown"
        by_era[era][0] += hit; by_era[era][1] += 1
        rows.append({"subject": s, "relation": rel, "hit": bool(hit), "era": era})
    k = sum(r["hit"] for r in rows)
    return {"pairs": len(rows), "hits": k, "rate": round(k / max(1, len(rows)), 3),
            "per_era": {e: {"hits": a, "pairs": b, "rate": round(a / b, 3)} for e, (a, b) in sorted(by_era.items())}, "rows": rows}


def frames(src, pairs, need, rng, cap: int = 4000) -> dict:
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
        if sum(stats["test"][x] for x in ("agree", "contra")) >= need and sum(stats["dev"][x] for x in ("agree", "contra")) >= need:
            break
        rels = subjects[s]
        asked += 1
        for t in src.facts(s, relations=rels):
            a = agree(t.rel, t.obj, pairs.get((s, t.rel), set()))
            per[t.rel]["read"] += 1
            if a is None:
                continue
            h = half(s)
            stats[h]["agree" if a else "contra"] += 1
            per[t.rel]["agree" if a else "contra"] += 1
            read.append({"subject": s, "relation": t.rel, "value": t.obj, "agrees": a, "half": h})
        for rel in rels:
            per[rel]["asked"] += 1
    ratio = {h: round(c["agree"] / max(1, c["agree"] + c["contra"]), 3) for h, c in stats.items()}
    return {"subjects_asked": asked, "dev": dict(stats["dev"]), "test": dict(stats["test"]), "agreement": ratio,
            "per_frame": {r: {**c, "yield": round(c["read"] / max(1, c["asked"]), 3)} for r, c in sorted(per.items())},
            "reads": read}


class Offline:
    """The source below the books: nothing (no network); the books' facts stand or fall on their own."""
    name = "offline"

    def facts(self, entity):
        return []


def bench(idx, pairs, reads, n, rng) -> dict:
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
    forms = Forms()
    rows, tense_rows = [], []
    for c in cands:
        s, rel = c["subject"], c["relation"]
        tense = TENSE.get(rel, "past")
        q = forms.render(rel, s, tense)[0]
        rec = loop.ask(q)
        ans, tier = rec.get("answer"), rec.get("tier")
        ok = agree(rel, ans, pairs[(s, rel)]) if rec["verified"] and ans else None
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
                     "reason": rec.get("reason"), "correct": ok, "citation_ok": cite_ok, "reply_names_no_source": names_ok,
                     "flat_on_two_witnesses": two_ok})
        wrong_q = forms.render(rel, s, WRONG[tense])[0]
        wr = loop.ask(wrong_q)
        tense_rows.append({"question": wrong_q, "verified": wr["verified"], "tier": wr.get("tier"), "reason": wr.get("reason")})
    spoken = [r for r in rows if r["verified"]]
    flat, hedged = [r for r in spoken if not r["tier"]], [r for r in spoken if r["tier"]]
    return {"questions": len(rows), "spoken": len(spoken), "flat": len(flat), "hedged": len(hedged),
            "wrong_flat": sum(r["correct"] is False for r in flat), "wrong_hedged": sum(r["correct"] is False for r in hedged),
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
    ap.add_argument("--out", default=os.path.join(HERE, "logs", f"exp_books_bench_{_dt.date.today().isoformat()}.json"))
    a = ap.parse_args()
    t0 = time.time()
    idx = BookIndex(a.index)
    pairs = gold_pairs()
    rep = {"args": {k: v for k, v in vars(a).items() if k not in ("index", "out")}, "index": idx.counts()}
    rep["finder"] = finder(idx, pairs, a.pairs, random.Random(a.seed))
    print(f"finder: {rep['finder']['hits']}/{rep['finder']['pairs']} gold in the top 5 [{time.time() - t0:.0f}s]", flush=True)
    rep["frames"] = frames(BookSource(idx), pairs, a.frames, random.Random(a.seed + 1))
    print(f"frames: agreement {rep['frames']['agreement']} (dev {rep['frames']['dev']}, test {rep['frames']['test']}) "
          f"[{time.time() - t0:.0f}s]", flush=True)
    rep["bench"] = bench(idx, pairs, rep["frames"]["reads"], a.bench, random.Random(a.seed + 2))
    b = rep["bench"]
    print(f"bench: {b['spoken']}/{b['questions']} spoken ({b['flat']} flat, {b['hedged']} hedged), wrong flat {b['wrong_flat']}, "
          f"wrong hedged {b['wrong_hedged']}; wrong tense answered flat {b['wrong_tense']['answered_flat']}/{b['wrong_tense']['asked']} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
