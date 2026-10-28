"""fame_set -- order 0046's fame set: names Wikidata5M gives to two or more items of one kind, built by name collision
(never by sitelinks or degree), asked bare (no single item may be picked) and qualified by one item's own fact (that
item, and only it, is meant).

1. **Names:** every English label (Wikidata's, as the hdc world holds them: Wikidata5M's entity file lists an item's
   names unordered and gives no label) that two or more Wikidata5M items share, where
   two or more of them are of one kind -- a shared P31, or one item's P31 a direct ``subclass of`` (P279) of the
   other's -- and two or more of those carry one relation of the fixed list (place of birth, place of death, country,
   capital, occupation). Date of birth is on the order's list and cannot be used: Wikidata5M holds item-to-item facts
   only, no dates.
2. **Sample:** 500 names, seed 0; per name, one relation those items carry (seed 0 among the eligible).
3. **Bare questions,** EN and FR, from fixed templates. Gold: no single pick.
4. **Qualified variants:** the same question with one discriminator from one item's own facts that no other of those
   items shares by its words -- its occupation (a person asked a place), or a place (``located in``, P131; a person
   asked its occupation: its birthplace). The rules, each found by a hand spot check of 50 rows (2026-10-28): a label
   shared or held whole by another's value tells nothing apart ("Kent County"; "Italy" in "Kingdom of Italy"); a
   place another item is not ruled out of -- its known places all at or above it, or any place of the same name
   holding it -- tells nothing apart ("Reid Avenue in Brooklyn", "Rye High School in Rye"); an item with nothing
   recorded for the relation cannot be ruled out. Gold: that item. A French variant only where the discriminator has a
   French label (the hdc world's, else Wikidata's by id) told apart in French too.
5. **Freeze:** ``fame_set.jsonl`` beside this file (committed), its SHA-256 in ``frozen.json``, before R2 runs.

    python validation/retrieval_sets/fame_set.py [--n 500]

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import SEED, raw_dir  # noqa: E402
from retrieval_sets.lexicon import freeze  # noqa: E402

__wiring__ = "STANDALONE"

OUT = os.path.join(HERE, "fame_set.jsonl")
HUMAN = "Q5"
ASK = {"P19": ("Where was {n} born?", "Où est né {n} ?"), "P20": ("Where did {n} die?", "Où est mort {n} ?"),
       "P17": ("Which country is {n} in?", "Dans quel pays se trouve {n} ?"),
       "P36": ("What is the capital of {n}?", "Quelle est la capitale de {n} ?"),
       "P106": ("What does {n} do for a living?", "Quel est le métier de {n} ?")}
KEEP = {"P31", "P279", "P131"} | set(ASK)


def triples(folder: str):
    for split in ("train", "valid", "test"):
        with open(os.path.join(folder, f"wikidata5m_transductive_{split}.txt"), encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")[:3]
                if r in KEEP:
                    yield h, r, t


def load(folder: str):
    facts = collections.defaultdict(lambda: collections.defaultdict(set))
    for h, r, t in triples(folder):
        facts[h][r].add(t)
    # the labels: Wikidata5M's entity file lists each item's names unordered (Titian's first is "ditiano"), so it
    # gives no label; the hdc world's English labels are Wikidata's own (2.46M of these items)
    from cubbyllm.worlds.hdc_store import DEFAULT
    db = sqlite3.connect(f"file:{DEFAULT}?mode=ro", uri=True)
    labels = {q: en for q, en in db.execute("SELECT qid, en FROM labels") if en}
    return facts, labels


def fr_labels(qids) -> dict:
    from cubbyllm.worlds.hdc_store import DEFAULT
    db = sqlite3.connect(f"file:{DEFAULT}?mode=ro", uri=True)
    out = {}
    qs = list(qids)
    for i in range(0, len(qs), 500):
        chunk = qs[i:i + 500]
        for q, fr in db.execute(f"SELECT qid, fr FROM labels WHERE qid IN ({','.join('?' * len(chunk))})", chunk):
            if fr:
                out[q] = fr
    missing = [q for q in qs if q not in out]                # the rest from Wikidata by id (the source's cache); a call
    if missing:                                              # that never succeeds stops the build: a frozen set must not
        from retrieval_sets.common import get, source       # depend on how loaded Wikidata was that hour
        src = source()
        for i in range(0, len(missing), 50):
            chunk = missing[i:i + 50]
            d = get(src, {"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels", "languages": "fr",
                          "sitefilter": "enwiki|frwiki"}, tries=9)
            if d is None:
                raise SystemExit("Wikidata did not answer a French-label call after 9 tries: rebuild when it does")
            for q, e in (d.get("entities") or {}).items():
                lab = ((e.get("labels") or {}).get("fr") or {}).get("value")
                if lab:
                    out[q] = lab
    return out


def one_kind(a: str, b: str, facts) -> bool:
    ka, kb = facts[a]["P31"], facts[b]["P31"]
    if ka & kb:
        return True
    up = lambda ks: set().union(*(facts[k]["P279"] for k in ks)) if ks else set()  # noqa: E731
    return bool(ka & up(kb) or kb & up(ka))


def eligible(items: list[str], facts) -> dict:
    """{relation: [items of one kind that carry it]} for the relations two or more same-kind items carry."""
    out = {}
    for r in ASK:
        carriers = [q for q in items if facts[q][r]]
        clique = [q for q in carriers if any(o != q and one_kind(q, o, facts) for o in carriers)]
        if len(clique) >= 2:
            out[r] = sorted(clique, key=lambda q: int(q[1:]))
    return out


def discriminator(q: str, others: list[str], r: str, facts, labels: dict) -> tuple[str, str] | None:
    """(relation, value) from `q`'s own facts whose words no item in `others` shares, never the relation asked. The
    question carries words, not ids: two different items both labelled "Kent County" tell nothing apart (the spot
    check of 2026-10-28 found "Caledonia in Kent County" this way), so the value's label must be unique, folded."""
    from retrieval_sets.common import fold
    human = HUMAN in facts[q]["P31"]
    # a person asked a place: its occupation; asked its occupation: its birthplace (a place, checked as one; the
    # second spot check found citizenship's words unreliable: "from Great Britain" fits a man recorded under the UK)
    order = (["P106"] if r != "P106" else ["P19"]) if human else (["P131"] if r != "P131" else [])
    for d in order:
        if any(not facts[o][d] and not (d == "P131" and facts[o]["P17"]) for o in others):
            continue                                     # an item with nothing recorded cannot be ruled out
        vals = {v for o in others for v in place_set(o, d, facts)} if d in PLACES else {v for o in others for v in facts[o][d]}
        theirs = {fold(labels.get(v, v)) for v in vals}
        mine = sorted((v for v in facts[q][d] if v in labels and not overlaps(fold(labels[v]), theirs)
                       and (d not in PLACES or not coarse_other(v, others, facts, labels[v], d))), key=lambda v: int(v[1:]))
        if mine:
            return d, mine[0]
    return None


PLACES = ("P131", "P19")


def place_set(o: str, d: str, facts) -> set:
    """Where an item is (P131: the item itself) or was born (P19: its birthplace), with every place above it."""
    if d == "P131":
        return places_up(o, facts)
    return set().union(*({b} | places_up(b, facts) for b in facts[o]["P19"])) if facts[o]["P19"] else set()


def overlaps(mine: str, theirs: set) -> bool:
    """The words are shared when one label holds the other whole ("italy" in "kingdom of italy": the second spot
    check, 2026-10-28, Andrea Maffei asked "(Italie)", both men Italian)."""
    return any(mine == t or f" {mine} " in f" {t} " or f" {t} " in f" {mine} " for t in theirs if t)


HOMONYMS: dict = {}                                       # folded label -> the items that carry it (set by `build`)


def coarse_other(place: str, others: list[str], facts, label: str = "", d: str = "P131") -> bool:
    """An item whose every known place is at or above the named place (or that has none) may be in it: the words
    cannot rule it out ("Reid Avenue in Brooklyn": the other station's place is New York City, which holds Brooklyn).
    Every place carrying the same name counts ("Rye High School in Rye": the other school is recorded only in New
    York state, where another Rye lies)."""
    from retrieval_sets.common import fold
    for p in set(HOMONYMS.get(fold(label), ())) | {place}:
        above = places_up(p, facts) | {p}
        for o in others:
            theirs = place_set(o, d, facts)
            if not theirs or theirs <= above:
                return True
    return False


def places_up(q: str, facts, steps: int = 3) -> set:
    """The places `q` is in: ``located in`` (P131) up `steps` steps, and every country (P17) on the way."""
    out, frontier = set(), {q}
    for _ in range(steps):
        frontier = set().union(*(facts[x]["P131"] for x in frontier)) - out if frontier else set()
        out |= frontier
    return out | set().union(*(facts[x]["P17"] for x in out | {q}))


def fr_unique(v: str, others: list[str], d: str, facts, fr: dict) -> bool:
    """The French label of the discriminator is told apart from the other items' values in French too."""
    from retrieval_sets.common import fold
    mine = fold(fr.get(v, ""))
    vals = {w for o in others for w in (places_up(o, facts) if d == "P131" else facts[o][d])}
    return bool(mine) and mine not in {fold(fr.get(w, "")) for w in vals}


def qualified_text(name: str, r: str, d: str, val_en: str, val_fr: str | None) -> tuple[str, str | None]:
    if d == "P106":
        en = ASK[r][0].replace("{n}", f"{name}, the {val_en},")
        fr = ASK[r][1].replace("{n}", f"{name}, {val_fr},") if val_fr else None
    elif d == "P19":
        en = ASK[r][0].replace("{n}", f"{name}, born in {val_en},")
        fr = ASK[r][1].replace("{n}", f"{name}, originaire de {val_fr},") if val_fr else None
    else:
        en = ASK[r][0].replace("{n}", f"{name} in {val_en}")
        fr = ASK[r][1].replace("{n}", f"{name} ({val_fr})") if val_fr else None
    tidy = lambda s: s.replace(",,", ",").replace(", ?", " ?").replace(",?", "?").replace(" ?", " ?") if s else s  # noqa: E731
    return tidy(en).replace(" ?", "?"), tidy(fr)


def build(n: int, folder: str) -> list[dict]:
    facts, labels = load(folder)
    from retrieval_sets.common import fold
    HOMONYMS.clear()
    for q, lab in labels.items():
        if facts.get(q) is not None and (facts[q]["P131"] or facts[q]["P17"]):     # places only: an item that is somewhere
            HOMONYMS.setdefault(fold(lab), []).append(q)
    by_name = collections.defaultdict(list)
    for q, lab in labels.items():
        if q in facts:
            by_name[lab].append(q)
    names = []
    for lab in sorted(by_name):
        items = by_name[lab]
        if len(items) >= 2:
            el = eligible(items, facts)
            if el:
                names.append((lab, el))
    rng = random.Random(SEED)
    picked = sorted(rng.sample(names, min(n, len(names))), key=lambda x: x[0])
    need = {v for _lab, el in picked for qs in el.values() for q in qs for v in facts[q]["P106"] | place_set(q, "P131", facts) | place_set(q, "P19", facts)}
    fr = fr_labels(need | {q for _lab, el in picked for qs in el.values() for q in qs})
    rows = []
    for k, (lab, el) in enumerate(picked):
        r = rng.choice(sorted(el))
        items = el[r]
        answers = {q: sorted(facts[q][r]) for q in items}
        base = {"name": lab, "relation": r, "items": items, "answers": answers, "n_namesakes": len(by_name[lab])}
        name_fr = fr.get(items[0]) if all(fr.get(q) == fr.get(items[0]) for q in items) and fr.get(items[0]) else lab
        rows.append({"id": f"fame:{k}:bare:en", "lang": "en", "form": "bare", "question": ASK[r][0].format(n=lab), **base})
        rows.append({"id": f"fame:{k}:bare:fr", "lang": "fr", "form": "bare", "question": ASK[r][1].format(n=name_fr), **base})
        for q in items:
            others = [o for o in items if o != q]
            d = discriminator(q, others, r, facts, labels)
            if not d:
                continue
            en, frq = qualified_text(lab, r, d[0], labels[d[1]], fr.get(d[1]) if fr_unique(d[1], others, d[0], facts, fr) else None)
            extra = {**base, "gold_item": q, "discriminator": {"relation": d[0], "value": d[1], "label": labels[d[1]]}}
            rows.append({"id": f"fame:{k}:q:{q}:en", "lang": "en", "form": "qualified", "question": en, **extra})
            if frq:
                rows.append({"id": f"fame:{k}:q:{q}:fr", "lang": "fr", "form": "qualified", "question": frq.replace("{n}", name_fr), **extra})
    return rows, len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--folder", default=raw_dir("wikidata5m"))
    a = ap.parse_args()
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists: the fame set is frozen once (move it aside to rebuild)")
    rows, pool = build(a.n, a.folder)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    c = collections.Counter((r["form"], r["lang"]) for r in rows)
    h = freeze("fame_set", OUT, f"order 0046's fame set: {a.n} names drawn by seed 0 from {pool} eligible; rows {dict(c)}")
    print(f"{pool} eligible names; {len(rows)} rows {dict(c)}; relations {dict(collections.Counter(r['relation'] for r in rows if r['form'] == 'bare' and r['lang'] == 'en'))}; sha256 {h}")


if __name__ == "__main__":
    main()
