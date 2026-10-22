"""hdc_world -- the hdc set's Wikidata5M facts as a world on disk (order 0014): nace-ai's hypernet-scaling-law-data
(Apache-2.0), every triplet column of every file and the `facts` sentences of the multi-hop rows, deduplicated,
named by their exact English Wikidata labels, in SQLite.

- Q-id rows (`Q.. P.. Q..`): the entities' labels from `hdc_names.tsv.gz` and `hdc_labels_api.tsv`
  (`fetch_hdc_labels.py`), the property's from `wikidata_properties_en_fr.json`;
- alias rows (`alias, edited label, alias`, the preservation files) and the `facts` sentences
  (`build_hdc_sft.Templates`): each alias to its Q-id through the label table, the edited label to the
  property's own (`relation_template_mapping.csv`);
- a fact with an end that has no usable label (a Wikimedia list, a disambiguation page, no English label, an
  alias the table cannot place) is dropped, and counted by reason.

Each fact is stored as the Wikidata source states it, `O is the <property label> of S`, with both Q-ids, so a live
refetch of the same claim gates as a duplicate. Names (labels and aliases, normalized) lead to Q-ids. Relations
whose value changes over time (heads of state and government, officeholders, populations, teams, employers,
CEOs: `TIME_SENSITIVE`) are marked; the snapshot is ~2019 (Wikidata5M). Also counted: the facts the wiki world
states about the same subject and relation, agreeing or not.

    python standin/data/hdc_world.py --src <hdc folder> [--db standin/data/out/hdc_world.sqlite]

Writes validation/logs/hdc_world_<date>.json (counts only).
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import gzip
import json
import os
import pathlib
import re
import sqlite3
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning.planner import normalize  # noqa: E402

__wiring__ = "STANDALONE"

OUT = ROOT / "standin" / "data" / "out"
SNAPSHOT = "2019"
PROVENANCE = "wikidata (Wikidata5M via hypernet-scaling-law-data, snapshot ~2019)"
TIME_SENSITIVE = {"P6", "P35", "P39", "P54", "P102", "P108", "P118", "P127", "P137", "P159", "P169", "P286", "P488",
                  "P1082", "P26", "P463", "P1037", "P2828", "P1308", "P451", "P551", "P937", "P1416", "P749"}
_Q, _P = re.compile(r"^Q\d+$"), re.compile(r"^P\d+$")
WIKI = {"BORN_IN": "place of birth", "DIED_IN": "place of death", "BURIED_AT": "place of burial", "FOUNDED_BY": "founded by",
        "PART_OF": "part of", "EDUCATED_AT": "educated at", "OCCUPATION": "occupation", "SPOUSE_OF": "spouse",
        "AUTHORED_BY": "author", "DIRECTED_BY": "director", "GENRE": "genre", "MEMBER_OF": "member of",
        "NATIONALITY": "country of citizenship", "EMPLOYED_BY": "employer", "WORKS_FOR": "employer"}
ONE_VALUED = {"place of birth", "place of death", "place of burial"}


class Templates:
    """relation_template_mapping.csv: a fact sentence is '<value> is <noun>', noun from noun_template with
    {relation} filled (the edited label when there is one) and {subject} the entity."""

    def __init__(self, path: str):
        self.by_pre: dict[str, list] = collections.defaultdict(list)
        nouns: dict[str, tuple] = {}
        for r in csv.DictReader(open(path, encoding="utf-8")):
            word = (r["edited_relation"] or r["relation"]).strip()
            if word[:1].isupper() and word[1:].islower():
                word = word.lower()                                # 'Predecessor' reads as a word, not a name
            noun = r["noun_template"].replace("{relation}", word)
            # two relations written to one sentence ('capital of' edited to 'capital' and 'capital' itself):
            # the sentence says 'the capital of S', so it is read as the relation it names
            if noun in nouns and not nouns[noun][1]:
                continue
            nouns[noun] = (r["relation"].strip(), r["edited_relation"].strip())
        for noun, (label, _) in nouns.items():
            pre, _, suf = noun.partition("{subject}")
            self.by_pre[pre].append((suf, label, noun))

    def parse(self, fact: str):
        """-> (entity, label, value, noun) or None. The first ' is ' that a template matches; the longest
        template there."""
        start = 0
        while True:
            i = fact.find(" is ", start)
            if i < 0:
                return None
            v, rem = fact[:i], fact[i + 4:]
            words, best = rem.split(" "), None
            for k in range(1, min(len(words), 9)):
                pre = " ".join(words[:k]) + " "
                for suf, label, noun in self.by_pre.get(pre, ()):
                    if len(rem) > len(pre) + len(suf) and (not suf or rem.endswith(suf)):
                        e = rem[len(pre):len(rem) - len(suf)]
                        if best is None or len(pre) + len(suf) > best[0]:
                            best = (len(pre) + len(suf), e.strip(), label, v.strip(), noun)
            if best:
                return best[1:]
            start = i + 1


def properties() -> dict[str, tuple[str, str]]:
    d = json.load(open(OUT / "wikidata_properties_en_fr.json", encoding="utf-8"))["properties"]
    return {p: ((v.get("label") or {}).get("en") or "", (v.get("label") or {}).get("fr") or "") for p, v in d.items()}


def label_names(src: pathlib.Path, props) -> dict[str, str]:
    """A relation as the set writes it (the property's label, its edited label, any case) -> the property's label."""
    out = {en.lower(): en for en, _fr in props.values() if en}
    for r in csv.DictReader(open(src / "relation_template_mapping.csv", encoding="utf-8")):
        out.setdefault(r["relation"].strip().lower(), r["relation"].strip())
        if r["edited_relation"].strip():
            out.setdefault(r["edited_relation"].strip().lower(), r["relation"].strip())
    return out


def entity_names():
    """(Q-id -> (en, fr), alias -> Q-id): the local table, then the API's answers."""
    labels, alias = {}, {}
    with gzip.open(OUT / "hdc_names.tsv.gz", "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            a, q, name, _how = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
            if q:
                alias.setdefault(a, q)
                if name:
                    labels.setdefault(q, (name, ""))
    api = OUT / "hdc_labels_api.tsv"
    if api.exists():
        for line in api.read_text(encoding="utf-8").splitlines():
            q, en, fr, how = (line.split("\t") + ["", "", "", ""])[:4]
            if how != "missing" and en:
                old = labels.get(q)
                labels[q] = (old[0] if old else en, fr)
    return labels, alias


def read(src: pathlib.Path):
    """Every (s, r, o) the set states, deduplicated, and how many rows each kind came from."""
    import pyarrow.parquet as pq
    tpl = Templates(str(src / "relation_template_mapping.csv"))
    rows, n = set(), collections.Counter()
    for f in sorted(src.rglob("*.pq")):
        cols = set(pq.read_schema(f).names)
        if {"triplet_subject", "triplet_relation", "triplet_object"} <= cols:
            t = pq.read_table(f, columns=["triplet_subject", "triplet_relation", "triplet_object"]).to_pydict()
            for s, r, o in zip(t["triplet_subject"], t["triplet_relation"], t["triplet_object"]):
                if s and r and o:
                    rows.add((str(s), str(r), str(o))); n["triplet rows"] += 1
        if "facts" in cols:
            for facts in pq.read_table(f, columns=["facts"]).column("facts").to_pylist():
                for fact in facts or []:
                    p = tpl.parse(str(fact))
                    if p:
                        e, label, v, _noun = p
                        rows.add((e, label, v)); n["fact sentences"] += 1
    return rows, n


SCHEMA = """
    CREATE TABLE facts(subj TEXT, rel TEXT, obj TEXT, skey TEXT, okey TEXT, sq TEXT, oq TEXT, pid TEXT, ts INTEGER);
    CREATE TABLE names(key TEXT, qid TEXT);
    CREATE TABLE labels(qid TEXT PRIMARY KEY, en TEXT, fr TEXT);
    CREATE TABLE rels(rel TEXT PRIMARY KEY, pid TEXT, n INTEGER, ts INTEGER);
    CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);"""


def write_world(db_path: pathlib.Path, facts: dict, labels: dict, alias: dict, pid_of: dict):
    """The world's file from `facts` {(subject Q-id, property label, object Q-id): P-id}, `labels` {Q-id: (en, fr)},
    `alias` {alias: Q-id}, `pid_of` {property label: P-id}. An existing file is moved aside, never deleted.
    -> (names written, facts per relation)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.replace(db_path.with_suffix(".sqlite.old"))
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
                   ((labels[sq][0], rel, labels[oq][0], normalize(labels[sq][0]), normalize(labels[oq][0]), sq, oq, pid,
                     int(pid in TIME_SENSITIVE)) for (sq, rel, oq), pid in facts.items()))
    used = {q for sq, _r, oq in facts for q in (sq, oq)}
    db.executemany("INSERT INTO labels VALUES (?,?,?)", ((q, *labels[q]) for q in used))
    keys = {(normalize(labels[q][0]), q) for q in used} | {(q.lower(), q) for q in used}
    keys |= {(normalize(a), q) for a, q in alias.items() if q in used and normalize(a)}
    db.executemany("INSERT INTO names VALUES (?,?)", keys)
    rels = collections.Counter(rel for _s, rel, _o in facts)
    db.executemany("INSERT INTO rels VALUES (?,?,?,?)", ((r, pid_of.get(r, ""), c, int(pid_of.get(r, "") in TIME_SENSITIVE))
                                                         for r, c in rels.items()))
    db.executemany("INSERT INTO meta VALUES (?,?)", [("snapshot", SNAPSHOT), ("provenance", PROVENANCE)])
    db.executescript("CREATE INDEX f_s ON facts(skey, rel); CREATE INDEX f_o ON facts(okey); CREATE INDEX n_k ON names(key);")
    db.commit()
    db.close()
    return len(keys), rels


def build(src: pathlib.Path, db_path: pathlib.Path) -> dict:
    t0 = time.time()
    props = properties()
    rel_of = label_names(src, props)
    pid_of = {en: p for p, (en, _fr) in props.items() if en}
    labels, alias = entity_names()
    raw, n = read(src)
    drop = collections.Counter()
    facts = {}
    for s, r, o in raw:
        if _P.match(r):
            pid, rel = r, props.get(r, ("", ""))[0]
        else:
            rel = rel_of.get(r.lower(), "")
            pid = pid_of.get(rel, "")
        if not rel:
            drop["relation without a label"] += 1; continue
        sq = s if _Q.match(s) else alias.get(s)
        oq = o if _Q.match(o) else alias.get(o)
        if not sq or not oq:
            drop["alias the table cannot place"] += 1; continue
        if sq not in labels or oq not in labels:
            drop["entity without an English label"] += 1; continue
        facts[(sq, rel, oq)] = pid
    keys, rels = write_world(db_path, facts, labels, alias, pid_of)
    used = {q for sq, _r, oq in facts for q in (sq, oq)}
    ents = {q for s, r, o in raw for q in ((s, o) if _Q.match(s) else ()) if _Q.match(q)}
    rep = {"rows_read": dict(n), "unique_statements": len(raw), "facts": len(facts), "dropped": dict(drop),
           "entities": len(used), "qid_entities_in_the_set": len(ents),
           "qid_entities_with_an_english_label": sum(q in labels for q in ents),
           "label_coverage": round(sum(q in labels for q in ents) / max(1, len(ents)), 4),
           "with_a_french_label": sum(bool(labels[q][1]) for q in used), "relations": len(rels),
           "time_sensitive": sorted(r for r in rels if pid_of.get(r, "") in TIME_SENSITIVE),
           "time_sensitive_facts": sum(c for r, c in rels.items() if pid_of.get(r, "") in TIME_SENSITIVE),
           "names": keys, "bytes": os.path.getsize(db_path), "seconds": round(time.time() - t0, 1)}
    rep["against_wikikg"] = against_wikikg(sqlite3.connect(db_path))
    return rep


def against_wikikg(db) -> dict:
    """The wiki world's facts on the same (subject, relation): agreeing, contradicting (a one-valued relation with
    no shared value), or other (a many-valued relation with none shared)."""
    import wikikg as wk
    out = collections.Counter()
    for s, r, o in wk.load_triples():
        rel = WIKI.get(r)
        if not rel:
            continue
        vals = {x for (x,) in db.execute("SELECT okey FROM facts WHERE skey = ? AND rel = ?", (normalize(wk.decamel(s)), rel))}
        if not vals:
            continue
        hit = normalize(wk.decamel(o)) in vals
        out["agree" if hit else "contradict" if rel in ONE_VALUED else "other value"] += 1
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--db", default=str(OUT / "hdc_world.sqlite"))
    ap.add_argument("--log", default=str(ROOT / "validation" / "logs" / f"hdc_world_{_dt.date.today().isoformat()}.json"))
    a = ap.parse_args()
    rep = build(pathlib.Path(a.src), pathlib.Path(a.db))
    json.dump(rep, open(a.log, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
