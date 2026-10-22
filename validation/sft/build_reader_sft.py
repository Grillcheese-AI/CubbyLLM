"""build_reader_sft -- the book reader's training records (order 0013), by distant supervision: a book sentence
that names a subject and an object the wiki world already relates is a labelled example, for free.

- **Positives:** a sentence naming the subject `s` (the mention table's capitalised match) and an object `o` with
  `s R o` in the wiki world; for a date relation (BORN_ON, DIED_ON), a date in the sentence of `o`'s year (and day,
  when both give one). A pair counts only where the sentence could state it: a relation with a verb of its own
  where its verb is (`CUED`), the two names at most `--gap` words apart, and never in a reference or an index line
  (`JUNK`); a pair that fails is neither kind, and its sentence is dropped. The answer is every such (R, `o` as the
  sentence writes it), one line each (`book_reader.answer`): the value occurs in the sentence, so the reader learns
  to copy, never to recall.
- **Negatives:** a sentence naming `s` and another name the wiki world does not relate to `s` either way -> NONE.
  A sentence where `s` and a related name meet under a relation outside the set is neither: it is dropped.
- **The set:** the relations with `--min-count` train positives, the associative ones out (`VAGUE`: a timeline
  event, 'related to': a sentence that names both rarely states them).
- **The subject** is shown as the sentence writes it ('Statue of Zeus at Olympia', not the key's case).
- **Held out by book** (its content hash), so a test book was never seen. Up to `--per-subject` paragraphs a
  subject, drawn by a seeded shuffle, so the famous do not fill the set.

The records hold the books' sentences, so they are written off-repo (`--out`, default beside the index), in the
talk trainer's form (`validation/talk_data.encode_records`: prompt, answer, split, family, id). The log,
`validation/logs/build_reader_sft_<date>.json`, holds counts only. `--audit` writes random train positives beside
the records, with an empty verdict column, for a hand check of the labels themselves (`--score-audit` reads it back):
distant supervision is only as good as the share of its pairs the sentence states.

    CUBBY_BOOKS_INDEX=<index.sqlite> python standin/data/build_reader_sft.py [--per-subject 40] [--min-count 30]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.worlds import book_frames as bf  # noqa: E402
from cubbyllm.worlds import book_reader as br  # noqa: E402
from cubbyllm.worlds.books import BookIndex, Mentions, norm  # noqa: E402

__wiring__ = "STANDALONE"

VAGUE = {"TIMELINE_EVENT", "RELATED_TO", "ASSOCIATED_WITH", "CONTEMPORARY_WITH"}
# a relation with a verb of its own is stated only where the verb is: a sentence that names Ney and 1815 at
# Waterloo does not say he died in 1815 (the smoke's first date). Without the cue the pair is dropped.
CUED = {"BORN_ON": ("born", "birth"), "BORN_IN": ("born", "birth", "birthplace"), "DIED_ON": ("died", "death", "killed", "executed"),
        "DIED_IN": ("died", "death", "killed", "executed"), "BURIED_AT": ("buried", "tomb", "grave", "interred"),
        "FOUNDED_BY": ("founded", "founder", "established"), "AUTHORED_BY": ("wrote", "written", "author", "by"),
        "AUTHOR_OF": ("wrote", "written", "author", "by"), "CREATED_BY": ("created", "creator", "built", "designed", "made",
        "by", "painted", "carved", "composed"), "CREATOR_OF": ("created", "creator", "built", "designed", "made", "by",
        "painted", "carved", "composed"), "SPOUSE_OF": ("married", "wife", "husband", "spouse", "consort", "wed"),
        "MARRIED_TO": ("married", "wife", "husband", "spouse", "consort", "wed"), "CHILD_OF": ("son", "daughter", "father",
        "mother", "child", "heir", "parents"), "PARENT_OF": ("son", "daughter", "father", "mother", "child", "heir", "parents"),
        "SUCCEEDED_BY": ("succeeded", "successor", "predecessor", "preceded", "followed"), "PRECEDED_BY": ("succeeded",
        "successor", "predecessor", "preceded", "followed"), "MEMBER_OF": ("member", "members", "joined", "among"),
        "STUDENT_OF": ("student", "pupil", "studied", "taught", "disciple"), "INFLUENCED_BY": ("influenced", "influence",
        "inspired"), "CAUSED_BY": ("caused", "cause", "because", "result", "resulted", "led")}
# a pair the sentence states sits close: more than `--gap` words between the two names is a list, not a statement;
# a reference or an index line ('pp. 4-5', '25, 30, 31', '(1983)') states nothing
JUNK = re.compile(r"\bpp?\.\s?\d|\d+,\s?\d+,\s?\d+|\(\d{4}[a-z]?\)|\bvol\.|\bibid\b|\bcf\.|\bed(?:s|ited)?\.(?:\s|$)", re.I)
DATE_RX = re.compile(bf.DATE)
LOG = ROOT / "validation" / "logs" / f"build_reader_sft_{_dt.date.today().isoformat()}.json"


def split_of(book_hash: str, test: float) -> str:
    return "test" if int(book_hash[:8], 16) / 0xFFFFFFFF < test else "train"


def find(sentence: str, name: str):
    """`name` as the sentence writes it (its words in order, any case, any gap of non-word characters): the match."""
    words = re.findall(r"[\w']+", name)
    if not words:
        return None
    return re.search(r"(?<![\w'])" + r"[^\w']+".join(map(re.escape, words)) + r"(?![\w'])", sentence, re.I)


def surface(sentence: str, name: str) -> str | None:
    m = find(sentence, name)
    return m.group(0) if m else None


def gap(sentence: str, a, b) -> int:
    """The words between two matches (spans)."""
    lo, hi = (a, b) if a[0] <= b[0] else (b, a)
    return len(re.findall(r"[\w']+", sentence[lo[1]:hi[0]])) if hi[0] >= lo[1] else 0


def date_span(sentence: str, value: str):
    """A date the sentence writes that is `value` (ISO, or a year): the same year, and the same month and day
    where both say one."""
    want = value.lstrip("-").split("-")
    for m in DATE_RX.finditer(sentence):
        got = bf._date(m)
        if got.endswith(" BC") != value.startswith("-"):
            continue
        g = got.replace(" BC", "").split("-")
        if int(g[0]) != int(want[0]):
            continue
        if len(g) == 3 and len(want) == 3 and g[1:] != want[1:]:
            continue
        return m
    return None


def world(triples):
    """subject -> [(code, object name, its key)], and subject -> every name it is related to (either way)."""
    from cubbyllm.worlds import wikikg as wk
    facts, related = collections.defaultdict(list), collections.defaultdict(set)
    for s, r, o in triples:
        S, O = wk.decamel(s), wk.decamel(o)
        facts[norm(S)].append((r, O, norm(O)))
        related[norm(S)].add(norm(O)); related[norm(O)].add(norm(S))
    return facts, related


def candidates(idx, facts, related, per_subject, rng, log, limit=None, max_gap=8):
    """Every (sentence, subject) the books give: its known lines (every code) or a negative's other names."""
    names = dict(idx.db.execute("SELECT name, id FROM names"))
    books = {b: (h, e or "unknown") for b, h, e in idx.db.execute("SELECT id, hash, era FROM books")}
    ment = Mentions(names, idx.setting("common") or ())
    subjects = sorted(s for s in facts if s in names)
    rng.shuffle(subjects)
    out, seen, t0 = [], set(), time.time()
    for n, s in enumerate(subjects[:limit]):
        paras = [p for (p,) in idx.db.execute("SELECT para FROM mentions WHERE name = ?", (names[s],))]
        rng.shuffle(paras)
        paras = paras[:per_subject]
        rows = idx.db.execute(f"SELECT id, book, text FROM paras WHERE id IN ({','.join('?' * len(paras))})", paras).fetchall()
        for pid, book, text in rows:
            for sent in bf.sentences(text):
                if len(sent) > 400 or (sent, s) in seen or JUNK.search(sent) or sum(c.isdigit() for c in sent) > 0.1 * len(sent):
                    continue
                ms = ment(sent)
                at = find(sent, s)
                if s not in ms or at is None:
                    continue
                seen.add((sent, s))
                words = set(re.findall(r"[a-z]+", sent.lower()))
                lines, uncued = [], False
                for code, O, ok in facts[s]:
                    m = date_span(sent, O) if br.is_date(O) else find(sent, O) if ok in ms and ok != s else None
                    if m is None:
                        continue
                    if (code in CUED and not words & set(CUED[code])) or gap(sent, at.span(), m.span()) > max_gap:
                        uncued = True                                       # named, not stated: neither kind
                    elif (code, m.group(0).strip()) not in lines:
                        lines.append((code, m.group(0).strip()))
                others = ms - {s} - related[s]
                if (lines or others) and not uncued:
                    out.append({"sentence": sent, "s": s, "shown": surface(sent, s), "book": books[book][0],
                                "era": books[book][1], "lines": lines, "others": len(others),
                                "related": bool((ms - {s}) & related[s])})
        if (n + 1) % 2000 == 0:
            log(f"  {n + 1:,} subjects, {len(out):,} sentences [{time.time() - t0:.0f}s]")
    return out


def score_audit(path) -> dict:
    """A hand-checked audit file -> how many of its labels the sentence states (verdict 'y'), of those judged."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(path, encoding="utf-8")][1:]
    judged = [r for r in rows if len(r) >= 5 and r[4].strip().lower() in ("y", "n")]
    yes = sum(r[4].strip().lower() == "y" for r in judged)
    return {"rows": len(rows), "judged": len(judged), "stated": yes, "precision": round(yes / max(1, len(judged)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=None, help="the books index (default: CUBBY_BOOKS_INDEX)")
    ap.add_argument("--out", default=None, help="the records, off-repo (default: reader_sft.jsonl beside the index)")
    ap.add_argument("--per-subject", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--gap", type=int, default=8, help="the most words between a pair's two names")
    ap.add_argument("--neg", type=float, default=1.0, help="negatives per positive, per split")
    ap.add_argument("--test", type=float, default=0.1, help="the share of books held out")
    ap.add_argument("--limit", type=int, default=None, help="subjects (a smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--audit", type=int, default=100, help="random positives written for a hand check of the labels")
    ap.add_argument("--score-audit", default=None, help="a filled audit file (verdict y / n): the labels' precision")
    a = ap.parse_args()
    if a.score_audit:
        print(json.dumps(score_audit(a.score_audit)))
        return
    from cubbyllm.worlds import wikikg as wk
    t0, rng = time.time(), random.Random(a.seed)
    idx = BookIndex(a.index)
    out = pathlib.Path(a.out or pathlib.Path(idx.path).parent / "reader_sft.jsonl")
    facts, related = world(wk.load_triples())
    print(f"wiki world: {len(facts):,} subjects [{time.time() - t0:.0f}s]", flush=True)
    cands = candidates(idx, facts, related, a.per_subject, rng, lambda m: print(m, flush=True), a.limit, a.gap)
    for c in cands:
        c["split"] = split_of(c["book"], a.test)
    count = collections.Counter(code for c in cands if c["split"] == "train" for code, _ in c["lines"])
    rels = sorted(r for r, k in count.items() if k >= a.min_count and r not in VAGUE)
    keep = set(rels)
    pos = [c for c in cands if any(code in keep for code, _ in c["lines"])]
    neg = [c for c in cands if not c["lines"] and not c["related"] and c["others"]]
    recs = collections.Counter()
    with open(out, "w", encoding="utf-8") as f:
        for split in ("train", "test"):
            p = [c for c in pos if c["split"] == split]
            q = [c for c in neg if c["split"] == split]
            rng.shuffle(q)
            for kind, rows in (("pos", p), ("neg", q[:int(a.neg * len(p))])):
                for c in rows:
                    lines = [(code, span) for code, span in c["lines"] if code in keep]
                    pr = br.prompt(c["sentence"], c["shown"] or c["s"])                  # the subject as the sentence writes it
                    f.write(json.dumps({"id": hashlib.blake2b(pr.encode(), digest_size=8).hexdigest(), "family": "reader",
                                        "split": split, "prompt": pr, "answer": br.answer(lines), "kind": kind,
                                        "subject": c["shown"] or c["s"], "book": c["book"], "era": c["era"],
                                        "codes": [code for code, _ in lines]}, ensure_ascii=False) + "\n")
                    recs[(split, kind)] += 1
    per_code = collections.Counter((c["split"], code) for c in pos for code, _ in c["lines"] if code in keep)
    rep = {"args": {k: v for k, v in vars(a).items() if k not in ("index", "out", "log")},
           "sentences_seen": len(cands), "positives": len(pos), "negative_pool": len(neg),
           "records": {f"{s}_{k}": n for (s, k), n in sorted(recs.items())},
           "relations": {r: {"train": per_code[("train", r)], "test": per_code[("test", r)]} for r in rels},
           "dropped_below_min_count": {r: k for r, k in count.most_common() if r not in keep},
           "per_era": dict(collections.Counter(c["era"] for c in pos)),
           "test_books": len({c["book"] for c in cands if c["split"] == "test"}),
           "train_books": len({c["book"] for c in cands if c["split"] == "train"}), "seconds": round(time.time() - t0, 1)}
    pathlib.Path(a.log).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    sample = [c for c in pos if c["split"] == "train"]
    rng.shuffle(sample)
    with open(out.with_name(out.stem + "_audit.tsv"), "w", encoding="utf-8") as f:     # off-repo, beside the records
        f.write("id\tsubject\tlabels\tsentence\tverdict\n")
        for c in sample[:a.audit]:
            pr = br.prompt(c["sentence"], c["shown"] or c["s"])
            f.write("\t".join([hashlib.blake2b(pr.encode(), digest_size=8).hexdigest(), c["shown"] or c["s"],
                               " ; ".join(f"{k} | {v}" for k, v in c["lines"] if k in keep), " ".join(c["sentence"].split()), ""]) + "\n")
    rel_file = out.with_name(out.stem + "_relations.json")
    rel_file.write_text(json.dumps(rels), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("relations", "dropped_below_min_count")}, indent=1), flush=True)
    print(f"{len(rels)} relations; wrote the records and their relation set beside the index; the log to {a.log}", flush=True)


if __name__ == "__main__":
    main()
