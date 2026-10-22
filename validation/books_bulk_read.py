"""books_bulk_read -- order 0013's bulk pass: every paragraph of every book read once, its facts put into the worlds.

1. **Read** (incremental by the book's content hash: a book already read by this reader is skipped). The paragraphs
   stream in id order, a book at a time, to `--workers` processes. In each, the names a paragraph mentions (the
   index's `Mentions`) are each read by frames (`book_frames.read`) in the sentences that hold the name and a
   frame's cue word, then, with `--reader`, by the book reader (`book_reader.BookReader`) in the sentences that
   name it and another name. Every read is a row: book, paragraph, offset, subject, relation, value, reader,
   sentence, when.
2. **Aggregate** (from every row, every run, so the result depends only on the books read): one fact per
   (subject, relation, value), its witnesses its rows; a year joins the one full date of that year; then 0012's
   tiers -- a book whose person does not fit the subject's life (the wiki world's birth and death years against the
   book's era and the dates it states for the name: `book_source.fits`) is set aside; two values for a relation that
   has one: HELD, both; two independent authors: CORROBORATED; one:
   ATTRIBUTED. Against the wiki world: the same value corroborates; another value of a relation that has one
   holds the fact. Each fact goes to the world of its witnesses' era.
3. **Write** `<out>/worlds/<era>.jsonl` (the sleep log's form, `AskLoop.load_learned` loads it: fact, rel, source
   with ' (attributed)' on one witness's fact, tier, era, witnesses), `<out>/worlds/_held.jsonl` (kept, never
   loaded to be spoken), and `validation/logs/books_bulk_read_<date>.json`: counts per tier, era and relation, the
   corroborated share, each file's hash (a re-run with no new book writes the same files). No text in the log.

The reads and the worlds hold the books' sentences and facts, so `--out` is off-repo (default beside the index).

    CUBBY_BOOKS_INDEX=<index.sqlite> python validation/books_bulk_read.py [--workers 4] [--limit-books N]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import re
import sqlite3
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import book_frames as bf  # noqa: E402
import book_reader as br  # noqa: E402
from book_source import FUNCTIONAL, credited, fits, independent  # noqa: E402
from books import Mentions, norm  # noqa: E402

LOG = HERE / "logs" / f"books_bulk_read_{_dt.date.today().isoformat()}.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS done(book TEXT, reader TEXT, at TEXT, reads INTEGER, PRIMARY KEY(book, reader));
CREATE TABLE IF NOT EXISTS reads(book TEXT, para INTEGER, off INTEGER, subj TEXT, code TEXT, value TEXT, by TEXT,
    sentence TEXT, wh TEXT);
CREATE INDEX IF NOT EXISTS reads_book ON reads(book);
"""
CUE_WORDS = {w for ws in bf.CUES.values() for w in ws}
ONE_VALUE = {br.FRAME_CODE.get(r, r) for r in FUNCTIONAL}          # the relations with one value: born on, founded by ...
_W = {}


def _init(names, common, reader_spec):
    _W["ment"] = Mentions(names, common)
    _W["reader"] = make_reader(reader_spec) if reader_spec else None


def make_reader(spec):
    """The trained reader, from `module:function` returning a BookReader (the adapter's loader lives with the adapter:
    order 0013's Colab notebook). None until an adapter exists."""
    mod, _, fn = spec.partition(":")
    import importlib
    return getattr(importlib.import_module(mod), fn)()


def read_book(job):
    """One book's paragraphs -> its read rows."""
    book, paras = job
    ment, reader, rows = _W["ment"], _W.get("reader"), []
    for pid, off, text in paras:
        names = ment(text)
        if not names:
            continue
        sents = bf.sentences(text)
        for n in names:
            m = re.search(r"(?<![\w'])" + r"[^\w']+".join(map(re.escape, n.split())) + r"(?![\w'])", text, re.I)
            if m is None:
                continue
            shown = m.group(0)
            for s in sents:
                if shown not in s:
                    continue
                words = set(re.findall(r"[a-z]+", s.lower()))
                if words & CUE_WORDS:
                    for t, sentence, when in bf.read(s, shown):
                        code, value = br.frame_fact(t)
                        rows.append((book, pid, off, t.subj, code, value, "frames", sentence, json.dumps(when or {})))
                if reader is not None and len(ment(s)) > 1:
                    for code, value in reader.read(s, shown):
                        rows.append((book, pid, off, shown, code, value, "reader", s, "{}"))
    return book, rows


def stream(idx_db, skip):
    """(book hash, [(para, offset, text)]) a book at a time, in paragraph order."""
    hashes = dict(idx_db.execute("SELECT id, hash FROM books"))
    cur, buf = None, []
    for pid, book, off, text in idx_db.execute("SELECT id, book, off, text FROM paras ORDER BY id"):
        if book != cur:
            if cur is not None and hashes[cur] not in skip:
                yield hashes[cur], buf
            cur, buf = book, []
        buf.append((pid, off, text))
    if cur is not None and hashes[cur] not in skip:
        yield hashes[cur], buf


def year(v: str) -> int | None:
    m = re.search(r"\d{1,4}", v)
    return None if m is None else -int(m.group(0)) if "BC" in v or v.startswith("-") else int(m.group(0))


def agrees(code: str, a: str, b: str) -> bool:
    """One value: a date by its year ('1769-08-15' / 'c. 1769'), a name by its words (one's within the other's)."""
    if code in br.DATE_CODES:
        return year(a) is not None and year(a) == year(b)
    x, y = set(norm(a).split()) - {"the"}, set(norm(b).split()) - {"the"}
    return bool(x and y and (x <= y or y <= x))


def relations(code: str) -> list[str]:
    """The relation each of `store_facts`' sentences states: frames' label, or the wiki noun and its inverse."""
    import wikikg as wk
    if code in br.LABEL or not code.isupper():
        return [br.LABEL.get(code, code)]
    noun, inv, _form = wk.relation_entry(code)
    return [noun, inv]


def aggregate(out_db, idx_db, wiki):
    """Every read row -> facts with tiers and eras."""
    books = {h: (a or "", e or "unknown") for h, a, e in idx_db.execute("SELECT hash, author, era FROM books")}
    groups = collections.defaultdict(list)
    shown = {}
    rows = out_db.execute("SELECT book, para, off, subj, code, value, by, sentence, wh FROM reads ORDER BY rowid").fetchall()
    stated = collections.defaultdict(list)                         # (subject, book) -> what the book says of the name
    for book, _pid, _off, subj, code, value, *_rest in rows:
        stated[(norm(subj), book)].append((br.LABEL.get(code, code), value))
    life = lambda s: tuple(next((year(v) for v in wiki.get((s, c), ()) if year(v) is not None), None)   # noqa: E731
                           for c in ("BORN_ON", "DIED_ON"))
    unfit = {sb for sb, said in stated.items() if not fits(life(sb[0]), books.get(sb[1], ("", ""))[1], said)}
    for book, _pid, off, subj, code, value, by, sentence, wh in rows:
        if (norm(subj), book) in unfit:                            # the same-name guard: another person of that name
            continue
        k = (norm(subj), code, norm(value))
        shown.setdefault(k, (subj, value))
        groups[k].append({"book": book, "author": books.get(book, ("", ""))[0], "era": books.get(book, ("", "unknown"))[1],
                          "offset": off, "by": by, "when": json.loads(wh or "{}")})
    for k in [k for k in groups if re.fullmatch(r"\d{1,4}", k[2])]:          # a year joins the one full date of that year
        full = [g for g in groups if g[:2] == k[:2] and re.fullmatch(rf"0*{k[2]} \d\d \d\d", g[2])]
        if len(full) == 1:
            groups[full[0]] += groups.pop(k)
    values = collections.Counter(k[:2] for k in groups)
    facts = []
    for k, ws in sorted(groups.items()):
        subj, value = shown[k]
        gold = wiki.get((k[0], k[1]), [])
        same = any(agrees(k[1], value, g) for g in gold)
        if (k[1] in ONE_VALUE and values[k[:2]] > 1) or (k[1] in ONE_VALUE and gold and not same):
            tier = "held"
        elif independent(ws) or same:
            tier = "corroborated"
        else:
            tier = "attributed"
        era = collections.Counter(w["era"] for w in ws).most_common(1)[0][0]
        names = list(dict.fromkeys("book:" + ("&".join(sorted(credited(w["author"]))) or f"#{w['book'][:8]}") for w in ws))
        source = "+".join(names + (["wiki"] if same else [])) + (" (attributed)" if tier == "attributed" else "")
        when = next((w["when"] for w in ws if w["when"]), None)
        for i, (fact, rel) in enumerate(zip(br.store_facts(subj, k[1], value), relations(k[1]))):
            facts.append({"fact": fact, "rel": rel, "source": source, "tier": tier, "era": era, "witnesses": len(ws),
                          "by": sorted({w["by"] for w in ws}), "subject": subj, **({"inverse": True} if i else {}),
                          **({"time": when} if when and not i else {})})
    return facts


def wiki_values():
    """(subject key, code) -> the wiki world's values."""
    import wikikg as wk
    out = collections.defaultdict(list)
    for s, r, o in wk.load_triples():
        out[(norm(wk.decamel(s)), r)].append(wk.decamel(o))
    return out


def write(facts, worlds: pathlib.Path) -> dict:
    worlds.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    for f in facts:
        by["_held" if f["tier"] == "held" else f["era"]].append(f)
    sums = {}
    for name in sorted(set(by) | {p.stem for p in worlds.glob("*.jsonl")}):
        text = "".join(json.dumps({k: v for k, v in f.items() if v is not None}, ensure_ascii=False) + "\n" for f in by.get(name, []))
        (worlds / f"{name}.jsonl").write_text(text, encoding="utf-8")
        sums[name] = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return sums


def run(index, out, workers: int = 4, reader=None, wiki=None, limit_books=None, log=print) -> dict:
    """Read what is new, aggregate everything, write the worlds -> the report. `workers=0` reads in this process;
    `wiki`: (subject key, code) -> values (`wiki_values()`; {} for none)."""
    t0 = time.time()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    idx_db = sqlite3.connect(f"file:{index}?mode=ro", uri=True, check_same_thread=False)   # the pool's feeder thread streams it
    db = sqlite3.connect(out / "reads.sqlite")
    db.executescript(SCHEMA)
    tag = "frames" + (f"+{reader}" if reader else "")
    done = {h for (h,) in db.execute("SELECT book FROM done WHERE reader = ?", (tag,))}
    names = [n for (n,) in idx_db.execute("SELECT name FROM names")]
    common = json.loads((idx_db.execute("SELECT v FROM meta WHERE k = 'common'").fetchone() or ["[]"])[0])
    jobs = stream(idx_db, done)
    if limit_books:
        jobs = (j for _i, j in zip(range(limit_books), jobs))
    n_books = n_reads = 0
    pool = mp.Pool(workers, initializer=_init, initargs=(names, common, reader)) if workers else None
    if pool is None:
        _init(names, common, reader)
    try:
        for book, rows in (pool.imap_unordered(read_book, jobs, chunksize=1) if pool else map(read_book, jobs)):
            db.executemany("INSERT INTO reads VALUES (?,?,?,?,?,?,?,?,?)", rows)
            db.execute("INSERT OR REPLACE INTO done VALUES (?,?,?,?)", (book, tag, time.strftime("%Y-%m-%d %H:%M"), len(rows)))
            db.commit()
            n_books, n_reads = n_books + 1, n_reads + len(rows)
            if n_books % 100 == 0:
                log(f"  {n_books:,} books read, {n_reads:,} reads [{time.time() - t0:.0f}s]")
    finally:
        if pool is not None:
            pool.close(); pool.join()
    facts = aggregate(db, idx_db, wiki if wiki is not None else wiki_values())
    sums = write(facts, out / "worlds")
    main = [f for f in facts if not f.get("inverse")]                  # one per (subject, relation, value)
    tiers = collections.Counter(f["tier"] for f in main)
    rep = {"reader": tag, "books_read_now": n_books, "reads_now": n_reads,
           "books_done": db.execute("SELECT COUNT(*) FROM done WHERE reader = ?", (tag,)).fetchone()[0],
           "reads": db.execute("SELECT COUNT(*) FROM reads").fetchone()[0], "triples": len(main), "facts_written": len(facts),
           "tiers": dict(tiers), "corroborated_share": round(tiers["corroborated"] / max(1, len(main) - tiers["held"]), 3),
           "per_era": dict(collections.Counter(f["era"] for f in main if f["tier"] != "held")),
           "per_era_tier": {e: dict(collections.Counter(f["tier"] for f in main if f["era"] == e))
                            for e in sorted({f["era"] for f in main})},
           "per_relation": dict(collections.Counter(f["rel"] for f in main).most_common(40)),
           "entities": len({norm(f["subject"]) for f in main}),
           "by_reader": dict(collections.Counter(b for f in main for b in f["by"])),
           "files": sums, "seconds": round(time.time() - t0, 1)}
    db.close(); idx_db.close()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=os.environ.get("CUBBY_BOOKS_INDEX"))
    ap.add_argument("--out", default=None, help="off-repo: the reads and the worlds (default: bulk/ beside the index)")
    ap.add_argument("--workers", type=int, default=4, help="0: read in this process")
    ap.add_argument("--reader", default=None, help="module:function giving a trained BookReader (none yet)")
    ap.add_argument("--limit-books", type=int, default=None)
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args()
    rep = run(a.index, a.out or pathlib.Path(a.index).parent / "bulk", a.workers, a.reader, None, a.limit_books,
              lambda m: print(m, flush=True))
    pathlib.Path(a.log).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
