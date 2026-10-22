"""standin/books.py -- the history books as evidence (order 0012): a SQLite FTS5 index over their
paragraphs, an entity-mention table, and the passage finder.

Each paragraph is a row carrying its book (author and title from the file name, era / region / topic
from the folders below the root) and its character offset in the book. The mention table holds,
for each paragraph, the names of a world's `SpanIndex` it mentions (a capitalised match: a name that
is also a word, 'war' or 'time', is not a mention when lowercase), so an entity's paragraphs come
back without a text search. The build is incremental by file hash: a re-run adds the books it has
not seen and skips the rest; nothing is ever deleted. Per root, twins (one book as .epub.txt and
.pdf.txt in one folder) are read once, and where the root has a cleaned mirror (the same tree, the
format marker dropped from the name) the mirror's copy is read instead. `historical_facts.jsonl`'s
dated events join as one book per era.

The index and the books stay off-repo; their paths come from the environment (`CUBBY_BOOKS_INDEX`,
`CUBBY_BOOKS`: roots separated by os.pathsep, each `folder` or `folder|mirror`). The text is internal
evidence: it never reaches a reply (order 0016).

Wired: WIRED (`book_source.BookSource` reads it; the ask loop takes the source with `--books`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import NamedTuple

__wiring__ = "WIRED"

SCHEMA = """
CREATE TABLE IF NOT EXISTS books(id INTEGER PRIMARY KEY, path TEXT, hash TEXT UNIQUE, author TEXT, title TEXT,
    era TEXT, region TEXT, topic TEXT, paras INTEGER, chars INTEGER, added TEXT);
CREATE TABLE IF NOT EXISTS paras(id INTEGER PRIMARY KEY, book INTEGER, off INTEGER, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, content='paras', content_rowid='id');
CREATE TABLE IF NOT EXISTS names(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS mentions(name INTEGER, para INTEGER, PRIMARY KEY(name, para)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
_FMT = re.compile(r"\.(pdf|epub|rtf|mobi)$", re.I)
_RANK = {"epub": 0, "pdf": 1, "rtf": 2, "mobi": 3}
_TAGS = re.compile(r"\s*(?:\((?:v\d[\d.]*|retail)\)|\[retail\]|\(\d+(?:st|nd|rd|th) edition\))\s*", re.I)
_SKIP_LEVEL = {"epub", "pdf", "mobi", "txt", "rtf"}
_WORD = re.compile(r"[\w']+")
_STOP = {"the", "a", "an", "in", "on", "of", "at", "to", "he", "she", "it", "they", "we", "i", "his", "her",
         "this", "that", "but", "and", "or", "as", "by", "for", "with", "from", "is", "was", "not", "no", "so"}


def world_names() -> set[str]:
    """The wiki world's subjects and objects, as the serve stack's name index holds them (`serve.slot_index`)."""
    import wikikg as wk
    w = wk.wiki_world()
    return set(w.index._by_subj) | set(w.index._by_obj)


class Passage(NamedTuple):
    author: str
    title: str
    era: str
    offset: int
    text: str
    book: int
    para: int


def norm(s: str) -> str:
    """A name as the mention table keys it: lowercase words (`slots._norm`)."""
    return " ".join(re.sub(r"[^\w']+", " ", s.lower()).split())


def era_of_year(y: int) -> str:
    return "Prehistory" if y < -3000 else "AncientClassical" if y < 500 else "MiddleAges" if y < 1500 \
        else "EarlyModern" if y < 1800 else "Modern"


def meta(path: Path, root: Path) -> dict:
    """author / title from 'Author - Title [Retail].pdf.txt'; era / region / topic from the folders below
    `root`. A format folder (EPUB) is not a level, nor is the book's own folder ('1066 - Andrew
    Bridgeford', the last level, named for the book): its 'Title - Author' gives the title whole where
    the file's is cut ('You Can't Be Neutral on a Movi_mes')."""
    stem = _TAGS.sub(" ", _FMT.sub("", path.stem)).strip()
    author, _, title = stem.partition(" - ") if " - " in stem else ("", "", stem)
    parts = list(path.relative_to(root).parts[:-1])
    fmt = next((i for i, p in enumerate(parts) if p.lower() in _SKIP_LEVEL), None)
    own = parts[fmt + 1:] if fmt is not None else []                 # under a format folder: the book's own folder
    levels = parts[:fmt] if fmt is not None else parts
    if not own and levels and author and norm(author) in norm(levels[-1]) and " - " in levels[-1]:
        own = [levels.pop()]
    if own and " - " in own[-1]:
        title = own[-1].rpartition(" - ")[0].replace("_ ", ": ")
    levels = [re.sub(r"^\d+\.\s*", "", p) for p in levels]
    return {"author": author.strip(), "title": title.strip(), "era": levels[0] if levels else "",
            "region": levels[1] if len(levels) > 1 else "", "topic": "/".join(levels[2:])}


def _rank(p: Path) -> tuple:
    f = _FMT.search(p.stem)
    return _RANK.get(f.group(1).lower(), 9) if f else 9, -p.stat().st_size


def discover(roots: list[tuple[Path, Path | None]]) -> list[tuple[Path, Path]]:
    """(file to read, the root its folders are read from) for every book: twins once (epub before pdf),
    the mirror's copy where there is one."""
    out = []
    for root, mirror in roots:
        groups: dict[tuple, list[Path]] = {}
        for p in sorted(root.rglob("*.txt")):
            groups.setdefault((p.parent, _FMT.sub("", p.stem).lower().strip()), []).append(p)
        for members in groups.values():
            best = min(members, key=_rank)
            twin = mirror / best.relative_to(root).parent / (_FMT.sub("", best.stem) + ".txt") if mirror else None
            out.append((twin, mirror) if twin is not None and twin.exists() else (best, root))
    return out


def paragraphs(text: str, cap: int = 1500, floor: int = 40) -> list[tuple[int, str]]:
    """(character offset, paragraph) for a book's text: blank-line blocks; a block of short lines is one
    paragraph (a PDF's wrapped lines, joined, hyphens closed), a block of long lines a paragraph each
    (the cleaned books); one past `cap` is cut at a sentence end; a fragment under `floor` (a page
    number, a running head) is dropped."""
    out, block, pos = [], [], 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s:
            block.append((pos + len(line) - len(line.lstrip()), s))
        if not s or pos + len(line) >= len(text):
            if block:
                if sum(len(b) for _o, b in block) / len(block) > 300:
                    out += block
                else:
                    joined = re.sub(r"(?<=\w)- (?=[a-z])", "", " ".join(b for _o, b in block))
                    out.append((block[0][0], joined))
            block = []
        pos += len(line)
    cut = []
    for o, p in out:
        while len(p) > cap:
            k = p.rfind(". ", cap // 2, cap)
            k = k + 1 if k > 0 else cap
            cut.append((o, p[:k].strip())); o, p = o + k, p[k:].lstrip()
        cut.append((o, p))
    return [(o, p) for o, p in cut if len(p) >= floor]


def common_words(texts, names) -> set[str]:
    """The one-word names the books mostly write in lowercase ('because', 'art', 'figure'): words, not
    names; a sentence's first word is capitalised, so capitals alone do not tell."""
    one = {w for w in (norm(n) for n in names) if w and " " not in w}
    low: dict[str, int] = {}
    cap: dict[str, int] = {}
    for t in texts:
        for w in _WORD.findall(t):
            k = w.lower()
            if k in one:
                d = cap if w[0].isupper() else low
                d[k] = d.get(k, 0) + 1
    return {k for k, n in low.items() if n >= cap.get(k, 0)}


class Mentions:
    """The names of a `SpanIndex`-like set, matched in a paragraph: longest first, left to right, a match's
    first word capitalised in the text; a lone stop word, number or `common` word is never a name."""

    def __init__(self, names, common=()) -> None:
        self.first: dict[str, int] = {}
        self.names: set = set()
        common = set(common)
        for n in names:
            w = tuple(norm(n).split())
            if not w or len(w) > 8 or (len(w) == 1 and (w[0] in _STOP or w[0] in common or w[0].isdigit() or len(w[0]) < 3)):
                continue
            self.names.add(w)
            self.first[w[0]] = max(self.first.get(w[0], 0), len(w))

    def __call__(self, text: str) -> set[str]:
        toks = _WORD.findall(text)
        low = [t.lower() for t in toks]
        found, i = set(), 0
        while i < len(toks):
            L = self.first.get(low[i], 0) if toks[i][0].isupper() else 0
            for n in range(min(L, len(toks) - i), 0, -1):
                if tuple(low[i:i + n]) in self.names:
                    found.add(" ".join(low[i:i + n])); i += n - 1; break
            i += 1
        return found


class BookIndex:
    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = str(path or os.environ.get("CUBBY_BOOKS_INDEX", ""))
        if not self.path:
            raise ValueError("no index: give a path or set CUBBY_BOOKS_INDEX")
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._ids: dict[str, int] = {}

    def setting(self, k: str, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self, k: str, v) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, json.dumps(v)))
        self.db.commit()

    def _name(self, n: str) -> int:
        i = self._ids.get(n)
        if i is None:
            self.db.execute("INSERT OR IGNORE INTO names(name) VALUES (?)", (n,))
            i = self._ids[n] = self.db.execute("SELECT id FROM names WHERE name = ?", (n,)).fetchone()[0]
        return i

    def add(self, text_paras: list[tuple[int, str]], digest: str, path: str, info: dict, mentions=None) -> bool:
        """One book (its paragraphs given); False when its hash is already in."""
        if self.db.execute("SELECT 1 FROM books WHERE hash = ?", (digest,)).fetchone():
            return False
        cur = self.db.execute("INSERT INTO books(path, hash, author, title, era, region, topic, paras, chars, added) "
                              "VALUES (?,?,?,?,?,?,?,?,?,?)", (path, digest, info.get("author", ""), info.get("title", ""),
                              info.get("era", ""), info.get("region", ""), info.get("topic", ""), len(text_paras),
                              sum(len(p) for _o, p in text_paras), time.strftime("%Y-%m-%d %H:%M")))
        book = cur.lastrowid
        for off, p in text_paras:
            pid = self.db.execute("INSERT INTO paras(book, off, text) VALUES (?,?,?)", (book, off, p)).lastrowid
            self.db.execute("INSERT INTO fts(rowid, text) VALUES (?,?)", (pid, p))
            if mentions is not None:
                self.db.executemany("INSERT OR IGNORE INTO mentions(name, para) VALUES (?,?)",
                                    [(self._name(n), pid) for n in mentions(p)])
        return True

    def build(self, roots: list[tuple[Path, Path | None]], events: Path | None = None, mentions=None, log=print) -> dict:
        """Every book under `roots` not yet in (by hash), then the events file's eras. -> counts."""
        t0, added, skipped = time.time(), 0, 0
        for path, root in discover(roots):
            raw = path.read_bytes()
            digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
            ok = self.add(paragraphs(raw.decode("utf-8", errors="replace")), digest,
                          f"{root.name}/{path.relative_to(root).as_posix()}", meta(path, root), mentions)
            added, skipped = added + ok, skipped + (not ok)
            if ok:
                self.db.commit()
                if added % 100 == 0:
                    log(f"  {added:,} books added, {skipped:,} already in [{time.time() - t0:.0f}s]")
        if events is not None and events.exists():
            groups: dict[str, list] = {}
            for line in events.read_text(encoding="utf-8").splitlines():
                try:                                           # a raw control character inside a text (seen): kept
                    r = json.loads(line, strict=False)
                except json.JSONDecodeError:
                    continue
                if r.get("year_start") is not None:
                    groups.setdefault(era_of_year(int(r["year_start"])), []).append(r)
            for era, rs in sorted(groups.items()):
                paras = [(i, f"{r.get('title', '')}. {r.get('text', '')}".strip()) for i, r in enumerate(rs)]
                digest = hashlib.blake2b(json.dumps(rs, sort_keys=True).encode(), digest_size=16).hexdigest()
                ok = self.add(paras, digest, f"{events.name}#{era}", {"title": f"historical facts ({era})", "era": era}, mentions)
                added, skipped = added + ok, skipped + (not ok)
            self.db.commit()
        return {"added": added, "skipped": skipped, "seconds": round(time.time() - t0, 1), **self.counts()}

    def counts(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]                                  # noqa: E731
        return {"books": q("SELECT COUNT(*) FROM books"), "paragraphs": q("SELECT COUNT(*) FROM paras"),
                "names_mentioned": q("SELECT COUNT(*) FROM names"), "mentions": q("SELECT COUNT(*) FROM mentions"),
                "bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0}

    def passages(self, entity: str, words=None, era: str | None = None, k: int = 10) -> list[Passage]:
        """The paragraphs about `entity`, best first: those the mention table names it in (or, for a name it
        does not hold, those holding the phrase), ranked by BM25 over the entity's words and `words` (a
        relation's cues); `era` keeps one era's books. The mention set is read once by its key and applied to
        the FTS ranking's best 20 k: inside the SQL it stops FTS5 ranking by its index (23 s, not 0.06 s, for
        an entity of 10,000 mentions, 2026-10-22)."""
        ent = [w for w in _WORD.findall(norm(entity))]
        if not ent:
            return []
        cues = [w for w in _WORD.findall(norm(" ".join(words or []))) if w not in _STOP and w not in ent]
        phrase = '"' + " ".join(ent) + '"'
        row = self.db.execute("SELECT id FROM names WHERE name = ?", (" ".join(ent),)).fetchone()
        mention = {p for (p,) in self.db.execute("SELECT para FROM mentions WHERE name = ?", (row[0],))} if row else None
        sql = ("SELECT b.author, b.title, b.era, p.off, p.text, b.id, p.id FROM fts JOIN paras p ON p.id = fts.rowid "
               "JOIN books b ON b.id = p.book WHERE fts MATCH :q ORDER BY bm25(fts) LIMIT :k")
        out, seen = [], set()
        for q in ([f"{phrase} AND ({' OR '.join(chr(34) + c + chr(34) for c in cues)})"] if cues else []) + [phrase]:
            for r in self.db.execute(sql, {"q": q, "k": 20 * k}):
                if (mention is None or r[6] in mention) and (not era or r[2] == era) and r[6] not in seen and len(out) < k:
                    seen.add(r[6]); out.append(Passage(*r))
        return out


def roots_from_env(value: str | None = None) -> list[tuple[Path, Path | None]]:
    """`CUBBY_BOOKS`: roots separated by os.pathsep, each `folder` or `folder|mirror`."""
    out = []
    for item in filter(None, (value if value is not None else os.environ.get("CUBBY_BOOKS", "")).split(os.pathsep)):
        folder, _, mirror = item.partition("|")
        out.append((Path(folder), Path(mirror) if mirror else None))
    return out
