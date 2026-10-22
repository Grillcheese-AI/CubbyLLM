"""standin/worlds_adapters.py -- one adapter per route of the manifest (order 0015). Each takes an entry (a folder or a
file, its world and tier) and the run's context, writes its rows into the world's store under the run's root, and
returns its counts: {rows, entities, files_added, files_skipped, left_out, as_of}. Every adapter is incremental by
file hash (a file already in the registry is skipped); nothing is ever deleted.

- passages: `books.BookIndex`, one per world (`passages/<world>.sqlite`), names mentioned from the wiki world;
- events: dated records into `events.sqlite`, each to its year's world (`reality:events:<year>`);
- quotes: speaker -> quote into `quotes.sqlite`, tier attributed;
- table: rows into `tables.sqlite`, one table per file, the columns named where the source names them;
- mount: an existing store (order 0014's hdc world, 0012's books index, the self world) registered, counted;
- classify: each book read as fiction or reality (the rights tracker's mark, else its text: `looks_like_fiction`),
  then to `worlds_fiction` or to passages -- the reason kept per book.

Wired: STANDALONE (`worlds_ingest`).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sqlite3

__wiring__ = "STANDALONE"

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEXT = (".txt",)
UCI = {"agaricus-lepiota": ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
                            "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                            "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                            "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"],
       "car": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
       "tic-tac-toe": ["top-left", "top-middle", "top-right", "middle-left", "middle-middle", "middle-right",
                       "bottom-left", "bottom-middle", "bottom-right", "class"]}


def slug(world: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", world.lower()).strip("_")


def file_hash(path: pathlib.Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The text files under a folder (or the file), sync-tool duplicates ('<name> (1)') skipped, in a stable order."""
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in TEXT and not re.search(r" \(\d+\)\.", p.name))


def looks_like_fiction(text: str) -> tuple[bool, str]:
    """A novel tells, a reference book states: dialogue lines and 'said' against the density of numbers and headings."""
    sample = text[:400_000]
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if not lines:
        return False, "empty"
    dialogue = sum(1 for ln in lines if re.match(r'\s*["“‘\']', ln)) / len(lines)
    said = len(re.findall(r"\b(?:said|asked|replied|whispered|shouted)\b", sample)) / max(1, len(sample) / 10_000)
    digits = sum(c.isdigit() for c in sample) / max(1, len(sample))
    fiction = dialogue > 0.12 or (said > 6 and digits < 0.01)
    return fiction, f"dialogue {dialogue:.2f}, 'said' {said:.1f}/10k chars, digits {digits:.3f}"


class Context:
    """The run: the root, the registry, the release filter, the names a mention table matches."""

    def __init__(self, root: pathlib.Path, registry: sqlite3.Connection, release=None, log=print) -> None:
        self.root, self.reg, self.release, self.log = root, registry, release, log
        self._mentions = None

    def seen(self, digest: str) -> bool:
        return self.reg.execute("SELECT 1 FROM files WHERE hash = ?", (digest,)).fetchone() is not None

    def record(self, digest, dataset, world, path, cleared) -> None:
        self.reg.execute("INSERT OR IGNORE INTO files VALUES (?,?,?,?,?,datetime('now'))",
                         (digest, dataset, world, pathlib.Path(path).name, int(bool(cleared))))

    def mentions(self):
        if self._mentions is None:
            from .books import BookIndex, Mentions
            from .books import world_names
            common = BookIndex(ROOT_BOOKS).setting("common") if ROOT_BOOKS.exists() else ()
            self._mentions = Mentions(world_names(), common or ())
        return self._mentions


ROOT_BOOKS = pathlib.Path(os.environ.get("CUBBY_BOOKS_INDEX") or "_no_books_index")   # its common-word list: one rule for every index


def passages(entry, dataset, ctx, files=None) -> dict:
    from .books import BookIndex, meta, paragraphs
    root = pathlib.Path(entry.path)
    db = ctx.root / "passages" / f"{slug(entry.world)}.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    idx = BookIndex(db)
    added = skipped = left = 0
    for p in files if files is not None else text_files(root):
        digest = file_hash(p)
        if ctx.seen(digest):
            skipped += 1; continue
        info = meta(p, root if root.is_dir() else p.parent)
        cleared = ctx.release.cleared(p, info) if ctx.release else True
        if ctx.release and ctx.release.only and not cleared:
            left += 1; continue
        text = p.read_bytes().decode("utf-8", errors="replace")
        info["topic"] = info.get("topic") or entry.world
        idx.add(paragraphs(text), digest, f"{dataset}/{p.name}", info, ctx.mentions())
        idx.db.commit()
        ctx.record(digest, dataset, entry.world, p, cleared); ctx.reg.commit()
        added += 1
    c = idx.counts()
    return {"store": str(db), "rows": c["paragraphs"], "entities": c["names_mentioned"], "files_added": added,
            "files_skipped": skipped, "left_out": left, "as_of": "2005" if "2005" in entry.tier else None}


def events(entry, dataset, ctx) -> dict:
    from .books import era_of_year
    p = pathlib.Path(entry.path)
    db = sqlite3.connect(ctx.root / "events.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS events(title TEXT, text TEXT, year_start INTEGER, year_end INTEGER, era TEXT, "
               "world TEXT, dataset TEXT)")
    digest, added = file_hash(p), 0
    if not ctx.seen(digest):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue
            y = r.get("year_start")
            if y is None:
                continue
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (r.get("title", ""), r.get("text", ""), int(y),
                       r.get("year_end"), era_of_year(int(y)), f"reality:events:{int(y)}", dataset))
            added += 1
        db.commit(); ctx.record(digest, dataset, entry.world, p, True); ctx.reg.commit()
    n = db.execute("SELECT COUNT(*), COUNT(DISTINCT world) FROM events WHERE dataset = ?", (dataset,)).fetchone()
    return {"store": str(ctx.root / "events.sqlite"), "rows": n[0], "entities": n[1], "files_added": int(added > 0),
            "files_skipped": int(added == 0), "left_out": 0, "as_of": None, "dated_share": 1.0}


def quotes(entry, dataset, ctx) -> dict:
    p = pathlib.Path(entry.path)
    db = sqlite3.connect(ctx.root / "quotes.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS quotes(speaker TEXT, quote TEXT, category TEXT, tier TEXT, dataset TEXT)")
    digest = file_hash(p)
    new = not ctx.seen(digest)
    if new:
        rows = [(r.get("author", ""), r.get("quote", ""), r.get("category", ""), "attributed", dataset)
                for r in json.loads(p.read_text(encoding="utf-8")) if r.get("quote") and r.get("author")]
        db.executemany("INSERT INTO quotes VALUES (?,?,?,?,?)", rows)
        db.commit(); ctx.record(digest, dataset, entry.world, p, True); ctx.reg.commit()
    n = db.execute("SELECT COUNT(*), COUNT(DISTINCT speaker) FROM quotes WHERE dataset = ?", (dataset,)).fetchone()
    return {"store": str(ctx.root / "quotes.sqlite"), "rows": n[0], "entities": n[1], "files_added": int(new),
            "files_skipped": int(not new), "left_out": 0, "as_of": None}


def table(entry, dataset, ctx) -> dict:
    db = sqlite3.connect(ctx.root / "tables.sqlite")
    added = skipped = rows = 0
    for p in sorted(pathlib.Path(entry.path).glob("*.data")):
        name = slug(p.stem)
        digest = file_hash(p)
        if ctx.seen(digest):
            skipped += 1
        else:
            cols = UCI.get(p.stem) or [f"c{i}" for i in range(len(p.read_text().splitlines()[0].split(",")))]
            db.execute(f"CREATE TABLE IF NOT EXISTS t_{name}({', '.join(f'[{c}] TEXT' for c in cols)})")
            data = [ln.split(",") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            db.executemany(f"INSERT INTO t_{name} VALUES ({','.join('?' * len(cols))})", [r[:len(cols)] for r in data if len(r) >= len(cols)])
            db.commit(); ctx.record(digest, dataset, entry.world, p, True); ctx.reg.commit()
            added += 1
        rows += db.execute(f"SELECT COUNT(*) FROM t_{name}").fetchone()[0]
    return {"store": str(ctx.root / "tables.sqlite"), "rows": rows, "entities": added + skipped, "files_added": added,
            "files_skipped": skipped, "left_out": 0, "as_of": None}


def mount(entry, dataset, ctx) -> dict:
    """An existing store, registered as the world it is, and counted."""
    p = pathlib.Path(entry.path) if pathlib.Path(entry.path).is_absolute() else ROOT / entry.path
    rows = entities = None
    if p.suffix == ".sqlite":
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "facts" in tables:
            rows = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            entities = db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] if "labels" in tables else None
        elif "paras" in tables:
            rows = db.execute("SELECT COUNT(*) FROM paras").fetchone()[0]
            entities = db.execute("SELECT COUNT(*) FROM names").fetchone()[0]
    return {"store": str(p), "rows": rows, "entities": entities, "files_added": 0, "files_skipped": 0, "left_out": 0,
            "as_of": "2019" if "hdc" in p.name else None, "mounted": True}
