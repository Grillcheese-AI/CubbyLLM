"""build_books_index -- the books' passage index (order 0012, `standin/books.py`): every book under the roots not
yet in it (by file hash), their paragraphs in FTS5, the wiki world's names they mention, and the events file's
dated records as one book per era. A re-run adds only what is new; nothing is deleted.

The one-word names the books mostly write in lowercase ('art', 'because') are not mentions; that list is
computed once from a sample of the books and kept in the index, so a re-run matches as the first run did.

    CUBBY_BOOKS_INDEX=<index.sqlite> CUBBY_BOOKS="<folder>|<cleaned mirror>;<folder>" \
        python standin/data/build_books_index.py [--events <historical_facts.jsonl>] [--sample 300]

Writes validation/logs/build_books_index_<date>.json (counts, time, size; no paths).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from books import BookIndex, Mentions, common_words, discover, paragraphs, roots_from_env  # noqa: E402

__wiring__ = "STANDALONE"


def world_names() -> set[str]:
    """The wiki world's subjects and objects, as the serve stack's name index holds them (`serve.slot_index`)."""
    import wikikg as wk
    w = wk.wiki_world()
    return set(w.index._by_subj) | set(w.index._by_obj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=os.environ.get("CUBBY_BOOKS_INDEX"))
    ap.add_argument("--books", default=os.environ.get("CUBBY_BOOKS"))
    ap.add_argument("--events", default=None, help="historical_facts.jsonl (dated events, one book per era)")
    ap.add_argument("--sample", type=int, default=300, help="books read to find the names that are common words")
    ap.add_argument("--log", default=str(ROOT / "validation" / "logs" / f"build_books_index_{_dt.date.today().isoformat()}.json"))
    args = ap.parse_args()
    if not args.index or not args.books:
        ap.error("give --index and --books (or CUBBY_BOOKS_INDEX / CUBBY_BOOKS)")
    t0 = time.time()
    roots = roots_from_env(args.books)
    idx = BookIndex(args.index)
    names = world_names()
    common = idx.setting("common")
    if common is None:
        found = discover(roots)
        step = max(1, len(found) // max(1, args.sample))
        sample = [p for p, _r in found[::step]][:args.sample]
        common = sorted(common_words((t for p in sample for _o, t in paragraphs(p.read_text(encoding="utf-8", errors="replace"))), names))
        idx.put("common", common)
        idx.put("names", len(names))
    mentions = Mentions(names, common)
    print(f"{len(names):,} world names, {len(mentions.names):,} matched as mentions ({len(common):,} common words kept out) "
          f"[{time.time() - t0:.0f}s]", flush=True)
    rep = idx.build(roots, pathlib.Path(args.events) if args.events else None, mentions)
    eras = dict(idx.db.execute("SELECT era, COUNT(*) FROM books GROUP BY era").fetchall())
    rep.update(world_names=len(names), mention_names=len(mentions.names), common_words=len(common), eras=eras,
               roots=len(roots), wall_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    json.dump(rep, open(args.log, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
