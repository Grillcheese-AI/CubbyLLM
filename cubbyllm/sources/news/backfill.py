"""standin/news_backfill.py -- Global Voices' history into the archive (order 0024; Nick said yes, 2026-10-22).

Cowork's backfill saves each raw feed page once, gzipped (``<dir>/<lang>/page_NNNNN.xml.gz``, ``en`` and ``fr``; the
folder is in the local notes). This reads them as the live poll reads a page -- headline, summary and, for this
publisher, the story text and credits (``news_globalvoices``) -- under the live feed's id (``globalvoices-main``,
``globalvoices-fr``), so an item the live poll already holds, or a neighbouring page repeats, is kept once (by guid).
Rows go to ``<archive>/backfill/<feed>/<published year>.jsonl``: the live day files keep their meaning (the day an
item was first seen), and the 7-day acceptance counts the live poll alone. A page is read once (``_pages.txt`` beside
its rows), and only when it has been on disk a minute (the backfill may still be writing it).

``pairs(root)`` writes ``<archive>/globalvoices_pairs.jsonl``: a translation and its source story (the credits'
``source-link``), both archived, by their two links -- parallel text.

    python standin/news_backfill.py --root <archive> --pages <dir> [--pairs]

Wired: WIRED (``news_archive.Archive.poll_all`` when ``CUBBY_GV_BACKFILL`` names the pages' folder).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import pathlib
import sys
import time

__wiring__ = "WIRED"

FEED_OF = {"en": "globalvoices-main", "fr": "globalvoices-fr"}


def ingest(archive, pages: pathlib.Path, feed: dict, min_age_s: float = 60.0) -> dict:
    from .archive import key_of, parse
    from .globalvoices import categories, story
    out = archive.root / "backfill" / feed["id"]
    out.mkdir(parents=True, exist_ok=True)
    done_p = out / "_pages.txt"
    done = set(done_p.read_text(encoding="utf-8").split()) if done_p.exists() else set()
    now, row = time.time(), {"feed": f"backfill:{feed['id']}", "pages": 0, "new": 0, "repeated": 0}
    for p in sorted(pages.glob("page_*.xml.gz")):
        if p.name in done or now - p.stat().st_mtime < min_age_s:
            continue
        files = {}
        try:
            for it in parse(gzip.decompress(p.read_bytes()), full=True):
                k = key_of(feed["id"], it)
                if k in archive.keys:
                    row["repeated"] += 1
                    continue
                archive.keys.add(k)
                it.update(story(it.pop("content", "")), categories=categories(it.get("categories", [])))
                year = (it.get("published") or "unknown")[:4]
                f = files.get(year) or files.setdefault(year, (out / f"{year}.jsonl").open("a", encoding="utf-8", newline="\n"))
                f.write(json.dumps({"key": k, "feed": feed["id"], "lang": feed.get("lang"), "kind": feed.get("kind"),
                                    "first_seen": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                    "backfill_page": p.name, **it}, ensure_ascii=False) + "\n")
                row["new"] += 1
        finally:
            for f in files.values():
                f.close()
        with done_p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(p.name + "\n")
        row["pages"] += 1
    return row


def ingest_all(archive, pages_root, manifest=None) -> dict:
    from .archive import MANIFEST, feeds
    meta = {f["id"]: f for f in feeds(manifest or MANIFEST)}
    rows = [ingest(archive, pathlib.Path(pages_root) / lang, meta[fid]) for lang, fid in FEED_OF.items()
            if (pathlib.Path(pages_root) / lang).is_dir() and fid in meta]
    return {"feed": "backfill", "new": sum(r["new"] for r in rows), "detail": rows}


def pairs(root) -> int:
    from .globalvoices import pairs as pair_rows
    root = pathlib.Path(root)
    rows = []
    for f in sorted(root.glob("????-??-??.jsonl")) + sorted(root.glob("backfill/*/*.jsonl")):
        rows += [json.loads(l) for l in f.open(encoding="utf-8") if l.strip() and '"globalvoices' in l]
    got = pair_rows(rows)
    with (root / "globalvoices_pairs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for a, b in got:
            f.write(json.dumps({"translation": a, "source": b}) + "\n")
    return len(got)


def main() -> None:
    from .archive import Archive
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.environ.get("CUBBY_NEWS_ARCHIVE"), required=not os.environ.get("CUBBY_NEWS_ARCHIVE"))
    ap.add_argument("--pages", default=os.environ.get("CUBBY_GV_BACKFILL"))
    ap.add_argument("--pairs", action="store_true")
    a = ap.parse_args()
    if a.pages:
        print(json.dumps(ingest_all(Archive(a.root), a.pages)))
    if a.pairs:
        print(f"{pairs(a.root)} pairs -> {pathlib.Path(a.root) / 'globalvoices_pairs.jsonl'}")


if __name__ == "__main__":
    main()
