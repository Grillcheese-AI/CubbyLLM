"""cubby_serve/sources/news.py -- the news archive's network part (order 0024; moved out of
``cubbyllm/sources/news/archive.py`` in order 0031, verbatim): the GET (``get``: the poller's name ``CubbyLLM/0.1``,
gzip read either way), the hourly poller thread the chat server runs (``start_poller``) and the command line. The
archive itself (its day files, robots and crawl delay over the fetch it is handed, the feeds' parsing) is
``cubbyllm.sources.news.archive.Archive``.

    python -m cubby_serve.sources.news --root <dir> --once       a manual or catch-up poll
    python -m cubby_serve.sources.news --root <dir> --every 3600 the loop
    python -m cubby_serve.sources.news --root <dir> --stats      items per feed per day, duplicates, size
(``python standin/news_archive.py ...`` runs this too, for one cycle.) The root defaults to ``CUBBY_NEWS_ARCHIVE``.

Wired: WIRED (the server's ``--news-archive``; the plugins' fetch).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

from cubbyllm.sources.news.archive import MANIFEST, Archive, stats
from cubbyllm.sources.news.source import UA

__wiring__ = "WIRED"


def get(url: str, timeout: int = 20) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
            body = r.read()
            if r.headers.get("Content-Encoding", "").lower() == "gzip" or body[:2] == b"\x1f\x8b":
                import gzip
                body = gzip.decompress(body)              # UN News answers gzip even when asked for plain bytes
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, b""


def start_poller(root, every_s: int = 3600, manifest=MANIFEST, log=print) -> threading.Thread:
    """The hourly poll in a daemon thread (the chat server's; order 0024, Nick's choice)."""
    def run():
        while True:
            try:
                rows = Archive(root, fetch=get).poll_all(manifest)
                log(f"news archive: +{sum(r['new'] for r in rows)} items from {len(rows)} feeds")
            except FileExistsError:
                log("news archive: another poller holds the lock; skipped")
            except Exception as e:                            # noqa: BLE001 -- the server keeps serving
                log(f"news archive: poll failed: {type(e).__name__}: {e}")
            time.sleep(every_s)
    t = threading.Thread(target=run, name="news-archive", daemon=True)
    t.start()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="the news archive's poller (order 0024)")
    ap.add_argument("--root", default=os.environ.get("CUBBY_NEWS_ARCHIVE"), help="the archive folder (CUBBY_NEWS_ARCHIVE)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--every", type=int, default=0, help="poll every N seconds until stopped")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if not a.root:
        ap.error("give --root or set CUBBY_NEWS_ARCHIVE")
    if a.stats:
        print(json.dumps(stats(a.root, a.manifest), indent=1))
        return
    while True:
        for r in Archive(a.root, fetch=get).poll_all(a.manifest):
            print(json.dumps(r, ensure_ascii=False), flush=True)
        if not a.every or a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
