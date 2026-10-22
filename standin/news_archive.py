"""standin/news_archive.py -- the news archive (order 0024): every feed item kept once, dated, never overwritten.

`news_source.NewsSource` reads a feed's current page and forgets it (its cache is overwritten every 15 minutes), so
nothing older than a page could be asked about. This keeps what it reads:
- **the feeds** come from a manifest (`standin/data/news_feeds.json`): each with its id, url, publisher site, language,
  licence, and a status (`active` or `held`, with the reason). Only `active` feeds are polled;
- **one file a day** (`<root>/<YYYY-MM-DD>.jsonl`, the day an item was first seen, UTC): an item is appended once, keyed
  by its feed and its id (the guid, else the link, else its title and date); a re-poll adds nothing;
- **headlines and summaries only**: the feed's own title and summary (the summary cut at `SUMMARY_MAX` characters; a
  feed that carries the whole release in its description gives its opening only); never an article page, never a link
  followed;
- **robots, read every poll** at the feed's host and the publisher's site: a rule refusing `CubbyLLM` (or `*`) the
  feed's path is a refusal, never worked around; an AI-agent block the manifest has not recorded holds the feed until a
  person looks (`new_ai_block` in the poll log). The poller's name is always `CubbyLLM/0.1`;
- **a poll log** (`<root>/_polls.jsonl`): per feed per poll, new items, refusals, errors.

CLI: `python standin/news_archive.py --root <dir> --once` (a manual or catch-up poll), `--every 3600` (the loop),
`--stats` (items per feed per day, duplicates, size). The chat server runs the loop in a thread
(`serve_api.py --news-archive <dir>`; off unless given). The root defaults to `CUBBY_NEWS_ARCHIVE`.

Wired: WIRED (`serve_api.py --news-archive`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request

__wiring__ = "WIRED"

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from news_parse import DC, SUMMARY_MAX, date_of, parse  # noqa: E402,F401 -- parse and the rest re-exported
from news_source import UA  # noqa: E402
from robots_rules import ai_named, crawl_delay, refuses  # noqa: E402

MANIFEST = HERE / "data" / "news_feeds.json"
ROBOTS_TTL_S = 6 * 3600


def feeds(path=MANIFEST) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["feeds"]


def key_of(feed_id: str, item: dict) -> str:
    ident = item.get("id") or item.get("link") or f"{item['title']}|{item.get('published')}"
    return hashlib.sha1(f"{feed_id}\t{ident}".encode()).hexdigest()


def _get(url: str, timeout: int = 20) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
            body = r.read()
            if r.headers.get("Content-Encoding", "").lower() == "gzip" or body[:2] == b"\x1f\x8b":
                import gzip
                body = gzip.decompress(body)              # UN News answers gzip even when asked for plain bytes
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, b""


class Archive:
    """Append-only day files under `root`, the keys already held, and a lock so two pollers never write at once."""

    def __init__(self, root, fetch=_get, sleep=time.sleep) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fetch, self.sleep, self._last = fetch, sleep, {}
        self.keys = set()
        for f in sorted(self.root.glob("????-??-??.jsonl")) + sorted(self.root.glob("backfill/*/*.jsonl")):
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    self.keys.add(json.loads(line)["key"])
        self._robots: dict[str, tuple[float, str | None]] = {}

    # -- robots ---------------------------------------------------------------------------------------------------------
    def robots(self, host: str) -> str | None:
        """The host's robots.txt (cached `ROBOTS_TTL_S`); None when there is none to read (404, unreachable)."""
        hit = self._robots.get(host)
        if hit and time.time() - hit[0] < ROBOTS_TTL_S:
            return hit[1]
        try:
            code, body = self.fetch(f"https://{host}/robots.txt")
            self._last[host] = time.time()                    # a request too: the crawl delay counts from it
            txt = body.decode("utf-8", "replace") if code == 200 else None
        except Exception:                                    # noqa: BLE001 -- no robots.txt is not a yes and not a no
            txt = None
        self._robots[host] = (time.time(), txt)
        return txt

    def polite(self, url: str):
        """``fetch(url)``, never sooner after this host's last request than its robots ``Crawl-delay`` asks."""
        host = urllib.parse.urlsplit(url).netloc
        txt = self.robots(host)
        wait = (crawl_delay(txt, UA) if txt else 0.0) - (time.time() - self._last.get(host, 0.0))
        if wait > 0:
            self.sleep(wait)
        try:
            return self.fetch(url)
        finally:
            self._last[host] = time.time()

    def check(self, feed: dict) -> str | None:
        """None when the feed may be polled; else the reason it is not, this poll."""
        hosts = [urllib.parse.urlsplit(feed["url"]).netloc] + [h for h in feed.get("publisher_hosts", []) if h]
        named = set()
        for h in dict.fromkeys(hosts):
            txt = self.robots(h)
            if txt is None:
                continue
            if h == hosts[0] and refuses(txt, UA, feed["url"]):
                return "refused"                              # addressed to us (or to everyone): never worked around
            named.update(ai_named(txt))
        new = named - set(feed.get("ai_agents_named", []))
        return f"new_ai_block: {','.join(sorted(new))}" if new else None

    # -- polling --------------------------------------------------------------------------------------------------------
    def poll(self, feed: dict, now: dt.datetime | None = None) -> dict:
        now = now or dt.datetime.now(dt.timezone.utc)
        row = {"feed": feed["id"], "at": now.isoformat(timespec="seconds"), "new": 0}
        why = self.check(feed)
        if why:
            return {**row, "skipped": why}
        try:
            code, body = self.polite(feed["url"])
            if code != 200:
                return {**row, "error": f"HTTP {code}"}
            items = parse(body, full=bool(feed.get("full_text")))
            if feed.get("full_text") == "globalvoices":       # Nick: this publisher's story text is kept (CC BY 3.0)
                from news_globalvoices import categories, story
                for it in items:
                    it.update(story(it.pop("content", "")), categories=categories(it.get("categories", [])))
        except Exception as e:                                # noqa: BLE001 -- a feed that is down is not an answer
            return {**row, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        day = self.root / f"{now.date().isoformat()}.jsonl"
        with day.open("a", encoding="utf-8", newline="\n") as f:
            for it in items:
                k = key_of(feed["id"], it)
                if k in self.keys:
                    continue
                self.keys.add(k)
                f.write(json.dumps({"key": k, "feed": feed["id"], "lang": feed.get("lang"), "kind": feed.get("kind", "news"),
                                    **({"topic": feed["topic"]} if feed.get("topic") else {}),
                                    "first_seen": row["at"], **it}, ensure_ascii=False) + "\n")
                row["new"] += 1
        return {**row, "items": len(items)}

    def last_polled(self) -> dict[str, float]:
        """Each feed's last poll that read it (not a skip, not an error), from the poll log: a feed polled every 6 hours
        or daily (`every_h` in the manifest) waits that long; 5 minutes' slack keeps an hourly loop on time."""
        out, log = {}, self.root / "_polls.jsonl"
        if log.exists():
            for line in log.open(encoding="utf-8"):
                r = json.loads(line) if line.strip() else {}
                if "items" in r and "at" in r:
                    out[r["feed"]] = dt.datetime.fromisoformat(r["at"]).timestamp()
        return out

    def poll_all(self, manifest=MANIFEST) -> list[dict]:
        with self._lock():
            last = self.last_polled()
            now = time.time()
            rows = [self.poll(f) for f in feeds(manifest) if f.get("status") == "active"
                    and now - last.get(f["id"], 0) >= f.get("every_h", 1) * 3600 - 300]
            if os.environ.get("CUBBY_GV_BACKFILL"):          # Global Voices' history pages, as they land (news_backfill)
                from news_backfill import ingest_all
                rows.append(ingest_all(self, os.environ["CUBBY_GV_BACKFILL"], manifest))
            cfg = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8")).get("gdelt") or {}
            if cfg.get("status") == "active":                 # the structured event source (news_gdelt.py)
                try:
                    from news_gdelt import Gdelt, hdc_known
                    g = Gdelt(self.root / "gdelt", self.fetch, hdc_known(os.environ.get("CUBBY_HDC_WORLD")))
                    rows.append(g.poll(tuple(cfg.get("streams", ["en"]))))
                except Exception as e:                        # noqa: BLE001 -- the feeds' poll still counts
                    rows.append({"feed": "gdelt", "new": 0, "error": f"{type(e).__name__}: {str(e)[:100]}"})
            with (self.root / "_polls.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    def _lock(self):
        archive = self

        class _L:
            def __enter__(self):
                self.p = archive.root / "_poll.lock"
                if self.p.exists() and time.time() - self.p.stat().st_mtime > 3600:
                    self.p.unlink()                           # a poller that died an hour ago holds nothing
                self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self

            def __exit__(self, *exc):
                os.close(self.fd)
                self.p.unlink()
        return _L()


def stats(root, manifest=MANIFEST) -> dict:
    """Items per feed per day, per region and per language, duplicate keys (should be 0), size on disk."""
    root, per, keys, dup = pathlib.Path(root), {}, set(), 0
    files = sorted(root.glob("????-??-??.jsonl"))
    for f in files:
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            dup += o["key"] in keys
            keys.add(o["key"])
            per.setdefault(o["feed"], {}).setdefault(f.stem, 0)
            per[o["feed"]][f.stem] += 1
    meta = {f["id"]: f for f in feeds(manifest)}
    cover = {k: {} for k in ("region", "lang")}
    for fid, days in per.items():
        for k in cover:
            v = (meta.get(fid) or {}).get(k) or "?"
            cover[k][v] = cover[k].get(v, 0) + sum(days.values())
    return {"items": len(keys), "duplicates": dup, "days": [f.stem for f in files],
            "bytes": sum(f.stat().st_size for f in files), "per_region": cover["region"], "per_lang": cover["lang"],
            "per_feed_per_day": per}


def topics(rows) -> dict[str, list[str]]:
    """Each item's topics: the topics of every feed that carried it (by link; Global Issues' topic feeds, Nick)."""
    out: dict[str, set] = {}
    for r in rows:
        if r.get("topic") and r.get("link"):
            out.setdefault(r["link"], set()).add(r["topic"])
    return {k: sorted(v) for k, v in out.items()}


def start_poller(root, every_s: int = 3600, manifest=MANIFEST, log=print) -> threading.Thread:
    """The hourly poll in a daemon thread (the chat server's; order 0024, Nick's choice)."""
    def run():
        while True:
            try:
                rows = Archive(root).poll_all(manifest)
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
        for r in Archive(a.root).poll_all(a.manifest):
            print(json.dumps(r, ensure_ascii=False), flush=True)
        if not a.every or a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
