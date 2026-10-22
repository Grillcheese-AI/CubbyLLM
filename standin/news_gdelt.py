"""standin/news_gdelt.py -- GDELT 2.0's events and mentions, kept by day and filtered on ingest (order 0024).

GDELT publishes every 15 minutes (`data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip`, `.mentions.CSV.zip`,
and the translingual `.translation.export.CSV.zip` / `.translation.mentions.CSV.zip`): events coded from the news (who
did what to whom, where, when; mention, source and article counts; tone; one source URL) and every article mentioning
an event. They are **reported** claims, machine-coded and noisy: a count of sources is salience, never truth.

`poll(root)` catches up every 15-minute file since the last one it took (at most `MAX_BACK`, a day), so an hourly
caller misses nothing, and keeps:
- **an event** when it touches Canada (an actor's country `CAN`, or a place in country `CA`, Québec being `CA10`) or
  when an actor's name is a person or an organisation a world holds (`known(name)`: the hdc world's `instance of`
  in `KINDS`; names of two words or more);
- **a mention** when its event was kept (that file or earlier).
Rows go to `<root>/<YYYY-MM-DD>.events.jsonl` and `.mentions.jsonl` (the day the file was published); the files taken
are listed in `<root>/_taken.txt`. Never a GKG file (the full stream needs Nick's yes) and never a source URL followed.
GDELT's terms: open, with a citation and a link to gdeltproject.org (the audit view carries it; replies do not, 0016).

Wired: WIRED (`news_archive.Archive.poll_all`, when the manifest's `gdelt.status` is `active`).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import pathlib
import zipfile

__wiring__ = "WIRED"

BASE = "https://data.gdeltproject.org/gdeltv2/"
MAX_BACK = 96
STEP = dt.timedelta(minutes=15)
STREAMS = {"en": ("export", "mentions"), "translation": ("translation.export", "translation.mentions")}

# GDELT 2.0 event columns (61, tab-separated) and mention columns (16) that are kept, by position.
EVENT = {"id": 0, "day": 1, "a1_code": 5, "a1_name": 6, "a1_country": 7, "a2_code": 15, "a2_name": 16, "a2_country": 17,
         "root": 25, "code": 26, "base": 27, "root_code": 28, "quad": 29, "goldstein": 30, "n_mentions": 31,
         "n_sources": 32, "n_articles": 33, "tone": 34, "a1_geo": 36, "a1_geo_cc": 37, "a1_adm1": 38, "a2_geo": 44,
         "a2_geo_cc": 45, "a2_adm1": 46, "geo": 52, "geo_cc": 53, "adm1": 54, "lat": 56, "lon": 57, "added": 59,
         "source_url": 60}
MENTION = {"id": 0, "event_time": 1, "mention_time": 2, "type": 3, "source": 4, "url": 5, "confidence": 11,
           "doc_tone": 13, "translation": 14}


def stamp(t: dt.datetime) -> str:
    return t.strftime("%Y%m%d%H%M%S")


def latest(fetch, streams=("en",)) -> dt.datetime | None:
    """The newest stamp every stream has published (the translingual one can lag a file behind)."""
    out = []
    for s in streams:
        code, body = fetch(BASE + ("lastupdate.txt" if s == "en" else "lastupdate-translation.txt"))
        if code != 200:
            return None
        url = body.decode("utf-8", "replace").split()[2]
        out.append(dt.datetime.strptime(url.rsplit("/", 1)[1][:14], "%Y%m%d%H%M%S"))
    return min(out)


def due(last: str | None, newest: dt.datetime) -> list[str]:
    """The 15-minute stamps after `last` up to `newest`, at most `MAX_BACK` (the newest kept when more are due)."""
    start = dt.datetime.strptime(last, "%Y%m%d%H%M%S") + STEP if last else newest
    out, t = [], start
    while t <= newest:
        out.append(stamp(t))
        t += STEP
    return out[-MAX_BACK:]


def rows(body: bytes):
    z = zipfile.ZipFile(io.BytesIO(body))
    with z.open(z.namelist()[0]) as f:
        yield from csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t")


def pick(row: list[str], cols: dict) -> dict:
    return {k: row[i] for k, i in cols.items() if i < len(row)}


def canadian(ev: dict) -> bool:
    return ("CAN" in (ev.get("a1_country"), ev.get("a2_country"), ev.get("a1_code", "")[:3], ev.get("a2_code", "")[:3])
            or "CA" in (ev.get("geo_cc"), ev.get("a1_geo_cc"), ev.get("a2_geo_cc")))


def keep(ev: dict, known=None) -> bool:
    if canadian(ev):
        return True
    if known is None:
        return False
    return any(len(n.split()) >= 2 and known(n) for n in (ev.get("a1_name", ""), ev.get("a2_name", "")) if n)


class Gdelt:
    def __init__(self, root, fetch, known=None) -> None:
        self.root, self.fetch, self.known = pathlib.Path(root), fetch, known
        self.root.mkdir(parents=True, exist_ok=True)
        self.taken = self.root / "_taken.txt"
        self.ids = set()
        for f in self.root.glob("*.events.jsonl"):
            self.ids.update(json.loads(l)["id"] for l in f.open(encoding="utf-8") if l.strip())

    def last(self) -> str | None:
        if not self.taken.exists():
            return None
        lines = [l.split()[0] for l in self.taken.read_text(encoding="utf-8").splitlines() if l.strip()]
        return max(lines) if lines else None

    def take(self, st: str, streams=("en", "translation")) -> dict:
        day = f"{st[:4]}-{st[4:6]}-{st[6:8]}"
        out = {"stamp": st, "events": 0, "kept": 0, "mentions": 0, "missing": []}
        ev_f = (self.root / f"{day}.events.jsonl").open("a", encoding="utf-8", newline="\n")
        mn_f = (self.root / f"{day}.mentions.jsonl").open("a", encoding="utf-8", newline="\n")
        with ev_f, mn_f:
            for stream in streams:
                ev_name, mn_name = STREAMS[stream]
                code, body = self.fetch(f"{BASE}{st}.{ev_name}.CSV.zip")
                if code != 200:
                    out["missing"].append(f"{stream}:events:{code}")
                    continue
                for r in rows(body):
                    out["events"] += 1
                    ev = pick(r, EVENT)
                    if ev.get("id") not in self.ids and keep(ev, self.known):
                        self.ids.add(ev["id"])
                        ev_f.write(json.dumps({**ev, "stream": stream, "file": st}, ensure_ascii=False) + "\n")
                        out["kept"] += 1
                code, body = self.fetch(f"{BASE}{st}.{mn_name}.CSV.zip")
                if code != 200:
                    out["missing"].append(f"{stream}:mentions:{code}")
                    continue
                for r in rows(body):
                    m = pick(r, MENTION)
                    if m.get("id") in self.ids:
                        mn_f.write(json.dumps({**m, "stream": stream, "file": st}, ensure_ascii=False) + "\n")
                        out["mentions"] += 1
        with self.taken.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"{st} {json.dumps(out)}\n")
        return out

    def poll(self, streams=("en", "translation")) -> dict:
        newest = latest(self.fetch, streams)
        if newest is None:
            return {"feed": "gdelt", "error": "lastupdate unreadable", "new": 0}
        done = [self.take(st, streams) for st in due(self.last(), newest)]
        return {"feed": "gdelt", "files": len(done), "events": sum(d["events"] for d in done),
                "new": sum(d["kept"] for d in done), "mentions": sum(d["mentions"] for d in done),
                "missing": [m for d in done for m in d["missing"]]}


# The kinds of entity whose news can change a fact the worlds hold (a person named, elected, dead; a company acquired).
# Any subject would do worse: "United Kingdom", "Supreme Court" and "Prime Minister" are subjects too, and kept a quarter
# of all events (445 of 1,790 in one 15-minute file, 2026-10-22).
KINDS = ("human", "business", "enterprise", "company", "public company", "political party", "organization")


def hdc_known(path):
    """`known(name)`: the hdc world holds `name` as one of `KINDS`; None when there is no world file."""
    import sqlite3
    import sys
    if not path:
        from hdc_store import DEFAULT as path
    p = pathlib.Path(path)
    if not p.exists():
        return None
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from cubbyllm.reasoning.planner import normalize
    db = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    q = (f"SELECT 1 FROM facts WHERE skey = ? AND rel = 'instance of' AND obj IN ({','.join('?' * len(KINDS))}) LIMIT 1")
    return lambda name: db.execute(q, (normalize(name), *KINDS)).fetchone() is not None
