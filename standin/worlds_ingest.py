"""standin/worlds_ingest.py -- everything in the datasets into worlds, one tool driven by the manifest (order 0015).

    python validation/build/ingest_worlds.py [--root <worlds dir>] [--only <dataset substring>] [--frames] [--release]
    (``python standin/worlds_ingest.py`` runs the same)

`worlds_manifest.plan` gives every manifest row a decision (ingest / excluded / queued); this runs the ingests through
`worlds_adapters` (passages, events, quotes, table, mount) and `worlds_fiction`, and records every decision in the
registry (`<root>/registry.sqlite`): datasets (status, reason, counts, time), files (hash -> dataset, world, release),
worlds. A re-run reads only files whose hash is new; nothing is deleted. `classify` rows read each book as fiction or
reality (the rights tracker's mark first, else `looks_like_fiction`) and send it on, keeping the reason. `--frames`
runs order 0013's bulk pass over each passage world whose manifest route says "frames": its facts, tiers and eras
beside the index. `--release` writes a separate store under `<root>/release` from the works the rights tracker
clears (owned, licensed, public domain) and reports what it left out.

Writes `validation/logs/worlds_ingest_<date>.json`: one row a dataset -- status, reason, world, rows, entities, files,
as-of, time. No paths, titles or text.

Wired: STANDALONE (a build tool; the ask loop mounts what it writes through `worlds_store`).
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import pathlib
import re
import sqlite3
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import worlds_adapters as wa  # noqa: E402
import worlds_fiction as wf  # noqa: E402
import worlds_manifest as wm  # noqa: E402

__wiring__ = "STANDALONE"

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(name TEXT PRIMARY KEY, tree TEXT, kind TEXT, status TEXT, reason TEXT, report TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS files(hash TEXT PRIMARY KEY, dataset TEXT, world TEXT, name TEXT, cleared INTEGER, at TEXT);
CREATE TABLE IF NOT EXISTS worlds(id TEXT PRIMARY KEY, kind TEXT, store TEXT, tier TEXT, as_of TEXT);
"""
# Local files, named by the environment (CLAUDE.local.md): the rights tracker, and the Gutenberg catalog under the root.
TRACKER = os.environ.get("CUBBY_RIGHTS_TRACKER") or ""
CATALOG = pathlib.Path("_catalog") / "pg_catalog.csv"
LOG = ROOT / "validation" / "logs" / f"worlds_ingest_{_dt.date.today().isoformat()}.json"
CLEARED = ("owned", "licensed", "public_domain", "public domain")


def key(author: str, title: str) -> str:
    k = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())                    # noqa: E731
    return f"{k(author)}|{k(title)}"


class Release:
    """The rights tracker's verdict per work (status, else its suggestion): cleared or not; `fiction` its mark."""

    def __init__(self, tracker=TRACKER, only: bool = False) -> None:
        self.only, self.rows = only, {}
        if tracker and pathlib.Path(tracker).exists():
            for r in csv.DictReader(open(tracker, encoding="utf-8-sig")):
                self.rows[r["key"]] = r

    def row(self, p, info):
        from books import meta
        m = info if info.get("title") else meta(pathlib.Path(p), pathlib.Path(p).parent)
        return self.rows.get(key(m.get("author", ""), m.get("title", "")))

    def cleared(self, p, info) -> bool:
        r = self.row(p, info)
        verdict = ((r or {}).get("status") or (r or {}).get("suggested") or "").lower()
        return verdict.startswith(CLEARED) and "?" not in verdict

    def fiction(self, p) -> str | None:
        r = self.row(p, {})
        return (r or {}).get("fiction") or None


def catalog(root) -> dict:
    path = pathlib.Path(root) / CATALOG
    if not path.exists():
        return {}
    return {r["Text#"]: r["Title"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def classify(entry, dataset, ctx, cat) -> dict:
    """Each book to fiction or to passages: the tracker's mark first, its text otherwise; every reason kept."""
    import collections
    fiction, reality, reasons = [], [], collections.Counter()
    for p in wa.text_files(pathlib.Path(entry.path)):
        mark = ctx.release.fiction(p) if ctx.release else None
        if mark in ("yes", "no"):
            is_f, why = mark == "yes", "the rights tracker's fiction mark"
        else:
            is_f, why = wa.looks_like_fiction(p.read_bytes()[:400_000].decode("utf-8", errors="replace"))[0], "its text"
        (fiction if is_f else reality).append(p)
        reasons[f"{'fiction' if is_f else 'reality'} by {why}"] += 1
    out = {"classified": {"fiction": len(fiction), "reality": len(reality), "why": dict(reasons)}}
    if fiction:
        out["fiction"] = wf.ingest(entry, dataset, ctx, files=fiction, catalog=cat)
    if reality:
        out["passages"] = wa.passages(entry, dataset, ctx, files=reality)
    return out


def run(root, only=None, frames=False, release=False, log=print, plan=None, mentions=None, tracker=TRACKER,
        bulk=None) -> list[dict]:
    """`plan`: the decisions (default: the local manifest's); `mentions`: the names a passage index matches (default:
    the wiki world's); `tracker`: the rights tracker (a missing file: every work cleared, none marked fiction);
    `bulk`: with `frames`, order 0013's bulk reader (`books_bulk_read.run`, handed in by the command line)."""
    root = pathlib.Path(root) / ("release" if release else "")
    root.mkdir(parents=True, exist_ok=True)
    reg = sqlite3.connect(root / "registry.sqlite")
    reg.executescript(SCHEMA)
    ctx = wa.Context(root, reg, Release(tracker, only=release) if (tracker and pathlib.Path(tracker).exists()) or release else None, log)
    ctx._mentions = mentions
    cat, rows = catalog(root), []
    for d in plan if plan is not None else wm.plan():
        if only and only.lower() not in d.dataset.lower():
            continue
        t0, rep = time.time(), {"dataset": d.dataset, "tree": d.tree, "status": d.status, "reason": d.reason, "entries": []}
        for e in d.entries:
            if e.route == "fiction":
                r = wf.ingest(e, d.dataset, ctx, catalog=cat)
            elif e.route == "classify":
                r = classify(e, d.dataset, ctx, cat)
            else:
                r = getattr(wa, e.route)(e, d.dataset, ctx)
            if frames and e.route == "passages" and "frames" in d.reason and r.get("files_added"):
                if bulk is None:
                    raise ValueError("frames needs the bulk reader (bulk=books_bulk_read.run)")
                r["frames"] = {k: v for k, v in bulk(r["store"], pathlib.Path(r["store"]).with_suffix(".bulk"), workers=3,
                                                     log=lambda m: None).items() if k in ("triples", "tiers", "entities")}
            reg.execute("INSERT OR REPLACE INTO worlds VALUES (?,?,?,?,?)", (e.world, e.world.split(":")[0], r.get("store") or
                        (r.get("passages") or r.get("fiction") or {}).get("store"), e.tier, r.get("as_of")))
            rep["entries"].append({"world": e.world, "route": e.route, "tier": e.tier,
                                   **{k: v for k, v in r.items() if k != "store"}})
        rep["seconds"] = round(time.time() - t0, 1)
        reg.execute("INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,datetime('now'))",
                    (d.dataset, d.tree, d.kind, d.status, d.reason, json.dumps(rep)))
        reg.commit()
        log(f"{d.status:8s} {d.dataset[:60]:60s} {rep['seconds']:8.1f}s " + " ".join(
            f"[{x['world']}: {x.get('rows')} rows, +{x.get('files_added', 0)} files]" for x in rep["entries"]))
        rows.append(rep)
    return rows


def main() -> None:
    """The command line lives in ``validation/build/ingest_worlds.py`` (it hands in the bulk reader); this runs it."""
    import runpy
    runpy.run_path(str(ROOT / "validation" / "build" / "ingest_worlds.py"), run_name="__main__")


def _relative(o):
    """Every `store` in a report, at any depth, as its last two parts (`passages/<world>.sqlite`): the worlds folder is
    a local path."""
    if isinstance(o, dict):
        return {k: ("/".join(pathlib.PureWindowsPath(v).parts[-2:]) if k == "store" and isinstance(v, str) else _relative(v))
                for k, v in o.items()}
    return [_relative(v) for v in o] if isinstance(o, list) else o


def public(rows) -> dict:
    """The report as it may be committed: a dataset by its manifest row only, never its name, its tree (a drive) or
    the manifest's reason (reasons name collections); those stay in the local manifest and registry."""
    names = [d.dataset for d in wm.plan()]
    out = [{"row": names.index(r["dataset"]) + 2 if r["dataset"] in names else None, "status": r["status"],
            "seconds": r.get("seconds"), "entries": [_relative(e) for e in r["entries"]]} for r in rows]
    return {"datasets": out, "counts": {s: sum(r["status"] == s for r in rows) for s in ("ingest", "excluded", "queued")}}


if __name__ == "__main__":
    main()
