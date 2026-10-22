"""standin/worlds_manifest.py -- the datasets' manifest, read for ingestion (order 0015).

Two local files, both git-ignored (the per-dataset list names local folders and private sets):
- `docs/worlds_manifest.local.tsv` (Cowork's inventory): one row a dataset -- tree, dataset, files, GB, kind, route
  (in words: "passages + frames", "EXCLUDED (eval)", "order 0014", ...), tier, world, notes;
- `docs/worlds_paths.local.tsv` (Claude Code's): for the datasets this pass ingests, each folder or file with its
  route (passages | fiction | events | quotes | table | classify | mount), world, tier and note.

`plan(manifest, paths)` joins them: every manifest row becomes one decision -- **ingest** (its paths and route),
**excluded** (the manifest says so, or it names a use that is not a world: pretraining, an SFT set, an order's test
data, an external model; the reason is the manifest's own words), or **queued** (a later tranche). Nothing is left
without a decision, which is the order's coverage line.

Wired: STANDALONE (`worlds_ingest`).
"""
from __future__ import annotations

import csv
import pathlib
import re
from typing import NamedTuple

__wiring__ = "STANDALONE"

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "worlds_manifest.local.tsv"
PATHS = ROOT / "docs" / "worlds_paths.local.tsv"
ROUTES = {"passages", "fiction", "events", "quotes", "table", "classify", "mount"}
# what the manifest's route says, in its words: EXCLUDED (and kin) first; else a world route (passages, a fiction /
# quote / event / entity / data / self / era world, a mount, the books' and hdc's orders) is a world to ingest; anything
# else names a use that is not a world (an order's test set, SFT, a routing aid) -- excluded, with its words as reason
NOT_A_WORLD = re.compile(r"EXCLUDED|pretraining|SFT|not a world|classifier data|not used for|duplicate of|voice reference|"
                         r"prior art|later: OCR|a cache|EVAL", re.I)
A_WORLD = re.compile(r"passages|fiction worlds?|quote world|event world|entity world|data world|self world|era worlds|"
                     r"mount|orders? 001[234]", re.I)


class Entry(NamedTuple):
    path: str
    route: str
    world: str
    tier: str
    note: str


class Decision(NamedTuple):
    dataset: str
    tree: str
    kind: str
    status: str                 # ingest | excluded | queued
    reason: str
    entries: tuple              # Entry, for status ingest


def read_tsv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f, delimiter="\t")]


def plan(manifest=MANIFEST, paths=PATHS) -> list[Decision]:
    rows = read_tsv(manifest)
    by: dict[str, list[Entry]] = {}
    for p in read_tsv(paths) if pathlib.Path(paths).exists() else []:
        if p["route"] not in ROUTES:
            raise ValueError(f"unknown route {p['route']!r} for {p['dataset']}")
        by.setdefault(p["dataset"], []).append(Entry(p["path"], p["route"], p["world"], p["tier"], p.get("note", "")))
    out, seen = [], set()
    for r in rows:
        name = r["dataset"]
        if name in seen:                        # the inventory lists a few datasets twice: one decision each
            continue
        seen.add(name)
        if name in by:
            out.append(Decision(name, r["tree"], r["kind"], "ingest", r["route"], tuple(by[name])))
        elif NOT_A_WORLD.search(r["route"]) or NOT_A_WORLD.search(r["kind"]) or not A_WORLD.search(r["route"]):
            out.append(Decision(name, r["tree"], r["kind"], "excluded", r["route"] or r["kind"], ()))
        else:
            out.append(Decision(name, r["tree"], r["kind"], "queued", f"a later tranche (Nick, 2026-10-22): {r['route']}", ()))
    missing = set(by) - seen
    if missing:
        raise ValueError(f"paths name datasets the manifest does not: {sorted(missing)}")
    return out
