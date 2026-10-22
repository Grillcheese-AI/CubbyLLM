"""build_same_place -- the wiki world's "located in" facts that are an ancient name in its modern one (order 0015,
addendum 3: "Antium" LOCATED_IN "Anzio" is an identity, not a location).

The evidence is the books' own wording, looked up in their passage index for every LOCATED_IN pair (X, Y):
- **an explicit modern name:** "X, today Y", "X, the present Y", "X, modern Y", "X, now Y", "X (modern Y)" -- an identity;
- **a bare parenthesis,** "X (Y)": an identity only when the wiki world types X as ancient (an "AncientCoastalTown",
  a "RomanCity"); otherwise the parenthesis is as often the region around X ("Ludlow (Shropshire)", "Aurillac
  (Cantal)"), and the fact stays a location;
- **never when Y is a country** (an object of the hdc world's "country"): "Meroe, modern Sudan" and "Catalonia,
  modern Spain" name the modern country around the place, a containment the marker words alike.

Writes `standin/data/out/same_place.tsv` (git-ignored; `world_hygiene` reads it at the mount) and
`validation/logs/build_same_place_<date>.json` (counts only).

    CUBBY_BOOKS_INDEX=<index.sqlite> python standin/data/build_same_place.py
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import pathlib
import re
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.worlds import wikikg as wk  # noqa: E402
from cubbyllm.worlds.hygiene import SAME_PLACE  # noqa: E402

__wiring__ = "STANDALONE"

ANCIENT = re.compile(r"Ancient|Roman|Greek|Etruscan|Phoenician|Hellenistic|Classical|Archaeolog|Former", re.I)
MARKER = r"(?:,\s*(?:the\s+)?(?:present(?:-day)?|modern|now|today)\s+|\s*\((?:modern|now|today|present(?:-day)?)\s+)"
PATTERNS = ["{x} the present {y}", "{x} modern {y}", "{x} now {y}", "{x} today {y}", "{x} present day {y}", "{x} {y}"]


def find(db, x: str, y: str):
    """('marker' | 'parenthesis', the matched words) where a book writes the pair so, else None."""
    for p in PATTERNS:
        try:
            r = db.execute("SELECT p.text FROM fts JOIN paras p ON p.id = fts.rowid WHERE fts MATCH ? LIMIT 3",
                           ('"' + p.format(x=x, y=y) + '"',)).fetchall()
        except sqlite3.OperationalError:
            continue
        for (text,) in r:
            if re.search(re.escape(x) + MARKER + re.escape(y), text):
                return "marker"
            if re.search(re.escape(x) + r"\s*\(" + re.escape(y) + r"\)", text):
                return "parenthesis"
    return None


def main() -> None:
    db = sqlite3.connect(f"file:{os.environ['CUBBY_BOOKS_INDEX']}?mode=ro", uri=True)
    triples = wk.load_triples()
    types = collections.defaultdict(set)
    for s, r, o in triples:
        if r in ("INSTANCE_OF", "IS_A"):
            types[s].add(o)
    pairs = [(s, o) for s, r, o in triples if r == "LOCATED_IN" and s != o]
    from cubbyllm.worlds.hygiene import fold
    hdc = ROOT / "standin" / "data" / "out" / "hdc_world.sqlite"
    countries = {fold(o) for (o,) in sqlite3.connect(f"file:{hdc}?mode=ro", uri=True).execute(
        "SELECT DISTINCT obj FROM facts WHERE rel = 'country'")} if hdc.exists() else set()
    found, c = [], collections.Counter()
    for s, o in pairs:
        x, y = wk.decamel(s), wk.decamel(o)
        if not re.fullmatch(r"[A-Za-z' -]+", x + y):
            continue
        how = find(db, x, y)
        if how is None:
            continue
        c[how] += 1
        if fold(y) in countries:
            c["a country around it: kept a location"] += 1
            continue
        ancient = any(ANCIENT.search(t) for t in types[s])
        if how == "marker" or ancient:
            found.append((s, o, how))
        else:
            c["parenthesis, not ancient-typed: kept a location"] += 1
    SAME_PLACE.write_text("subject\tobject\tevidence\n" + "".join(f"{s}\t{o}\t{h}\n" for s, o, h in found), encoding="utf-8")
    rep = {"located_in_pairs": len(pairs), "written_so_in_the_books": dict(c), "same_place": len(found)}
    log = ROOT / "validation" / "logs" / f"build_same_place_{_dt.date.today().isoformat()}.json"
    log.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
