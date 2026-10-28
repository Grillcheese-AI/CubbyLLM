"""Order 0046 -- the fame set's rules, on a synthetic Wikidata5M (``validation/retrieval_sets/fame_set.py``), each found by
a hand spot check of 50 rows (2026-10-28): two items are of one kind by a shared P31 or one direct P279 step; a
discriminator's words must tell the items apart (a shared label, or one held whole by another's, does not); a place
another item is not ruled out of does not (its known places at or above it, or a place of the same name holding it);
an item with nothing recorded cannot be ruled out; the committed set matches its frozen SHA-256."""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_sets import fame_set as F  # noqa: E402


def world(triples):
    facts = collections.defaultdict(lambda: collections.defaultdict(set))
    for h, r, t in triples:
        facts[h][r].add(t)
    return facts


LABELS = {"Q900": "United States", "Q901": "Michigan", "Q902": "Illinois", "Q903": "Kent County", "Q904": "Kent County",
          "Q905": "Boone County", "Q906": "New York", "Q907": "Rye", "Q908": "Rye", "Q909": "Colorado",
          "Q910": "Pueblo County", "Q911": "New York City", "Q912": "Brooklyn", "Q913": "Italy", "Q914": "Kingdom of Italy",
          "Q915": "politician", "Q916": "actor", "Q917": "Delaware"}


def test_two_items_are_of_one_kind_by_a_shared_class_or_one_subclass_step():
    f = world([("A", "P31", "Q918"), ("B", "P31", "Q918"), ("C", "P31", "Q919"), ("Q919", "P279", "Q918"),
               ("D", "P31", "Q920"), ("Q920", "P279", "Q921"), ("Q918", "P279", "Q921")])
    assert F.one_kind("A", "B", f) and F.one_kind("A", "C", f)
    assert not F.one_kind("A", "D", f)                                       # siblings under a parent: two steps


def test_a_label_shared_or_held_whole_by_anothers_tells_nothing_apart():
    f = world([("G", "P31", "Q5"), ("G", "P106", "Q915"), ("O", "P31", "Q5"), ("O", "P106", "Q916"),
               ("Cal1", "P131", "Q903"), ("Cal2", "P131", "Q904"), ("Q903", "P131", "Q901"), ("Q904", "P131", "Q917")])
    assert F.discriminator("G", ["O"], "P19", f, LABELS) == ("P106", "Q915")
    assert F.discriminator("Cal1", ["Cal2"], "P17", f, LABELS) is None       # both "Kent County"
    assert F.overlaps("italy", {"kingdom of italy"}) and not F.overlaps("italy", {"italian peninsula"})


def test_a_place_another_item_is_not_ruled_out_of_tells_nothing_apart():
    f = world([("S1", "P131", "Q912"), ("Q912", "P131", "Q911"), ("Q911", "P131", "Q906"), ("S2", "P131", "Q911"),
               ("H1", "P131", "Q908"), ("Q908", "P131", "Q910"), ("Q910", "P131", "Q909"), ("H2", "P131", "Q906"),
               ("Q907", "P131", "Q906"), ("V1", "P131", "Q905"), ("Q905", "P131", "Q902"), ("V2", "P131", "Q901")])
    F.HOMONYMS.clear()
    F.HOMONYMS.update({"rye": ["Q907", "Q908"], "brooklyn": ["Q912"], "illinois": ["Q902"], "boone county": ["Q905"]})
    assert F.discriminator("S1", ["S2"], "P17", f, LABELS) is None           # S2 is in New York City, which holds Brooklyn
    assert F.discriminator("H1", ["H2"], "P17", f, LABELS) is None           # another Rye lies in New York, where H2 is
    assert F.discriminator("V1", ["V2"], "P17", f, LABELS) == ("P131", "Q905")
    assert F.places_up("V1", f) == {"Q905", "Q902"}


def test_an_item_with_nothing_recorded_cannot_be_ruled_out():
    f = world([("G", "P31", "Q5"), ("G", "P106", "Q915"), ("O", "P31", "Q5")])
    assert F.discriminator("G", ["O"], "P19", f, LABELS) is None
    g = world([("A", "P131", "Q905"), ("Q905", "P131", "Q902"), ("B", "P31", "Q918")])
    assert F.discriminator("A", ["B"], "P17", g, LABELS) is None             # B is nowhere on record


def test_the_committed_fame_set_matches_its_frozen_checksum():
    frozen = json.load(open(ROOT / "validation" / "retrieval_sets" / "frozen.json", encoding="utf-8"))["fame_set"]
    data = (ROOT / "validation" / "retrieval_sets" / "fame_set.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == frozen["sha256"]
    rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
    bare = [r for r in rows if r["form"] == "bare"]
    assert len(bare) == 1000 and all(len(r["items"]) >= 2 for r in bare)
    assert all(r["gold_item"] in r["items"] for r in rows if r["form"] == "qualified")
