"""The loci reader in the ask loop (standin/loci_reader.py, order 0007) -- with a stand-in reader (no model): the
trunk's reading is spoken only when it equals the store's value; a reading that differs is refused; a miss is
looked up once and read again under the same check; a relation the shard never addressed is a miss; without the
reader the loop is what it was. And the scene is the shard's own layout (the real tokenizer, skipped without it).
Run: python -m pytest standin/tests/test_loci_reader.py -q
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation", ROOT / "standin" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from ask import AskLoop  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402
from test_books import TimedStore, exact_vm  # noqa: E402
from test_search_learn import STORE, DictSource  # noqa: E402

TOKENIZER = os.path.join(os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS"), "base450m", "grilly", "tokenizer.json")


class Reader:
    """Reads back what the store holds for the asked relation -- or what it is told to read."""
    name = "loci"

    def __init__(self, says=None):
        self.says, self.reads = {(normalize(e), r): v for (e, r), v in (says or {}).items()}, []

    def shard_relation(self, relation):
        return {"date of birth": "BORN_ON", "employer": "EMPLOYED_BY"}.get(normalize(relation))

    def read(self, entity, relation, facts):
        self.reads.append((entity, relation, len(facts)))
        held = [o for _s, r, o in facts if self.shard_relation(r) == self.shard_relation(relation)]
        if (normalize(entity), relation) in self.says:
            return {"answer": self.says[(normalize(entity), relation)], "miss": False, "why": None}
        if not held:
            return {"answer": None, "miss": True, "why": "nothing written about it"}
        return {"answer": held[0], "miss": False, "why": None}


def _loop(reader, source, facts=()):
    from test_ask import FakeEmitter
    world = TimedStore(STORE + list(facts) + ["1749 is the date of birth of Goethe"])
    world.index.declare_relation("date of birth")
    plans = {"What is the date of birth of Ada Lovelace?": ("Ada Lovelace", ["date of birth"])}
    return AskLoop(FakeEmitter(plans), world=world, source=source, lexicon=False, run_fn=exact_vm(), loci=reader)


def test_a_reading_equal_to_the_stored_value_is_spoken_with_no_source_called():
    src = DictSource({})
    loop = _loop(Reader(), src, ["1815-12-10 is the date of birth of Ada Lovelace"])
    rec = loop.ask("What is the date of birth of Ada Lovelace?")
    assert rec["verified"] and rec["answer"] == "1815-12-10" and rec["path"] == "loci" and rec["source_calls"] == 0
    assert src.calls == []


def test_a_reading_that_differs_from_the_store_is_refused():
    loop = _loop(Reader({("Ada Lovelace", "date of birth"): "1816"}), DictSource({}),
                 ["1815-12-10 is the date of birth of Ada Lovelace"])
    rec = loop.ask("What is the date of birth of Ada Lovelace?")
    assert not rec["verified"] and rec["answer"] is None and rec["reason"] == "loci_mismatch"


def test_a_miss_is_looked_up_once_then_read_again_under_the_same_check():
    src = DictSource({"ada lovelace": ["1815-12-10 is the date of birth of Ada Lovelace"]})
    reader = Reader()
    loop = _loop(reader, src)
    rec = loop.ask("What is the date of birth of Ada Lovelace?")
    assert rec["verified"] and rec["answer"] == "1815-12-10" and rec["path"] == "loci_after_lookup"
    assert rec["source_calls"] == 1 and [r[2] for r in reader.reads] == [0, 1]       # nothing written, then the fact
    rec = loop.ask("What is the date of birth of Ada Lovelace?")                                    # next time: the trunk, no lookup
    assert rec["verified"] and rec["path"] == "loci" and rec["source_calls"] == 0 and len(src.calls) == 1
    bad = _loop(Reader({("Ada Lovelace", "date of birth"): "1820"}), DictSource({"ada lovelace": ["1815-12-10 is the date of birth of Ada Lovelace"]}))
    rec = bad.ask("What is the date of birth of Ada Lovelace?")
    assert not rec["verified"] and rec["reason"] == "loci_mismatch"                 # learned, misread: never said


def test_without_the_reader_the_loop_is_what_it_was():
    src = DictSource({"ada lovelace": ["1815-12-10 is the date of birth of Ada Lovelace"]})
    loop = _loop(None, src)
    rec = loop.ask("What is the date of birth of Ada Lovelace?")
    assert rec["verified"] and rec["answer"] == "1815-12-10" and "path" not in rec and loop.loci is None


@pytest.mark.skipif(not os.path.exists(TOKENIZER), reason="no bbpe128k tokenizer here")
def test_the_scene_is_the_shards_own_layout():
    from tokenizers import Tokenizer
    import loci_reader as lr
    from make_fact_shard import address, render
    r = object.__new__(lr.LociReader)
    r.tk = Tokenizer.from_file(TOKENIZER)
    r.rel_index = {"BORN_ON": 0, "EMPLOYED_BY": 1}
    r.rel_of = {"date of birth": "BORN_ON", "employer": "EMPLOYED_BY"}
    ids, pos, roles = r.scene("Ada Lovelace", "date of birth", [("Ada Lovelace", "1815-12-10", "BORN_ON")])
    eos = r.tk.token_to_id("</s>")
    full, places, sroles = render(r.tk, [("Ada Lovelace", "BORN_ON", "1815-12-10")], r.rel_index,
                                  [("What is the born on of Ada Lovelace?", "1815-12-10", address("Ada Lovelace", 0))],
                                  "places", eos, 200, lead=True)
    n = len(ids)
    assert list(full[:n]) == ids and list(sroles[:n]) == roles                       # token for token, role for role
    assert [tuple(int(v) for v in places[i]) if roles[i] else None for i in range(n)] == [p if roles[i] else None for i, p in enumerate(pos)]
