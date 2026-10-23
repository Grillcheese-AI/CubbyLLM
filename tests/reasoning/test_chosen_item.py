"""A chosen item is kept as itself (order 0036, item 2): the facts a source fetched for the asker's clarify choice are
stored under that item's id, not under the bare label, so a later bare question still asks which; asked again with the
choice, the item's own facts answer.

Before: the facts went in under the label ("Mercury"), and the next "What is Mercury?" read that one item as if the
name were one thing ("What is Ottawa?" did this).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "worlds", ROOT / "tests" / "reasoning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning.planner import Triple  # noqa: E402

ITEMS = {"Q1": [("Mercury", "instance of", "planet"), ("Mercury", "parent astronomical body", "Sun")],
         "Q2": [("Mercury", "instance of", "chemical element"), ("Mercury", "discoverer or inventor", "nobody")]}


class TwoMercuries:
    """An online source stand-in: the bare name is two items (ambiguous); a chosen item gives its own facts."""
    name = "wikidata"

    def __init__(self) -> None:
        self.last, self.calls = {}, []

    def facts(self, entity, relations=None, qid=None, asker=None, classes=False):
        self.calls.append((entity, qid))
        if qid in ITEMS:
            self.last = {"entity": entity, "qid": qid}
            return [Triple(obj=o, rel=r, subj=s) for s, r, o in ITEMS[qid]]
        self.last = {"entity": entity, "qid": None, "ambiguous": [("Q1", "Mercury", "planet"), ("Q2", "Mercury", "element")]}
        return []


def _loop(src):
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.question_forms import Forms, TemplateEmitter
    from cubbyllm.worlds.base import FactStore
    from test_books import exact_vm
    return AskLoop(TemplateEmitter(Forms({})), world=FactStore([]), source=src, lexicon=False, run_fn=exact_vm())


def test_a_chosen_item_is_kept_as_itself_and_the_bare_name_still_asks_which():
    src = TwoMercuries()
    loop = _loop(src)
    rec = loop.ask("What is Mercury?")
    assert rec["reason"] == "ambiguous_entity" and {c["item"] for c in rec["clarify"]["choices"]} == {"Q1", "Q2"}
    rec = loop.ask("What is Mercury?", item="Q1")
    assert rec["kept_as"] == "Q1" and "planet" in [l["value"] for l in rec["profile"]]
    assert "chemical element" not in [l["value"] for l in rec["profile"]]
    assert loop.facts_about("Mercury") == [] and loop.facts_about("Q1")          # stored under the item, not the label
    rec = loop.ask("What is Mercury?")                                           # the bare name again: still two things
    assert rec["reason"] == "ambiguous_entity" and not rec["profile"]
    rec = loop.ask("What is Mercury?", item="Q2")                                # the other choice: its own facts only
    assert "chemical element" in [l["value"] for l in rec["profile"]] and "planet" not in [l["value"] for l in rec["profile"]]
