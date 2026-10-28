"""Order 0043 -- how the answer reads when the source's rule decided a shared name, through the ask loop on a canned
Wikidata source (``reasoning.ask_source_namesakes``; ``CB_SOURCE_NAMESAKES=1``).

Pinned:
- two kinds that both carry the relation are each answered, labelled by one of their own facts, every value tied to
  its own item (a scratch first world per walk: nothing of one lands under the other's name);
- namesakes that agree are said once, with no offer line;
- one fits: its value, then the other namesake named by its kind with an offer;
- off: the strict refusal as before.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation", ROOT / "tests" / "reasoning", ROOT / "tests" / "sources", ROOT / "tests" / "worlds"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from test_ask import STORE, FakeEmitter  # noqa: E402
from test_books import exact_vm  # noqa: E402
from test_search_learn import LookupStore  # noqa: E402
from test_wikidata_namesakes import LABELS, Fake, georgia, hit, item  # noqa: E402


def loop_with(src, plans):
    from cubbyllm.reasoning.ask import AskLoop
    # the time rule off: a canned claim carries no date, and "at one point" is that rule's, not this order's
    return AskLoop(FakeEmitter(plans), world=LookupStore(STORE), source=src, lexicon=False, run_fn=exact_vm(), time_rule=False)


def test_two_kinds_are_each_answered_labelled_and_nothing_lands_under_the_others_name(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    src = Fake(*georgia())
    loop = loop_with(src, {"What is the capital of Georgia?": ("georgia", ["capital"])})
    rec = loop.ask("What is the capital of Georgia?")
    assert rec["verified"] and rec["reply"] == "Georgia (instance: U.S. state): Atlanta; Georgia (instance: country): Tbilisi."
    assert rec["namesakes"]["rule"] == "several fit (source)" and [a["item"] for a in rec["answers"]] == ["Q1", "Q3"]
    assert all(" of Georgia" not in t for t in loop.world.texts)          # the first world holds neither
    assert src.choices == {}                                              # the ledger untouched


def test_namesakes_that_agree_are_said_once_with_no_offer(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"petershausen": [hit("Q21", "Petershausen", "municipality"), hit("Q22", "Petershausen", "quarter")]}
    ents = {"Q21": {"claims": {"P31": [item("Qmun")], "P17": [item("Qde")]}},
            "Q22": {"claims": {"P31": [item("Qquarter")], "P17": [item("Qde")]}}, **LABELS}
    loop = loop_with(Fake(search, ents), {"What is the country of Petershausen?": ("petershausen", ["country"])})
    rec = loop.ask("What is the country of Petershausen?")
    assert rec["verified"] and rec["answer"] == "Germany" and rec["reply"] == "Germany."
    assert rec["namesakes"]["rule"] == "agree (source)" and {a["value"] for a in rec["answers"]} == {"Qde"}


def test_one_fits_its_value_then_the_other_named_by_its_kind(monkeypatch):
    monkeypatch.setenv("CB_SOURCE_NAMESAKES", "1")
    search = {"olmo": [hit("Q5", "Olmo", "city in Italy"), hit("Q6", "Olmo", "singer")]}
    ents = {"Q5": {"claims": {"P31": [item("Qcity")], "P17": [item("Qusa")]}},
            "Q6": {"claims": {"P31": [item("Qhuman")], "P19": [item("Qsmall")]}}, **LABELS}
    loop = loop_with(Fake(search, ents), {"Where was Olmo born?": ("olmo", ["place of birth"])})
    rec = loop.ask("Where was Olmo born?")
    assert rec["verified"] and rec["answer"] == "Smallville"
    assert rec["reply"] == "Smallville. If you meant something else, there is also a city named Olmo. Want to know about it?"
    assert rec["namesakes"]["rule"] == "one fits (source)"


def test_off_the_strict_refusal_stands(monkeypatch):
    monkeypatch.delenv("CB_SOURCE_NAMESAKES", raising=False)
    loop = loop_with(Fake(*georgia()), {"What is the capital of Georgia?": ("georgia", ["capital"])})
    rec = loop.ask("What is the capital of Georgia?")
    assert not rec["verified"] and "namesakes" not in rec
