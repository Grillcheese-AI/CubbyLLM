"""Order 0048's data (``validation/emitter_wd_data.py``): a record is v12e's one-hop plan with the item as a slot and
HOP1 the property's label in the question's language; the audit catches a 0046 dev or test question or item, and a
label in the wrong language; the gold rules are R3's (the set's own property; Mintaka's links, forward only)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import emitter_wd_data as W  # noqa: E402

LX = {"label": {"P36": {"en": "capital", "fr": "capitale"}, "P17": {"en": "country"}}}


def test_a_record_is_the_one_hop_plan_with_the_slot_and_the_label():
    q = "What is the capital of France?"
    r = W.record("wd-x-1", "mintaka/en", q, "Q142", [23, 29], "P36", "capital")
    assert r["prompt"] == "Question:\nWhat is the capital of [E1: France]?\nProgram:\n"
    assert 'bind frame, SEED, "$E1";' in r["program"] and 'bind frame, HOP1, "capital";' in r["program"]
    assert 'bind frame, SEED, "France";' in r["reference"] and r["spans"][0]["text"] == "France"
    assert r["task"] == "wd" and r["lang"] == "en" and r["item"] == "Q142" and r["split"] in ("train", "val")
    v12e = ('use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n'
            '        create frame: number;\n        bind frame, SEED, "$E1";\n        bind frame, HOP1, "given name";\n'
            '        return recover(frame, SEED);\n    }\n}\n')
    assert W.program("$E1", "given name") == v12e                                   # v12e's text, byte for byte


def test_the_audit_catches_eval_questions_items_and_a_label_in_another_language():
    ok = W.record("wd-a", "mintaka/fr", "Quelle est la capitale de la France ?", "Q142", [29, 35], "P36", "capitale")
    en_label = W.record("wd-b", "mintaka/fr", "Quelle est la capitale de la France ?", "Q142", [29, 35], "P36", "capital")
    held = W.record("wd-c", "mintaka/en", "What is the capital of France?", "Q90", [23, 29], "P36", "capital")
    bad = W.audit([ok, en_label, held], {"what is the capital of france"}, {"Q90"}, LX)
    assert bad == {"eval_question": ["wd-c"], "eval_item": ["wd-c"], "label_not_in_language": ["wd-b"]}
    assert W.audit([ok], set(), set(), LX) == {"eval_question": [], "eval_item": [], "label_not_in_language": []}
