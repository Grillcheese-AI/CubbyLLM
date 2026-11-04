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


def test_mkqa_s_item_is_found_by_rule_and_written_only_on_one_link():
    import emitter_wd_mkqa as K
    q = "qui a écrit le petit prince"
    texts = [t for _s, _e, t in K.spans_of(q)]
    assert "le petit prince" in texts and "petit prince" in texts and "le" not in texts   # a function word alone is no span
    spans = [(15, 27, "petit prince"), (12, 27, "le petit prince"), (7, 12, "écrit")]
    items_of = {"petit prince": ["Q25338", "Q999"], "le petit prince": ["Q25338"], "écrit": ["Q5"]}
    why, rec = K.decide(spans, items_of, [("Q25338", "P50")])
    assert why == "written" and rec == {"span": [12, 27], "item": "Q25338", "pid": "P50"}   # the longest span holding it
    assert K.decide(spans, items_of, [("Q25338", "P50"), ("Q25338", "P170")])[0] == "several properties"
    assert K.decide(spans, items_of, [("Q25338", "P50"), ("Q5", "P50")])[0] == "linking items under several spans"
    assert K.decide(spans, items_of, [])[0] == "no link"
