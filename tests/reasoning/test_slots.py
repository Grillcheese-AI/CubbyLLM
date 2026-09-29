"""H-E15 pins: the emitter never copies a name or a number. The host extracts the spans, annotates the question,
checks the program (unknown slot, copied literal, unused spans) and fills it; the training harvest converts to
the slot form without a name index. No model, no VM."""
from __future__ import annotations

import pytest

from cubbyllm.reasoning.slots import SlotEmitter, SlotRefused, SpanIndex, extract, to_slots

NAMES = ["Siege of Vienna", "Vienna", "Treaty of Karlowitz", "Iridomyrmex bigi", "Ottoman Empire"]


def test_extract_takes_the_longest_match_left_to_right_and_numbers_by_position():
    idx = SpanIndex(NAMES)
    t = extract("What led to the Siege of Vienna in 1683, and how did the Ottoman Empire respond?", idx)
    assert [(s.id, s.text) for s in t.spans] == [("$E1", "Siege of Vienna"), ("$E2", "Ottoman Empire"), ("$N1", "1683")]
    assert t.annotate() == "What led to the [E1: Siege of Vienna] in [N1: 1683], and how did the [E2: Ottoman Empire] respond?"
    t2 = extract("Vienna, then the siege of vienna", idx)
    assert [s.text for s in t2.spans] == ["Vienna", "siege of vienna"], "case-insensitive, and 'Vienna' alone when the longer name is not there"


def test_referents_come_from_the_context_graph_and_are_not_in_the_text():
    t = extract("and what came of it?", SpanIndex(NAMES), referents={"it": "Siege of Vienna"})
    assert [(s.id, s.text, s.kind) for s in t.spans] == [("$R1", "Siege of Vienna", "R")]
    assert t.annotate() == "and what came of it?  [R1 = Siege of Vienna]"
    assert t.fill('bind frame, SEED, "$R1";') == 'bind frame, SEED, "Siege of Vienna";'


PROGRAM = '''use vsa;
program CotPlan implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, SEED, "%s";
        bind frame, HOP1, "parent taxon";
        return recover(frame, SEED);
    }
}'''


def test_check_refuses_an_unknown_slot_and_a_copied_name_and_reports_unused_spans():
    idx = SpanIndex(NAMES)
    t = extract("What is the parent taxon of Iridomyrmex bigi, unlike Vienna?", idx)
    ok = t.check(PROGRAM % "$E1")
    assert ok.ok and ok.referenced == ["$E1"] and ok.unused_entities == ["$E2"] and ok.copied == []
    unknown = t.check(PROGRAM % "$E7")
    assert not unknown.ok and "unknown slot $E7" in unknown.reason
    copied = t.check(PROGRAM % "Iridomyrmex bigi")
    assert not copied.ok and "copied instead of a slot" in copied.reason and copied.copied == ["Iridomyrmex bigi"]
    relation_literal = t.check(PROGRAM % "$E1")          # "parent taxon" is a relation, not a span: allowed
    assert relation_literal.ok
    assert t.fill(PROGRAM % "$E1") == PROGRAM % "Iridomyrmex bigi"


def test_numbers_are_slots_too_and_a_copied_number_is_refused():
    t = extract("Natalia sold 48 clips in April and half as many in May. How many in all?")
    assert [(s.id, s.text) for s in t.spans] == [("$N1", "48")]
    good = "create s0 : quantity;\n assign s0 = $N1;\n div s0, 2;\n create s1 : quantity;\n assign s1 = $N1;\n add s1, s0;"
    assert t.check(good).ok
    assert t.fill(good).count("48") == 2 and "$N1" not in t.fill(good)
    bad = good.replace("$N1", "48", 1)
    v = t.check(bad)
    assert not v.ok and v.copied == ["48"]
    assert t.check("assign s0 = 2;").ok, "a constant the question does not carry (half = 2) is the model's to write"


def test_the_harvest_converts_without_a_name_index():
    prompt = "What is the component of the instance of the parent taxon of iridomyrmex bigi?"
    ann, prog, table, stats = to_slots(prompt, PROGRAM % "iridomyrmex bigi")
    assert ann == "What is the component of the instance of the parent taxon of [E1: iridomyrmex bigi]?"
    assert 'bind frame, SEED, "$E1";' in prog and 'bind frame, HOP1, "parent taxon";' in prog
    assert stats == {"entities_slotted": 1, "entities_kept": 0, "numbers_slotted": 0, "spans": 1}
    assert table.fill(prog) == PROGRAM % "iridomyrmex bigi", "fill is the exact inverse on the harvest"
    # a literal the prompt does not carry stays as it is, and is counted
    ann2, prog2, _, stats2 = to_slots("Who wrote it?", PROGRAM % "Hamlet")
    assert 'bind frame, SEED, "Hamlet";' in prog2 and stats2["entities_kept"] == 1 and ann2 == "Who wrote it?"
    arith = "Natalia sold 48 clips and then 24 more."
    _, p3, _, s3 = to_slots(arith, "assign s0 = 48;\n add s0, 24;\n div s0, 2;")
    assert p3 == "assign s0 = $N1;\n add s0, $N2;\n div s0, 2;" and s3["numbers_slotted"] == 2


class FakeInner:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def emit(self, prompt, max_new_tokens=768, **kw):
        self.seen.append(prompt)
        return self.reply


def test_slot_emitter_annotates_checks_and_fills_and_refuses_a_copy():
    idx = SpanIndex(NAMES)
    inner = FakeInner(PROGRAM % "$E1")
    e = SlotEmitter(inner, idx)
    out = e.emit("What is the parent taxon of Iridomyrmex bigi?")
    assert inner.seen == ["What is the parent taxon of [E1: Iridomyrmex bigi]?"]
    assert out == PROGRAM % "Iridomyrmex bigi" and e.last["verdict"].ok
    copier = SlotEmitter(FakeInner(PROGRAM % "Iridomyrmex bigi"), idx)
    with pytest.raises(SlotRefused) as ex:
        copier.emit("What is the parent taxon of Iridomyrmex bigi?")
    assert "copied" in str(ex.value) and not copier.last["verdict"].ok
    wrong = SlotEmitter(FakeInner(PROGRAM % "$E2"), idx)
    with pytest.raises(SlotRefused):
        wrong.emit("What is the parent taxon of Iridomyrmex bigi?")      # only $E1 exists: unknown slot
