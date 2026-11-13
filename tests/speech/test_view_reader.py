"""Order 0067, the second reader's form and grammar: a target parses back to its own segments and program; a segment
the instruction does not hold is refused; every target is complete under the grammar and every prefix of it viable;
text off the grammar (a label not in the set, a segment not copied from the instruction, a line no generator writes)
is not."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders")]
import view_join as VJ  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VRP  # noqa: E402
import view_wording as W  # noqa: E402

from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_reader as R  # noqa: E402


def rows(n=12, seed=0):
    rng, out = random.Random(seed), []
    while len(out) < n:
        reps = VRP.pool(rng, 20)
        parts = [p for p in (VP.sample_atom(rng, [RV.views(t) for t in reps]) for _ in range(rng.randint(1, 3))) if p]
        if not parts:
            continue
        texts = [W.say(p, "en", rng) for p in parts]
        instr, segs, at = "", [], 0
        for i, s in enumerate(texts):
            if i:
                instr += " "
            segs.append({"kind": "check", "part": i, "start": len(instr), "end": len(instr) + len(s)})
            instr += s
        groups = [("one", i) for i in range(len(parts))]
        out.append({"instruction": "Explain the tides. " + instr if rng.random() < 0.5 else instr,
                    "segments": segs, "join": VJ.join(parts, groups)})
        if out[-1]["instruction"].startswith("Explain"):
            for s in segs:
                s["start"] += 19
                s["end"] += 19
            segs.insert(0, {"kind": "task", "start": 0, "end": 18})
    return out


def test_a_target_parses_back_to_itself():
    for r in rows():
        got = R.parse(R.target(r), r["instruction"])
        assert [(s["kind"], s["part"], s["start"], s["end"]) for s in got["segments"]] == \
            [(s["kind"], s.get("part"), s["start"], s["end"]) for s in r["segments"]]
        assert got["program"].strip() == r["join"].strip()


def test_a_segment_the_instruction_does_not_hold_is_refused():
    assert "error" in R.parse("\ncheck 1 | Write a poem." + R.SEP + "program", "Explain the tides.")
    assert "error" in R.parse("\ncheck 1 | Explain the tides.", "Explain the tides.")


def test_the_grammar_holds_every_target_and_refuses_what_is_off_it():
    pytest.importorskip("regex")
    from cubbyllm.speech import view_grammar as G
    for r in rows(8, 1):
        g, t = G.Grammar(r["instruction"]), R.target(r)
        assert g.complete(t)
        assert all(g.viable(t[:k]) for k in range(0, len(t), 7))
    g = G.Grammar("Write fewer than 5 sentences.")
    head = "\ncheck 1 | Write fewer than 5 sentences." + R.SEP
    assert not g.viable("\nverdict 1 | Write")                        # a label not in the set
    assert not g.viable("\ncheck 1 | Write more")                    # not copied from the instruction
    assert not g.viable(head + "program Other implements ISolve {")  # a line no generator writes
    assert not g.viable(head + "program Instruction implements ISolve {\n    function part1(n_sentence: number): number {\n"
                        "        if (n_sentence && 1)")


def test_the_chooser_gives_the_best_viable_token_and_eos_only_at_the_end():
    pytest.importorskip("regex")
    from cubbyllm.speech import view_grammar as G
    r = rows(1, 2)[0]
    t = R.target(r)
    g = G.Grammar(r["instruction"])
    pieces = {1: t[:5], 2: "zzz", 3: t[5:9], 0: ""}
    choose = G.chooser(g, lambda i: pieces[i], eos=0, prompt_len=0)
    assert choose([], [2, 1, 0]) == [1]                  # "zzz" is off the grammar; eos is not yet allowed
    assert choose([1], [0, 2, 3]) == [3]
    full = {9: t}
    done = G.chooser(g, lambda i: full.get(i, ""), eos=0, prompt_len=0)
    assert done([9], [5, 0]) == [0]
