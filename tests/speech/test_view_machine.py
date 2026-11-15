"""Order 0073, step 4 (cubbyllm/speech/view_machine.py): the reader's grammar as a state machine. A gold answer is
taken whole in both arms and both variants and is whole only when the instruction is used up; one refusal each (a
slot outside its set, a column of another view, a comparison its column's type lacks, arm W's ``not`` in arm M, a
tag or a description on a line that is not a check, text out of order, a line that ends on white space or crosses
a line break); with ``own_text`` a value the line's text does not hold is refused and a line cannot end before it
holds what its description used; no choice is the start of another; from every state walked there is a way on to
a whole answer."""
from __future__ import annotations

import random

import pytest

from cubbyllm.speech import view_split as V
from cubbyllm.speech.view_machine import START, Machine

TEXT = ('Write a note. If the reply has at least 3 sentences, use fewer than 40 words.\n'
        'Do not use the word "so". Stay kind.')
W = ('\ntask | Write a note.\ncheck if next n sentence >= 3 | If the reply has at least 3 sentences,\n'
     'check n word < 40 | use fewer than 40 words.\ncheck not any word where lower == "so" | Do not use the word "so".\n'
     'asked | Stay kind.')
M = W.replace("not any word", "none word")
HEAD = "\ntask | Write a note.\n"


@pytest.mark.parametrize("own", [False, True])
def test_a_gold_answer_is_taken_whole_only_when_the_instruction_is_used_up(own):
    for arm, answer in (("w", W), ("m", M), ("w", M)):
        m = Machine(TEXT, arm, own)
        assert m.accepts(answer)
        ends = [i for i, ch in enumerate(answer) if ch == "\n" and i]
        assert all(m.allows(answer[:i]) and not m.accepts(answer[:i]) for i in ends)       # a line done, text left
        assert not m.allows(answer + "\n") and not m.accepts(answer[:-1])                  # nothing after; nothing short
        assert V.parse(answer, TEXT, arm, own_text=own).get("lines")
    assert not Machine(TEXT, "m", own).allows(W)                                           # arm M has no `not`


@pytest.mark.parametrize("bad", [
    "\nnonx",                                                # a kind outside its set
    HEAD + "check ax",                                       # a quantifier outside its set
    HEAD + "check n words",                                  # a view outside its set
    HEAD + "check any sentence where lo",                    # a column of another view
    HEAD + "check any sentence where first_word >",          # a comparison a string does not have
    HEAD + 'check any sentence where n_word == "',           # a string where the column is a number
    HEAD + "check n sentence >= 12345",                      # a number past four digits
    HEAD + "check n sentence >= 03",                         # a number written two ways
    HEAD + "check at sentence 0",                            # a position starts at 1
    HEAD + "check either p",                                 # a tag outside its set
    "\ntask i",                                              # a tag on a line that is not a check
    "\ntask n",                                              # a description on a line that is not a check
    "\ntask | Write a note.\nnone | u",                      # text out of order
    "\ntask | Write a \n",                                   # a line that ends on white space
    HEAD + "check n sentence >= 3 | If the reply has at least 3 sentences, use fewer than 40 words.D",   # a line break crossed
])
def test_one_refusal_each(bad):
    for arm in V.ARMS:
        m = Machine(TEXT, arm)
        assert m.allows(bad[:-1]) and not m.allows(bad), bad


def test_with_own_text_a_description_says_only_what_its_line_holds():
    free, own = Machine(TEXT, "m"), Machine(TEXT, "m", own_text=True)
    line = HEAD + "check if next n sentence >= "
    assert free.allows(line + "14 ") and not own.allows(line + "14") and own.allows(line + "3 ")
    assert own.allows(line + "1 ") and own.allows(line + "0 ")                             # 0 and 1 always
    whole = line + "3 | If the reply has at least"
    assert free.allows(whole + "\n") and not own.allows(whole + "\n") and own.allows(whole + " 3\n")   # it must reach its 3
    far = HEAD + "check n sentence >= 40 | If the reply has at least 3 sentences,"
    assert not own.allows(far + "\n") and own.allows(far + " use fewer than 40\n")          # and a later 40, if it takes it
    named = HEAD + "check n paragraph >= 3 | If the reply has at least 3"                   # "paragraph" is not named:
    assert own.allows(named + "\n") and own.allows(named + " sentenc") and not own.allows(named + " sentence")   # stop before a view is
    assert free.allows(named + " sentences,\n") and not own.allows(HEAD + "check n paragraph >= 4")   # so no 40 for it
    at = HEAD + "check if next n sentence >= 3 | If the reply has at least 3 sentences,\ncheck n word < 40 | use fewer than 40 words.\n"
    said, other = at + 'check none word where lower == "so" | Do not use the word', at + 'check none word where lower == "then" | Do not use the word'
    assert own.allows(said + "\n") and own.allows(said + ' "so".\n')                        # free while the line quotes nothing
    assert not own.allows(said + ' "s\n')                                                   # no line ends inside a quote
    assert own.allows(other + "\n") and not own.allows(other + " ") and free.allows(other + ' "so".\n')   # "then" is not quoted


def walk(m: Machine, rng, steps: int = 300) -> list:
    """A random walk by characters: at each state a character drawn from the ones the machine takes."""
    seen, st = [START], START
    letters = sorted(set(TEXT) | set('\n |"\\0123456789<>=!_abcdefghijklmnopqrstuvwxyz'))
    for _ in range(steps):
        nxt = [s for s in (m.step(st, c) for c in letters) if s is not None]
        if not nxt:
            break
        st = rng.choice(nxt)
        seen.append(st)
    return seen


def way_on(m: Machine, st: tuple, limit: int = 4000) -> bool:
    """Whether the plainest way on from a state reaches a whole answer: finish the choice in hand, close a number
    or a string, copy the text while it may be copied, end the line."""
    for _ in range(limit):
        if m.accepting(st):
            return True
        if st[0] == "pick":
            ch = min(o for o in m._options(st[1]) if o.startswith(st[2]))[len(st[2])]
        elif st[0] == "text":
            ch = TEXT[st[2]] if st[2] < len(TEXT) and m.step(st, TEXT[st[2]]) else "\n"
        elif st[0] == "num":
            ch = " " if st[2] else "1"
        else:
            ch = {"start": "\n", "quote": '"', "str": '"'}[st[0]]
        st = m.step(st, ch)
        if st is None:
            return False
    return False


@pytest.mark.parametrize("arm,own", [("m", False), ("w", False), ("m", True), ("w", True)])
def test_no_choice_starts_another_and_every_state_walked_has_a_way_on(arm, own):
    m, rng = Machine(TEXT, arm, own), random.Random(7)
    states = sorted({s for _ in range(40) for s in walk(m, rng)}, key=repr)
    assert len(states) > 500 and all(way_on(m, s) for s in rng.sample(states, 300))
    for key, opts in m._opts.items():
        assert opts, key
        assert not any(a != b and b.startswith(a) for a in opts for b in opts), key
