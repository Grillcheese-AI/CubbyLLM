"""validation/exp_view_counted_tables.py: what one answer and one reply add to the tables. A reply the VM could not
check is counted as a reply and nothing else; a false pass is a pass on a reply built to fail, a false fail a refusal
of a reply built to pass; an answer's shape says whether it finished and how its check lines compare with the parts
its program wrote."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("regex")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation")]
import exp_view_counted_tables as T  # noqa: E402

ANSWER = ("\ncheck 1 | Write two lines.\ntask | Be kind.\ncheck 2 | End with a full stop.\n---\n"
          "program Instruction implements ISolve {\n    function part1(n_line: number): number {\n    }\n")


def test_check_lines_and_part_functions_are_counted_on_their_own_side_of_the_separator():
    assert T.lines_and_parts(ANSWER) == (2, 1)
    assert T.lines_and_parts("check 1 | x") == (1, 0)                       # no program yet


def test_a_reply_is_tallied_by_what_it_was_built_for_and_what_the_vm_said():
    t = Counter()
    assert T.tally(t, True, True) is True                                   # built to pass, passed
    assert T.tally(t, False, True) is False                                 # built to pass, refused: a false fail
    assert T.tally(t, True, False) is False                                 # built to fail, passed: a false pass
    assert T.tally(t, False, False) is True                                 # built to fail, refused
    assert T.tally(t, None, False) is False                                 # the VM could not check it
    assert dict(t) == {"replies": 5, "replies built to fail": 3, "checked": 4, "agree": 2,
                       "checked, built to pass": 2, "false fails": 1, "checked, built to fail": 2, "false passes": 1}


def test_an_unfinished_answer_says_where_it_stopped():
    assert T.shape(5, 4, False, 5) == ["check lines = gold parts", "unfinished: more check lines than parts written",
                                       "unfinished: 4 parts written"]
    assert T.shape(6, 6, False, 5) == ["check lines > gold parts", "unfinished: as many check lines as parts written",
                                       "unfinished: 5 or more parts written"]


def test_a_finished_answer_says_whether_its_parts_match_its_check_lines():
    assert T.shape(3, 3, True, 4) == ["check lines < gold parts", "finished: as many parts as check lines",
                                      "finished: 4 parts or fewer written"]
    assert T.shape(5, 4, True, 5) == ["check lines = gold parts", "finished: parts and check lines differ",
                                      "finished: 4 parts or fewer written"]
