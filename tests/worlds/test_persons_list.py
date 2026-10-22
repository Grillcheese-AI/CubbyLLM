"""Order 0028's plays pack: a play's list of persons read as stated (the three forms: "daughter to Capulet",
"Hamlet's mother", "his Son" after the line before), checked against the speaker tags; the fiction world answers a
relation as stated, a spouse asked the other way, and refuses a parent asked the other way (the list does not say
which parent). Synthetic play, no source text.
Run: python -m pytest standin/tests/test_persons_list.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from worlds_fiction import persons_facts  # noqa: E402

PLAY = """THE TRAGEDY OF THE TEST

Dramatis Personæ

DUKE ORSO, ruler of Vella.
LEO, his Son.
MARA, wife to Duke Orso.
ADA, daughter to Duke Orso.
TOMAS, Ada’s brother, a soldier.
BRUNO, husband to Ada, a merchant of Vella.
NURSE to Ada.
Servants, Guards.

SCENE. Vella.

ACT I

SCENE I. A hall.

DUKE ORSO.
Welcome, all.

ADA.
Father, I am here.

FIRST GUARD.
My lord.

NURSE.
Come, child.

*** END OF THE PROJECT GUTENBERG EBOOK ***
DAMAGE.
"""


def test_the_list_is_read_as_stated_and_checked_against_the_speakers():
    facts, check = persons_facts(PLAY)
    got = {(s, r, o) for s, r, o, _line in facts}
    assert got == {("Duke Orso", "son", "Leo"), ("Duke Orso", "wife", "Mara"), ("Duke Orso", "daughter", "Ada"),
                   ("Ada", "brother", "Tomas"), ("Ada", "husband", "Bruno")}
    assert check["list"] and check["speakers_not_listed"] == ["FIRST GUARD"]       # the licence's words are not speakers
    assert persons_facts("A novel with no list.\n\nShe walked.")[0] == []


def test_the_world_answers_as_stated_and_refuses_what_it_would_infer(tmp_path):
    from plugins_world import FictionWorld
    from worlds_fiction import FACTS, read_facts, work_names
    db = tmp_path / "fiction.sqlite"
    con = sqlite3.connect(db)
    con.executescript(FACTS)
    con.executemany("INSERT INTO fiction_facts VALUES (?,?,?,?,?,?)",
                    [("The Test Tragedy", s, r, o, "h", line) for s, r, o, line in read_facts(PLAY, work_names(PLAY))])
    con.commit()
    w = FictionWorld("plugin:plays", "plays", db)
    assert w.answer("In The Test Tragedy, who is Duke Orso's daughter?") == ["Ada is the daughter of Duke Orso"]
    assert w.answer("In The Test Tragedy, who is Bruno's wife?") == ["Bruno is the husband of Ada"]   # entailed: asked the other way
    assert w.answer("In The Test Tragedy, who is Ada's father?") == []          # "daughter to Duke Orso" does not say father
    assert w.answer("In The Test Tragedy, who is Tomas's sister?") == []        # nor does "Ada's brother" say sister
    assert w.covers("Who is Ada's father?") == 0.0                              # no work named
