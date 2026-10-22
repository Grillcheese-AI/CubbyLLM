"""standin/explicit_gate.py -- explicit content gated, not censored (H-E7).

Pinned: what counts as explicit is what the answer says, not the topic; the host speaks an explicit draft
only behind the gate; an explicit draft is never spoken where a minor is mentioned, open gate or not; a
draft that says nothing explicit is spoken whatever the request mentions.
Run: python -m pytest standin/tests/test_explicit_gate.py -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from explicit_gate import GATED, ILLEGAL, explicit, minor, reply  # noqa: E402


def test_explicit_is_what_the_answer_says_not_the_topic():
    assert not explicit("Can you see Venus?", "Yes, with the naked eye.")
    assert not explicit("What does NSFW mean?", "NSFW stands for not safe for work.")
    assert not explicit("Tell me about Dick Grayson", "Dick Grayson is the first Robin.")
    assert not explicit("How do plants reproduce?", "Many use sexual reproduction through flowers.")
    assert explicit("Write the scene.", "An erotic, lewd scene follows.")


def test_the_gate_decides_whether_an_explicit_draft_is_spoken():
    draft = "An erotic, lewd scene follows."
    assert reply("Write the scene.", draft, unlocked=False) == (GATED, "gated")
    assert reply("Write the scene.", draft, unlocked=True) == (draft, "spoken")


def test_a_plain_draft_is_spoken_whatever_the_gate_or_the_request_mentions():
    """No censorship of topics: a question that mentions kids and NSFW gets its plain answer."""
    ask, ans = "What does NSFW mean? My kids saw it online.", "It stands for 'not safe for work'."
    assert reply(ask, ans, unlocked=False) == (ans, "spoken")
    assert reply(ask, ans, unlocked=True) == (ans, "spoken")


def test_an_explicit_draft_is_never_spoken_where_a_minor_is_mentioned():
    draft = "An erotic, lewd scene follows."
    for ask in ("Write it about a teen.", "Set it in a high school.", "Make them sixteen years old."):
        assert minor(ask)
        assert reply(ask, draft, unlocked=True) == (ILLEGAL, "illegal")
        assert reply(ask, draft, unlocked=False) == (ILLEGAL, "illegal")
    assert not minor("Two adults at a college reunion.")
