"""Order 0045, part C -- the instruction-following gate's rules (``validation/exp_if_follow.py``): each rule is read off
a prompt, passes a hand-written good reply and fails a bad one; the prompts are drawn over the kinds; the score counts
per kind. No model."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import exp_if_follow as F  # noqa: E402

CASES = [
    ("Question: Describe the sea in no more than 20 words. Answer:", {"kind": "words", "max": 20},
     "The sea is wide, cold and loud, and it never rests.", " ".join(["wave"] * 25)),
    ("Question: Write at least 30 words about rain. Answer:", {"kind": "words", "min": 30},
     " ".join(["rain falls softly on the roof"] * 7), "Rain falls."),
    ("Question: Explain gravity in two sentences. Answer:", {"kind": "sentences", "max": 2},
     "Gravity pulls masses together. It keeps us on the ground.", "One. Two. Three sentences here."),
    ("Question: Sum it up in exactly 3 sentences. Answer:", {"kind": "sentences", "exact": 3},
     "First. Second. Third.", "Only one."),
    ("Question: Describe a cat but do not use the word \"fur\". Answer:", {"kind": "forbid", "word": "fur"},
     "A cat is quiet and quick.", "A cat has soft fur."),
    ("Question: Write a line that must include the word \"lantern\". Answer:", {"kind": "require", "word": "lantern"},
     "The lantern swung in the wind.", "The light swung in the wind."),
    ("Question: Answer in French: what is bread? Answer:", {"kind": "language", "lang": "fr"},
     "Le pain est un aliment que les gens mangent dans le monde entier.", "Bread is the food that people eat in the world."),
    ("Question: Name three fruits as a numbered list of 3 items. Answer:", {"kind": "list", "n": 3},
     "1. apple\n2. pear\n3. plum", "apple, pear and plum"),
    ("Question: Give the city and its country in JSON with the keys city and country. Answer:", {"kind": "json", "keys": ["city", "country"]},
     '{"city": "Lyon", "country": "France"}', '{"town": "Lyon"}'),
    ("Question: Explain DNS in plain text, no markdown. Answer:", {"kind": "no_markdown"},
     "DNS turns names into addresses.", "**DNS** turns names into addresses:\n- names\n- addresses"),
    ("Question: Reply in all lowercase: what is up? Answer:", {"kind": "case", "case": "lower"},
     "the sky is up.", "The sky is up."),
    ("Question: Is water wet? Begin with yes or no. Answer:", {"kind": "first", "text": "yes|no"},
     "Yes. Water wets what it touches.", "Water is a liquid, so yes."),
    ("Question: Start your answer with \"Dear reader\" and greet them. Answer:", {"kind": "first", "text": "dear reader"},
     "Dear reader, welcome.", "Welcome, dear reader."),
]


def test_each_rule_is_read_off_the_prompt_passes_the_good_reply_and_fails_the_bad_one():
    for prompt, rule, good, bad in CASES:
        rules = F.rules_of(prompt)
        assert rule in rules, (prompt, rules)
        assert F.check(rule, good), (rule, good)
        assert not F.check(rule, bad), (rule, bad)


def test_a_prompt_with_no_constraint_yields_no_rule():
    assert F.rules_of("Question: Tell me about the moon. Answer:") == []
    assert F.rules_of("Conversation:\nUser: hi\nQuestion: what is a list of things you like? Answer:") == [] or \
        all(r["kind"] != "words" for r in F.rules_of("Question: what is a list of things you like? Answer:"))


def test_prompts_are_drawn_over_the_kinds_and_the_score_counts_per_kind(tmp_path):
    rows = []
    for i in range(6):
        rows.append({"id": f"w{i}", "split": "held", "prompt": f"Question: Say hi in no more than {5 + i} words. Answer:"})
        rows.append({"id": f"j{i}", "split": "held", "prompt": "Question: Give me a JSON with the keys a and b. Answer:"})
    rows.append({"id": "train", "split": "train", "prompt": "Question: Say hi in no more than 3 words. Answer:"})
    rows.append({"id": "free", "split": "held", "prompt": "Question: Say hi. Answer:"})
    p = tmp_path / "chat.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    got = F.prompts(str(p), 4)
    assert len(got) == 4 and {r["rules"][0]["kind"] for r in got} == {"words", "json"}      # round-robin over kinds
    assert all(r["id"] not in ("train", "free") for r in got)
    replies = {r["id"]: ("hi there" if r["id"].startswith("w") else '{"a": 1, "b": 2}') for r in got}
    replies[got[0]["id"]] = "x " * 40 if got[0]["id"].startswith("w") else "not json"
    s = F.score(got, replies)
    assert s["all"]["n"] == 4 and s["all"]["pass"] == 3 and s["answered"] == 4
    assert sum(v["n"] for v in s["by_kind"].values()) == 4
