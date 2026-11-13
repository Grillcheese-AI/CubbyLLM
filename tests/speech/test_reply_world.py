"""Order 0067, step 2: the reply world. Each measurement on hand-written replies, and for each of exp_if_follow's ten
kinds a rule written over the statements alone agrees with that measure's own check on every reply here."""
from __future__ import annotations

import importlib.util
import pathlib

from cubbyllm.speech import reply_world as rw

_spec = importlib.util.spec_from_file_location("_if", pathlib.Path(__file__).parents[2] / "validation" / "exp_if_follow.py")
IF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IF)


def vals(text, rel):
    return [v for _s, r, v in rw.measure(text, [rel]) if r == rel]


def test_counts_and_forms():
    t = "Paris is the capital. It sits on the Seine! Is it big?"
    assert vals(t, "word count") == ["12"] and vals(t, "sentence count") == ["3"]
    assert vals("- a\n- b\n3. c\nplain", "list item count") == ["3"] and vals("a\n\nb\n", "line count") == ["2"]
    assert vals("all quiet here", "case") == ["lower"] and vals("ALL LOUD", "case") == ["upper"] and vals("Mixed", "case") == ["mixed"]
    assert vals("Yes, it is. The answer is two.", "opening")[:2] == ["yes", "yes it"]
    assert vals('{"name": "Ada", "year": 1815}', "json") == ["yes"] and vals('{"name": "Ada", "year": 1815}', "json key") == ["name", "year"]
    assert vals("{not json", "json") == ["no"] and vals("**bold**", "markdown") == ["yes"] and vals("plain words", "markdown") == ["no"]
    assert "seine" in vals(t, "word") and vals(t, "last word") == ["big"]


def test_language_and_the_french_measurements():
    assert vals("La ville est belle et le fleuve est grand.", "language") == ["fr"]
    assert vals("The city is old and the river is wide.", "language") == ["en"]
    assert vals("Qwerty zxcv.", "language") == ["none"]
    fr = "Ça va, tu as reçu le cœur de l'été en 2024 ?"
    assert vals(fr, "cedilla count") == ["2"] and vals(fr, "oe ligature count") == ["1"] and vals(fr, "digit count") == ["4"]
    assert vals(fr, "accented letter count") == ["2"] and vals(fr, "form of address") == ["tu"]
    assert vals("Vous avez raison.", "form of address") == ["vous"] and vals("Il pleut.", "form of address") == ["none"]


def test_statements_read_like_facts():
    assert "2 is the sentence count of the reply" in rw.statements("One. Two.")
    assert all(s.endswith(" of the reply") for s in rw.statements("Hello there."))


def held(rule, reply):
    """exp_if_follow's ten kinds, as a rule over the reply world's statements only (the shape step 3's plans take)."""
    m = {}
    for _s, r, v in rw.measure(reply):
        m.setdefault(r, []).append(v)
    one = lambda r: m[r][0]                                   # noqa: E731
    k = rule["kind"]
    if k == "words":
        return int(one("word count")) <= rule["max"] if "max" in rule else int(one("word count")) >= rule["min"]
    if k == "sentences":
        n = int(one("sentence count"))
        return n == rule["exact"] if "exact" in rule else 0 < n <= rule["max"]
    if k in ("forbid", "require"):
        has = rule["word"].lower() in m.get("word", [])
        return not has if k == "forbid" else has
    if k == "language":
        return one("language") == rule["lang"]
    if k == "list":
        n = int(one("list item count"))
        return n >= 2 and ("n" not in rule or n == rule["n"])
    if k == "json":
        return one("json") == "yes" and all(key in m.get("json key", []) for key in rule.get("keys", []))
    if k == "no_markdown":
        return one("markdown") == "no"
    if k == "case":
        return one("case") == rule["case"]
    if k == "first":
        return any(alt.strip() in m.get("opening", []) for alt in rule["text"].split("|"))
    raise KeyError(k)


REPLIES = ["Yes. Paris is the capital of France.", "no", "La réponse est simple et claire.", "- one\n- two\n- three",
           '{"name": "Ada", "born": 1815}', "**Bold** claim with `code`.", "all lowercase words here", "LOUD REPLY",
           "The answer uses the word river twice: river.", "", "Sure! Here is a long reply with many words in it, "
           "more than ten for sure, and it goes on."]
RULES = [{"kind": "words", "max": 10}, {"kind": "words", "min": 5}, {"kind": "sentences", "exact": 2},
         {"kind": "sentences", "max": 1}, {"kind": "forbid", "word": "river"}, {"kind": "require", "word": "paris"},
         {"kind": "language", "lang": "fr"}, {"kind": "language", "lang": "en"}, {"kind": "list"}, {"kind": "list", "n": 3},
         {"kind": "json", "keys": ["name", "born"]}, {"kind": "json", "keys": []}, {"kind": "no_markdown"},
         {"kind": "case", "case": "lower"}, {"kind": "case", "case": "upper"}, {"kind": "first", "text": "yes|no"},
         {"kind": "first", "text": "sure"}]


def test_a_rule_over_the_statements_agrees_with_the_measures_own_check():
    for rule in RULES:
        for reply in REPLIES:
            assert held(rule, reply) == IF.check(rule, reply), (rule, reply)


def test_the_seen_kinds_measurements():
    t = "Intro\n*** \nSECTION 1 a *hi* **bold** <<The Title>>\n***\n\n***\nSECTION 2 end"
    assert vals(t, "paragraph count") == ["3"] and vals(t, "empty paragraph count") == ["1"]
    assert vals(t, "highlight count") == ["2"] and vals(t, "title count") == ["1"] and vals("<< >>", "title count") == ["0"]
    two = "First answer.\n******\nSecond answer."
    assert vals(two, "response count") == ["2"] and vals(two, "distinct response count") == ["2"]
    assert vals("same ****** same", "distinct response count") == ["1"]
    assert vals("a ****** ****** b", "empty response count") == ["1"]


def arg(text, rel, a):
    return [v for _s, _r, v in rw.measure_arg(text, rel, a)]


def test_relations_with_an_argument():
    assert arg("The river, the River! rivers", "count of the word", "river") == ["2"]
    assert arg("Banana", "count of the letter", "a") == ["3"] and arg("a,b,c", "count of the character", ",") == ["2"]
    assert arg('  "Quoted reply"  ', "starts with", '"') == ["yes"] and arg('"Quoted reply"', "ends with", '"') == ["yes"]
    assert arg("Repeat this first. Then more.", "starts with", "repeat THIS first") == ["yes"]
    assert arg("Intro SECTION 1 a SECTION 2 b", "section count", "SECTION") == ["2"]
    assert rw.measure_arg("x", "count of the word", "x") == [("the reply", 'count of the word "x"', "1")]


def test_json_only_is_the_whole_reply():
    assert vals('{"a": 1}', "json only") == ["yes"] and vals('```json\n{"a": 1}\n```', "json only") == ["yes"]
    assert vals('Here it is: {"a": 1}', "json only") == ["no"] and vals('Here it is: {"a": 1}', "json") == ["yes"]
    assert vals("[1, 2]", "json only") == ["yes"] and vals("plain", "json only") == ["no"]
