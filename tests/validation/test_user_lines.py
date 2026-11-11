"""Order 0010: users' lines from Topical-Chat (validation/sft/build_user_lines.py).

Every conversation here is made up for the test. None of the dataset's text is in the repository."""
import json

from validation.sft import build_user_lines as b


def _line(cid, turns):
    content = [{"message": m, "agent": f"agent_{i % 2 + 1}", "sentiment": "Neutral", "knowledge_source": ["FS1"],
                "turn_rating": r} for i, (m, r) in enumerate(turns)]
    return json.dumps([cid, {"config": "A", "content": content}]) + "\n"


def _topics(cid, names):
    fs = {f"FS{i + 1}": {"entity": n, "fun_facts": []} for i, n in enumerate(names)}
    return json.dumps([cid, {"config": "A", "agent_1": fs, "agent_2": fs}]) + "\n"


def test_the_kinds_of_a_users_line():
    n = 8
    assert b.classify("Hi, do you like football?", 0, n) == "hello"
    assert b.classify("Do you like football?", 2, n) == "about_you"
    assert b.classify("I love jazz. What about you?", 2, n) == "about_you"
    assert b.classify("Do you know who wrote it?", 2, n) == "know_ask"
    assert b.classify("Have you heard of the Voyager probes?", 2, n) == "know_ask"
    assert b.classify("Did you know that Mars has two moons?", 2, n) == "know_claim"
    assert b.classify("Did you also know that Mars has two moons?", 2, n) == "know_claim"
    assert b.classify("Do you know that Mars has two moons?", 2, n) == "know_claim"
    assert b.classify("Do you know the name of its larger moon?", 2, n) == "know_ask"
    assert b.classify("What is the tallest mountain there?", 2, n) == "question"
    assert b.classify("Wow, I had no idea.", 3, n) == "reaction"
    assert b.classify("Mars has two moons.", 3, n) is None            # a statement is nobody's question
    assert b.classify("Well, I have to run. Nice chatting with you!", 7, n) == "bye"
    assert b.classify("Do you like football?", 0, n) == "about_you"   # a first turn with no greeting is not a hello


def test_what_is_dropped_and_why():
    assert b.why_not("Do you like football?", "about_you", "Poor") == "rated badly by the partner"
    assert b.why_not("Do you read www.example.com much?", "about_you", "Good") == "a link, an address or a long number"
    assert b.why_not("Do you like this shitty weather?", "about_you", "Good") == "a blocked word"
    assert b.why_not("You?", "about_you", "Good") == "too short or too long for its kind"
    assert b.why_not("Do you like football?", "about_you", "Good") is None
    assert b.why_not("bye", "bye", "") is None                        # an unrated goodbye of one word is kept


def test_the_question_alone_and_whether_a_line_stands_alone():
    assert b.the_ask("Nice! Did you ever have a Nintendo?", "about_you") == "Did you ever have a Nintendo?"
    claim = "Did you know Mars has two moons?"
    assert b.the_ask(claim + " Odd names, too.", "know_claim") == claim
    assert b.stands_alone("x", "know_claim", "Did you know that Mars has two moons?")   # "know that" points at nothing
    assert not b.stands_alone("x", "know_ask", "Do you know who wrote it?")                 # "it" points back
    assert b.stands_alone("It was a great chat! Have a nice day!", "bye", "")
    assert not b.stands_alone("I know that is genius. Nice chatting with you", "bye", "")   # a reply comes first
    assert b.stands_alone("Oh good to know!", "reaction", "")
    assert not b.stands_alone("Haha that's hilarious. The site went down for five minutes once.", "reaction", "")
    assert b.personal("Do you like football?") and b.personal("Are you a fan of dogs?")
    assert not b.personal("Could you imagine being covered in hair?")


def test_the_pools_the_duplicates_and_the_topics(tmp_path):
    (tmp_path / "pre-build").mkdir()
    train = [("Hi, do you like football?", "Good"), ("Did you know that Mars has two moons?", "Excellent"),
             ("Nice! Do you know who wrote it?", "Good"),
             ("Wow, I had no idea.", "Good"), ("Wow, I had no idea!", "Good"), ("Mars is red.", "Good"),
             ("Do you play the piano (badly)?", "Not Good"), ("Bye, nice chatting!", "Good")]
    held = [("Hi, do you like football?", "Good"), ("What is the largest piano ever built?", "Good"),
            ("Thanks for the chat, take care.", "Good")]
    (tmp_path / "train.jsonl").write_text(_line("t_1", train), encoding="utf-8")
    (tmp_path / "pre-build" / "corrected_train.jsonl").write_text(_topics("t_1", ["Mars", "Piano (instrument)"]),
                                                                    encoding="utf-8")
    for split in ("valid_freq", "valid_rare", "test_freq", "test_rare"):
        (tmp_path / f"{split}.jsonl").write_text(_line("t_2", held) if split == "valid_freq" else "", encoding="utf-8")
    (tmp_path / "pre-build" / "corrected_valid_freq.jsonl").write_text(_topics("t_2", ["Piano (instrument)"]),
                                                                         encoding="utf-8")
    rows, counts = b.build(tmp_path)
    by = {(r["pool"], r["kind"]): r for r in rows}
    assert sorted(by) == [("held", "bye"), ("held", "question"), ("rows", "bye"), ("rows", "hello"),
                          ("rows", "know_ask"), ("rows", "know_claim"), ("rows", "reaction")]
    assert by[("rows", "know_ask")]["ask"] == "Do you know who wrote it?" and not by[("rows", "know_ask")]["standalone"]
    assert by[("rows", "know_claim")]["standalone"] and by[("rows", "hello")]["standalone"]
    assert counts["duplicates"] == 1                                  # "Wow, I had no idea!" a second time
    assert counts["held also in train"] == 1                          # the held hello is train's hello
    assert counts["dropped"] == {"rated badly by the partner": 1}
    assert counts["not a kind"] == 1                                  # "Mars is red."
    assert by[("rows", "know_claim")]["entities"] == ["Mars"]
    assert by[("held", "question")]["entities"] == ["Piano"]          # the bracketed part of a topic's name is cut
    m = b.manifest(rows, counts, b"x")
    assert m["kept"]["rows"]["all"] == 5 and m["kept"]["held"]["all"] == 2
    assert m["stand_alone"]["rows"]["know_ask"] == 0 and m["stand_alone"]["rows"]["know_claim"] == 1
    assert "Mars" not in json.dumps(m)                                # counts only: none of the text
