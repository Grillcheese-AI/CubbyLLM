"""Order 0010's blind A/B harness (validation/exp_talk_ab.py): how a voice row's block is read, which pairs go to the
judge, and that a chat turn is rendered and read by the host's own chat path. No model is run."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "validation"))

import exp_talk_ab as ab  # noqa: E402

BLOCK = ("Premise: contradicted\nFacts:\n- Barood — director: Muthyala Subbaiah\n- Barood — country of origin: India\n"
         "- Other Film — country of origin: Chile\nQuestion: So the country of origin of Barood is Australia?\nAnswer:")


def test_a_voice_row_gives_its_triples_its_question_and_the_hosts_verdict_line():
    facts, q, premise = ab._voice_parts(BLOCK)
    assert facts == [("Barood", "director", "Muthyala Subbaiah"), ("Barood", "country of origin", "India"),
                     ("Other Film", "country of origin", "Chile")]
    assert q == "So the country of origin of Barood is Australia?" and premise == "Premise: contradicted"


def test_a_fact_turn_returns_the_asked_relation_and_a_list_turn_every_value_of_the_entity():
    facts, q, _ = ab._voice_parts(BLOCK)
    entity, returned, others = ab._asked(facts, q, whole=False)
    assert (entity, returned) == ("Barood", ["India"])
    assert others == ["Muthyala Subbaiah", "Other Film", "Chile"]          # what a reply must not give for the answer
    entity, returned, others = ab._asked(facts, "Tell me about Barood.", whole=True)
    assert returned == ["Muthyala Subbaiah", "India"] and others == []
    assert ab._asked(facts, "What is the budget of Barood?", whole=False) is None   # a relation the block does not hold
    assert ab._asked(facts, "Tell me about Chile.", whole=True) is None            # no entity of the block is named


def test_only_a_pair_both_arms_speak_and_that_differs_goes_to_the_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "LOGS", str(tmp_path))
    prompts = [{"id": f"ab{i:02d}", "kind": "hello", "group": "chat", "lang": "en", "user": "hi", "facts": []} for i in range(4)]
    with open(tmp_path / (ab.TAG + "_prompts.jsonl"), "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    voice = ["Hey! What's up?", "Same words.", None, "Hello, I'm Cubby."]
    control = ["Hello.", "Same words.", "Hi.", "Hi there."]
    for arm, replies in (("voice", voice), ("control", control)):
        rows = [{"id": p["id"], "draft": r or "", "reply": r, "why": "spoken" if r else "ungrounded: X"} for p, r in zip(prompts, replies)]
        json.dump({"rows": rows}, open(tmp_path / f"{ab.TAG}_{arm}.json", "w", encoding="utf-8"))
    ab.sheet()
    sheet = json.load(open(tmp_path / (ab.TAG + "_sheet.json"), encoding="utf-8"))
    key = json.load(open(tmp_path / (ab.TAG + "_key.json"), encoding="utf-8"))
    assert sorted(p["id"] for p in sheet["pairs"]) == ["ab00", "ab03"]
    assert [x["id"] for x in sheet["auto"]["identical"]] == ["ab01"] and [x["id"] for x in sheet["auto"]["refused"]] == ["ab02"]
    for p in sheet["pairs"]:                                   # the key names the voice's side, and only the key does
        i = int(p["id"][2:])
        assert p[key["voice_is"][p["id"]]] == voice[i] and "voice" not in json.dumps(p)
    blob = json.dumps(key["voice_is"], sort_keys=True).encode()
    assert key["sha256"] == sheet["key_sha256"] == hashlib.sha256(blob).hexdigest()


def test_a_chat_turn_is_rendered_and_read_by_the_hosts_own_chat_path(monkeypatch):
    monkeypatch.setenv("CB_CUBBY_RECORD", "1")
    monkeypatch.setenv("CB_OPENER_VIEW", "1")
    host = ab._host()
    about = host.context("are you a person or a program?", "en")
    assert about["question"] == "are you a person or a program?" and about["facts"]      # Cubby's record as the block
    assert host.context("hi", "en")["facts"] == about["facts"]            # the host reads a greeting as a turn about Cubby
    assert host.context("tell me a joke", "en") == {"turns": [("user", "tell me a joke")]}   # any other turn goes bare
    # the first run's "kill" reply: the host does not let it out
    assert host.guard("I'm not a person or program. I can't provide information about any person, organization, or event.",
                      "are you a person or a program?") is not None
    assert host.guard("Glad I could help!", "ok thank you!") is None
