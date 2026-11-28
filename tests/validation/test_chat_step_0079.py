"""Order 0079's final run, the step tool (`validation/sft/chat_step_0079.py`): the sheet's new checks on the board's
three turns, the test's personas never drawn, a sample that does not depend on the file's order, and a conversation
through the tool to a row in the second talk shape, with its writer, model and session."""
from __future__ import annotations

import io
import json
import os
import random
import urllib.request

import pytest

from validation.sft import chat_batches_0079 as B
from validation.sft import chat_gate_0079 as G
from validation.sft import chat_persona_0079 as L
from validation.sft import chat_rows_0079 as R
from validation.sft import chat_plans_0079 as P
from validation.sft import chat_step_0079 as S

CALM = {k: 0 for k in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")}


def g(reply, user, earlier=(), lang="en", places=()):
    return G.gate(reply, user, list(earlier), CALM, False, lang, places, tuple(earlier))[0]


def test_the_board_turns_are_refused():
    assert "emoji" in g("Glad you are doing okay! \U0001F4A5", "Thanks and congrat on the reply those are some of "
                        "your first words :)")
    assert g("Glad you are doing okay!", "Thanks and congrat on the reply").startswith("answers a mood")
    assert g("Glad I was enjoying your first words!", "thanks",
             earlier=["congrats on your first words"]).startswith("perspective")
    assert g("I always enjoy our conversations.", "hi there").startswith("speaks of a past")
    assert g("Two \U0001F31E and \U0001F31E.", "hey \U0001F642") == "2 emoji in one reply"


def test_what_the_sheet_allows_passes():
    reply, ran = G.gate("Thanks! My first words, and you were here for them.", "congrats on your first words",
                        [], CALM, False)
    assert reply is None and ran == list(G.CHECKS)
    assert g("Mine's blue. What's your favourite?", "what's your favourite colour?") is None   # asking back
    assert g("Glad you're doing well! What's new?", "i'm good thanks") is None                # the user said so
    assert g("You can always ask me again.", "ok thanks") is None
    assert g("One \U0001F642 back.", "hello \U0001F44B") is None


def test_pronouns_in_french_and_the_users_i():
    assert g("Ton humeur est bonne aujourd'hui.", "c'est quoi ton humeur ?", lang="fr").startswith("perspective")
    assert g("Mon humeur est plutôt calme.", "c'est quoi ton humeur ?", lang="fr") is None
    assert g("I just got off work too.", "I just got off work").startswith("perspective")
    assert g("I think so too!", "I think so") is None                                         # a stance, shared
    assert g("Madison sounds lovely.", "i live in Madison, near the lake", places=("Madison",)).startswith("names a place")


def test_test_personas_are_never_drawn_and_order_does_not_matter():
    uuids = [f"{i:032x}" for i in range(400)]
    assert 0 < sum(map(P.is_test, uuids)) < 60                    # about one in sixteen
    items = [(u, (lambda u=u: {"uuid": u})) for u in uuids]
    a = P._bottom(items, 30, 79)
    random.Random(1).shuffle(items)
    b = P._bottom(items, 30, 79)
    assert [p["uuid"] for p in a] == [p["uuid"] for p in b] and not any(P.is_test(p["uuid"]) for p in a)


def test_english_personas_are_joined_from_their_chunks(tmp_path):
    keep = next(f"{i:032x}" for i in range(99) if not P.is_test(f"{i:032x}"))
    test = next(f"{i:032x}" for i in range(99) if P.is_test(f"{i:032x}"))
    chunks = [f"uuid: {keep}\npersona: A baker who loves early mornings\noccupation: bak", "er\ncity: Springfield\n",
              f"uuid: {test}\npersona: someone held for the test\n"]
    f = tmp_path / "personas.jsonl"
    f.write_text("".join(json.dumps({"id": str(i), "text": t}) + "\n" for i, t in enumerate(chunks)), encoding="utf-8")
    got = P.personas_en(str(f), 5, 79)
    assert [p["uuid"] for p in got] == [keep]
    assert got[0]["occupation"] == "baker" and P.seed_of(got[0], random.Random(0))["places"] == ["Springfield"]


@pytest.fixture
def work(tmp_path):
    plan = {"id": "c0079f:00000", "lang": "en", "length": 2, "kind_turn": 1, "kind": "about_cubby",
            "source": "persona life", "persona": {"uuid": "u1", "occupation": "nurse", "age": 41, "hobby": "running",
                                                  "line": "a night-shift nurse", "places": ["Springfield"]}}
    (tmp_path / "plans.json").write_text(json.dumps([plan]), encoding="utf-8")
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "b000.json").write_text(json.dumps({"batch": "b000", "writer": "claude-haiku",
                                                                "ids": [plan["id"]]}), encoding="utf-8")
    return tmp_path


def run(work, *args, capsys):
    code = S.main(["--work", str(work), *args])
    return code, capsys.readouterr().out


def test_a_conversation_through_the_tool(work, capsys, monkeypatch):
    monkeypatch.setattr(R, "MIN_TURNS", 2)                                  # a 2-turn fixture; the real floor is 4
    cid = "c0079f:00000"
    assert run(work, "turn", cid, "--text", "hi", capsys=capsys)[0] == 2                     # the batch is not started
    assert "not" in run(work, "start", "b000", "--model", "claude-sonnet-x", capsys=capsys)[1]
    assert run(work, "start", "b000", "--model", "claude-haiku-test", "--session", "s1", capsys=capsys)[0] == 0
    code, out = run(work, "turn", cid, "--seems", "joy", "--text", "just got off a long shift, finally home",
                    capsys=capsys)
    assert code == 0 and out.startswith("Mood:") and "Facts: none" in out
    assert run(work, "reply", cid, "--text", "Welcome home! That sounds like a long one.", capsys=capsys)[1].startswith("OK")
    assert "THIS user turn" in run(work, "show", cid, capsys=capsys)[1]
    # a turn that needs facts: the host's ask loop answers it (here a stand-in for the server's /ask)
    rec = {"verified": True, "trace": ["Lou Reed — occupation: musician"], "answer": "musician", "id": "r1"}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: io.BytesIO(json.dumps(rec).encode()))
    assert run(work, "turn", cid, "--seems", "curiosity", "--text", "who was Lou Reed?", capsys=capsys)[0] == 2   # no host
    code, out = run(work, "--host", "http://127.0.0.1:1", "turn", cid, "--seems", "curiosity", "--text",
                    "who was Lou Reed?", capsys=capsys)
    assert code == 0 and "- Lou Reed — occupation: musician" in out
    assert run(work, "reply", cid, "--text", "Lou Reed was a musician.", capsys=capsys)[1].startswith("DONE")
    code, out = run(work, "merge", capsys=capsys)
    rows = [json.loads(ln) for ln in (work / "chat_0079_final_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {r["model"] for r in rows} == {"claude-haiku-test"} and rows[1]["kind"] == "about_cubby"
    assert rows[1]["prompt"].endswith("Cubby: Welcome home! That sounds like a long one.\nUser: who was Lou Reed?\nCubby:")
    assert rows[1]["prompt"].startswith("Mood: ") and "\nFacts:\n- Lou Reed" in rows[1]["prompt"]
    conv = json.loads((work / "convs" / "c0079f_00000.json").read_text(encoding="utf-8"))
    conv["turns"][0]["checks"] = conv["turns"][0]["checks"][:-1]                                 # a check that did not run
    (work / "convs" / "c0079f_00000.json").write_text(json.dumps(conv), encoding="utf-8")
    run(work, "merge", capsys=capsys)
    assert len((work / "chat_0079_final_rows.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_refused_reply_is_written_again_then_cut(work, capsys):
    cid = "c0079f:00000"
    run(work, "start", "b000", "--model", "claude-haiku-test", capsys=capsys)
    run(work, "turn", cid, "--seems", "gratitude", "--text", "thanks!", capsys=capsys)
    assert run(work, "reply", cid, "--text", "Glad you're doing okay!", capsys=capsys)[1].startswith("REFUSED")
    assert run(work, "reply", cid, "--text", "Glad you're doing well!", capsys=capsys)[1].startswith("REFUSED")
    assert run(work, "reply", cid, "--text", "Glad you're feeling fine!", capsys=capsys)[1].startswith("CUT")
    assert run(work, "turn", cid, "--text", "hello?", capsys=capsys)[0] == 2
    conv = json.loads((work / "convs" / "c0079f_00000.json").read_text(encoding="utf-8"))
    assert len(conv["turns"][-1]["rejected"]) == 2


def test_a_passing_rewrite_keeps_the_refused_try_as_a_pair(work, capsys, monkeypatch):
    monkeypatch.setattr(R, "MIN_TURNS", 1)
    cid = "c0079f:00000"
    run(work, "start", "b000", "--model", "claude-haiku-test", capsys=capsys)
    run(work, "turn", cid, "--seems", "gratitude", "--text", "thanks!", capsys=capsys)
    run(work, "reply", cid, "--text", "Glad you're doing okay!", capsys=capsys)
    assert run(work, "reply", cid, "--text", "Anytime! What's next on your list?", capsys=capsys)[1].startswith("OK")
    run(work, "merge", capsys=capsys)
    pairs = (work / "chat_0079_final_rows_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(pairs) == 1 and json.loads(pairs[0])["rejected"] == "Glad you're doing okay!"


def test_regate_puts_a_one_try_cut_back_to_pending_and_relength(work, capsys):
    cid = "c0079f:00000"
    run(work, "start", "b000", "--model", "claude-haiku-test", capsys=capsys)
    run(work, "turn", cid, "--seems", "gratitude", "--text", "thanks!", capsys=capsys)
    conv = json.loads((work / "convs" / "c0079f_00000.json").read_text(encoding="utf-8"))
    t = {**conv["pending"], "cubby": "Glad you're doing okay!", "gate": "answers a mood", "checks": ["pilot"]}
    conv.update(turns=[t], pending=None, cut="answers a mood")                 # cut on one try, as before 2026-11-28
    (work / "convs" / "c0079f_00000.json").write_text(json.dumps(conv), encoding="utf-8")
    assert json.loads(run(work, "regate", capsys=capsys)[1]) == {"back to pending": 1}
    conv = json.loads((work / "convs" / "c0079f_00000.json").read_text(encoding="utf-8"))
    assert conv["turns"] == [] and conv["pending"]["user"] == "thanks!" and len(conv["pending"]["rejected"]) == 1
    assert "waits for Cubby's reply" in run(work, "show", cid, capsys=capsys)[1]
    run(work, "relength", capsys=capsys)
    conv = json.loads((work / "convs" / "c0079f_00000.json").read_text(encoding="utf-8"))
    assert 8 <= conv["plan"]["length"] <= 14 and conv["plan"]["second_turn"] >= 1
    assert run(work, "sheet", capsys=capsys)[1].startswith("## Cubby's character sheet")


def test_the_tool_leaves_the_hosts_flags_as_they_were(monkeypatch):
    for k in ("CB_MOOD_REGULATION", "CB_CHAT_ABILITIES"):
        monkeypatch.delenv(k, raising=False)
    g("Hi! What's new?", "hey")
    S.mood_after(["hey", "long day"], ["neutral", "sadness"])
    assert "CB_MOOD_REGULATION" not in os.environ and "CB_CHAT_ABILITIES" not in os.environ


def test_cubby_knows_only_the_conversation():
    seed = {"uuid": "u", "occupation": "", "work": "A food bank volunteer coordinator", "hobby": "baking bread",
            "line": "Dana Ruiz sings in the choir at her church every Sunday and tinkers with home automation."}
    df = L.counts([seed] + [{"line": f"someone who likes walks number {i}"} for i in range(40)])
    said = ["lol he'd talk for an hour. I'm the one who has to get the food bank shifts covered"]
    assert "church" in L.leak("Could you ask a friend from church to share a shift?", said, seed, df, {})
    assert L.leak("Could you ask a friend from church?", said + ["we met at church"], seed, df, {}) is None
    assert L.leak("Your baking must help.", said + ["I bake too"], seed, df, {}) is None      # an inflection of hers
    assert "Dana" in L.leak("Thanks, Dana!", said, seed, df, {})
    assert L.leak("Church bells make a nice sound on a Sunday.", said, seed, df, {}) is None   # not about the user                               # a name she never gave
    assert L.leak("Do you organize your home automation projects?", said, seed, df, {"automation": 0.5}) is None


def test_the_cut_kinds_and_the_brief(work):
    assert R._cut_key("d: ungrounded: Note") == "d: ungrounded" and R._cut_key("h: ability") == "h: ability"
    assert R._cut_key("feeling not backed: I'm glad (joy +0)") == "feeling not backed"
    text = R.brief(work, "b000", "http://127.0.0.1:8775")
    assert "the persona is the user's side" in text and "the user's town (a place in the Facts lines" in text
    assert "its turns before the cut are kept" in text and '--host "http://127.0.0.1:8775"' in text


def test_rebatch_into_fives_outside_the_repo(work, tmp_path, capsys):
    plans = [{**json.loads((work / "plans.json").read_text(encoding="utf-8"))[0], "id": f"c0079f:{i:05d}"} for i in range(12)]
    (work / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (work / "batches" / "b000.json").write_text(json.dumps({"batch": "b000", "writer": "claude-haiku", "model": "m",
                                                            "ids": [p["id"] for p in plans[:2]]}), encoding="utf-8")
    (work / "batches" / "b001.json").write_text(json.dumps({"batch": "b001", "writer": "claude-sonnet",
                                                            "ids": [p["id"] for p in plans[2:]]}), encoding="utf-8")
    with pytest.raises(R.Stop):
        B.cmd_rebatch(work, 5, str(S.ROOT / "tmp_sessions"), None)                             # inside the repo
    got = json.loads(B.cmd_rebatch(work, 5, str(tmp_path / "sessions"), "http://127.0.0.1:8775"))
    assert got["kept_batches"] == ["b000"] and got["new_batches"] == 2 and got["first"] == "b001"
    assert [len(json.loads((work / "batches" / f"b00{k}.json").read_text())["ids"]) for k in (1, 2)] == [5, 5]
    assert list((work / "batches" / "_old").glob("*/b001.json"))
    assert "--host" in (tmp_path / "sessions" / "step.sh").read_text(encoding="utf-8")
