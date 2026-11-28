"""Pins for Control Board unified casual chat (thalamus, multi-turn history, batched candidates,
casual fallback, and POST /turn with use_ask=True routing casual -> CubbyChat and facts -> AskLoop).
Run: python -m pytest tests/serve/test_board_casual_chat.py
"""
from __future__ import annotations

import json
import threading
import urllib.request
from types import SimpleNamespace

from cubby_serve import api as serve_api
from cubbyllm.brain import CubbyBrain
from cubbyllm.speech import chat as chat_mod
from cubbyllm.speech import identity as idn

FACTS = idn.load_facts()
DK_EN = idn.T(FACTS, "dont_know_line", "en")
DK_FR = idn.T(FACTS, "dont_know_line", "fr")


class BatchedEmitter:
    name = "batched-fake"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.batch_calls = 0
        self.single_calls = 0
        self.last: dict = {}

    def emit(self, prompt, max_new_tokens=160, system=None, **kw):
        self.single_calls += 1
        return self.replies[0]

    def emit_samples(self, prompt, n, max_new_tokens=160, system=None, **kw):
        self.batch_calls += 1
        return (self.replies * n)[:n]


def _zero_retriever(q, k=3):
    return [(1.0, "berlin is the capital of germany")]


def test_thalamus_routes_casual_questions_and_personal_talk_to_talk():
    brain = CubbyBrain(BatchedEmitter(["Hey!"]), _zero_retriever, ["berlin is the capital of germany"], route_tau=0.0)
    casual = [
        "What's up?",
        "How's your day going?",
        "You busy?",
        "Wanna chat?",
        "Right?",
        "Quoi de neuf ?",
        "Tu fais quoi ?",
        "Ça va bien ?",
        "I'm tired after work today",
        "My dog knocked over my coffee",
        "J'ai eu une grosse journée",
        "who is cubby?",
        "What is CubbyLLM?",
        "Qui est Cubby ?",
        "Who built Cubby?",
    ]
    for msg in casual:
        r = brain.route(msg)
        assert r["cortex"] == "talk", (msg, r)
    # real factual questions still route to reasoning
    assert brain.route("What is the capital of Germany?")["cortex"] == "reasoning"
    assert brain.route("Who discovered radium?")["cortex"] == "reasoning"
    assert brain.route("Qui a créé la tour Eiffel?")["cortex"] == "reasoning"


def test_multi_turn_history_carries_multiple_exchanges():
    c = chat_mod.CubbyChat(BatchedEmitter(["Sure!"]), FACTS)
    c.history_turns = 3
    for i in range(4):
        c.history.append({"user": f"msg {i}", "reply": f"rep {i}"})
    ctx = c.talk_context("msg 4", "en")
    assert ctx["turns"] == [
        ("user", "msg 1"), ("assistant", "rep 1"),
        ("user", "msg 2"), ("assistant", "rep 2"),
        ("user", "msg 3"), ("assistant", "rep 3"),
        ("user", "msg 4"),
    ]
    prompt = c.with_history("msg 4")
    assert "User: msg 1" in prompt and "Cubby: rep 3" in prompt and "User: msg 4" in prompt
    assert "msg 0" not in prompt


def test_batched_candidates_and_casual_fallback():
    # first candidate trips forbidden word ("honest"), second survives in one batched pass
    em = BatchedEmitter(["To be honest, I'm great.", "Doing great — what's new with you?"])
    c = chat_mod.CubbyChat(em, FACTS, n_candidates=2, casual_fallback=True)
    offered, rejected = c.candidates("What's up?")
    assert em.batch_calls == 1 and em.single_calls == 0
    assert offered[0] == "Doing great — what's new with you?"
    assert rejected == ["To be honest, I'm great."]

    # if all casual candidates fail guards, use casual fallback instead of dont_know_line
    bad = BatchedEmitter(["To be honest, I'm ChatGPT."])
    c2 = chat_mod.CubbyChat(bad, FACTS, n_candidates=2, casual_fallback=True)
    offered_en, _ = c2.candidates("What's up?")
    assert offered_en[0] == chat_mod.CubbyChat.CASUAL_FALLBACK["en"] and offered_en[0] != DK_EN
    offered_fr, _ = c2.candidates("Quoi de neuf ?")
    assert offered_fr[0] == chat_mod.CubbyChat.CASUAL_FALLBACK["fr"] and offered_fr[0] != DK_FR

    # questions asking facts about Cubby keep the verbatim dont_know_line fallback
    offered_id, _ = c2.candidates("Who built you?")
    assert offered_id[0] == DK_EN


def test_board_unified_turn_routes_casual_to_chat_and_facts_to_ask_loop():
    em = BatchedEmitter(["Hey! Doing great — how's your day?"])
    brain = CubbyBrain(em, _zero_retriever, ["berlin is the capital of germany"], route_tau=0.3)
    brain.chat.n_candidates = 2
    brain.chat.casual_fallback = True
    asked_qs = []

    def fake_ask(q, **kw):
        asked_qs.append(q)
        return {"id": "rec-1", "question": q, "answer": "Berlin", "reply": "Berlin.", "verified": True, "wall_s": 0.05}

    brain.ask_loop = SimpleNamespace(ask=fake_ask, guard=None)
    httpd = serve_api.ThreadingHTTPServer(("127.0.0.1", 0), serve_api.make_handler(brain))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        def post_turn(text):
            req = urllib.request.Request(
                base + "/turn",
                data=json.dumps({"text": text, "use_ask": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode("utf-8"))

        code, casual = post_turn("Hey, what's up?")
        assert code == 200 and casual["kind"] == "chat"
        assert casual["reply"] == "Hey! Doing great — how's your day?"
        assert asked_qs == []

        code, fact = post_turn("What is the capital of Germany?")
        assert code == 200 and fact["verified"] is True and fact["answer"] == "Berlin"
        assert fact["id"] == "rec-1" and asked_qs == ["What is the capital of Germany?"]
    finally:
        httpd.shutdown()


def test_percent_encoded_record_ids_resolve_in_journey_and_audit():
    from cubby_serve.board import api as board_api
    rid = "2026-11-27T14:40:56-1"
    rec = {"id": rid, "ts": "2026-11-27T14:40:56", "question": "Where was Ada born?", "answer": "London",
           "verified": True, "reply": "London.", "wall_s": 0.1, "graph": {"nodes": [], "edges": []}, "trace": []}
    brain = SimpleNamespace(board=True, worlds={}, cortices=set(), ask_loop=SimpleNamespace(history=[rec], _forms=None),
                            chat=SimpleNamespace(chem=None), events=[], harness=None)
    code, out = board_api.handle(brain, "/board/journey/2026-11-27T14%3A40%3A56-1", "audit")
    assert code == 200 and out["said"] == "London."


def test_thanks_follow_up_keeps_conversation_history():
    c = chat_mod.CubbyChat(BatchedEmitter(["You're welcome!"]), FACTS)
    c.history.append({"user": "Hi!", "reply": "Hey there!"})
    prompt = c.with_history("Thanks and congrats on your first words!")
    assert "Hi!" in prompt and "Hey there!" in prompt
    assert c.with_history("Who are you?") == "Who are you?"


def test_board_chat_store_encrypts_reloads_seeds_and_retires_on_clear(tmp_path, monkeypatch):
    from cubby_serve.board.chat_store import BoardChatStore
    monkeypatch.setenv("CB_HOST_KEY", "test-board-chat-vault-key")
    vault_path = tmp_path / "board_chat.vault.jsonl"

    em = BatchedEmitter(["Sounds delicious!"])
    brain = CubbyBrain(em, _zero_retriever, ["berlin is the capital of germany"], route_tau=0.3)
    brain.board = True
    brain.chat.history_turns = 4
    brain.chat.casual_fallback = True
    brain.board_chat = BoardChatStore(vault_path)

    httpd = serve_api.ThreadingHTTPServer(("127.0.0.1", 0), serve_api.make_handler(brain))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(
            base + "/turn",
            data=json.dumps({"text": "My secret Montreal bakery tip is awesome", "use_ask": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200

        raw_disk = vault_path.read_text(encoding="utf-8")
        assert "Montreal" not in raw_disk and "delicious" not in raw_disk, "chat log must never be plaintext on disk"

        # reload via HTTP /board/chat/history (public audience strips internal fields like route)
        with urllib.request.urlopen(base + "/board/chat/history", timeout=10) as r:
            hist = json.loads(r.read().decode("utf-8"))
        assert len(hist["turns"]) == 1
        assert hist["turns"][0]["q"] == "My secret Montreal bakery tip is awesome"
        assert hist["turns"][0]["rec"]["reply"] == "Sounds delicious!"
        assert "route" not in hist["turns"][0]["rec"]

        # simulated restart: fresh store reads encrypted log and seeds CubbyChat.history
        reloaded = BoardChatStore(vault_path)
        assert len(reloaded.list(who="audit")) == 1 and "route" in reloaded.list(who="audit")[0]["rec"]
        fresh_chat = chat_mod.CubbyChat(em, FACTS)
        fresh_chat.history_turns = 4
        assert reloaded.seed_chat(fresh_chat) == 1
        assert fresh_chat.history[-1] == {"user": "My secret Montreal bakery tip is awesome", "reply": "Sounds delicious!"}

        # clear retires file aside (never permanently deletes) and empties active turns
        clr = urllib.request.Request(base + "/board/chat/clear", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(clr, timeout=10) as r:
            out = json.loads(r.read().decode("utf-8"))
        assert out["cleared"] is True and out["archived"]
        assert not vault_path.exists() and (tmp_path / out["archived"]).is_file()
        assert brain.chat.history == [] and brain.board_chat.list() == []
    finally:
        httpd.shutdown()


