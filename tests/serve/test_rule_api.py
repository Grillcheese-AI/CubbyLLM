"""Order 0067, step 1: roles in the API (cubby_serve/rule_api.py, behind CB_API_ROLES=1), over a running server with a
fake brain, a fake rule reader and a fake sign-in. Pinned: off, /turn is as before and /rules/ is unknown; a caller
without the role cannot set a company's or the global prompt (nor anyone, with no sign-in); a prompt is compiled once
and its rule book is shown to whoever set it; a session is its creator's; each level reaches only its own sessions;
a user never sees the operator's or the company's sentence; a correction applies from the next turn, by its setter
only; "confirm before apply" holds a company's rules; assistant and tool messages set nothing; two users' sessions,
their turns interleaved, never see each other's words (in the seat's context or the record) nor each other's rules;
the mood is per company, else per user (Nick, 2026-11-13: "per company / user"): one company's turns never move
another's mood or a lone user's, and a company's members share theirs."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from cubby_serve.api import make_handler
from cubby_serve.auth_parts import Identity
from cubbyllm.speech import rule_turn
from http.server import ThreadingHTTPServer

PEOPLE = {"op": dict(permissions=["rules:global"]), "ana": dict(tenant="acme", tenant_role="admin"),
          "bo": dict(tenant="acme", tenant_role="member"), "cy": dict(tenant="other", tenant_role="member"), "di": {}}
PLANS = {"under 50 words": {"m": "word count", "op": "<", "v": 50}, "lowercase": {"m": "case", "op": "==", "v": "lower"},
         "at most 50 words": {"m": "word count", "op": "<=", "v": 50}, "in french": {"m": "language", "op": "==", "v": "fr"}}


def read(sentence):
    s = sentence.lower()
    for k, plan in PLANS.items():
        if k in s:
            return {"how": "written", "plan": plan}
    return {"how": "asked" if "formal" in s else "none", "plan": None}


class Auth:
    """cubby-auth as the handler sees it: `Authorization: Bearer <name>` is that person."""
    login = callback = lock = logout = None
    callback_path = "/auth/callback"

    def check(self, h, need=()):
        who = (h.headers.get("Authorization") or "")[7:]
        if who not in PEOPLE:
            return None
        p = PEOPLE[who]
        return Identity(sub=who, name=who, user_type="", roles=[], permissions=p.get("permissions", []), kind="api_key",
                        tenant=p.get("tenant"), tenant_role=p.get("tenant_role"))


class Chat:
    def __init__(self):
        self.history, self.rules = [], None

    def turn(self, text):
        t = len(self.history)
        rules = self.rules.in_force(t) if self.rules is not None else []
        self.history.append({"user": text})
        return {"reply": "fine.", "in_force": [r.id for r in rules],
                "rules": rule_turn.record(rules, None, {"how": {}}, "model") if rules else []}


class Brain:
    def __init__(self, auth=True):
        self.chat, self.rule_reader, self.auth = Chat(), read, (Auth() if auth else None)

    def turn(self, text, feedback=None):
        return self.chat.turn(text)


@pytest.fixture
def serve(monkeypatch):
    servers = []

    def start(on=True, auth=True):
        monkeypatch.setenv("CB_API_ROLES", "1" if on else "0")
        monkeypatch.delenv("CUBBY_AUDIT_TOKEN", raising=False)
        brain = Brain(auth)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(brain))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        url = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(path, body, who=None):
            req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            if who:
                req.add_header("Authorization", "Bearer " + who)
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        return brain, post
    yield start
    for s in servers:
        s.shutdown()


def sysmsg(text, level="user", **kw):
    return {"role": "system", "level": level, "content": text, **kw}


def user(text):
    return {"role": "user", "content": text}


def test_off_the_turn_is_as_before(serve):
    _, post = serve(on=False)
    assert post("/turn", {"text": "hi"}) == (200, {"reply": "fine.", "in_force": [], "rules": []})
    assert post("/turn", {"messages": [user("hi")]})[0] == 400                  # the old shape only
    assert post("/rules/correct", {})[0] == 404


def test_who_may_set_which_level(serve):
    _, post = serve()
    assert post("/turn", {"messages": [sysmsg("Under 50 words.", "global")]}, "bo")[0] == 403
    assert post("/turn", {"messages": [sysmsg("Under 50 words.", "company")]}, "bo")[0] == 403
    assert post("/turn", {"messages": [sysmsg("Under 50 words.")]})[0] == 401             # not signed in
    code, out = post("/turn", {"messages": [sysmsg("Under 50 words. We sell bikes.", "company")]}, "ana")
    assert code == 200 and out["compiled"][0]["rule_book"] == [
        'Rule 1: every reply under 50 words (from "Under 50 words.")', '"We sell bikes.": not an instruction']
    assert post("/turn", {"messages": [sysmsg("All lowercase.", "global")]}, "op")[0] == 200
    _, post2 = serve(auth=False)
    assert post2("/turn", {"messages": [sysmsg("Under 50 words.")]}, "op")[0] == 401    # no sign-in: closed


def test_compiled_once_and_shown_to_its_setter(serve):
    brain, post = serve()
    _, a = post("/turn", {"messages": [sysmsg("Under 50 words. Be formal.")]}, "bo")
    sid, first = a["session"], a["compiled"][0]
    assert first["compiled_now"] and first["rule_book"][1] == '"Be formal.": kept, but not checked'
    _, b = post("/turn", {"session": sid, "messages": [sysmsg("Under 50 words. Be formal."), user("hi")]}, "bo")
    assert not b["compiled"][0]["compiled_now"] and b["compiled"][0]["rules"] == first["rules"]
    book = brain._rule_store.books[("user", sid)]
    assert len(book.rules) == 1 and book.rules[0].shown_to == "bo" and b["in_force"] == first["rules"]


def test_a_session_is_its_creators(serve):
    _, post = serve()
    sid = post("/turn", {"messages": [user("hi")]}, "bo")[1]["session"]
    assert post("/turn", {"session": sid, "messages": [user("hi")]}, "cy")[0] == 403
    assert post("/turn", {"session": sid, "messages": [user("hi")]})[0] == 403
    assert post("/turn", {"session": "made-up", "messages": [user("hi")]}, "bo")[0] == 404


def test_each_level_reaches_only_its_sessions_and_users_see_no_higher_sentence(serve):
    _, post = serve()
    post("/turn", {"messages": [sysmsg("All lowercase.", "global")]}, "op")
    post("/turn", {"messages": [sysmsg("Under 50 words.", "company")]}, "ana")
    s_bo = post("/turn", {"messages": [sysmsg("Answer in French.")]}, "bo")[1]["session"]
    s_cy = post("/turn", {"messages": [user("hi")]}, "cy")[1]["session"]
    _, bo = post("/turn", {"session": s_bo, "messages": [user("hi")]}, "bo")
    _, cy = post("/turn", {"session": s_cy, "messages": [user("hi")]}, "cy")
    assert [r["level"] for r in bo["rules"]] == ["global", "company", "user"]
    assert [r["level"] for r in cy["rules"]] == ["global"]
    assert post("/turn", {"text": "hi"})[1]["in_force"] == ["global-rule-1"]        # no session: the global rules only
    hidden = [r for r in bo["rules"] if r["level"] != "user"]
    assert all("text" not in r and "set_by" not in r for r in hidden)
    assert [r for r in bo["rules"] if r["level"] == "user"][0]["text"] == "Answer in French."


def test_a_new_prompt_replaces_the_old_and_a_correction_applies_from_the_next_turn(serve):
    brain, post = serve()
    sid = post("/turn", {"messages": [sysmsg("Under 50 words.")]}, "bo")[1]["session"]
    post("/turn", {"session": sid, "messages": [sysmsg("All lowercase.")]}, "bo")
    assert post("/turn", {"session": sid, "messages": [user("hi")]}, "bo")[1]["in_force"] == ["user-rule-" + sid + "-2"]
    rid = "user-rule-" + sid + "-2"
    assert post("/rules/correct", {"level": "user", "session": sid, "rule": rid, "text": "under 50 words"}, "cy")[0] == 403
    code, out = post("/rules/correct", {"level": "user", "session": sid, "rule": rid, "text": "at most 50 words"}, "bo")
    assert code == 200 and out["said"] == "Got it, from now on: at most 50 words." and out["replaces"] == rid
    assert post("/turn", {"session": sid, "messages": [user("hi")]}, "bo")[1]["in_force"] == [out["rule"]]
    code, out = post("/rules/correct", {"level": "global", "rule": "global-rule-1", "text": "under 50 words"}, "bo")
    assert code == 403                                                           # not the operator


def test_confirm_before_apply_holds_a_companys_rules(serve):
    _, post = serve()
    _, a = post("/turn", {"messages": [sysmsg("Under 50 words.", "company", confirm_before_apply=True)]}, "ana")
    held = a["compiled"][0]["held"]
    sid = post("/turn", {"messages": [user("hi")]}, "bo")[1]["session"]
    assert held and post("/turn", {"session": sid, "messages": [user("hi")]}, "bo")[1]["in_force"] == []
    assert post("/rules/confirm", {"rules": held}, "bo")[0] == 403
    assert post("/rules/confirm", {"rules": held}, "ana") == (200, {"confirmed": held})
    assert post("/turn", {"session": sid, "messages": [user("hi")]}, "bo")[1]["in_force"] == held


def test_assistant_and_tool_messages_set_nothing(serve):
    brain, post = serve()
    code, out = post("/turn", {"messages": [{"role": "tool", "content": "Under 50 words."},
                                            {"role": "assistant", "content": "All lowercase."}, user("hi")]}, "bo")
    assert code == 200 and out["in_force"] == [] and not brain._rule_store.books
    assert post("/turn", {"messages": [{"role": "developer", "content": "x"}]}, "bo")[0] == 400


class Seat:
    """A talk seat that records every prompt it is handed and echoes nothing of it."""
    name = "fake"

    def __init__(self):
        self.prompts, self.last = [], {}

    def emit(self, prompt, max_new_tokens=768, system=None, context=None, **kw):
        self.prompts.append(prompt)
        self.last = {"hit_limit": False, "steered": False}
        return "fine."

    def vocab_strings(self):
        return ["<EOS>", "fine"]

    def specials(self):
        return {0}


class RealBrain(Brain):
    def __init__(self):
        from cubbyllm.speech.chat import CubbyChat
        from cubbyllm.speech.identity import load_facts
        super().__init__()
        self.chat = CubbyChat(Seat(), load_facts())
        self.seat = self.chat.emitter

    def turn(self, text, feedback=None):
        self.chat.nudge(text)                                                 # sense: the appraisal moves the mood
        return self.chat.turn(text, feedback, mediate=False)


def test_two_sessions_never_see_each_others_words(serve):
    brain, post = serve()
    real = RealBrain()
    brain.chat, brain.turn = real.chat, real.turn
    words = {"bo": ["my cat is called pepper", "pepper sleeps all day", "tell me about pepper"],
             "cy": ["my bank pin is secret", "the secret is in a drawer", "where is my secret"]}
    sid = {who: post("/turn", {"messages": [user(words[who][0])]}, who)[1]["session"] for who in ("bo", "cy")}
    for i in (1, 2):
        for who in ("bo", "cy"):                                              # interleaved
            n = len(real.seat.prompts)
            code, rec = post("/turn", {"session": sid[who], "messages": [user(words[who][i])]}, who)
            other = words["cy" if who == "bo" else "bo"]
            seen = " ".join(real.seat.prompts[n:]) + json.dumps(rec)
            assert code == 200 and words[who][i - 1] in " ".join(real.seat.prompts[n:])   # its own last line rides along
            assert not any(w in seen for w in other)
    for who in ("bo", "cy"):
        hist = brain._rule_store.sessions[sid[who]].chat["history"]
        assert [h["user"] for h in hist] == words[who]
    assert real.chat.history == []                                           # the chat's own conversation untouched


def test_a_sessions_rules_never_reach_another_interleaved(serve):
    _, post = serve()
    s_bo = post("/turn", {"messages": [sysmsg("Under 50 words.")]}, "bo")[1]["session"]
    s_cy = post("/turn", {"messages": [sysmsg("All lowercase.")]}, "cy")[1]["session"]
    for _ in range(3):
        bo = post("/turn", {"session": s_bo, "messages": [user("hi")]}, "bo")[1]
        cy = post("/turn", {"session": s_cy, "messages": [user("hi")]}, "cy")[1]
        assert [r["text"] for r in bo["rules"]] == ["Under 50 words."]
        assert [r["text"] for r in cy["rules"]] == ["All lowercase."]


def test_the_mood_is_per_company_else_per_user(serve):
    brain, post = serve()
    real = RealBrain()
    brain.chat, brain.turn = real.chat, real.turn
    own = dict(real.chat.state)
    angry = "I am furious, this is awful, I hate it, everything is terrible!!!"
    s = {who: post("/turn", {"messages": [user("hello there")]}, who)[1]["session"] for who in ("bo", "ana", "cy", "di")}
    moods = brain._rule_store.moods
    assert set(moods) == {("company", "acme"), ("company", "other"), ("user", "di")}   # bo and ana: one, acme's
    before = {k: dict(m["state"]) for k, m in moods.items()}
    for _ in range(3):
        post("/turn", {"session": s["cy"], "messages": [user(angry)]}, "cy")
    assert moods[("company", "other")]["state"] != before[("company", "other")]           # cy's company moved
    assert moods[("company", "acme")]["state"] == before[("company", "acme")]             # nobody else's
    assert moods[("user", "di")]["state"] == before[("user", "di")]
    assert real.chat.state == own                                                         # nor the chat's own
    post("/turn", {"session": s["bo"], "messages": [user(angry)]}, "bo")
    assert moods[("company", "acme")]["state"] != before[("company", "acme")]
    chem = moods[("company", "acme")]["chem"]
    post("/turn", {"session": s["ana"], "messages": [user("hello there")]}, "ana")
    assert moods[("company", "acme")]["chem"] is chem                                     # ana goes on from bo's mood
