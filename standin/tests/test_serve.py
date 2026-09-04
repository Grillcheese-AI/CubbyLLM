"""Pins for the serve brain (standin/serve.py): routing, fact expansion,
ground check, live learning, plugin mount, and the VM-mediated turn (live
tests skip without the exe). Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation"), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import identity as idn  # noqa: E402
import serve as sv  # noqa: E402
from worlds import FactStore  # noqa: E402

F = idn.load_facts()
DK_EN = idn.T(F, "dont_know_line", "en")

STORE = [
    "Fnordovia is the homeland of Zorblax the Painter",
    "Quuxville is the capital of Fnordovia",
    "berlin is the capital of germany",
    "mars is the planet of olympus mons",
]

EVT_PROGRAM = ("program Ev implements ISolve {\n"
               "    public function solve(mention: str): str {\n"
               "        create said : str;\n        assign said = \"ok\";\n"
               "        remember said;\n        return said;\n    }\n}\n")


def overlap_retriever(query, k):
    """Deterministic token-overlap retriever for tests (read-only world)."""
    q = set(query.lower().split())
    scored = sorted(((len(q & set(f.lower().split())) / (len(q) or 1), f) for f in STORE), reverse=True)
    return [(s, f) for s, f in scored[:k] if s > 0]


class ChainEmitter:
    """A correct fake trunk: event-writes get a minimal Evt program; task
    prompts chain through the offered facts (top fact's object -> the fact
    that mentions it -> its object; single fact -> its own object)."""
    name = "fake-chain"

    def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
        if prompt.startswith("Record this as an event"):
            return EVT_PROGRAM
        if "Facts:" not in prompt:                       # no facts offered: an honest trunk has nothing to bind
            return ""
        assert prefix.startswith("use vsa"), "task turns must pin the CotChain style"
        facts = [l[2:] for l in prompt.split("Facts:\n", 1)[1].splitlines() if l.startswith("- ")]
        from cubbyllm.reasoning.planner import parse_fact
        first = parse_fact(facts[0])
        try:
            nxt = next(f for f in facts[1:] if first.obj.lower() in f.lower())
            answer = parse_fact(nxt).obj
        except StopIteration:
            answer = first.obj
        return ("use vsa;\n\nprogram CotChain implements ISolve {\n"
                "    public function solve(mention: str): str {\n"
                "        create frame: number;\n"
                f'        bind frame, H1_ANSWER, "{answer}";\n'
                "        return recover(frame, H1_ANSWER);\n    }\n}\n")


class ChatterEmitter:
    name = "fake-chat"

    def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
        if prompt.startswith("Record this as an event"):
            return EVT_PROGRAM
        return "I'm Cubby — a small model that thinks big. Glad to help!"


def test_gather_facts_expands_through_the_top_facts_object():
    s = sv.CubbyServe(ChatterEmitter(), overlap_retriever, STORE)
    facts = [f for _, f in s.reason.gather_facts("What is the homeland of Zorblax the Painter?",
                                                 overlap_retriever)]
    assert facts[0] == STORE[0]
    assert STORE[1] in facts, "the expansion hop must pull the capital-of-Fnordovia fact"


def test_clean_strips_think_and_fences():
    assert sv._clean("<think>hm</think>\n```\nprogram A {}\n```") == "program A {}\n"


def test_clean_cuts_echoed_header_and_rejects_babble():
    echoed = "# Which country?\n- a fact echoed as a bullet\nuse vsa;\nprogram A {}"
    assert sv._clean(echoed) == "use vsa;\nprogram A {}\n"
    assert sv._clean("# The country of the currency of the country is the United States.") == ""


def test_memory_detect_reads_remember_prefixes_and_bare_template_facts():
    d = sv.MemoryCortex.detect
    assert d("Remember that Quuxville is the capital of Fnordovia.") == \
        "Quuxville is the capital of Fnordovia"
    assert d("retiens que Quuxville is the capital of Fnordovia") == \
        "Quuxville is the capital of Fnordovia"
    assert d("mars is the planet of olympus mons") == "mars is the planet of olympus mons"
    assert d("What is the capital of Fnordovia?") is None
    assert d("Hello there!") is None


def _exe_or_skip():
    from cubbyllm.bridges import cubelang_client as cc
    try:
        return cc.find_cubelang_exe()
    except cc.CubelangNotFound as e:
        pytest.skip(f"cubelang.exe not found ({e})")


def test_live_task_turn_grounded_answer_is_spoken_through_the_vm():
    _exe_or_skip()
    s = sv.CubbyServe(ChainEmitter(), overlap_retriever, STORE, route_tau=0.2)
    rec = s.turn("What is the capital of the homeland of Zorblax the Painter?")
    assert rec["kind"] == "task"
    assert rec["task"]["grounded"] is True and rec["task"]["vm_answer"] == "Quuxville"
    assert rec["reply"] == "Quuxville" and rec["offered"] == ["Quuxville", DK_EN]
    assert rec["emotion"] and rec["route"]["cortex"] == "reasoning"


def test_live_ungrounded_task_answer_degrades_to_the_dont_know_line():
    _exe_or_skip()

    class Confabulator(ChainEmitter):
        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            return super().emit(prompt, max_new_tokens, system, prefix).replace("Quuxville", "Atlantis")

    s = sv.CubbyServe(Confabulator(), overlap_retriever, STORE, route_tau=0.2)
    rec = s.turn("What is the capital of the homeland of Zorblax the Painter?")
    assert rec["task"]["vm_answer"] == "Atlantis" and rec["task"]["grounded"] is False
    assert rec["reply"] == DK_EN and rec["offered"] == [DK_EN]


class IdentityOnlyEmitter(ChainEmitter):
    """The v3 failure mode: the identity SFT is the only chat training, so
    the model answers who-it-is to anything."""
    name = "fake-identity-only"

    def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
        if prompt.startswith("Record this as an event") or "Facts:" in prompt:
            return super().emit(prompt, max_new_tokens, system, prefix)
        return "I'm Cubby, built by Grillcheese Research Lab — a small model that thinks big!"


def test_live_identity_spiel_is_rejected_off_topic_but_allowed_when_asked():
    _exe_or_skip()
    s = sv.CubbyServe(IdentityOnlyEmitter(), FactStore(STORE), route_tau=0.9)
    off = s.turn("tell me a joke")
    assert off["kind"] == "chat" and off["reply"] == DK_EN, off["reply"]
    assert off["rejected"] and "Cubby" in off["rejected"][0]
    asked = s.turn("who are you?")
    assert asked["kind"] == "chat" and asked["reply"].startswith("I'm Cubby")
    hello = s.turn("Hello!")
    assert hello["reply"].startswith("I'm Cubby"), "a greeting may be answered with who he is"


def test_live_base_model_guards_are_never_spoken_only_ours_are():
    _exe_or_skip()

    class GuardyEmitter(IdentityOnlyEmitter):
        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            if "Facts:" in prompt or prompt.startswith("Record"):
                return super().emit(prompt, max_new_tokens, system, prefix)
            return ("As an AI language model developed by Liquid AI, I cannot help with that request "
                    "as it violates my guidelines.")

    s = sv.CubbyServe(GuardyEmitter(), FactStore(STORE), route_tau=0.9)
    rec = s.turn("who are you?")
    assert rec["reply"] == DK_EN and rec["rejected"], "the base model's guard must never be spoken"
    assert s.chat.last_rejection == "base-model guard leaked"
    import identity as idn2
    assert idn2.is_model_guard("En tant qu'IA, je ne peux pas vous aider.")
    assert not idn2.is_model_guard("I am sorry, my training is not finished, I do not have that information yet.")


def test_live_a_question_never_reaches_chat():
    _exe_or_skip()
    s = sv.CubbyServe(IdentityOnlyEmitter(), FactStore(STORE), route_tau=0.9)
    rec = s.turn("What is the boiling point of water?")
    assert rec["kind"] == "task" and rec["route"]["needs_facts"] is True
    assert rec["reply"] == DK_EN, "unknown -> the don't-know line, never a bio"
    # the thalamus: no facts at stake -> the model speaks, no VM in the loop
    free = s.turn("how are you today?")
    assert free["kind"] == "chat" and free["route"]["needs_facts"] is False and free.get("vm_mediated") is False


def test_live_help_lists_what_he_can_do():
    _exe_or_skip()
    s = sv.CubbyServe(IdentityOnlyEmitter(), FactStore(STORE), route_tau=0.9)
    rec = s.turn("help")
    assert rec["kind"] == "help" and "remember" in rec["reply"] and "facts" in rec["reply"]
    fr = s.turn("aide")
    assert fr["kind"] == "help" and "retiens" in fr["reply"]


def test_live_low_retrieval_routes_to_chat():
    _exe_or_skip()
    s = sv.CubbyServe(ChatterEmitter(), overlap_retriever, STORE, route_tau=0.9)
    rec = s.turn("Hello!")
    assert rec["kind"] == "chat" and "Cubby" in rec["reply"]


def test_live_learn_then_answer_the_remember_forever_loop():
    _exe_or_skip()
    world = FactStore(["mars is the planet of olympus mons"])
    s = sv.CubbyServe(ChainEmitter(), world, route_tau=0.2)
    rec = s.turn("Remember that Quuxville is the capital of Fnordovia.")
    assert rec["kind"] == "learn" and rec["learn"]["accepted"] and rec["learn"]["vm_written"]
    assert rec["reply"].startswith("Got it") and "Quuxville" in rec["reply"]
    rec2 = s.turn("What is the capital of Fnordovia?")
    assert rec2["kind"] == "task" and rec2["reply"] == "Quuxville", \
        "a fact learned at turn N must be answerable at turn N+1"
    # contradiction: gated, the stored fact wins and is cited
    rec3 = s.turn("Remember that Atlantis is the capital of Fnordovia.")
    assert rec3["learn"]["accepted"] is False and rec3["learn"]["reason"] == "contradiction"
    assert "Quuxville is the capital of Fnordovia" in rec3["reply"]
    # duplicate
    rec4 = s.turn("Remember that Quuxville is the capital of Fnordovia.")
    assert rec4["learn"]["reason"] == "duplicate"


def test_live_plugin_mounts_worlds_and_cortex_mindforge_style():
    _exe_or_skip()

    class GamePlugin:
        name = "cubbyverse-toy"

        def __init__(self):
            self.seen = []

        def worlds(self):
            return {"game": FactStore(["cubbyman is the hero of cubbyverse"], name="game")}

        def cortices(self):
            class GameCortex:
                @staticmethod
                def match(text):
                    return 1.0 if "play" in text.lower() else 0.0

                @staticmethod
                def handle(text):
                    return {"offered": ["Let's play — loading the cubbyverse!"], "meta": {"scene": "lobby"}}
            return {"game": GameCortex()}

        def on_turn(self, rec):
            self.seen.append(rec["kind"])

    plug = GamePlugin()
    s = sv.CubbyServe(ChainEmitter(), FactStore(STORE), route_tau=0.2)
    s.mount(plug)
    assert "game" in s.worlds and "game" in s.cortices
    rec = s.turn("Can we play?")
    assert rec["kind"] == "plugin:game" and rec["reply"].startswith("Let's play")
    assert plug.seen == ["plugin:game"], "mounted plugins observe finished turns"
    rec2 = s.turn("who is the hero of cubbyverse?")
    assert rec2["route"].get("world") == "game", "routing must reach plugin worlds"


def test_live_api_surface_turn_state_worlds():
    _exe_or_skip()
    from serve_api import serve_http
    s = sv.CubbyServe(ChainEmitter(), FactStore(STORE), route_tau=0.2)
    httpd = serve_http(s, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
                return json.load(r)
        assert get("/health")["ok"] is True
        st = get("/state")
        assert {"dopamine", "cortisol", "emotion"} <= set(st)
        assert get("/worlds")["worlds"]["facts"] == len(STORE)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/turn",
                                     data=json.dumps({"text": "What is the capital of the homeland "
                                                              "of Zorblax the Painter?"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            rec = json.load(r)
        assert rec["reply"] == "Quuxville" and rec["emotion"]
        # the console feed: the turn's reasoning/actions are visible events
        ev = get("/events?since=0")
        kinds = [e["kind"] for e in ev["events"]]
        for k in ("user", "sense", "route", "walk", "gate", "speak"):
            assert k in kinds, f"console must show {k}: {kinds}"
        assert get(f"/events?since={ev['next']}")["events"] == []
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30) as r:
            page = r.read().decode("utf-8")
        assert "Cubby Console" in page
    finally:
        httpd.shutdown()


def test_live_a_question_about_something_else_is_not_a_game_command():
    """Owner's chat: 'how would you like to have a new plugin to explore the web?' went to pacman."""
    from verse import CubbyMan
    man = CubbyMan()
    assert man.match("explore for 30") == 1.0 and man.match("explore") == 0.6 and man.match("let's play") == 0.6
    assert man.match("how would you like to have a new plugin to explore the web?") == 0.0
    assert man.match("what is the cubbyverse?") == 0.0 and man.match("explore the cubbyverse") == 0.6, "the world's name alone is not a command"

    class Cmd:                                           # a plugin that only knows bare commands
        @staticmethod
        def match(text):
            return 0.6 if "explore" in text.lower() else 0.0

        @staticmethod
        def handle(text):
            return {"offered": ["exploring"], "meta": {}}
    s = sv.CubbyServe(ChainEmitter(), FactStore(STORE), route_tau=0.2)
    s.cortices["game"] = Cmd()
    assert s.route("explore a bit")["cortex"] == "game", "a statement is the plugin's"
    r = s.route("how would you like to have a new plugin to explore the web?")
    assert r["cortex"] == "talk" and r["needs_facts"] is False, f"a question to Cubby is talk, got {r}"
    assert s.route("would you like a new plugin?")["cortex"] == "talk"
    assert s.route("what is the cubbyverse")["cortex"] == "talk", "the world is an identity fact: Cubby answers himself"


def test_live_a_multi_hop_answer_must_carry_the_final_relation():
    """v6 self-test (2026-09-03): the one wrong chain was a 2-hop question whose walk exhausted at hop 2
    while retrieval was confident on hop 1's fact; the emitter bound hop 1's object and the ground check took it."""
    _exe_or_skip()
    s = sv.CubbyServe(ChainEmitter(), overlap_retriever, STORE, route_tau=0.2)
    rec = s.turn("What is the capital of the planet of olympus mons?")   # hop 1 is in the store, "capital of mars" is not
    assert rec["kind"] == "task" and rec["route"]["cortex"] == "reasoning"
    assert rec["task"]["vm_answer"] == "mars", "the emitter binds hop 1's object (the partial walk offered it)"
    assert rec["task"]["grounded"] is False, "a 'planet' fact cannot ground a 'capital' answer"
    assert rec["reply"] == DK_EN and rec["offered"] == [DK_EN]
    assert "gate_relation" in [e.get("kind") for e in list(s.events)]
    # a single-hop question with a confident flat hit still answers
    rec1 = s.turn("What is the capital of germany?")
    assert rec1["reply"] == "berlin"


def test_live_the_context_selects_the_adapter_and_callers_hold_one_emitter():
    """v8, fitted to the architecture (theta = f(c)): every cortex calls ONE emitter with a context role;
    a ContextualEmitter resolves "programs" / "talk" to adapters; a fact question never reaches the talk
    adapter, a no-facts turn never reaches the program adapter; one model serves both roles unchanged."""
    _exe_or_skip()
    from emitter import ContextualEmitter

    class Counting(ChainEmitter):
        def __init__(self, name):
            self.name, self.calls = name, []

        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            self.calls.append((kw.get("context"), prompt[:30]))
            return super().emit(prompt, max_new_tokens, system, prefix)

    class Talker(Counting):
        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            self.calls.append((kw.get("context"), prompt[:30]))
            return "Doing well, thanks for asking! What can I do for you?"

    prog, talk = Counting("prog"), Talker("talk")
    ctx = ContextualEmitter({"programs": prog, "talk": talk}, default="programs")
    assert ctx.is_split and ctx.resolve("talk") == "talk" and ctx.resolve({"role": "talk"}) == "talk" \
        and ctx.resolve(None) == "programs" and ctx.resolve("mystery") == "programs"
    s = sv.CubbyServe(ctx, overlap_retriever, STORE, route_tau=0.2)
    assert s.emitter is ctx and s.chat.emitter is ctx and s.reason.emitter is ctx and s.memory.emitter is ctx
    rec = s.turn("What is the capital of the homeland of Zorblax the Painter?")
    assert rec["kind"] == "task" and rec["reply"] == "Quuxville"
    assert prog.calls and prog.calls[0][0] == "programs" and not talk.calls
    rec2 = s.turn("how are you today?")
    assert rec2["kind"] == "chat" and talk.calls and len(prog.calls) == 1, "the no-facts turn never touched the program adapter"
    assert isinstance(talk.calls[0][0], dict) and talk.calls[0][0]["role"] == "talk" and "dopamine" in talk.calls[0][0]["state"], \
        "the hormonal state rides in the context (the trunk's c)"
    assert ctx.calls == {"programs": 1, "talk": 1}
    # the pre-v8 shape still works: two handles fold into one contextual emitter
    p2, t2 = Counting("p2"), Talker("t2")
    s2 = sv.CubbyServe(p2, overlap_retriever, STORE, route_tau=0.2, talk_emitter=t2)
    assert isinstance(s2.emitter, ContextualEmitter) and s2.emitter.is_split
    # one model: both roles, no ContextualEmitter needed
    solo = sv.CubbyServe(Counting("only"), overlap_retriever, STORE, route_tau=0.2)
    assert not isinstance(solo.emitter, ContextualEmitter) and solo.turn("how are you today?")["kind"] == "chat"


def test_the_adapter_bank_grows_and_counts_what_it_cannot_serve():
    """The automatic half of "specialization, automatic": a context that asks for a specialist the bank
    lacks falls back to the default AND is counted (the need monitor); a promoted specialist registers at
    runtime and takes its role over; retirement keeps history, never deletes."""
    from emitter import ContextualEmitter

    class Fake:
        def __init__(self, name):
            self.name, self.calls = name, 0

        def emit(self, prompt, max_new_tokens=768, system=None, prefix="", **kw):
            self.calls += 1
            return f"{self.name}:{prompt}"

    base, talk = Fake("base"), Fake("talk")
    bank = ContextualEmitter({"programs": base, "talk": talk}, default="programs")
    assert bank.emit("q", context={"role": "pacman", "world": "pacman"}) == "base:q"   # no pacman specialist yet
    assert bank.emit("q", context="pacman") == "base:q" and bank.fallbacks == {"pacman": 2}
    u = bank.usage()
    assert u["fallback_rate"] == 1.0 and u["roles"] == ["programs", "talk"]
    spec = Fake("pacman-v1")
    bank.register("pacman", spec)                       # a spawned specialist, promoted
    assert bank.emit("q", context="pacman") == "pacman-v1:q" and spec.calls == 1 and bank.fallbacks == {"pacman": 2}
    assert bank.name == "ctx[programs=base,talk=talk,pacman=pacman-v1]"
    bank.register("pacman", Fake("pacman-v2"))           # replaced: v1 retired into history, not lost
    assert [h for h in bank.history if h[2] == "retired"] == [("pacman", "pacman-v1", "retired")]
    bank.unregister("pacman")
    assert "pacman" not in bank.adapters and bank.emit("q", context="pacman") == "base:q"
    import pytest
    with pytest.raises(ValueError):
        bank.unregister("programs")

    # the maturation ladder: a spawned adapter waits in SHADOW — comparable, never routed — until promoted
    cand = Fake("history-v1")
    bank.shadow("history", cand)
    assert bank.stage("history") == "candidate" and bank.emit("q", context="history") == "base:q" and cand.calls == 0
    assert bank.emit_candidate("history", "q") == "history-v1:q" and cand.calls == 1, "the promote step can compare it"
    bank.promote("history")
    assert bank.stage("history") == "routed" and bank.emit("q", context="history") == "history-v1:q"
    bank.shadow("history", Fake("history-v2")); bank.reject("history")
    assert bank.stage("history") == "routed" and [h for h in bank.history if h[2] == "rejected"] == [("history", "history-v2", "rejected")]
    assert bank.idle_roles(min_calls=2) == ["history", "talk"], "history used once, talk never: both below the prune line; the default is never listed"
    assert bank.usage()["candidates"] == [] and "history" in bank.usage()["roles"]



def test_a_definite_no_facts_read_is_never_overridden_by_retrieval():
    """Live misroute (2026-09-03): 'do you know how to write code?' is a capability question about Cubby, but the
    store resembled it (an 'audio album ... write you a song' fact at 0.377 > tau 0.319) and 'confident retrieval
    engages reasoning' sent it to the VM, which answered 'audio album'. Retrieval may pull only residual small talk."""
    s = sv.CubbyServe(ChainEmitter(), overlap_retriever, STORE, route_tau=0.0)   # tau 0: every hit is 'confident'
    assert s.needs_facts("do you know how to write code?") == (False, "about Cubby himself")
    assert s.needs_facts("can you write a poem about berlin?")[0] is False
    assert s.needs_facts("What is the capital of germany?") == (True, "the fact grammar parses it")   # parse comes first
    for q in ("do you know how to write code?", "tell me a joke about berlin", "peux-tu écrire du code ?"):
        assert s.route(q)["cortex"] == "talk", q
    r = s.route("berlin berlin berlin")                    # residual small talk that the store resembles: retrieval may still engage
    assert r["cortex"] == "reasoning" and r["why"] == "retrieval"


def test_live_an_unparsed_question_needs_a_confident_hit_to_ground():
    """Live misroute (2026-09-03): 'awesome, what are the things you learned?' is a wh-question the grammar does not
    parse; the flat fallback offered a 0.137 hit, the emitter bound its object ('lists') and the ground check took
    it — an unparsed question has no relation to hold the answer to. Now the grounding fact must be a hit for the
    question at >= tau_ret; a parsed question keeps the relation gate and still answers."""
    _exe_or_skip()
    s = sv.CubbyServe(ChainEmitter(), overlap_retriever, STORE, route_tau=0.2)
    rec = s.turn("what are the things you learned about berlin")   # 'what' -> a question about the world; overlap with the berlin fact is 2/8
    assert rec["kind"] == "task" and rec["route"]["cortex"] == "reasoning"
    assert rec["task"]["vm_answer"] is not None, "the emitter bound an offered object (that is the leak)"
    assert rec["task"]["grounded"] is False and rec["reply"] == DK_EN
    assert "gate_retrieval" in [e.get("kind") for e in list(s.events)]
    assert s.turn("What is the capital of germany?")["reply"] == "berlin"
