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

    def emit(self, prompt, max_new_tokens=768, system=None, prefix=""):
        if prompt.startswith("Record this as an event"):
            return EVT_PROGRAM
        assert "Facts:" in prompt, "task prompts must carry the Facts block"
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

    def emit(self, prompt, max_new_tokens=768, system=None, prefix=""):
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
        def emit(self, prompt, max_new_tokens=768, system=None, prefix=""):
            return super().emit(prompt, max_new_tokens, system, prefix).replace("Quuxville", "Atlantis")

    s = sv.CubbyServe(Confabulator(), overlap_retriever, STORE, route_tau=0.2)
    rec = s.turn("What is the capital of the homeland of Zorblax the Painter?")
    assert rec["task"]["vm_answer"] == "Atlantis" and rec["task"]["grounded"] is False
    assert rec["reply"] == DK_EN and rec["offered"] == [DK_EN]


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
