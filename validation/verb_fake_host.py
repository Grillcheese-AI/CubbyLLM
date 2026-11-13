"""Order 0072, build step 4: act verbs with a fake host only. Nothing real is called: the host records each call and
gives back a scripted read-back.

`run_act(case, vm, host)`: the gate program decides (2 do, 1 ask first, 0 refuse) from what the host binds: the act's
class in this context (`verb_kinds.act_class`), a tool or none, and a yes **given in this turn** (a stored yes from an
earlier turn is never passed in). Only a 2 reaches the host; then the read-back program compares what the host reads
back with what was asked, and "done" is said only when it agrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm.speech import verb_guard, verb_programs  # noqa: E402

__wiring__ = "STANDALONE"


class FakeHost:
    def __init__(self):
        self.calls = []

    def call(self, verb: str, target: str, read_back: str) -> str:
        self.calls.append({"verb": verb, "target": target})
        return read_back


def _run(vm, kind, variant, data) -> int:
    src = verb_programs.program(kind, variant, data)
    why = verb_guard.check(src)
    if why:
        raise RuntimeError(f"the static guard refused a {kind}/{variant} program: {why}")
    return int(float(vm.run(src, fn="solve")["result"]))


def run_act(case: dict, vm, host: FakeHost) -> str:
    data = {"arg_tool": case["tool"], "arg_class": case["class"], "arg_yes": case["yes_this_turn"]}   # never stored_yes
    gate = _run(vm, "act", "gate", data)
    if gate == 0:
        return "refuse"
    if gate == 1:
        return "ask"
    want = f"{case['verb']}:as asked"
    got = host.call(case["verb"], case["instruction"], want if case["read_back_agrees"] else f"{case['verb']}:otherwise")
    return "done" if _run(vm, "act", "read_back", {"arg_want": want, "arg_got": got}) == 1 else "not done"


__all__ = ["FakeHost", "run_act"]
