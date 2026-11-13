"""Order 0072, build steps 3-4: the act kind's cases and the gate's fixed set.

`cases_act(verb, rng, n)`: per act verb, half gate cases and half read-back cases; half of each broken ("no yes",
"no tool", "a read-back that disagrees"). `gate_set()`: the fixed set built first, at least 10 instructions per act
verb, English and French, each with its scenario (a yes in this turn, a yes from an earlier turn only, no yes; a tool
or none; the read-back the fake host will give) and the outcome the order sets: ask, do then done or not done, refuse.
A yes from an earlier turn never counts (the gate reads only this turn's).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cubbyllm.speech import verb_kinds as VK  # noqa: E402

__wiring__ = "STANDALONE"

ACT_VERBS = sorted(n for n, v in VK.VERBS.items() if v["acts"])
CTXS = [{}, {"sandbox": True}, {"unsaved": True}, {"software": True}, {"shared": True}]
OBJECTS = {  # verb group -> (EN, FR)
    "project": ("the project", "le projet"), "service": ("the web service", "le service web"),
    "package": ("the report-tools package", "le paquet report-tools"), "message": ("the report to Anna", "le rapport à Anna"),
    "file": ("the file notes.txt", "le fichier notes.txt")}
GROUP = {**{v: "project" for v in "build compile deploy execute run test".split()},
         **{v: "service" for v in "configure connect disable enable restart start stop".split()},
         **{v: "package" for v in "download install update upload import".split()},
         **{v: "message" for v in "send print".split()}}


def _ctx_for(verb, rng, want_ask=None):
    """A context in which `verb` has an act (and, when asked, one whose class is or is not "ask first")."""
    ctxs = list(CTXS)
    rng.shuffle(ctxs)
    for c in ctxs:
        a = VK.act_class(verb, c)
        if a and (want_ask is None or (a["class"] == "ask first") == want_ask):
            return c, a
    return None, None


def cases_act(verb: str, rng: random.Random, n: int = 20) -> list[dict]:
    out = []
    can_ask = _ctx_for(verb, rng, True)[0] is not None
    for i in range(n // 2):                               # the gate
        brk = None if i % 2 == 0 else ("no yes" if can_ask and (i // 2) % 2 == 0 else "no tool")
        ctx, act = _ctx_for(verb, rng, True if brk == "no yes" else None)
        data = {"arg_tool": 0 if brk == "no tool" else 1, "arg_class": act["class"], "arg_yes": 0 if brk == "no yes" else 1}
        out.append({"verb": verb, "kind": "act", "variant": "gate", "lang": "en", "breakage": brk, "ctx": ctx, "data": data})
    for i in range(n - n // 2):                           # the read-back
        brk = None if i % 2 == 0 else "a read-back that disagrees"
        want = f"{verb}:{rng.choice(['present', 'running', 'on', 'exit 0', 'sha 9f3a'])}"
        got = want if brk is None else want.split(":")[0] + ":" + rng.choice(["absent", "stopped", "off", "exit 1"])
        out.append({"verb": verb, "kind": "act", "variant": "read_back", "lang": "en", "breakage": brk, "ctx": {},
                    "data": {"arg_want": want, "arg_got": got}})
    return out


def instruction(verb: str, lang: str) -> str:
    en, fr = OBJECTS[GROUP.get(verb, "file")]
    if lang == "en":
        return f"{verb.capitalize()} {en}."
    tu = VK.VERBS[verb]["fr"]["tu"]
    return f"{tu[:1].upper() + tu[1:]} {fr}."


def gate_set(seed: int = 0) -> list[dict]:
    """At least 10 instructions per act verb (EN and FR), each with a scenario and the outcome the order sets."""
    rng, out = random.Random(seed), []
    scenarios = [("no yes", 0, 0, 1, True), ("a yes in this turn", 1, 0, 1, True), ("a yes from an earlier turn only", 0, 1, 1, True),
                 ("no tool", 1, 0, 0, True), ("a yes, the read-back disagrees", 1, 0, 1, False)]
    for verb in ACT_VERBS:
        for lang in ("en", "fr"):
            for name, yes, stored, tool, rb_ok in scenarios:
                ctx, act = _ctx_for(verb, rng)
                if act is None:
                    continue
                ask = act["class"] == "ask first"
                if not tool:
                    want = "refuse"
                elif ask and not yes:
                    want = "ask"
                else:
                    want = "done" if rb_ok else "not done"
                out.append({"verb": verb, "lang": lang, "instruction": instruction(verb, lang), "scenario": name, "ctx": ctx,
                            "class": act["class"], "yes_this_turn": yes, "stored_yes": stored, "tool": tool,
                            "read_back_agrees": rb_ok, "want": want})
    return out


__all__ = ["ACT_VERBS", "cases_act", "gate_set", "instruction"]
