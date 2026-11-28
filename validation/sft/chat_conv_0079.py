"""chat_conv_0079 -- a drafted conversation's state, and what the step tool does to it after a reply: the checks
(`chat_gate_0079`), a refused reply written again, and `regate` (split out of `chat_step_0079.py`, at its 300 lines).

Nick, 2026-11-28 ("more turns and avoid the small 1 turn ones"), the end of order 0079:
- **A refused reply is written again,** up to twice (`MAX_TRIES` in all), before the conversation is cut; each
  refused try stays with the turn (`rejected`), so the one that passed and the ones that did not are a chosen /
  rejected pair for the preference pass.
- **`regate`** runs today's checks over every kept turn. A cut that no longer holds (the reply passes now, or it was
  cut on fewer than `MAX_TRIES` tries) comes back: a passing reply is kept and the conversation goes on; a still
  refused one comes out, into the turn's `rejected`, and the user's turn is pending again, for the next session.
- **`relength`** draws 8 to 14 Cubby turns for every plan not yet done, and for the open conversations, keeping what
  is drafted and the fact plans' asks; the kind's turn and a second subject about halfway are drawn inside it.

Wired: STANDALONE (`chat_step_0079.py reply | regate | relength`).
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import re

from cubbyllm.speech import cubby_record
from validation.sft import chat_gate_0079 as G
from validation.sft.chat_rows_0079 import Stop, _read, _write

__wiring__ = "STANDALONE"

MAX_TRIES = 3
WORK = pathlib.Path(".")
_WORDS: tuple | None = None


def _p(*parts) -> pathlib.Path:
    return WORK.joinpath(*parts)


def plans() -> dict:
    return {p["id"]: p for p in _read(_p("plans.json"), [])}


def batch_of(cid: str) -> dict:
    for f in sorted(_p("batches").glob("b*.json")):
        b = _read(f)
        if cid in b["ids"]:
            return b
    raise Stop(f"{cid} is in no batch")


def conv_path(cid: str) -> pathlib.Path:
    return _p("convs", cid.replace(":", "_") + ".json")


def load(cid: str) -> dict:
    c = _read(conv_path(cid))
    if c is None:
        plan = plans().get(cid)
        if plan is None:
            raise Stop(f"no plan {cid}")
        c = {"id": cid, "plan": plan, "turns": [], "pending": None, "cut": None, "done": False}
    return c


def about_cubby(text: str) -> bool:
    return cubby_record.is_about_cubby(text) or bool(re.search(r"\b(are you|es-tu|tu es)\b", text, re.I))


def persona_words() -> tuple:
    """The run's persona word counts, and what Cubby says anyway (frozen from the drafted replies on first use)."""
    global _WORDS
    if _WORDS is None:
        from validation.sft import chat_persona_0079 as L
        f = _p("persona_background.json")
        if not f.exists():
            _write(f, L.background([t["cubby"] for g in sorted(_p("convs").glob("*.json")) for t in _read(g)["turns"]]))
        _WORDS = (L.counts([p["persona"] for p in plans().values()]), _read(f))
    return _WORDS


def check(t: dict, prior: list[dict], plan: dict) -> tuple[str | None, list[str]]:
    context = t["facts"] + [x for u in prior for x in (u["user"], u["cubby"])]
    return G.gate(t["cubby"], t["user"], context, t["mood"], about_cubby(t["user"]), plan["lang"],
                  tuple(plan["persona"].get("places") or ()), tuple(u["user"] for u in prior),
                  (plan["persona"], *persona_words()))


def _try(why: str, reply: str, ran: list[str]) -> dict:
    return {"cubby": reply, "why": why, "checks": ran}


def cmd_reply(cid: str, text: str) -> str:
    c = load(cid)
    if c["cut"] or c["done"] or not c["pending"]:
        raise Stop("no user turn is waiting for a reply" if not c["pending"] else f"finished ({c['cut']})")
    tries = list(c["pending"].get("rejected") or [])
    t = {**{k: v for k, v in c["pending"].items() if k != "rejected"}, "cubby": text.strip()}
    why, ran = check(t, c["turns"], c["plan"])
    if why and len(tries) + 1 < MAX_TRIES:
        c["pending"]["rejected"] = tries + [_try(why, t["cubby"], ran)]
        _write(conv_path(cid), c)
        return (f"REFUSED: {why}. Write Cubby's reply to the same user turn again, avoiding that "
                f"(try {len(tries) + 2} of {MAX_TRIES}), and send it with: reply")
    c["turns"].append({**t, "gate": why or "ok", "checks": ran, "rejected": tries})
    c["pending"] = None
    if why:
        c["cut"] = why
    elif len(c["turns"]) >= c["plan"]["length"]:
        c["done"] = True
    _write(conv_path(cid), c)
    if why:
        return f"CUT: {why} ({MAX_TRIES} tries). The turns before it are kept; go to the next conversation."
    if c["done"]:
        return "DONE: this conversation is complete. Go to the next one."
    return f"OK ({len(c['turns'])} of {c['plan']['length']} Cubby turns). Write the user's next message (turn)."


def cmd_regate() -> str:
    counts = collections.Counter()
    for f in sorted(_p("convs").glob("*.json")):
        c = _read(f)
        if c.get("pending"):                             # a session is on it: left as it is
            counts["left: a turn pending"] += 1
            continue
        state = "kept"
        for k, t in enumerate(c["turns"]):
            why, ran = check(t, c["turns"][:k], c["plan"])
            t.update(gate=why or "ok", checks=ran)
            if not why:
                continue
            tries = list(t.get("rejected") or [])
            c["turns"], c["done"] = c["turns"][:k], False
            if len(tries) + 1 < MAX_TRIES:               # the cut no longer holds: the reply comes out, the turn waits
                pend = {x: v for x, v in t.items() if x not in ("cubby", "gate", "checks", "rejected")}
                c["pending"], c["cut"] = {**pend, "rejected": tries + [_try(why, t["cubby"], ran)]}, None
                state = "back to pending"
            else:
                c["turns"].append(t)
                c["cut"], state = why, "cut"
            break
        else:
            was = c.get("cut")
            c["cut"], c["done"] = None, len(c["turns"]) >= c["plan"]["length"]
            state = "reopened" if was else ("done" if c["done"] else "open")
        counts[state] += 1
        _write(f, c)
    return json.dumps(dict(counts))


def _draw(rng: random.Random, have: int, plan: dict) -> dict:
    """8 to 14 Cubby turns, more than `have` already drafted; the kind's turn and the second subject inside it."""
    length = max(rng.randint(8, 14), have + 2)
    out = {"length": length}
    if plan.get("kind_turn", 0) >= have:               # the kind's turn not yet reached: drawn again
        out["kind_turn"] = rng.randint(max(1, have), length - 2)
    half = length // 2
    second = half if half != out.get("kind_turn", plan.get("kind_turn")) else half + 1
    out["second_turn"] = max(second, have)
    return out


def cmd_relength(seed: int = 79) -> str:
    """Every plan not yet done, and every open conversation, drawn again at 8 to 14 Cubby turns."""
    allp = _read(_p("plans.json"))
    counts = collections.Counter()
    for p in allp:
        c = _read(conv_path(p["id"]))
        rng = random.Random(f"0079:relength:{seed}:{p['id']}")
        if c is None:
            p.update(_draw(rng, 0, p))
            counts["plans"] += 1
        elif not (c["done"] or c["cut"]):
            have = len(c["turns"]) + bool(c.get("pending"))
            c["plan"].update(_draw(rng, have, c["plan"]))
            p.update({k: c["plan"][k] for k in ("length", "kind_turn", "second_turn")})
            _write(conv_path(p["id"]), c)
            counts["open conversations"] += 1
    _write(_p("plans.json"), allp)
    return json.dumps(dict(counts))
