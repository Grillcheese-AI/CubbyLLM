"""chat_step_0079 -- the step tool of 0079's final run: the pilot's tool (Cowork's, off the repo) ported onto the real
`cubbyllm`. A drafting agent (a local Claude Code session, Haiku or Sonnet) writes both sides of a conversation; the
tool does what it cannot: Cubby's mood (the hormone model over the user's turns, read as the host serves:
`LabelAppraiser` with the agent's GoEmotions label), the facts for the turn (the host's own resolver: the brain's
router says whether the turn needs facts, and the served ask loop answers it), and every check of
`chat_gate_0079` on Cubby's reply. A refused reply cuts the conversation there; the turns before it are kept.

    python validation/sft/chat_step_0079.py plans --n 500 --personas-en F --personas-fr F --topical F --store-rows F
    python validation/sft/chat_step_0079.py sheet | brief <batch>
    python validation/sft/chat_step_0079.py start <batch> --model <the session's exact model id> [--session S]
    python validation/sft/chat_step_0079.py show <id>
    python validation/sft/chat_step_0079.py turn <id> --seems <GoEmotions label> [--strength s] [--text T]  (or stdin)
    python validation/sft/chat_step_0079.py reply <id> [--text T]                                         (or stdin)
    python validation/sft/chat_step_0079.py status | regate | merge [--out rows.jsonl] [--manifest log.json]

The work folder (`--work`, default the data root's ``chat_0079_final/``) holds the plans, the batches, one file a
conversation and the rows: all off the repo. Facts need the server with ``--ask`` (``--host`` or ``CUBBY_HOST``) and
the audit token (``CUBBY_AUDIT_TOKEN`` or its file; never printed). Every turn records the batch's writer, model id and
session; `merge` keeps a reply only when every check of `chat_gate_0079.CHECKS` ran on it and passed.

Wired: STANDALONE (a data builder; the agents run it, `chat_rows_0079` writes the rows).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.affect import mood  # noqa: E402
from validation.sft import chat_gate_0079 as G  # noqa: E402
from validation.sft import chat_plans_0079 as P  # noqa: E402
from validation.sft import chat_conv_0079 as C  # noqa: E402
from validation.sft.chat_conv_0079 import _p, batch_of, cmd_regate, cmd_reply, conv_path, load  # noqa: E402
from validation.sft.chat_rows_0079 import Stop, _read, _write  # noqa: E402

__wiring__ = "STANDALONE"

LABELS = set("admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval "
             "disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization "
             "relief remorse sadness surprise neutral".split())
ORDER = ROOT / "docs" / "orders" / "0079-llm-level-chat.md"
def mood_after(users: list[str], labels: list[str]) -> dict:
    """Cubby's eight scales after the user's turns: the hormones, nudged by the serving read with the agent's label."""
    with G.env(CB_MOOD_REGULATION="1"):                 # the regulated mood, as the pilot and the final run read it
        body = G.pilot.Body()
        body.appraiser = G.pilot.LabelAppraiser()
        for text, label in zip(users, labels):
            body.appraiser.label = label
            body.nudge(text)
        return mood.scales(body.chem)


def resolve(text: str, host: str | None) -> dict:
    """The host's own resolver for the turn: does it need facts (the brain's router), and the served ask loop's walked
    facts when it does (verified, no namesake left to ask about). A refusal hands over none, as at serving."""
    from cubbyllm.brain_route import BrainRoute
    from cubbyllm.reasoning import ask_commands
    with G.env(CB_ASK_SELF="1", CB_ASK_COMMANDS="1"):  # routed as serving will route with the rows' flags (Cowork)
        need, why = BrainRoute().needs_facts(text)
        asked = ask_commands.restate(text)               # "do you know who X is" -> "Who is X?"
    if not need:
        return {"route": why, "facts": [], "ask": None}
    if not host:
        raise Stop("this turn needs the host's facts: start the server with --ask and give --host (or CUBBY_HOST)")
    import urllib.request
    from cubbyllm.speech.audience import configured_token
    tok = configured_token()
    req = urllib.request.Request(host.rstrip("/") + "/ask", data=json.dumps({"text": asked}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **({"X-Audit-Token": tok} if tok else {})})
    rec = json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8"))
    if rec.get("verified") and "trace" not in rec:
        raise Stop("the host answered with the user's view (no walked facts): set CUBBY_AUDIT_TOKEN for the tool")
    ok = bool(rec.get("verified")) and not rec.get("clarify")
    return {"route": why, "facts": list(rec.get("trace") or []) if ok else [],
            "ask": {"id": rec.get("id"), "asked": asked, "verified": bool(rec.get("verified")), "reason": rec.get("reason"),
                    "clarify": bool(rec.get("clarify"))}}


def cmd_start(bid: str, model: str, session: str | None) -> str:
    f = _p("batches", f"{bid}.json")
    b = _read(f)
    if b is None:
        raise Stop(f"no batch {bid}")
    family = b["writer"].split("-")[-1]
    if family not in model.lower():
        raise Stop(f"batch {bid} is {b['writer']}'s; this session's model {model!r} is not")
    if b.get("model") and b["model"] != model:
        raise Stop(f"batch {bid} was started by {b['model']}")
    b.update(model=model, session=session or b.get("session") or f"{bid}-{int(time.time())}",
             started=b.get("started") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write(f, b)
    return f"batch {bid} started: {len(b['ids'])} conversations, model {model}, session {b['session']}"


def cmd_show(cid: str) -> str:
    c = load(cid)
    p = c["plan"]
    out = [f"id {cid} | language {p['lang']} | Cubby turns planned {p['length']} | done {len(c['turns'])} | "
           f"cut {c['cut'] or '-'} | finished {c['done']}"]
    s = p["persona"]
    who = ", ".join(str(x) for x in (s.get("occupation"), s.get("age") and f"{s['age']} years old") if x)
    out.append(f"the user: {who or 'see below'}; their work: {s.get('work') or '-'}; a hobby: {s['hobby'] or '-'}; "
               f"who they are: {s['line']}")
    if c["turns"] and not (c["cut"] or c["done"]):
        out.append("under way; continue it:")
        out += [f"  User: {t['user']}\n  Cubby: {t['cubby']}" for t in c["turns"]]
    if c.get("pending") and not (c["cut"] or c["done"]):
        pend = c["pending"]
        out += [f"  User: {pend['user']}", "this user turn waits for Cubby's reply (send it with: reply):",
                mood.line(pend["mood"]), "Shade: " + (", ".join(pend["shade"]) or "none"), "They seem: " + pend["seems"],
                "Facts:" + ("" if pend["facts"] else " none for this turn")] + [f"- {f}" for f in pend["facts"]]
        out += [f"  (a refused try: {r['why']})" for r in pend.get("rejected") or []]
        return "\n".join(out)
    elif not c["turns"]:
        out.append({"topical-chat": f"the user's first message, word for word: {p.get('opener')}",
                    "feeling": f"the user writes first: they feel {p.get('feeling')} about {p.get('situation')}",
                    "persona life": "the user writes first, about their own day, work or hobby",
                    "asks about a fact": (f"the user writes first and asks Cubby this, in their own words, keeping the "
                                          f"name as it is and what is asked: {p['ask_question']}" if p.get("ask_question")
                                          else f"the user writes first and asks Cubby about: {p.get('ask')} (named like that)")}
                   [p["source"]])
    n = len(c["turns"])
    if not (c["cut"] or c["done"]):
        if n == p["kind_turn"]:
            out.append(f"THIS user turn: {P.KIND_BRIEF[p['kind']]}")
        elif n == p.get("second_turn"):
            out.append("THIS user turn: the user brings a second subject of their own (their work, a hobby, a question)")
        out.append("the conversation may close now: one closing exchange at most" if n >= p["length"] - 2
                   else f"no goodbye yet: {p['length'] - n} Cubby turns to go")
    return "\n".join(out)


def cmd_turn(cid: str, seems: str, strength: str, text: str, host: str | None) -> str:
    c, b = load(cid), batch_of(cid)
    if not b.get("model"):
        raise Stop(f"start the batch first: start {b['batch']} --model <your exact model id>")
    if c["cut"] or c["done"]:
        raise Stop(f"this conversation is finished ({c['cut'] or 'done'}); go to the next one")
    if c["pending"]:
        raise Stop("reply to the pending user turn first (reply)")
    text, seems = text.strip(), seems.strip().lower() if seems.strip().lower() in LABELS else "neutral"
    if not text:
        raise Stop("the user's message is empty")
    sc = mood_after([t["user"] for t in c["turns"]] + [text], [t["seems"] for t in c["turns"]] + [seems])
    got = resolve(text, host)
    kind = c["plan"]["kind"] if len(c["turns"]) == c["plan"]["kind_turn"] else None
    c["pending"] = {"user": text, "seems": seems, "strength": strength, "mood": sc, "named": mood.named(sc),
                    "shade": mood.shade(sc), "facts": got["facts"], "route": got["route"], "ask": got["ask"],
                    "kind": kind, "model": b["model"], "session": b["session"], "batch": b["batch"],
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(conv_path(cid), c)
    lines = [mood.line(sc), "Shade: " + (", ".join(c["pending"]["shade"]) or "none"), "They seem: " + seems,
             "Facts:" + ("" if got["facts"] else " none for this turn")] + [f"- {f}" for f in got["facts"]]
    return "\n".join(lines + ["Now write Cubby's reply and send it with: reply"])


def cmd_status() -> str:
    from validation.sft.chat_rows_0079 import tally
    return json.dumps(tally(C.WORK), indent=1, ensure_ascii=False)


def main(argv=None) -> int:
    from cubbyllm.local_paths import data_dir
    for s in (sys.stdin, sys.stdout):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plans", "sheet", "brief", "start", "show", "turn", "reply", "status", "regate",
                                    "merge", "reseed", "rebatch", "relength"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--work", default=os.environ.get("CUBBY_CHAT_0079") or str(data_dir("chat_0079_final")))
    ap.add_argument("--host", default=os.environ.get("CUBBY_HOST"))
    ap.add_argument("--seems", default="neutral")
    ap.add_argument("--strength", default="medium", choices=["weak", "medium", "strong"])
    ap.add_argument("--text")
    ap.add_argument("--model")
    ap.add_argument("--session")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    from validation.sft import chat_batches_0079 as B
    from validation.sft import chat_rows_0079 as R
    R.plan_args(ap)
    a = ap.parse_args(argv)
    C.WORK, C._WORDS = pathlib.Path(a.work), None
    WORK = C.WORK
    try:
        if a.cmd in ("show", "turn", "reply", "start", "brief") and not a.id:
            raise Stop(f"{a.cmd} needs an id")
        text = (a.text if a.text is not None else sys.stdin.read()) if a.cmd in ("turn", "reply") else ""
        out = {"plans": lambda: R.cmd_plans(WORK, a), "sheet": lambda: R.sheet(ORDER),
               "brief": lambda: R.brief(WORK, a.id, a.host), "start": lambda: cmd_start(a.id, a.model or "", a.session),
               "show": lambda: cmd_show(a.id), "turn": lambda: cmd_turn(a.id, a.seems, a.strength, text, a.host),
               "reply": lambda: cmd_reply(a.id, text), "status": cmd_status, "regate": cmd_regate,
               "merge": lambda: R.merge(WORK, a.out, a.manifest),
               "reseed": lambda: B.cmd_reseed(WORK, a.host, a.max_asks, a.seed, a.gen3 or str(data_dir("gen3_free_text.jsonl"))),
               "rebatch": lambda: B.cmd_rebatch(WORK, a.batch_size, a.sessions, a.host),
               "relength": lambda: C.cmd_relength(a.seed)}[a.cmd]()
    except Stop as e:
        print(f"STOP: {e}")
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
