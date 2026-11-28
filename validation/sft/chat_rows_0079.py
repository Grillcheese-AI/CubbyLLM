"""chat_rows_0079 -- the final run's ends, for `chat_step_0079`: the plans and the agents' batches (with each one's
brief), the character sheet as the drafting agent reads it (from the order, so there is one copy), and the merge:
every kept reply as a talk row in the second shape (`talk_format.render`, as serving renders it), with a manifest of
counts only.

A reply is a row only when its gate said ok and every check of `chat_gate_0079.CHECKS` ran on it: a turn gated before
a check existed is regated first, never kept as it is. One conversation in twenty is held (by the SHA-256 of its id),
for the trainer's held loss.

Wired: STANDALONE (`chat_step_0079.py plans | sheet | brief | merge | status`).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys

from cubbyllm.emit import talk_format
from validation.sft import chat_gate_0079 as G
from validation.sft import chat_plans_0079 as P

__wiring__ = "STANDALONE"

STEP = pathlib.Path(__file__).resolve().parent / "chat_step_0079.py"


class Stop(Exception):
    """A step the tool will not take; its message tells the agent what to do instead."""


def _read(path: pathlib.Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def plan_args(ap) -> None:
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=79)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--fr-share", type=float, default=0.2)
    ap.add_argument("--personas-en", help="Nemotron-Personas-USA as text chunks (jsonl)")
    ap.add_argument("--personas-fr", help="one parquet file of Nemotron-Personas-France")
    ap.add_argument("--topical", help="Topical-Chat's user lines (talk_lines/user_lines_topical_chat.jsonl)")
    ap.add_argument("--store-rows", help="talk_v2_2.jsonl: the store's names a user may ask about")


def cmd_plans(work: pathlib.Path, a) -> str:
    pilot = G.pilot
    if (work / "plans.json").exists():
        raise Stop(f"{work / 'plans.json'} exists: its conversations are under way; move it aside to plan again")
    if not (a.personas_en and a.personas_fr and a.topical and a.store_rows):
        raise Stop("plans needs --personas-en, --personas-fr, --topical and --store-rows")
    en = P.personas_en(a.personas_en, a.n, a.seed)
    fr = P.personas_fr(a.personas_fr, a.n, a.seed)
    users = [json.loads(ln) for ln in open(a.topical, encoding="utf-8")]
    openers = [r.get("ask") or r["text"] for r in users if r["split"] == "train" and r["standalone"]
               and not r.get("sensitive") and r["kind"] in ("hello", "about_you", "question", "know_ask")
               and 3 <= r["words"] <= 25]
    idx = pilot.fact_index(pathlib.Path(a.store_rows))
    pilot.mentions("", idx)
    names = sorted(e for e in pilot._LOWER[id(idx)].values() if 4 <= len(e) <= 40 and len(idx[e]) >= 3)
    plans = P.build(a.n, a.seed, en, fr, openers, names, a.fr_share)
    assert not any(P.is_test(p["persona"]["uuid"]) for p in plans)
    _write(work / "plans.json", plans)
    for b in P.batches(plans, a.batch_size):
        _write(work / "batches" / f"{b['batch']}.json", b)
        (work / "batches" / f"{b['batch']}.md").write_text(brief(work, b["batch"]), encoding="utf-8")
    c = collections.Counter
    return json.dumps({"plans": len(plans), "batches": -(-len(plans) // a.batch_size),
                       "by_lang": dict(c(p["lang"] for p in plans)), "by_source": dict(c(p["source"] for p in plans)),
                       "by_kind": dict(c(p["kind"] for p in plans)),
                       "cubby_turns_planned": sum(p["length"] for p in plans)}, indent=1)


def sheet(order: pathlib.Path) -> str:
    """The character sheet of 2026-11-28 with its examples, as the order holds it."""
    text = order.read_text(encoding="utf-8")
    start = text.index("## Cubby's character sheet")
    return text[start:text.index("### What changed from the sheet", start)].strip()


def brief(work: pathlib.Path, bid: str) -> str:
    b = _read(work / "batches" / f"{bid}.json")
    if b is None:
        raise Stop(f"no batch {bid}")
    run = f'"{sys.executable}" "{STEP}" --work "{work}"'
    return f"""# Order 0079, batch {bid}: {len(b['ids'])} conversations, drafted by {b['writer']}

You draft conversations between a user and Cubby, through a step tool. You write both sides, one turn at a time: the
user (seeded from a persona) and Cubby (from its character sheet). The tool works out Cubby's mood and the facts it
may use, and checks every Cubby reply; a reply it refuses ends that conversation. Never edit the tool's files.

1. Once: `{run} start {bid} --model <your exact model id>`, then read the sheet: `{run} sheet`.
2. For each id below: `{run} show <id>`. Then, until it says DONE or CUT:
   - the user's message, with a GoEmotions label for how the user feels in it (admiration ... neutral):
     `{run} turn <id> --seems <label> --strength weak|medium|strong --text "<message>"`
   - read the Mood, Shade, They seem and Facts lines it prints; then Cubby's reply:
     `{run} reply <id> --text "<reply>"`
   (or leave out --text and give the message on stdin, `<<'EOF'` ... `EOF`, when it holds quotes)
3. When every id is DONE or CUT, say so and stop.

The user: a real-sounding person from the persona `show` prints (their job, a hobby, who they are); one or two short
sentences, typed the way people type; they talk about their life, react to Cubby, ask it things; now and then an
emoji. Never a model, and never the name of their town, city or state (Cubby never names a place). When `show`
says THIS user turn, write that kind of turn.
Cubby: the sheet, always. Facts only from the Facts lines; when there are none and the user asked for one, it says
so plainly. Never a place name, never a past or a body, at most one emoji and only when the user's turn has one; it
turns the user's pronouns ("your X" is Cubby's "my X"; the user's "I"/"my" is "you"/"your"), and it never says how
the user is unless they said it. One to three sentences unless asked for more; not every reply ends with a question.

Your ids: {' '.join(b['ids'])}
"""


def _held(cid: str) -> bool:
    return int(hashlib.sha256(cid.encode("utf-8")).hexdigest(), 16) % 20 == 0


def rows(work: pathlib.Path):
    """(row, why not) for every drafted reply: the row when it is kept, else the reason."""
    for f in sorted((work / "convs").glob("*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        for k, t in enumerate(c["turns"]):
            if t.get("gate") != "ok":
                yield None, "refused"
                continue
            if list(t.get("checks") or []) != list(G.CHECKS):
                yield None, "a check did not run on it (regate)"
                continue
            turns = [x for u in c["turns"][:k] for x in (("user", u["user"]), ("cubby", u["cubby"]))]
            ctx = {"mood": t["mood"], "shade": t["shade"], "seems": t["seems"], "facts": t["facts"],
                   "turns": turns + [("user", t["user"])]}
            yield {"id": f"chat-{c['id']}:{k}", "family": "chat", "split": "held" if _held(c["id"]) else "train",
                   "prompt": talk_format.render("", ctx), "answer": t["cubby"], "format": 2,
                   "source": "chat_0079_final", "lang": c["plan"]["lang"], "writer": "local Claude Code agent",
                   "model": t["model"], "session": t["session"], "batch": t["batch"], "kind": t.get("kind"),
                   "persona": c["plan"]["persona"]["uuid"], "facts_route": t["route"]}, None


def tally(work: pathlib.Path) -> dict:
    convs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((work / "convs").glob("*.json"))]
    turns = [t for c in convs for t in c["turns"]]
    kept = [r for r, _ in rows(work) if r is not None]
    c = collections.Counter
    plans = json.loads((work / "plans.json").read_text(encoding="utf-8")) if (work / "plans.json").exists() else []
    times = c()
    first, last = {}, {}
    for t in turns:
        first[t["batch"]] = min(first.get(t["batch"], t["at"]), t["at"])
        last[t["batch"]] = max(last.get(t["batch"], t["at"]), t["at"])
        times[t["batch"]] += 1
    return {"plans": len(plans), "conversations_started": len(convs),
            "conversations_whole": sum(bool(x["done"]) for x in convs), "conversations_cut": sum(bool(x["cut"]) for x in convs),
            "cubby_turns_drafted": len(turns), "rows_kept": len(kept),
            "not_kept": dict(c(w for _, w in rows(work) if w)),
            "cut_by": dict(c(str(x["cut"]).split(":")[0][:40] for x in convs if x["cut"]).most_common()),
            "rows_by_model": dict(c(r["model"] for r in kept)), "rows_by_lang": dict(c(r["lang"] for r in kept)),
            "rows_by_kind": dict(c(r["kind"] for r in kept if r["kind"])),
            "kind_share_of_rows": round(sum(bool(r["kind"]) for r in kept) / max(1, len(kept)), 3),
            "rows_with_facts": sum("\nFacts:\n" in "\n" + r["prompt"] for r in kept),
            "turns_routed_to_facts": sum(bool(t.get("ask")) for t in turns),
            "host_asks_verified": sum(bool((t.get("ask") or {}).get("verified")) for t in turns),
            "rows_held": sum(r["split"] == "held" for r in kept),
            "batches": {b: {"turns": times[b], "first": first[b], "last": last[b]} for b in sorted(times)}}


def merge(work: pathlib.Path, out: str, manifest: str) -> str:
    kept = [r for r, _ in rows(work) if r is not None]
    path = pathlib.Path(out) if out else work / "chat_0079_final_rows.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = {"what": "order 0079's final run: conversations drafted by local Claude Code agents (Haiku and Sonnet) through "
                 "chat_step_0079, users from Nemotron personas (CC BY 4.0), moods from the hormone model, facts from "
                 "the host's resolver, every reply through chat_gate_0079; rows in the second talk shape",
         "personas": "nvidia/Nemotron-Personas-USA and nvidia/Nemotron-Personas-France, CC BY 4.0; test personas "
                     "(uuid SHA-256 ending in 0) never drawn",
         "not_for_shipping": "Claude's output: never in an adapter that ships (redrafted before Cubby goes live)",
         "checks": list(G.CHECKS), **tally(work)}
    if manifest:
        pathlib.Path(manifest).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return json.dumps({k: m[k] for k in ("rows_kept", "conversations_started", "cut_by")}, ensure_ascii=False)
