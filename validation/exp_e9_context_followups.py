"""H-E9: with a context graph, is a follow-up that names no event answered about the event the conversation is about?

Dialogues are built from held-split events of the history graph (`build_ground_sft.split_of`: the talk adapter never
trained on a question about them), seeded, and written with a manifest; the first `--dev` of each type are the dev
split, the rest the test split. Every first turn names its event in the host's own template (history_lookup.CANON).

  follow       the event, then two more asks about it by pronoun or bare ("When did it happen?", "Why?", "And then what?")
  description  the event, then an ask naming it by its head noun ("What led to the battle?")
  switch       event X, a pronoun about X, event Y, a pronoun that must mean Y
  clarify      a name the graph holds for two events of different dates that answer the kind differently (the host
               should ask which), then "the one in <year>"

Two arms over the same turns: the context graph (standin/context_graph.py, one per dialogue) and each turn answered
alone (H-E8's host, `HistoryLookup.ask`). The talk adapter drafts over the served block and the host's ask; `ask.
talk_reply` decides what is said. Per turn: resolved to the right event, and the outcome (correct / wrong / refused /
asked / none / unparsed, as H-E8). The gate reads the test split's follow-up turns (pronoun, description, switched):
resolved >= 95%, correct >= 85%, wrong = 0; switched resolved >= 95%; a pick binds >= 90% where the host asked;
kill: wrong > 2% of follow-up turns.

    python validation/exp_e9_context_followups.py --export <export dir> --adapter <adapter dir> [--split test] [--tag _x]
    python validation/exp_e9_context_followups.py --no-model [--split dev]        # resolution only, CPU
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ask import ABSENT_REPLY, has_value, talk_reply  # noqa: E402
from build_ground_sft import render, split_of  # noqa: E402
from context_graph import ContextGraph  # noqa: E402
from history_graph import HistoryGraph, key, overlap, year_text  # noqa: E402
from history_lookup import HistoryLookup, canonical, serve  # noqa: E402

LOG: list[str] = []
OUT = os.path.join(ROOT, "standin", "data", "out")
KINDS = ("cause", "effect", "when", "where", "who", "downstream")
PRONOUN_Q = {"cause": ["What led to it?", "Why did it happen?", "What caused it?", "Why?"],
             "effect": ["What did it lead to?", "What came of it?", "And then what?", "What were the consequences?"],
             "when": ["When did it happen?", "When was that?", "When?"],
             "where": ["Where did it take place?", "Where was that?", "Where?"],
             "who": ["Who took part in it?", "Who was involved?", "Who?"],
             "downstream": ["What would change if it had not happened?", "What if it had never happened?"]}
HEAD_Q = {"cause": "What led to the {h}?", "effect": "What did the {h} lead to?", "when": "When did the {h} happen?",
          "where": "Where did the {h} take place?", "who": "Who took part in the {h}?",
          "downstream": "What would change if the {h} had not happened?"}
HEADS = ("battle war siege treaty revolt rebellion election campaign death fall founding conquest reign massacre invasion "
         "strike trial coronation assassination expedition council crusade uprising riot fire flood earthquake murder "
         "execution surrender alliance raid mutiny coup revolution capture sack occupation blockade persecution").split()
FOLLOW_ROLES = ("pronoun", "description", "switched")


def log(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------- dialogues
def answers(g, eid) -> list[str]:
    ev = g.events[eid]
    book = not (ev.sources and all(s[0] == "nyt" for s in ev.sources))
    return [k for k in KINDS if (book or k in ("when", "where", "who")) and serve(g, eid, k)[1]]


def turn(g, eid, kind, q, role) -> dict:
    return {"q": q, "event": eid, "name": g.events[eid].name, "kind": kind, "role": role, "gold": serve(g, eid, kind)[1]}


def build(g, per_type: int, dev: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    held = [e for e, ev in g.events.items() if split_of(ev.name) == "held"]
    rng.shuffle(held)
    taken, out = set(), []

    def events(min_kinds, pred=lambda e: True):
        for e in held:
            if e in taken or not pred(e):
                continue
            ks = answers(g, e)
            if len(ks) >= min_kinds:
                taken.add(e)
                yield e, ks

    def named(e, k):
        return canonical(k, g.events[e].name)

    it = events(3)
    for n in range(per_type):
        e, ks = next(it)
        k1, k2, k3 = rng.sample(ks, 3)
        out.append({"type": "follow", "n": n, "turns": [turn(g, e, k1, named(e, k1), "first"),
                    turn(g, e, k2, rng.choice(PRONOUN_Q[k2]), "pronoun"), turn(g, e, k3, rng.choice(PRONOUN_Q[k3]), "pronoun")]})

    def head(e):
        toks = key(g.events[e].name).split()
        return next((h for h in HEADS if h in toks), None)
    it = events(2, lambda e: head(e) is not None)
    for n in range(per_type):
        e, ks = next(it)
        k1, k2 = rng.sample(ks, 2)
        out.append({"type": "description", "n": n, "turns": [turn(g, e, k1, named(e, k1), "first"),
                    turn(g, e, k2, HEAD_Q[k2].format(h=head(e)), "description")]})
    it = events(2)
    for n in range(per_type):
        (x, kx), (y, ky) = next(it), next(it)
        a1, a2 = rng.sample(kx, 2)
        b1, b2 = rng.sample(ky, 2)
        out.append({"type": "switch", "n": n, "turns": [turn(g, x, a1, named(x, a1), "first"),
                    turn(g, x, a2, rng.choice(PRONOUN_Q[a2]), "pronoun"), turn(g, y, b1, named(y, b1), "first"),
                    turn(g, y, b2, rng.choice(PRONOUN_Q[b2]), "switched")]})
    pairs = []
    for k_, ids in g.by_key.items():
        if len(ids) < 2 or split_of(g.events[ids[0]].name) != "held":
            continue
        a, b = ids[0], ids[1]
        wa, wb = g.events[a].when, g.events[b].when
        if not wa or not wb or overlap(wa, wb, 0) or a in taken or b in taken or min(abs(wa["y0"]), abs(wb["y0"])) < 100:
            continue                                      # a year a reply can say ("the one in 56" reads as no year)
        shared = [k for k in set(answers(g, a)) & set(answers(g, b))
                  if sorted(serve(g, a, k)[1]) != sorted(serve(g, b, k)[1])]
        if shared:
            pairs.append((a, b, sorted(shared)))
    rng.shuffle(pairs)
    for n, (a, b, shared) in enumerate(pairs[:per_type]):
        k = rng.choice(shared)
        pick = rng.choice([a, b])
        w = g.events[pick].when
        taken.update((a, b))
        first = turn(g, pick, k, canonical(k, g.events[a].name), "clarify_ask")
        first["between"] = [a, b]
        out.append({"type": "clarify", "n": n, "turns": [first, turn(g, pick, k, f"The one in {year_text(w['y0'])}.", "clarify_pick")]})
    counts = collections.Counter(d["type"] for d in out)
    for d in out:
        d["split"] = "dev" if d["n"] < dev else "test"
    log(f"dialogues: {dict(counts)} (clarify pairs available: {len(pairs)})")
    return out


# ---------------------------------------------------------------- scoring
def outcome(status, members, returned, t, reply) -> str:
    if status != "clear":
        return {"ask": "asked", "none": "none", "unparsed": "unparsed"}[status]
    if reply is None:
        return "refused"
    expected = returned if t["event"] in members else t["gold"]
    said_absent = reply.strip().lower().startswith(ABSENT_REPLY.lower().rstrip("."))
    if not expected:
        return "correct" if said_absent else "wrong"
    if said_absent:
        return "refused"
    return "correct" if any(has_value(reply, v) for v in expected) else "wrong"


def table(rows, label):
    log(f"\n[{label}]")
    log(f"  {'role':12s} {'n':>4s} {'resolved':>8s} | {'correct':>7s} {'wrong':>5s} {'refused':>7s} {'asked':>5s} {'none':>4s} {'unparsed':>8s}")
    out = {}
    groups = [(r, [x for x in rows if x["role"] == r]) for r in ("first", "pronoun", "description", "switched",
                                                                   "clarify_ask", "clarify_pick")]
    groups.append(("FOLLOW-UPS", [x for x in rows if x["role"] in FOLLOW_ROLES]))
    for role, rs in groups:
        if not rs:
            continue
        n = len(rs)
        c = collections.Counter(x["outcome"] for x in rs)
        res = sum(x["resolved"] for x in rs) / n
        out[role] = {"n": n, "resolved": res, **{o: c[o] for o in ("correct", "wrong", "refused", "asked", "none", "unparsed")}}
        log(f"  {role:12s} {n:4d} {res:8.1%} | {c['correct'] / n:7.1%} {c['wrong']:5d} {c['refused']:7d} {c['asked']:5d} "
            f"{c['none']:4d} {c['unparsed']:8d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(OUT, "history_graph_v21.jsonl"))
    ap.add_argument("--dialogues", default=os.path.join(OUT, "e9_dialogues.jsonl"))
    ap.add_argument("--rebuild", action="store_true", help="rebuild the dialogues even if the file exists")
    ap.add_argument("--per-type", type=int, default=80)
    ap.add_argument("--dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--export", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    g = HistoryGraph.load(args.graph)
    lk = HistoryLookup(g)
    log(f"exp_e9 context follow-ups | graph {len(g.events):,} events ({time.time() - t0:.0f}s)")
    if args.rebuild or not os.path.exists(args.dialogues):
        ds = build(g, args.per_type, args.dev, args.seed)
        with open(args.dialogues, "w", encoding="utf-8") as f:
            for d in ds:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        sha = hashlib.sha256(open(args.dialogues, "rb").read()).hexdigest()
        json.dump({"dialogues": os.path.basename(args.dialogues), "sha256": sha, "graph": os.path.basename(args.graph),
                   "per_type": args.per_type, "dev": args.dev, "seed": args.seed,
                   "counts": dict(collections.Counter(f"{d['split']}:{d['type']}" for d in ds))},
                  open(os.path.join(HERE, "logs", "e9_dialogues.manifest.json"), "w", encoding="utf-8"), indent=1)
    ds = [d for d in map(json.loads, open(args.dialogues, encoding="utf-8")) if d["split"] == args.split]
    log(f"{len(ds)} {args.split} dialogues, {sum(len(d['turns']) for d in ds)} turns")

    draft_of = lambda lines, q: None                              # noqa: E731 -- replaced when the model is loaded
    meta = None
    if not args.no_model:
        from tokenizers import Tokenizer
        from exp_e5_base450m_probes import load_grilly
        from exp_e6_talk_gate import answer, load_adapter
        tk = Tokenizer.from_file(os.path.join(args.export, "tokenizer.json"))
        eos = tk.token_to_id("</s>")
        model, _ = load_grilly(args.export)
        meta = load_adapter(model, args.adapter)
        log(f"adapter: step {meta.get('step')} r={meta['rank']} alpha={meta['alpha']}")
        cache = {}

        def draft_of(lines, q):
            p = render(lines, q)
            if p not in cache:
                cache[p] = answer(model, tk, eos, p)
            return cache[p]

    def speak(status, lines, returned, others, name, canon):
        if status != "clear":
            return None, None, status
        draft = draft_of(lines, canon) if returned else None
        if returned and draft is None:                            # --no-model: resolution only
            return None, None, "no_model"
        facts = [f"{e} {r}: {v}" for e, r, v in lines] + [f"{v} is the {r} of {e}" for e, r, v in lines]
        reply, reason = talk_reply(returned, draft, facts, name, others)
        return draft, reply, reason

    rows = {"context": [], "alone": []}
    t1 = time.time()
    for d in ds:
        cg = ContextGraph(lk)
        for i, t in enumerate(d["turns"]):
            a = cg.ask(t["q"])
            draft, reply, reason = speak(a.status, a.lines, a.returned, a.others, a.name, a.canonical)
            if a.status == "clear":
                cg.record(a, draft, reply, reason)
            right = a.status == "clear" and t["event"] in a.members
            if t["role"] == "clarify_pick":
                right = right and a.via == "clarified"
            rows["context"].append({"type": d["type"], "n": d["n"], "i": i, "role": t["role"], "q": t["q"], "gold": t["name"],
                                    "got": a.name, "via": a.via, "status": a.status, "resolved": right, "reply": reply,
                                    "reason": reason, "outcome": outcome(a.status, a.members, a.returned, t, reply)
                                    if not args.no_model or a.status != "clear" else ("correct" if right else "wrong")})
            h = lk.ask(t["q"])
            draft, reply, reason = speak(h.status, h.lines, h.returned, h.others, h.name, h.canonical)
            right = h.status == "clear" and t["event"] in h.members
            rows["alone"].append({"type": d["type"], "n": d["n"], "i": i, "role": t["role"], "q": t["q"], "gold": t["name"],
                                  "got": h.name, "status": h.status, "resolved": right, "reply": reply, "reason": reason,
                                  "outcome": outcome(h.status, h.members, h.returned, t, reply)
                                  if not args.no_model or h.status != "clear" else ("correct" if right else "wrong")})
    log(f"turns: {len(rows['context'])} per arm, {time.time() - t1:.0f}s")
    result = {"args": vars(args), "adapter_meta": meta}
    for arm in ("context", "alone"):
        result[arm] = table(rows[arm], f"{'with the context graph' if arm == 'context' else 'each turn alone'}, {args.split} split")
    c = result["context"]
    asked = [r for r in rows["context"] if r["role"] == "clarify_ask" and r["status"] == "ask"]
    picks = {(r["type"], r["n"]) for r in asked}
    bound = [r for r in rows["context"] if r["role"] == "clarify_pick" and (r["type"], r["n"]) in picks]
    bind = sum(r["resolved"] for r in bound) / max(1, len(bound))
    log(f"\nclarify: host asked on {len(asked)} of {sum(1 for r in rows['context'] if r['role'] == 'clarify_ask')} first turns; "
        f"the pick bound to the chosen event {bind:.1%} of those")
    f = c.get("FOLLOW-UPS", {})
    if f:
        bars = [("resolved >= 95%", f["resolved"] >= 0.95), ("switched resolved >= 95%", c.get("switched", {}).get("resolved", 0) >= 0.95),
                ("bind >= 90%", bind >= 0.90)]
        if not args.no_model:
            bars += [("correct >= 85%", f["correct"] / f["n"] >= 0.85), ("wrong = 0", f["wrong"] == 0)]
        kill = not args.no_model and f["wrong"] / f["n"] > 0.02
        log(f"gate ({args.split}): " + "; ".join(f"{b} {'PASS' if ok else 'MISS'}" for b, ok in bars) +
            f"; kill (wrong > 2%): {'FIRES' if kill else 'does not fire'}")
    for label, sel in (("context arm, follow-ups not resolved to the gold event", lambda r: r["role"] in FOLLOW_ROLES + ("clarify_pick",) and not r["resolved"]),
                       ("context arm, spoken and wrong", lambda r: r["outcome"] == "wrong")):
        bad = [r for r in rows["context"] if sel(r)][:12]
        if bad:
            log(f"\n{label}, first {len(bad)}:")
            for r in bad:
                log(f"  [{r['type']}/{r['role']}] {r['q']}  gold: {r['gold']} | got: {r['got'] or r['status']} ({r['via']})"
                    + (f"\n     said: {r['reply']}" if r.get("reply") else ""))
    result["rows"] = rows
    out = os.path.join(HERE, "logs", f"exp_e9_context_followups{args.tag}")
    json.dump(result, open(out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\nwrote {out}.json / .log")


if __name__ == "__main__":
    main()
