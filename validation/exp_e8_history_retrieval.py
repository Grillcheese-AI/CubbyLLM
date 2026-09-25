"""H-E8: does a question in plain words reach the history graph -- the right event found, and the answer spoken
right or not at all?

Every question of `build_e8_questions.py` (a split, both tiers) goes through the host as it would serve it:
`history_lookup.HistoryLookup.ask` reads the kind, finds the event, and -- when the match is clear -- serves the
block, the returned values and the host's ask (the kind asked, about the found event by its name: the program the
asker's words came to); the talk adapter drafts over `render(block, that ask)` (greedy, grilly2) and
`ask.talk_reply` decides what is said. Retrieval is scored on its own first (no model):

  top1 / top5     the gold event is the best / among the five best candidates
  clear           the host used a match (else it asks which one was meant, or finds none, or reads no kind)
  kind            the kind the host read off the question is the one asked
  line            the served block holds a line with the answer (the gold event's answer lines)

and then end to end, one outcome per question:

  correct         spoken, and it states a gold value (absent: the host or the adapter says the facts don't say)
  wrong           spoken, and it states no gold value -- a wrong event's block, faithfully read, lands here
  refused         the host's checks refused the draft, or "don't say" where the graph does say
  asked / none / unparsed   the host asked which event, found none over the floor, or read no kind: safe, not answers

The gate (pre-registered, H-E8) reads the test split's reworded questions: top1 >= 90%, line >= 90%, correct >= 80%,
wrong = 0; kill: top5 < 50%.

    python validation/exp_e8_history_retrieval.py --export <export dir> --adapter <adapter dir> [--split test] [--tag _x]
    python validation/exp_e8_history_retrieval.py --no-model [--split dev]        # retrieval only, CPU
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ask import ABSENT_REPLY, has_value, talk_reply  # noqa: E402
from build_ground_sft import render  # noqa: E402
from history_graph import HistoryGraph  # noqa: E402
from history_lookup import MARGIN, MIN_SCORE, HistoryLookup  # noqa: E402

LOG: list[str] = []
KINDS = ("cause", "effect", "when", "where", "who", "downstream", "absent")


def log(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def retrieval(hit, q) -> dict:
    """top1: the gold event is the host's top event (or one of the readings it serves as that event)."""
    ranks = [c["event"] for c in hit.candidates]
    served = {tuple(l) for l in hit.lines}
    gold_lines = {tuple(l) for l in q["gold_lines"]}
    right = hit.status == "clear" and q["event"] in hit.members
    return {"top1": q["event"] in hit.members if hit.members else False, "top5": q["event"] in ranks[:5] + hit.members,
            "status": hit.status, "kind_ok": hit.kind == q["ask"], "clear": hit.status == "clear", "right_event": right,
            "line": bool(gold_lines & served) if gold_lines else right}


def outcome(hit, q, reply) -> str:
    """The answer against what the graph holds for the asked event: when the host served the gold event (with any
    other readings of it), what it returned for them together; when it served another event, the gold values."""
    if hit.status != "clear":
        return {"ask": "asked", "none": "none", "unparsed": "unparsed"}[hit.status]
    if reply is None:
        return "refused"
    expected = hit.returned if q["event"] in hit.members else q["gold"]
    said_absent = reply.strip().lower().startswith(ABSENT_REPLY.lower().rstrip("."))
    if not expected:
        return "correct" if said_absent else "wrong"
    if said_absent:
        return "refused"
    return "correct" if any(has_value(reply, v) for v in expected) else "wrong"


def table(rows, label):
    log(f"\n[{label}] {len(rows)} questions")
    head = f"  {'kind':10s} {'n':>4s} {'top1':>6s} {'top5':>6s} {'clear':>6s} {'kind':>6s} {'line':>6s} | " \
           f"{'correct':>7s} {'wrong':>5s} {'refused':>7s} {'asked':>5s} {'none':>4s} {'unparsed':>8s}"
    log(head)
    groups = [(k, [r for r in rows if r["kind"] == k]) for k in KINDS] + [("ALL", rows)]
    out = {}
    for k, rs in groups:
        if not rs:
            continue
        n = len(rs)
        c = collections.Counter(r.get("outcome", "-") for r in rs)
        m = {x: sum(r[x] for r in rs) / n for x in ("top1", "top5", "clear", "kind_ok", "line")}
        m.update({o: c[o] for o in ("correct", "wrong", "refused", "asked", "none", "unparsed")}, n=n)
        out[k] = m
        log(f"  {k:10s} {n:4d} {m['top1']:6.1%} {m['top5']:6.1%} {m['clear']:6.1%} {m['kind_ok']:6.1%} {m['line']:6.1%} | "
            f"{c['correct'] / n:7.1%} {c['wrong']:5d} {c['refused']:7d} {c['asked']:5d} {c['none']:4d} {c['unparsed']:8d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(ROOT, "standin", "data", "out", "history_graph_v21.jsonl"))
    ap.add_argument("--questions", default=os.path.join(ROOT, "standin", "data", "out", "e8_questions.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--export", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    qs = [q for q in map(json.loads, open(args.questions, encoding="utf-8")) if q["split"] == args.split]
    t0 = time.time()
    g = HistoryGraph.load(args.graph)
    t1 = time.time()
    lk = HistoryLookup(g, min_score=args.min_score, margin=args.margin)
    t2 = time.time()
    lk._facts_index()                                   # built on first use; built here so no question's time holds it
    log(f"exp_e8 history retrieval | {len(qs)} {args.split} questions | graph {len(g.events):,} events "
        f"(load {t1 - t0:.0f}s, index {t2 - t1:.0f}s, {len(lk.name_event):,} names; facts index {time.time() - t2:.0f}s) | "
        f"min_score {args.min_score} margin {args.margin}")
    rows = []
    for q in qs:
        hit = lk.ask(q["question"])
        rows.append({"id": q["id"], "tier": q["tier"], "kind": q["kind"], "question": q["question"],
                     "copies_name": q.get("copies_name", False),
                     "gold_event": q["event_name"], "got": hit.name, "candidates": hit.candidates[:3], "kind_read": hit.kind,
                     "via": hit.via, "ms": round(hit.ms, 1), **retrieval(hit, q), "_hit": hit, "_q": q})
    log(f"lookup: median {sorted(r['ms'] for r in rows)[len(rows) // 2]:.0f} ms, max {max(r['ms'] for r in rows):.0f} ms")
    for tier in ("reworded", "named"):
        fr = [r for r in rows if r["tier"] == tier and r["clear"] and r["via"] == "facts"]
        log(f"found by the description's facts ({tier}): {len(fr)}, the right event {sum(r['right_event'] for r in fr)}")
    model_meta = None
    if not args.no_model:
        from tokenizers import Tokenizer
        from exp_e5_base450m_probes import load_grilly
        from exp_e6_talk_gate import answer, load_adapter
        tk = Tokenizer.from_file(os.path.join(args.export, "tokenizer.json"))
        eos = tk.token_to_id("</s>")
        model, _ = load_grilly(args.export)
        model_meta = load_adapter(model, args.adapter)
        log(f"adapter: step {model_meta.get('step')} r={model_meta['rank']} alpha={model_meta['alpha']}")
        t2 = time.time()
        for r in rows:
            hit, q = r["_hit"], r["_q"]
            reply, reason, draft = None, "", None
            if hit.status == "clear":
                if hit.returned:                             # the adapter reads the host's ask about the found event
                    draft = answer(model, tk, eos, render(hit.lines, hit.canonical))
                reply, reason = talk_reply(hit.returned, draft, [f"{e} {rel}: {v}" for e, rel, v in hit.lines] +
                                           [f"{v} is the {rel} of {e}" for e, rel, v in hit.lines], hit.name, hit.others)
            r.update(draft=draft, reply=reply, reason=reason, outcome=outcome(hit, q, reply))
        log(f"talk: {len(rows)} questions, {time.time() - t2:.0f}s")
    result = {"args": vars(args), "adapter_meta": model_meta}
    for tier in ("reworded", "named"):
        result[tier] = table([r for r in rows if r["tier"] == tier], f"{tier}, {args.split} split")
    result["reworded_no_name"] = table([r for r in rows if r["tier"] == "reworded" and not r["copies_name"]],
                                       f"reworded without the event's name, {args.split} split")
    rw = result["reworded"].get("ALL", {})
    if rw:
        bars = [("top1 >= 90%", rw["top1"] >= 0.90), ("line >= 90%", rw["line"] >= 0.90)]
        if not args.no_model:
            bars += [("correct >= 80%", rw["correct"] / rw["n"] >= 0.80), ("wrong = 0", rw["wrong"] == 0)]
        log("\ngate (reworded, " + args.split + "): " + "; ".join(f"{b} {'PASS' if ok else 'MISS'}" for b, ok in bars)
            + f"; kill (top5 < 50%): {'FIRES' if rw['top5'] < 0.5 else 'does not fire'}")
    for label, pick in (("wrong", lambda r: r.get("outcome") == "wrong"),
                        ("missed (gold not top1)", lambda r: not r["top1"] and r["tier"] == "reworded")):
        sel = [r for r in rows if pick(r)][:12]
        if sel:
            log(f"\n{label}, first {len(sel)}:")
            for r in sel:
                log(f"  [{r['kind']}/{r['tier']}] {r['question']}\n     gold: {r['gold_event']} | got: {r['got'] or r['status']}"
                    f" | top: {[c['name'] for c in r['candidates']]}" + (f"\n     said: {r.get('reply')}" if r.get('reply') else ""))
    for r in rows:
        r.pop("_hit"), r.pop("_q")
    result["rows"] = rows
    out = os.path.join(HERE, "logs", f"exp_e8_history_retrieval{args.tag}")
    json.dump(result, open(out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\nwrote {out}.json / .log")


if __name__ == "__main__":
    main()
