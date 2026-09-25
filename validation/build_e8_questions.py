"""H-E8's questions: held-split events of the history graph, each asked as named and reworded.

For each kind (cause, effect, when, where, who, downstream, absent) `--per-kind` events are drawn (seeded) from
the events whose name hashes to the held split (`build_ground_sft.split_of`: the talk adapter never trained on a
question about them) and that the graph can answer for that kind -- absent: an event with effects and no cause,
asked for its cause. when / where / who draw half from the newspaper's events and half from the books'; the
others are the books' (the newspaper has no links). The first `--dev` of each kind are the dev split, the only
questions the lookup's thresholds may be tuned on; the rest are the test split the gate reads.

Each event is asked twice:
  named     the training template with the graph's own name (build_history_sft's questions) -- the ceiling
  reworded  the same ask rewritten by a model the way a person would type it, told not to copy the name --
            dataset building for a test, the only use of an external model (OpenRouter, cached per prompt);
            a rewrite is dropped if it is not one question or copies a name of two or more words verbatim

The gold is what the graph returns for the gold event (history_lookup.serve): the values and the lines holding
them. Writes standin/data/out/e8_questions.jsonl and validation/logs/e8_questions.manifest.json (sha256 of the
questions and of the graph they were drawn from).

    python validation/build_e8_questions.py [--graph standin/data/out/history_graph_v21.jsonl] [--per-kind 80 --dev 20]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_ground_sft import split_of  # noqa: E402
from build_history_sft import CAUSE_Q, EFFECT_Q, WHEN_Q, WHERE_Q, WHO_Q, the  # noqa: E402
from history_graph import HistoryGraph, key, year_text  # noqa: E402
from history_lookup import parse_kind, serve  # noqa: E402

KINDS = ("cause", "effect", "when", "where", "who", "downstream", "absent")
ASKS = {"cause": "cause", "effect": "effect", "when": "when", "where": "where", "who": "who",
        "downstream": "downstream", "absent": "cause"}
TEMPLATES = {"cause": CAUSE_Q, "effect": EFFECT_Q, "when": WHEN_Q, "where": WHERE_Q, "who": WHO_Q,
             "downstream": ["What would change if {x} had not happened?"], "absent": CAUSE_Q}
ASKED = {"cause": "what caused it or led to it", "effect": "what it led to, its consequences",
         "when": "when it happened", "where": "where it happened", "who": "who took part in it",
         "downstream": "what would have been different if it had never happened", "absent": "what caused it or led to it"}
SYSTEM = """You rewrite questions about historical events the way an ordinary person would type them into a search box or ask a friend.
For each numbered item you get the event's name, when and where it happened, who took part, and WHAT IS ASKED about it.
Write ONE question per item that asks exactly that thing about exactly that event.
Refer to the event the way a person would -- by what happened, who was involved, where or when -- and do NOT copy the
event's name word for word. Never put the answer in the question: a question about when does not give the date, one
about where does not name the place, one about who does not name who took part.
One sentence, under 25 words, ending with a question mark.
Reply with ONLY a JSON list of strings, one per item, in the same order."""
RETRY = ("Your earlier rewrites of these items copied the event's name or gave the answer away. Rewrite each again: "
         "describe the event in other words, and leave the answer out.")
MODEL = "inclusionai/ling-3.0-flash"


def origin(ev) -> str:
    return "news" if ev.sources and all(s[0] == "nyt" for s in ev.sources) else "book"


def eligible(g: HistoryGraph, eid: str, kind: str) -> bool:
    ev = g.events[eid]
    if kind == "absent":
        return origin(ev) == "book" and not serve(g, eid, "cause")[1] and bool(serve(g, eid, "effect")[1])
    lines, returned, _ = serve(g, eid, kind)
    if not returned:
        return False
    if kind in ("cause", "effect", "downstream") and origin(ev) != "book":
        return False
    if kind == "downstream":
        return 2 <= len(returned) <= 6
    if kind == "where":
        return "," not in returned[0]
    return True


def answer_lines(lines, returned) -> list:
    r = set(returned)
    return [l for l in lines if l[2] in r or l[0] in r]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(g, eid, kind) -> str:
    """The event for the rewriter -- without the field the question asks for, so it cannot leak into it."""
    ev = g.events[eid]
    w = ev.when
    when = (w.get("day") or (year_text(w["y0"]) if w["y0"] == w["y1"] else f"{year_text(w['y0'])} to {year_text(w['y1'])}")) if w else "unknown"
    parts = [f"event: {ev.name}"]
    if kind != "when":
        parts.append(f"when: {when}")
    if kind != "where":
        parts.append(f"where: {', '.join(ev.where[:2]) or 'unknown'}")
    if kind != "who":
        parts.append(f"who: {', '.join(ev.who[:3]) or 'unknown'}")
    return " | ".join(parts + [f"asks: {ASKED[kind]}"])


def reword(llm, items: list[tuple[str, str]], g, note: str = "") -> list[str | None]:
    """items: [(eid, kind)] -> one rewrite each (None when the reply had none for it)."""
    user = (note + "\n\n" if note else "") + "\n".join(f"{n + 1}. {describe(g, eid, kind)}" for n, (eid, kind) in enumerate(items))
    reply = llm.chat(user)
    m = re.search(r"\[.*\]", reply or "", re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        out = []
    if len(out) != len(items):
        return [None] * len(items) if len(items) == 1 else [x for it in items for x in reword(llm, [it], g, note)]
    return [x if isinstance(x, str) else None for x in out]


def gives_answer(q: str, gold: list[str]) -> bool:
    kq = f" {key(q)} "
    for v in gold:
        v = str(v)
        if re.fullmatch(r"-?\d{4}-\d{2}-\d{2}", v):
            v = v.lstrip("-")[:4]                      # a date's year in the question gives it away
        if key(v) and f" {key(v)} " in kq:
            return True
    return False


def valid(q: str | None, name: str, gold: list[str]) -> str:
    """'' when the rewrite is kept, else why not ('copies_name' is kept, flagged, when a second try copies too)."""
    if not q:
        return "missing"
    q = q.strip()
    if not q.endswith("?") or not 3 <= len(q.split()) <= 35 or q.count("?") > 1:
        return "not_one_question"
    if gives_answer(q, gold):
        return "gives_answer"
    k = key(name)
    if len(k.split()) >= 2 and f" {k} " in f" {key(q)} ":
        return "copies_name"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default=str(ROOT / "standin" / "data" / "out" / "history_graph_v21.jsonl"))
    ap.add_argument("--per-kind", type=int, default=80)
    ap.add_argument("--dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "e8_questions.jsonl"))
    args = ap.parse_args()
    t0 = time.time()
    g = HistoryGraph.load(args.graph)
    print(f"graph: {len(g.events):,} events, {len(g.links):,} links ({time.time() - t0:.0f}s)", flush=True)
    rng = random.Random(args.seed)
    held = [eid for eid, ev in g.events.items() if split_of(ev.name) == "held"]
    rng.shuffle(held)
    picked, taken, funnel = [], set(), collections.Counter()
    for kind in KINDS:
        want = {"news": args.per_kind // 2, "book": args.per_kind - args.per_kind // 2} if kind in ("when", "where", "who") \
            else {"book": args.per_kind}
        got = collections.Counter()
        for eid in held:
            if sum(got.values()) >= args.per_kind:
                break
            o = origin(g.events[eid])
            if eid in taken or got[o] >= want.get(o, 0) or not eligible(g, eid, kind):
                continue
            got[o] += 1
            taken.add(eid)
            picked.append((eid, kind))
        funnel[f"picked:{kind}"] = sum(got.values())
        print(f"  {kind}: {dict(got)}", flush=True)

    from openrouter import OpenRouterProposer
    from build_history_graph import reasoning_of
    llm = OpenRouterProposer(MODEL, system=SYSTEM, max_tokens=2000, reasoning=reasoning_of("off"), offline=args.offline)
    gold = {(eid, kind): serve(g, eid, ASKS[kind])[1] for eid, kind in picked}
    rewrites = []
    for b in range(0, len(picked), args.batch):
        rewrites += reword(llm, picked[b:b + args.batch], g)
    again = [n for n, ((eid, kind), rq) in enumerate(zip(picked, rewrites)) if valid(rq, g.events[eid].name, gold[(eid, kind)])]
    funnel["second_try"] = len(again)
    for b in range(0, len(again), args.batch):
        idx = again[b:b + args.batch]
        for n, rq in zip(idx, reword(llm, [picked[n] for n in idx], g, RETRY)):
            eid, kind = picked[n]
            if valid(rq, g.events[eid].name, gold[(eid, kind)]) in ("", "copies_name"):
                rewrites[n] = rq                          # the second try, where it is at least as good
    rows, per_kind = [], collections.Counter()
    for (eid, kind), rq in zip(picked, rewrites):
        ev = g.events[eid]
        lines, returned, _ = serve(g, eid, ASKS[kind])
        n = per_kind[kind]
        per_kind[kind] += 1
        base = {"kind": kind, "ask": ASKS[kind], "split": "dev" if n < args.dev else "test", "event": eid,
                "event_name": ev.name, "origin": origin(ev), "gold": returned, "gold_lines": answer_lines(lines, returned)}
        named = rng.choice(TEMPLATES[kind]).format(x=the(ev.name))
        rows.append({"id": f"e8:{kind}:{n}:named", "tier": "named", "question": named, **base})
        why = valid(rq, ev.name, returned)
        funnel[f"reworded:{why or 'kept'}"] += 1
        if why in ("", "copies_name"):
            rows.append({"id": f"e8:{kind}:{n}:reworded", "tier": "reworded", "question": rq.strip(),
                         "copies_name": why == "copies_name", **base})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parsed = collections.Counter(f"{r['tier']}:{'ok' if parse_kind(r['question'])[0] == r['ask'] else 'other'}" for r in rows)
    counts = collections.Counter(f"{r['split']}:{r['tier']}:{r['kind']}" for r in rows)
    manifest = {"graph": pathlib.Path(args.graph).name, "graph_sha256": sha256(args.graph), "questions": pathlib.Path(args.out).name,
                "questions_sha256": sha256(args.out), "rows": len(rows), "per_kind": args.per_kind, "dev": args.dev,
                "seed": args.seed, "model": MODEL, "usage": llm.usage, "funnel": dict(funnel),
                "kind_read_by_the_host": dict(parsed), "counts": dict(sorted(counts.items())), "wall_s": round(time.time() - t0, 1)}
    (HERE / "logs" / "e8_questions.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "usage", "funnel", "kind_read_by_the_host")}, indent=1))
    for r in rng.sample([r for r in rows if r["tier"] == "reworded"], min(14, len(rows))):
        print(f"  [{r['kind']}] {r['event_name']}  ->  {r['question']}")


if __name__ == "__main__":
    main()
