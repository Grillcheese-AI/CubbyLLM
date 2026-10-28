"""audit_source_namesakes -- order 0043's acceptance: the rule that decides a shared name at the Wikidata source by the
question's relation (``sources.wikidata_namesakes``, behind ``CB_SOURCE_NAMESAKES=1``), for Cowork to read by hand.

**The 50 names.** Every name among the scoreboard's 187 (its last live run) and the stack's 20 that the strict rule
refuses as shared at the source (two or more exact hits left), with the question's own relation; then names drawn by
seed 2 from the hdc world's namesakes, each with a relation one of its items holds, asked against the live source,
until there are 50. Each is asked through the ask loop with the rule on -- an empty first world and no hdc world
behind it, so the source alone decides -- and listed with:
- the exact hits, their kinds (P31) and whether each carries the relation (holds / may / placed), or is ruled out;
- the rule's outcome (one fits, several fit answered each, they agree, asking back, no fit);
- the answer as spoken.
The line: at most 1 wrong pick in 50, and 0 values tied to the wrong item in any "answer each".

Latency: the decision's own time, and the source calls each question spent.

    python validation/audit_source_namesakes.py [--n 50] [--seed 2] [--run validation/logs/<scoreboard>.json]
-> validation/logs/audit_source_namesakes_<date>.json (and .md: the list to read)

Wired: STANDALONE (a measurement; the live source through cubby_serve's fetch).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.planner import normalize  # noqa: E402

__wiring__ = "STANDALONE"

LAST_RUN = os.path.join(HERE, "logs", "everyday_webq_on_live_tense_names2_2026-10-24.json")


class PlanEmitter:
    """Writes the plan for the one question asked (seed, relation), as the 450M would on a good day."""
    name = "plan"

    def __init__(self) -> None:
        self.plans: dict[str, tuple[str, list[str]]] = {}

    def emit(self, prompt: str, max_new_tokens: int = 300, **kw) -> str:
        p = self.plans.get(" ".join(prompt.split()))
        if p is None:
            return "no plan\n"
        seed, rels = p
        lines = [f'    bind frame, SEED, "{seed}";'] + [f'    bind frame, HOP{i + 1}, "{r}";' for i, r in enumerate(rels)]
        return "function plan(): str {\n" + "\n".join(lines) + "\n    return recover(frame, SEED);\n}\n"


def live_source():
    from cubbyllm.sources.wikidata import WikidataSource
    from cubby_serve.sources.http import wikidata as fetch
    return WikidataSource(offline=False, use_choices=False, fetch=fetch())


def shared_at_source(src, name: str, rel: str) -> bool:
    """Strict: does the source refuse `name` as shared (two or more exact hits left) for a hop on `rel`?"""
    os.environ["CB_SOURCE_NAMESAKES"] = "0"
    try:
        src.resolve(name, relations=[rel])
        return bool(src.last.get("ambiguous"))
    finally:
        os.environ["CB_SOURCE_NAMESAKES"] = "1"


def from_runs(src, path: str) -> list[dict]:
    from validation.stack_answers import QUESTIONS
    rows = []
    if os.path.exists(path):
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            if r.get("seed") and r.get("plan"):
                rows.append({"question": r["question"], "seed": r["seed"], "rel": r["plan"][0], "from": "scoreboard"})
    from cubbyllm.reasoning.everyday import EverydayForms
    forms = EverydayForms()
    for q in QUESTIONS:
        p = forms.restate(q)
        if p:
            rows.append({"question": q, "seed": p[2], "rel": p[1], "from": "stack"})
    seen, out = set(), []
    for r in rows:
        key = (normalize(r["seed"]), normalize(r["rel"]))
        if key in seen:
            continue
        seen.add(key)
        if shared_at_source(src, r["seed"], r["rel"]):
            out.append(r)
    return out


def from_hdc(src, n: int, seed: int, skip: set) -> list[dict]:
    from cubbyllm.worlds.hdc_store import HdcWorld
    from validation.audit_namesake_relation import shared_labels
    w = HdcWorld()
    rng, out = random.Random(seed), []
    for key in shared_labels(w, seed):
        if len(out) >= n:
            break
        by = w.namesakes(key)
        rels = sorted({t.rel for its in by.values() for _f, t in its if normalize(t.rel) != "instance of"})
        if len(by) < 2 or not rels:
            continue
        rel, name = rng.choice(rels), next(iter(by.values()))[0][1].subj
        if (normalize(name), normalize(rel)) in skip or not shared_at_source(src, name, rel):
            continue
        out.append({"question": f"What is the {rel} of {name}?", "seed": name, "rel": rel, "from": f"hdc seed {seed}"})
    return out


def ask_all(src, rows: list[dict]) -> list[dict]:
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.worlds.base import FactStore
    em = PlanEmitter()
    loop = AskLoop(em, world=FactStore(name="audit"), source=src, lexicon=True, history_path=None)
    times = []
    real = src.decide

    def timed(entity, cands, relations):
        t0 = time.perf_counter()
        try:
            return real(entity, cands, relations)
        finally:
            times.append((time.perf_counter() - t0) * 1000)
    src.decide = timed
    for r in rows:
        em.plans[" ".join(r["question"].split())] = (r["seed"], [r["rel"]])
        calls0, t0 = src.calls, time.perf_counter()
        src.clear_decisions()
        try:
            rec = loop.ask(r["question"])
        except Exception as e:                           # noqa: BLE001 -- a crash is a row, not the end of the audit
            rec = {"answer": None, "verified": False, "reason": f"error: {str(e)[:160]}"}
        d = src.namesake_decision(r["seed"]) or {}
        ns = rec.get("namesakes") or {}
        r.update(reply=rec.get("reply") or rec.get("answer"), verified=bool(rec.get("verified")), reason=rec.get("reason"),
                 outcome=ns.get("rule") or ("asking back" if rec.get("reason") == "ambiguous_entity" else
                                            "answered (no rule needed)" if rec.get("verified") else f"refused: {rec.get('reason')}"),
                 fit=d.get("fit"), how=d.get("how"), ruled_out=d.get("ruled_out"), cues=d.get("cues"),
                 hits=[{"item": q, "label": d["labels"][q], "kinds": d["kinds"].get(q),
                        "mark": (f"fits ({d['how'][q]})" if q in d.get("fit", []) else
                                 f"ruled out by '{d['ruled_out'][q]}'" if q in d.get("ruled_out", {}) else "does not fit")}
                       for q in d.get("labels", {})],
                 answers=rec.get("answers"), source_calls=src.calls - calls0, wall_s=round(time.perf_counter() - t0, 2))
        print(f"  {r['outcome']:28} | {r['question'][:50]:50} | {str(r['reply'])[:70]}", flush=True)
    return rows, times


def markdown(rows: list[dict], summary: dict) -> str:
    out = [f"# Order 0043: shared names at the source, {summary['date']} (seed {summary['seed']})", "",
           "Each name: the question, the exact hits with their kinds and how each fits (holds: its own claims carry the relation; "
           "may: the property's subject-type constraint allows its kind; placed: a place relation and a fact that places it), "
           "the rule's outcome, the answer as spoken. Mark each line: right; wrong pick; a value tied to the wrong item; "
           "or asked back where the question was obvious.", ""]
    for i, r in enumerate(rows, 1):
        said = r["reply"] if r.get("verified") else f"nothing ({r.get('reason')})"
        out.append(f"{i}. **{r['question']}** ({len(r.get('hits') or [])} exact hits; {r['from']}): {r['outcome']}. Said: {said}")
        for h in r.get("hits") or []:
            out.append(f"   - {h['mark']}: {h['item']} {h['label']} ({', '.join(h['kinds'] or []) or 'no kind'})")
        for a in r.get("answers") or []:
            out.append(f"   - answered: {a['label']}: {a['said'] if a['verified'] else 'I do not know'} ({a['item']})")
    out += ["", "## Summary", "```", json.dumps(summary, indent=1), "```"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--run", default=LAST_RUN)
    a = ap.parse_args()
    src = live_source()
    rows = from_runs(src, a.run)
    print(f"{len(rows)} shared names among the scoreboard's and the stack's", flush=True)
    rows = rows[:a.n]
    if len(rows) < a.n:
        rows += from_hdc(src, a.n - len(rows), a.seed, {(normalize(r["seed"]), normalize(r["rel"])) for r in rows})
    os.environ["CB_SOURCE_NAMESAKES"] = "1"
    rows, times = ask_all(src, rows)
    import collections
    summary = {"what": "audit_source_namesakes (order 0043)", "date": _dt.date.today().isoformat(), "seed": a.seed,
               "n": len(rows), "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
               "from": dict(collections.Counter(r["from"] for r in rows)),
               "decision_ms": {"p50": round(statistics.median(times), 1) if times else None, "max": round(max(times), 1) if times else None, "n": len(times)},
               "source_calls_per_question": {"p50": statistics.median(r["source_calls"] for r in rows) if rows else None,
                                             "max": max((r["source_calls"] for r in rows), default=None)}}
    stem = os.path.join(HERE, "logs", f"audit_source_namesakes_{summary['date']}")
    json.dump({"summary": summary, "rows": rows}, open(stem + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(stem + ".md", "w", encoding="utf-8").write(markdown(rows, summary))
    print(json.dumps(summary, indent=1), "->", stem + ".json", flush=True)


if __name__ == "__main__":
    main()
