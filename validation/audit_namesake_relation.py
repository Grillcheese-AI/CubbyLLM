"""audit_namesake_relation -- order 0037's acceptance: the rule that decides a shared name by the question's relation
and its cue words (``hdc_relation.decide``, on by default since 2026-10-27), on the real hdc world, for Cowork to
read by hand.

1. **The audit:** 50 shared-name hops, drawn by seed 1 (0036's audit drew seed 0). Each label gets one relation,
   drawn from the relations its items hold, so the hops cover relations one item holds and relations several do.
   Each hop is listed with:
   - the label and the relation;
   - every item, with its kinds (recorded or read) and a few of its own values;
   - the rule's outcome (one item, each item labelled, or asking back);
   - the answer as spoken, phrased by the loop's own functions (``ask_namesake_answer``). The loop's time checks
     come on top: a changing value from this 2019 snapshot is said "As of 2019, ...".
   The line: at most 1 wrong pick in 50. A wrong pick is one item chosen when a person asking could as well mean
   another. Asking back where the question was obvious is a cost, counted apart.
2. **The cue-word probe:** 20 shared names, drawn by seed 1, among those with two or more items of a known kind. Each
   is asked about a relation one item (the target) holds, with a cue word taken from the target's own values.
   Each must pick the target or ask back: 0 wrong picks.
3. **The fixed checks:** France's capital, Japan's currency, Italy's capital, "What Nobel Prize did Marie Curie win?"
   (the cue "nobel"; the person, Q7186, holds no Nobel fact here, so the reply is that it does not know), and Steve
   Mills (two of a kind).

Each decision's time is recorded (the order's latency line).

    python validation/audit_namesake_relation.py [--n 50] [--probe 20] [--seed 1]
-> validation/logs/audit_namesake_relation_<date>.json (and .md: the lists to read)

Wired: STANDALONE (a measurement).
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

os.environ["CB_NAMESAKE_RELATION"] = "1"                 # the rule under audit (on by default since 2026-10-27; this pins it on)

from cubbyllm.reasoning import ask_namesake_answer as A  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan, normalize  # noqa: E402
from cubbyllm.worlds.hdc_cues import words  # noqa: E402
from cubbyllm.worlds.hdc_relation import item_kinds, others, qnum  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld  # noqa: E402

__wiring__ = "STANDALONE"

FIXED = [("capital", "France", ()), ("currency", "Japan", ()), ("capital", "Italy", ()),
         ("award received", "Marie Curie", ("nobel",)), ("country of citizenship", "Steve Mills", ())]


def hop(w: HdcWorld, rel: str, name: str, cues=()):
    """(values read, the decision, ms)."""
    w.clear_withheld()
    t0 = time.perf_counter()
    with w.cued(list(cues)):
        got = w.hop(QuestionPlan(relations=[None], tail=f"{rel} of {name}", n_hop=1), 0, None)
    ms = (time.perf_counter() - t0) * 1000
    return sorted({t.obj for _f, t in got}), w.decision(name), round(ms, 1)


def spoken(w: HdcWorld, d: dict | None, by: dict, reads: list, question: str, seed: str) -> tuple[str, str]:
    """(outcome, the reply as the loop phrases it)."""
    if d is None:
        return "no decision", ""
    fit = d["fit"]
    if len(fit) == 1 and not d.get("holds"):
        return "asking back (the one that fits holds none)", f"Which one do you mean? ({len(by)} items named {seed})"
    if len(fit) == 1:
        if not reads:
            return "one item, holds no such value", "I don't know."
        want = A.class_words(question, seed)
        said = ", ".join(reads)
        if want and not all(x in said.lower() for x in want):
            return "one item, not what was asked", f"I don't know (its value, {said}, is not a {' '.join(want)})."
        line = A.also_line(others(d, by, w._kinds))
        return "one item", f"{said}." + (f" {line}" if line else "")
    labels = A.label_for(fit, by, d.get("rels") or ()) if 1 < len(fit) <= A.EACH_MAX else None
    if labels:
        rels = set(d["rels"])
        parts = [f"{labels[q]}: {', '.join(sorted({t.obj for _f, t in by[q] if t.rel in rels})) or 'I do not know'}"
                 for q in fit]
        return "each item labelled", "; ".join(parts) + "."
    return "asking back", f"Which one do you mean? ({len(by)} items named {seed})"


def items_of(w: HdcWorld, d: dict | None, by: dict) -> list[dict]:
    out = []
    for q in sorted(by, key=qnum):
        got = (d or {}).get("kinds", {}).get(q)
        if got is None:                                  # (no decision: read the kinds as the rule would)
            ks, how = item_kinds(by[q], w._kinds)
            got = [sorted(ks), how]
        ks, how = got
        mark = ("fits (placed)" if q in (d or {}).get("placed", []) else      # (no kind; a fact places it: 2026-10-27)
                "fits" if q in (d or {}).get("fit", []) else
                f"ruled out by '{d['ruled_out'][q]}'" if q in (d or {}).get("ruled_out", {}) else "does not fit")
        out.append({"item": q, "kinds": ks, "how": how, "mark": mark,
                    "values": [f"{t.rel}: {t.obj}" for _f, t in by[q] if normalize(t.rel) != "instance of"][:4]})
    return out


def shared_labels(w: HdcWorld, seed: int) -> list[str]:
    labels = sorted(k for (k,) in w.db.execute("SELECT skey FROM facts GROUP BY skey HAVING COUNT(DISTINCT sq) >= 2")
                    if len(k) >= 3 and not k.isdigit())
    random.Random(seed).shuffle(labels)
    return labels


def audit(w: HdcWorld, labels: list[str], n: int, seed: int) -> list[dict]:
    rng, rows = random.Random(seed), []
    for key in labels:
        if len(rows) >= n:
            break
        by = w.namesakes(key)
        rels = sorted({t.rel for its in by.values() for _f, t in its if normalize(t.rel) != "instance of"})
        if len(by) < 2 or not rels:
            continue
        rel = rng.choice(rels)
        name = next(iter(by.values()))[0][1].subj
        q = f"What is the {rel} of {name}?"
        reads, d, ms = hop(w, rel, name)
        outcome, reply = spoken(w, d, by, reads, q, name)
        rows.append({"question": q, "label": name, "relation": rel, "items_n": len(by), "outcome": outcome,
                     "reply": reply, "fit": (d or {}).get("fit"), "ms": ms, "items": items_of(w, d, by)[:8]})
    return rows


def probe(w: HdcWorld, labels: list[str], n: int, seed: int) -> list[dict]:
    rng, rows = random.Random(seed), []
    for key in labels:
        if len(rows) >= n:
            break
        by = w.namesakes(key)
        known = [q for q in sorted(by, key=qnum) if item_kinds(by[q], w._kinds)[0]]
        if len(by) < 2 or len(known) < 2:
            continue
        target = rng.choice(known)
        name = by[target][0][1].subj
        own = set(words(name))
        cue = sorted({x for _f, t in by[target] for x in words(t.obj)
                      if x not in own and w._cues is not None and w._cues.kinds(x) is not None})
        rels = sorted({t.rel for _f, t in by[target] if normalize(t.rel) != "instance of"})
        if not cue or not rels:
            continue
        word, rel = rng.choice(cue), rng.choice(rels)
        reads, d, ms = hop(w, rel, name, [word])
        fit = (d or {}).get("fit") or []
        outcome, reply = spoken(w, d, by, reads, f"What is the {rel} of {name}?", name)
        picked = outcome.startswith("one item")              # (a single fit that holds none decides nothing)
        verdict = ("picks the target" if picked and fit == [target] else
                   "WRONG PICK" if picked else
                   f"{outcome}, the target among them" if target in fit else
                   f"{outcome}, the target ruled out by the cue" if target in (d or {}).get("ruled_out", {}) else outcome)
        rows.append({"label": name, "relation": rel, "cue": word, "target": target, "verdict": verdict, "fit": fit,
                     "reads": reads, "reply": reply, "ms": ms, "items": items_of(w, d, by)[:8]})
    return rows


def fixed(w: HdcWorld) -> list[dict]:
    rows = []
    for rel, name, cues in FIXED:
        by = w.namesakes(name)
        reads, d, ms = hop(w, rel, name, cues)
        q = "What Nobel Prize did Marie Curie win?" if cues else f"What is the {rel} of {name}?"
        outcome, reply = spoken(w, d, by, reads, q, name)
        rows.append({"question": q, "cues": list(cues), "fit": (d or {}).get("fit"), "outcome": outcome,
                     "reply": reply, "ms": ms, "items_n": len(by)})
    return rows


def write_md(path: str, out: dict) -> None:
    md = [f"# Order 0037: the relation rule on shared names (seed {out['seed']}, {out['date']})", "",
          "## The fixed checks", ""]
    for r in out["fixed"]:
        md.append(f"- **{r['question']}** ({r['items_n']} items) -> {r['outcome']}: {r['reply']} (fit {r['fit']})")
    md += ["", f"## The audit: {len(out['audit'])} hops",
           "Mark each line: right; wrong pick (one item chosen when a person could as well mean another); or asked "
           "back where the question was obvious (a cost).", ""]
    for i, r in enumerate(out["audit"], 1):
        md.append(f"{i}. **{r['question']}** ({r['items_n']} items): {r['outcome']}. Said: {r['reply']}")
        md += [f"   - {it['mark']}: {it['item']} ({', '.join(it['kinds']) or 'no kind'}; {it['how']}): "
               f"{'; '.join(it['values'])}" for it in r["items"]]
    md += ["", f"## The cue-word probe: {len(out['probe'])} names", "Each must pick the target or ask back.", ""]
    for i, r in enumerate(out["probe"], 1):
        md.append(f"{i}. **{r['label']}**, {r['relation']}, cue '{r['cue']}': {r['verdict']} (target {r['target']}, "
                  f"fit {r['fit']}). Said: {r['reply']}")
        md += [f"   - {it['mark']}: {it['item']} ({', '.join(it['kinds']) or 'no kind'}; {it['how']}): "
               f"{'; '.join(it['values'])}" for it in r["items"]]
    md += ["", "## Counts", "", "```", json.dumps(out["counts"], indent=1), "```"]
    open(path, "w", encoding="utf-8").write("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--probe", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--cues", default=None, help="another cue table to read (a rebuild not yet swapped in)")
    a = ap.parse_args()
    w = HdcWorld()
    if a.cues:
        from cubbyllm.worlds.hdc_cues import Cues
        w._cues = Cues(a.cues)
    if w._kinds is None or w._cues is None:
        sys.exit("the kinds table or the cue table is missing beside the world (hdc_kinds.build, hdc_cues.build)")
    labels = shared_labels(w, a.seed)
    out = {"what": "order 0037: the relation rule on shared hdc names, for a reading by hand", "date": a.date,
           "seed": a.seed, "fixed": fixed(w), "audit": audit(w, labels, a.n, a.seed), "probe": probe(w, labels, a.probe, a.seed)}
    ms = [r["ms"] for part in ("fixed", "audit", "probe") for r in out[part]]
    out["counts"] = {"audit_outcomes": {o: sum(r["outcome"] == o for r in out["audit"]) for o in sorted({r["outcome"] for r in out["audit"]})},
                     "probe_verdicts": {v: sum(r["verdict"] == v for r in out["probe"]) for v in sorted({r["verdict"] for r in out["probe"]})},
                     "decision_ms": {"p50": statistics.median(ms), "max": max(ms), "n": len(ms)}}
    path = os.path.join(HERE, "logs", f"audit_namesake_relation_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_md(path[:-5] + ".md", out)
    print(json.dumps(out["counts"], indent=1), "->", path)


if __name__ == "__main__":
    main()
