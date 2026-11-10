"""audit_wd_kinds -- order 0055 step 3, option (a): order 0037's relation rule on the hot Wikidata store's shared names,
read with the store's own kinds table (`cubbyllm/worlds/wd_store_kinds.py`), for Cowork to read by hand before the
rule speaks (``CB_WD_KINDS=1`` stays off until then).

The same audit as 0037's (`audit_namesake_relation.py`, whose functions this reuses), on the store's world:
1. **The fixed checks:** France's capital, Japan's currency, Italy's capital, Marie Curie's award, Steve Mills's
   citizenship. In the full store these names are 50, 33, 36, 18 and 10 items.
2. **The audit:** 50 shared-name hops. The names are drawn as 0037's were (the Wikidata5M world's shared labels,
   seed 1), so the two audits read side by side; each gets one relation drawn from the relations its Wikidata5M
   items hold that the store also has. A name the store gives one item or none is skipped. Each hop lists the items
   the rule weighed (for a name of more than 25 items: the candidates the kinds narrowed it to, `wd_store_crowd`),
   their kinds, a few values, the outcome and the reply as the loop phrases it.
   The line (0037's): at most 1 wrong pick in 50. A wrong pick is one item chosen when a person asking could as well
   mean another. Asking back where the question was obvious is a cost, counted apart.
There is no cue-word probe: the store's world has no cue table (a cue rules nothing out there).

    python validation/audit_wd_kinds.py [--n 50] [--seed 1]
-> validation/logs/audit_wd_kinds_<date>.json (and .md: the lists to read)

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

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["CB_WD_KINDS"] = "1"

import audit_namesake_relation as A0  # noqa: E402  (sets CB_NAMESAKE_RELATION=1: the rule under audit)

from cubbyllm.reasoning.planner import normalize  # noqa: E402
from cubbyllm.worlds.hdc_relation import PLACING  # noqa: E402
from cubbyllm.worlds.hdc_store import HdcWorld  # noqa: E402

__wiring__ = "STANDALONE"


def store_by(w, name: str, rel: str) -> tuple[dict, int]:
    """(the items the rule weighs for this hop, the name's item count): every item for a name of 25 or fewer, the
    kinds' candidates for a crowded one."""
    labels, aliases = w._tiers(normalize(name))
    ids = labels or aliases
    if len(ids) <= w.NAMESAKE_CAP:
        return w.namesakes(name), len(ids)
    rels = w._world_rels(rel)
    cand = w._candidates(ids, rels) if rels and not rels & PLACING else None
    return {q: w._triples(w._entity(q)[1]) for q in cand or [] if w._entity(q)[1]}, len(ids)


def one(w, rel: str, name: str, cues=()) -> dict:
    by, n = store_by(w, name, rel)
    reads, d, ms = A0.hop(w, rel, name, cues)
    q = "What Nobel Prize did Marie Curie win?" if cues else f"What is the {rel} of {name}?"
    outcome, reply = A0.spoken(w, d, by, reads, q, name) if d else (
        ("one item", f"{', '.join(reads)}.") if reads else ("nothing read", "I don't know."))
    return {"question": q, "label": name, "relation": rel, "items_n": n, "weighed": len(by), "outcome": outcome,
            "reply": reply, "fit": (d or {}).get("fit"), "ms": ms, "items": A0.items_of(w, d, by)[:8]}


def audit(w, hdc: HdcWorld, n: int, seed: int) -> list[dict]:
    rng, rows = random.Random(seed), []
    for key in A0.shared_labels(hdc, seed):
        if len(rows) >= n:
            break
        hby = hdc.namesakes(key)
        rels = sorted({t.rel for its in hby.values() for _f, t in its
                       if normalize(t.rel) != "instance of" and normalize(t.rel) in w._rel_n})
        if not rels:
            continue
        rel = rng.choice(rels)
        name = next(iter(hby.values()))[0][1].subj
        labels, aliases = w._tiers(normalize(name))
        if len(labels or aliases) < 2:
            continue                                     # one item or none in the store: no shared name to decide
        rows.append(one(w, rel, name))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    from cubby_serve.wd_world import build
    w = build()
    if w._kinds is None:
        sys.exit("no kinds table for this store (wd_store_kinds)")
    out = {"what": "order 0055 step 3 (a): order 0037's relation rule on the store's shared names, for a reading by hand",
           "date": a.date, "seed": a.seed, "store": w.state, "fixed": [one(w, r, nm, c) for r, nm, c in A0.FIXED],
           "audit": audit(w, HdcWorld(), a.n, a.seed), "probe": []}
    ms = [r["ms"] for part in ("fixed", "audit") for r in out[part]]
    out["counts"] = {"audit_outcomes": {o: sum(r["outcome"] == o for r in out["audit"]) for o in sorted({r["outcome"] for r in out["audit"]})},
                     "crowded": sum(r["items_n"] > w.NAMESAKE_CAP for r in out["audit"]),
                     "decision_ms": {"p50": statistics.median(ms), "max": max(ms), "n": len(ms)}}
    path = os.path.join(HERE, "logs", f"audit_wd_kinds_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    A0.write_md(path[:-5] + ".md", out)
    md = open(path[:-5] + ".md", encoding="utf-8").read().split("\n", 1)[1]            # its own title, not 0037's
    open(path[:-5] + ".md", "w", encoding="utf-8").write(
        f"# Order 0055 step 3 (a): the relation rule on the Wikidata store's shared names (seed {a.seed}, {a.date})\n" + md)
    print(json.dumps(out["counts"], indent=1), "->", path)


if __name__ == "__main__":
    main()
