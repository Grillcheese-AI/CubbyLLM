"""exp_r35 -- real families: the skill library on Wikidata's own kinship statements. WO-2.6 follow-up.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHAT THIS IS
------------
exp_r34 learned kinship rules on CLUTRR, a synthetic world. Rules learned there should not answer questions
about real people: in CLUTRR a wife's son is a son; in a real family he may be a stepson. So here the episodes
are REAL, from `validation/wikidata_kin.py`:

  * the gold: Wikidata's "relative" statements with a "kinship to subject" qualifier (P1038 + P1039) -- an
    editor saying how two real people are related;
  * the chains: the family graph Wikidata holds around them (father, mother, child, sibling, spouse; sex
    or gender names the relation: a male child is a son), each path from the one to the other;
  * the grain: the kinship terms' own hierarchy (P279) -- "father's brother" is an "uncle".

Day 1 is the TRAIN pairs as refused relation asks with that gold; one sleep night; day 2 the TEST pairs
(split by the subject, so no subject is on both days), each composed over every path through the real VM.

SCORING, never accuracy
-----------------------
    correct   the term spoken is the term Wikidata states
    coarser   the spoken term is entailed by the stated one ("uncle" where Wikidata says "father's brother"):
              true, less specific
    finer     the spoken term entails the stated one ("paternal grandfather" for "grandfather"): not
              contradicted, not confirmed either -- reported apart
    WRONG     neither entails the other
    refused   and why

ARMS
----
    rules            the night's library, learned on real train pairs
    null             the last relation of every path relabelled with a token nothing stated: a spoken answer
                     is a BREACH
    shuffled         instrument check: conclusions permuted; the harness must see wrongs
    clutrr_transfer  DIAGNOSTIC: the library a CLUTRR night learned (exp_r34's last run), on real families --
                     what it costs to carry rules out of the world whose episodes made them
    reversed         the test pairs Wikidata ALSO states the other way round ("s is the K2 of r"), asked that
                     way: the same rules, a second question set
    inverse          the same reversed questions answered only from the paths that lead from s to r, turned
                     round by an INVERSE rule for s's sex (skills.relate_back). Inverse rules are learned
                     from the train pairs stated both ways and from family edges Wikidata states on both
                     people (a father on the child, the child on the father) -- never from our own inverses

KILL: any WRONG in `rules`, `reversed` or `inverse`, any spoken answer in `null`; `shuffled` with no WRONG means the
instrument is blind.

    python validation/exp_r35_wikidata_kinship.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
OUT = ROOT / "standin" / "data" / "out" / "exp_r35"
KIN = ROOT / "standin" / "data" / "out" / "wikidata_kin" / "kin.json"
CLUTRR_LIB = ROOT / "standin" / "data" / "out" / "exp_r34" / "rules" / "skills.jsonl"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NULL_TOKEN = "zubnog"
TAU = {1: 1.0, 2: 0.4736328125}
MALE = {"nephew", "uncle", "grandson", "grandfather", "son in law", "brother in law", "father in law", "brother",
        "son", "stepson", "cousin"}
FEMALE = {"niece", "aunt", "granddaughter", "grandmother", "daughter in law", "sister in law", "mother in law",
          "sister", "daughter", "stepdaughter"}


def store_facts(kin: dict) -> list[str]:
    """Every family edge as facts the rules speak of, both ways (`skills.family_facts`: named by sex, with the
    edge back that Wikidata's own inverse declarations give) -- the same function the live loop uses."""
    from cubbyllm.reasoning.skills import family_facts
    out = set()
    for x, es in kin["edges"].items():
        for relation, y in es:
            out.update(family_facts(x, relation, y, kin["gender"]))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-share", type=int, default=5, help="1 in N subjects is a test subject")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--shuffled-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning import sleep as S
    from cubbyllm.reasoning.index import TripleIndex
    from cubbyllm.reasoning.skills import Library, Rule, rel, relate

    kin = json.loads(KIN.read_text(encoding="utf-8"))
    names = kin["labels"]
    term = {q: rel(names.get(q) or q) for q in kin["taxonomy"]}
    entails = {term[k]: sorted({rel(names.get(s) or s) for s in sups}) for k, sups in kin["taxonomy"].items()}
    facts = store_facts(kin)
    ix = TripleIndex(facts)
    log(f"Wikidata kinship: {len(kin['pairs'])} stated pairs, {len(kin['edges'])} people, {len(facts)} facts "
        f"(gendered, with Wikidata's declared inverses); {len(kin['kinds'])} kinship terms")
    grain = Library(entails=entails)

    # ── the pairs: one term each (comparable terms merge to the finest), and the direction checked ──
    items, dropped = [], collections.Counter()
    agree = collections.Counter()
    for p in kin["pairs"]:
        terms = [term.get(k, rel(names.get(k) or k)) for k in p["kinds"]]
        stated = grain.finest(terms)
        if stated is None:
            dropped["incomparable_terms"] += 1
            continue
        want = "male" if stated in MALE else "female" if stated in FEMALE else None
        if want:
            for who in ("r", "s"):
                if kin["gender"].get(p[who]):
                    agree[who] += kin["gender"][p[who]] == want
                    agree[who + "_n"] += 1
        items.append({"s": p["s"], "r": p["r"], "gold": stated,
                      "test": int(hashlib.sha256(p["s"].encode()).hexdigest(), 16) % a.test_share == 0})
    log(f"direction: the stated term's sex matches the RELATIVE {agree['r']}/{agree['r_n']}, the subject "
        f"{agree['s']}/{agree['s_n']} -> 'the relative is the <term> of the subject'"
        + ("" if agree["r"] * agree["s_n"] > agree["s"] * agree["r_n"] else "  !! THE OTHER WAY: check the loader"))
    log(f"pairs used {len(items)}; dropped {dict(dropped)}")

    for it in items:
        ps, over = ix.paths(it["s"], it["r"], max_hops=a.max_hops, limit=16, max_nodes=20000)
        it["paths"], it["overflow"] = ps, over
    train = [it for it in items if not it["test"]]
    test = [it for it in items if it["test"]]
    log(f"train {len(train)} pairs ({sum(1 for i in train if i['paths'])} with a path), test {len(test)} "
        f"({sum(1 for i in test if i['paths'])} with a path); path lengths "
        f"{dict(sorted(collections.Counter(len(p) for i in items for p in i['paths']).items()))}")

    # ── day 1 and the night ─────────────────────────────────────────────────────────────────────────
    if OUT.exists():
        for f in sorted(OUT.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "skills_taxonomy.json").write_text(json.dumps(entails, ensure_ascii=False), encoding="utf-8")
    rows = [{"id": f"{it['s']}-{it['r']}", "question": f"How is {it['r']} related to {it['s']}?", "kind": "relation",
             "verified": False, "reason": "no_rule", "paths": [[t.rel for t in p] for p in it["paths"]],
             "gold": it["gold"]} for it in train if it["paths"] and not it["overflow"]]
    (OUT / "day.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tn = time.perf_counter()
    sk = S.sleep([OUT / "day.jsonl"], OUT, night="wikidata-train")["skills"]
    lib = Library(OUT / "skills.jsonl")
    log(f"night: {sk['adopted']} rules adopted in {sk['rounds']} rounds from {sk['episodes_total']} episodes, "
        f"{sk['retired']} retired; rejected {sk['rejected']} ({time.perf_counter() - tn:.1f}s)")
    for c in sk.get("contested", [])[:8]:
        log(f"   contested: {c['premise']} -> {c.get('counts') or c.get('states')}")

    # ── inverse rules, from statements made both ways ─────────────────────────────────────────────────
    from cubbyllm.reasoning.skills import GENDERED, InverseEpisode, mine_inverses, relate_back
    sex = kin["gender"]
    rel_rows = kin.get("relatives") or {}

    def stated(x: str, y: str) -> str | None:
        """What Wikidata says y is to x (x's relative statement about y), at the finest comparable grain."""
        terms = [term.get(k, rel(names.get(k) or k)) for yy, ks in rel_rows.get(x, []) if yy == y for k in ks]
        return grain.finest(terms) if terms else None

    inv_eps = []
    for it in train:
        k2 = stated(it["r"], it["s"])
        if k2:
            inv_eps.append(InverseEpisode(f"{it['s']}~{it['r']}", it["gold"], sex.get(it["s"]), k2))
            inv_eps.append(InverseEpisode(f"{it['r']}~{it['s']}", k2, sex.get(it["r"]), it["gold"]))
    back_of = {"father": "child", "mother": "child", "child": None, "sibling": "sibling", "spouse": "spouse"}
    test_people = {it["s"] for it in test} | {it["r"] for it in test}
    for x, es in kin["edges"].items():
        if x in test_people:
            continue                                          # no episode from a test pair's own people
        for relation, y in es:
            want = back_of[relation] or ("father" if sex.get(x) == "male" else "mother" if sex.get(x) == "female" else None)
            back_rel = {r2 for r2, yy in kin["edges"].get(y, []) if yy == x}
            if not want or want not in back_rel or y in test_people:
                continue                                      # stated one way only, or on a test person
            b_name = GENDERED.get((relation, sex.get(y)), relation)
            a_name = GENDERED.get((want, sex.get(x)), want)
            inv_eps.append(InverseEpisode(f"{x}>{y}", b_name, sex.get(x), a_name))
    inv = mine_inverses(inv_eps, lib, "wikidata-train-inverse")
    log(f"inverse rules: {len(inv['adopted'])} adopted from {len(inv_eps)} two-way episodes; rejected "
        f"{dict(collections.Counter(v['why'] for v in inv['rejected'].values()))}")
    reversed_items = []
    for it in test:
        k2 = stated(it["r"], it["s"])
        if k2:
            fwd, fo = ix.paths(it["r"], it["s"], max_hops=a.max_hops, limit=16, max_nodes=20000)
            reversed_items.append({"s": it["r"], "r": it["s"], "gold": k2, "paths": fwd, "overflow": fo,
                                   "back": it["paths"], "back_overflow": it["overflow"]})
    log(f"reversed questions (the test pairs Wikidata states both ways): {len(reversed_items)}")

    rng = random.Random(a.seed)
    shuffled = Library(entails=entails)
    rules = [e["rule"] for e in lib.rules.values()]
    concl = [r.conclusion for r in rules]
    rng.shuffle(concl)
    for r, c in zip(rules, concl):
        shuffled.adopt(Rule(r.first, r.second, c), support=0, evidence=[], night="shuffled")
    clutrr = Library(CLUTRR_LIB, entails=entails) if CLUTRR_LIB.exists() else None
    log(f"shuffled: {sum(1 for r, c in zip(rules, concl) if r.conclusion != c)}/{len(rules)} moved; clutrr library: "
        f"{len(clutrr) if clutrr else 'absent'} rules\n")

    # ── day 2 ───────────────────────────────────────────────────────────────────────────────────────
    session = cc.CubelangSession(exe=a.exe)
    calls = collections.Counter()

    def run_fn(source, fn):
        calls["vm"] += 1
        return session.run(source, fn=fn)

    def score(said: str | None, gold: str) -> str:
        if said is None:
            return "refused"
        if said == gold:
            return "correct"
        if said in grain.sup(gold):
            return "coarser"
        if gold in grain.sup(said):
            return "finer"
        return "WRONG"

    def nulled(p):
        from cubbyllm.reasoning.planner import Triple
        return p[:-1] + [Triple(obj=p[-1].obj, rel=NULL_TOKEN, subj=p[-1].subj)]

    arms = {"rules": (lib, test, False), "null": (lib, test, True),
            "shuffled": (shuffled, rng.sample(test, min(a.shuffled_n, len(test))), False)}
    if clutrr is not None:
        arms["clutrr_transfer"] = (clutrr, test, False)
    arms["reversed"] = (lib, reversed_items, False)
    arms["inverse"] = (lib, reversed_items, "inverse")
    results = {}
    from cubbyllm.reasoning.planner import Triple as _T
    for arm, (library, pool, null) in arms.items():
        ta, n0 = time.perf_counter(), calls["vm"]
        rs = []
        for it in pool:
            if null == "inverse":                              # only the paths that lead the other way
                who = it["r"]                                  # the person asked about: s of the original pair
                sx = (sex[who], _T(obj=sex[who], rel="sex or gender", subj=who)) if sex.get(who) else None
                res = (relate_back(it["back"], sx, library, run_fn, tau_vm=TAU, chunk=2)
                       if not it["back_overflow"] else {"answer": None, "verified": False,
                                                        "reason": "relation_paths_overflow", "paths": []})
                said = rel(res["answer"]) if res["verified"] else None
                outcome = score(said, it["gold"])
                hops = min((len(p) for p in it["back"]), default=0)
                rs.append({"s": it["s"], "r": it["r"], "gold": it["gold"], "answer": said, "outcome": outcome,
                           "reason": res["reason"], "hops": hops,
                           "paths": [{k: p.get(k) for k in ("relations", "status", "conclusion", "steps", "failed")}
                                     for p in res["paths"][:4]]})
                continue
            if null and any(len(p) == 1 for p in it["paths"]):
                # a stated fact between the two is its own answer: relabelling it would invent a fact the
                # store does not hold (the first run did, and counted the invented fact as a breach)
                rs.append({"s": it["s"], "r": it["r"], "gold": it["gold"], "answer": None, "outcome": "direct",
                           "reason": "a stated fact, not a composition", "hops": 1, "paths": []})
                continue
            ps = [nulled(p) for p in it["paths"]] if null else it["paths"]
            res = relate(ps, library, run_fn, tau_vm=TAU, chunk=2, overflow=it["overflow"])
            said = rel(res["answer"]) if res["verified"] else None
            outcome = score(said, it["gold"])
            if null and said is not None:
                outcome = "BREACH"
            hops = min((len(p) for p in it["paths"]), default=0)
            rs.append({"s": it["s"], "r": it["r"], "gold": it["gold"], "answer": said, "outcome": outcome,
                       "reason": res["reason"], "hops": hops,
                       "paths": [{k: p.get(k) for k in ("relations", "status", "conclusion", "steps", "failed")}
                                 for p in res["paths"][:4]]})
        results[arm] = rs
        log(f"[{arm}] {len(rs)} pairs, {calls['vm'] - n0} VM calls, {time.perf_counter() - ta:.1f}s")

    for arm, rs in results.items():
        by = collections.defaultdict(collections.Counter)
        for r in rs:
            by[r["hops"]][r["outcome"]] += 1
        log(f"\n{arm} -- by the shortest path's length (0 = no path)")
        log("   hops      n  correct  coarser    finer  refused   WRONG  BREACH  (direct: n/a in null)")
        tot = collections.Counter()
        for h in sorted(by):
            c = by[h]
            tot.update(c)
            log(f"   {h:>4} {sum(c.values()):>6} {c['correct']:>8} {c['coarser']:>8} {c['finer']:>8} {c['refused']:>8} "
                f"{c['WRONG']:>7} {c['BREACH']:>7}")
        log(f"   {'all':>4} {sum(tot.values()):>6} {tot['correct']:>8} {tot['coarser']:>8} {tot['finer']:>8} "
            f"{tot['refused']:>8} {tot['WRONG']:>7} {tot['BREACH']:>7}")
        log(f"   refusals: {dict(collections.Counter(r['reason'] for r in rs if r['outcome'] == 'refused').most_common())}")
        for r in [r for r in rs if r["outcome"] == "WRONG"][:4]:
            log(f"   WRONG {r['s']} -> {r['r']}: said {r['answer']}, Wikidata {r['gold']}; {r['paths'][:1]}")

    wrong = sum(1 for arm in ("rules", "reversed", "inverse") for r in results[arm] if r["outcome"] == "WRONG")
    breach = sum(1 for r in results["null"] if r["outcome"] == "BREACH")
    seen = sum(1 for r in results["shuffled"] if r["outcome"] == "WRONG")
    verdict = "KILLED" if wrong or breach else ("INSTRUMENT BLIND" if not seen else "survives")
    log(f"\nVERDICT: {verdict} -- {wrong} wrong in rules + reversed + inverse, {breach} breaches in null, "
        f"{seen} wrong in shuffled")
    log(f"library: {len(lib)} rules; {calls['vm']} VM calls; {time.perf_counter() - t0:.1f}s")
    LOGS.mkdir(parents=True, exist_ok=True)
    stem = "exp_r35_wikidata_kinship" + (f"_{a.tag}" if a.tag else "")
    (LOGS / f"{stem}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / f"{stem}.json").write_text(json.dumps({"night": sk, "rules": lib.table(), "results": results,
                                                   "verdict": verdict}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
