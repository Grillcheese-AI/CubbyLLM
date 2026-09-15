"""exp_r31 -- does host-owned branching recover a refusal without buying a guess?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE QUESTION
------------
`pipeline._walk` raises `_Ambiguous` the moment a hop has more than one admissible
fact, and `answer` turns that into an immediate refusal naming the candidates. It
is a blind refusal: the right answer may well be among them, and nothing tries.

cubemind's `_archive/execution/decision_tree.py` (March 2026, 419 lines of its own
passing tests) is the structure for trying: an immutable-history tree whose nodes
hold ranked candidates, with `select()` to branch and `backtrack_to(depth)` to
come back. WO-3.1 needs exactly this, because branching is the part of the VM that
does not work -- so the search has to live in the host.

WHAT IT WOULD BUY, AND WHAT IT MUST NOT
---------------------------------------
The tempting claim is "branching converts refusals into answers". That claim would
be a kill-line breach dressed as a feature: picking one of several admissible
candidates is a guess, and a guess is exactly what this architecture exists not to
do. The claim actually worth testing is narrower, and it has two halves:

  * exactly ONE branch verifies  -> the ambiguity was apparent, not real. The
    answer is certified and safe to speak. A refusal is RECOVERED, not guessed.
  * SEVERAL branches verify      -> genuinely ambiguous. Still a refusal, but an
    ENUMERATED one that names the alternatives instead of a bare `ambiguous_hop`.

So branching buys certified answers where the ambiguity was illusory, and better
refusals where it was not. It must buy no wrong answers at all.

THE DATA
--------
CLUTRR's own test split, no synthesis. 411 of 1,146 records are forks; in 88 of
them two facts share a (relation, subject) -- "Jenny is the mother of Mary" and
"Jenny is the mother of Anne" -- so a hop asking for the daughter of Jenny has two
answers. Records whose GOLD path is itself non-unique are excluded: there the
question is ambiguous, which is a different problem.

THE ARMS
--------
    single_path   the shipped walk                      (expect: refuses)
    branching     host-owned DecisionTree over the hops (the port under test)

KILL CRITERION, two clauses
---------------------------
  * correctness -- ANY wrong spoken answer kills it. Speaking when several
    branches verify is the failure mode to watch, and the harness counts it
    separately as GUESSED so it cannot hide inside "correct".
  * rate -- if branching recovers no single-verifier cases, it buys nothing the
    bare refusal did not already give, and the structure is not worth porting.

    python validation/exp_r31_branching.py --bench-root <dir>
"""
from __future__ import annotations

import argparse, collections, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}
TAU_FALLBACK = 0.2202
CHUNK = 2                                  # exp_r28's shape; depth is not the variable here


def tau_for(n_hop: int, chunk: int = CHUNK) -> float:
    return TAU_VM.get(min(chunk, n_hop) if chunk else n_hop, TAU_FALLBACK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--max-branches", type=int, default=16,
                    help="node budget; the host owns it, which is the point")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import clutrr as C
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.pipeline import answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import normalize, parse_question
    from exp_r11_search_learn import match
    from worlds import FactStore

    items = C.fork_items("test", root=a.bench_root)
    by_hops = collections.Counter(i["hops"] for i in items)
    log(f"CLUTRR fork records with an ambiguous (relation, subject) and a UNIQUE "
        f"gold path: {len(items)}")
    log(f"  by depth: {json.dumps(dict(sorted(by_hops.items())))}")
    log(f"  ambiguous keys per record: "
        f"{collections.Counter(len(i['ambiguous_keys']) for i in items).most_common()}")
    if items:
        e = items[0]
        log(f"\nexample {e['id']}  ({e['hops']} hops, gold {e['gold']})")
        log(f"  Q: {e['question']}")
        for f in e["facts"]:
            log(f"    {f}" + ("   <- the fork"
                              if any(normalize(f).startswith(normalize(f.split(' is the ')[0]))
                                     and (k[0] in normalize(f)) for k in e["ambiguous_keys"])
                              else ""))
    log("")

    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    def no_search(q, k):
        return []

    c = collections.Counter()
    branch_rows, wrongs, guessed = [], [], []

    for it in items:
        store = FactStore(name=f"clutrr-fork:{it['id']}")
        for f in it["facts"]:
            store.add(f)
        known = StoreRelations(store.index._seen)
        plan = parse_question(it["question"])
        if plan is None:
            c["no_plan"] += 1; continue
        tau = tau_for(it["hops"])

        # ---- arm 1: the shipped single-path walk -----------------------------
        base = answer(it["question"], no_search, run_fn, tau_vm=tau, tau_ret=0.0,
                      top_k=3, max_repairs=1, lookup=store.lookup, known=known,
                      plan=plan, chunk=CHUNK)
        if base.verified:
            c["single:verified"] += 1
            m = match(base.answer, it["gold"], normalize)
            c[f"single:{m}"] += 1
            if m == "WRONG":
                wrongs.append({"arm": "single", "q": it["question"],
                               "gold": it["gold"], "got": base.answer})
            continue                       # nothing for branching to recover
        c["single:refused"] += 1
        c[f"single:why:{base.reason or 'failed'}"] += 1
        if base.reason != "ambiguous_hop":
            continue                       # a different refusal; branching is not the lever

        # ---- arm 2: host-owned branching -------------------------------------
        # The tree's candidates at the ambiguous hop are the objects the walk
        # named in its own refusal -- the host is not inventing options, it is
        # trying the ones the VM already said were admissible. Each branch is a
        # fresh walk with the OTHER candidates banned, which is the same
        # mechanism `answer` already uses for verify-stage repair.
        cands = list((base.refused or {}).get("objects") or [])
        facts_at_hop = list((base.refused or {}).get("facts") or [])
        c["ambiguous"] += 1
        c[f"ambiguous:{len(cands)}way"] += 1

        verified_branches = []
        for keep in facts_at_hop[: a.max_branches]:
            banned = {f for f in facts_at_hop if f != keep}
            # a store that holds every fact except the rival candidates: the
            # branch the tree is exploring, made concrete
            sub = FactStore(name=f"branch:{it['id']}")
            for f in it["facts"]:
                if f not in banned:
                    sub.add(f)
            sub_known = StoreRelations(sub.index._seen)
            r = answer(it["question"], no_search, run_fn, tau_vm=tau, tau_ret=0.0,
                       top_k=3, max_repairs=1, lookup=sub.lookup, known=sub_known,
                       plan=plan, chunk=CHUNK)
            if r.verified:
                verified_branches.append({"kept": keep, "answer": r.answer})

        n_ok = len(verified_branches)
        answers = {normalize(b["answer"] or "") for b in verified_branches}
        row = {"id": it["id"], "hops": it["hops"], "candidates": len(cands),
               "verified_branches": n_ok, "distinct_answers": len(answers),
               "gold": it["gold"]}

        if n_ok == 0:
            c["branch:still_refused"] += 1
            row["outcome"] = "still refused"
        elif len(answers) == 1:
            # ONE answer survives certification -> the ambiguity was apparent
            spoken = verified_branches[0]["answer"]
            m = match(spoken, it["gold"], normalize)
            c["branch:recovered"] += 1; c[f"branch:{m}"] += 1
            row["outcome"] = f"recovered ({m})"; row["spoke"] = spoken
            if m == "WRONG":
                wrongs.append({"arm": "branch", "q": it["question"],
                               "gold": it["gold"], "got": spoken,
                               "branches": verified_branches})
        else:
            # several survive -> still a refusal, but it can now NAME them
            c["branch:enumerated"] += 1
            row["outcome"] = "enumerated refusal"
            row["alternatives"] = sorted(str(b["answer"]) for b in verified_branches)
            if it["gold"] and normalize(it["gold"]) in answers:
                c["branch:enumerated:gold_present"] += 1
            # a system that PICKED one here would be guessing; counted so the
            # temptation is visible in the log rather than hidden in "correct"
            guessed.append({"q": it["question"], "gold": it["gold"],
                            "would_have_picked": verified_branches[0]["answer"],
                            "alternatives": row["alternatives"]})
        branch_rows.append(row)

    session.close()

    log(f"{'single-path walk':<34}{c['single:verified'] + c['single:refused']:>6}")
    log(f"  verified{'':<24}{c['single:verified']:>6}   "
        f"(correct {c['single:correct']}, WRONG {c['single:WRONG']})")
    log(f"  refused{'':<25}{c['single:refused']:>6}")
    for k, v in sorted(((k[len('single:why:'):], v) for k, v in c.items()
                        if k.startswith("single:why:")), key=lambda kv: -kv[1]):
        log(f"    {k:<30}{v:>6}")

    log(f"\n{'branching over the ambiguous hop':<34}{c['ambiguous']:>6}")
    for k in sorted(k for k in c if k.startswith("ambiguous:")):
        log(f"  {k[len('ambiguous:'):]:<32}{c[k]:>6}  candidates")
    log(f"  recovered (one answer certified){c['branch:recovered']:>6}   "
        f"correct {c['branch:correct']}, WRONG {c['branch:WRONG']}")
    log(f"  enumerated refusal{'':<14}{c['branch:enumerated']:>6}   "
        f"gold among the alternatives: {c['branch:enumerated:gold_present']}")
    log(f"  still refused{'':<19}{c['branch:still_refused']:>6}")

    if branch_rows:
        log(f"\n{'id':<14}{'hops':>5}{'cands':>7}{'ok':>4}{'answers':>9}   outcome")
        log("-" * 72)
        for r in branch_rows[:14]:
            log(f"{str(r['id'])[:12]:<14}{r['hops']:>5}{r['candidates']:>7}"
                f"{r['verified_branches']:>4}{r['distinct_answers']:>9}   {r['outcome']}")
        if len(branch_rows) > 14:
            log(f"... {len(branch_rows) - 14} more")

    for w in wrongs[:5]:
        log(f"\nWRONG ({w['arm']}): {w['q'][:100]}")
        log(f"   gold {w['gold']}, got {w['got']}")
    if guessed:
        log(f"\nwhat a GUESSING system would have said (it did not):")
        for g in guessed[:3]:
            log(f"   {g['q'][:90]}")
            log(f"     gold {g['gold']} | alternatives {g['alternatives']} "
                f"| would have picked {g['would_have_picked']}")

    wrong = c["single:WRONG"] + c["branch:WRONG"]
    if wrong:
        verdict = f"KILLED (correctness clause): {wrong} wrong spoken answers"
    elif c["branch:recovered"] == 0:
        verdict = ("KILLED (rate clause): branching recovered nothing -- it buys "
                   "no answer the bare refusal did not already give")
    else:
        verdict = (f"survives; {c['branch:recovered']} blind refusals became certified "
                   f"answers ({c['branch:correct']} correct, 0 wrong) and "
                   f"{c['branch:enumerated']} became enumerated refusals")
    log(f"\nVERDICT: {verdict}")

    stem = f"exp_r31_branching{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "derived": True,
        "note": "CLUTRR-DERIVED chain questions over the fork records; not a CLUTRR score.",
        "n_items": len(items), "chunk": CHUNK, "max_branches": a.max_branches,
        "by_depth": dict(sorted(by_hops.items())),
        "counts": dict(c), "rows": branch_rows,
        "wrongs": wrongs, "would_have_guessed": guessed,
        "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
