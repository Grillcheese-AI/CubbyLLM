"""exp_r32 -- branching moved INTO the walk: does it keep exp_r31's zero, and what does the open world cost?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE QUESTION
------------
exp_r31 branched OUTSIDE the walk -- one fresh `answer()` per candidate, the rivals banned -- and found
that 32 of 53 blind `ambiguous_hop` refusals on CLUTRR's fork records were apparent ambiguities: 32
correct, 0 wrong. On 2026-09-24 branching moved into `pipeline.answer` (`_explore` / `_converged`),
with one rule r31 did not have: in an OPEN world a branch the store cannot finish is a gap, not a
refutation, so it blocks the others from speaking. A CLUTRR story is a CLOSED world (every edge is
in it), which is what `closed_world=True` declares.

THE ARMS (the same 65 records r31 used)
---------------------------------------
    refuse   branch=0                       the refuse-on-sight walk (r31's single-path arm)
    closed   branch=16, closed_world=True   must reproduce r31: 32 recovered, 0 wrong
    open     branch=16 (the serve default)  speaks only when every branch finished and agreed
    ask      each open-arm refusal whose split is MID-chain is offered to the asker (the real
             `standin/ask.branch_clarify`); the asker picks the gold path's object at the split --
             they know whom they meant -- and the question is re-asked with that `choose`, up to
             `hops` rounds. A split at the last hop is never offered: a pick there would BE the answer.

KILL CRITERION: any WRONG spoken answer in any arm. The closed arm must also reproduce r31's count,
or the move into the walk changed what branching does.

    python validation/exp_r32_branching_inpipe.py --bench-root <dir>
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

from exp_r31_branching import CHUNK, tau_for  # noqa: E402  -- the same shape and thresholds as r31


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--max-branches", type=int, default=16)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import clutrr as C
    from ask import branch_clarify, parse_choose
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.pipeline import answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import normalize, parse_question
    from exp_r11_search_learn import match
    from worlds import FactStore

    items = C.fork_items("test", root=a.bench_root)
    log(f"CLUTRR fork records (unique gold path, an ambiguous (relation, subject)): {len(items)}")
    session = cc.CubelangSession(exe=a.exe)
    calls = collections.Counter()

    def vm(arm):
        def run_fn(source, fn):
            calls[arm] += 1
            return session.run(source, fn=fn)
        return run_fn

    def no_search(q, k):
        return []

    c = collections.Counter(); rows = []; wrongs = []
    wall = collections.Counter()
    ARMS = (("refuse", dict(branch=0)), ("closed", dict(branch=a.max_branches, closed_world=True)),
            ("open", dict(branch=a.max_branches)))
    for it in items:
        store = FactStore(name=f"clutrr-fork:{it['id']}")
        for f in it["facts"]:
            store.add(f)
        known = StoreRelations(store.index._seen)
        plan = parse_question(it["question"])
        if plan is None:
            c["no_plan"] += 1; continue
        tau = tau_for(it["hops"])
        row = {"id": it["id"], "hops": it["hops"], "gold": it["gold"]}

        def ask(arm, **kw):
            s = time.perf_counter()
            r = answer(it["question"], no_search, vm(arm), tau_vm=tau, tau_ret=0.0, top_k=3, max_repairs=1,
                       lookup=store.lookup, known=known, plan=plan, chunk=CHUNK, **kw)
            wall[arm] += time.perf_counter() - s
            return r

        def score(arm, r):
            if r.verified:
                m = match(r.answer, it["gold"], normalize)
                c[f"{arm}:spoke"] += 1; c[f"{arm}:{m}"] += 1
                if m == "WRONG":
                    wrongs.append({"arm": arm, "q": it["question"], "gold": it["gold"], "got": r.answer,
                                   "branches": r.branches})
                return f"spoke ({m})"
            c[f"{arm}:refused"] += 1; c[f"{arm}:why:{r.reason}"] += 1
            if r.reason == "ambiguous_hop" and r.branches:
                # what blocked it, counted over the branches the arm's world lets block: a closed world
                # has already refuted its pruned branches, so they are never the reason there
                live = [b for b in r.branches if not (arm == "closed" and b["status"] == "pruned")]
                st = collections.Counter(b["status"] for b in live)
                ans = (r.refused or {}).get("verified_answers") or []
                kind = ("overflow" if (r.refused or {}).get("overflow") else
                        "divergent" if len(ans) > 1 else "all_pruned" if not live else
                        "blocked:vm" if st.get("vm_failed") else "blocked:pruned" if st.get("pruned") else
                        "blocked:type" if st.get("type_mismatch") else "other")
                c[f"{arm}:ambiguous:{kind}"] += 1
                return f"refused ({kind})"
            return f"refused ({r.reason})"

        results = {}
        for arm, kw in ARMS:
            results[arm] = ask(arm, **kw)
            row[arm] = score(arm, results[arm])

        # the ask arm: the open world's refusals, offered to an asker who knows whom they meant
        r = results["open"]
        if not r.verified:
            prior: dict[int, str] = {}
            rounds = 0
            clar = branch_clarify(plan, r, prior)
            if clar is None:
                c["ask:not_offered"] += 1; row["ask"] = "not offered"
            while clar is not None and rounds < it["hops"]:
                h = clar["hop"]
                want = normalize(it["chain"][h][2])            # the object the gold path takes at the split
                pick = next((ch for ch in clar["choices"] if normalize(ch["label"]) == want), None)
                if pick is None:
                    c["ask:gold_not_offered"] += 1; row["ask"] = "gold not among the choices"; break
                rounds += 1
                prior = parse_choose(pick["choose"])
                r = ask("ask", branch=a.max_branches, choose=prior)
                clar = branch_clarify(plan, r, prior) if not r.verified else None
            if rounds:
                c["ask:offered"] += 1; c[f"ask:rounds:{rounds}"] += 1
                row["ask"] = f"{score('ask', r)} after {rounds} pick{'s' if rounds > 1 else ''}"
        rows.append(row)
    session.close()

    n = len(rows)
    log(f"\n{'arm':<10}{'spoke':>7}{'correct':>9}{'WRONG':>7}{'refused':>9}   VM calls   wall")
    for arm, _kw in ARMS + (("ask", None),):
        log(f"{arm:<10}{c[arm + ':spoke']:>7}{c[arm + ':correct']:>9}{c[arm + ':WRONG']:>7}{c[arm + ':refused']:>9}"
            f"   {calls[arm]:>8}   {wall[arm]:.1f}s")
    for arm in ("closed", "open"):
        ks = sorted(k for k in c if k.startswith(f"{arm}:ambiguous:"))
        if ks:
            log(f"  {arm} refusals: " + ", ".join(f"{k.split(':', 2)[2]} {c[k]}" for k in ks))
    log(f"  ask: offered {c['ask:offered']} (rounds " +
        ", ".join(f"{k.split(':')[2]}: {c[k]}" for k in sorted(c) if k.startswith("ask:rounds:")) +
        f"), not offered {c['ask:not_offered']} (a last-hop split), gold not among the choices {c['ask:gold_not_offered']}")

    log(f"\n{'id':<14}{'hops':>5}   {'refuse':<26}{'closed':<26}{'open':<27}ask")
    log("-" * 120)
    for r in rows[:16]:
        log(f"{str(r['id'])[:12]:<14}{r['hops']:>5}   {r['refuse']:<26}{r['closed']:<26}{r['open']:<27}{r.get('ask', '')}")
    if len(rows) > 16:
        log(f"... {len(rows) - 16} more")
    for w in wrongs[:5]:
        log(f"\nWRONG ({w['arm']}): {w['q'][:100]}\n   gold {w['gold']}, got {w['got']}")

    wrong = sum(c[f"{arm}:WRONG"] for arm in ("refuse", "closed", "open", "ask"))
    r31 = json.loads((LOGS / "exp_r31_branching.json").read_text(encoding="utf-8"))["counts"]
    r31_spoke = r31.get("single:verified", 0) + r31.get("branch:recovered", 0)
    if wrong:
        verdict = f"KILLED (correctness): {wrong} wrong spoken answers"
    elif c["closed:spoke"] != r31_spoke:
        verdict = (f"KILLED (reproduction): the closed arm spoke {c['closed:spoke']}, r31 spoke {r31_spoke} "
                   f"-- moving branching into the walk changed what it does")
    else:
        verdict = (f"survives; 0 wrong in every arm. closed reproduces r31 ({c['closed:spoke']}/{n} spoken); "
                   f"open speaks {c['open:spoke']}/{n}, and one pick by the asker brings it to "
                   f"{c['open:spoke'] + c['ask:spoke']}/{n}")
    log(f"\nVERDICT: {verdict}")

    stem = f"exp_r32_branching_inpipe{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "derived": True,
        "note": "CLUTRR-DERIVED chain questions over the fork records; not a CLUTRR score.",
        "n_items": n, "chunk": CHUNK, "max_branches": a.max_branches, "counts": dict(c),
        "vm_calls": dict(calls), "wall_s_by_arm": {k: round(v, 2) for k, v in wall.items()},
        "rows": rows, "wrongs": wrongs, "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1, default=str), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
