"""exp_r33 -- the web behind Wikidata, held out: does a fact the store lacks come back from the web,
attested by two independent sites, and is anything it speaks ever wrong?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

THE BATTERY
-----------
N facts from the wiki world's triples (Wikidata-derived) for relations the web states in prose:
born on / born in / died on / died in -> date of birth / place of birth / date of death / place of
death. Each becomes a one-hop question, "What is the date of birth of Girard Desargues?". The store
is EMPTY -- the fact is withheld -- and the primary source answers nothing (Wikidata's stand-in), so
the web is the only way to the answer; gold is the withheld object. The Wikipedia -> Wikidata handoff
is OFF by default: it would hand back the very claims the gold came from, and what is measured here is
the web's own readers. The loop is the served one: `learn_and_answer(fallbacks=[WebSource(...)])`, the
real cubelang VM, tau 1.0 for one hop.

WHAT IS SCORED
--------------
  spoken   verified answers -- correct / near / WRONG (dates: an answer more precise than a
           month- or year-precision gold that agrees with it is correct; a less precise one is near)
  held     `latent_only` -- the web found it on one site only; the would-be answer is scored too, so
           the log shows what the two-site rule withheld (held-correct: recall it cost; held-WRONG:
           errors it stopped)
  split    two attested values for the relation (Warsaw / Poland): the walk refuses, nothing spoken
  none     the web found nothing for the relation

KILL (H-A12): any WRONG spoken.

    python validation/exp_r33_web_heldout.py --n 60 --reader auto --tag _lfm
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RELS = {"BORN_ON": "date of birth", "BORN_IN": "place of birth", "DIED_ON": "date of death", "DIED_IN": "place of death"}


class Withheld:
    """Wikidata's stand-in: it has nothing, so the web is asked."""
    name = "wikidata (withheld)"

    def facts(self, entity, **_kw):
        return []


CONTAINED_BY = ("P131", "P17")      # located in the administrative entity, country ("part of" wanders to the Commonwealth and Earth)


def wikidata_up(wd, place: str, normalize, memo: dict) -> set[str]:
    """The places that contain `place` per Wikidata, as normalized labels: P131 / P17 walked up
    six levels, for EVERY item whose label is exactly the name, intersected -- a 'Paris' in Texas and one in
    France contain nothing in common, so neither vouches for 'France'. A lookup, not a judge."""
    sets = items_up(wd, place, normalize, memo)
    return set.intersection(*sets) if sets else set()


def items_up(wd, place: str, normalize, memo: dict) -> list[set[str]]:
    """Per item labelled exactly `place`: the places that contain it."""
    key = normalize(place)
    if key in memo:
        return memo[key]
    got = wd._get({"action": "wbsearchentities", "search": place, "language": "en", "limit": 7}) or {}
    items = [h["id"] for h in got.get("search") or [] if normalize(h.get("label", "")) == key][:4]
    sets = []
    for qid in items:
        seen, frontier = set(), {qid}
        for _ in range(6):
            nxt = set()
            for q in frontier:
                claims = wd._claims(q)
                for p in CONTAINED_BY:
                    for st in claims.get(p, []):
                        v = (((st.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
                        if isinstance(v, dict) and v.get("id") and v["id"] not in seen:
                            nxt.add(v["id"])
            seen |= nxt
            frontier = nxt
            if not frontier or len(seen) > 80:
                break
        labels = wd._labels_for(sorted(seen)) if seen else {}
        sets.append({normalize(l) for l in labels.values() if l})
    memo[key] = sets
    return sets


def same_place(answer: str, gold: str, wd, normalize, memo: dict) -> bool:
    """'Lugo, Italy' and the gold 'Lugo Emilia Romagna' (a name whose comma the dataset dropped) are one
    place when ONE item named Lugo lies in Italy AND in Emilia-Romagna -- the answer's own qualifiers and
    the gold's both check out on the same item. 'Paris, Texas' against 'Paris France' finds no such item."""
    parts = [normalize(x) for x in answer.split(",") if normalize(x)]
    gw = normalize(gold).split()
    for i in range(1, len(gw)):
        if " ".join(gw[:i]) != (parts[0] if parts else None):
            continue
        tail = " ".join(gw[i:])
        for anc in items_up(wd, answer.split(",")[0], normalize, memo):
            if tail in anc and all(q in anc for q in parts[1:]):
                return True
    return False


def score(answer: str | None, gold: str, normalize, up: dict | None = None, wd=None, memo: dict | None = None) -> str:
    """correct | near | coarse | finer | WRONG. `coarse`: the answer is a place that CONTAINS the gold
    ('Sweden' for Bromma) -- the question asks the PLACE of birth, not the city, so a containing place
    answers it (Nick, 2026-09-24): scored as not wrong, reported apart. `finer`: the gold contains the
    answer. Containment comes from the wiki world's own LOCATED_IN graph and, when a Wikidata source is
    given, from Wikidata's (`wikidata_up`)."""
    from exp_r11_search_learn import match
    if not answer:
        return "none"
    if (up is not None or wd is not None) and not answer.strip()[:4].isdigit():
        up = up or {}
        a_parts = [normalize(x) for x in answer.split(",") if normalize(x)]
        g = normalize(gold)

        def ancestors(x):
            out, frontier = set(), {x}
            for _ in range(6):
                frontier = {p for f in frontier for p in up.get(f, ())} - out
                out |= frontier
            return out
        m0 = match(answer, gold, normalize)
        if m0 != "WRONG":
            return m0
        if a_parts and a_parts[0] in ancestors(g):
            return "coarse"
        if a_parts and g in ancestors(a_parts[0]):
            return "finer"
        if wd is not None and a_parts:
            memo = memo if memo is not None else {}
            if same_place(answer, gold, wd, normalize, memo):
                return "correct"
            if a_parts[0] in wikidata_up(wd, gold, normalize, memo):
                return "coarse"
            if g in wikidata_up(wd, a_parts[0], normalize, memo):
                return "finer"
        return "WRONG"
    a, g = answer.strip(), gold.strip()
    if g[:4].isdigit() and a[:4].isdigit():                 # a date: compare at the coarser precision
        if a == g or (a.startswith(g) and len(g) < len(a)):
            return "correct"
        if g.startswith(a) and len(a) < len(g):
            return "near"
        return "WRONG"
    return match(a, g, normalize)


def verdict_of(c, n: int) -> str:
    if c["spoke:WRONG"]:
        return f"KILLED: {c['spoke:WRONG']} wrong spoken"
    return (f"survives; {c['spoke']}/{n} spoken ({c['spoke:correct']} correct, {c['spoke:near']} near, "
            f"{c['spoke:coarse']} a containing place, {c['spoke:finer']} finer, 0 wrong), "
            f"{c['held']} held for a second site ({c['held:correct'] + c['held:coarse'] + c['held:near']} of them right), "
            f"{c['split']} split")


def rescore(path: pathlib.Path) -> None:
    """Re-score a finished run's rows with Wikidata containment (no search, no VM): the spoken and held
    answers are in the log; only their verdicts change."""
    from cubbyllm.reasoning.planner import normalize
    from sources import WikidataSource
    d = json.loads(path.read_text(encoding="utf-8"))
    wd, memo = WikidataSource(), {}
    c = collections.Counter({k: v for k, v in d["counts"].items() if not k.startswith(("spoke:", "held:"))})
    changed = []
    for r in d["rows"]:
        kind = r["outcome"].split()[0]
        if kind not in ("spoke", "held"):
            continue
        m = score(r["answer"], r["gold"], normalize, None, wd, memo)
        if m != r["outcome"].split()[1]:
            changed.append((r["q"], r["gold"], r["answer"], r["outcome"], f"{kind} {m}"))
            r["outcome"] = f"{kind} {m}"
        c[f"{kind}:{m}"] += 1
    d["counts"] = dict(c); d["verdict"] = verdict_of(c, d["n"]); d["rescored"] = [list(x) for x in changed]
    out = path.with_name(path.stem + "_rescored.json")
    out.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    for q, g, a_, was, now in changed:
        print(f"{was:<14} -> {now:<14} {q[:60]}  gold {g}  got {a_}")
    print("VERDICT:", d["verdict"]); print("wrote", out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reader", default="off", help="the local reader: 'auto' (the LFM base), a GGUF path, or 'off'")
    ap.add_argument("--handoff", action="store_true", help="let Wikipedia hits hand over to live Wikidata")
    ap.add_argument("--dissent", action="store_true", help="one site's different value vetoes the claim (default: ignored)")
    ap.add_argument("--offline", action="store_true", help="searches and pages from the cache only (a re-score of the same web)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache", default=None, help="the cache tag to read (default: --tag) -- to re-run another run's web")
    ap.add_argument("--rescore", default=None, help="a finished run's json: re-score it with Wikidata containment")
    a = ap.parse_args()
    if a.rescore:
        rescore(pathlib.Path(a.rescore)); return
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    import wikikg as wk
    from cubbyllm.bridges import cubelang_client as cc
    from cubbyllm.reasoning.learn import learn_and_answer
    from cubbyllm.reasoning.plan_verify import StoreRelations
    from cubbyllm.reasoning.planner import QuestionPlan, normalize
    from exp_r11_search_learn import LookupStore
    from web_source import BraveBackend, WebSource

    triples = wk.load_triples(wk.TRIPLETS)
    by_rel: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    seen_subj: set[tuple[str, str]] = set()
    for s, r, o in triples:
        if r in RELS:
            subj, obj = wk.decamel(s), wk.decamel(o) if not o[:4].isdigit() else o
            if len(subj.split()) >= 2 and obj and (subj, r) not in seen_subj:
                seen_subj.add((subj, r)); by_rel[r].append((subj, obj))
    up: dict[str, set[str]] = collections.defaultdict(set)      # place -> the places that contain it
    for s, r, o in triples:
        if r in ("LOCATED_IN", "PART_OF"):
            up[normalize(wk.decamel(s))].add(normalize(wk.decamel(o)))
        elif r in ("LOCATION_OF", "HAS_PART"):
            up[normalize(wk.decamel(o))].add(normalize(wk.decamel(s)))
    rng = random.Random(a.seed)
    per = max(1, a.n // len(RELS))
    items = [(subj, RELS[r], obj) for r in RELS for subj, obj in rng.sample(by_rel[r], min(per, len(by_rel[r])))]
    log(f"battery: {len(items)} withheld facts ({', '.join(f'{RELS[r]} {min(per, len(by_rel[r]))}' for r in RELS)}), seed {a.seed}")

    reader = None
    if a.reader != "off":
        from lfm_source import LfmReader
        gguf = str(ROOT / "standin" / "models" / "LFM2.5-2.6B.Q4_K_M.gguf") if a.reader == "auto" else a.reader
        reader = LfmReader(gguf)
    wikidata = None
    if a.handoff:
        from sources import WikidataSource
        wikidata = WikidataSource()
    from sources import WikidataSource as _WD
    backend = BraveBackend()
    web = WebSource(backend, reader=reader, wikidata=wikidata, containment=_WD(), dissent_blocks=a.dissent,
                    offline=a.offline, cache_dir=ROOT / "standin" / "data" / "out" / f"web_r33{a.cache or a.tag}")
    log(f"web: {backend.name} ({backend.license}), reader {'LFM base (CPU)' if reader else 'off'}, "
        f"handoff {'on' if wikidata else 'off'}, two independent sites to attest, containment Wikidata, "
        f"dissent {'vetoes' if a.dissent else 'ignored'}{', OFFLINE (cache)' if a.offline else ''}")
    from sources import WikidataSource
    wd, memo = WikidataSource(), {}                      # the scorer's containment lookups (never the loop's)
    session = cc.CubelangSession(exe=a.exe)

    def run_fn(source, fn):
        return session.run(source, fn=fn)

    c = collections.Counter(); rows = []; wrong = []
    for i, (subj, rel, gold) in enumerate(items):
        q = f"What is the {rel} of {subj}?"
        plan = QuestionPlan(relations=[None], tail=f"{rel} of {subj}", n_hop=1)
        store = LookupStore([]); store.provenance = {}; store.times = {}
        known = StoreRelations([f"x is the {rel} of y"])
        s0 = time.perf_counter()
        lr = learn_and_answer(q, lambda _q, _k: [], run_fn, store=store, known=known, source=Withheld(),
                              tau_vm=1.0, plan=plan, fallbacks=[web])
        wall = time.perf_counter() - s0
        r = lr.result
        facts = web.last.get("facts", {}) if lr.entities else {}
        mine = {k: f for k, f in facts.items() if k.endswith(normalize(f" is the {rel} of {subj}"))}
        attested = [k for k, f in mine.items() if not f.get("held")]
        if r.verified:
            m = score(r.answer, gold, normalize, up, wd, memo); outcome = f"spoke {m}"
            c["spoke"] += 1; c[f"spoke:{m}"] += 1
            if m == "WRONG":
                wrong.append({"q": q, "gold": gold, "got": r.answer, "sites": web.provenance.get(normalize(f"{r.answer} is the {rel} of {subj}"))})
        elif r.reason == "latent_only":
            m = score((r.refused or {}).get("answer"), gold, normalize, up, wd, memo); outcome = f"held {m}"
            c["held"] += 1; c[f"held:{m}"] += 1
        elif r.reason == "ambiguous_hop":
            outcome = "split"; c["split"] += 1
            ans = (r.refused or {}).get("verified_answers") or (r.refused or {}).get("objects") or []
            c["split:gold_among"] += int(any(score(x, gold, normalize, up, wd, memo) in ("correct", "near", "finer", "coarse") for x in ans))
        else:
            outcome = "none" if not mine else f"refused {r.reason}"; c["none" if not mine else "other"] += 1
        c[f"rel:{rel}:{outcome.split()[0]}"] += 1
        rows.append({"q": q, "gold": gold, "outcome": outcome, "answer": r.answer or (r.refused or {}).get("answer"),
                     "reason": r.reason, "values": {k: {"sites": f.get("sites"), "held": f.get("held"),
                                                        "readers": f.get("readers")} for k, f in mine.items()},
                     "attested": len(attested), "wall_s": round(wall, 2), "how": web.last.get("how")})
        log(f"{i + 1:>3} {outcome:<16} {q[:70]:<70} gold {gold[:24]:<24} got {str(rows[-1]['answer'])[:24]}")
    session.close()

    n = len(rows)
    log(f"\n{'':<12}{'n':>5}")
    for k in ("spoke", "held", "split", "none", "other"):
        log(f"{k:<12}{c[k]:>5}   " + ", ".join(f"{m} {c[f'{k}:{m}']}" for m in ("correct", "near", "finer", "coarse", "WRONG") if c[f"{k}:{m}"]))
    log(f"split with the gold among the certified values: {c['split:gold_among']}")
    log("by relation: " + "; ".join(f"{rel}: " + ", ".join(f"{o} {c[f'rel:{rel}:{o}']}" for o in ("spoke", "held", "split", "none", "refused")
                                                             if c[f"rel:{rel}:{o}"]) for rel in RELS.values()))
    log(f"searches {web.calls['search']}, pages {web.calls['page']}, reads {web.calls['read']}, "
        f"wall {time.perf_counter() - t0:.0f}s ({(time.perf_counter() - t0) / max(1, n):.1f}s a question)")
    for w in wrong:
        log(f"WRONG: {w['q']} gold {w['gold']} got {w['got']} [{w['sites']}]")
    verdict = verdict_of(c, n)
    log(f"\nVERDICT: {verdict}")
    stem = f"exp_r33_web_heldout{a.tag}"
    (LOGS / f"{stem}.json").write_text(json.dumps({"n": n, "seed": a.seed, "reader": a.reader, "handoff": a.handoff,
                                                   "counts": dict(c), "calls": web.calls, "rows": rows, "wrong": wrong,
                                                   "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1)},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
