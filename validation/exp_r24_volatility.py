"""
Wired: STANDALONE (validation script; never imported by cubbyllm/).

WO-0.5 -- the per-predicate volatility probe.

The problem this exists to solve is the one that makes every other instrument
untrustworthy:

    The gold labels, the verifier and the answer all descend from the SAME
    snapshot, and 0 of 552,297 stored facts carry a time qualifier. A fact
    that was true when the snapshot was taken and is false today verifies
    against the snapshot, passes tau_vm, and is spoken. The kill line reads
    green BY CONSTRUCTION, because the only oracle it has is the stale thing.

Every existing measurement is downstream of the snapshot. This is the only one
that is not: it samples facts the system would speak, re-fetches them LIVE, and
records whether the live world still agrees.

The output is a **per-predicate volatility table**, accumulated across runs.
That table is the missing time qualifier, learned empirically for free. Two
uses:

  1. A high-volatility predicate becomes **fetch-required**: deriving an answer
     for it from the snapshot alone becomes a refusal, not an answer.
  2. The aggregate disagreement rate is the only available measure of
     **snapshot decay** -- how stale the world has become since it was taken.

Budget is deliberately small (50 fetches/day by default). The table is
cumulative, so 30 days of 50 is a real distribution.

    python validation/exp_r24_volatility.py --n 50

Notes on honesty of the measurement, because they decide whether the number
means anything:

  * The re-fetch runs with the HTTP cache DISABLED. A cached re-fetch would
    compare the snapshot against itself and report 0% volatility forever,
    which is the exact failure mode this probe exists to avoid.
  * A predicate the live source does not state at all is recorded as `absent`,
    NOT as disagreement. The source dropping a claim is a different event from
    the source changing it, and folding them together would inflate volatility.
  * Sampling is stratified by predicate by default, so the table gains
    coverage across predicates rather than re-measuring `instance of` 50 times.
    Pass --uniform for a sample that reflects what actually gets spoken.

*Retire to weekly* if after 30 days every predicate's disagreement is under 1%.
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
TABLE = LOGS / "volatility_table.json"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_table() -> dict:
    if TABLE.exists():
        try:
            return json.loads(TABLE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"predicates": {}, "runs": []}


def save_table(table: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")


def same_value(stored: str, live: str) -> bool:
    """Is the live value the SAME claim as the stored one, allowing for the
    source relabelling an entity?

    The first live run of this probe reported `perpetrator` as 100% volatile
    on the strength of stored 'Eric Harris' vs live 'Eric David Harris'. That
    is one person under two labels, not a fact that changed. Counting a
    relabelling as volatility would make this table measure Wikidata's naming
    churn instead of the world's, and the whole point of the table is to
    decide which predicates must be re-fetched before being spoken -- a
    decision that would then be driven by noise.

    Token containment in either direction is the test: every word of one
    appears in the other. It accepts 'Eric Harris' vs 'Eric David Harris' and
    'Paris' vs 'Paris, France', and rejects 'Eric Harris' vs 'Dylan Klebold'.
    """
    from cubbyllm.reasoning.planner import normalize

    a, b = normalize(stored), normalize(live)
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


def sample_encyclopedia(path: pathlib.Path, n: int, seed: int, stratified: bool) -> list:
    """Sample from the encyclopedia's frame-read facts.

    This is the DEFAULT source, and the reason is a finding rather than a
    preference. The WikiKG world cannot be re-verified against a live source
    at all:

      * its triples carry `subject`/`relation`/`object` strings and **no
        QIDs**, so a subject can only be resolved by label -- and its labels
        are decased/despaced ('HD189733b', 'Soft Bank Group Corp',
        'Charles IV Of France'). A 30-fact live run resolved 13 of 30.
      * its relation vocabulary is its own (`TIMELINE_EVENT`, `HAS_PART`,
        `IS_A`, ... rendered as 'part'/'whole'/'instance'), not Wikidata
        property labels, so even a subject that DOES resolve never states the
        relation being checked. That same run decided 0 of 30.

    So a volatility rate computed over the KG world is not 0% -- it is
    UNMEASURED, and reporting it as 0% would be precisely the false green this
    work order exists to prevent.

    The encyclopedia facts do not have that problem: `entity` is a real name
    ('Henry Louis Aaron'), and `rel` is already a Wikidata property label
    ('date of birth', 'located in the administrative territorial entity'), so
    a re-fetch asks the live source a question it can actually answer.
    """
    rows = json.loads(path.read_text(encoding="utf-8")).get("facts") or []
    rng = random.Random(seed)

    class _T:                                     # duck-types the Triple fields used below
        __slots__ = ("obj", "rel", "subj")
        def __init__(self, obj, rel, subj):
            self.obj, self.rel, self.subj = obj, rel, subj

    by_rel: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        ent, rel, obj = r.get("entity"), r.get("rel"), r.get("obj")
        if ent and rel and obj:
            t = _T(obj=str(obj), rel=str(rel), subj=str(ent))
            by_rel[str(rel).lower()].append((r.get("sentence") or "", t))
    if not by_rel:
        return []

    if not stratified:
        flat = [row for rows_ in by_rel.values() for row in rows_]
        rng.shuffle(flat)
        return flat[:n]

    rels = sorted(by_rel)
    rng.shuffle(rels)
    picked, i = [], 0
    while len(picked) < n and rels:
        rows_ = by_rel[rels[i % len(rels)]]
        if rows_:
            picked.append(rows_[rng.randrange(len(rows_))])
        i += 1
        if i > n * 20:
            break
    return picked[:n]


def sample_facts(world, n: int, seed: int, stratified: bool, askable: set[str] | None) -> list:
    """Pick facts to re-check, as (fact_text, Triple).

    `askable`: only sample relations the LIVE source can actually be asked
    about. Without this filter the first live run came back 11-of-12 "absent",
    not because the source had dropped those claims but because the relations
    sampled (`coined by of unity`, `mutually intelligible with`,
    `generalization`) are this KG's vocabulary, not Wikidata property labels.
    An "absent" that only means "we asked in a language it does not speak"
    carries no information about staleness, and mixed into the table it would
    swamp the ones that do.
    """
    from cubbyllm.reasoning.planner import normalize

    rng = random.Random(seed)
    by_rel: dict[str, list] = collections.defaultdict(list)
    for bucket in world.index._by_subj.values():
        for fact, t in bucket:
            key = normalize(t.rel)
            if askable is not None and key not in askable:
                continue
            by_rel[key].append((fact, t))
    if not by_rel:
        return []

    if not stratified:
        flat = [row for rows in by_rel.values() for row in rows]
        rng.shuffle(flat)
        return flat[:n]

    # Round-robin over predicates so the table gains coverage instead of
    # re-measuring the commonest relation over and over.
    rels = sorted(by_rel)
    rng.shuffle(rels)
    picked, i = [], 0
    while len(picked) < n and rels:
        rel = rels[i % len(rels)]
        rows = by_rel[rel]
        if rows:
            picked.append(rows[rng.randrange(len(rows))])
        i += 1
        if i > n * 20:                      # safety: never spin
            break
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="facts to re-fetch (the daily budget)")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: derived from the date, so a daily run samples "
                         "differently each day without needing state")
    ap.add_argument("--uniform", action="store_true",
                    help="sample uniformly instead of stratifying by predicate")
    ap.add_argument("--any-relation", action="store_true",
                    help="do not restrict sampling to relations the live source "
                         "can be asked about. Off by default: without the filter "
                         "the first live run came back 11-of-12 'absent' purely "
                         "because the KG's relation names are not Wikidata "
                         "property labels.")
    ap.add_argument("--source", choices=("encyclopedia", "wikikg"), default="encyclopedia",
                    help="which stored facts to re-check. Default 'encyclopedia': "
                         "its facts carry real entity names and real Wikidata "
                         "property labels, so a live re-fetch can actually answer. "
                         "'wikikg' is currently UNVERIFIABLE -- see "
                         "sample_encyclopedia's docstring for the measurement.")
    ap.add_argument("--encyclopedia",
                    default=str(LOGS / "exp_r16_encyclopedia_all.json"))
    ap.add_argument("--sleep", type=float, default=0.25, help="seconds between API calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="sample and report what WOULD be fetched; no network")
    a = ap.parse_args()

    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    day = time.strftime("%Y-%m-%d")
    seed = a.seed if a.seed is not None else int(time.strftime("%Y%m%d"))

    from cubbyllm.reasoning.planner import normalize

    if a.source == "encyclopedia":
        enc = pathlib.Path(a.encyclopedia)
        if not enc.is_file():
            log(f"no encyclopedia log at {enc}. Pass --encyclopedia <path>, or "
                f"--source wikikg (which cannot currently be verified -- see "
                f"sample_encyclopedia's docstring).")
            return
        picked = sample_encyclopedia(enc, a.n, seed, stratified=not a.uniform)
        log(f"source: encyclopedia ({enc.name})")
        log(f"sampled {len(picked)} facts "
            f"({'uniform' if a.uniform else 'stratified by predicate'}, seed {seed})\n")
        _run_probe(a, picked, log, t0, day, seed, normalize, lines)
        return

    import wikikg as wk

    wk.ensure_data(("triplets",))
    tw = time.perf_counter()
    world = wk.wiki_world()
    log(f"world: {len(world)} facts | built in {time.perf_counter() - tw:.0f}s")

    askable = None
    if not a.any_relation:
        from sources import property_aliases
        pa = property_aliases()
        # `_texts_of` maps a normalized English label -> every English wording
        # of that property (label + aliases), in their ORIGINAL spelling. Take
        # the raw wordings and re-normalize them with the planner's normalizer,
        # which is the one `sample_facts` keys relations by -- normalizing with
        # the wrong one would silently produce an empty intersection.
        raw_wordings: set[str] = set()
        for texts in getattr(pa, "_texts_of", {}).values():
            raw_wordings.update(texts)
        askable = {normalize(w) for w in raw_wordings} or None
        log(f"askable relations (Wikidata property labels/aliases): "
            f"{len(askable) if askable else 0}")
        if not askable:
            log("  WARNING: could not read the property table; sampling every relation "
                "instead. Expect 'absent' verdicts that mean 'not a Wikidata property', "
                "not 'claim dropped'.")

    picked = sample_facts(world, a.n, seed, stratified=not a.uniform, askable=askable)
    log(f"source: wikikg world")
    log(f"sampled {len(picked)} facts "
        f"({'uniform' if a.uniform else 'stratified by predicate'}, seed {seed})\n")
    _run_probe(a, picked, log, t0, day, seed, normalize, lines)


def _run_probe(a, picked, log, t0, day, seed, normalize, lines) -> None:
    """Re-fetch each sampled fact live and fold the result into the table."""
    if a.dry_run:
        for fact, t in picked[:20]:
            log(f"  [{t.rel}] {t.subj} -> {t.obj}")
        log(f"\n(dry run: {len(picked)} facts would be re-fetched live)")
        return

    from sources import WikidataSource
    # cache_dir=None is the whole point: a cached re-fetch compares the
    # snapshot against itself and reports 0% volatility forever.
    src = WikidataSource(cache_dir=None, sleep_s=a.sleep)

    table = load_table()
    preds = table["predicates"]
    run = collections.Counter()
    disagreements: list[dict] = []
    unresolved: list[str] = []

    for i, (fact, t) in enumerate(picked):
        rel_key = normalize(t.rel)
        try:
            # Pass the relation: `facts` uses it to choose among items that
            # share a name, and without it the source's strict rule rejects
            # ambiguous labels outright. Omitting it cost 60% unresolved on the
            # first encyclopedia run ('San Diego', 'Lecce' -- names that
            # obviously exist).
            live = src.facts(t.subj, relations=[t.rel])
        except Exception as e:                                   # noqa: BLE001
            run["fetch_error"] += 1
            log(f"  fetch error for {t.subj!r}: {e}")
            continue

        # `WikidataSource.facts` returns `Triple`s, already structured -- so
        # the comparison is on (relation, object) rather than on a sentence.
        # Wording differences are not volatility, and this avoids re-parsing a
        # sentence that was built from a triple in the first place.
        live_objs = [lt.obj for lt in live if normalize(lt.rel) == rel_key]

        entry = preds.setdefault(rel_key, {"checked": 0, "agreed": 0, "changed": 0,
                                           "absent": 0, "relabelled": 0})
        entry.setdefault("relabelled", 0)
        entry["checked"] += 1
        run["checked"] += 1

        if not live_objs:
            # Split the two very different reasons a re-fetch comes back empty.
            # Lumping them together is what made the first runs unreadable: an
            # unresolved ENTITY says nothing about the fact, while a resolved
            # entity that no longer states the RELATION is real evidence.
            if not live:
                run["entity_unresolved"] += 1
                unresolved.append(t.subj)
                verdict = "entity_unresolved"
                entry["checked"] -= 1            # carries no information; don't bank it
                run["checked"] -= 1
                if (i + 1) % 10 == 0:
                    log(f"  {i + 1}/{len(picked)} ({time.perf_counter() - t0:.0f}s, "
                        f"{src.calls} API calls) last: {verdict}")
                continue
            entry["absent"] += 1; run["absent"] += 1
            verdict = "absent"
        elif any(normalize(o) == normalize(t.obj) for o in live_objs):
            entry["agreed"] += 1; run["agreed"] += 1
            verdict = "agreed"
        elif any(same_value(t.obj, o) for o in live_objs):
            # Same claim, different label ('Eric Harris' / 'Eric David Harris').
            # Counted as agreement for volatility, tracked separately so the
            # source's naming churn stays visible instead of being hidden.
            entry["agreed"] += 1; entry["relabelled"] += 1
            run["agreed"] += 1; run["relabelled"] += 1
            verdict = "relabelled"
        else:
            # DISPUTED, not "changed". The probe cannot tell a fact that moved
            # from a subject label that resolved to the WRONG item, and the
            # difference decides whether a predicate becomes fetch-required.
            # The first encyclopedia run made the case concrete: `date of birth`
            # of 'Robert Edward Lee' came back stored 1807-01-19 -> live 1942,
            # which is not Wikidata correcting the Confederate general's
            # birthday -- it is a different Robert Edward Lee. Scoring that as
            # volatility would have made `date of birth` fetch-required on the
            # strength of one misresolution.
            #
            # So disputes are collected for confirmation and kept OUT of the
            # rate and out of the fetch-required computation until something
            # can rule out misresolution. A predicate is promoted on confirmed
            # changes only.
            entry.setdefault("disputed", 0)
            entry["disputed"] += 1; run["disputed"] += 1
            verdict = "disputed"
            disagreements.append({"relation": t.rel, "subject": t.subj,
                                  "stored": t.obj, "live": live_objs[:4]})

        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(picked)} ({time.perf_counter() - t0:.0f}s, "
                f"{src.calls} API calls) last: {verdict}")

    # ---- report ---------------------------------------------------------------------
    checked = run["checked"]
    decided = run["agreed"] + run["changed"]
    disagree_rate = run["changed"] / decided if decided else float("nan")

    log(f"\nVOLATILITY -- this run ({day})")
    log(f"  re-fetched   : {checked}")
    log(f"  agreed       : {run['agreed']}   (of which {run['relabelled']} were the same "
        f"claim under a different label)")
    log(f"  CHANGED      : {run['changed']}   (confirmed)")
    log(f"  disputed     : {run['disputed']}   (live value differs, but misresolution is not "
        f"ruled out; excluded from the rate -- see below)")
    log(f"  absent live  : {run['absent']}   (entity resolved, relation no longer stated; "
        f"NOT counted as change)")
    log(f"  unresolved   : {run['entity_unresolved']}   (the subject label did not resolve to a "
        f"live item at all; carries no information, excluded)")
    log(f"  fetch errors : {run['fetch_error']}")
    log(f"  API calls    : {src.calls}")
    if decided:
        log(f"\n  snapshot disagreement rate = {disagree_rate:.1%} of decided re-fetches")
    else:
        log("\n  snapshot disagreement rate: UNDEFINED -- nothing was CONFIRMED this run.")
        log("  That is the honest reading, not 0% volatility: with every difference")
        log("  still disputed, this run measured no staleness either way.")

    if unresolved:
        log(f"\n  subjects that did not resolve ({len(unresolved)}), first few:")
        for s in unresolved[:10]:
            log(f"    {s!r}")
        if run["entity_unresolved"] > max(1, len(picked)) // 2:
            log("\n  MOST subjects failed to resolve, so this run sampled far less than it")
            log("  fetched. Treat the rate as thin rather than as a finding.")

    if disagreements:
        log("\n  DISPUTED -- live value differs from the stored one. Each needs one look to")
        log("  say whether the fact moved or the name resolved to the wrong item; only")
        log("  confirmed moves should promote a predicate to fetch-required:")
        for d in disagreements[:15]:
            log(f"    [{d['relation']}] {d['subject']}: stored {d['stored']!r} "
                f"-> live {d['live']!r}")

    # cumulative table
    table["runs"].append({"day": day, "seed": seed, "checked": checked,
                          "agreed": run["agreed"], "changed": run["changed"],
                          "disputed": run["disputed"], "absent": run["absent"],
                          "unresolved": run["entity_unresolved"],
                          "source": a.source, "api_calls": src.calls})
    table.setdefault("disputed", []).extend(
        dict(d, day=day) for d in disagreements)
    save_table(table)

    ranked = sorted(
        ((k, v) for k, v in preds.items() if v["agreed"] + v["changed"] >= 1),
        key=lambda kv: -(kv[1]["changed"] / max(1, kv[1]["agreed"] + kv[1]["changed"])))
    log(f"\nPER-PREDICATE TABLE (cumulative over {len(table['runs'])} run(s), "
        f"{len(preds)} predicates seen)")
    log(f"  {'predicate':<40}{'checked':>8}{'changed':>9}{'disputed':>10}{'rate':>7}")
    for k, v in ranked[:25]:
        d = v["agreed"] + v["changed"]
        log(f"  {k[:39]:<40}{v['checked']:>8}{v['changed']:>9}"
            f"{v.get('disputed', 0):>10}{v['changed'] / max(1, d):>7.0%}")

    hot = [k for k, v in ranked if (v["agreed"] + v["changed"]) >= 5
           and v["changed"] / max(1, v["agreed"] + v["changed"]) > 0.05]
    if hot:
        log(f"\n  FETCH-REQUIRED candidates (>5% disagreement over >=5 checks): {len(hot)}")
        for k in hot[:20]:
            log(f"    {k}")
        log("  Deriving one of these from the snapshot alone should become a refusal.")
    else:
        log("\n  No predicate yet has enough checks to qualify as fetch-required.")
        log("  The table needs volume: this is a 50/day probe by design, and a rate")
        log("  over fewer than 5 checks is noise, not a measurement.")

    # Relative, not absolute: this line lands in a committed .log, and an
    # absolute path bakes one machine's layout into the repo.
    log(f"\nwrote {TABLE.relative_to(ROOT).as_posix()}")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"exp_r24_volatility_{day}.log").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
