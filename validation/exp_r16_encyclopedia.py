"""exp_r16 -- the encyclopedia as an offline Source: how many entries the frames read, and are the
facts true?

Nick (2026-09-13): "we could easily get Q/A extracted from these encyclopedia records". Before any
of it feeds a dataset, the OCR's precision is measured two ways, neither a model: (1) every year
and date the frames read is compared with what the wiki world (Wikidata-derived, 552k facts)
holds for the same entity and relation -- agree / disagree / unheld -- and every disagreement is
printed in full with its sentence; (2) a seeded sample of admitted facts is printed with their
sentences for hand reading. Places are compared by containment (the frames read the immediate
place, Wikidata may hold the country).

Run (one volume): python validation/exp_r16_encyclopedia.py --folder <dir> --volumes 13
Logs: validation/logs/exp_r16_encyclopedia<tag>.{log,json}
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cubbyllm.reasoning.planner import normalize          # noqa: E402
from encyclopedia import OPENING, EncyclopediaSource, entries, read_entry   # noqa: E402

DATE_RELS = ("date of birth", "date of death")


def fold(s: str) -> str:
    """ASCII-folded, normalized: 'Mölln' and the OCR's 'Molln' are one spelling."""
    import unicodedata
    return normalize(unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())


def compare(rel: str, obj: str, held: list[str]) -> str:
    """agree | near | disagree | unheld, against the world's objects for (entity, rel). `near`
    (places): the same name in another spelling or transliteration (difflib >= 0.7 after
    folding) -- listed with the disagreements for hand reading, never counted as agreement."""
    if not held:
        return "unheld"
    if rel in DATE_RELS:
        y = obj[:4]
        years = {h[:4] for h in held if h[:1].isdigit()}
        if not years:
            return "unheld"
        return "agree" if y in years else "disagree"
    import difflib
    o = fold(obj)
    best = 0.0
    for h in held:
        hn = fold(h)
        if o == hn or (o and (o in hn or hn in o)):
            return "agree"
        best = max(best, difflib.SequenceMatcher(None, o, hn).ratio())
    return "near" if best >= 0.7 else "disagree"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--volumes", nargs="*", default=None, help="volume numbers; all when omitted")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-world", action="store_true", help="skip the wiki-world cross-check")
    ap.add_argument("--wikidata", action="store_true",
                    help="cross-check against Wikidata too (cached API, learning phase; the entity-ambiguity rule applies: "
                         "a name two items share is checked against neither)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"exp_r16_encyclopedia{a.tag}.log"
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    t0 = time.perf_counter()
    src = EncyclopediaSource(a.folder, volumes=a.volumes)
    log(f"encyclopedia: {len(src.volumes)} volume(s) from {pathlib.Path(a.folder).name}, "
        f"{sum(len(i) for _v, _p, i in src.volumes):,} named entries, index {time.perf_counter() - t0:.1f}s")
    world = None
    if not a.no_world:
        import wikikg as wk
        wk.ensure_data(("triplets",))
        t1 = time.perf_counter(); world = wk.wiki_world()
        log(f"wiki world {len(world):,} facts ({time.perf_counter() - t1:.0f}s)")
    wd = None
    if a.wikidata:
        from sources import WikidataSource
        from cubbyllm.reasoning.planner import parse_fact
        wd = WikidataSource()
    checking = world is not None or wd is not None

    c = collections.Counter(); facts_out: list[dict] = []; disagreements: list[dict] = []
    for vol, p, _idx in src.volumes:
        text = src._read(vol, p)
        for head, s, b, e in entries(text):
            name, person, pairs = read_entry(head, text[b:b + OPENING])
            c["entries"] += 1; c["person"] += person; c["named"] += name is not None
            if name is None:
                c["person_unnamed"] += 1
                continue
            held_by_rel: dict[str, list[str]] = collections.defaultdict(list)
            if world is not None and pairs:
                for _f, t in world.index._by_subj.get(normalize(name), []):
                    held_by_rel[normalize(t.rel)].append(t.obj)
            if wd is not None and pairs:
                wf = wd.facts(name)
                if wf:
                    c["wd:resolved"] += 1
                elif wd.last.get("ambiguous"):
                    c["wd:ambiguous"] += 1
                else:
                    c["wd:unresolved"] += 1
                for f in wf:
                    t = f if hasattr(f, "rel") else parse_fact(f)
                    if t is not None:
                        held_by_rel[normalize(t.rel)].append(t.obj)
            for t, sentence in pairs:
                verdict = compare(t.rel, t.obj, held_by_rel.get(normalize(t.rel), [])) if checking else "unchecked"
                c[f"fact:{t.rel}"] += 1; c[f"check:{verdict}"] += 1; c[f"check:{t.rel}:{verdict}"] += 1
                row = {"volume": vol, "headword": head, "entity": name, "rel": t.rel, "obj": t.obj, "sentence": sentence,
                       "check": verdict, "held": held_by_rel.get(normalize(t.rel), [])[:4]}
                facts_out.append(row)
                if verdict in ("disagree", "near"):
                    disagreements.append(row)
    n_facts = sum(v for k, v in c.items() if k.startswith("fact:"))
    log(f"entries {c['entries']:,} | persons {c['person']:,} (unnamed {c['person_unnamed']:,}) | facts {n_facts:,} | "
        f"{time.perf_counter() - t0:.0f}s")
    log("facts by relation: " + ", ".join(f"{k[5:]} {v:,}" for k, v in sorted(c.items()) if k.startswith("fact:")))
    if checking:
        checked = c["check:agree"] + c["check:near"] + c["check:disagree"]
        if wd is not None:
            log(f"wikidata: {c['wd:resolved']} entities resolved, {c['wd:ambiguous']} ambiguous (checked against neither item), "
                f"{c['wd:unresolved']} unresolved | api calls {wd.calls}")
        log(f"cross-check against {'the wiki world' if world is not None else ''}{' + ' if world is not None and wd is not None else ''}"
            f"{'wikidata' if wd is not None else ''}: agree {c['check:agree']:,} near {c['check:near']:,} disagree {c['check:disagree']:,} "
            f"unheld {c['check:unheld']:,} -> agreement on the checked part {c['check:agree'] / checked if checked else 0:.3f} ({checked:,} checked)")
        for rel in sorted({k.split(':')[1] for k in c if k.startswith('check:') and k.count(':') == 2}):
            ag, nr, dis, un = c[f"check:{rel}:agree"], c[f"check:{rel}:near"], c[f"check:{rel}:disagree"], c[f"check:{rel}:unheld"]
            log(f"  {rel}: agree {ag} near {nr} disagree {dis} unheld {un}" + (f" -> {ag / (ag + nr + dis):.3f}" if ag + nr + dis else ""))
        log(f"\nevery near / disagreement ({len(disagreements)}):")
        for r in disagreements:
            log(f"  [{r['check']:8}] {r['entity']!r} {r['rel']} -> {r['obj']!r} | world {r['held']} | {r['sentence'][:160]}")
    rng = random.Random(a.seed)
    sample = rng.sample(facts_out, min(a.sample, len(facts_out)))
    log(f"\nhand-reading sample ({len(sample)}, seed {a.seed}):")
    for r in sample:
        log(f"  [{r['check']:8}] {r['obj']!r} is the {r['rel']} of {r['entity']!r}  ||  {r['sentence'][:170]}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / f"exp_r16_encyclopedia{a.tag}.json").write_text(json.dumps(
        {"counts": dict(c), "n_facts": n_facts, "volumes": [v for v, _p, _i in src.volumes], "facts": facts_out},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {log_path.name}")


if __name__ == "__main__":
    main()
