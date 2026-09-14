"""The retrieval shape as training data: 'who / what / where is X' -> the program SEED + ASK.

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

2026-09-14 (Nick): "the model needs to learn to retrieve -- if it's a question about who is an entity,
it needs to call a program that will retrieve the details about the entity"; "the model's job is to write
the right program so the VM can retrieve it; then it will reply with the facts". The program is
`standin/ask.cot_profile(seed, kind)` -- SEED bound, ASK bound to who / what / where, no hop -- and the
host (`ask.AskLoop.profile`) executes it: the store's facts about X, each recovered by the VM, the reply.
This builder writes the (question, program) records the emitter learns it from: entities of the wiki
world typed by the relations they hold (a birth date or an occupation -> who; a location an entity is
in, a type that is a place -> where; the rest -> what), several wordings each, split by the entity's
hash like gen 3 (held = 10%). No model writes anything here; the gold is the program.

  python standin/data/build_profile_sft.py --per-kind 3000 --out standin/data/out/profile_sft.jsonl
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WORDINGS = {
    "who": ["Who is {X}?", "who is {x}", "Who was {X}?", "Who's {X}?", "Tell me about {X}.", "What do you know about {X}?",
            "Qui est {X} ?", "Who is {X}", "Give me a profile of {X}."],
    "what": ["What is {X}?", "what is {x}", "What was {X}?", "What's {X}?", "Tell me about {X}.", "What do you know about {X}?",
             "Qu'est-ce que {X} ?", "Describe {X}."],
    "where": ["Where is {X}?", "where is {x}", "Where's {X}?", "Where can I find {X}?", "Où est {X} ?", "Where is {X} located?",
              "Tell me where {X} is."],
}
PLACE_TYPES = frozenset("city town village country state province region county municipality island river lake mountain "
                        "continent capital district territory commune department ocean sea bay valley desert forest park "
                        "airport station street square building castle church cathedral bridge stadium university museum".split())
PERSON_RELS = frozenset(["birth date", "birthplace", "occupation", "spouse", "alma mater", "death date"])


def kind_of(rels: set[str], types: set[str], is_location: bool) -> str:
    if rels & PERSON_RELS:
        return "who"
    if is_location or (types & PLACE_TYPES):
        return "where"
    return "what"


def main() -> None:
    from ask import cot_profile
    from build_gen3 import cased, rec_id
    from cubbyllm.reasoning.planner import normalize
    import wikikg as wk
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=3000)
    ap.add_argument("--wordings", type=int, default=4, help="wordings sampled per entity")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-facts", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "profile_sft.jsonl"))
    a = ap.parse_args()
    t0 = time.perf_counter()
    wk.ensure_data(("triplets",))
    world = wk.wiki_world()
    by = world.index._by_subj
    location_objs: set[str] = set()
    for entries in by.values():
        for _f, t in entries:
            if t is not None and t.rel == "location":
                location_objs.add(normalize(t.obj))
    pools: dict[str, list[str]] = collections.defaultdict(list)
    for subj, entries in by.items():
        triples = [t for _f, t in entries if t is not None]
        if len(triples) < a.min_facts or len(subj) < 3 or len(subj.split()) > 5 or any(ch.isdigit() for ch in subj[:1]):
            continue
        rels = {t.rel for t in triples}
        types = {normalize(t.obj) for t in triples if t.rel in ("type", "instance")}
        pools[kind_of(rels, types, subj in location_objs)].append(triples[0].subj)
    rng = random.Random(a.seed)
    records: list[dict] = []
    c: collections.Counter = collections.Counter()
    for kind, ents in pools.items():
        rng.shuffle(ents)
        for ent in ents[:a.per_kind]:
            split = "held" if int(hashlib.sha256(normalize(ent).encode()).hexdigest(), 16) % 10 == 0 else "train"
            program = cot_profile(normalize(ent), kind)
            for tmpl in rng.sample(WORDINGS[kind], min(a.wordings, len(WORDINGS[kind]))):
                q = tmpl.replace("{X}", cased(ent)).replace("{x}", normalize(ent))
                rec = {"source": "cubbyllm/gen3_profile", "split": split, "system": None, "state": None, "repeat": 1,
                       "gold": None, "provenance": {"source": "wiki", "entity": ent, "kind": kind, "n_facts": len(by[normalize(ent)])},
                       "wording": tmpl, "aliased": [], "n_hop": 0, "task": "plan", "subtype": f"ask={kind}", "prompt": q,
                       "program": program, "vm_ok": None, "vm_result": None, "vm_error": None, "gold_match": None}
                rec["id"] = rec_id("plan", q, program)
                records.append(rec); c[f"{kind}:{split}"] += 1
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"version": "gen3_profile", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_records": len(records),
                "pools": {k: len(v) for k, v in pools.items()}, "per_kind": a.per_kind, "wordings_per_entity": a.wordings,
                "counts": dict(c), "wordings": WORDINGS, "program": "standin/ask.cot_profile(seed, kind): SEED + ASK, no hop",
                "note": "the retrieval shape: the emitter learns to write the program; the host runs it (ask.AskLoop.profile), "
                        "the VM recovers each fact, the facts are the reply. Split by the entity's hash (held = 10%)."}
    (ROOT / "validation" / "logs" / "profile_sft.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pools {manifest['pools']} | records {len(records):,} {dict(c)} | {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
