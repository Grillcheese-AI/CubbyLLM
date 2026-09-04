"""exp_g4_wikikg_screen — is AutomatedScientist/wikikg-trajectories "Wikipedia already formatted for the VM"?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Owner (2026-09-04): "this dataset is gold its like wikipedia already formated for vm". The dataset (MIT):
  triplets   365,923 (subject, relation, object) from 45,416 Wikipedia articles, 180+ UPPER_SNAKE relations
  paths      1,500,000 two-hop random walks (entities, relations, directions)
  trajectories 2 x 1,000,000 tool-calling conversations: query_relations(subject, rel_type) / get_neighbors(entity, direction)
             — the exact semantics of TripleIndex.hop / by_subject / by_object.

This screen measures, without a model:
  1. the relation inventory and how the triples verbalize into TEMPLATE facts ("O is the R of S", the shape
     planner.parse_fact reads and TripleIndex indexes): X_OF relations map directly (S is the X of O), the rest
     take a noun from a small hand map, the remainder a generic lowercase phrase;
  2. how many two-hop paths become questions the planner grammar parses ("What is the R2 of the R1 of E?") when
     every hop's direction has a direct verbalization (no inverse-relation map yet);
  3. whether the lookup-first walk + the real VM verifies a sample of those chains (the chain family at scale:
     real Wikipedia entities instead of the harvest corpus's typo'd Wikidata);
  4. what the triple index costs at 366k facts (build time, memory) — and why the cosine FactStore cannot hold it
     (366k x 10,240 float32 = 15 GB): the wiki world is lookup + a lexical fallback, not cosine.

  python validation/exp_g4_wikikg_screen.py [--vm-sample 300]   # -> validation/logs/exp_g4_wikikg_screen.{json,log}
Data: standin/data/out/wikikg/{triplets,paths}.parquet (the Hub's parquet export; gitignored).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import random
import re
import subprocess
import sys
import time
import tracemalloc

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cubbyllm.reasoning import TripleIndex, answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question  # noqa: E402

DATA = ROOT / "standin" / "data" / "out" / "wikikg"
OUT_JSON = ROOT / "validation" / "logs" / "exp_g4_wikikg_screen.json"
OUT_LOG = ROOT / "validation" / "logs" / "exp_g4_wikikg_screen.log"
TAU_VM = 0.22021484375        # serve's operating point (ReasoningCortex.TAU_VM)

# non-OF relations -> the noun N in "object is the N of subject" (subject R object). Filled from the inventory
# below; anything absent takes the generic phrase (lowercased words) so the screen counts it, not hides it.
NOUN = {
    "BORN_ON": "birth date", "DIED_ON": "death date", "BORN_IN": "birthplace", "DIED_IN": "place of death",
    "WORKS_FOR": "employer", "MEMBER_OF": "group", "LOCATED_IN": "location", "PART_OF": None,     # PART_OF: X_OF form
    "PERFORMED_BY": "performer", "WRITTEN_BY": "writer", "DIRECTED_BY": "director", "PRODUCED_BY": "producer",
    "COMPOSED_BY": "composer", "FOUNDED_BY": "founder", "OWNED_BY": "owner", "PUBLISHED_BY": "publisher",
    "DEVELOPED_BY": "developer", "DESIGNED_BY": "designer", "LED_BY": "leader", "FOUNDED_IN": "founding year",
    "RELEASED_ON": "release date", "RELEASED_IN": "release year", "OCCURRED_ON": "date", "OCCURRED_IN": "place",
    "HELD_IN": "venue", "PLAYS_FOR": "team", "PLAYED_FOR": "former team", "COACHED_BY": "coach",
    "MARRIED_TO": "spouse", "CHILD_OF": None, "PARENT_OF": None, "SIBLING_OF": None,
    "POSITION_HELD": "position", "EDUCATED_AT": "alma mater", "SPEAKS": "language", "AWARDED": "award",
    "NATIONALITY": "nationality", "OCCUPATION": "occupation", "GENRE": "genre", "CAPITAL": "capital",
    "HEADQUARTERED_IN": "headquarters", "INSTANCE_OF": None, "TYPE_OF": None, "SUBCLASS_OF": None,
    "TIMELINE_EVENT": "timeline event", "BETWEEN": "neighbour", "VARIANT_OF": None, "USED_BY": "user",
    "USES": "tool", "CAUSED_BY": "cause", "CAUSES": "effect", "PRECEDED_BY": "predecessor", "FOLLOWED_BY": "successor",
    "INFLUENCED_BY": "influence", "INFLUENCED": "influencee", "STUDENT_OF": None, "TEACHER_OF": None,
}


def decamel(s: str) -> str:
    """'KeikoMatsuzaka' -> 'Keiko Matsuzaka', 'AlphabetInc_Creation' -> 'Alphabet Inc Creation'; dates untouched."""
    if re.fullmatch(r"[\d\-/.:]+", s):
        return s
    s = s.replace("_", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return " ".join(s.split())


def verbalize(subject: str, relation: str, obj: str) -> tuple[str, str, str, str] | None:
    """(fact, obj_text, rel_phrase, subj_text) in the template 'obj is the rel of subj', or None.
    X_OF: 'S is the X of O' (the triple's subject is the fact's object). Otherwise: 'O is the NOUN of S'."""
    s, o = decamel(subject), decamel(obj)
    if not s or not o or s == o:
        return None
    if relation.endswith("_OF") and NOUN.get(relation, "x") is None or (relation.endswith("_OF") and relation not in NOUN):
        rel = relation[:-3].lower().replace("_", " ")
        return (f"{s} is the {rel} of {o}", s, rel, o)
    noun = NOUN.get(relation) or relation.lower().replace("_", " ")
    return (f"{o} is the {noun} of {s}", o, noun, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm-sample", type=int, default=300)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    import pyarrow.parquet as pq
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    log(f"exp_g4_wikikg_screen  git {rev}  python {platform.python_version()}  {platform.platform()}")
    trip = pq.read_table(DATA / "triplets.parquet").to_pylist()
    paths = pq.read_table(DATA / "paths.parquet")
    log(f"triplets {len(trip):,}  paths {paths.num_rows:,}")

    # ── 1. inventory + verbalization ─────────────────────────────────────────
    rel_count = collections.Counter(r["relation"] for r in trip)
    ents = set()
    for r in trip:
        ents.add(r["subject"]); ents.add(r["object"])
    dates = sum(1 for r in trip if re.fullmatch(r"[\d\-/.:]+", r["object"]))
    log(f"entities {len(ents):,}  relations {len(rel_count)}  date-valued objects {dates:,} ({100*dates/len(trip):.1f}%)")
    of_rels = [r for r in rel_count if r.endswith("_OF")]
    mapped = [r for r in rel_count if r in NOUN and NOUN[r]]
    generic = [r for r in rel_count if not r.endswith("_OF") and not (r in NOUN and NOUN[r])]
    cov = lambda rs: sum(rel_count[r] for r in rs)
    log(f"relation forms: X_OF {len(of_rels)} rels / {cov(of_rels):,} triples ({100*cov(of_rels)/len(trip):.1f}%) | "
        f"noun-mapped {len(mapped)} / {cov(mapped):,} ({100*cov(mapped)/len(trip):.1f}%) | generic phrase {len(generic)} / {cov(generic):,} ({100*cov(generic)/len(trip):.1f}%)")
    log("top 40 relations: " + ", ".join(f"{r}={c}" for r, c in rel_count.most_common(40)))
    log("top generic (need a noun): " + ", ".join(f"{r}={rel_count[r]}" for r in sorted(generic, key=lambda r: -rel_count[r])[:25]))

    facts: dict[tuple[str, str, str], str] = {}     # (subject, relation, object) -> fact
    n_dropped = 0
    tv = time.perf_counter()
    for r in trip:
        v = verbalize(r["subject"], r["relation"], r["object"])
        if v is None:
            n_dropped += 1
            continue
        facts[(r["subject"], r["relation"], r["object"])] = v[0]
    log(f"template facts {len(facts):,} (dropped {n_dropped:,}: empty/self loops) in {time.perf_counter()-tv:.1f}s")
    reparse = sum(1 for f in list(facts.values())[:20000] if parse_fact(f) is not None)
    log(f"parse_fact round-trip on 20k facts: {reparse/200:.1f}%")
    for f in list(facts.values())[:6]:
        log(f"  e.g. {f}")

    # ── 4. the index at scale ───────────────────────────────────────────────
    tracemalloc.start()
    ti = time.perf_counter()
    index = TripleIndex(facts.values())
    build_s = time.perf_counter() - ti
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"TripleIndex over {index.n_facts:,} facts: {len(index):,} indexed, build {build_s:.1f}s, peak {peak/1e6:.0f} MB "
        f"(a cosine FactStore would need {index.n_facts*10240*4/1e9:.1f} GB of float32 rows)")

    # ── 2. paths -> questions the grammar parses ────────────────────────────
    # a hop (e -R-> e', direction) has a DIRECT question phrasing when the template fact reads "e' is the P of e":
    #   X_OF relation, backward hop (from the object side): "S is the X of O", walking O -> S asks "the X of O"
    #   non-OF relation, forward hop:                        "O is the N of S", walking S -> O asks "the N of S"
    rng = random.Random(args.seed)
    n = paths.num_rows
    sample_idx = sorted(rng.sample(range(n), 20000))
    tbl = paths.take(sample_idx).to_pylist()
    direct = ambiguous = parsed = 0
    questions: list[tuple[str, str, list[str]]] = []      # (question, gold, chain facts)
    hop_forms = collections.Counter()
    for row in tbl:
        es, rs, ds = row["entities"], row["relations"], row["directions"]
        ok = True
        phrases = []
        chain = []
        for i, (rel, d) in enumerate(zip(rs, ds)):
            a, b = es[i], es[i + 1]
            is_of = rel.endswith("_OF") and NOUN.get(rel, "x") is None or (rel.endswith("_OF") and rel not in NOUN)
            if is_of and d == "backward":       # triple (b, rel, a): "b is the X of a"; from a ask "the X of a"
                key = (b, rel, a)
            elif not is_of and d == "forward":  # triple (a, rel, b): "b is the N of a"
                key = (a, rel, b)
            else:
                ok = False
                hop_forms["needs_inverse"] += 1
                break
            if key not in facts:
                ok = False
                hop_forms["triple_missing"] += 1
                break
            f = facts[key]
            t = parse_fact(f)
            phrases.append(t.rel)
            chain.append(f)
            hop_forms["direct"] += 1
        if not ok:
            continue
        direct += 1
        # "What is the R2 of the R1 of E?" — the walk order is R1 first (tail = "R1 of E")
        q = f"What is the {phrases[1]} of the {phrases[0]} of {decamel(es[0])}?"
        plan = parse_question(q)
        if plan is None:
            continue
        parsed += 1
        c0 = index.hop(plan, 0, None)
        if len(c0) > 1:
            ambiguous += 1
        questions.append((q, decamel(es[-1]), chain))
    log()
    log(f"paths sample 20,000: fully direct {direct} ({100*direct/len(tbl):.1f}%), of which the grammar parses {parsed} "
        f"({100*parsed/max(1,direct):.1f}%); hop forms {dict(hop_forms)}; ambiguous hop 0 among parsed {ambiguous} ({100*ambiguous/max(1,parsed):.1f}%)")
    for q, g, _ in questions[:5]:
        log(f"  e.g. {q}  -> {g}")

    # ── 3. lookup walk + the VM on a sample ─────────────────────────────────
    from cubbyllm.bridges import cubelang_client as cc

    def run_fn(source: str, fn: str) -> dict:
        return cc.run_program_proto(source, fn=fn, exe=args.exe)

    def no_search(query: str, k: int):
        return []

    vm_sample = questions[: args.vm_sample]
    outcomes = collections.Counter()
    wrong_examples = []
    tvm = time.perf_counter()
    for q, gold, chain in vm_sample:
        res = pipeline_answer(q, no_search, run_fn, tau_vm=TAU_VM, tau_ret=0.0, lookup=index.hop)
        if res.verified:
            if normalize(res.answer) == normalize(gold):
                outcomes["verified_correct"] += 1
            else:
                outcomes["verified_other_answer"] += 1       # a different valid path end (ambiguity), or wrong
                if len(wrong_examples) < 6:
                    wrong_examples.append({"q": q, "answer": res.answer, "gold": gold, "facts": [h.fact for h in res.trace]})
        else:
            outcomes[res.reason or "unverified"] += 1
    vm_s = time.perf_counter() - tvm
    log()
    log(f"VM sample {len(vm_sample)}: {dict(outcomes)}  ({1000*vm_s/max(1,len(vm_sample)):.0f} ms/question, {vm_s:.0f}s)")
    for e in wrong_examples:
        log(f"  other-answer e.g. {e['q']} -> {e['answer']} (path end {e['gold']}) via {e['facts']}")

    wall = time.perf_counter() - t0
    log(f"wall {wall:.0f}s")
    OUT_JSON.write_text(json.dumps({
        "git": rev, "triplets": len(trip), "paths": n, "entities": len(ents), "relations": len(rel_count),
        "date_objects": dates, "relation_forms": {"x_of": [len(of_rels), cov(of_rels)], "noun_mapped": [len(mapped), cov(mapped)],
                                                  "generic": [len(generic), cov(generic)]},
        "top_relations": rel_count.most_common(60), "generic_relations": sorted(generic, key=lambda r: -rel_count[r]),
        "template_facts": len(facts), "index": {"indexed": len(index), "build_s": round(build_s, 2), "peak_mb": round(peak / 1e6)},
        "paths_sample": {"n": len(tbl), "direct": direct, "parsed": parsed, "hop_forms": dict(hop_forms), "ambiguous_hop0": ambiguous},
        "vm": {"n": len(vm_sample), "outcomes": dict(outcomes), "ms_per_question": round(1000 * vm_s / max(1, len(vm_sample)))},
        "wall_s": round(wall)}, indent=1), encoding="utf-8")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON.name, OUT_LOG.name)


if __name__ == "__main__":
    main()
