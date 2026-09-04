"""exp_g4b_wiki_world — the wikikg world as served: cost, fallback latency, functional-hop chains through the VM,
and the first SimpleQA reachability read.

Wired: STANDALONE (validation script; never imported by cubbyllm/).

Follows exp_g4 (the screen) with the two maps built (standin/data/wikikg.py: relation nouns + inverse readings):
  1. the world with inverse facts: fact count, index build time, peak memory, and the lexical fallback's latency
     per query (route_world calls every mounted world once per turn);
  2. chain questions from FUNCTIONAL hops (both lookups unique) out of the two-hop paths: yield per path, and a
     VM-verified sample through the lookup-first walk — precision against the one gold answer;
  3. SimpleQA (openai/simple-evals, 4.3k short factual questions) reachability: is the gold answer an entity of
     the graph at all, and is there a stored fact "gold is the R of E" with E named in the question — the ceiling
     of what the wiki world can answer before any search-and-learn.

  python validation/exp_g4b_wiki_world.py [--vm-sample 300] [--paths 200000]
  -> validation/logs/exp_g4b_wiki_world.{json,log}
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import tracemalloc
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import wikikg as wk  # noqa: E402
from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize  # noqa: E402

OUT_JSON = ROOT / "validation" / "logs" / "exp_g4b_wiki_world.json"
OUT_LOG = ROOT / "validation" / "logs" / "exp_g4b_wiki_world.log"
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SIMPLEQA_CACHE = ROOT / "standin" / "data" / "out" / "simpleqa" / "simple_qa_test_set.csv"
TAU_VM = 0.22021484375


def simpleqa_rows() -> list[dict]:
    if not SIMPLEQA_CACHE.exists():
        SIMPLEQA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SIMPLEQA_URL, headers={"User-Agent": "cubbyllm/standin"})
        SIMPLEQA_CACHE.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    text = SIMPLEQA_CACHE.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm-sample", type=int, default=300)
    ap.add_argument("--paths", type=int, default=200_000, help="how many random-walk paths to scan for functional chains")
    ap.add_argument("--exe", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    log(f"exp_g4b_wiki_world  git {rev}  python {platform.python_version()}  {platform.platform()}")

    # ── 1. the world ────────────────────────────────────────────────────────
    wk.ensure_data(("triplets", "paths"))
    triples = wk.load_triples()
    tracemalloc.start()
    tb = time.perf_counter()
    world = wk.wiki_world(triples=triples)
    build_s = time.perf_counter() - tb
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_fwd = len(wk.facts_from_triples(triples, inverse=False))
    log(f"world: {len(world):,} facts ({n_fwd:,} forward + {len(world) - n_fwd:,} inverse readings) from {len(triples):,} triples; "
        f"{len(world.index):,} indexed; build {build_s:.1f}s, peak {peak / 1e6:.0f} MB")
    rel_c = collections.Counter(r for _, r, _ in triples)
    mapped = sum(c for r, c in rel_c.items() if r in wk.RELATIONS)
    with_inv = sum(c for r, c in rel_c.items() if r in wk.RELATIONS and wk.RELATIONS[r][1])
    log(f"relation map: {len(wk.RELATIONS)} types cover {100 * mapped / len(triples):.1f}% of triples; inverse reading for {100 * with_inv / len(triples):.1f}%")
    queries = ["What is the capital of France?", "who is the creator of the Desarguesian plane", "Sundar Pichai employer",
               "what is the birth date of Girard Desargues", "how are you today", "Norwegian black metal scene founder"]
    tq = time.perf_counter()
    for _ in range(3):
        for q in queries:
            world(q, 3)
    fallback_ms = 1000 * (time.perf_counter() - tq) / (3 * len(queries))
    log(f"lexical fallback: {fallback_ms:.1f} ms/query (inverted index; near-universal tokens skipped past {world.BIG:,} facts)")
    for q in queries[:4]:
        hits = world(q, 2)
        log(f"  {q!r} -> {[(round(s, 3), f) for s, f in hits]}")

    # ── 2. functional-hop chains + the VM ───────────────────────────────────
    import pyarrow.parquet as pq
    tbl = pq.read_table(wk.PATHS).slice(0, args.paths)
    paths = list(zip(tbl.column("entities").to_pylist(), tbl.column("relations").to_pylist(), tbl.column("directions").to_pylist()))
    tc = time.perf_counter()
    qs = wk.chain_questions(paths, world.index, n=len(paths))          # no cap: the true functional yield
    log()
    log(f"chain questions: {len(qs):,} functional two-hop questions from {len(paths):,} paths ({100 * len(qs) / len(paths):.1f}% yield) in {time.perf_counter() - tc:.0f}s")
    for q in qs[:5]:
        log(f"  e.g. {q['question']} -> {q['answer']}")
    from cubbyllm.bridges import cubelang_client as cc

    def run_fn(source: str, fn: str) -> dict:
        return cc.run_program_proto(source, fn=fn, exe=args.exe)

    def no_search(query: str, k: int):
        return []

    outcomes = collections.Counter()
    bad = []
    tv = time.perf_counter()
    sample = qs[: args.vm_sample]
    for q in sample:
        res = pipeline_answer(q["question"], no_search, run_fn, tau_vm=TAU_VM, tau_ret=0.0, lookup=world.lookup)
        if res.verified and normalize(res.answer) == normalize(q["answer"]):
            outcomes["verified_correct"] += 1
        elif res.verified:
            outcomes["verified_wrong"] += 1
            if len(bad) < 5:
                bad.append({"q": q["question"], "answer": res.answer, "gold": q["answer"]})
        else:
            outcomes[res.reason or "unverified"] += 1
    vm_s = time.perf_counter() - tv
    log(f"VM sample {len(sample)}: {dict(outcomes)} ({1000 * vm_s / max(1, len(sample)):.0f} ms/question)")
    for b in bad:
        log(f"  wrong e.g. {b}")

    # ── 3. SimpleQA reachability ────────────────────────────────────────────
    rows = simpleqa_rows()
    qcol = next(c for c in rows[0] if c.lower() in ("problem", "question"))
    acol = next(c for c in rows[0] if c.lower() in ("answer", "gold", "target"))
    obj_norm: set[str] = set()
    subj_norm: set[str] = set()
    from cubbyllm.reasoning.planner import parse_fact
    for f in world.texts:
        t = parse_fact(f)
        if t is not None:
            obj_norm.add(normalize(t.obj))
            subj_norm.add(normalize(t.subj))
    known = keyed = 0
    keyed_examples = []
    for r in rows:
        gold = normalize(r[acol])
        qn = normalize(r[qcol])
        if gold in obj_norm or gold in subj_norm:
            known += 1
            # a stored fact "gold is the R of E" with E named in the question
            for f, t in world.index.by_object(r[acol]):
                if normalize(t.subj) in qn:
                    keyed += 1
                    if len(keyed_examples) < 5:
                        keyed_examples.append((r[qcol][:90], f))
                    break
    log()
    log(f"SimpleQA ({len(rows):,} questions): gold answer is a graph entity for {known} ({100 * known / len(rows):.1f}%); "
        f"a stored fact 'gold is the R of E' with E named in the question for {keyed} ({100 * keyed / len(rows):.1f}%) — the lookup ceiling before any search-and-learn")
    for q, f in keyed_examples:
        log(f"  e.g. {q!r} <- {f}")

    wall = time.perf_counter() - t0
    log(f"wall {wall:.0f}s")
    OUT_JSON.write_text(json.dumps({
        "git": rev, "triples": len(triples), "facts": len(world), "forward": n_fwd, "indexed": len(world.index),
        "build_s": round(build_s, 1), "peak_mb": round(peak / 1e6), "relations_mapped": len(wk.RELATIONS),
        "mapped_triples_pct": round(100 * mapped / len(triples), 1), "inverse_triples_pct": round(100 * with_inv / len(triples), 1),
        "fallback_ms_per_query": round(fallback_ms, 1), "paths_scanned": len(paths), "chain_questions": len(qs),
        "vm": {"n": len(sample), "outcomes": dict(outcomes), "ms_per_question": round(1000 * vm_s / max(1, len(sample)))},
        "simpleqa": {"n": len(rows), "entity_known": known, "keyed": keyed}, "wall_s": round(wall)}, indent=1), encoding="utf-8")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON.name, OUT_LOG.name)


if __name__ == "__main__":
    main()
