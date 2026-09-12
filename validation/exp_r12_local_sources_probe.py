"""exp_r12_local_sources_probe — two numbers before any build: does the subdomain taxonomy tag
our questions (routing), and does the local DBpedia corpus hold the entities SimpleQA asks about
(a local Source for search-and-learn)?

Wired: STANDALONE. Reads off-repo data (paths as arguments); writes validation/logs/exp_r12_*.

  python validation/exp_r12_local_sources_probe.py --taxonomy <jsonl> --dbpedia <corpus parquet>
"""
from __future__ import annotations

import argparse, collections, csv, io, json, pathlib, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=str(ROOT / "standin" / "data" / "out" / "subdomain_taxonomy_patterns.jsonl"))
    ap.add_argument("--dbpedia", default=None, help="BeIR dbpedia-entity corpus parquet (_id, title, text)")
    ap.add_argument("--r11", default=str(LOGS / "exp_r11_search_learn_wikidata_alias2.json"))
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)
    from cubbyllm.reasoning.planner import normalize
    import exp_m3_cot_pipeline as m
    qs, _a, _h, _c, _store = m.load_sample(800, seed=0)
    sq = list(csv.DictReader(io.StringIO((ROOT / "standin/data/out/simpleqa/simple_qa_test_set.csv").read_text(encoding="utf-8"))))
    sqq = [r["problem"] for r in sq]
    out: dict = {}

    # -- 1. the taxonomy as a router --------------------------------------------------------
    rows = [json.loads(l) for l in open(a.taxonomy, encoding="utf-8")]
    tax = [(r["subdomain"], [re.compile(p) for p in r["patterns"]]) for r in rows]
    def tag(t): return [s for s, ps in tax if any(p.search(t) for p in ps)]
    log(f"taxonomy: {len(tax)} subdomains, {sum(len(ps) for _s, ps in tax)} patterns")
    for name, texts in (("eval 800", qs), ("SimpleQA 4326", sqq)):
        tags = [tag(t) for t in texts]
        c = collections.Counter(s for ts in tags for s in ts)
        hit = sum(1 for t in tags if t); multi = sum(1 for t in tags if len(t) > 1)
        log(f"  {name}: tagged {hit}/{len(texts)} ({100 * hit / len(texts):.0f}%), >1 tag {multi}; top {c.most_common(10)}")
        out[f"taxonomy:{name}"] = {"tagged": hit, "multi": multi, "n": len(texts), "top": c.most_common(25)}
    ex = [(t, tag(t)) for t in sqq[:400] if tag(t)][:6]
    for t, ts in ex: log(f"    e.g. {t[:80]!r} -> {ts}")

    # -- 2. DBpedia as a local Source: does it hold the entities the walk stalled on? ---------
    if a.dbpedia:
        import pyarrow.parquet as pq
        td = time.perf_counter()
        titles = set()
        pf = pq.ParquetFile(a.dbpedia)
        for i in range(pf.num_row_groups):
            for t in pf.read_row_group(i, columns=["title"]).column("title").to_pylist():
                titles.add(normalize(t))
        log(f"\ndbpedia corpus: {len(titles):,} titles in {time.perf_counter() - td:.0f}s")
        r11 = json.loads(pathlib.Path(a.r11).read_text(encoding="utf-8"))
        seeds = [x["seed"] for x in r11["rows"] if x.get("seed")]
        stalled = [x["seed"] for x in r11["rows"] if x.get("seed") and x["final"] in ("unknown_relation", "retrieval_exhausted")]
        golds = [r["answer"] for r in sq]
        for name, ents in (("emitter seeds (600 SimpleQA)", seeds), ("seeds the walk stalled on", stalled), ("SimpleQA gold answers", golds)):
            hit = sum(1 for e in ents if normalize(e) in titles)
            log(f"  {name}: {hit}/{len(ents)} have a DBpedia abstract by exact title ({100 * hit / max(1, len(ents)):.0f}%)")
            out[f"dbpedia:{name}"] = {"hit": hit, "n": len(ents)}
        # what a deterministic extractor would read off an abstract: the '(born DATE)' and 'is a/an ...' frames
        born = re.compile(r"\(born ([A-Z][a-z]+ \d{1,2}, \d{4})\)")
        isa = re.compile(r"^(?P<t>[^.]+?) (?:is|was) an? (?P<what>[^.,;]{3,60})")
        nb = nis = nrows = 0; ex = []
        for i in range(min(4, pf.num_row_groups)):
            tb = pf.read_row_group(i, columns=["title", "text"])
            for t, x in zip(tb.column("title").to_pylist(), tb.column("text").to_pylist()):
                nrows += 1
                if born.search(x or ""): nb += 1
                mm = isa.match(x or "")
                if mm: nis += 1
                if len(ex) < 4 and born.search(x or "") and mm: ex.append((t, born.search(x).group(1), mm.group("what")))
        log(f"  frames on {nrows:,} abstracts: '(born DATE)' {nb} ({100 * nb / nrows:.1f}%), 'X is a/an ...' {nis} ({100 * nis / nrows:.1f}%)")
        for t, d, w in ex: log(f"    e.g. {t!r}: born {d}; is a {w!r}")
        out["dbpedia:frames"] = {"rows": nrows, "born": nb, "isa": nis}

    wall = time.perf_counter() - t0
    (LOGS / "exp_r12_local_sources_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (LOGS / "exp_r12_local_sources_probe.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwall {wall:.0f}s | wrote exp_r12_local_sources_probe.{{json,log}}")


if __name__ == "__main__":
    main()
