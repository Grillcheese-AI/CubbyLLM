"""import_science - MegaScience rows -> science fact worlds, through the worlds' possibility gate (H-E17).

A textbook question and its worked answer are not facts; the general statements inside them are
(other names, formulas, units, symbols, constants, classifications, parts, causes). The dataset LLM
PROPOSES them as (subject, relation, object); the HOST keeps one only when both ends are copied from the
row, the relation is in a closed vocabulary, the fact template round-trips through `parse_fact`, and every
name is ASCII after a small transliteration (the VM lexer mangles the rest). Then the worlds decide: a
fact the science worlds contradict on a functional relation is refused into the ledger, a new one waits
in the LATENT tier, and a second row stating it attests it. The LLM never judges and never supplies a
fact the row does not state; nothing here serves (`mount_science` mounts the ATTESTED facts only).

License: MegaScience and TextbookReasoning are CC-BY-NC-SA-4.0 (non-commercial, share-alike) -- research
worlds only; every world this writes carries the license and every fact the row it came from.

Stages (each re-runnable; the extraction is cached by the OpenRouter layer):
  fetch    rows from the Hub's datasets-server (a sample, spread over the split; no file download)
  extract  rows -> candidate facts, each with the host's verdict (kept or why not)
  build    kept facts -> one CapsuleStore per subject area, judged by StorePossibility, saved

  python standin/data/import_science.py fetch --n 400 --out standin/data/out/science/rows.jsonl
  python standin/data/import_science.py extract --rows standin/data/out/science/rows.jsonl \
      --extractor openrouter:<model> --out standin/data/out/science/facts.jsonl
  python standin/data/import_science.py build --facts standin/data/out/science/facts.jsonl \
      --out standin/data/out/science/worlds
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "standin"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cubbyllm.reasoning.planner import normalize, parse_fact  # noqa: E402

LICENSES = {"MegaScience/MegaScience": "CC-BY-NC-SA-4.0", "MegaScience/TextbookReasoning": "CC-BY-NC-SA-4.0"}
ROWS_API = "https://datasets-server.huggingface.co/rows"

# the closed relation vocabulary: reused relations are what the index, the relation gate and the path sampler
# need, and none contains ' of ' so 'moment of momentum' stays one subject under parse_fact(known=...)
# No 'value': in textbook exercises it is almost always this exercise's number ('50 deg is the value of dip
# angle'; smoke 2026-09-29) -- physical constants belong to a curated table, not to extraction.
RELATIONS = ("other name", "abbreviation", "symbol", "formula", "unit", "definition", "type",
             "example", "part", "property", "cause", "effect", "function", "location", "discoverer",
             "product", "component")
KNOWN = frozenset(normalize(r) for r in RELATIONS)
# The relations on which a second object CONTRADICTS the first (the store gate treats every relation as
# functional). Measured on textbook science, none is: 'formula' has several true forms ('D_c = 4r' and
# 'D_c = sqrt(3)L', the one refusal of the 144-row sample), and at 4,888 rows the functional set {abbreviation,
# symbol, unit, discoverer} refused 68 facts of which a sample of 15 held 13 truths -- unit systems ('ft*lb' vs
# 'J' for work), notations ('Nm^2/kg^2' vs 'N*m^2/kg^2', 's' vs 'seconds'), one- vs three-letter codes ('A' vs
# 'Ala' for alanine). String inequality is not a contradiction here; a law world is (dimensional analysis for
# units). Until one exists, a second object is a new LATENT fact and attestation is the only gate.
FUNCTIONAL: frozenset = frozenset()

# the micro sign as in 'um', 'uF'; the Greek letters by name
_TRANSLIT = {"µ": "u", "μ": "mu", "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "Delta",
             "ε": "epsilon", "θ": "theta", "λ": "lambda", "π": "pi", "ρ": "rho", "σ": "sigma", "Σ": "Sigma",
             "τ": "tau", "φ": "phi", "ω": "omega", "Ω": "ohm", "°": " deg", "×": "x", "·": "*", "−": "-",
             "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "≈": "~", "≤": "<=", "≥": ">=",
             "→": "->", "²": "^2", "³": "^3", "⁻": "^-", "¹": "^1", "Å": "angstrom", "⁺": "^+", "⁰": "^0",
             **{chr(0x2080 + i): f"_{i}" for i in range(10)},           # subscript digits: H₂O -> H_2O
             **{c: f"^{i}" for i, c in zip((4, 5, 6, 7, 8, 9), "⁴⁵⁶⁷⁸⁹")}}


def delatex(s: str) -> str:
    """LaTeX math as plain text: wrappers dropped, \\frac as a/b, \\times as x, \\alpha as alpha."""
    s = re.sub(r"\\[()\[\]]", " ", s or "").replace("$", " ")
    for _ in range(3):                                   # nested wrappers
        s = re.sub(r"\\(?:mathbf|mathrm|text|textbf|textit|mathit|operatorname|boldsymbol|vec|hat|bar)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\times\b", " x ", s)
    s = re.sub(r"\\cdot\b", " * ", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    return s


def plain(s: str) -> str:
    s = delatex(str(s))
    for a, b in _TRANSLIT.items():
        s = s.replace(a, b)
    return " ".join(s.split())


def _grounded(term: str, text_norm: str) -> bool:
    t = normalize(plain(term))
    return bool(t) and f" {t} " in f" {text_norm} "


def fact_text(subject: str, relation: str, obj: str) -> str:
    return f"{obj} is the {relation} of {subject}"


def check_fact(c: dict, text_norm: str) -> tuple[str | None, str | None]:
    """(None, fact) when the host keeps a candidate, else (why, None)."""
    if not isinstance(c, dict) or not all(isinstance(c.get(k), str) for k in ("subject", "relation", "object")):
        return "shape", None
    s, r, o = plain(c["subject"]).strip(" .,;:"), plain(c["relation"]).strip(" .,;:").lower(), plain(c["object"]).strip(" .,;:")
    if normalize(r) not in KNOWN:
        return "relation", None
    if len(normalize(s)) <= 1:                           # a bare variable ('x'): this problem's, not the world's
        return "variable", None
    if not (2 <= len(s) <= 80 and 1 <= len(o) <= 120):
        return "length", None
    if not (s.isascii() and o.isascii()):
        return "non_ascii", None
    if normalize(s) == normalize(o):
        return "trivial", None
    if not (_grounded(s, text_norm) and _grounded(o, text_norm)):
        return "ungrounded", None
    f = fact_text(s, r, o)
    t = parse_fact(f, known=KNOWN)
    if t is None or (normalize(t.obj), normalize(t.rel), normalize(t.subj)) != (normalize(o), normalize(r), normalize(s)):
        return "template", None                          # an ' is the ' inside a name, or a split that moves the subject
    return None, f


# ── fetch ─────────────────────────────────────────────────────────────────────
def _get_json(url: str, tries: int = 6) -> dict:
    """The datasets-server answers 5xx while its index loads ('this can take a minute'): wait and retry."""
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=90).read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
        time.sleep(min(60, 10 * (k + 1)))                # a 429 (rows API rate limit) waits the same way
    return {}


def fetch_rows(dataset: str, n: int, seed: int = 0, total: int | None = None, page: int = 100,
               get=None, drop_areas: tuple = (), threads: int = 1, log=lambda *a: None) -> list[dict]:
    """`n` rows in pages of `page` at seeded, shuffled offsets spread over the split (the datasets-server rows
    API; nothing is downloaded but the rows), leaving out rows whose subject is in `drop_areas` -- filtered here:
    the server's filter API answers in tens of seconds a page (2026-09-29), the rows API in about one.
    `get(url) -> dict` is injectable for tests."""
    get = get or _get_json
    params = {"dataset": dataset, "config": "default", "split": "train"}
    q = lambda off, ln: f"{ROWS_API}?" + urllib.parse.urlencode({**params, "offset": off, "length": ln})
    if total is None:
        total = int(get(q(0, 1)).get("num_rows_total") or 0)
    offsets = list(range(0, max(1, total - page + 1), page))
    random.Random(seed).shuffle(offsets)
    drop = set(drop_areas)
    out: list[dict] = []
    batch = max(1, threads) * 4
    done = 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for b in range(0, len(offsets), batch):
            for data in pool.map(lambda off: get(q(off, page)), offsets[b: b + batch]):
                done += 1
                for r in data.get("rows", []):
                    row = r.get("row") or {}
                    if row.get("subject") in drop:
                        continue
                    out.append({"id": f"{dataset}#{r.get('row_idx')}", "dataset": dataset, **row})
            if done % 50 < batch:
                log(f"  fetch {done} pages, {len(out)} rows kept")
            if len(out) >= n:
                return out[:n]
    return out


# ── extract ───────────────────────────────────────────────────────────────────
EXTRACT_SYSTEM = """You extract general scientific facts from a textbook question and its worked answer, for a knowledge base.
Answer with ONE JSON list and nothing else: [{"subject": ..., "relation": ..., "object": ...}, ...], each read as "<object> is the <relation> of <subject>".
Rules:
- Only facts the text itself states that stay true outside this exercise: other names, abbreviations, symbols, general formulas, units, definitions, types, examples, parts, properties, causes, effects, functions, discoverers.
- Never a value that belongs to this exercise only (x = 3, this problem's answer), never a step, never an opinion, never something the text does not say.
- subject and object are copied from the text: a term, a name, a formula or a number with its unit. Keep them short.
- relation is exactly one of: """ + ", ".join(RELATIONS) + """.
- At most 6 facts. [] when the text states none.
Example text: "Moment of momentum (also called angular momentum) is given by L = r x p. Its SI unit is kg m^2/s."
Example answer: [{"subject": "moment of momentum", "relation": "other name", "object": "angular momentum"}, {"subject": "angular momentum", "relation": "formula", "object": "L = r x p"}, {"subject": "angular momentum", "relation": "unit", "object": "kg m^2/s"}]"""


def parse_candidates(text: str) -> list:
    t = (text or "").strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        out = json.loads(t[i: j + 1])
    except ValueError:
        return []
    return out if isinstance(out, list) else []


def extract(rows: list[dict], extractor, max_chars: int = 5000, log=print,
            max_cost: float | None = None, skip_empty: bool = False) -> tuple[list[dict], dict]:
    """Every candidate the extractor proposes, with the host's verdict: {'row', 'area', 'dataset', 'triple',
    'fact' | None, 'why' | None}. `extractor` is any `.chat(system, user, max_tokens=, temperature=)`.
    `max_cost` (dollars, read from the extractor's `usage`) stops the run before the next row once spent.
    `skip_empty`: a row whose reply is empty is not counted (the cached-only read of a run still going)."""
    stats: Counter = Counter()
    out: list[dict] = []
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        spent = float((getattr(extractor, "usage", None) or {}).get("cost", 0.0) or 0.0)
        if max_cost is not None and spent >= max_cost:
            stats["stopped_at_cost_cap"] = 1
            log(f"  stopped: ${spent:.3f} spent >= cap ${max_cost:.3f} after {k - 1} rows")
            break
        q, a = str(row.get("question") or ""), str(row.get("answer") or row.get("reference_answer") or "")
        area = normalize(str(row.get("subject") or "general")).replace(" ", "_") or "general"
        user = f"Subject area: {area}\nQuestion: {q}\nAnswer: {a}"[:max_chars]
        reply = extractor.chat(EXTRACT_SYSTEM, user, max_tokens=600, temperature=0.0)
        if skip_empty and not reply:
            stats["rows_skipped_uncached"] += 1
            continue
        cands = parse_candidates(reply)
        stats["rows"] += 1
        stats["rows_with_candidates"] += bool(cands)
        text_norm = normalize(plain(q + " " + a))
        for c in cands[:6]:
            why, fact = check_fact(c, text_norm)
            stats["candidates"] += 1
            stats["kept" if why is None else f"rejected:{why}"] += 1
            out.append({"row": row.get("id"), "dataset": row.get("dataset"), "area": area,
                        "triple": [c.get(x) for x in ("subject", "relation", "object")] if isinstance(c, dict) else None,
                        "fact": fact, "why": why})
        if k % 50 == 0:
            log(f"  extract {k}/{len(rows)} rows, kept {stats['kept']}/{stats['candidates']} ({time.time() - t0:.0f}s)")
    return out, dict(stats)


# ── build ─────────────────────────────────────────────────────────────────────
def build_worlds(facts: list[dict], enc=None, functional: frozenset | None = None) -> tuple[dict, dict]:
    """Kept facts -> {world name: CapsuleStore}, one per subject area. Every add is judged by the worlds' own
    contradiction rule (`same_subject_clash`, the rule `StorePossibility` applies), across every world by index
    lookup -- no similarity search, so a build of tens of thousands of facts stays linear. A first statement is
    LATENT (one row is one unverified source) unless it clashes on a functional relation (refused, ledgered);
    a second row stating it -- the same fact up to case and punctuation -- attests it."""
    from capsules import CapsuleStore
    from cubbyllm.bridges.possibility import Possibility, Verdict, _members_about, same_subject_clash
    functional = FUNCTIONAL if functional is None else frozenset(normalize(r) for r in functional)
    worlds: dict = {}
    stats: Counter = Counter()
    rows_of: dict[str, set] = {}
    home: dict[str, tuple[str, str]] = {}                # normalized fact -> (world, the stored text)
    for r in facts:
        if not r.get("fact"):
            continue
        name = f"science_{r['area']}"
        if name not in worlds:
            worlds[name] = CapsuleStore(name=name, enc=enc, gpu=False)   # small worlds: numpy; never touch the card a serve holds
        store, key = worlds[name], " ".join(r["fact"].split())
        canon = normalize(key)
        if canon in home:                                # stated again: a second row attests it
            w, stored = home[canon]
            seen = rows_of.setdefault(canon, set())
            if r["row"] not in seen:
                seen.add(r["row"])
                stats["attested" if worlds[w].attest(stored, source=r["row"]) else "restated"] += 1
            continue
        clash = next(((w, c) for w, s in worlds.items() if (c := same_subject_clash(key, _members_about(s, key)))), None)
        t = parse_fact(key, known=KNOWN)
        if clash is not None and normalize(t.rel) in functional:
            p = Possibility(clash[0], Verdict.IMPOSSIBLE, 1.0, None, support=((clash[1], 1.0),), why="contradiction")
        else:
            if clash is not None:
                stats["second_object_multivalued"] += 1  # not a contradiction for a many-valued relation
                stats[f"second_object:{normalize(t.rel)}"] += 1
            p = Possibility(name, Verdict.UNKNOWN, 0.0, None, latent=name,
                            why="multivalued" if clash is not None else "single_source")
        if store.add(key, source=r["row"], possibility=p):
            rows_of[canon], home[canon] = {r["row"]}, (name, key)
            stats["latent"] += 1
        else:
            stats["refused" if store.last_refusal else "duplicate"] += 1
    stats["attested_facts"] = sum(len(s.texts) - len(s.latent_facts()) for s in worlds.values())
    stats["latent_facts"] = sum(len(s.latent_facts()) for s in worlds.values())
    return worlds, dict(stats)


def save_worlds(worlds: dict, folder: str, datasets: list[str], stats: dict) -> None:
    os.makedirs(folder, exist_ok=True)
    for name, store in worlds.items():
        store.save(os.path.join(folder, name))
        with open(os.path.join(folder, name, "world.json"), "w", encoding="utf-8") as f:
            json.dump({"world": name, "datasets": datasets,
                       "license": sorted({LICENSES.get(d, "unknown") for d in datasets}),
                       "n": len(store.texts), "latent": len(store.latent_facts()), "refused": len(store.ledger),
                       "rule": "latent until a second row states it; a clash is refused only on a relation built as functional",
                       "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
    with open(os.path.join(folder, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump({"worlds": {n: len(s.texts) for n, s in worlds.items()}, "stats": stats}, f, indent=1)


def mount_science(brain, folder: str, attested_only: bool = True) -> dict[str, int]:
    """Mount every saved science world on a serving brain as `brain.worlds[name]`: the ATTESTED facts only
    (a latent fact is held from speech), retrieved with the brain's own encoder. -> {name: facts mounted}."""
    from capsules import CapsuleStore
    from worlds import FactStore
    enc = getattr(brain.worlds.get("facts"), "enc", None)
    out = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(os.path.join(path, "capsules.meta.json")):
            continue
        store = CapsuleStore.load(path, gpu=False)
        texts = [t for t in store.texts if not (attested_only and store.meta[t].get("latent") is not None)]
        if texts:
            brain.worlds[name] = FactStore(texts, enc=enc, name=name)
            out[name] = len(texts)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def _jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _dump(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dataset", default="MegaScience/TextbookReasoning", choices=sorted(LICENSES))
    f.add_argument("--n", type=int, default=400)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--exclude-areas", default="", help="comma-separated subject areas to leave out (e.g. math)")
    f.add_argument("--threads", type=int, default=4, help="parallel page requests")
    f.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--rows", required=True)
    e.add_argument("--extractor", required=True, help="openrouter:<model>[@effort] or local:<gguf>")
    e.add_argument("--areas", default="", help="comma-separated subject areas to keep (default: all)")
    e.add_argument("--limit", type=int, default=0, help="first N rows after the area filter (0: all)")
    e.add_argument("--shard", type=int, default=0, help="this process takes rows[shard::of]")
    e.add_argument("--of", type=int, default=1)
    e.add_argument("--max-cost", type=float, default=None, help="dollars: stop before the next row once spent")
    e.add_argument("--cached-only", action="store_true", help="no calls: only rows whose reply is already cached (openrouter)")
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--facts", required=True, nargs="+")
    b.add_argument("--functional", default="", help="comma-separated relations on which a second object is refused (default: none)")
    b.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    say = lambda *a: print(*a, flush=True)
    if args.cmd == "fetch":
        areas = tuple(a.strip() for a in args.exclude_areas.split(",") if a.strip())
        rows = fetch_rows(args.dataset, args.n, seed=args.seed, drop_areas=areas, threads=args.threads, log=say)
        _dump(rows, args.out)
        print(f"wrote {args.out}: {len(rows)} rows, areas {dict(Counter(r.get('subject') for r in rows).most_common())}")
    elif args.cmd == "extract":
        from build_program_first import make_writer
        rows = _jsonl(args.rows)
        if args.areas:
            keep = {a.strip() for a in args.areas.split(",")}
            rows = [r for r in rows if str(r.get("subject")) in keep]
        rows = rows[: args.limit] if args.limit else rows
        rows = rows[args.shard::args.of]
        ex = make_writer(args.extractor, temperature=0.0)
        if args.cached_only:                             # read what a running extraction has already paid for
            ex.p.offline = True
        out, stats = extract(rows, ex, log=say, max_cost=args.max_cost, skip_empty=args.cached_only)
        stats["extractor"], stats["usage"] = getattr(ex, "name", "?"), getattr(ex, "usage", {})
        _dump(out, args.out)
        with open(os.path.splitext(args.out)[0] + ".stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1, default=str)
        print(json.dumps(stats, indent=1, default=str))
    else:
        facts = [r for p in args.facts for r in _jsonl(p)]
        fn = [x.strip() for x in args.functional.split(",") if x.strip()]
        worlds, stats = build_worlds(facts, functional=fn or None)
        save_worlds(worlds, args.out, sorted({r["dataset"] for r in facts if r.get("dataset")}), stats)
        print(json.dumps({"worlds": {n: len(s.texts) for n, s in worlds.items()}, "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
