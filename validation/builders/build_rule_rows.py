"""Order 0067, step 3: the rule plan's rows, built in reverse by rule. A rule is drawn (a plan over the reply world),
its wording rendered from a template (English and French, `rule_templates.py`); no model writes a rule.

Seen kinds only (`validation/logs/rule_kinds_split_2026-11-12.json`): no held-out kind has a drawer or a wording here.
Splits:
- ``train`` / ``dev``: rows from the ``train`` wordings (dev = a seeded tenth);
- ``new_wording``: rows from the ``new`` wordings, held apart for line 3's "new wordings of seen kinds";
- one row in five (train and dev) is two rules of different kinds in one instruction, the plan their "and".
Every plan passes `rule_plan.check` (a refused plan stops the build). Pairs of kinds that cannot both hold (all
capitals and all lowercase, JSON and paragraphs, ...) are not composed (`CLASH`).

  python validation/builders/build_rule_rows.py [--out $CUBBY_DATASETS/rule_rows_0067] [--per 250] [--seed 0]
      [--wordings v1|v2]   # v2 (round 2): 10 training wordings a kind, a fresh new-wording split; out rule_rows_0067_v2
Writes rows.jsonl and manifest.json there; prints the counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import rule_templates as RT  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402

__wiring__ = "STANDALONE"


def leaf(m, op, v, arg=None) -> dict:
    out = {"m": m, "op": op, "v": v}
    if arg is not None:
        out["arg"] = arg
    return out


def both(*ps) -> dict:
    return ps[0] if len(ps) == 1 else {"and": list(ps)}


def _rel(rng, lang, op):
    return rng.choice(REL[lang][op])


def draw(kind: str, lang: str, rng: random.Random) -> tuple[dict, dict]:
    """(the template's slots, the plan) for one rule of `kind`."""
    op = rng.choice([">=", "<"])
    if kind == "number_words":
        n = rng.choice([20, 30, 50, 80, 100, 150, 200, 300])
        return {"n": n, "rel": _rel(rng, lang, op)}, leaf("word count", op, n)
    if kind == "number_sentences":
        n = rng.randint(1 if op == ">=" else 2, 10)
        return {"n": n, "rel": _rel(rng, lang, op)}, leaf("sentence count", op, n)
    if kind == "number_paragraphs":
        n = rng.randint(2, 5)
        return {"n": n}, both(leaf("paragraph count", "==", n), leaf("empty paragraph count", "==", 0))
    if kind == "multiple_sections":
        n, marker = rng.randint(2, 5), rng.choice(RT.MARKERS)
        return {"n": n, "marker": marker}, leaf("section count", ">=", n, marker)
    if kind == "number_highlighted_sections":
        n = rng.randint(1, 4)
        return {"n": n}, leaf("highlight count", ">=", n)
    if kind == "title":
        return {}, leaf("title count", ">=", 1)
    if kind == "json_format":
        return {}, leaf("json only", "==", "yes")
    if kind in ("english_capital", "english_lowercase"):
        return {}, both(leaf("case", "==", "upper" if kind == "english_capital" else "lower"), leaf("language", "==", "en"))
    if kind == "response_language":
        code = rng.choice(sorted(RT.LANG[lang]))
        return {"lang": RT.LANG[lang][code]}, leaf("language", "==", code)
    if kind == "existence":
        kw = rng.choice(RT.KEYWORDS[lang])
        return {"kw": kw}, leaf("count of the word", ">=", 1, kw)
    if kind == "frequency":
        kw, n = rng.choice(RT.KEYWORDS[lang]), rng.randint(1 if op == ">=" else 2, 4)
        return {"kw": kw, "n": n, "rel": _rel(rng, lang, op)}, leaf("count of the word", op, n, kw)
    if kind == "letter_frequency":
        letter, n = rng.choice("abcdefghijklmnopqrstuvwxyz"), rng.randint(2, 12)
        return {"letter": letter, "n": n, "rel": _rel(rng, lang, op)}, leaf("count of the letter", op, n, letter)
    if kind == "repeat_prompt":
        req = rng.choice(RT.REQUESTS[lang])
        return {"req": req}, leaf("starts with", "==", "yes", req)
    if kind == "two_responses":
        return {}, both(leaf("response count", "==", 2), leaf("empty response count", "==", 0),
                        leaf("distinct response count", "==", 2))
    if kind == "quotation":
        return {}, both(leaf("starts with", "==", "yes", '"'), leaf("ends with", "==", "yes", '"'))
    if kind == "no_markdown":
        return {}, leaf("markdown", "==", "no")
    if kind == "first":
        x = rng.choice(RT.OPENINGS[lang])
        return {"x": x}, leaf("opening", "has", x)
    raise KeyError(kind)


IFEVAL = {"number_words": "length_constraints:number_words", "number_sentences": "length_constraints:number_sentences",
          "number_paragraphs": "length_constraints:number_paragraphs", "multiple_sections": "detectable_format:multiple_sections",
          "number_highlighted_sections": "detectable_format:number_highlighted_sections", "title": "detectable_format:title",
          "json_format": "detectable_format:json_format", "english_capital": "change_case:english_capital",
          "english_lowercase": "change_case:english_lowercase", "response_language": "language:response_language",
          "existence": "keywords:existence", "frequency": "keywords:frequency", "letter_frequency": "keywords:letter_frequency",
          "repeat_prompt": "combination:repeat_prompt", "two_responses": "combination:two_responses",
          "quotation": "startend:quotation"}


def ifeval_form(kind: str, plan: dict) -> list | None:
    """[instruction id, kwargs]: the same rule as IFEval's checker states it (None for exp_if_follow's own kinds), so
    line 3 can run the benchmark's checker on it."""
    if kind not in IFEVAL:
        return None
    lf = rp.leaves(plan)[0]
    rel = {">=": "at least", "<": "less than"}.get(lf["op"])
    kw = {"number_words": lambda: {"num_words": lf["v"], "relation": rel},
          "number_sentences": lambda: {"num_sentences": lf["v"], "relation": rel},
          "number_paragraphs": lambda: {"num_paragraphs": lf["v"]},
          "multiple_sections": lambda: {"section_spliter": lf["arg"], "num_sections": lf["v"]},
          "number_highlighted_sections": lambda: {"num_highlights": lf["v"]},
          "response_language": lambda: {"language": lf["v"]},
          "existence": lambda: {"keywords": [lf["arg"]]},
          "frequency": lambda: {"keyword": lf["arg"], "frequency": lf["v"], "relation": rel},
          "letter_frequency": lambda: {"letter": lf["arg"], "let_frequency": lf["v"], "let_relation": rel},
          "repeat_prompt": lambda: {"prompt_to_repeat": lf["arg"]}}.get(kind, dict)()
    return [IFEVAL[kind], kw]


def render(kind, lang, which, rng) -> tuple[str, dict, int]:
    slots, plan = draw(kind, lang, rng)
    if which == "new" and "rel" in slots:                # the held-apart rows also word the comparison otherwise
        slots["rel"] = rng.choice(REL_NEW[lang][rp.leaves(plan)[0]["op"]])
    ts = WORDINGS[kind][lang][which]
    i = rng.randrange(len(ts))
    return ts[i].format(**slots), plan, i


def flat_and(*plans) -> dict:
    parts = [q for p in plans for q in (p["and"] if "and" in p else [p])]
    return {"and": parts}


CLASH = {frozenset({"english_capital", "english_lowercase"}), frozenset({"json_format", "number_paragraphs"}),
         frozenset({"json_format", "two_responses"}), frozenset({"json_format", "quotation"}),
         frozenset({"english_capital", "response_language"}), frozenset({"english_lowercase", "response_language"}),
         frozenset({"no_markdown", "number_highlighted_sections"}), frozenset({"repeat_prompt", "first"}),
         frozenset({"repeat_prompt", "quotation"}), frozenset({"first", "quotation"}),
         frozenset({"json_format", "first"}), frozenset({"json_format", "repeat_prompt"})}


def use_wordings(name: str) -> None:
    """``v1`` (round 1, `rule_templates.T`), ``v2`` (round 2, `rule_wordings_v2`) or ``v3`` (round 3,
    `rule_wordings_v3`, with its own training comparisons)."""
    global WORDINGS, REL, REL_NEW
    mod = None if name == "v1" else __import__(f"rule_wordings_{name}")
    WORDINGS = RT.T if mod is None else mod.templates(RT.T)
    REL = getattr(mod, "REL", RT.REL) if name in ("v3", "v4", "v4b") else RT.REL       # v4: everyday comparisons allowed
    REL_NEW = getattr(mod, "REL_NEW", RT.REL_NEW) if mod is not None else RT.REL_NEW
    global CLEAN
    CLEAN = None
    if name in ("v4", "v4b"):                            # everyday comparisons meet IFEval's short runs for some numbers:
        import bench_overlap as BO                       # a train/dev row that would share 5 words with a benchmark
        bench = BO.benchmark_grams()                     # prompt is drawn again (where the prompts are on this machine)
        CLEAN = (lambda text: not BO.overlaps(text, bench)) if bench else None


WORDINGS, REL, REL_NEW, CLEAN = RT.T, RT.REL, RT.REL_NEW, None


def _clean_train(draw):
    """`draw()` again until the training text shares 5 words with no benchmark prompt (v4; at most 50 draws)."""
    got = draw()
    for _ in range(50):
        if CLEAN is None or CLEAN(got[0]):
            return got
        got = draw()
    raise SystemExit(f"no clean wording after 50 draws: {got[0]}")


def build(per: int, seed: int) -> list[dict]:
    rng, rows = random.Random(seed), []
    kinds = sorted(WORDINGS)
    for kind in kinds:
        for lang in ("en", "fr"):
            for j in range(per):
                text, plan, t = _clean_train(lambda: render(kind, lang, "train", rng))
                rows.append({"kind": kind, "lang": lang, "instruction": text, "plan": plan, "template": t,
                             "ifeval": [f for f in [ifeval_form(kind, plan)] if f], "split": "dev" if rng.random() < 0.1 else "train"})
            for j in range(max(1, per // 5)):
                text, plan, t = render(kind, lang, "new", rng)
                rows.append({"kind": kind, "lang": lang, "instruction": text, "plan": plan, "template": t,
                             "ifeval": [f for f in [ifeval_form(kind, plan)] if f], "split": "new_wording"})
    for lang in ("en", "fr"):                             # two rules in one instruction
        for j in range(per * len(kinds) // 4):
            a, b = rng.sample(kinds, 2)
            if frozenset({a, b}) in CLASH:
                continue
            def pair(a=a, b=b):
                (ta, pa, _), (tb, pb, _) = render(a, lang, "train", rng), render(b, lang, "train", rng)
                return f"{ta} {tb}", pa, pb
            text, pa, pb = _clean_train(pair)
            rows.append({"kind": f"{a}+{b}", "lang": lang, "instruction": text, "plan": flat_and(pa, pb),
                         "template": None, "ifeval": [f for f in (ifeval_form(a, pa), ifeval_form(b, pb)) if f],
                         "split": "dev" if rng.random() < 0.1 else "train"})
    for i, r in enumerate(rows):
        why = rp.check(r["plan"]) if r["plan"] is not None else None
        if why:
            raise SystemExit(f"row {i} ({r['kind']}): its plan is refused: {why}")
        r["id"] = f"r{i:06d}"
    return rows


def with_alpaca(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """`rows` plus Nick's annotated Alpaca prompts (`alpaca_rows`): natural length rules and rule-free prompts. A train
    or dev prompt that shares 5 words with a benchmark prompt is dropped (a real prompt is not redrawn). Ids follow."""
    import alpaca_rows
    import bench_overlap as BO
    bench = BO.benchmark_grams()
    extra, dropped = [], 0
    for r in alpaca_rows.rows(seed):
        if r["split"] in ("train", "dev") and bench and BO.overlaps(r["instruction"], bench):
            dropped += 1
            continue
        extra.append(r)
    for i, r in enumerate(extra, start=len(rows)):
        r["id"] = f"r{i:06d}"
    return rows + extra, {"alpaca_rows": len(extra), "alpaca_dropped_for_benchmark_overlap": dropped,
                          "alpaca_licence": "Stanford Alpaca text CC BY-NC 4.0; annotations Nick's own"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--per", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpaca", action="store_true", help="add Nick's annotated Alpaca rows (round 4 on)")
    ap.add_argument("--wordings", default="v1", choices=("v1", "v2", "v3", "v4", "v4b"), help="round 1's wordings, or round 2's (more a kind)")
    a = ap.parse_args()
    use_wordings(a.wordings)
    out = Path(a.out or Path(local_paths.value("CUBBY_DATASETS") or ".") /
               ("rule_rows_0067" if a.wordings == "v1" else f"rule_rows_0067_{a.wordings}"))
    rows = build(a.per, a.seed)
    extra_man = {}
    if a.alpaca:
        rows, extra_man = with_alpaca(rows, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tsrc = (Path(__file__).parent / "rule_templates.py").read_bytes() + (
        b"".join((Path(__file__).parent / f"rule_wordings_{v}.py").read_bytes() for v in ("v2", "v3", "v4", "v4b")
                 if a.wordings >= v))
    counts = Counter((r["split"], r["lang"]) for r in rows)
    man = {"order": "0067", "seed": a.seed, "per": a.per, "rows": len(rows),
           "by_split_lang": {f"{s}/{l}": n for (s, l), n in sorted(counts.items())},
           "kinds": sorted(WORDINGS), "templates_sha256": hashlib.sha256(tsrc).hexdigest(), "wordings": a.wordings, **extra_man,
           "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
