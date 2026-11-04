"""emitter_wd_data -- order 0048: the programs adapter's training records with Wikidata's property labels as ``HOP1``'s
targets, in the question's language, mixed with v12e's records (which stay, so the adapter keeps its other jobs).

Questions come from the sets' train splits only, and no question or item of any order-0046 dev or test row (nor any
fame-set item) gets in, filtered by the folded question and by item id:
  SimpleQuestions-Wikidata (CC BY 3.0)  its train split; the gold property is the set's own (R3's rule). Rows asked
                                        from the value's side (``R`` properties) are left out: the program format has
                                        no inverse hop (a hop goes from the seed through its relation), so they are
                                        counted, not written.
  Mintaka (CC BY 4.0), EN and FR        its train split; the gold property is the one property linking a question item
                                        to an answer item in today's Wikidata (R3's rule, through the source's cache),
                                        written only when exactly one links them, item to answer. An answer-to-item
                                        link alone is an inverse row: counted, not written.
  QALD-9-plus                           none: its train split is 0046's dev.
A record: the question with its item as a slot (``[E1: ...]``, ``r3_450m.annotate``, as H-E15), and v12e's one-hop plan
with ``HOP1`` the property's label in the question's language from the frozen lexicon (none in that language: left out
and counted). A question whose item's name is not found in it is left out and counted. About 4% of the records (by a
hash of their id) are the ``val`` split the trainer reads.

    python validation/emitter_wd_data.py [--base <v12e records>] [--out <records.jsonl>]
-> the records, and ``<out>.manifest.json`` (licences, counts, shares per set and language, every file's sha256)

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

LICENCES = {"simplequestions_wd": "CC BY 3.0", "mintaka": "CC BY 4.0"}
EVAL_SETS = ("mintaka", "qald9plus", "webqsp_el", "graphq_el", "simplequestions_wd", "shadowlink", "amber", "cronquestions")
PROGRAM = ("use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n"
           "        create frame: number;\n        bind frame, SEED, \"{seed}\";\n        bind frame, HOP1, \"{rel}\";\n"
           "        return recover(frame, SEED);\n    }}\n}}\n")
VAL_SHARE = 0.04


def excluded() -> tuple[set, set]:
    """(folded questions, item ids) of every 0046 dev and test row, and the fame set's items."""
    from retrieval_sets.common import fold, read_rows
    qs, items = set(), set()
    for name in EVAL_SETS:
        for split in ("dev", "test"):
            for r in read_rows(name, split=split):
                qs.add(fold(r["question"]))
                items.update(m["qid"] for m in r["mentions"] if m.get("qid"))
    for line in open(os.path.join(HERE, "retrieval_sets", "fame_set.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        items.update(r.get("items") or ())
        if r.get("gold_item"):
            items.add(r["gold_item"])
    return qs, items


def program(seed: str, rel: str) -> str:
    return PROGRAM.replace("{seed}", seed).replace("{rel}", rel).replace("}}", "}")


def record(rid: str, subtype: str, question: str, qid: str, span: list, pid: str, label: str) -> dict:
    from retrieval_bench.r3_450m import annotate
    text = question[span[0]:span[1]].strip()
    split = "val" if int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 16 ** 8 < VAL_SHARE else "train"
    return {"id": rid, "task": "wd", "subtype": subtype, "split": split, "gold": pid, "item": qid,
            "lang": subtype.split("/")[-1], "repeat": 1,
            "reference": program(text, label), "question": question,
            "prompt": "Question:\n" + annotate(question, [{"qid": qid, "span": span}]) + "\nProgram:\n",
            "program": program("$E1", label),
            "spans": [{"id": "$E1", "text": text, "start": span[0], "end": span[1], "kind": "E"}],
            "stats": {"entities": 1, "numbers": 0, "comment_numbers": 0}}


def audit(recs: list[dict], ex_q: set, ex_items: set, lx: dict) -> dict:
    """The builder's own check on what it wrote: no question or item of 0046's dev or test, and HOP1 the gold
    property's label in the question's language. Returns the violations (empty lists when clean)."""
    import re
    from retrieval_sets.common import fold
    hop = re.compile(r'bind frame, HOP1, "([^"]*)";')
    out = {"eval_question": [], "eval_item": [], "label_not_in_language": []}
    for r in recs:
        if fold(r["question"]) in ex_q:
            out["eval_question"].append(r["id"])
        if r["item"] in ex_items:
            out["eval_item"].append(r["id"])
        m = hop.search(r["program"])
        if not m or m.group(1) != (lx["label"].get(r["gold"]) or {}).get(r["lang"]):
            out["label_not_in_language"].append(r["id"])
    return out


def candidates() -> list[dict]:
    """Every train question with its item and gold property (before the language, span and exclusion checks)."""
    from retrieval_sets.common import raw_dir
    out = []
    for i, line in enumerate(open(os.path.join(raw_dir("simplequestions_wd"), "annotated_wd_data_train_answerable.txt"), encoding="utf-8")):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            s, p, _o, q = parts[:4]
            out.append({"id": f"wd-sq-{i}", "set": "simplequestions_wd", "lang": "en", "question": q, "qid": s,
                        "pid": "P" + p[1:], "inverse": p.startswith("R")})
    for r in json.load(open(os.path.join(raw_dir("mintaka"), "mintaka_train.json"), encoding="utf-8")):
        ents = [e for e in r.get("questionEntity") or [] if e.get("entityType") == "entity" and str(e.get("name", "")).startswith("Q")]
        ans = [a["name"] for a in (r.get("answer") or {}).get("answer") or [] if isinstance(a, dict) and str(a.get("name", "")).startswith("Q")]
        for lang in ("en", "fr"):
            q = r["question"] if lang == "en" else (r.get("translations") or {}).get("fr")
            if q:
                out.append({"id": f"wd-mk-{r['id']}-{lang}", "set": "mintaka", "lang": lang, "question": q, "ents": ents,
                            "answers": ans})
    return out


def build(base: str, out: str) -> dict:
    from retrieval_bench.mentions import Gold, locate
    from retrieval_bench.r3_gold import ClaimIds
    from retrieval_sets.common import fold, source
    from retrieval_sets.lexicon import check, frozen, sha256
    lx, src = check(), source()
    ex_q, ex_items = excluded()
    rows = candidates()
    mk = [r for r in rows if r["set"] == "mintaka"]
    claims = ClaimIds(src).get(sorted({e["name"] for r in mk for e in r["ents"]}))
    gold = Gold(src)
    c = collections.Counter()
    for r in mk:                                         # Mintaka's one forward link, item to answer
        links = {(e["name"], p) for e in r["ents"] for p, vs in (claims.get(e["name"]) or {}).items() if set(vs) & set(r["answers"])}
        props = {p for _q, p in links}
        if len(props) == 1:
            r["qid"], r["pid"], r["inverse"] = sorted(links)[0][0], props.pop(), False
            r["span"] = next((e.get("span") for e in r["ents"] if e["name"] == r["qid"] and r["lang"] == "en" and e.get("span")), None)
        else:
            r["skip"] = "no forward link" if not props else "several links"
    need = sorted({r["qid"] for r in rows if r.get("qid") and not r.get("skip")})
    gold.fetch(need)
    recs = []
    for r in rows:
        k = f"{r['set']}/{r['lang']}"
        why = r.get("skip") or ("inverse" if r.get("inverse") else None)
        if not why and (fold(r["question"]) in ex_q or r["qid"] in ex_items):
            why = "in 0046's dev or test"
        label = None if why else (lx["label"].get(r["pid"]) or {}).get(r["lang"])
        if not why and not label:
            why = "no label in the question's language"
        span = None if why else (r.get("span") or (locate(r["question"], gold.names(r["qid"], r["lang"])) or (None, None))[1])
        if not why and not span:
            why = "item not found in the question"
        if why:
            c[f"{k}: {why}"] += 1
            continue
        recs.append(record(r["id"], k, r["question"], r["qid"], span, r["pid"], label))
        c[f"{k}: written"] += 1
    bad = audit(recs, ex_q, ex_items, lx)
    if any(bad.values()):
        raise SystemExit(f"the audit found records that must not be written: { {k: v[:5] for k, v in bad.items() if v} }")
    base_recs = [json.loads(line) for line in open(base, encoding="utf-8")]
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        for x in base_recs + recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    fam = collections.Counter(x["task"] for x in base_recs + recs)
    total = sum(fam.values())
    man = {"what": "order 0048: v12e's records + Wikidata's property labels as HOP1's targets (EN, FR)",
           "licences": LICENCES, "excluded": {"questions": len(ex_q), "items": len(ex_items)}, "counts": dict(sorted(c.items())),
           "share": {t: round(n / total, 4) for t, n in sorted(fam.items())},
           "wd_by_set_lang": dict(collections.Counter(x["subtype"] for x in recs)),
           "wd_split": dict(collections.Counter(x["split"] for x in recs)),
           "files": {os.path.basename(base): sha256(base), os.path.basename(out): sha256(out)},
           "lexicon_sha256": frozen()["relation_lexicon"]["sha256"],
           "audit": {k: len(v) for k, v in bad.items()}}
    json.dump(man, open(out + ".manifest.json", "w", encoding="utf-8"), indent=1)
    return man


def main() -> None:
    from cubbyllm.local_paths import expand
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(HERE), "standin", "data", "out", "emitter_sft_v12e_w_tg30_step.jsonl"))
    ap.add_argument("--out", default=None, help="default: $CUBBY_DRIVE/cubbyllm/emitter/emitter_sft_v12e_wd.jsonl")
    a = ap.parse_args()
    out = a.out or expand("$CUBBY_DRIVE/cubbyllm/emitter/emitter_sft_v12e_wd.jsonl")
    man = build(a.base, out)
    print(json.dumps({k: man[k] for k in ("counts", "share", "wd_by_set_lang", "wd_split")}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
