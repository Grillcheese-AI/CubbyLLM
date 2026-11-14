"""Order 0067: does a decoder that screens a whole-vocabulary scan by a piece's first characters (`ec7044c`) write
the answers the decoder before it wrote? Answered from stored answers, with no model run.

The screen only removes candidates, in the same ranked order. So at a step it changes the pick only if it refuses the
token the unscreened decoder picked: only if `viable(text + piece[:k])` is false for k = 1 or 2. `text + piece[:k]`
is a prefix of the stored answer, whatever the tokens were. If every prefix of an answer is viable under its row's
grammar, the screen refuses no token on that answer's path, and the screened decoder writes the same answer (same
model, same scores). A refused prefix proves nothing by itself: it names an answer to decode again.

Per split it prints: answers, answers with every prefix viable, answers with a refused prefix (ids and positions,
never text), prefixes checked; with ``--ids`` (the counted ids of a read) how many counted rows are among them.

  python validation/exp_view_prefix_closed.py <answers.jsonl> --rows <rows.jsonl> [--grammar typed|typed8|lines]
      [--ids <counted ids json>] [--procs 6] [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

__wiring__ = "STANDALONE"

SPLITS = ("dev", "test", "test_long")


def refused(grammar, answer: str) -> tuple[int | None, int]:
    """(length of the first prefix the grammar refuses, or None; how many prefixes it refuses)."""
    first, bad = None, 0
    for j in range(1, len(answer) + 1):
        if not grammar.viable(answer[:j]):
            bad += 1
            if first is None:
                first = j
    return first, bad


def grammar_class(name: str):
    from cubbyllm.speech import view_grammar as VG
    from cubbyllm.speech import view_types as VT
    return {"typed": VT.TypedGrammar, "typed8": VT.TypedGrammar8, "lines": VG.Grammar}[name]


def _one(job: tuple) -> dict:
    rid, split, instruction, answer, grammar = job
    first, bad = refused(grammar_class(grammar)(instruction), answer)
    return {"id": rid, "split": split, "prefixes": len(answer), "first": first, "refused": bad}


def summary(results: list, counted: dict) -> dict:
    out = {}
    for s in SPLITS:
        rs = [x for x in results if x["split"] == s]
        if not rs:
            continue
        broke = [x for x in rs if x["first"] is not None]
        out[s] = {"answers": len(rs), "every prefix viable": len(rs) - len(broke), "a refused prefix": len(broke),
                  "prefixes": sum(x["prefixes"] for x in rs),
                  "refused": [{"id": x["id"], "first": x["first"], "of": x["prefixes"]} for x in broke]}
        if s in counted:
            out[s]["counted rows"] = sum(x["id"] in counted[s] for x in rs)
            out[s]["counted with a refused prefix"] = sum(x["id"] in counted[s] for x in broke)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--grammar", default="typed", choices=("typed", "typed8", "lines"))
    ap.add_argument("--ids", default="", help="a read's counted ids ({split: {'ids': [...]}})")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    raw = open(a.answers, "rb").read()
    answers = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    want = {r["id"] for r in answers}
    instruction = {}
    for ln in open(a.rows, encoding="utf-8"):
        r = json.loads(ln)
        if r["id"] in want:
            instruction[r["id"]] = r["instruction"]
    counted = {}
    if a.ids:
        counted = {s: set(v["ids"]) for s, v in json.load(open(a.ids, encoding="utf-8")).items()}
    jobs = [(r["id"], r["split"], instruction[r["id"]], r["answer"], a.grammar) for r in answers]
    t0 = time.time()
    with Pool(a.procs) as pool:
        results = pool.map(_one, jobs, chunksize=4)
    splits = summary(results, counted)
    print(f"{len(results)} answers, grammar {a.grammar}, {time.time() - t0:.0f} s")
    for s, v in splits.items():
        line = (f"{s}: {v['answers']} answers; every prefix viable {v['every prefix viable']}; a refused prefix "
                f"{v['a refused prefix']}; prefixes checked {v['prefixes']}")
        if "counted rows" in v:
            line += f"; counted rows here {v['counted rows']}, with a refused prefix {v['counted with a refused prefix']}"
        print(line)
        for x in v["refused"][:20]:
            print(f"   {x['id']}: first refused at {x['first']} of {x['of']}")
    if a.out:
        rec = {"order": "0067", "what": "every character prefix of each stored answer judged by its row's grammar: "
               "with none refused, the screen of ec7044c changes no pick on these answers",
               "grammar": a.grammar, "answers": os.path.basename(a.answers),
               "answers_sha256": hashlib.sha256(raw).hexdigest(), "splits": splits}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("log:", os.path.basename(a.out))


if __name__ == "__main__":
    main()
