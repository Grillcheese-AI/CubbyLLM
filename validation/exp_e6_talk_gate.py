"""H-E6 gate: does the talk adapter say what the VM returned -- and only that?

Runs the 450M base on grilly2 with and without the adapter (`train_talk_grilly.py`'s LoRA) on
records it never trained on: the held split of `build_ground_sft.py` (entities hashed out of
training), greedy, in the talk format. Every answer is read by the rules the host serves with:

  relation / bind / counter   correct = the asked value is there and no forbidden value is (bind: the
                              other entities' values and names; counter: the real-world value the
                              facts replaced)
  absent                      correct = "The facts don't say."; wrong = it states a forbidden value
  profile                     correct = passes the host's guard and states at least one fact's value
  every answer                guarded = `ask.grounded_prose` passes (every name and number in the facts)

The number that matters most is **wrong-but-guarded**: an answer the name-and-number guard lets
through that is still wrong -- a binding error names only grounded things, so the guard cannot see
it. That is the class the adapter itself has to keep at zero.

Second part, the exp_e5 capitals in the talk format: the facts contradict the capital
('The capital of France is Madrid' as a fact line) -> follow the facts; the facts do not mention it
(an unrelated person's birthplace) -> "The facts don't say." rather than the memorised capital.

    python validation/exp_e6_talk_gate.py --export <export dir> --adapter <adapter dir> \
        --data standin/data/out/ground_sft.jsonl [--per-family 80] [--base] [--tag _x]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    sys.path.insert(0, p)

import grilly  # noqa: E402  (grilly2)
from grilly.infer.lora import apply_lora_  # noqa: E402

from ask import date_words, grounded_prose  # noqa: E402
from build_ground_sft import ABSENT, has_value, norm, render  # noqa: E402
from exp_e5_base450m_probes import CAPITALS, load_grilly  # noqa: E402

LOG: list[str] = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def load_adapter(model, adapter_dir):
    from safetensors.numpy import load_file
    meta = json.load(open(os.path.join(adapter_dir, "talk_lora.json"), encoding="utf-8"))
    wrapped = apply_lora_(model, meta["rank"], meta["alpha"], meta["targets"])
    tensors = load_file(os.path.join(adapter_dir, "talk_lora.safetensors"))
    with grilly.no_grad():
        for path, mod in wrapped.items():
            mod.lora_A.weight.copy_(grilly.from_numpy(tensors[f"{path}.lora_A.weight"]))
            mod.lora_B.weight.copy_(grilly.from_numpy(tensors[f"{path}.lora_B.weight"]))
    return meta


def answer(model, tk, eos, prompt, max_new=64):
    ids = tk.encode(prompt).ids
    with grilly.no_grad():
        out = model.generate(grilly.tensor([ids]), max_new_tokens=max_new, eos_token_id=eos).tolist()[0][len(ids):]
    if eos in out:
        out = out[:out.index(eos)]
    text = tk.decode(out)
    return " ".join(text.split("\nQuestion:")[0].split("\nFacts:")[0].strip().split("\n")[0].split())


def judge(rec, text):
    lines = [f"{v} is the {r} of {e}" for e, r, v in rec["facts"]] + [f"{e} {r}: {v}" for e, r, v in rec["facts"]]
    guarded, bad = grounded_prose(text, lines, rec["entity"])
    refused = text.strip().lower().startswith(ABSENT.lower().rstrip("."))
    forbidden = [v for v in rec.get("forbid", []) if norm(v) and f" {norm(v)} " in f" {norm(text)} "]
    fam = rec["family"]
    if fam == "absent":
        outcome = "correct" if refused else ("wrong" if forbidden else "spoke_other")
    elif fam == "profile":
        vals = [v for _, _, v in rec["facts"]]
        outcome = "correct" if guarded and any(has_value(text, v, date_words) for v in vals) else (
            "refused" if refused else "wrong")
    else:
        gold = any(has_value(text, g, date_words) for g in rec["gold"])
        outcome = "correct" if gold and not forbidden else ("refused" if refused else "wrong")
    return {"outcome": outcome, "guarded": guarded, "refused": refused, "forbidden": forbidden,
            "unguarded_tokens": bad}


def bind_items(n, rng):
    """Held-only bind blocks built by the host. The data's own held bind records are few (23 of 3,655:
    every entity in a block must hash to held), so the gate adds these: 2-3 held entities sharing a
    relation with different values, 1-2 other facts each, a templated question about one of them. No
    answer is needed -- the gate scores what the model says against the gold and the forbidden values."""
    from build_ground_sft import facts_of, load_entities, split_of
    ents = {e: r for e, r in load_entities(3).items() if split_of(e) == "held"}
    by_rel = collections.defaultdict(list)
    for e, rels in ents.items():
        for r, vals in rels.items():
            if len(vals) == 1:
                by_rel[r].append(e)
    shared = [r for r, es in by_rel.items() if len(es) >= 6]
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        r = rng.choice(shared)
        k = rng.choice([2, 2, 3])
        es = rng.sample(by_rel[r], k)
        vals = [ents[x][r][0] for x in es]
        if len({norm(v) for v in vals}) < k:
            continue
        f = []
        for x in es:
            f += facts_of(rng, x, ents[x], n_lo=1, n_hi=3, must=r)
        rng.shuffle(f)
        ask = rng.randrange(k)
        q = f"What is the {r} of {es[ask]}?"
        out.append({"family": "bind_t", "id": f"bt{len(out):04d}", "entity": es[ask], "relation": r, "facts": f,
                    "question": q, "gold": [vals[ask]],
                    "forbid": [v for i, v in enumerate(vals) if i != ask] + [x for i, x in enumerate(es) if i != ask],
                    "prompt": render(f, q)})
    return out


def capital_items(rng):
    items = []
    for country, cap in CAPITALS:
        other = rng.choice([c for _, c in CAPITALS if c != cap])
        q = f"What is the capital of {country}?"
        items.append({"arm": "counter", "facts": [[country, "capital", other]], "question": q,
                      "follow": other, "memory": cap})
        who = rng.choice(["Tessa Velsabson", "Idris Oskdunard", "Marlo Zelkorley"])
        items.append({"arm": "unrelated", "facts": [[who, "birthplace", rng.choice(["Lima", "Oslo", "Kyoto"])]],
                      "question": q, "follow": None, "memory": cap})
    return items


def run(model, tk, eos, held, caps, label):
    rows, t0 = [], time.time()
    for rec in held:
        text = answer(model, tk, eos, rec["prompt"])
        rows.append(dict(judge(rec, text), family=rec["family"], id=rec["id"], answer=text))
    log(f"\n[{label}] {len(rows)} held records, {time.time() - t0:.0f}s")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["family"]][r["outcome"]] += 1
        by[r["family"]]["guarded"] += r["guarded"]
        by[r["family"]]["wrong_but_guarded"] += (r["outcome"] == "wrong" and r["guarded"])
    for fam in ("relation", "bind", "bind_t", "counter", "absent", "profile"):
        c = by[fam]
        n = sum(c[k] for k in ("correct", "wrong", "refused", "spoke_other"))
        if n:
            log(f"  {fam:9s} n={n:3d}  correct {c['correct'] / n:6.1%}  wrong {c['wrong']:3d}  refused "
                f"{c['refused']:3d}  other {c['spoke_other']:3d}  | guarded {c['guarded'] / n:6.1%}  "
                f"wrong-but-guarded {c['wrong_but_guarded']}")
    crow = []
    for it in caps:
        text = answer(model, tk, eos, render(it["facts"], it["question"]))
        low = norm(text)
        mem = f" {norm(it['memory'])} " in f" {low} "
        fol = it["follow"] is not None and f" {norm(it['follow'])} " in f" {low} "
        ref = text.lower().startswith(ABSENT.lower().rstrip("."))
        crow.append({"arm": it["arm"], "answer": text, "memory": mem, "follows": fol, "refused": ref})
    for arm in ("counter", "unrelated"):
        sub = [r for r in crow if r["arm"] == arm]
        log(f"  capitals/{arm:9s} n={len(sub)}  follows facts {sum(r['follows'] for r in sub)}  "
            f"from memory {sum(r['memory'] for r in sub)}  says don't-know {sum(r['refused'] for r in sub)}")
    return rows, crow, {f: dict(c) for f, c in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-family", type=int, default=80)
    ap.add_argument("--base", action="store_true", help="also run the base without the adapter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(args.export, "tokenizer.json"))
    eos = tk.token_to_id("</s>")
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    held = []
    for fam in ("relation", "bind", "counter", "absent", "profile"):
        pool = [r for r in recs if r["split"] == "held" and r["family"] == fam]
        held += rng.sample(pool, min(args.per_family, len(pool)))
    held += bind_items(args.per_family, rng)
    caps = capital_items(rng)
    model, cfg = load_grilly(args.export)
    result = {"args": vars(args), "n_held": len(held)}
    log(f"exp_e6 talk gate | {len(held)} held records, {len(caps)} capital items")
    if args.base:
        result["base"] = dict(zip(("rows", "capitals", "summary"), run(model, tk, eos, held, caps, "base, no adapter")))
    if args.adapter:
        meta = load_adapter(model, args.adapter)
        log(f"adapter: step {meta.get('step')} r={meta['rank']} alpha={meta['alpha']} ({len(meta['targets'])} modules)")
        result["adapter_meta"] = meta
        result["adapter"] = dict(zip(("rows", "capitals", "summary"), run(model, tk, eos, held, caps, "talk adapter")))
    out = os.path.join(HERE, "logs", f"exp_e6_talk_gate{args.tag}")
    json.dump(result, open(out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\nwrote {out}.json / .log")


if __name__ == "__main__":
    main()
