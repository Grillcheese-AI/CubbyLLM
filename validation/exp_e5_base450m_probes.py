"""H-E5 follow-up: probes on the finished 450M base that the training log cannot give,
run on grilly2 (the local AMD GPU, Vulkan) from the export ``export_base.py`` writes.

0. Parity. grilly2's logits against this repo's torch model on CPU, same weights,
   same tokens: a prompt past the window and 24 decode steps. grilly2's own test
   (tests/python/test_cubby.py) holds the architecture to derived bounds on a small
   model; this is the check that the export and the real checkpoint agree too.
A. Copy by distance. The training probe repeats a 512-token random half, so every
   target sits 511 tokens back: the last offset the window-512 attention can reach.
   It measures the edge only. Here the repeat distance varies (15 to 511 inside the
   window, 599 and 799 beyond it). Null arm: the second half is fresh random tokens,
   so a low loss cannot come from the instrument. Plus natural text repeated once.
B. Answer from context, or from memory. Invented people with one fact each, two
   people per context (copying the only candidate is not enough), scored by the
   log-probability of each candidate answer. Arms:
     bound      the asked person is in context (first or second, balanced)
     null       the asked person is not in context (nothing is right; how often
                does it still copy an in-context value?)
     prior      real capitals, irrelevant context (what the weights know)
     counter    real capitals, context states another city (context or memory?)
   A CubbyLLM answer comes from what the VM returned, so the base must follow the
   context over its own memory.
C. Greedy samples on the training prompts and two grounded ones, on the decode path.

Usage, in an environment with grilly2, torch and tokenizers (grilly2's own):
    python validation/export_base.py --ckpt <base450m_final.pt> --out <export dir>
    python validation/exp_e5_base450m_probes.py --export <export dir> \
        --tokenizer <grillcheese_bbpe128k.json> [--ckpt <base450m_final.pt>] [--tag _x]
``--ckpt`` enables part 0. Writes validation/logs/exp_e5_base450m_probes<tag>.{json,log}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import grilly  # noqa: E402  (grilly2)
from grilly.infer.models import CubbyConfig, CubbyForCausalLM  # noqa: E402
from grilly.infer.safetensors import load_model  # noqa: E402
from grilly.nn import functional as GF  # noqa: E402

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


def load_grilly(export_dir: str) -> CubbyForCausalLM:
    with open(os.path.join(export_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    fields = {k: v for k, v in cfg.items() if k not in ("model_type", "source")}
    model = CubbyForCausalLM(CubbyConfig(**fields))
    missing, unexpected = load_model(model, os.path.join(export_dir, "model.safetensors"))
    if missing or unexpected:
        raise SystemExit(f"export does not match the model: missing {missing}, unexpected {unexpected}")
    return model, cfg


def logprobs_of(model, x: np.ndarray, y: np.ndarray, keep: int = 0) -> np.ndarray:
    """log p(y[b, t] | x[b, :t+1]) for the last ``keep`` positions (all when 0),
    gathered on the GPU so only (B, keep) comes back. Only the kept positions
    reach the head: a (B, T, 128k) logit tensor is 0.5 GB per thousand tokens.
    Normalised over the full padded vocabulary, as training's loss was."""
    keep = keep or x.shape[1]
    with grilly.no_grad():
        logits = model(grilly.from_numpy(x.astype(np.int64)), logits_to_keep=keep).logits
        lp = GF.log_softmax(logits, -1)
        target = grilly.from_numpy(np.ascontiguousarray(y[:, -keep:]).astype(np.int64)).unsqueeze(-1)
        return lp.gather(-1, target).numpy()[..., 0].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 0. parity against the torch model
# ─────────────────────────────────────────────────────────────────────────────
PARITY_TEXT = (
    "The river rises in the hills north of the town and runs south for about forty kilometres "
    "before it joins the sea. In spring the snowmelt floods the lower fields, and the farmers "
    "have built low stone walls to keep the water off the orchards. ")


def parity(model, ckpt, encode):
    import torch

    from train_base import load_base

    ref = load_base(ckpt, "cpu")
    ids = (encode(PARITY_TEXT) * 8)[:600]                 # past the 512 window
    with torch.no_grad():
        want = ref.forward(torch.tensor([ids])).float().numpy()[0]
    with grilly.no_grad():
        got = model(grilly.tensor([ids])).logits.numpy()[0]
    scale = float(np.abs(want).max())
    diff = np.abs(got - want)
    top = float((got.argmax(-1) == want.argmax(-1)).mean())
    log(f"  prompt, 600 tokens: max |logit diff| {diff.max():.3e} (scale {scale:.1f}, "
        f"{diff.max() / scale:.1e} relative); per-position max, median {np.median(diff.max(-1)):.2e}; "
        f"argmax agrees at {top:.4f} of positions")
    # decode: prefill 40, then 24 steps teacher-forced, both sides on their step paths
    steps = ids[:64]
    with torch.no_grad():
        st = ref.init_state(1, torch.device("cpu"))
        t_out = []
        for tok in steps:
            lg, st = ref.step(torch.tensor([tok]), st)
            t_out.append(lg.float().numpy()[0])
    t_out = np.stack(t_out[39:])
    with grilly.no_grad():
        state = model.make_cache(1, 0)
        g_out = [model(grilly.tensor([steps[:40]]), past_key_values=state).logits.numpy()[0, -1]]
        for tok in steps[40:]:
            g_out.append(model(grilly.tensor([[tok]]), past_key_values=state).logits.numpy()[0, -1])
    g_out = np.stack(g_out)
    d2 = np.abs(g_out - t_out)
    log(f"  decode, 25 positions after a 40-token prefill: max |logit diff| {d2.max():.3e} "
        f"({d2.max() / np.abs(t_out).max():.1e} relative); argmax agrees at "
        f"{float((g_out.argmax(-1) == t_out.argmax(-1)).mean()):.4f}")
    del ref
    return {"prompt_max_abs": float(diff.max()), "prompt_scale": scale, "prompt_argmax_agree": top,
            "decode_max_abs": float(d2.max()),
            "decode_argmax_agree": float((g_out.argmax(-1) == t_out.argmax(-1)).mean())}


# ─────────────────────────────────────────────────────────────────────────────
# A. copy by distance
# ─────────────────────────────────────────────────────────────────────────────
def copy_arm(model, n, windows, null, seed):
    """Random half of n tokens, then the same half again (or a fresh one when
    ``null``). The target a[k+1] sits n-1 positions before the query that needs
    it, so the repeat distance is n-1. Loss over the second half, first token of
    the repeat excluded (it is not predictable)."""
    rng = np.random.default_rng(seed * 1000 + n + (7 if null else 0))
    a = rng.integers(1000, 30000, size=(windows, n))
    b = rng.integers(1000, 30000, size=(windows, n)) if null else a
    seq = np.concatenate([a, b], 1)
    per_row = []
    for w in range(windows):
        # the last n-1 positions (n .. 2n-2) predict b[1..n-1]
        per_row.append(-logprobs_of(model, seq[w:w + 1, :-1], seq[w:w + 1, 1:], keep=n - 1)[0])
    tl = np.stack(per_row)
    q = max(1, tl.shape[1] // 4)
    return {"n": n, "distance": n - 1, "null": null, "loss": float(tl.mean()),
            "first_quarter": float(tl[:, :q].mean()), "last_quarter": float(tl[:, -q:].mean()),
            "top1": float((tl < math.log(2)).mean())}


NATURAL = [
    "The river rises in the hills north of the town and runs south for about forty "
    "kilometres before it joins the sea. In spring the snowmelt floods the lower "
    "fields, and the farmers have built low stone walls to keep the water off the "
    "orchards. A wooden bridge crossed it until 1911, when the council replaced it "
    "with the iron one that still stands beside the old mill.",
    "Le marché ouvre à sept heures le samedi matin. Les producteurs arrivent avant "
    "l'aube pour installer leurs tables sous les arcades, et les premiers clients "
    "viennent surtout pour le pain et les fromages. Vers midi, la place se vide et "
    "il ne reste que les pigeons et quelques cageots de pommes invendues.",
    "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
]


def natural_copy(model, encode):
    rows = []
    for text in NATURAL:
        ids = np.asarray(encode(text))
        seq = np.concatenate([ids, ids])[None, :]
        tl = -logprobs_of(model, seq[:, :-1], seq[:, 1:])[0]
        n = len(ids)
        rows.append({"tokens": n, "first_copy": float(tl[: n - 1].mean()),
                     "second_copy": float(tl[n:].mean())})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# B. context vs memory
# ─────────────────────────────────────────────────────────────────────────────
FIRST = ["Marlo", "Tessa", "Idris", "Oona", "Bram", "Lysa", "Corin", "Fenna", "Dario",
         "Mireille", "Anouk", "Tobin", "Selka", "Ravi", "Ines", "Joren", "Kaija", "Emeric",
         "Nadia", "Pallas", "Quentin", "Rhea", "Soren", "Talia", "Ulric", "Vesna"]
SYL = ["tar", "vel", "osk", "mir", "dun", "bra", "zel", "kor", "phin", "lau", "quen", "sab"]
SUF = ["ley", "son", "ard", "ine", "ov", "ek", "wick", "at"]

RELATIONS = {
    "born": ("{p} was born in {v}.", "Where was {p} born?",
             ["Lisbon", "Oslo", "Denver", "Nairobi", "Kyoto", "Montreal", "Dublin", "Lima",
              "Prague", "Seoul", "Perth", "Cairo", "Glasgow", "Hanoi", "Tunis", "Boston"]),
    "job": ("{p} works as a {v}.", "What does {p} do for a living?",
            ["baker", "pilot", "nurse", "carpenter", "lawyer", "farmer", "painter", "dentist",
             "plumber", "teacher", "chemist", "sailor"]),
    "color": ("{p}'s favorite color is {v}.", "What is {p}'s favorite color?",
              ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink",
               "brown"]),
    "pet": ("{p} has a dog named {v}.", "What is the name of {p}'s dog?",
            ["Biscuit", "Pepper", "Shadow", "Maple", "Rocket", "Ziggy", "Luna", "Otis",
             "Clover", "Bruno"]),
}

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"),
            ("Spain", "Madrid"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
            ("England", "London"), ("Canada", "Ottawa"), ("China", "Beijing"),
            ("Greece", "Athens"), ("Ireland", "Dublin"), ("Portugal", "Lisbon"),
            ("Austria", "Vienna"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
            ("Poland", "Warsaw"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
            ("Turkey", "Ankara"), ("Thailand", "Bangkok"), ("Cuba", "Havana"),
            ("Hungary", "Budapest"), ("Argentina", "Buenos Aires")]


def person(rng, used):
    while True:
        last = (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SUF)).capitalize()
        p = f"{rng.choice(FIRST)} {last}"
        if p not in used:
            used.add(p)
            return p


def fewshot(rng):
    """Three solved examples, fixed per run, invented people only."""
    used = set()
    blocks = []
    for rel in ("born", "job", "pet"):
        fact, q, pool = RELATIONS[rel]
        p1, p2 = person(rng, used), person(rng, used)
        v1, v2 = rng.sample(pool, 2)
        who, ans = (p1, v1) if rel == "job" else (p2, v2)
        blocks.append(f"Context: {fact.format(p=p1, v=v1)} {fact.format(p=p2, v=v2)}\n"
                      f"Question: {q.format(p=who)}\nAnswer: {ans}\n\n")
    return "".join(blocks), used


def build_items(rng, n_bound, n_null, used):
    items = []
    rels = list(RELATIONS)
    for i in range(n_bound + n_null):
        rel = rels[i % len(rels)]
        fact, q, pool = RELATIONS[rel]
        p1, p2 = person(rng, used), person(rng, used)
        vals = rng.sample(pool, 4)                 # v1, v2 in context; two outside
        ctx = f"{fact.format(p=p1, v=vals[0])} {fact.format(p=p2, v=vals[1])}"
        if i < n_bound:
            second = (i // len(rels)) % 2 == 1
            who, gold = (p2, vals[1]) if second else (p1, vals[0])
            items.append({"arm": "bound", "rel": rel, "position": 2 if second else 1,
                          "context": ctx, "question": q.format(p=who), "gold": gold,
                          "in_context": vals[:2], "candidates": vals})
        else:
            p3 = person(rng, used)
            items.append({"arm": "null", "rel": rel, "position": 0, "context": ctx,
                          "question": q.format(p=p3), "gold": None,
                          "in_context": vals[:2], "candidates": vals})
    return items


def capital_items(rng, used):
    items = []
    fact, _, pool = RELATIONS["born"]
    for country, cap in CAPITALS:
        others = [c for _, c in CAPITALS if c != cap]
        wrong = rng.sample(others, 3)
        q = f"What is the capital of {country}?"
        p = person(rng, used)
        irrelevant = fact.format(p=p, v=rng.choice([v for v in pool if v != cap]))
        items.append({"arm": "prior", "rel": "capital", "context": irrelevant, "question": q,
                      "gold": cap, "memory": cap, "candidates": [cap] + wrong})
        items.append({"arm": "counter", "rel": "capital",
                      "context": f"The capital of {country} is {wrong[0]}.", "question": q,
                      "gold": wrong[0], "memory": cap, "candidates": [cap] + wrong})
    return items


def score_items(model, encode, prefix, items, batch_seqs):
    """Sum log-probability of each candidate answer after '... Answer:'. Right-padded
    batches: the model is causal (running-mean context, windowed attention, scan),
    so padding after a candidate cannot change its score."""
    seqs = []
    for it_i, it in enumerate(items):
        prompt = encode(f"{prefix}Context: {it['context']}\nQuestion: {it['question']}\nAnswer:")
        for c_i, cand in enumerate(it["candidates"]):
            seqs.append((it_i, c_i, prompt, encode(" " + cand)))
    scores = [[None] * len(it["candidates"]) for it in items]
    t0 = time.time()
    for s in range(0, len(seqs), batch_seqs):
        chunk = seqs[s:s + batch_seqs]
        L = max(len(p) + len(c) for _, _, p, c in chunk)
        x = np.zeros((len(chunk), L), np.int64)
        for r, (_, _, p, c) in enumerate(chunk):
            x[r, :len(p) + len(c)] = p + c
        y = np.concatenate([x[:, 1:], np.zeros((len(chunk), 1), np.int64)], 1)
        # candidate token j is predicted at position len(p)-1+j; keep only from the
        # shortest prompt's last position on, so the head sees a few positions a row
        keep = L - min(len(p) for _, _, p, _ in chunk) + 1
        lp = logprobs_of(model, x, y, keep=keep)
        for r, (it_i, c_i, p, c) in enumerate(chunk):
            at = len(p) - 1 - (L - keep)
            scores[it_i][c_i] = (float(lp[r, at:at + len(c)].sum()), len(c))
    return scores, time.time() - t0


def verdicts(items, scores):
    rows = []
    for it, sc in zip(items, scores):
        best = max(range(len(sc)), key=lambda k: sc[k][0])
        pick = it["candidates"][best]
        row = {"arm": it["arm"], "rel": it["rel"], "pick": pick, "gold": it["gold"],
               "scores": [round(s, 3) for s, _ in sc]}
        if it["arm"] == "bound":
            row["outcome"] = ("correct" if pick == it["gold"] else
                              "other_in_context" if pick in it["in_context"] else "outside")
            row["position"] = it["position"]
        elif it["arm"] == "null":
            row["outcome"] = "copied_in_context" if pick in it["in_context"] else "outside"
        elif it["arm"] == "prior":
            row["outcome"] = "memory" if pick == it["memory"] else "other"
        else:
            row["outcome"] = ("context" if pick == it["gold"] else
                              "memory" if pick == it["memory"] else "other")
        rows.append(row)
    return rows


def tally(rows, arm):
    out = {}
    for r in rows:
        if r["arm"] == arm:
            out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return sum(out.values()), out


# ─────────────────────────────────────────────────────────────────────────────
# C. samples
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PROMPTS = [
    "The history of Quebec City begins",
    "La ville de Lévis est située sur la rive sud",
    "Question: Why does ice float on water?\nAnswer:",
    "def is_prime(n):\n",
    "Context: Bill Haslam's father is Jim Haslam.\nQuestion: Who is Bill Haslam's father?\nAnswer:",
    "Question: What is the capital of Canada?\nAnswer:",
]


def sample_rows(model, encode, decode, vocab_real, n_tokens):
    rows = []
    for p in SAMPLE_PROMPTS:
        ids = encode(p)
        t0 = time.time()
        out = model.generate(grilly.tensor([ids]), max_new_tokens=n_tokens,
                             repetition_penalty=1.1).tolist()[0][len(ids):]
        secs = time.time() - t0
        rows.append({"prompt": p, "output": decode([t for t in out if t < vocab_real]),
                     "seconds": round(secs, 2), "tok_s": round(len(out) / secs, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export_base.py's output directory")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt", default="", help="the torch checkpoint, for part 0 (parity)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-windows", type=int, default=8)
    ap.add_argument("--bound", type=int, default=160)
    ap.add_argument("--null", type=int, default=40)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--sample-tokens", type=int, default=40)
    ap.add_argument("--skip", default="", help="comma list of parts to skip: copy,context,samples")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)

    def encode(s):
        return tk.encode(s).ids

    def decode(ids):
        return tk.decode(list(ids))

    t0 = time.time()
    model, cfg = load_grilly(args.export)
    vocab_real = model.config.vocab_real
    src = cfg.get("source", {})
    log(f"exp_e5 base450m probes | grilly2 on the local GPU | load {time.time() - t0:.0f}s | "
        f"{src.get('checkpoint')} step {src.get('step')}")
    result = {"source": src, "args": vars(args)}

    if args.ckpt:
        log("\n0. parity: grilly2 vs this repo's torch model on CPU, same weights")
        result["parity"] = parity(model, args.ckpt, encode)

    if "copy" not in skip:
        log("\nA. copy by distance (random tokens 1000..29999; loss on the repeat, nats)")
        log(f"{'n':>5} {'dist':>5} {'arm':>6} {'loss':>7} {'q1':>7} {'q4':>7} {'p>.5':>6}   note")
        arms = []
        for n in (16, 64, 128, 256, 384, 512, 600, 800):
            r = copy_arm(model, n, args.copy_windows, False, args.seed)
            note = ("training probe (window edge)" if n == 512 else
                    "beyond window" if n - 1 >= 512 else "inside window")
            arms.append(r)
            log(f"{n:5d} {n - 1:5d} {'repeat':>6} {r['loss']:7.3f} {r['first_quarter']:7.3f} "
                f"{r['last_quarter']:7.3f} {r['top1']:6.3f}   {note}")
        for n in (64, 512):
            r = copy_arm(model, n, args.copy_windows, True, args.seed)
            arms.append(r)
            log(f"{n:5d} {n - 1:5d} {'null':>6} {r['loss']:7.3f} {r['first_quarter']:7.3f} "
                f"{r['last_quarter']:7.3f} {r['top1']:6.3f}   fresh tokens, nothing to copy")
        log(f"  (uniform over the 29,000 probe ids = {math.log(29000):.2f}; p>.5 = share of repeat "
            f"tokens given more than half the probability)")
        nat = natural_copy(model, encode)
        for label, r in zip(("EN prose", "FR prose", "code"), nat):
            log(f"  natural {label:9s} {r['tokens']:4d} tok: first copy {r['first_copy']:.3f}, "
                f"second copy {r['second_copy']:.3f}")
        result["copy"] = arms
        result["natural_copy"] = nat

    if "context" not in skip:
        rng = random.Random(args.seed)
        prefix, used = fewshot(rng)
        items = build_items(rng, args.bound, args.null, used) + capital_items(rng, used)
        scores, secs = score_items(model, encode, prefix, items, args.batch_seqs)
        rows = verdicts(items, scores)
        log(f"\nB. context vs memory ({len(items)} items, 4 candidates each, {secs:.0f}s)")
        for arm, what in (("bound", "asked person in context"),
                          ("null", "asked person NOT in context"),
                          ("prior", "real capitals, irrelevant context"),
                          ("counter", "real capitals, context states another city")):
            n, t = tally(rows, arm)
            log(f"  {arm:8s} n={n:4d}  {json.dumps(t)}   ({what})")
        for pos in (1, 2):
            sub = [r for r in rows if r["arm"] == "bound" and r.get("position") == pos]
            c = sum(r["outcome"] == "correct" for r in sub)
            log(f"    bound, asked person {'first' if pos == 1 else 'second'}: {c}/{len(sub)} correct")
        for rel in RELATIONS:
            sub = [r for r in rows if r["arm"] == "bound" and r["rel"] == rel]
            c = sum(r["outcome"] == "correct" for r in sub)
            log(f"    bound, {rel:6s}: {c}/{len(sub)} correct")
        known = {it["question"] for it, r in zip(items, rows)
                 if r["arm"] == "prior" and r["outcome"] == "memory"}
        t = {}
        for it, r in zip(items, rows):
            if r["arm"] == "counter" and it["question"] in known:
                t[r["outcome"]] = t.get(r["outcome"], 0) + 1
        log(f"  counter, on the {len(known)} capitals the weights know: {json.dumps(t)}")
        log(f"  few-shot prefix:\n{prefix}")
        result["context"] = {"prefix": prefix, "items": items, "rows": rows}

    if "samples" not in skip:
        log("\nC. greedy samples (repetition penalty 1.1), decode path")
        rows = sample_rows(model, encode, decode, vocab_real, args.sample_tokens)
        for r in rows:
            log(f"  [{r['seconds']}s, {r['tok_s']} tok/s] {r['prompt']!r}\n     -> {r['output']!r}")
        result["samples"] = rows

    out = os.path.join(HERE, "logs", f"exp_e5_base450m_probes{args.tag}")
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    with open(out + ".log", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log(f"\nwrote {out}.json / .log  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
