"""Does the sampling repetition penalty actually work, and what does it cost?

The guards in ``train_colab.sample_text`` (frequency-aware repetition penalty +
n-gram block) were added 2026-08-01 and validated only against a RIGGED model —
a fake logit vector that always wanted the same token. That proves the control
flow, not that it helps real generations, and it says nothing about the cost.
A repetition penalty is not free: it suppresses legitimately frequent tokens
("the", "of"), so pushed too hard it trades looping for unnatural text.

So this sweeps the two knobs against a real checkpoint and MEASURES, rather than
reading five samples and forming an impression. Metrics, all word-level because
that is the repetition a reader actually notices:

  rep-n      fraction of duplicate n-grams WITHIN a generation (lower = better)
  max-run    longest run of one word repeated back-to-back ("and and and" = 3)
  distinct-2 unique bigrams / total bigrams ACROSS all generations. This is the
             counterweight: a penalty can cut rep-n while collapsing every sample
             onto the same safe text, and only a cross-sample metric shows that.

It calls the SHIPPED ``sample_text``, not a reimplementation, so what is measured
is what will run.

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt.pt \
  CB_CORPUS=/content/token_cache CUBBY_SPM=.../grillcheese_bbpe128k.json \
  python validation/exp_gen_repetition.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

CKPT = os.environ.get("CB_CKPT", "")
SPM = os.environ.get("CUBBY_SPM", "")
N_GEN = int(os.environ.get("CB_GEN_N", "12"))
MAX_NEW = int(os.environ.get("CB_MAX_NEW", "200"))     # long enough for loops to show
TEMP = float(os.environ.get("CB_TEMP", "0.8"))
TOP_K = int(os.environ.get("CB_TOP_K", "40"))
# (rep_pen, no_repeat) — 1.0/0 is the unguarded baseline
SWEEP = [(1.0, 0), (1.0, 3), (1.15, 3), (1.3, 3), (1.3, 0), (1.5, 3)]


def rep_n(words, n):
    if len(words) < n + 1:
        return 0.0
    g = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def max_run(words):
    best = cur = 1 if words else 0
    for a, b in zip(words, words[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def main():
    t0 = time.perf_counter()
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT to a trained checkpoint")
    meta = torch.load(CKPT, map_location="cpu")["meta"]
    d, layers, vocab = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    # checkpoints predating the basis generator carry no "gen" key -> flat
    gen_kind = meta.get("gen", "flat")

    # train_colab reads its config from the environment AT IMPORT, and sample_text
    # closes over module-level SEQ/_autocast — so configure before importing it.
    os.environ.update({"CB_D": str(d), "CB_L": str(layers), "CB_CTX": os.environ.get("CB_CTX", "32"),
                       "CB_GEN_KIND": gen_kind, "CB_S": os.environ.get("CB_S", "1024"),
                       "CUBBY_SPM": SPM})
    spec = importlib.util.spec_from_file_location(
        "tc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_colab.py"))
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"generation repetition sweep — {CKPT}")
    print(f"  model d={d} L={layers} vocab={vocab} gen={gen_kind} | {dev}")
    print(f"  {N_GEN} samples x {MAX_NEW} tokens per config | temp {TEMP} top_k {TOP_K}\n")

    from cubbyllm.training.data import _load_tokenizer
    _, decode, eos_id, _ = _load_tokenizer(SPM)

    model = tc.build(vocab, dev)
    ck = torch.load(CKPT, map_location=dev)
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))

    print("  rep_pen  n-gram |  rep-2   rep-3   rep-4  max-run | distinct-2 | avg chars")
    print("  ---------------+------------------------------------+------------+----------")
    rows = {}

    # TARGET, measured from the real corpus rather than assumed. "Lower rep-n is
    # better" is only true down to what human text actually does — natural English
    # runs rep-2 ~0.22 / rep-3 ~0.06, so a sampler tuned until rep-n hits zero is
    # producing UNNATURALLY diverse text. This corpus is multilingual and
    # code-heavy, so its own number is the one to match, not a prose baseline.
    corpus = os.environ.get("CB_CORPUS", "")
    if corpus:
        import glob
        import json as _json

        from cubbyllm.training import WeightedCorpusPipeline
        src = os.environ.get("CB_SOURCES", "")
        srcs = (_json.load(open(src, encoding="utf-8"))["sources"] if src else
                [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                 for p in sorted(glob.glob(os.path.join(corpus, "*.u32")))])
        pipe = WeightedCorpusPipeline(srcs, SPM, corpus, seed=7, cache_only=True).prepare()
        x, _ = next(pipe.batches(N_GEN, MAX_NEW))
        rw = [decode([int(t) for t in row]).replace("\n", " ").split() for row in x]
        r2 = sum(rep_n(w, 2) for w in rw) / len(rw)
        r3 = sum(rep_n(w, 3) for w in rw) / len(rw)
        r4 = sum(rep_n(w, 4) for w in rw) / len(rw)
        mr = max(max_run(w) for w in rw)
        ag = [tuple(w[i:i + 2]) for w in rw for i in range(len(w) - 1)]
        print(f"  {'REAL TEXT':>15} | {r2:6.3f} {r3:6.3f} {r4:6.3f} {mr:>8} |"
              f"   {len(set(ag))/max(len(ag),1):6.3f}   |"
              f" {sum(len(' '.join(w)) for w in rw)/len(rw):>8.0f}  <-- TARGET")
        print("  ---------------+------------------------------------+------------+----------")
    for pen, nr in SWEEP:
        torch.manual_seed(0)                       # same draw for every config
        outs = tc.sample_text(model, decode, dev, vocab, N_GEN, max_new=MAX_NEW,
                              temp=TEMP, top_k=TOP_K, seed_id=eos_id,
                              rep_pen=pen, no_repeat=nr)
        ws = [o.split() for o in outs]
        r2 = sum(rep_n(w, 2) for w in ws) / len(ws)
        r3 = sum(rep_n(w, 3) for w in ws) / len(ws)
        r4 = sum(rep_n(w, 4) for w in ws) / len(ws)
        mr = max(max_run(w) for w in ws)
        allg = [tuple(w[i:i + 2]) for w in ws for i in range(len(w) - 1)]
        d2 = len(set(allg)) / max(len(allg), 1)
        ch = sum(len(o) for o in outs) / len(outs)
        rows[(pen, nr)] = (r2, r3, r4, mr, d2, outs)
        tag = "  <-- shipped default" if (pen, nr) == (1.3, 3) else ""
        base = "  <-- unguarded" if (pen, nr) == (1.0, 0) else ""
        print(f"  {pen:>7.2f}  {nr:>6} | {r2:6.3f} {r3:6.3f} {r4:6.3f} {mr:>8} |"
              f"   {d2:6.3f}   | {ch:>8.0f}{tag}{base}")

    b = rows[(1.0, 0)]; s = rows[(1.3, 3)]
    print(f"\n  shipped default vs unguarded: rep-3 {b[1]:.3f} -> {s[1]:.3f}"
          f"  ({(b[1]-s[1])/max(b[1],1e-9):+.0%}), max-run {b[3]} -> {s[3]},"
          f" distinct-2 {b[4]:.3f} -> {s[4]}")
    print("  Read distinct-2 as the cost check: if it FALLS while rep-n falls, the")
    print("  penalty is flattening samples toward one safe text, not fixing looping.")

    print("\n  --- samples: unguarded (1.0/0) ---")
    for i, o in enumerate(rows[(1.0, 0)][5][:3], 1):
        print(f"    [{i}] {o[:240]}")
    print("\n  --- samples: shipped default (1.3/3) ---")
    for i, o in enumerate(rows[(1.3, 3)][5][:3], 1):
        print(f"    [{i}] {o[:240]}")
    print(f"\n  wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
