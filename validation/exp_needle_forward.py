"""Needle-in-a-haystack by forward passes: ``exp_needle_recall``'s probe for ``train_base``
checkpoints, batched, on whatever device is free (the CPU, here: the card serves).

Same question, same method, same controls as ``exp_needle_recall.py`` (read its docstring):
a verbatim fact planted at a depth in real-corpus filler, scored by PMI -- the candidate's
mean log-prob after the haystack minus its mean log-prob after the probe alone -- against
seven distractors, trained model against the shuffled-needle control. What differs is how a
score is computed: one forward over ``filler + needle + probe + candidate`` per candidate,
the eight candidates of a trial in one batch, the head applied only to the candidate's
positions. The step path would be the same numbers (the decode step is pinned to the forward)
at 4,096 memory-bound matvecs per prefix; the forward is a few GEMMs.

H-E20, gate 2: on the grown final (16 episodic reads, window 512) against the 450M
continuation's final (no memory) -- beyond-window depths above chance where the 450M sits at
it, or the read did not train.

    CB_CKPT=<ckpt.pt> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json> \\
    [CB_LENGTHS=1024,2048,4096] [CB_DEPTHS=0.0,0.5,1.0] [CB_TRIALS=8] [CB_DISTRACT=7] \\
    [CB_FILLER=books,wikipedia,fineweb_edu] [CB_UNTRAINED=0] [CB_OUT=<json>] \\
    python validation/exp_needle_forward.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = os.environ.get("CB_CKPT", "")
CORPUS = os.environ.get("CB_CORPUS", "")
SPM = os.environ.get("CUBBY_SPM", "")
LENGTHS = [int(x) for x in os.environ.get("CB_LENGTHS", "1024,2048,4096").split(",")]
DEPTHS = [float(x) for x in os.environ.get("CB_DEPTHS", "0.0,0.5,1.0").split(",")]
TRIALS = int(os.environ.get("CB_TRIALS", "8"))
N_DISTRACT = int(os.environ.get("CB_DISTRACT", "7"))
FILLER = os.environ.get("CB_FILLER", "books,wikipedia,fineweb_edu").split(",")
UNTRAINED = os.environ.get("CB_UNTRAINED", "0") == "1"
OUT = os.environ.get("CB_OUT", "")
THREADS = int(os.environ.get("CB_THREADS", "0"))

FACTS = [
    ("The secret access code for the north vault is {}.", "The secret access code for the north vault is"),
    ("Dr. Halloran's laboratory notebook is stored in room {}.", "Dr. Halloran's laboratory notebook is stored in room"),
    ("The shipment from Trieste weighed exactly {} kilograms.", "The shipment from Trieste weighed exactly"),
    ("Agent Petrov's callsign during the operation was {}.", "Agent Petrov's callsign during the operation was"),
]
ANSWERS = ["47213", "3092", "8814", "60517", "27384", "91650", "5573", "10428"]


class Filler:
    """Random windows of real text from the token cache's ``.u32`` shards."""

    def __init__(self, cache: str, names: list, seed: int = 99) -> None:
        self.arrays = []
        for n in names:
            p = os.path.join(cache, f"{n}.u32")
            if os.path.exists(p):
                self.arrays.append(np.memmap(p, dtype=np.uint32, mode="r"))
        if not self.arrays:
            self.arrays = [np.memmap(p, dtype=np.uint32, mode="r")
                           for p in sorted(glob.glob(os.path.join(cache, "*.u32")))[:3]]
        self.rng = np.random.default_rng(seed)

    def take(self, n: int) -> list:
        arr = self.arrays[self.rng.integers(len(self.arrays))]
        i = int(self.rng.integers(0, len(arr) - n - 1))
        return [int(t) for t in arr[i:i + n]]


@torch.no_grad()
def scores(model, prefix: list, cands: list, eos: int, dev) -> list:
    """Mean log-prob of each candidate's tokens after ``prefix``: one batched forward,
    right-padded, the head on the candidates' positions only."""
    k = max(len(c) for c in cands)
    rows = [prefix + c + [eos] * (k - len(c)) for c in cands]
    x = torch.tensor(rows, dtype=torch.long, device=dev)
    h = model.features(x)                                        # (B, S, d): the trunk, hardened
    n = len(prefix)
    lg = F.log_softmax(model.logits_from(h[:, n - 1:n - 1 + k]).float(), dim=-1)   # predicts the candidate
    out = []
    for b, c in enumerate(cands):
        out.append(float(sum(lg[b, t, c[t]] for t in range(len(c))) / len(c)))
    return out


def table(model, encode, eos, dev, filler: Filler, base: dict, shuffled: bool) -> dict:
    rows = {}
    winners: dict = {}
    print("   length |" + "".join(f"  d={int(dd * 100):>3}% " for dd in DEPTHS), flush=True)
    for ln in LENGTHS:
        row, t_row = [], time.perf_counter()
        for dep in DEPTHS:
            hits = 0
            for tr in range(TRIALS):
                fact, probe = FACTS[tr % len(FACTS)]
                ans = ANSWERS[(tr * 3) % len(ANSWERS)]
                distract = [a for a in ANSWERS if a != ans][:N_DISTRACT]
                shown = ans if not shuffled else [a for a in ANSWERS if a != ans][tr % (len(ANSWERS) - 1)]
                n_ids = encode(fact.format(shown))
                fill = filler.take(ln)
                pos = int(dep * max(0, len(fill) - len(n_ids)))
                prefix = fill[:pos] + n_ids + fill[pos:] + encode(" " + probe)
                cands = [ans] + distract
                s = scores(model, prefix, [encode(" " + c) for c in cands], eos, dev)
                pmi = [s[i] - base[(probe, c)] for i, c in enumerate(cands)]
                win = max(range(len(pmi)), key=lambda i: pmi[i])
                winners[cands[win]] = winners.get(cands[win], 0) + 1
                hits += int(win == 0)
            row.append(hits / TRIALS)
        rows[ln] = row
        print(f"  {ln:>7} |" + "".join(f"  {v:>5.1%} " for v in row) + f"   ({time.perf_counter() - t_row:.0f}s)", flush=True)
    top, n = max(winners.items(), key=lambda kv: kv[1]), sum(winners.values())
    print(f"  argmax spread: {len(winners)} distinct winners over {n} trials; most frequent {top[0]!r} "
          f"took {top[1]}/{n} ({top[1] / n:.0%})" + ("   <<< CONSTANT ARGMAX" if len(winners) == 1 else ""), flush=True)
    return {"rows": {str(k): v for k, v in rows.items()}, "winners": winners}


def main() -> None:
    t0 = time.perf_counter()
    if THREADS:
        torch.set_num_threads(THREADS)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_base import build_model
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = dict(ck["meta"])
    model = build_model(meta, dev)
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    del ck
    from cubbyllm.training.data import _load_tokenizer
    encode, _decode, eos, _ = _load_tokenizer(SPM)
    filler = Filler(CORPUS, FILLER)
    print(f"needle by forward passes — {os.path.basename(CKPT)}: d={meta['D']} L={meta['L']} "
          f"mem_every={meta.get('mem_every', 0)} window={meta['window']} | {dev}, {torch.get_num_threads()} threads")
    print(f"  {TRIALS} trials/cell, 1 true answer vs {N_DISTRACT} distractors (chance {1 / (N_DISTRACT + 1):.1%}); "
          f"filler from {[os.path.basename(a.filename) for a in filler.arrays]}\n", flush=True)
    base = {}
    for _, probe in FACTS:
        s = scores(model, encode(" " + probe), [encode(" " + c) for c in ANSWERS], eos, dev)
        for c, v in zip(ANSWERS, s):
            base[(probe, c)] = v
    print(f"  [{time.perf_counter() - t0:.0f}s] baselines done\n", flush=True)
    report = {"ckpt": CKPT, "meta": meta, "lengths": LENGTHS, "depths": DEPTHS, "trials": TRIALS,
              "distract": N_DISTRACT, "filler": FILLER}
    print("  --- TRAINED MODEL ---")
    report["trained"] = table(model, encode, eos, dev, filler, base, shuffled=False)
    print("\n  --- CONTROL: shuffled needle (no true signal present) ---")
    report["shuffled"] = table(model, encode, eos, dev, filler, base, shuffled=True)
    if UNTRAINED:
        fresh = build_model(meta, dev, seed=0)
        base0 = {}
        for _, probe in FACTS:
            s = scores(fresh, encode(" " + probe), [encode(" " + c) for c in ANSWERS], eos, dev)
            for c, v in zip(ANSWERS, s):
                base0[(probe, c)] = v
        print("\n  --- CONTROL: untrained model ---")
        report["untrained"] = table(fresh, encode, eos, dev, filler, base0, shuffled=False)
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"\n  wrote {OUT}")
    print(f"  wall {report['wall_s']}s")


if __name__ == "__main__":
    main()
