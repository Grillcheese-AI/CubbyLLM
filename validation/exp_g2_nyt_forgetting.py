"""H-G2 — the Zero-Forgetting benchmark on REAL dated NYT keys.

exp_a_forgetting.py established the candidate ranking on random bipolar keys.
Random keys are the memory-friendly best case: mutually quasi-orthogonal by
construction. Real text keys are CORRELATED (headlines share vocabulary), which
is exactly the regime a real memory layer lives in — and H-G2's stated purpose
for `temporal/nyt_data/` is a time-ordered forgetting benchmark over real data.

Protocol:
  - E eras (year-spread NYT archive files), P headlines each, chronological.
  - key   = sign-bundled word hypervectors of the headline (deterministic
            per-word vectors via hashed seeds) -> bipolar D, but CORRELATED
            across headlines; measured correlation reported.
  - value = random bipolar ID per article (episodic association).
  - All associations written ONLINE in strict chronological order through the
    SAME four rules as exp_a_forgetting (imported from it — one source of
    truth): Hebbian λ=0, Hebbian flat-decay, NLMS, SDM.
  - Report recall per era after everything is written: the dated forgetting
    curve, oldest era first.

Standalone; reads only headline strings + dates from D:\\ data.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_a_forgetting import (  # noqa: E402  — same rules, one source of truth
    hebbian_recall, nlms_recall, sdm_recall, cos_rows, RECALL_SIM,
)

NYT_DIR = r"D:\grillcheese_training_data\temporal\nyt_data"
D = 2048
ERAS = 8
PER_ERA = 128


def word_vec(word, d=D):
    """Deterministic bipolar vector per word (hash-seeded, BLAKE-style spirit)."""
    seed = abs(hash(("nyt-key", word.lower()))) % (2**32)
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float64)


_word_cache: dict[str, np.ndarray] = {}


def headline_key(text):
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split()
             if len(w) > 2]
    if not words:
        return None
    acc = np.zeros(D)
    for w in words:
        v = _word_cache.get(w)
        if v is None:
            v = word_vec(w)
            _word_cache[w] = v
        acc += v
    key = np.sign(acc)
    key[key == 0] = 1
    return key


def keys_from_file(path):
    docs = json.load(open(path, encoding="utf-8"))
    keys = []
    seen = set()
    for doc in docs:
        h = (doc.get("headline") or {}).get("main") or ""
        if len(h) < 20:
            continue
        k = headline_key(h)
        if k is None:
            continue
        sig = k.tobytes()
        if sig in seen:                # identical word-multiset -> skip dup
            continue
        seen.add(sig)
        keys.append(k)
        if len(keys) >= PER_ERA:
            break
    return keys


def load_eras():
    """ERAS year-spread eras; a slot walks forward past files with too few
    usable headlines (some archive months are sparse or headline-less)."""
    files = sorted(glob.glob(os.path.join(NYT_DIR, "*.json")),
                   key=lambda p: (int(os.path.basename(p).split("_")[0]),
                                  int(os.path.basename(p).split("_")[1][:-5])))
    slots = np.linspace(0, len(files) - 1, ERAS).astype(int)
    eras, used = [], set()
    for s in slots:
        for i in range(s, len(files)):
            if i in used:
                continue
            keys = keys_from_file(files[i])
            if len(keys) >= PER_ERA:
                used.add(i)
                eras.append((os.path.basename(files[i])[:-5], np.stack(keys)))
                break
        else:
            raise RuntimeError(f"no usable era file from slot {s} onward")
    return eras


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"H-G2  Dated Zero-Forgetting benchmark on real NYT headlines"
          f"  (D={D}, {ERAS} eras x {PER_ERA})")
    print("=" * 78)

    eras = load_eras()
    rng = np.random.default_rng(11)
    all_keys = np.concatenate([k for _, k in eras])           # chronological
    n = len(all_keys)
    all_vals = (rng.integers(0, 2, size=(n, D)) * 2 - 1).astype(np.float64)

    # measured key correlation vs the random-key baseline regime
    sample = rng.choice(n, size=(500, 2))
    sims = np.abs(np.einsum("ij,ij->i", all_keys[sample[:, 0]],
                            all_keys[sample[:, 1]]) / D)
    print(f"  eras: {', '.join(tag for tag, _ in eras)}")
    print(f"  total associations: {n}  |  mean |cos| between random key pairs:"
          f" {sims.mean():.4f}  (random-bipolar baseline ~{np.sqrt(2/np.pi/D):.4f})")

    methods = [
        ("Hebbian l=0 (Hopfield)", lambda k, v: hebbian_recall(k, v, D, lam=0.0)),
        ("Hebbian l=0.02 (decay)", lambda k, v: hebbian_recall(k, v, D, lam=0.02)),
        ("NLMS (error-corrective)", lambda k, v: nlms_recall(k, v, D)),
        ("SDM (Kanerva, M=D)", lambda k, v: sdm_recall(k, v, D, D)),
    ]

    q = PER_ERA
    print(f"\n  recall per era AFTER writing all {n} associations"
          f" (recall@cos>{RECALL_SIM}):")
    print("  method                    " +
          "".join(f"{tag.split('_')[0]:>7}" for tag, _ in eras) + "    all")
    for name, fn in methods:
        retrieved = fn(all_keys, all_vals)
        ok = cos_rows(retrieved, all_vals) > RECALL_SIM
        per = [float(ok[i * q:(i + 1) * q].mean()) for i in range(ERAS)]
        print(f"  {name:<26}" + "".join(f"{p*100:>6.0f}%" for p in per)
              + f"{ok.mean()*100:>6.0f}%")

    print("\n[verdict]")
    print("  Compare against exp_a_forgetting's random-key run at N=1024:"
          " correlated real-text keys are the harder, honest regime —"
          " whichever ordering survives BOTH regimes is the one to trust.")
    print(f"\n(wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
