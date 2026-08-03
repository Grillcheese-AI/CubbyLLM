"""Is the trained FFN low-rank? The gate for the whole FFN-compression track.

WHY. SwiGLU is ~40% of CubbyLLM's parameters (`_SwiGLU(d, mult=2)` = 3 x d x 2d
per layer), so an FFN bake-off — dense vs MoE vs low-rank hypernet — is the
largest structural experiment left. cubby-lm's `experiments/ffn_compression/
NOTES.md` says do NOT run it blind:

    "Low-rank + muP. muP is the real insight (naive low-rank *does* fail), but it
     fixes *trainability*, not the fact that FFN weight matrices are near
     full-rank -> quality ceiling stays low. Gate on the spectrum diagnostic
     (singular values of a real trained FFN: sharp knee -> low-rank viable; fat
     spectrum -> skip)."

That gate costs minutes on a checkpoint that already exists, and it either
justifies the bake-off or cancels it. Same file also separates the three axes
that get conflated in cost claims: BYTES (ternary: ~10x, measured near-free —
fp32 1.231 vs ternary 1.225), versus PARAMS/FLOPs (structurally not free).
Low-rank is a params/FLOPs play, which is the expensive kind. Hence the gate.

THE CONTROL THAT MAKES IT A MEASUREMENT. A singular-value curve always decays;
read alone it will always look "compressible" to a hopeful eye. So every trained
matrix is compared against a FRESHLY INITIALISED matrix of identical shape. Only
the GAP between them is evidence of learned low-rank structure — a trained matrix
whose spectrum matches its own init has learned nothing rank-reducible, however
steep the curve looks in isolation.

READING IT. `r90/R` is the fraction of directions holding 90% of the spectral
energy. Compression by 1/f needs r90/R comfortably below f. The verdict compares
trained against init and states which way the gate falls.

  CB_CKPT=/content/drive/MyDrive/cubbyllm/ckpt_v22.pt python validation/exp_ffn_spectrum.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

sys.argv = sys.argv[:1]
from exp_needle_recall import build  # noqa: E402

CKPT = os.environ.get("CB_CKPT", "")


def spectrum(w):
    """Singular values (desc) plus the rank fractions that decide the question."""
    s = torch.linalg.svdvals(w.float().cpu())
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    R = len(s)
    out = {"R": R, "s": s}
    for q in (0.50, 0.90, 0.95, 0.99):
        out[f"r{int(q*100)}"] = int((e < q).sum()) + 1
    # participation ratio: a scale-free "how many directions actually matter"
    out["eff"] = float((s.sum() ** 2) / (s ** 2).sum())
    return out


def main():
    if not CKPT or not os.path.exists(CKPT):
        raise SystemExit("set CB_CKPT to a checkpoint path")
    dev = torch.device("cpu")
    ck = torch.load(CKPT, map_location="cpu")
    meta = ck["meta"]
    model, d, L, V, kind = build(meta, dev)

    # init spectra FIRST, from the freshly built model — this is the control, and
    # it must be captured before the trained weights overwrite it.
    init = [{n: spectrum(getattr(f, n).weight) for n in ("g", "u", "o")}
            for f in model.backbone.ffn]

    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    trained = [{n: spectrum(getattr(f, n).weight) for n in ("g", "u", "o")}
               for f in model.backbone.ffn]
    step = meta.get("step", "?") if isinstance(meta, dict) else "?"

    n_ffn = sum(p.numel() for f in model.backbone.ffn for p in f.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print("FFN spectrum — is there learned low-rank structure to exploit?")
    print(f"  {CKPT}")
    print(f"  d={d} L={L} step={step} | FFN = {n_ffn:,} of {n_all:,} params "
          f"({n_ffn/n_all:.0%} of the model)\n")

    print("  r90/R = fraction of directions holding 90% of spectral energy.")
    print("  Compare TRAINED against its own INIT: the gap is the only evidence.\n")
    print("   layer | mat |    R |  r90/R init | r90/R trained |   delta")
    print("  -------+-----+------+-------------+---------------+---------")
    gaps = []
    for i, (a, b) in enumerate(zip(init, trained)):
        for n in ("g", "u", "o"):
            R = b[n]["R"]
            fi, ft = a[n]["r90"] / R, b[n]["r90"] / R
            gaps.append(fi - ft)
            if i in (0, L // 2, L - 1):
                print(f"  {i:>6} |  {n}  | {R:>4} | {fi:>10.3f}  |  {ft:>11.3f}  |"
                      f" {fi-ft:>+7.3f}")
    mean_t = sum(b[n]["r90"] / b[n]["R"] for b in trained for n in "guo") / (3 * L)
    mean_i = sum(a[n]["r90"] / a[n]["R"] for a in init for n in "guo") / (3 * L)
    print(f"\n  mean over all {3*L} matrices:  init {mean_i:.3f}  ->  trained {mean_t:.3f}"
          f"  (delta {mean_i-mean_t:+.3f})")

    print("\n  energy captured at a given compression factor (trained, mean):")
    print("   keep 1/f of directions |" + "".join(f"  1/{f:<3} " for f in (2, 4, 8, 16)))
    row = []
    for f in (2, 4, 8, 16):
        vals = []
        for b in trained:
            for n in "guo":
                s = b[n]["s"]
                k = max(1, b[n]["R"] // f)
                vals.append(float((s[:k] ** 2).sum() / (s ** 2).sum()))
        row.append(sum(vals) / len(vals))
    print("   energy kept            |" + "".join(f" {v:>5.1%} " for v in row))

    print("\n  === VERDICT ===")
    if mean_t > 0.5:
        print(f"  FAT SPECTRUM. 90% of the energy needs {mean_t:.0%} of the directions.")
        print("  The FFN is near full-rank; low-rank/hypernet FFN has a quality")
        print("  ceiling that muP does not lift. SKIP that arm of the bake-off and")
        print("  spend the credits on ternary (bytes, ~free) or MoE (capacity).")
    elif mean_t < 0.25 and (mean_i - mean_t) > 0.05:
        print(f"  SHARP KNEE. 90% of the energy sits in {mean_t:.0%} of directions,")
        print(f"  and that is {mean_i-mean_t:.3f} tighter than this matrix's own init —")
        print("  so it is LEARNED structure, not the shape every random matrix has.")
        print("  Low-rank FFN is worth a matched-budget A/B.")
    else:
        print(f"  AMBIGUOUS. trained {mean_t:.0%} vs init {mean_i:.0%} of directions")
        print("  for 90% energy. Either the compression on offer is mild or the")
        print("  trained spectrum barely differs from its own initialisation.")
        print("  Not a basis for spending the bake-off; re-check on a converged")
        print("  checkpoint before deciding.")
    print("\n  NOTE: this gates the low-rank/hypernet arm ONLY. Ternary is a BYTES")
    print("  play and is unaffected by rank — cubby-lm measured it near-free")
    print("  (fp32 1.231 vs ternary 1.225) and it stays the cheapest real win.")


if __name__ == "__main__":
    main()
