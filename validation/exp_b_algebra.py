"""H-B1 vs H-B2 — pick a binding-head algebra, and check the predecessor fits.

Three questions §9 step 5 and H-B3 raise:

  (H-B3 corollary) cubby-lm's VSABindingHead operates in CONTINUOUS space (a
      learned linear query projected into R^D, cosine vs a bipolar codebook).
      cubby-concepts' bipolar bind/bundle/permute (H-B1) require {-1,+1} inputs.
      So: is the head's output already bipolar-compatible, or does dropping in
      the H-B1 algebra need a binarization step? We measure the sign() info loss.

  (H-B1 vs H-B2) The two verified reference algebras are genuinely different:
      - BSC / MAP  (cubby-concepts): bind = elementwise product, self-inverse.
      - HRR        (GrillCheese):    bind = circular convolution, unbind = corr.
      HRR handles CONTINUOUS vectors natively; BSC is bipolar. Since the head's
      query is continuous, that difference is the whole decision. We benchmark
      both on the SAME role-filler unbind task at D=10240 (cubby-lm's head dim)
      across noise/superposition load, and report which degrades more gracefully.

Imports the real cubby-concepts core (H-B1) and cubby-lm's real head. HRR is the
~10-line textbook algebra (NumPy+FFT), matching GrillCheese's holographic test.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-concepts")
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")
from vsa.core import bind as bsc_bind, generate_random_hypervector  # noqa: E402
from cubby.trunk_torch.vsa_binding_head import VSABindingHead  # noqa: E402

D = 10240


# ── HRR algebra (GrillCheese holographic_relation_test form) ────────────────
def hrr_vec(n, d, rng):
    """Standard HRR atoms: N(0, 1/d) real vectors (unit expected norm)."""
    return rng.standard_normal((n, d)) / np.sqrt(d)


def hrr_bind(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def hrr_unbind(c, b):
    # correlation = convolution with the involution of b
    return np.real(np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(b))))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Part 1: is cubby-lm's head output bipolar-compatible? ───────────────────
def head_binarization_loss():
    """Two regimes:
    (no-signal) an UNTRAINED head's random query — argmax over the codebook is
        decided by ~0.04-cosine crosstalk noise, so sign() freely reshuffles the
        winner. Worst case, reported for honesty, but not the operating point.
    (signal)    a query with cos=0.55 alignment to its target codeword (a
        realistically trained projection, cf. exp_c7) — does sign(query) keep
        the argmax on the target? This is the number that decides H-B1's
        'needs binarization' condition."""
    torch.manual_seed(0)
    head = VSABindingHead(d_model=256, vocab_size=4096, d_vsa=D)
    h = torch.randn(512, 256)
    with torch.no_grad():
        z = head.query_proj(h)                       # the continuous VSA query
        z = torch.nn.functional.normalize(z, dim=-1)
    z = z.numpy()
    z_bipolar = np.sign(z)
    z_bipolar[z_bipolar == 0] = 1
    per = [cos(z[i], z_bipolar[i]) for i in range(len(z))]
    cb = head.codebook_unit.numpy()                  # (V, D) bipolar/sqrt(D)
    agree_nosig = float((np.argmax(z @ cb.T, 1) ==
                         np.argmax(z_bipolar @ cb.T, 1)).mean())

    # signal regime: synth queries aligned cos=0.55 with a chosen codeword
    rng = np.random.default_rng(0)
    n = 512
    targets = rng.integers(0, cb.shape[0], size=n)
    a = 0.55
    b = float(np.sqrt(1 - a * a))
    noise = rng.standard_normal((n, D))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    q = a * cb[targets] * np.sqrt(D) / np.sqrt(D) + b * noise   # unit-ish
    q_bip = np.sign(q); q_bip[q_bip == 0] = 1
    top_cont = np.argmax(q @ cb.T, 1)
    top_bip = np.argmax(q_bip @ cb.T, 1)
    keep_cont = float((top_cont == targets).mean())
    keep_bip = float((top_bip == targets).mean())
    return float(np.mean(per)), agree_nosig, keep_cont, keep_bip


# ── Part 2: HRR vs BSC on role-filler unbind under superposition load ───────
def unbind_under_load(algebra, n_pairs_list, trials=20, seed=0):
    """Bind n role-filler pairs, bundle them, unbind one role, measure whether
    the correct filler is recovered (nearest-neighbour in a 100-filler codebook).
    Higher n_pairs = more crosstalk. Returns accuracy per n."""
    rng = np.random.default_rng(seed)
    accs = []
    for n in n_pairs_list:
        hits = 0
        total = 0
        for _ in range(trials):
            if algebra == "HRR":
                roles = hrr_vec(n, D, rng)
                fillers_cb = hrr_vec(100, D, rng)     # codebook of candidate fillers
                pick = rng.integers(0, 100, size=n)
                fillers = fillers_cb[pick]
                bound = np.stack([hrr_bind(roles[i], fillers[i]) for i in range(n)])
                sup = bound.sum(0)                    # superposition (bundle)
                for i in range(n):
                    recovered = hrr_unbind(sup, roles[i])
                    sims = fillers_cb @ recovered
                    hits += int(sims.argmax() == pick[i]); total += 1
            else:  # BSC / MAP — cubby-concepts elementwise bipolar
                roles = generate_random_hypervector(dim=D, num=n, rng=rng)
                fillers_cb = generate_random_hypervector(dim=D, num=100, rng=rng)
                pick = rng.integers(0, 100, size=n)
                fillers = fillers_cb[pick]
                bound = np.stack([bsc_bind(roles[i], fillers[i]) for i in range(n)])
                sup = np.sign(bound.sum(0).astype(np.int32))   # bundled bipolar
                sup[sup == 0] = 1
                for i in range(n):
                    recovered = bsc_bind(roles[i].astype(np.int8), sup.astype(np.int8))
                    sims = fillers_cb.astype(np.float64) @ recovered.astype(np.float64)
                    hits += int(sims.argmax() == pick[i]); total += 1
        accs.append(hits / total)
    return accs


def main():
    print("=" * 76)
    print(f"H-B1 vs H-B2  Binding-head algebra comparison   (D={D} = cubby-lm head)")
    print("=" * 76)

    sign_cos, agree_nosig, keep_cont, keep_bip = head_binarization_loss()
    print("\n[H-B3 corollary] cubby-lm VSABindingHead query is CONTINUOUS.")
    print(f"  cos(continuous query, sign(query))              = {sign_cos:.3f}"
          f"  (theory sqrt(2/pi)=0.798)")
    print(f"  argmax agreement, NO-signal (untrained) regime  = {agree_nosig*100:.1f}%"
          f"  (worst case: argmax over pure crosstalk)")
    print(f"  top-1 = target, cos=0.55 signal, continuous q   = {keep_cont*100:.1f}%")
    print(f"  top-1 = target, cos=0.55 signal, sign(q)        = {keep_bip*100:.1f}%")
    lossless = keep_bip >= keep_cont - 0.02
    print(f"  => in the OPERATING regime binarization is "
          f"{'nearly lossless' if lossless else 'LOSSY'} for readout.")
    print("  => H-B1's bipolar algebra needs a sign() step; H-B2's HRR does not.")

    ns = [1, 3, 5, 8, 12, 20, 40, 80, 160, 320]
    print(f"\n[H-B1 vs H-B2] role-filler unbind from a superposition of n pairs")
    print(f"  (100-filler codebook; accuracy = correct filler is nearest)")
    hrr = unbind_under_load("HRR", ns)
    bsc = unbind_under_load("BSC", ns)
    print("  n pairs " + "".join(f"{n:>7}" for n in ns))
    print("  HRR     " + "".join(f"{a*100:>6.0f}" for a in hrr) + "   (%)")
    print("  BSC/MAP " + "".join(f"{a*100:>6.0f}" for a in bsc) + "   (%)")

    # capacity = largest n with >=90% recall
    def cap(accs):
        ok = [ns[i] for i, a in enumerate(accs) if a >= 0.90]
        return max(ok) if ok else 0
    print(f"\n[verdict]")
    print(f"  HRR  >=90%-recall capacity: n<= {cap(hrr)} pairs")
    print(f"  BSC  >=90%-recall capacity: n<= {cap(bsc)} pairs")
    print(f"  Both are viable; the deciding factor is the head's CONTINUOUS query:")
    print(f"  HRR unbinds it natively; BSC needs sign(), which is"
          f" {'~free' if lossless else 'lossy'} in the signal regime.")
    return {"sign_cos": sign_cos, "keep_cont": keep_cont, "keep_bip": keep_bip,
            "hrr": hrr, "bsc": bsc}


if __name__ == "__main__":
    main()
