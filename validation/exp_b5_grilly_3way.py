"""Group B step-5 — 3-way prototype over grilly's ACTUAL algebra families.

Decision recorded 2026-07-23 (user): before standardizing the binding head on
one algebra, prototype all three families that ship in the confirmed production
substrate `grilly.experimental.vsa` (H-B5):

  binary   BinaryOps        flat D=10240 bipolar, bind = elementwise product
  hrr      HolographicOps   flat D=10240 continuous, bind = circular conv (FFT)
  block    BlockCodeOps     (k=80, l=128) NVSA sparse block codes, per-block
                            circular conv — cubemind's world-arena shape

Four parts, each mapped to a real binding-head duty:
  [1] parity + roundtrip sanity + op latency (incl. similarity_batch vs a
      V=32k codebook — the head's per-token readout op)
  [2] vocab-scale readout accuracy vs signal fraction f (query = f*target +
      (1-f)*family-appropriate noise), V=32k
  [3] role-filler unbind from a bundled superposition of n pairs (100-filler
      codebook), n up to 320 — same protocol as exp_b_algebra, now on grilly's
      own implementations + block codes
  [4] trainability probe (torch): learn a projection into each representation
      to classify 256 inputs into 512 codewords through the family's canonical
      readout — tests BlockCodeOps' central claim ("prevents ... the
      zero-gradient problem of bipolar sign()"). Includes hard-sign(STE) and
      tanh-relaxed bipolar rows so the claimed failure mode is actually
      exercised, not just asserted.

grilly modules are imported directly from source files (importlib) so the
C++/Vulkan backend init is never touched. Standalone; nothing wired.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRILLY = r"C:\Users\grill\Documents\GitHub\grilly"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = load_module("grilly_vsa_ops", GRILLY + r"\experimental\vsa\ops.py")
block_ops = load_module("grilly_vsa_block_ops",
                        GRILLY + r"\experimental\vsa\block_ops.py")
BinaryOps, HolographicOps = ops.BinaryOps, ops.HolographicOps
BlockCodeOps = block_ops.BlockCodeOps

# cubby-concepts reference, for cross-implementation parity
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-concepts")
from vsa.core import bind as cc_bind  # noqa: E402

D = 10240
K, L = 80, 128            # cubemind/world-arena block shape; K*L = D
V = 32_000
N_FILLERS = 100


def rng_(seed=0):
    return np.random.default_rng(seed)


# ── family adapters: uniform interface over the three algebras ──────────────
class Fam:
    """random(n) -> stacked vectors; bind/unbind; bundle(list); sim_batch."""


class Binary(Fam):
    name = "binary"
    @staticmethod
    def random(n, rng):
        return (rng.integers(0, 2, size=(n, D)) * 2 - 1).astype(np.float32)
    bind = staticmethod(BinaryOps.bind)
    unbind = staticmethod(BinaryOps.unbind)
    @staticmethod
    def bundle(vs):
        return BinaryOps.bundle(list(vs), normalize=True)
    @staticmethod
    def sim_batch(q, cb):
        return BinaryOps.similarity_batch(q, cb)
    @staticmethod
    def noisy(target, f, rng):
        """bit-flip noise: keep each bit with prob (1+f)/2 -> E[cos]=f."""
        flip = rng.random(D) > (1 + f) / 2
        out = target.copy()
        out[flip] *= -1
        return out


class HRR(Fam):
    name = "hrr"
    @staticmethod
    def random(n, rng):
        return (rng.standard_normal((n, D)) / np.sqrt(D)).astype(np.float32)
    bind = staticmethod(HolographicOps.convolve)
    unbind = staticmethod(HolographicOps.correlate)
    @staticmethod
    def bundle(vs):
        return HolographicOps.bundle(list(vs), normalize=True)
    @staticmethod
    def sim_batch(q, cb):
        return HolographicOps.similarity_batch(q, cb)
    @staticmethod
    def noisy(target, f, rng):
        n = rng.standard_normal(D).astype(np.float32)
        n /= np.linalg.norm(n)
        t = target / (np.linalg.norm(target) + 1e-12)
        return (f * t + np.sqrt(1 - f * f) * n).astype(np.float32)


class Block(Fam):
    name = "block"
    @staticmethod
    def random(n, rng):
        out = np.zeros((n, K, L), dtype=np.float32)
        idx = rng.integers(0, L, size=(n, K))
        np.put_along_axis(out, idx[:, :, None], 1.0, axis=2)
        return out
    bind = staticmethod(BlockCodeOps.bind)
    unbind = staticmethod(BlockCodeOps.unbind)
    @staticmethod
    def bundle(vs):
        return BlockCodeOps.bundle(list(vs), normalize=True)
    @staticmethod
    def sim_batch(q, cb):
        return BlockCodeOps.similarity_batch(q, cb)
    @staticmethod
    def noisy(target, f, rng):
        noise = rng.random((K, L)).astype(np.float32)
        noise /= noise.sum(axis=1, keepdims=True)
        return (f * target + (1 - f) * noise).astype(np.float32)


FAMS = [Binary, HRR, Block]


def part1_parity_and_latency():
    print("\n[1] parity, roundtrip, latency")
    rng = rng_(1)
    # parity: grilly BinaryOps.bind == cubby-concepts bind (elementwise product)
    a8 = (rng.integers(0, 2, size=D) * 2 - 1).astype(np.int8)
    b8 = (rng.integers(0, 2, size=D) * 2 - 1).astype(np.int8)
    assert np.array_equal(BinaryOps.bind(a8, b8), cc_bind(a8, b8).astype(np.float32)), \
        "grilly BinaryOps.bind != cubby-concepts bind"
    print("  parity: grilly BinaryOps.bind == cubby-concepts vsa.core.bind  OK")

    print(f"  {'family':>7} {'roundtrip sim':>14} {'bind us':>9}"
          f" {'unbind us':>10} {'readout ms (V=32k)':>19}")
    for FamC in FAMS:
        rng = rng_(2)
        vs = FamC.random(2, rng)
        a, b = vs[0], vs[1]
        comp = FamC.bind(a, b)
        rec = FamC.unbind(comp, b)
        s = float(np.sum(rec * a) /
                  (np.linalg.norm(rec) * np.linalg.norm(a) + 1e-12))
        t0 = time.perf_counter()
        for _ in range(200):
            FamC.bind(a, b)
        t_bind = (time.perf_counter() - t0) / 200 * 1e6
        t0 = time.perf_counter()
        for _ in range(200):
            FamC.unbind(comp, b)
        t_unbind = (time.perf_counter() - t0) / 200 * 1e6
        cb = FamC.random(V, rng)
        t0 = time.perf_counter()
        for _ in range(20):
            FamC.sim_batch(a, cb)
        t_read = (time.perf_counter() - t0) / 20 * 1e3
        print(f"  {FamC.name:>7} {s:>14.4f} {t_bind:>9.0f} {t_unbind:>10.0f}"
              f" {t_read:>19.2f}")
        del cb


def part2_readout():
    print("\n[2] vocab-scale readout: top-1 accuracy vs signal fraction f"
          f"  (V={V:,})")
    fs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
    print("  family  " + "".join(f"{f:>8}" for f in fs))
    for FamC in FAMS:
        rng = rng_(3)
        cb = FamC.random(V, rng)
        row = f"  {FamC.name:<8}"
        for f in fs:
            hits = 0
            for _ in range(50):
                t = int(rng.integers(0, V))
                q = FamC.noisy(cb[t], f, rng)
                sims = FamC.sim_batch(q, cb)
                hits += int(int(np.argmax(sims)) == t)
            row += f"{hits/50*100:>7.0f}%"
        print(row)
        del cb


def part3_superposition():
    print(f"\n[3] role-filler unbind from bundled superposition"
          f"  ({N_FILLERS}-filler codebook)")
    ns = [1, 5, 20, 80, 160, 320]
    print("  family  " + "".join(f"{n:>8}" for n in ns))
    for FamC in FAMS:
        rng = rng_(4)
        fillers = FamC.random(N_FILLERS, rng)
        row = f"  {FamC.name:<8}"
        for n in ns:
            hits = 0; total = 0
            for _ in range(10):
                roles = FamC.random(n, rng)
                pick = rng.integers(0, N_FILLERS, size=n)
                bound = [FamC.bind(roles[i], fillers[pick[i]])
                         for i in range(n)]
                sup = FamC.bundle(bound) if n > 1 else bound[0]
                for i in range(n):
                    rec = FamC.unbind(sup, roles[i])
                    sims = FamC.sim_batch(rec, fillers)
                    hits += int(int(np.argmax(sims)) == pick[i]); total += 1
            row += f"{hits/total*100:>7.0f}%"
        print(row)


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    @staticmethod
    def backward(ctx, g):
        return g            # straight-through


def part4_trainability():
    print("\n[4] trainability probe: learn 32-dim inputs -> 512 codewords"
          " through each canonical readout (2k steps, Adam 3e-3)")
    torch.manual_seed(0)
    rng = rng_(5)
    n_cls, d_in, steps = 512, 32, 2000
    X = torch.randn(4096, d_in)
    y = torch.randint(0, n_cls, (4096,))

    # codebooks per representation
    cb_bip = torch.from_numpy(Binary.random(n_cls, rng))            # (C, D)
    cb_hrr = torch.from_numpy(HRR.random(n_cls, rng))               # (C, D)
    cb_blk = torch.from_numpy(Block.random(n_cls, rng).reshape(n_cls, -1))

    def run(tag, head_fn, cb, scale):
        proj = nn.Linear(d_in, D)
        opt = torch.optim.Adam(proj.parameters(), lr=3e-3)
        first_loss = last_loss = None
        for s in range(steps):
            i = torch.randint(0, len(X), (256,))
            q = head_fn(proj(X[i]))
            logits = q @ cb.t() * scale
            loss = F.cross_entropy(logits, y[i])
            if s == 0:
                first_loss = float(loss)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = float(loss)
        with torch.no_grad():
            q = head_fn(proj(X))
            acc = float(((q @ cb.t()).argmax(1) == y).float().mean())
        print(f"  {tag:<22} loss {first_loss:5.2f} -> {last_loss:5.2f}"
              f"   train acc {acc*100:5.1f}%")
        return acc

    run("binary hard sign(STE)", lambda z: _SignSTE.apply(z), cb_bip,
        1 / np.sqrt(D))
    run("binary tanh (relaxed)", torch.tanh, cb_bip, 1 / np.sqrt(D))
    run("hrr linear", lambda z: F.normalize(z, dim=-1), cb_hrr, np.sqrt(D) / 8)
    run("block softmax/blocks",
        lambda z: F.softmax(z.view(-1, K, L), dim=-1).view(-1, D),
        cb_blk, 40.0 / K)   # BlockCodeOps.cosine_to_pmf default temperature


def main():
    print("=" * 78)
    print("Group B step 5 — 3-way prototype on grilly's production algebras"
          f"   (D={D} / k={K},l={L}, V={V:,})")
    print("=" * 78)
    part1_parity_and_latency()
    part2_readout()
    part3_superposition()
    part4_trainability()
    print("\n[read the numbers, then decide] duties matrix: readout robustness"
          " (part 2), compositional load (part 3), gradient-friendliness"
          " (part 4), raw speed (part 1).")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n(wall {time.time()-t0:.1f}s)")
