"""r9b_grilly2 -- order 0046, R9b's arm 5: block codes on grilly2's Vulkan path (``grilly.vsa``: bind and unbind by
``g2-blockcode-bind``, similarity_batch as one matmul). grilly2 only, never grilly v1. Run in grilly2's own environment,
in a GPU slot Cowork opens, short calls only:

    cd <grilly2> && uv run python <CubbyLLM>/validation/retrieval_bench/r9b_grilly2.py --out <json>

- **Equal results first:** bind, unbind, bundle and similarity_batch against R9's arithmetic (``BlockCodes``' numpy
  path, the same formulas: per-block FFT convolution / correlation, the per-block normalized sum, (1/k) sum(a * b)) on a
  seeded sample: one-hot exact, float within 1e-5. A mismatch stops the timing.
- **Timed:** bind, bundle of 16 and similarity at batches of 1, 1,000 and 100,000 (where the batch fits the card), and
  the index job (one query, and 1,000, against N stored keys, top-10) for N the card holds; each twice: with upload and
  read-back (numpy in, numpy out), and resident (the tensors already on the card; completion forced by reading one
  element back, which is in the time). The median over ``--reps`` runs. grilly2's commit and the device recorded.

Wired: STANDALONE (order 0046's bench; runs under grilly2's environment, not CubbyLLM's).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import numpy as np

__wiring__ = "STANDALONE"

L = 128


def ref_bind(a, b, inv=False):
    fa, fb = np.fft.fft(a, axis=-1), np.fft.fft(b, axis=-1)
    return np.real(np.fft.ifft(fa * (np.conj(fb) if inv else fb), axis=-1)).astype(np.float32)


def ref_bundle(vs):
    s = vs.sum(axis=0)
    t = s.sum(axis=-1, keepdims=True)
    return (s / np.where(t == 0, 1.0, t)).astype(np.float32)


def onehot(rng, n, k):
    v = np.zeros((n, k, L), dtype=np.float32)
    v[np.arange(n)[:, None], np.arange(k)[None, :], rng.integers(0, L, size=(n, k))] = 1.0
    return v


def check(g, vsa, k) -> dict:
    rng = np.random.default_rng(7)
    cb = onehot(rng, 32, k)
    frame = ref_bundle(np.stack([ref_bind(cb[i], cb[i + 1]) for i in range(0, 8, 2)]))
    T = g.from_numpy
    out = {}
    out["bind_onehot_exact"] = bool(np.array_equal(vsa.bind(T(cb[0]), T(cb[1])).numpy().round(6), ref_bind(cb[0], cb[1]).round(6)))
    out["bind_frame_1e-5"] = bool(np.allclose(vsa.bind(T(frame), T(cb[3])).numpy(), ref_bind(frame, cb[3]), atol=1e-5))
    out["unbind_frame_1e-5"] = bool(np.allclose(vsa.unbind(T(frame), T(cb[3])).numpy(), ref_bind(frame, cb[3], inv=True), atol=1e-5))
    out["bundle_1e-5"] = bool(np.allclose(vsa.bundle(T(cb[:6])).numpy(), ref_bundle(cb[:6]), atol=1e-5))
    ref_sim = (cb.reshape(32, -1) @ frame.reshape(-1)) / k
    out["similarity_batch_1e-5"] = bool(np.allclose(vsa.similarity_batch(T(frame), T(cb)).numpy(), ref_sim, atol=1e-5))
    return out


def med(f, reps):
    xs = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        f()
        xs.append(time.perf_counter_ns() - t)
    return statistics.median(xs)


def main() -> None:
    import grilly as g
    from grilly import vsa
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--max-bytes", type=float, default=2e9, help="the largest tensor put on the card")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {"what": "order 0046 R9b arm 5: grilly2's block codes (Vulkan)",
           "grilly2_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "device": str(getattr(g, "get_device_name", lambda: "unknown")()), "checks": {}, "calls": [], "index_job": []}
    rng = np.random.default_rng(0)
    for k in (80, 512):
        rep["checks"][str(k)] = c = check(g, vsa, k)
        if not all(c.values()):
            print("not equal: no timing", c)
            continue
        per = k * L * 4
        for batch in (1, 1000, 100_000):
            if 3 * batch * per > a.max_bytes:
                rep["calls"].append({"k": k, "batch": batch, "skipped": "does not fit under --max-bytes"})
                continue
            x, y = onehot(rng, batch, k), onehot(rng, batch, k)
            vs = onehot(rng, 16, k)
            gx, gy, gv = g.from_numpy(x), g.from_numpy(y), g.from_numpy(vs)
            one = lambda t: t.reshape(-1)[0].item()               # completion forced by one element read back
            rep["calls"].append({
                "k": k, "batch": batch, "unit": "ns per call",
                "bind_upload": med(lambda: vsa.bind(g.from_numpy(x), g.from_numpy(y)).numpy(), a.reps),
                "bind_resident": med(lambda: one(vsa.bind(gx, gy)), a.reps),
                "bundle16_resident": med(lambda: one(vsa.bundle(gv)), a.reps),
                "similarity_resident": med(lambda: one(vsa.similarity_batch(gx[0] if batch > 1 else gx.reshape(k, L), gv)), a.reps)})
            del gx, gy
            print(k, batch, flush=True)
        for n in (10_000, 100_000, 1_000_000):
            if n * per > a.max_bytes:
                rep["index_job"].append({"k": k, "n": n, "skipped": "the codebook does not fit under --max-bytes"})
                continue
            cb = onehot(rng, n, k)
            gcb = g.from_numpy(cb)
            for batch in (1, 1000):
                qs = cb[rng.choice(n, batch, replace=False)]
                gq = g.from_numpy(qs)
                flat = gcb.reshape(n, k * L)
                score = lambda: (gq.reshape(batch, k * L) @ flat.transpose(0, 1)) / k
                t_res = med(lambda: score().numpy(), max(3, a.reps // 4)) / batch
                t_up = med(lambda: ((g.from_numpy(qs).reshape(batch, k * L) @ flat.transpose(0, 1)) / k).numpy(), 3) / batch
                rep["index_job"].append({"k": k, "n": n, "batch": batch, "ns_per_query_resident_scores_back": t_res,
                                         "ns_per_query_with_upload": t_up, "bytes_held": int(n * per)})
            del gcb
            print(k, n, flush=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
