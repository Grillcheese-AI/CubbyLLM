"""prepare_inputs -- the pod run's inputs, built at home and uploaded to a private repo (order 0045, part A).

1. A fresh slice of the full token cache (``make_base_cache.py --frac 0.3 --exclude-frac 0.2 --exclude-dir <token_cache_cont>``:
   blocks neither the first run nor the continuation took), then the decay shards (``make_decay_shards.py``) and the memory's
   shard (``make_fact_shard.py --far``) into it. The epoch table for the most tokens any arm can see (a B200 control at
   ``--hours`` 14, at ``--mfu``): every row at most 2 epochs, else the table says to raise ``--frac``.
2. The manifest ``pod_inputs.json``: every file's size and sha256 (the slice and its index files, the tokenizer, the start
   checkpoint, the continuation's metrics, the adapters' data) and the commit to run.
3. The upload to the private dataset repo ``CB_POD_REPO`` (``huggingface_hub``'s resumable large-folder upload; the token
   from ``HF_TOKEN``), after printing the size and an estimated time on this connection. Rerun to resume.

    python validation/pod/prepare_inputs.py --src <full cache> --dst <token_cache_r3> --exclude-dir <token_cache_cont> \\
        --tokenizer <bbpe128k.json> --start <base450m_cont_stable_end.pt> [--cont-metrics <base450m_cont_metrics.jsonl>] \\
        [--emitter-data <slots.jsonl>] [--talk-data <talk.jsonl>] [--frac 0.3] [--hours 14] [--card b200] [--dry] [--upload]

``--dry``: the plan only -- which blocks the slice would take (none an earlier subset took), the epoch table -- nothing
written, nothing uploaded. No path or account name lives here: they come from the arguments and the environment.

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATION = os.path.dirname(HERE)
REPO = os.path.dirname(VALIDATION)
for p in (HERE, VALIDATION):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_base_cache as mbc  # noqa: E402
from pod_remote import redact, sha256  # noqa: E402

__wiring__ = "STANDALONE"

PEAK_TFLOPS = {"h100": 989.0, "h200": 989.0, "b200": 2250.0}          # dense BF16
FLOP_PER_TOKEN = 2.9e9                                                 # the 450M at S 2048, forward + backward (train_base's count)
MAX_EPOCHS = 2.0
DECAY_SHARDS = ("instruct", "agent", "program", "facts_text_far")


def tokens_for(card: str, hours: float, mfu: float) -> float:
    return PEAK_TFLOPS[card.lower()] * 1e12 * mfu / FLOP_PER_TOKEN * hours * 3600


def plan_slice(src: str, frac: float, exclude_frac: list, exclude_dirs: list, val_tokens: int, keep_small: int) -> dict:
    """What ``make_base_cache`` would take, shard by shard: {name: {"n", "hold", "train", "starts", "block", "used", "cands"}}."""
    import glob
    import numpy as np
    prev = []
    for d in exclude_dirs:
        prev.append(json.load(open(os.path.join(d, "_subset.json"), encoding="utf-8"))["shards"])
    out = {}
    for p in sorted(glob.glob(os.path.join(src, "*.u32"))):
        name = os.path.splitext(os.path.basename(p))[0]
        n = os.path.getsize(p) // 4
        hold = mbc.holdout(n, val_tokens)
        train = n - hold
        if n <= keep_small or frac >= 1.0:
            out[name] = {"n": n, "hold": hold, "train": train, "starts": [0], "block": train, "used": [], "whole": True}
            continue
        k = max(1, int(np.ceil(frac * train / mbc.BLOCK)))
        used = [s for f in exclude_frac for s in mbc.spread(train, f)]
        for pr in prev:
            r = pr.get(name, {"starts": [], "block": mbc.BLOCK})
            used += [s + j for s in r["starts"] for j in range(0, int(r["block"]), mbc.BLOCK)]
        cands = mbc.gap_blocks(train, used) if used else None
        starts = mbc.pick(cands, k) if used else mbc.spread(train, frac)
        out[name] = {"n": n, "hold": hold, "train": train, "starts": starts, "block": mbc.BLOCK, "used": sorted(set(used)),
                     "wanted": k, "unseen": len(cands) if cands is not None else None, "whole": False}
    return out


def overlaps(plan: dict) -> list[str]:
    """The shards whose planned blocks overlap a block an earlier subset took (none, by construction)."""
    import bisect
    bad = []
    for name, r in plan.items():
        if r["whole"]:
            continue
        used = sorted(r["used"])                         # blocks of one size overlap iff their starts are < BLOCK apart
        def near(s):                                     # (a set of every taken token ran out of memory on the real cache)
            i = bisect.bisect_left(used, s)
            return any(abs(used[j] - s) < mbc.BLOCK for j in (i - 1, i) if 0 <= j < len(used))
        if any(near(s) for s in r["starts"]):
            bad.append(name)
    return bad


def epoch_table(sizes: dict, train_tokens: dict, run_tokens: float, decay_json: dict) -> list[dict]:
    """Epochs of each shard's training region in a run of `run_tokens`, under the run's mix (size-proportional x the
    quality multipliers in the stable phase, the decay multipliers plus `decay_json` in the decay)."""
    try:
        from train_base import DECAY, DECAY_FRAC, QUALITY
    except Exception:                                   # noqa: BLE001 -- no torch: the same defaults
        QUALITY, DECAY, DECAY_FRAC = {}, {}, 0.2
    decay = {**DECAY, **decay_json}

    def probs(mult):
        w = {k: sizes[k] * mult.get(k, 1.0) for k in sizes}
        s = sum(w.values()) or 1
        return {k: v / s for k, v in w.items()}
    ps, pd = probs(QUALITY), probs(decay)
    st, dc = run_tokens * (1 - DECAY_FRAC), run_tokens * DECAY_FRAC
    rows = []
    for k in sorted(sizes):
        kept = train_tokens.get(k, sizes[k])
        drawn = st * ps[k] + dc * pd[k]
        rows.append({"shard": k, "train": kept, "drawn": round(drawn), "epochs": round(drawn / max(kept, 1), 3)})
    return rows


def table_text(rows: list[dict], run_tokens: float) -> str:
    out = [f"{'shard':18s} {'subset train':>14s} {'drawn':>14s}  epochs in a {run_tokens / 1e9:.1f}B run (at most {MAX_EPOCHS:.0f})"]
    for r in rows:
        out.append(f"{r['shard']:18s} {r['train']:>14,} {r['drawn']:>14,}  {r['epochs']:6.2f}{'  <-- raise --frac' if r['epochs'] > MAX_EPOCHS else ''}")
    return "\n".join(out)


def build(a, run_tokens: float) -> None:
    py = sys.executable
    cmd = [py, os.path.join(VALIDATION, "make_base_cache.py"), "--src", a.src, "--dst", a.dst, "--frac", str(a.frac),
           "--run-tokens", str(run_tokens), "--val-tokens", str(a.val_tokens), "--keep-small", str(a.keep_small)]
    for f in a.exclude_frac:
        cmd += ["--exclude-frac", str(f)]
    for d in a.exclude_dir:
        cmd += ["--exclude-dir", d]
    subprocess.run(cmd, check=True)
    subprocess.run([py, os.path.join(VALIDATION, "make_decay_shards.py"), "--cache", a.dst, "--tokenizer", a.tokenizer], check=True)
    subprocess.run([py, os.path.join(VALIDATION, "make_fact_shard.py"), "--cache", a.dst, "--tokenizer", a.tokenizer, "--far",
                    "--mode", "text", "--doc", "2048", "--docs", str(a.far_docs)], check=True)


def stage(a) -> str:
    """The upload folder: inputs/ with the cache (hard links where the disk allows), the tokenizer, the start checkpoint,
    the continuation's metrics, the adapters' data; the manifest last."""
    sys.path.insert(0, HERE)
    from run_pod import MANIFEST, manifest_of
    root = os.path.join(a.stage, "inputs")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "corpus"), exist_ok=True)

    def link(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    for f in sorted(os.listdir(a.dst)):
        if f.endswith((".u32", ".json", ".i16", ".u8")) and not f.startswith("_subset"):
            link(os.path.join(a.dst, f), os.path.join(root, "corpus", f))
    named = {"corpus": "corpus"}
    for key, path, rel in (("tokenizer", a.tokenizer, "tokenizer/" + os.path.basename(a.tokenizer)), ("start", a.start, "start.pt"),
                           ("cont_metrics", a.cont_metrics, "cont_metrics.jsonl"), ("emitter_data", a.emitter_data, "emitter_data.jsonl"),
                           ("talk_data", a.talk_data, "talk_data.jsonl")):
        if path:
            link(path, os.path.join(root, rel))
            named[key] = rel
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    man = manifest_of(root, named, commit)
    json.dump(man, open(os.path.join(root, MANIFEST), "w", encoding="utf-8"), indent=1)
    gb = sum(f["bytes"] for f in man["files"].values()) / 1e9
    print(f"staged {len(man['files'])} files, {gb:.2f} GB, commit {commit[:9]} -> {root}")
    return root


def upload(a, root: str) -> None:
    from huggingface_hub import HfApi
    repo = os.environ.get("CB_POD_REPO")
    if not repo or not os.environ.get("HF_TOKEN"):
        raise SystemExit("--upload needs CB_POD_REPO and HF_TOKEN in the environment")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _ds, fs in os.walk(root) for f in fs)
    mbps = a.mbps
    if not mbps:                                        # a 16 MB probe, timed, then removed
        probe = os.path.join(a.stage, "_probe.bin")
        open(probe, "wb").write(os.urandom(16 << 20))
        t0 = time.perf_counter()
        api.upload_file(path_or_fileobj=probe, path_in_repo="_probe.bin", repo_id=repo, repo_type="dataset")
        mbps = 16 * 8 / max(time.perf_counter() - t0, 1e-3)
        api.delete_file("_probe.bin", repo_id=repo, repo_type="dataset")
        os.remove(probe)
    est_h = total * 8 / (mbps * 1e6) / 3600
    print(f"uploading {total / 1e9:.2f} GB to the private repo at ~{mbps:.0f} Mbit/s: about {est_h:.1f} h; resumable, rerun after an interruption")
    try:
        api.upload_large_folder(repo_id=repo, folder_path=a.stage, repo_type="dataset", allow_patterns=["inputs/**"])
    except Exception as e:                              # noqa: BLE001 -- the token never reaches the message
        raise SystemExit(redact(f"upload failed: {e}")) from None
    print("uploaded; the manifest's commit is what the pod checks out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the full token cache")
    ap.add_argument("--dst", required=True, help="the new slice (token_cache_r3)")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--exclude-frac", type=float, action="append", default=[0.2], help="the first run's subset, by its --frac")
    ap.add_argument("--exclude-dir", action="append", default=[], help="the continuation's cache (its _subset.json)")
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--keep-small", type=int, default=200_000_000)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--start", default="", help="base450m_cont_stable_end.pt")
    ap.add_argument("--cont-metrics", default="")
    ap.add_argument("--emitter-data", default="")
    ap.add_argument("--talk-data", default="")
    ap.add_argument("--far-docs", type=int, default=20_000)
    ap.add_argument("--hours", type=float, default=14.0, help="HOURS_PER_ARM the table is sized for")
    ap.add_argument("--card", default="b200", choices=sorted(PEAK_TFLOPS))
    ap.add_argument("--mfu", type=float, default=0.35)
    ap.add_argument("--tokens", type=float, default=0, help="override the run's tokens for the table")
    ap.add_argument("--stage", default="", help="the upload folder (default: <dst>_pod)")
    ap.add_argument("--mbps", type=float, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    a.stage = a.stage or a.dst.rstrip("/\\") + "_pod"
    run_tokens = a.tokens or tokens_for(a.card, a.hours, a.mfu)
    print(f"the most tokens an arm sees: {run_tokens / 1e9:.2f}B ({a.card.upper()} at {a.mfu:.0%} MFU for {a.hours} h)")
    plan = plan_slice(a.src, a.frac, a.exclude_frac, a.exclude_dir, a.val_tokens, a.keep_small)
    bad = overlaps(plan)
    assert not bad, f"planned blocks overlap an earlier subset's: {bad}"
    sizes = {k: r["train"] for k, r in plan.items()}
    kept = {k: len(r["starts"]) * r["block"] for k, r in plan.items()}
    short = [k for k, r in plan.items() if not r["whole"] and len(r["starts"]) < r.get("wanted", 0)]
    print(f"slice: {sum(kept.values()) / 1e9:.2f}B training tokens over {len(plan)} shards" + (f"; out of unseen blocks: {short}" if short else ""))
    print(table_text(epoch_table(sizes, kept, run_tokens, {}), run_tokens))
    print("(the decay shards and facts_text_far are added by the build; their rows appear in the build's own table)")
    if a.dry:
        return
    build(a, run_tokens)
    sizes2 = json.load(open(os.path.join(a.dst, "_sizes.json"), encoding="utf-8"))
    holds = json.load(open(os.path.join(a.dst, "_holdout.json"), encoding="utf-8"))
    kept2 = {k: os.path.getsize(os.path.join(a.dst, f"{k}.u32")) // 4 - holds.get(k, 0) for k in sizes2 if os.path.exists(os.path.join(a.dst, f"{k}.u32"))}
    sys.path.insert(0, HERE)
    from pod_phases import DECAY_JSON
    print(table_text(epoch_table(sizes2, kept2, run_tokens, DECAY_JSON), run_tokens))
    root = stage(a)
    if a.upload:
        upload(a, root)


if __name__ == "__main__":
    main()
