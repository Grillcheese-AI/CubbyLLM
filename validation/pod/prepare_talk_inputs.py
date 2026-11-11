"""prepare_talk_inputs -- what `talk_pod.py` fetches on a rented card, staged and uploaded from home (order 0010).

Wired: STANDALONE.

    python validation/pod/prepare_talk_inputs.py --base <base450m_cont_final.pt> --tokenizer <bbpe128k.json> \
        --arm control=<talk_v2_2.jsonl>:_v2_2_cont --arm voice=<talk_v3a.jsonl>:_v3a [--upload] [--remote hf:<repo>]
    python validation/pod/prepare_talk_inputs.py --smoke --tokenizer <bbpe128k.json> --arm voice=<sample.jsonl>:_smoke \
        --upload --remote folder:<tmp>/remote

Without `--upload` it prints the plan and writes nothing outside the stage folder. It stages:
- the code the trainer needs, as `git archive` of the commit (`cubbyllm/`, the three trainer files, `validation/pod`),
  so the pod runs committed code and nothing else. The paths must be committed first.
- a manifest (`talk_inputs.json`): each file's size and sha256, the arms (data, tag, adapter name), the
  notebook's settings (`pod_train.TALK`) and the commit.
`--upload` puts them on the remote (`pod_remote.py`: a private dataset repo named by `CB_POD_REPO`, the token
from `HF_TOKEN` in the environment). A file already there under the same name is not sent again unless `--force`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from pod_remote import open_remote, sha256  # noqa: E402

__wiring__ = "STANDALONE"

PREFIX = "talk/inputs"
CODE = ["cubbyllm", "validation/train_base.py", "validation/train_talk_torch.py", "validation/talk_data.py",
        "validation/pod"]
TALK = dict(epochs="1", batch="32", lr="2e-4", rank="16", alpha="32", max_len="1024")   # pod_train.TALK, the notebook's


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="", help="the base checkpoint (.pt) every arm trains on")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=<data.jsonl>:<tag>; repeat per arm, in run order")
    ap.add_argument("--stage", default=os.path.join(REPO, "standin", "data", "out", "talk_pod_inputs"))
    ap.add_argument("--remote", default=None, help="hf:<repo> or folder:<dir>; default CB_POD_REPO")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="send a file again even when the remote has its name")
    ap.add_argument("--smoke", action="store_true", help="no base: the pod's smoke builds a tiny random one")
    a = ap.parse_args()
    if not a.smoke and not a.base:
        raise SystemExit("--base is needed (or --smoke)")
    dirty = git("status", "--porcelain", "--", *CODE)
    if dirty:
        raise SystemExit("commit these first: the pod runs the committed code\n" + dirty)
    commit = git("rev-parse", "HEAD")
    os.makedirs(a.stage, exist_ok=True)
    bundle = os.path.join(a.stage, "code.tar.gz")
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", bundle, commit, *CODE], cwd=REPO, check=True)

    local = {"code": bundle, "tokenizer": a.tokenizer}
    if not a.smoke:
        local["base"] = a.base
    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        data, tag = rest.rsplit(":", 1)
        key = os.path.basename(data)
        local[key] = data
        arms[name] = {"data": key, "tag": tag, "out": "talk" + tag}
    files = {}
    for key, path in local.items():
        t0 = time.time()
        files[key] = {"rel": f"{PREFIX}/{os.path.basename(path)}", "bytes": os.path.getsize(path), "sha256": sha256(path)}
        print(f"  {key:16s} {files[key]['bytes'] / 1e6:10,.1f} MB  sha256 {files[key]['sha256'][:16]}  ({time.time() - t0:.0f}s)")
    manifest = {"commit": commit, "made": time.strftime("%F %T"), "files": files, "arms": arms, "settings": TALK,
                "smoke": a.smoke}
    man_path = os.path.join(a.stage, "talk_inputs.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    total = sum(f["bytes"] for f in files.values())
    print(f"commit {commit[:12]} | {len(files)} files, {total / 1e9:.2f} GB | arms: {', '.join(arms)}")
    if not a.upload:
        print("plan only: nothing sent (add --upload)")
        return 0
    remote = open_remote(a.remote)
    if hasattr(remote, "ensure"):
        remote.ensure()                                   # a PRIVATE dataset repo, created if it is not there
    for key, path in local.items():
        rel = files[key]["rel"]
        if remote.exists(rel) and not a.force and key != "code":
            print(f"  {rel}: already on the remote, not sent again")
            continue
        t0 = time.time()
        remote.put(path, rel)
        print(f"  sent {rel} in {time.time() - t0:.0f}s")
    remote.put(man_path, f"{PREFIX}/talk_inputs.json")
    print(f"on the remote: {remote.name()} (talk/inputs/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
