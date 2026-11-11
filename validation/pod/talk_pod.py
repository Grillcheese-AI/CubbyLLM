"""talk_pod -- the talk adapter trained on a rented card (order 0010): fetch, train each arm, upload, stop.

Wired: STANDALONE (run on the pod; `prepare_talk_inputs.py` stages what it fetches).

    python validation/pod/talk_pod.py --work /workspace/talk [--remote hf:<repo> | folder:<dir>] [--arms control,voice]
    python validation/pod/talk_pod.py --smoke --work <tmp>/work --remote folder:<tmp>/remote     # CPU, minutes

The remote is order 0045's (`pod_remote.py`): a private dataset repo named by `CB_POD_REPO`, the token from
`HF_TOKEN` in the environment, redacted from everything printed. It holds `talk/inputs/talk_inputs.json` (each
file's sha256, the arms, the commit) and the files. An arm is `train_talk_torch.py` with the notebook's settings
(`pod_train.TALK`); its adapter, its json and its logs go to `talk/runs/<arm's name>/`. An arm whose adapter is
already on the remote is not run again, so a second pod picks up where the first stopped. No arm starts past
`--max-minutes`. When the last upload is done `CB_POD_STOP` runs, so the card never bills idle.
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

from pod_remote import fetch, open_remote, redact  # noqa: E402

__wiring__ = "STANDALONE"

PREFIX = "talk"
MANIFEST = f"{PREFIX}/inputs/talk_inputs.json"
ADAPTER = ("talk_lora.safetensors", "talk_lora.json")


class Run:
    def __init__(self, work: str, remote, smoke: bool) -> None:
        self.work, self.remote, self.smoke, self.t0 = work, remote, smoke, time.time()
        self.inputs, self.logs, self.out = (os.path.join(work, d) for d in ("inputs", "logs", "runs"))
        for d in (self.inputs, self.logs, self.out):
            os.makedirs(d, exist_ok=True)
        self.log = os.path.join(self.logs, "talk_pod.log")
        self.state: dict = {"arms": {}, "started": time.strftime("%F %T")}

    def say(self, msg: str) -> None:
        line = time.strftime("%F %T ") + redact(msg)
        print(line, flush=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def minutes(self) -> float:
        return (time.time() - self.t0) / 60

    def sh(self, cmd: list[str], log_name: str) -> int:
        """A command with its output shown and kept, the token redacted from both."""
        with open(os.path.join(self.logs, log_name), "a", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                line = redact(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
            return p.wait()

    def put_state(self) -> None:
        self.state["updated"], self.state["minutes"] = time.strftime("%F %T"), round(self.minutes(), 1)
        path = os.path.join(self.out, "talk_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)
        self.remote.put(path, f"{PREFIX}/runs/talk_state.json")
        self.remote.put(self.log, f"{PREFIX}/runs/talk_pod.log")


def preflight(r: Run) -> None:
    """The card and a write to the remote, before anything is fetched: a card is not billed for a run that cannot
    save its result."""
    import torch
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    r.state["card"], r.state["torch"] = card, torch.__version__
    r.say(f"card: {card} | torch {torch.__version__} | remote: {r.remote.name()}")
    if not r.smoke and not torch.cuda.is_available():
        raise SystemExit("no GPU on this machine: nothing to train on")
    r.put_state()


def held_losses(metrics: str) -> list[float]:
    out = []
    if os.path.exists(metrics):
        for line in open(metrics, encoding="utf-8"):
            row = json.loads(line)
            if "held_loss" in row:
                out.append(row["held_loss"])
    return out


def train_arm(r: Run, name: str, arm: dict, files: dict, settings: dict) -> bool:
    out_dir = os.path.join(r.out, arm["out"])
    cmd = [sys.executable, "-u", "validation/train_talk_torch.py", "--tokenizer", files["tokenizer"],
           "--data", files[arm["data"]], "--out", out_dir, "--tag", arm["tag"]]
    for k, val in settings.items():
        cmd += [f"--{k.replace('_', '-')}", str(val)]
    cmd += ["--smoke"] if r.smoke else ["--ckpt", files["base"]]
    t0 = time.time()
    rc = r.sh(cmd, f"train_{name}.log")
    rel = f"{PREFIX}/runs/{arm['out']}/"
    logs = [os.path.join(r.logs, f"train_{name}.log")] + \
        [os.path.join(REPO, "validation", "logs", f"train_talk_torch{arm['tag']}{ext}") for ext in (".jsonl", ".log")]
    for f in logs:
        if os.path.exists(f):
            r.remote.put(f, rel + os.path.basename(f))
    ok = rc == 0 and all(os.path.exists(os.path.join(out_dir, f)) for f in ADAPTER)
    if ok:
        meta = json.load(open(os.path.join(out_dir, ADAPTER[1]), encoding="utf-8"))
        held = held_losses(logs[1])
        for f in ADAPTER[::-1]:                          # the json first: the safetensors on the remote means "done"
            r.remote.put(os.path.join(out_dir, f), rel + f)
        r.state["arms"][name] = {"out": arm["out"], "steps": meta.get("step"), "data": meta.get("data"),
                                 "base_sha256": meta.get("ckpt_sha256"), "minutes": round((time.time() - t0) / 60, 1),
                                 "held_loss_first": held[0] if held else None, "held_loss_last": held[-1] if held else None}
        r.say(f"{name}: done, {meta.get('step')} steps in {(time.time() - t0) / 60:.1f} min, uploaded to {rel}")
    else:
        r.state["arms"][name] = {"failed": f"exit code {rc}", "minutes": round((time.time() - t0) / 60, 1)}
        r.say(f"{name}: FAILED (exit code {rc}); its log is on the remote")
    r.put_state()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--remote", default=None, help="hf:<repo> or folder:<dir>; default CB_POD_REPO")
    ap.add_argument("--arms", default="", help="comma-separated; default every arm of the manifest, in its order")
    ap.add_argument("--max-minutes", type=float, default=150.0, help="no arm starts past this many minutes")
    ap.add_argument("--smoke", action="store_true", help="a tiny random base on the CPU: the code path, not a result")
    a = ap.parse_args()
    r = Run(a.work, open_remote(a.remote), a.smoke)
    failed = []
    try:
        preflight(r)
        man = json.load(open(r.remote.get(MANIFEST, os.path.join(r.inputs, "talk_inputs.json")), encoding="utf-8"))
        r.state["commit"] = man.get("commit")
        arms = [x for x in a.arms.split(",") if x] or list(man["arms"])
        need = {"tokenizer"} | {man["arms"][x]["data"] for x in arms} | (set() if a.smoke else {"base"})
        files = {}
        for key in sorted(need):
            f = man["files"][key]
            t0 = time.time()
            files[key] = fetch(r.remote, f["rel"], os.path.join(r.inputs, os.path.basename(f["rel"])), f.get("sha256"))
            r.say(f"fetched {key}: {f.get('bytes', 0) / 1e6:,.0f} MB in {time.time() - t0:.0f}s, sha256 checked")
        for name in arms:
            arm = man["arms"][name]
            if r.remote.exists(f"{PREFIX}/runs/{arm['out']}/{ADAPTER[0]}"):
                r.say(f"{name}: its adapter is already on the remote, not run again")
                continue
            if r.minutes() > a.max_minutes:
                r.say(f"{name}: not started, {r.minutes():.0f} min already used (--max-minutes {a.max_minutes:.0f})")
                failed.append(name)
                continue
            if not train_arm(r, name, arm, files, man["settings"]):
                failed.append(name)
        r.state["finished"] = time.strftime("%F %T")
        r.say("TALK_POD_DONE" + (f" with failures: {', '.join(failed)}" if failed else ": every arm trained and uploaded"))
    except BaseException as e:                              # noqa: BLE001 -- said, saved, and the card still stops
        r.say(f"TALK_POD_STOPPED: {type(e).__name__}: {e}")
        failed.append("run")
    finally:
        try:
            r.put_state()
        except Exception as e:                              # noqa: BLE001
            print(redact(f"the last upload failed: {e}"), flush=True)
        stop = os.environ.get("CB_POD_STOP", "")
        if stop and not a.smoke:
            r.say("stopping the card (CB_POD_STOP)")
            subprocess.run(stop, shell=True, check=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
