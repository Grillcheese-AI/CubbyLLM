"""run_pod -- the rented card's one command (order 0045): H-E20's two arms, their probes, the readout and the adapters,
in phases, resumable from the remote after a spot preemption.

    HF_TOKEN=... CB_POD_REPO=<private dataset repo> CB_POD_ARMS=control [HOURS_PER_ARM=14] [CB_POD_STOP='<stop command>'] \\
        python validation/pod/run_pod.py --work /workspace/pod
    python validation/pod/run_pod.py --smoke --work <tmp> --remote folder:<tmp remote>     # a tiny shape, CPU, minutes

Phases: preflight, fetch, kernels, speed, control, grown, probes, readout, adapters, done (``pod_phases``,
``pod_train``). The state (``pod_state.json``) lives on the remote: a new pod resumes at the phase and step the last one
reached, pulling the newest uploaded checkpoint slot. A phase already done is not run again. The knobs, each written to the
state once and kept by a resumed pod:
  CB_POD_ARMS     the arms to train: ``control`` (one arm now, Nick 2026-11-04), or ``control,grown`` (the default, as
                  pre-registered). An arm not run is skipped and recorded; the readout marks its gates "waiting on the
                  grown arm"; the adapters train on the control.
  HOURS_PER_ARM   the cap (default 14). The hours themselves are decided after the bench (``pod_phases.hours_from_bench``):
                  the most at which every shard stays within 2 epochs, at the measured tokens per second, at most the cap.
                  ``--hours`` on the command line sets them outright instead (recorded as explicit).
  the read count  decided by the speed phase. Both arms are benched whichever train, and the grown arm's MFU over the
                  control's is recorded against the 0.75 line (``--until speed``: the bench, then stop).
The pre-registered numbers are the readout's (``grow_mem_readout``).

Wired: STANDALONE (the pod's entry point; ``setup.sh`` starts it in tmux).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if p not in sys.path:
        sys.path.insert(0, p)

import pod_phases as P  # noqa: E402
import pod_train as T  # noqa: E402
from pod_remote import fetch, open_remote, read_state, sha256, write_state  # noqa: E402

__wiring__ = "STANDALONE"

PHASES = ("preflight", "fetch", "kernels", "speed", "control", "grown", "probes", "readout", "adapters", "done")
ARMS = ("control", "grown")
MANIFEST = "pod_inputs.json"
SMOKE_SHARDS = ("fineweb_edu", "books", "nemotron_qa", "nemotron_code", "instruct", "agent", "program", "facts_text_far")


class Ctx:
    def __init__(self, a, remote, manifest: dict, state: dict) -> None:
        self.work, self.remote, self.manifest, self.state = a.work, remote, manifest, state
        self.inputs, self.runs, self.logs = (os.path.join(a.work, d) for d in ("inputs", "runs", "logs"))
        for d in (self.inputs, self.runs, self.logs):
            os.makedirs(d, exist_ok=True)
        self.smoke, self.hours, self.hours_cap, self.child = a.smoke, a.hours, a.hours_cap, None
        self.arms = tuple(state.get("knobs", {}).get("arms") or ARMS)
        self.card = json.loads(a.card) if a.card else None
        self.kernel_tests = a.kernel_tests or list(P.KERNEL_TESTS)

    def path(self, key: str) -> str:
        return os.path.join(self.inputs, self.manifest["named"][key])

    def run_dir(self, arm: str) -> str:
        d = os.path.join(self.runs, arm)
        os.makedirs(d, exist_ok=True)
        return d


def manifest_of(folder: str, named: dict, commit: str) -> dict:
    """Every file under `folder`, with its size and sha256; `named`: the roles (corpus, tokenizer, start, ...)."""
    files = {}
    for d, _ds, fs in os.walk(folder):
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, folder).replace(os.sep, "/")
            if rel == MANIFEST:
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    return {"commit": commit, "built": time.strftime("%F %T"), "files": files, "named": named}


def smoke_inputs(work: str, remote) -> dict:
    """The tiny inputs for ``--smoke``: eight random shards, a start checkpoint from two tiny steps, the manifest; all
    uploaded to the remote under inputs/ (once: a second pod fetches them)."""
    import numpy as np
    folder = os.path.join(work, "_build_inputs")
    corpus = os.path.join(folder, "corpus")
    os.makedirs(corpus, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes, holds = {}, {}
    for i, name in enumerate(SMOKE_SHARDS):
        n = 4000 + 100 * i
        rng.integers(1, 40, size=n, dtype=np.uint32).tofile(os.path.join(corpus, f"{name}.u32"))
        sizes[name], holds[name] = n - 200, 200
    json.dump(sizes, open(os.path.join(corpus, "_sizes.json"), "w"))
    json.dump(holds, open(os.path.join(corpus, "_holdout.json"), "w"))
    json.dump({"facts_text_far": 16}, open(os.path.join(corpus, "_align.json"), "w"))
    start_dir = os.path.join(work, "_start_run")
    env = {**os.environ, **P.TINY, **P.TINY_SHAPES["control"], "CB_STEPS": "2", "CB_EVAL": "1000", "CB_STABLE_END": "0",
           "CB_CORPUS": corpus, "CB_DRIVE_DIR": start_dir, "CB_LOCAL_DIR": os.path.join(start_dir, "local"), "CB_NAME": "start"}
    subprocess.run([sys.executable, "validation/train_base.py"], cwd=P.REPO, env=env, check=True, capture_output=True)
    shutil.copyfile(os.path.join(start_dir, "start_final.pt"), os.path.join(folder, "start.pt"))
    man = manifest_of(folder, {"corpus": "corpus", "start": "start.pt"}, "smoke")
    json.dump(man, open(os.path.join(folder, MANIFEST), "w", encoding="utf-8"), indent=1)
    for rel in list(man["files"]) + [MANIFEST]:
        remote.put(os.path.join(folder, rel), f"inputs/{rel}")
    return man


def load_manifest(a, remote) -> dict:
    local = os.path.join(a.work, "inputs", MANIFEST)
    if a.smoke and not remote.exists(f"inputs/{MANIFEST}"):
        return smoke_inputs(a.work, remote)
    fetch(remote, f"inputs/{MANIFEST}", local)
    return json.load(open(local, encoding="utf-8"))


def phase_fn(name: str):
    return {"preflight": P.preflight, "fetch": P.fetch_inputs, "kernels": P.kernels, "speed": P.speed,
            "control": lambda c: T.train_arm(c, "control"), "grown": lambda c: T.train_arm(c, "grown") if "grown" in c.arms
            else skipped(c, "grown"),
            "probes": T.probes, "readout": T.readout, "adapters": T.adapters, "done": done}[name]


def skipped(ctx, phase: str) -> None:
    ctx.state.setdefault("skipped", {})[phase] = f"CB_POD_ARMS={','.join(ctx.arms)}"
    P.say(ctx, f"{phase}: not run (CB_POD_ARMS={','.join(ctx.arms)}); recorded")


def arms_of(text: str | None) -> tuple:
    arms = tuple(a.strip() for a in (text or ",".join(ARMS)).split(",") if a.strip())
    if not arms or "control" not in arms or any(a not in ARMS for a in arms):
        raise SystemExit(f"CB_POD_ARMS={text!r}: the control and, optionally, the grown arm ({','.join(ARMS)})")
    return tuple(a for a in ARMS if a in arms)


def done(ctx) -> None:
    ctx.state["finished"] = time.strftime("%F %T")
    write_state(ctx.remote, ctx.work, ctx.state)
    for f in os.listdir(ctx.logs):
        ctx.remote.put(os.path.join(ctx.logs, f), f"logs/{f}")
    P.say(ctx, "done: every output is on the remote" + (f" ({ctx.remote.name()})"))
    stop = os.environ.get("CB_POD_STOP")
    if stop and not ctx.smoke:
        P.say(ctx, f"stopping the pod: {stop}")
        subprocess.run(stop, shell=True, check=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="local NVMe folder: inputs/, runs/, logs/")
    ap.add_argument("--remote", default=None, help="folder:<dir> | hf:<repo> (default: hf:$CB_POD_REPO)")
    ap.add_argument("--hours", type=float, default=None, help="the hours outright (default: decided after the bench)")
    ap.add_argument("--hours-cap", type=float, default=float(os.environ.get("HOURS_PER_ARM", P.MAX_HOURS)))
    ap.add_argument("--smoke", action="store_true", help="a tiny shape on the CPU, minutes; a folder remote")
    ap.add_argument("--phase", default="", help="run this one phase only")
    ap.add_argument("--until", default="", choices=("",) + PHASES, help="stop once this phase is done (the bench first: speed)")
    ap.add_argument("--card", default="", help="(tests) a JSON card record standing in for the real one")
    ap.add_argument("--kernel-tests", nargs="*", default=None, help="(tests) the kernel test files to run")
    ap.add_argument("--smoke-steps", type=int, default=8, help="(smoke) optimizer steps an arm trains")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    remote = open_remote(a.remote)
    if remote.kind == "hf":
        remote.ensure()
    manifest = load_manifest(a, remote)
    state = read_state(remote, a.work)
    # the knobs are written down once, before training, and a resumed pod keeps them (order 0045)
    knobs = state.setdefault("knobs", {})
    arms = arms_of(os.environ.get("CB_POD_ARMS"))
    if "arms" in knobs and tuple(knobs["arms"]) != arms:
        print(f"CB_POD_ARMS was {','.join(knobs['arms'])} when this run started; keeping it", flush=True)
    knobs.setdefault("arms", list(arms))
    if "hours_cap" in knobs and abs(knobs["hours_cap"] - a.hours_cap) > 1e-9:
        print(f"HOURS_PER_ARM (the cap) was {knobs['hours_cap']} when this run started; keeping it", flush=True)
    a.hours_cap = knobs.setdefault("hours_cap", a.hours_cap)
    if a.hours is not None and "hours" not in knobs:
        knobs.update(hours=a.hours, hours_from="explicit")
    a.hours = knobs.get("hours", a.hours_cap)              # until the bench decides them
    if a.smoke:
        a.smoke_steps = knobs.setdefault("smoke_steps", a.smoke_steps)
        P.TINY["CB_STEPS"] = str(a.smoke_steps)
    ctx = Ctx(a, remote, manifest, state)
    if state.get("arm") and state.get("phase") in ("control", "grown") and state["phase"] not in state["done"]:
        state["preemptions"] = state.get("preemptions", 0) + 1       # a pod picking up an arm another pod left
        P.say(ctx, f"resuming the {state['arm']} arm (preemption {state['preemptions']})")
    P.say(ctx, f"work {a.work}; remote {remote.kind} {remote.name()}; arms {','.join(ctx.arms)}; "
               f"hours per arm {a.hours if 'hours' in knobs else f'from the bench (at most {a.hours_cap})'}; "
               f"{'SMOKE; ' if a.smoke else ''}done so far: {state['done']}")
    for name in ([a.phase] if a.phase else PHASES):
        if name in state["done"] and name not in ("preflight", "fetch") and not a.phase:
            continue                                     # (a new pod runs the preflight and the fetch again: its own
        if name in ("preflight", "fetch") and name in state["done"] and not a.phase:   # card, its own empty disk)
            P.say(ctx, f"--- phase {name} (again, on this pod)")
            phase_fn(name)(ctx)
            continue
        state["phase"] = name
        write_state(remote, a.work, state)
        P.say(ctx, f"--- phase {name}")
        ok = phase_fn(name)(ctx)
        if ok is False:
            P.say(ctx, f"phase {name} did not finish; rerun to resume")
            return 3
        if name not in state["done"]:
            state["done"].append(name)
        write_state(remote, a.work, state)
        if a.until and name == a.until:
            P.say(ctx, f"stopped after {name} (--until); rerun without it to go on")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
