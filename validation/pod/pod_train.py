"""pod_train -- phases 4-9 of the pod run (order 0045): the two arms trained with every checkpoint uploaded as it
lands, the probes, the readout, the adapters on the winner, done.

Spot preemption: ``train_base`` checkpoints every ``CB_CKPT_MIN`` minutes into the run folder (two slots, a sidecar
json each); a watcher thread sees a slot's sidecar turn valid and uploads the slot, its sidecar, the metrics and the
logs, then records the arm and step in the state file. A new pod fetches that slot (``pod_phases.fetch_inputs``) and
``train_base`` resumes from it: at most ``CB_CKPT_MIN`` of training plus the fetch is lost. The slot on the remote is
the truth; no SIGTERM grace is counted on.

Wired: WIRED (``run_pod.py``).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time

from pod_phases import REPO, base_env, run, say
from pod_remote import write_state

__wiring__ = "WIRED"

PROBE_LENGTHS = "1024,2048,4096"
EMITTER = dict(epochs="2", batch="16", lr="2e-4", rank="16", alpha="32", max_len="1024", gen_per_task="64", gen_max_new="640")
TALK = dict(epochs="1", batch="32", lr="2e-4", rank="16", alpha="32", max_len="1024")


class Watcher(threading.Thread):
    """Uploads a new checkpoint slot as soon as its sidecar says it is whole; the metrics and the logs with it."""

    def __init__(self, ctx, arm: str, every_s: float) -> None:
        super().__init__(daemon=True)
        self.ctx, self.arm, self.every_s, self.stop = ctx, arm, every_s, threading.Event()
        self.seen = (ctx.state.get("last_step_uploaded") or {}).get(arm, {}).get("step", -1)

    def slots(self) -> list[tuple[int, str]]:
        out = []
        for side in glob.glob(os.path.join(self.ctx.run_dir(self.arm), "*_slot?.pt.json")):
            try:
                meta = json.load(open(side, encoding="utf-8"))
                pt = side[:-5]
                if meta.get("step", -1) >= 0 and os.path.exists(pt) and os.path.getsize(pt) == meta.get("bytes"):
                    out.append((meta["step"], pt))
            except (OSError, ValueError):
                pass
        return sorted(out)

    def once(self) -> None:
        for step, pt in self.slots():
            if step <= self.seen:
                continue
            rel = f"runs/{self.arm}/"
            files = [pt, pt + ".json"]
            for extra in glob.glob(os.path.join(self.ctx.run_dir(self.arm), "*_metrics.jsonl")):
                files.append(extra)
            for f in files:
                self.ctx.remote.put(f, rel + os.path.basename(f))
            for lg in glob.glob(os.path.join(self.ctx.logs, f"train_{self.arm}*.log")):
                self.ctx.remote.put(lg, "logs/" + os.path.basename(lg))
            self.seen = step
            self.ctx.state.setdefault("last_step_uploaded", {})[self.arm] = {
                "step": step, "files": [rel + os.path.basename(f) for f in files], "at": time.strftime("%F %T")}
            write_state(self.ctx.remote, self.ctx.work, self.ctx.state)
            say(self.ctx, f"uploaded {self.arm} step {step}")

    def run(self) -> None:
        while not self.stop.wait(self.every_s):
            try:
                self.once()
            except Exception as e:                   # noqa: BLE001 -- the training goes on; the next pass retries
                say(self.ctx, f"upload failed, retrying next pass: {e}")
        try:
            self.once()
        except Exception as e:                       # noqa: BLE001
            say(self.ctx, f"final upload pass failed: {e}")


def grow_step0(ctx, env: dict) -> str:
    """The grown model at step 0 from the start checkpoint, exact or no training."""
    out = os.path.join(ctx.run_dir("grown"), "grow_step0.pt")
    tag = "pod_smoke" if ctx.smoke else f"pod_m{env['CB_MEM_EVERY']}"
    logs = ctx.logs if ctx.smoke else os.path.join(REPO, "validation", "logs")   # a smoke's report stays in its work dir
    rep = os.path.join(logs, f"grow_step0_{tag}.json")
    if os.path.exists(out) and os.path.exists(rep) and json.load(open(rep, encoding="utf-8")).get("exact"):
        return out
    cmd = [sys.executable, "-u", "validation/grow_base.py", "--ckpt", ctx.path("start"), "--cache", ctx.path("corpus"),
           "--depth", "2", "--ffn", "2", "--mem-every", env["CB_MEM_EVERY"], "--save", out, "--tag", tag, "--log-dir", logs,
           "--windows", "1", "--seq", "64" if ctx.smoke else "1024"]
    if ctx.smoke:
        cmd += ["--sources", "fineweb_edu,books"]
    else:
        cmd += ["--sources", "arxiv,books,fineweb_edu,nemotron_code"]
    genv = dict(env)
    genv.pop("CB_L", None); genv.pop("CB_FFN_MULT", None); genv.pop("CB_MEM_EVERY", None)   # grow_base reads the start's meta
    run(ctx, cmd, genv, "grow_step0.log")
    r = json.load(open(rep, encoding="utf-8"))
    if not r.get("exact"):
        raise SystemExit("the grown model at step 0 is not exact (grow_step0 json); no training on an inexact growth")
    ctx.remote.put(rep, "runs/grown/grow_step0.json")
    return out


def train_arm(ctx, arm: str) -> bool:
    """One arm to its final; False when the trainer stopped before it (the wall, a kill): rerun to resume."""
    env = base_env(ctx, arm)
    if arm == "grown":
        env["CB_INIT_FROM"] = grow_step0(ctx, env)
    else:
        env["CB_INIT_FROM"] = ctx.path("start")
    final = os.path.join(ctx.run_dir(arm), f"grow_mem_{arm}_final.pt")
    if os.path.exists(final):
        return True
    ctx.state["arm"] = arm
    ctx.state.setdefault("arms_started", {}).setdefault(arm, time.strftime("%F %T"))
    write_state(ctx.remote, ctx.work, ctx.state)
    w = Watcher(ctx, arm, 0.5 if ctx.smoke else 15.0)
    w.start()
    try:
        run(ctx, [sys.executable, "-u", "validation/train_base.py"], env, f"train_{arm}.log", check=False)
    finally:
        w.stop.set()
        w.join(timeout=600)
    if not os.path.exists(final):
        say(ctx, f"{arm}: the trainer stopped before its final; rerun resumes from the newest slot")
        return False
    for f in (final, os.path.join(ctx.run_dir(arm), f"grow_mem_{arm}_stable_end.pt")):
        if os.path.exists(f):
            ctx.remote.put(f, f"runs/{arm}/{os.path.basename(f)}")
    for f in glob.glob(os.path.join(ctx.run_dir(arm), "*_metrics.jsonl")):
        ctx.remote.put(f, f"runs/{arm}/{os.path.basename(f)}")
    rows = [json.loads(l) for l in open(glob.glob(os.path.join(ctx.run_dir(arm), "*_metrics.jsonl"))[0], encoding="utf-8") if l.strip()]
    hours = max((r.get("hours", 0) for r in rows), default=0)
    ctx.state.setdefault("arms_done", {})[arm] = {"hours": round(hours, 3), "steps": max((r.get("step", 0) for r in rows), default=0),
                                                 "tok_s": [r["tok_s"] for r in rows if "tok_s" in r][-10:]}
    write_state(ctx.remote, ctx.work, ctx.state)
    return True


def probes(ctx) -> None:
    out = {"skipped": "smoke: no tokenizer, no real shards"} if ctx.smoke else {}
    if not ctx.smoke:
        env = dict(os.environ, CB_CORPUS=ctx.path("corpus"), CUBBY_SPM=ctx.path("tokenizer"), CB_SHARDS="facts_text_far", CB_DOCS="400", CB_BATCH="8")
        for name, ck in (("grown", os.path.join(ctx.run_dir("grown"), "grow_mem_grown_final.pt")),
                         ("control", os.path.join(ctx.run_dir("control"), "grow_mem_control_final.pt")), ("base", ctx.path("start"))):
            o = os.path.join(ctx.runs, f"hypertoken_recall_{name}.json")
            run(ctx, [sys.executable, "-u", "validation/exp_hypertoken_recall.py"], dict(env, CB_CKPT=ck, CB_OUT=o), f"probe_recall_{name}.log")
            out[f"recall_{name}"] = o
            if name != "base":
                n = os.path.join(ctx.runs, f"needle_{name}.json")
                run(ctx, [sys.executable, "-u", "validation/exp_needle_forward.py"], dict(env, CB_CKPT=ck, CB_OUT=n, CB_LENGTHS=PROBE_LENGTHS),
                    f"probe_needle_{name}.log")
                out[f"needle_{name}"] = n
    json.dump(out, open(os.path.join(ctx.runs, "probes.json"), "w", encoding="utf-8"), indent=1)
    for f in glob.glob(os.path.join(ctx.runs, "*.json")):
        ctx.remote.put(f, "runs/" + os.path.basename(f))


def readout(ctx) -> dict:
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import grow_mem_readout as gr
    load = lambda p: json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None  # noqa: E731
    pr = load(os.path.join(ctx.runs, "probes.json")) or {}
    probes_ = {"grown": load(pr.get("recall_grown")), "control": load(pr.get("recall_control")), "base": load(pr.get("recall_base")),
               "needle_grown": load(pr.get("needle_grown")), "needle_control": load(pr.get("needle_control"))}
    m = lambda a: os.path.join(ctx.run_dir(a), f"grow_mem_{a}_metrics.jsonl")  # noqa: E731
    cards = {a: ctx.state.get("card") for a in ("grown", "control")}
    r = gr.gates(gr.rows_of(m("grown")), gr.rows_of(m("control")), gr.rows_of(ctx.path("cont_metrics")) if ctx.manifest.get("cont_metrics") else [],
                 int(base_env(ctx, "control")["CB_S"]), probes_, cards)
    r["arms"] = ctx.state.get("arms_done", {})
    r["preemptions"] = ctx.state.get("preemptions", 0)
    r["kernels"] = ctx.state.get("kernels")
    r["bench"] = {k: v for k, v in (ctx.state.get("bench") or {}).items() if k != "rows"}
    json.dump(r, open(os.path.join(ctx.runs, "readout.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(ctx.runs, "readout.md"), "w", encoding="utf-8").write(gr.markdown(r))
    for f in ("readout.json", "readout.md"):
        ctx.remote.put(os.path.join(ctx.runs, f), "runs/" + f)
    say(ctx, "readout:\n" + gr.markdown(r))
    return r


def adapters(ctx) -> None:
    r = json.load(open(os.path.join(ctx.runs, "readout.json"), encoding="utf-8"))
    winner = "grown" if r["checks"].get("1. growth: grown mix <= control mix + 0.02") else "control"
    out_dir = os.path.join(ctx.runs, "adapters", winner)
    os.makedirs(out_dir, exist_ok=True)
    rep = {"winner": winner}
    if ctx.smoke or not (ctx.manifest.get("emitter_data") and ctx.manifest.get("talk_data")):
        rep["skipped"] = "smoke" if ctx.smoke else "no adapter data in the manifest"
    else:
        env = base_env(ctx, winner)                       # the trainers read the winner's shape from the environment
        final = os.path.join(ctx.run_dir(winner), f"grow_mem_{winner}_final.pt")
        for script, data, cfg, tag in (("train_emitter_torch.py", "emitter_data", EMITTER, "_v12e_w_pod"),
                                       ("train_talk_torch.py", "talk_data", TALK, "_talk_v2_2_pod")):
            cmd = [sys.executable, "-u", f"validation/{script}", "--ckpt", final, "--tokenizer", ctx.path("tokenizer"),
                   "--data", ctx.path(data), "--out", os.path.join(out_dir, tag.strip("_")), "--tag", tag]
            for k, v in cfg.items():
                cmd += [f"--{k.replace('_', '-')}", v]
            run(ctx, cmd, env, f"adapter_{script}.log")
            rep[script] = os.path.join(out_dir, tag.strip("_"))
    json.dump(rep, open(os.path.join(ctx.runs, "adapters.json"), "w", encoding="utf-8"), indent=1)
    ctx.remote.put(os.path.join(ctx.runs, "adapters.json"), "runs/adapters.json")
    for d, _ds, fs in os.walk(out_dir):
        for f in fs:
            p = os.path.join(d, f)
            ctx.remote.put(p, "runs/adapters/" + os.path.relpath(p, os.path.join(ctx.runs, "adapters")).replace(os.sep, "/"))


__all__ = ["Watcher", "adapters", "grow_step0", "probes", "readout", "train_arm"]
