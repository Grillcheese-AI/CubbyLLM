"""pod_phases -- phases 0-3 of the pod run (order 0045): preflight, fetch, kernels, speed; and the environment every
arm trains under. Each phase takes the run's context (``run_pod.Ctx``), writes its output into the run folder and
returns when done; a failure raises SystemExit with its reason, so no card bills for a run that cannot start.

Wired: WIRED (``run_pod.py``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

from pod_remote import fetch, redact, write_state

__wiring__ = "WIRED"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERNEL_TESTS = ("tests/model/backbone/test_mingru_triton.py", "tests/model/backbone/test_window_attn_triton.py")
KERNEL_FLAGS = {"test_mingru_triton.py": "CB_MINGRU_TRITON", "test_window_attn_triton.py": "CB_ATTN_TRITON"}
CARDS = ("H100", "H200", "B200")
MIN_GB, DISK_SPARE_GB, MEM_SHARE = 80, 100, 0.92
MEM_EVERY, MEM_EVERY_SLOW, SLOW_RATIO = 4, 8, 0.5
# round 2's config (notebooks/_build_grow_mem.py), the pre-registered knobs: S 2048, 131,072 tokens a step, the decay
# mix with the three new shards and facts_text_far, samples every 500 steps
ROUND2 = dict(CB_D="1024", CB_HEADS="4", CB_WINDOW="512", CB_ATTN_EVERY="3", CB_S="2048", CB_TOKENS_PER_STEP="131072",
              CB_LR="3e-4", CB_BETA2="0.95", CB_WD="0.1", CB_WARMUP="100", CB_DECAY_FRAC="0.2", CB_SEED="3",
              CB_EVAL="250", CB_GEN="500", CB_STABLE_END="1", CB_CKPT_MIN="15", CB_WALL_H="10000")
DECAY_JSON = {"instruct": 91, "agent": 36, "program": 78, "facts_text_far": 23}
SHAPES = {"control": dict(CB_L="32"), "grown": dict(CB_L="64", CB_FFN_MULT="4", CB_GRAD_CKPT="1")}
TINY = dict(CB_D="16", CB_HEADS="2", CB_WINDOW="4", CB_ATTN_EVERY="2", CB_CTX="6", CB_SLOTS="3", CB_GEN_BASIS="3",
            CB_GEN_RANK="2", CB_VOCAB="40", CB_S="16", CB_MICRO="2", CB_TOKENS_PER_STEP="64", CB_STEPS="8", CB_EVAL="4",
            CB_GEN="0", CB_NO_FLEX="1", CB_CKPT_MIN="0", CB_WARMUP="1", CB_LOG="1", CB_VAL_TOKENS="200", CB_WALL_H="10000",
            CB_STABLE_END="1", CB_DECAY_FRAC="0.25", CB_SEED="3", CB_LR="1e-3")
TINY_SHAPES = {"control": dict(CB_L="4"), "grown": dict(CB_L="8", CB_FFN_MULT="4")}


def say(ctx, text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {redact(text)}"
    print(line, flush=True)
    with open(os.path.join(ctx.logs, "run_pod.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(ctx, cmd: list[str], env: dict, log_name: str, check: bool = True) -> int:
    """A subprocess with its output teed to a log in the run's logs folder (the token redacted)."""
    say(ctx, "run: " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)) + f"  -> logs/{log_name}")
    with open(os.path.join(ctx.logs, log_name), "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        ctx.child = p
        for line in p.stdout:
            log.write(redact(line))
            log.flush()
        p.wait()
        ctx.child = None
    if check and p.returncode != 0:
        raise SystemExit(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed (exit {p.returncode}); see logs/{log_name}")
    return p.returncode


def base_env(ctx, arm: str) -> dict:
    """The arm's training environment: round 2's config (or the tiny smoke's), the arm's shape, the inputs, the run
    folder, the kernels' flags and the bench's choice when they are decided."""
    env = dict(os.environ)
    env.update(TINY if ctx.smoke else ROUND2)
    env.update((TINY_SHAPES if ctx.smoke else SHAPES)[arm])
    inputs = ctx.manifest
    env.update(CB_CORPUS=ctx.path("corpus"), CB_DRIVE_DIR=ctx.run_dir(arm), CB_LOCAL_DIR=os.path.join(ctx.run_dir(arm), "local"),
               CB_NAME=f"grow_mem_{arm}", CB_HOURS=str(ctx.hours), CB_DECAY_JSON=json.dumps(DECAY_JSON))
    if inputs.get("tokenizer"):
        env["CUBBY_SPM"] = ctx.path("tokenizer")
    if inputs.get("mix_json"):
        env["CB_MIX_JSON"] = json.dumps(inputs["mix_json"])
    for flag, verdict in (ctx.state.get("kernels") or {}).items():
        if verdict.startswith("off"):
            env[flag] = "0"
    bench = (ctx.state.get("bench") or {}).get(arm) or {}
    env.update(bench.get("env") or {})
    if bench.get("micro"):
        env["CB_MICRO"] = str(bench["micro"])
    if arm == "grown":
        env["CB_MEM_EVERY"] = str((ctx.state.get("bench") or {}).get("mem_every") or (2 if ctx.smoke else MEM_EVERY))
    return env


# -- phase 0: preflight -------------------------------------------------------------------------------------------
def card_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"name": "cpu", "gb": 0.0, "torch": torch.__version__, "cuda": None}
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "gb": round(p.total_memory / 2 ** 30, 1), "torch": torch.__version__, "cuda": torch.version.cuda}


def _ver(s: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", s or "0")[:2]) or (0,)


def preflight(ctx) -> None:
    card = ctx.card or card_info()
    say(ctx, f"card: {card}")
    if not ctx.smoke or ctx.card:                    # (the smoke on a bare CPU skips the card's checks; a card record given, no)
        if not any(c in card["name"] for c in CARDS) or card["gb"] < MIN_GB:
            raise SystemExit(f"not a card this run uses ({CARDS}, >= {MIN_GB} GB): {card}")
        if "B200" in card["name"] and (_ver(card["torch"]) < (2, 7) or _ver(card["cuda"] or "") < (12, 8)):
            raise SystemExit(f"a B200 needs torch >= 2.7 with CUDA >= 12.8; this image has torch {card['torch']} / CUDA {card['cuda']}."
                             " Install the cu128 build (docs/POD.md); nothing is downgraded silently")
    if ctx.state.get("card") and ctx.state["card"] != card["name"]:
        raise SystemExit(f"this run trained on a {ctx.state['card']}; a {card['name']} compares nothing. "
                         "Rent the same card model to resume, or start a new run folder")
    need_gb = 3 * sum(f["bytes"] for f in ctx.manifest["files"].values()) / 2 ** 30 + (0 if ctx.smoke else DISK_SPARE_GB)
    free_gb = shutil.disk_usage(ctx.work).free / 2 ** 30
    if free_gb < need_gb:
        raise SystemExit(f"free disk {free_gb:.0f} GB < {need_gb:.0f} GB needed (3 x the inputs + {DISK_SPARE_GB} GB)")
    if ctx.remote.kind == "hf" and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (the private repo needs it)")
    probe = os.path.join(ctx.logs, "_preflight.txt")
    open(probe, "w").write(time.strftime("%F %T"))
    ctx.remote.put(probe, f"runs/_preflight_{int(time.time())}.txt")   # write access, or it raises here
    ctx.state["card"] = card["name"]
    ctx.state["versions"] = {"torch": card["torch"], "cuda": card["cuda"], "python": sys.version.split()[0]}
    ctx.state["free_gb"] = round(free_gb, 1)
    write_state(ctx.remote, ctx.work, ctx.state)


# -- phase 1: fetch ---------------------------------------------------------------------------------------------------
def fetch_inputs(ctx) -> None:
    for rel, f in ctx.manifest["files"].items():
        local = os.path.join(ctx.inputs, rel)
        fetch(ctx.remote, f"inputs/{rel}", local, f["sha256"])
        say(ctx, f"fetched inputs/{rel} ({f['bytes'] / 1e6:.1f} MB)")
    arm = ctx.state.get("arm")
    last = (ctx.state.get("last_step_uploaded") or {}).get(arm or "")
    if arm and last:                                     # a new pod: the newest slot of the arm under way
        for rel in last["files"]:
            if not os.path.exists(os.path.join(ctx.run_dir(arm), os.path.basename(rel))):
                fetch(ctx.remote, rel, os.path.join(ctx.run_dir(arm), os.path.basename(rel)))
                say(ctx, f"fetched {rel} (resume at step {last['step']})")


# -- phase 2: kernels -------------------------------------------------------------------------------------------------
def kernels(ctx) -> None:
    verdicts = {}
    for t in ctx.kernel_tests:
        flag = KERNEL_FLAGS.get(os.path.basename(t), "CB_" + os.path.basename(t).upper())
        code = run(ctx, [sys.executable, "-m", "pytest", t, "-q", "-x", "-p", "no:cacheprovider"], dict(os.environ),
                   f"kernels_{os.path.basename(t)}.log", check=False)
        verdicts[flag] = "on" if code == 0 else "skipped (no card)" if code == 5 else f"off (tests failed, exit {code})"
        say(ctx, f"{t}: {verdicts[flag]}")
    ctx.state["kernels"] = verdicts
    json.dump(verdicts, open(os.path.join(ctx.runs, "kernels.json"), "w", encoding="utf-8"), indent=1)
    ctx.remote.put(os.path.join(ctx.runs, "kernels.json"), "runs/kernels.json")
    write_state(ctx.remote, ctx.work, ctx.state)


# -- phase 3: speed ---------------------------------------------------------------------------------------------------
_ROW = re.compile(r"micro\s+(\d+):\s+([\d.]+) ms/micro\s+([\d,]+) tok/s\s+MFU\s+([\d.]+|nan)%\s+peak\s+([\d.]+|nan) GB")


def bench_arm(ctx, arm: str, env: dict) -> dict:
    """``train_base.py CB_MODE=bench`` at the arm's shape: the fastest micro that fits under 92% of the memory."""
    env = dict(env, CB_MODE="bench", CB_BENCH_MICRO=("1,2" if ctx.smoke else "4,8,16,32"))
    run(ctx, [sys.executable, "-u", "validation/train_base.py"], env, f"bench_{arm}.log")
    rows = []
    for line in open(os.path.join(ctx.logs, f"bench_{arm}.log"), encoding="utf-8"):
        m = _ROW.search(line)
        if m:
            rows.append({"micro": int(m.group(1)), "tok_s": float(m.group(3).replace(",", "")),
                         "mfu": 0.0 if m.group(4) == "nan" else float(m.group(4)) / 100,   # (no peak known: the CPU)
                         "peak_gb": float("nan") if m.group(5) == "nan" else float(m.group(5))})
    if not rows:
        raise SystemExit(f"the bench measured nothing for the {arm} arm; see logs/bench_{arm}.log")
    gb = (ctx.card or card_info())["gb"]
    fit = [r for r in rows if not gb or r["peak_gb"] != r["peak_gb"] or r["peak_gb"] <= MEM_SHARE * gb]
    best = max(fit or rows, key=lambda r: r["tok_s"])
    return {"micro": best["micro"], "tok_s": best["tok_s"], "mfu": best["mfu"], "peak_gb": best["peak_gb"], "rows": rows}


def speed(ctx) -> None:
    if ctx.state.get("bench"):
        say(ctx, "bench already decided; not measured again")
        return
    bench = {"card": ctx.state.get("card")}
    for arm in ("control", "grown"):
        env = base_env(ctx, arm)
        if arm == "grown":
            env["CB_MEM_EVERY"] = str(2 if ctx.smoke else MEM_EVERY)
        chosen_env = {}
        if (ctx.card or card_info())["name"] != "cpu" and not ctx.smoke:
            out = os.path.join(ctx.runs, f"speed_{arm}.json")
            run(ctx, [sys.executable, "-u", "validation/speed_check.py"], dict(env, CB_SPEED_OUT=out), f"speed_{arm}.log", check=False)
            if os.path.exists(out):
                chosen_env = json.load(open(out, encoding="utf-8")).get("env") or {}
        env.update(chosen_env)
        bench[arm] = {**bench_arm(ctx, arm, env), "env": chosen_env}
        say(ctx, f"{arm}: micro {bench[arm]['micro']}, {bench[arm]['tok_s']:,.0f} tok/s, MFU {100 * bench[arm]['mfu']:.1f}%")
    slow = bench["grown"]["mfu"] < SLOW_RATIO * bench["control"]["mfu"] if bench["control"]["mfu"] else False
    bench["mem_every"] = 2 if ctx.smoke else (MEM_EVERY_SLOW if slow else MEM_EVERY)
    bench["reads"] = int(TINY_SHAPES["grown"]["CB_L"] if ctx.smoke else SHAPES["grown"]["CB_L"]) // bench["mem_every"]
    bench["decided"] = time.strftime("%F %T")
    say(ctx, f"the memory read every {bench['mem_every']}th layer ({bench['reads']} reads)" + (" -- the grown arm ran under half the control's MFU" if slow else ""))
    json.dump(bench, open(os.path.join(ctx.runs, "bench.json"), "w", encoding="utf-8"), indent=1)
    ctx.remote.put(os.path.join(ctx.runs, "bench.json"), "runs/bench.json")
    ctx.state["bench"] = bench
    write_state(ctx.remote, ctx.work, ctx.state)


__all__ = ["DECAY_JSON", "KERNEL_TESTS", "ROUND2", "SHAPES", "TINY", "TINY_SHAPES", "base_env", "bench_arm", "card_info",
           "fetch_inputs", "kernels", "preflight", "run", "say", "speed"]
