"""Order 0045, part B -- the pod runner end to end on a tiny shape (``run_pod.py --smoke``, CPU): every phase runs and
writes its outputs; killed mid-arm (the process killed, not signalled) and started again in an empty folder it fetches
from the remote and resumes at the last uploaded step, and a phase already done is not run again; a failing kernel test
turns its flag off for the run; another card model refuses to resume an arm. Minutes on the CPU."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "validation" / "pod" / "run_pod.py"
pytestmark = pytest.mark.skipif(os.environ.get("CB_POD_TESTS") == "0", reason="CB_POD_TESTS=0")


def runner(work, remote, *extra, env=None, **kw):
    cmd = [sys.executable, "-u", str(RUNNER), "--smoke", "--work", str(work), "--remote", f"folder:{remote}", *extra]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env or dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **kw)


def state_of(remote):
    p = pathlib.Path(remote) / "pod_state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        return {}


def test_every_phase_runs_and_writes_its_outputs_then_a_kill_mid_arm_resumes_from_the_remote(tmp_path):
    remote = tmp_path / "remote"
    work1 = tmp_path / "work1"
    p = runner(work1, remote, "--smoke-steps", "60")
    uploaded = None
    for _ in range(1200):                                     # kill once the control arm has uploaded a slot
        time.sleep(0.25)
        st = state_of(remote)
        last = (st.get("last_step_uploaded") or {}).get("control")
        if st.get("phase") == "control" and last and last["step"] >= 3:
            uploaded = last["step"]
            p.kill()
            break
        if p.poll() is not None:
            break
    out1 = p.communicate()[0]
    assert uploaded is not None, out1[-3000:]
    assert "control" not in state_of(remote)["done"]
    assert not (work1 / "runs" / "control" / "grow_mem_control_final.pt").exists()
    work2 = tmp_path / "work2"                               # an empty folder: a new pod
    p2 = runner(work2, remote)
    out2 = p2.communicate(timeout=900)[0]
    assert p2.returncode == 0, out2[-4000:]
    log = (work2 / "logs" / "train_control.log").read_text(encoding="utf-8", errors="replace")
    assert "resumed" in log and f"@ step {uploaded}" in log or any(f"@ step {s}" in log for s in range(uploaded, uploaded + 3)), log[-1500:]
    st = state_of(remote)
    assert st["done"] == ["preflight", "fetch", "kernels", "speed", "control", "grown", "probes", "readout", "adapters", "done"]
    assert st["preemptions"] == 1 and st["card"] == "cpu" and st["bench"]["mem_every"] == 2
    assert not list((work2 / "logs").glob("kernels_*.log"))            # a phase done before is not run again
    assert not list((work2 / "logs").glob("bench_*.log"))
    for f in ("bench.json", "kernels.json", "probes.json", "readout.json", "readout.md", "adapters.json"):
        assert (remote / "runs" / f).exists(), f
    for arm in ("control", "grown"):
        assert (remote / "runs" / arm / f"grow_mem_{arm}_final.pt").exists()
        assert (remote / "runs" / arm / f"grow_mem_{arm}_metrics.jsonl").exists()
    r = json.loads((remote / "runs" / "readout.json").read_text(encoding="utf-8"))
    assert r["checks"]["0. equal compute: the same tokens a step, the same card, hours within 0.5 h"] is True
    assert (remote / "runs" / "grown" / "grow_step0.json").exists() and json.loads((remote / "runs" / "grown" / "grow_step0.json").read_text())["exact"]
    steps = [json.loads(l)["step"] for l in (remote / "runs" / "control" / "grow_mem_control_metrics.jsonl").read_text().splitlines() if '"tokens"' in l]
    assert max(steps) == 60
    # the runner's third start finds everything done: only this pod's own preflight and fetch run again
    p3 = runner(work2, remote)
    out3 = p3.communicate(timeout=300)[0]
    assert p3.returncode == 0 and "(again, on this pod)" in out3
    ran = [line.split("--- phase ", 1)[1].split()[0] for line in out3.splitlines() if "--- phase " in line]
    assert ran == ["preflight", "fetch"], ran


def test_a_failing_kernel_test_turns_its_flag_off_and_another_card_refuses_to_resume(tmp_path):
    remote, work = tmp_path / "remote", tmp_path / "work"
    bad = tmp_path / "test_mingru_triton.py"
    bad.write_text("def test_kernel():\n    assert False, 'the kernel disagrees with the reference on this card'\n", encoding="utf-8")
    p = runner(work, remote, "--phase", "preflight", "--card", json.dumps({"name": "B200", "gb": 180.0, "torch": "2.8.0", "cuda": "12.8"}))
    out = p.communicate(timeout=600)[0]
    assert p.returncode == 0, out[-2000:]
    p = runner(work, remote, "--phase", "kernels", "--kernel-tests", str(bad))
    out = p.communicate(timeout=600)[0]
    assert p.returncode == 0, out[-2000:]
    k = json.loads((work / "runs" / "kernels.json").read_text(encoding="utf-8"))
    assert k["CB_MINGRU_TRITON"].startswith("off")
    sys.path.insert(0, str(ROOT / "validation" / "pod"))
    import pod_phases
    from run_pod import Ctx
    import argparse
    st = state_of(remote)
    ctx = Ctx(argparse.Namespace(work=str(work), smoke=True, hours=1.0, hours_cap=1.0, card="", kernel_tests=None), None,
              json.loads((work / "inputs" / "pod_inputs.json").read_text()), st)
    assert pod_phases.base_env(ctx, "control")["CB_MINGRU_TRITON"] == "0"
    p = runner(tmp_path / "w4", tmp_path / "r4", "--phase", "preflight", "--card", json.dumps({"name": "NVIDIA A100-SXM4-80GB", "gb": 79.2, "torch": "2.8.0", "cuda": "12.6"}))
    out = p.communicate(timeout=600)[0]
    assert p.returncode == 0, out[-2000:]                                  # Colab's A100 80 GB is a card this run uses
    # the state says B200; a pod with an H200 refuses before it fetches anything
    p = runner(work, remote, "--phase", "preflight", "--card", json.dumps({"name": "NVIDIA H200", "gb": 141.0, "torch": "2.8.0", "cuda": "12.8"}))
    out = p.communicate(timeout=600)[0]
    assert p.returncode != 0 and "compares nothing" in out
    # a B200 on an old torch is refused with what is missing, not downgraded
    p = runner(tmp_path / "w3", tmp_path / "r3", "--phase", "preflight", "--card", json.dumps({"name": "B200", "gb": 180.0, "torch": "2.5.1", "cuda": "12.4"}))
    out = p.communicate(timeout=600)[0]
    assert p.returncode != 0 and "torch >= 2.7" in out


def test_setup_sh_parses_and_never_puts_the_token_on_a_command_line():
    import re
    import shutil
    text = (ROOT / "validation" / "pod" / "setup.sh").read_text(encoding="utf-8")
    assert not re.search(r"HF_TOKEN=\$", text), "the token must reach run_pod.py through the environment"
    assert "github.com/" not in text                                       # no account name in a committed file
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(ROOT / "validation" / "pod" / "setup.sh")]).returncode == 0


def test_a_token_never_reaches_a_log_line(monkeypatch):
    sys.path.insert(0, str(ROOT / "validation" / "pod"))
    from pod_remote import redact
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 30)
    assert "hf_" not in redact(f"upload failed: 401 for token {'hf_' + 'x' * 30} and hf_{'A1' * 15}")


def test_one_arm_now_trains_the_control_only_and_its_readout_waits_on_the_grown_arm(tmp_path):
    """Order 0045, 2026-11-04 (Nick: one arm now, the control): CB_POD_ARMS=control runs every phase, skips the grown
    arm and records why, decides the hours after the bench, marks the grown arm's gates as waiting, and trains the
    adapters on the control."""
    remote, work = tmp_path / "remote", tmp_path / "work"
    p = runner(work, remote, "--until", "speed", env=dict(os.environ, CB_POD_ARMS="control"))     # the bench first, then stop
    out = p.communicate(timeout=900)[0]
    assert p.returncode == 0 and "stopped after speed" in out, out[-4000:]
    st = state_of(remote)
    assert st["done"] == ["preflight", "fetch", "kernels", "speed"] and "control" in st["bench"] and "grown" in st["bench"]
    assert st["bench"]["mfu_line"]["line"] == 0.75 and "ratio" in st["bench"]["mfu_line"]
    p = runner(work, remote, env=dict(os.environ, CB_POD_ARMS="control"))
    out = p.communicate(timeout=900)[0]
    assert p.returncode == 0, out[-4000:]
    st = state_of(remote)
    assert st["done"][-1] == "done" and st["knobs"]["arms"] == ["control"] and "grown" in st["skipped"]
    rule = st["bench"]["hours_rule"]
    assert rule["limit_shard"] and rule["hours_at_2_epochs"] > 0 and st["knobs"]["hours_from"] == "bench"
    assert (remote / "runs" / "control" / "grow_mem_control_final.pt").exists()
    assert not (remote / "runs" / "grown").exists()
    md = (remote / "runs" / "readout.md").read_text(encoding="utf-8")
    assert "waiting on the grown arm" in md and "1. growth" in md
    assert json.loads((remote / "runs" / "adapters.json").read_text(encoding="utf-8"))["winner"] == "control"
    p = runner(work, remote, env=dict(os.environ, CB_POD_ARMS="control,grown"))      # a resumed pod keeps the first knob
    out = p.communicate(timeout=300)[0]
    assert p.returncode == 0 and "keeping it" in out and state_of(remote)["knobs"]["arms"] == ["control"]


def test_the_hours_are_the_most_within_two_epochs_and_at_most_the_cap(tmp_path):
    import numpy as np
    sys.path.insert(0, str(ROOT / "validation" / "pod"))
    sys.path.insert(0, str(ROOT / "validation"))
    import pod_phases
    from run_pod import arms_of
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, n in (("fineweb_edu", 9000), ("books", 3000), ("instruct", 500)):
        np.zeros(n, dtype=np.uint32).tofile(corpus / f"{name}.u32")
    (corpus / "_sizes.json").write_text(json.dumps({"fineweb_edu": 8900, "books": 2900, "instruct": 450}))
    (corpus / "_holdout.json").write_text(json.dumps({"fineweb_edu": 100, "books": 100, "instruct": 50}))
    r = pod_phases.hours_from_bench(str(corpus), tok_s=1.0, cap=100.0)          # 1 token a second: hours = tokens / 3600
    assert r["by"].endswith("at 2 epochs") and abs(max(r["epochs"].values()) - 2.0) < 1e-3
    assert r["epochs"][r["limit_shard"]] == max(r["epochs"].values())
    capped = pod_phases.hours_from_bench(str(corpus), tok_s=1.0, cap=r["hours_at_2_epochs"] / 2)
    assert capped["by"] == "the cap" and capped["hours"] == round(r["hours_at_2_epochs"] / 2, 4)
    assert max(capped["epochs"].values()) < 2.0
    assert arms_of("control") == ("control",) and arms_of(None) == ("control", "grown") and arms_of("grown,control") == ("control", "grown")
    with pytest.raises(SystemExit):
        arms_of("grown")
