"""Generate pod_colab.ipynb: order 0045 on Colab (Nick and Cowork, 2026-11-04: "an affordable platform is the thesis").
The pod runner (`validation/pod/run_pod.py`) on one Colab A100 80 GB, its remote a folder on the mounted Drive, so a
session that ends is a preemption the next session resumes. The notebook is thin: the logic is the runner's. Re-run
after editing:
    python notebooks/_build_pod_colab.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "pod_colab.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0045 on Colab: the bench first, then the control

**Runtime:** an **A100 80 GB** for every session (the same card model each time: gate 0). The run's remote is
`MyDrive/cubbyllm/pod/` on Drive: its state, the inputs (`inputs/`, with `pod_inputs.json`), every checkpoint slot as it
lands, the logs. A session that ends is a preemption: run the cells again in a new session and it resumes.

1. **The bench** (cell 3, about an hour): both arms' tokens per second and MFU, then it stops. Cowork reads
   `pod/runs/bench.json` before anything trains. **Line:** the grown arm's MFU at least 0.75x the control's.
2. **The control** (cell 4, only after Cowork's go): the hours follow from the bench (the most at which every shard
   stays within 2 epochs: about 50 A100-hours, over as many sessions as it takes), then the probes, the readout and the adapters.
**Disconnect the runtime** when a cell ends and nothing else is to run: an idle runtime burns units.
"""),
    code(r"""
!git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM 2>/dev/null || git -C /content/CubbyLLM pull -q
from google.colab import drive; drive.mount('/content/drive')
import json, os, subprocess, sys
REPO, WORK = '/content/CubbyLLM', '/content/pod'
REMOTE = '/content/drive/MyDrive/cubbyllm/pod'
for pkg in ('numpy', 'safetensors', 'tokenizers', 'pytest'):
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=False)
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, round(p.total_memory / 2 ** 30, 1), 'GiB; torch', torch.__version__, 'CUDA', torch.version.cuda)
assert 'A100' in p.name and p.total_memory / 2 ** 30 > 79, 'pick an A100 80 GB runtime (every session the same card model)'
man = os.path.join(REMOTE, 'inputs', 'pod_inputs.json')
assert os.path.exists(man), f'the inputs are not on Drive yet ({man}): see the order, 0045'
m = json.load(open(man))
print(len(m['files']), 'input files,', round(sum(f['bytes'] for f in m['files'].values()) / 1e9, 1), 'GB; commit', m['commit'][:9])


def run(*args, env=None):
    # stream the runner's output as it comes (a captured run shows nothing until it ends)
    p = subprocess.Popen([sys.executable, '-u', *args], cwd=REPO, text=True, bufsize=1, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end='')
    return p.wait()
"""),
    md(r"""
## The commit the inputs were built at

The runner trains from the repo as it is; the manifest names the commit the inputs were built at. A different
commit is printed here, not hidden.
"""),
    code(r"""
head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
print('repo at', head[:9], '| inputs built at', m['commit'][:9], '' if head == m['commit'] else '(differs: the runner is newer)')
"""),
    md(r"""
## 3. The bench, then stop

Preflight (this card, its disk, write access to Drive), the inputs fetched from Drive to the local disk (each file
checked against its sha256), the kernels' tests, then the speed phase for **both** arms. It stops after the bench.
"""),
    code(r"""
ARMS = 'control'          # the arm(s) to train (one arm now: the control); both arms are benched either way
# on Colab the 2-epoch line decides the hours, not a rental's cap: ~50 A100-hours at the continuation's 63k tok/s,
# over as many sessions as it takes (each resumes); 60 only keeps the cap out of the line's way
KNOBS = {'CB_POD_ARMS': ARMS, 'HOURS_PER_ARM': '60'}
code_ = run('validation/pod/run_pod.py', '--work', WORK, '--remote', f'folder:{REMOTE}', '--until', 'speed', env=KNOBS)
print('exit', code_)
b = json.load(open(os.path.join(REMOTE, 'runs', 'bench.json')))
for arm in ('control', 'grown'):
    print(arm, {k: b[arm][k] for k in ('micro', 'tok_s', 'mfu', 'peak_gb')})
print('MFU line:', b['mfu_line'], '| hours per arm:', b['hours_rule']['hours'], f"({b['hours_rule']['by']})")
"""),
    md(r"""
## 4. The control (only after Cowork has read the bench)

Resumable: rerun this cell in each new session until it prints `done`.
"""),
    code(r"""
GO = False          # set True once Cowork has read the bench and said go
assert GO, 'the bench first: Cowork reads pod/runs/bench.json, then GO = True'
print('exit', run('validation/pod/run_pod.py', '--work', WORK, '--remote', f'folder:{REMOTE}', env=KNOBS))
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
