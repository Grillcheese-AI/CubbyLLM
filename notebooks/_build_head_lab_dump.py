"""Generate head_lab_dump.ipynb: order 0068, fresh held-out states for route 5 by distance alone (Nick's Colab session).
The 13 shards' held-out tails, 4 windows of 1,024 at half the earlier spacing, every window that would share a token
with the ones read before dropped (39 windows, 39,936 positions). One forward each through ``base450m_cont``, the full
head's top tokens kept: nothing is trained, no weight changes. The read is at home, on the CPU
(``validation/head_lab_distance.py``), once. Re-run after editing:
    python notebooks/_build_head_lab_dump.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "head_lab_dump.ipynb"
BRANCH = "cowork/0068"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# CubbyLLM 0068: fresh held-out states for route 5 by distance alone

One dump, about 5 minutes on any CUDA card (an A100 or an L4 is fine):
- the 13 shards' held-out tails, 4 windows of 1,024 tokens at **half** the earlier spacing;
- any window that would share a token with the held-out windows read before is dropped (the last of each shard), so
  39 windows, 39,936 positions no route has seen;
- per position: the state the head reads, the full head's top-16, its log-sum-exp and the target's logit.

Nothing is trained and no weight changes. The file lands in `MyDrive/cubbyllm/head_shortlist/heldout_b.npz`; Cowork
records its SHA-256 in the order, then reads it once at home on the CPU. **Run all, then disconnect the runtime.**
"""),
    code(r"""
import os, shutil, subprocess, sys, hashlib, json
from google.colab import drive; drive.mount('/content/drive')
BRANCH = '""" + BRANCH + r"""'
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q -b {BRANCH} --single-branch https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git fetch -q origin {BRANCH} && git checkout -q {BRANCH} && git pull -q --ff-only
!cd /content/CubbyLLM && git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv
DRIVE = '/content/drive/MyDrive/cubbyllm'
OUTD = f'{DRIVE}/head_shortlist'; os.makedirs(OUTD, exist_ok=True)
assert not os.path.exists(f'{OUTD}/heldout_b.npz'), 'heldout_b.npz is already on Drive: it is dumped once'
CK450 = '/content/base450m_cont_final.pt'
if not os.path.exists(CK450):                      # Drive-FUSE reads are slow: one copy to the local disk
    shutil.copy(f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt', CK450)
CACHE = f'{DRIVE}/token_cache_cont'
"""),
    code(r"""
# --- the dump (about 5 min) ---
p = subprocess.Popen([sys.executable, '-u', 'validation/head_share_torch.py', 'dump', '--ckpt', CK450, '--cache', CACHE,
                      '--split', 'heldout', '--windows', '4', '--phase', '0.5', '--avoid-phase', '0',
                      '--name', 'heldout_b', '--out', '/content/head_shortlist_b'],
                     cwd='/content/CubbyLLM', text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout:
    print(line, end='')
assert p.wait() == 0
"""),
    code(r"""
# --- check, keep on Drive, print the SHA-256 ---
import numpy as np
sys.path.insert(0, '/content/CubbyLLM/validation')
from head_dump import SHARDS, starts
src = '/content/head_shortlist_b/heldout_b.npz'
d = np.load(src)
hold = json.load(open(f'{CACHE}/_holdout.json'))
gaps = []
for si, s in zip(d['win_shard'], d['win_start']):
    name = SHARDS[int(si)]
    total = os.path.getsize(f'{CACHE}/{name}.u32') // 4
    gaps.append(min(abs(int(s) - o) for o in starts(total, int(hold[name]), 'heldout', 4, 1024, 0.0)))
print('windows', len(d['win_start']), 'positions', len(d['target']), 'closest gap to an earlier window', min(gaps))
assert min(gaps) > 1024 and len(d['target']) == 1024 * len(d['win_start'])
sha = hashlib.sha256(open(src, 'rb').read()).hexdigest()
shutil.copy(src, f'{OUTD}/heldout_b.npz')
import datetime
today = datetime.date.today().isoformat()                # only this run's log, not 0050's committed one
shutil.copy(f'/content/CubbyLLM/validation/logs/head_share_dump_cuda_{today}.json', f'{OUTD}/heldout_b_dump_log.json')
print('kept on Drive: heldout_b.npz, sha256', sha)
print('Done: disconnect the runtime.')
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
