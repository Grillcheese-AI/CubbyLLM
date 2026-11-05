"""Generate head_shortlist.ipynb: order 0050, stage 1 on Nick's A100 session -- the 128k output head's share of a
decode step and of a training step (the 450M and the grown 1.36B), and the held-in states the shortlist index is
built from (the home card shut the PC down on a long dump, 2026-11-04). The logic is validation/head_share_torch.py.
Nothing is trained; no weight changes. Re-run after editing:
    python notebooks/_build_head_shortlist.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "head_shortlist.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# CubbyLLM 0050: the output head's share, and the shortlist's build states

Stage 1 of order 0050, on an A100 (any CUDA card works; the order asks for the A100):
1. **Decode:** one decode step of `base450m_cont` against its 128k head alone, a full 32k head and K gathered rows,
   batch 1 and 16, fp32 (~3 min).
2. **Training:** a forward + backward with the 128k head and its cross-entropy against the trunk alone, for the 450M
   and the grown 1.36B shape, time and peak memory (~5 min).
3. **Held-in states:** 64 windows of 1,024 tokens from each of the 13 shards' training region: the state the head
   reads, the full head's top-16, its log-sum-exp and the target's logit (~852k positions, ~20 min, ~1.7 GB on Drive).

4. **Stage 2a, the exact chunked cross-entropy** (`CB_FUSED_CE`): its tests on the card, then the training step with
   it beside the full path at micro 8 (memory freed, time), and the largest micro batch each path fits with its
   tokens per second, both shapes (~15 min). Lines: loss and gradients equal (the tests), at least 10 GB freed, at most
   3% slower.

Cells 1-3 ran on 2026-11-04 (their logs are in the repo): **run the setup cell and cell 4 only.**

Everything lands in `MyDrive/cubbyllm/head_shortlist/`. Claude Code builds and reads the index from there at home
(CPU). **Disconnect the runtime when the last cell ends.**
"""),
    code(r"""
import os, shutil, subprocess, sys
from google.colab import drive; drive.mount('/content/drive')
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!cd /content/CubbyLLM && git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv
DRIVE = '/content/drive/MyDrive/cubbyllm'
OUTD = f'{DRIVE}/head_shortlist'; os.makedirs(OUTD, exist_ok=True)
CK450 = '/content/base450m_cont_final.pt'
if not os.path.exists(CK450):                      # Drive-FUSE reads are slow: one copy to the local disk
    shutil.copy(f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt', CK450)
CKGROWN = f'{DRIVE}/runs/grow_mem_grown/grow_mem_grown_final.pt'     # its shape only (weights left at init)
CACHE = f'{DRIVE}/token_cache_cont'

def stream(*args):
    p = subprocess.Popen([sys.executable, '-u', 'validation/head_share_torch.py', *args], cwd='/content/CubbyLLM',
                         text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0

def keep(mode):
    import glob
    for f in glob.glob(f'/content/CubbyLLM/validation/logs/head_share_{mode}_cuda_*.json'):
        shutil.copy(f, OUTD); print('kept', os.path.basename(f))
"""),
    code(r"""
# --- 1. decode: the head's share of a step ---
stream('decode', '--ckpt', CK450)
keep('decode')
"""),
    code(r"""
# --- 2. training: the head + cross-entropy's share of a step, the 450M and the grown 1.36B ---
stream('train', '--ckpt', CK450, '--ckpt', CKGROWN, '--micro', '8', '--seq', '1024')
keep('train')
"""),
    code(r"""
# --- 3. held-in states for the shortlist's build (the training region only; never the held-out tails) ---
stream('dump', '--ckpt', CK450, '--cache', CACHE, '--split', 'heldin', '--windows', '64', '--seq', '1024',
       '--name', 'heldin_w64', '--out', '/content/head_shortlist')
shutil.copy('/content/head_shortlist/heldin_w64.npz', OUTD)
keep('dump')
print('done: disconnect the runtime')
"""),
    code(r"""
# --- 4. stage 2a: the chunked cross-entropy -- its tests on this card, then memory, time and the largest micro ---
t = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/model/test_fused_ce.py'],
                   cwd='/content/CubbyLLM', capture_output=True, text=True)
print(t.stdout[-1500:], t.stderr[-500:])
assert t.returncode == 0, 'the chunked cross-entropy disagrees with the full path on this card: stop here'
stream('train', '--ckpt', CK450, '--ckpt', CKGROWN, '--micro', '8', '--seq', '1024', '--fused', '--reps', '10')
keep('train')
print('done: disconnect the runtime')
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
