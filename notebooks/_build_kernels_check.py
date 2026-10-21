"""Generate kernels_check.ipynb: the Triton kernels (fused MinGRU, banded flash attention) tested and timed
on a Colab GPU, against what they replace -- a short session on a cheap card (an L4 has the RTX PRO 6000's
99 KB of shared memory a block) before a training run takes them. The logic is
tests/model/backbone/test_{mingru,window_attn}_triton.py and validation/bench_kernels.py. Re-run after editing:
    python notebooks/_build_kernels_check.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "kernels_check.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# CubbyLLM: the Triton kernels, tested and timed on this GPU

The fused MinGRU recurrence (`cubbyllm/model/backbone/mingru_triton.py`, `CB_MINGRU_TRITON=1`) and the banded
flash attention (`window_attn_triton.py`, `CB_ATTN_TRITON=1`), each against what it replaces: their tests on the
card (values and gradients against the references, the mixers under their flags), then one layer's forward +
backward at the 450M's shapes (`validation/bench_kernels.py`) -- with flex attention at its default tiles and at
tiles for a 99 KB card beside them. No Drive, no checkpoint: ~5 minutes on any GPU.
"""),
    code(r"""
import os, subprocess, sys
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!cd /content/CubbyLLM && git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__)

def stream(cmd, **env):
    p = subprocess.Popen(cmd, cwd='/content/CubbyLLM', env=dict(os.environ, **env), text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        if 'Warning' not in line and 'warn' not in line:
            print(line, end='')
    return p.wait()
"""),
    code(r"""
# --- the tests, on this card ---
stream([sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider',
        'tests/model/backbone/test_mingru_triton.py', 'tests/model/backbone/test_window_attn_triton.py'])
"""),
    code(r"""
# --- one layer, forward + backward, at the 450M's shapes (micro 32 x 1024 tokens; 8 on a card under 40 GB) ---
B = '32' if torch.cuda.get_device_properties(0).total_memory > 40 * 2**30 else '8'
stream([sys.executable, '-u', 'validation/bench_kernels.py'], CB_B=B, CB_OUT='/content/bench_kernels.json')
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "L4"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
