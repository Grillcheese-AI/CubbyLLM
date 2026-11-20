"""Generate emitter_s1.ipynb: order 0078's arm S1 on Colab -- S0's recipe (the write-back step format, H-E19) on
S1's first cut (Nick, 2026-11-20: ~1.5M step rows). The notebook is thin; the logic is validation/train_emitter_torch.py.
Re-run after editing:
    python notebooks/_build_emitter_s1.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "emitter_s1.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# Order 0078, arm S1: the 450M step emitter on S1's first cut

**What this trains.** S0 is today's step adapter (`cubby450m_v12e_w_step_cont`): the Gate A adapter continued one epoch
on `emitter_sft_v12e_w_tg30_step.jsonl` (148,437 step rows: v12e_w re-cut + 30K TinyGSM step rows), batch 16, lr 2e-4,
rank 16 / alpha 32, up to 1,024 tokens. **S1 is the same recipe and the same budget per row on more rows**
(`s1_0078_step.jsonl`, 1,499,996 step rows; `validation/logs/s1_0078_2026-11-20.json`):

| part | step rows |
|---|---|
| v12e_w re-cut (S0's base; its 5,819 GSM8K train problems tagged `gsm8k_id`; its val split = GSM8K's 329 dev) | 34,873 |
| GSM8K train's other golds as programs (`gsm8k_programs.py`, 112) | 479 |
| TinyGSM, 348,006 whole programs drawn (seed 78) to GSM8K train's mix of step counts | 1,464,644 |

Every TinyGSM program was run in the VM when it was converted, and agreed (step 1, 6,350,518 of 6,350,518).

**Budget.** One epoch at batch 16: about 93,700 steps, ten times S0's (S0 trained in 20-30 min on an A100), so about
**3.5-5 hours on an A100**. The trainer saves the adapter every 500 steps but does not resume at a step: run it in one
session. Training only (`--no-gen`); disconnect when it prints `trained:`.

**The read (local, free; Code B):** the step loop on GSM8K dev (329) with the real VM, beside S0's on the same rows
(`validation/exp_he19_step_loop.py`). The scale-up rule is in 0078: S1's dev right short of S0's + 10 points -> the
next cut (1M programs). GSM8K test is not read here.
"""),
    code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, subprocess, time, json, math
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy "protobuf>=7.35.1"
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code(r"""
# --- the run (S0's settings; only the data differs) ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
S1_DATA   = f'{DRIVE}/emitter/s1_0078_step.jsonl'                  # 1,499,996 step rows (1.9 GB)
S1_FROM   = f'{DRIVE}/emitter/cubby450m_v12e_w_tg_cont'             # the Gate A adapter, as S0 started from
OUT       = f'{DRIVE}/emitter/cubby450m_s1_0078'
EPOCHS, BATCH, LR, RANK, ALPHA, SEED, MAX_LEN = 1, 16, 2e-4, 16, 32, 0, 1024
for p in (CKPT, TOKENIZER, S1_DATA, f'{S1_FROM}/emitter_lora.safetensors'):
    assert os.path.exists(p), p

def train_rows(path):
    n = 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r['split'] == 'train':
            n += int(r.get('repeat', 1) or 1)
    return n
ROWS = train_rows(S1_DATA)
STEPS = math.ceil(EPOCHS * ROWS / BATCH)
print('S1', ROWS, 'train rows ->', STEPS, 'steps')
"""),
    code(r"""
if os.path.exists(f'{OUT}/emitter_lora.safetensors') and os.path.exists(f'{OUT}/TRAINED'):
    print('done already ->', OUT)
else:
    os.makedirs(OUT, exist_ok=True)
    tag = '_s1_0078'
    cmd = [sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
           '--data', S1_DATA, '--out', OUT, '--steps', str(STEPS), '--batch', str(BATCH), '--lr', str(LR),
           '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
           '--resume-adapter', S1_FROM, '--tag', tag, '--no-gen']
    with open(f'{OUT}/train.log', 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch{tag}.* {OUT}/ 2>/dev/null')
    if p.returncode == 0:
        open(f'{OUT}/TRAINED', 'w').write(time.strftime('%F %T'))
    print('exit', p.returncode, '->', OUT)
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
