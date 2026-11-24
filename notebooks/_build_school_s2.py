"""Generate school_s2.ipynb: order 0078's arm S2 on Colab -- S1's adapter, then the school (`validation/
train_school_torch.py`, H-E19) on GSM8K train outside dev, each wrong step named by cubelang's `examples/gsm8k.cube`.
The notebook is thin; the logic is the trainer and `validation/school_gsm8k.py`. Re-run after editing:
    python notebooks/_build_school_s2.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "school_s2.ipynb"
CUBELANG_COMMIT = "efa2559"      # cubelang main with gsm8k.cube and the decimal-compare fix (cross-checked, d9f9f5a)


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# Order 0078, arm S2: the school on GSM8K, from S1's adapter

**Run it only after Code B's dev read of S1 is written and it calls no scale-up** (order 0078's rule).

**What this trains.** S2 is S1 (`emitter/cubby450m_s1_0078`) followed by the school (H-E19): rounds of GSM8K train
problems outside dev (S1's step rows with a `gsm8k_id`), each step tried until the VM says right, the solve
reinforced by its dopamine, each wrong try paired with the right one and **named by `gsm8k.cube`'s `classify_error`**
(stopped early, reversed, unit conversion, one number off, other). **A round is kept only if dev does not drop**
(the gate: the step loop on GSM8K's 329 dev questions, greedy, scored against the gold); otherwise the adapter goes
back to the best one.

**Needs on Drive:** the base, the tokenizer, S1's adapter and step rows, `gsm8k/train.jsonl` and `gsm8k/dev_ids.json`
(GSM8K train with ids; off-repo), `emitter/emitter_sft_v12e_w_slots.jsonl` (its val split is the 329 dev).
**Writes** `emitter/cubby450m_s2_0078/` (the kept adapter, `school_rounds.jsonl`, `school.log`). Training only:
disconnect when it prints `school done`. GSM8K test is not read here.
"""),
    code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, subprocess, time, json
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
# --- the VM: cubelang built at the commit the reads at home used (gsm8k.cube, the decimal-compare fix) ---
COMMIT = '__COMMIT__'
if not os.path.exists('/content/cubelang/target/release/cubelang'):
    !curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null
    !git clone -q https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
    !cd /content/cubelang && git checkout -q {COMMIT} && $HOME/.cargo/bin/cargo build --release -q
EXE, CUBE = '/content/cubelang/target/release/cubelang', '/content/cubelang/examples/gsm8k.cube'
os.environ['CUBELANG_EXE'] = EXE
print(subprocess.run([EXE, 'check', CUBE], capture_output=True, text=True).stdout.strip())
""".replace("__COMMIT__", CUBELANG_COMMIT)),
    code(r"""
# --- the run ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
S1        = f'{DRIVE}/emitter/cubby450m_s1_0078'
S1_ROWS   = f'{DRIVE}/emitter/s1_0078_step.jsonl'
GSM8K     = f'{DRIVE}/gsm8k'
HELDOUT   = f'{DRIVE}/emitter/emitter_sft_v12e_w_slots.jsonl'
OUT       = f'{DRIVE}/emitter/cubby450m_s2_0078'
ROUNDS, PROBLEMS, MAX_TRIES, GATE_N = 20, 256, 16, 329
for p in (CKPT, TOKENIZER, f'{S1}/emitter_lora.safetensors', f'{S1}/TRAINED', S1_ROWS, f'{GSM8K}/train.jsonl',
          f'{GSM8K}/dev_ids.json', HELDOUT):
    assert os.path.exists(p), p
POOL = '/content/s1_gsm8k_step.jsonl'                 # S1's GSM8K rows only (the school's pool), cut once
if not os.path.exists(POOL):
    with open(S1_ROWS, encoding='utf-8') as f, open(POOL, 'w', encoding='utf-8') as g:
        for line in f:
            if '"gsm8k_id"' in line:
                g.write(line)
print(sum(1 for _ in open(POOL, encoding='utf-8')), 'GSM8K step rows in the pool file')
os.makedirs(OUT, exist_ok=True)
cmd = [sys.executable, '-u', 'validation/train_school_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
       '--adapter', S1, '--pool', POOL, '--heldout', HELDOUT, '--out', OUT, '--step',
       '--rounds', str(ROUNDS), '--problems', str(PROBLEMS), '--max-tries', str(MAX_TRIES), '--gate-n', str(GATE_N),
       '--gate-skip', '80:200',                          # 0078's split: dev 80:200 is read once, never gated on
       '--gsm8k', GSM8K, '--cube-exe', EXE, '--cube', CUBE]
with open(f'{OUT}/train.log', 'a', encoding='utf-8', buffering=1) as f:
    f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end=''); f.write(line)
    p.wait()
print('school done' if p.returncode == 0 else f'exit {p.returncode}', '->', OUT)
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
