"""Generate gsm8k_dev_read.ipynb: order 0078's dev read on Colab (Nick: "colab", 2026-11-23) -- S0 then S1 on GSM8K
dev's 329 questions, one greedy chain and 16 sampled chains each, every try's VM value logged
(`validation/exp_0078_dev_read.py --ckpt`), then k and m chosen from the log (`exp_0078_dev_score.py`). The notebook
resumes from its own logs on Drive: a question already written is skipped, so a disconnect costs only the question in
flight. Re-run after editing:
    python notebooks/_build_gsm8k_dev_read.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "gsm8k_dev_read.ipynb"
CUBELANG_COMMIT = "efa2559"      # the build the school and the cross-checks used (the decimal-compare fix)


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# Order 0078: the dev read, S0 then S1 (Cowork's ruling, 2026-11-20)

**What it reads.** GSM8K dev (the 329 val arithmetic problems of `emitter_sft_v12e_w_slots.jsonl`), step by step in
the VM (H-E19's loop). Per question: one greedy chain (reported, not deciding) and 16 sampled chains (temperature 0.7,
top-k 40), every try's VM value written to a log, so the speak rule's k (8 or 16) and m are chosen afterwards with no
second pass. **S0** is `cubby450m_v12e_w_step_cont`, **S1** is `cubby450m_s1_0078`. Test is not read here.

**Time:** about 5.3 h per arm at home (58 s a question). The tries are decoded in batches by prompt length with the
question read once (`validation/dev_read_torch.py`, exact; 2026-11-24). Before that a T4 took 306 s a question; take an
A100 or an L4 when one is free (a T4 has no fast bf16). An L4 still took ~75 s a question, nearly all of it Python
launching the step's kernels one token at a time; on CUDA each batch size's step is now recorded once as a CUDA graph and
replayed (`GraphStep`, checked against the eager step when recorded; the check cell below runs the same test on the
card; `CB_DEV_READ_GRAPHS=0` turns it off). Rows read with it carry `"decode": "rows-graphs"`. **It resumes:** the logs are on Drive
(`emitter/gsm8k_dev_read/`); run the same cells again after a disconnect and it skips what is written. Rows written by
the first decode carry no `decode` field, the new ones `"decode": "rows"`: both are the same sampling (temperature 0.7,
top-k 40), seeded differently. Disconnect when it prints `dev read done`.
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
# --- the VM: cubelang at the commit the home reads used ---
COMMIT = '__COMMIT__'
if not os.path.exists('/content/cubelang/target/release/cubelang'):
    !curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null
    !git clone -q https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
    !cd /content/cubelang && git checkout -q {COMMIT} && $HOME/.cargo/bin/cargo build --release -q
EXE = '/content/cubelang/target/release/cubelang'
os.environ['CUBELANG_EXE'] = EXE
print(subprocess.run([EXE, 'check', '/content/cubelang/examples/gsm8k.cube'], capture_output=True, text=True).stdout.strip())
""".replace("__COMMIT__", CUBELANG_COMMIT)),
    code(r"""
# --- check: the recorded step decodes exactly as the eager one, on this card ---
!cd {REPO} && python -m pytest -q tests/validation/test_dev_read_torch.py
"""),
    code(r"""
# --- the read: S0 then S1, each resumable from its log ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
DATA      = f'{DRIVE}/emitter/emitter_sft_v12e_w_slots.jsonl'
LOGS      = f'{DRIVE}/emitter/gsm8k_dev_read'
ARMS      = {'s0': f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont', 's1': f'{DRIVE}/emitter/cubby450m_s1_0078'}
for p in (CKPT, TOKENIZER, DATA, *(f'{d}/emitter_lora.safetensors' for d in ARMS.values())):
    assert os.path.exists(p), p
os.makedirs(LOGS, exist_ok=True)
for arm, adapter in ARMS.items():
    out = f'{LOGS}/gsm8k_dev_read_{arm}.jsonl'
    done = sum(1 for _ in open(out, encoding='utf-8')) if os.path.exists(out) else 0
    print(f'=== {arm}: {done} of 329 already written', flush=True)
    if done >= 329:
        continue
    cmd = [sys.executable, '-u', 'validation/exp_0078_dev_read.py', '--arm', arm, '--ckpt', CKPT,
           '--adapter', adapter, '--tokenizer', TOKENIZER, '--data', DATA, '--out', out]
    with open(f'{LOGS}/run.log', 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    assert p.returncode == 0, f'{arm}: exit {p.returncode} (run the cell again: it resumes)'
print('dev read done ->', LOGS)
"""),
    code(r"""
# --- k and m from the logs (no second pass), chosen on dev without 80:200, then 80:200 read once at them (the split,
# Cowork 2026-11-24); S0 also without the five overlapping dev questions ---
DROP = f'{REPO}/validation/logs/gsm8k_dev_s0_overlap_2026-11-21.json'
for arm in ARMS:
    for drop in ([''] + ([DROP] if arm == 's0' else [])):
        args = [sys.executable, 'validation/exp_0078_dev_read.py', '--score', f'{LOGS}/gsm8k_dev_read_{arm}.jsonl']
        r = subprocess.run(args + (['--drop', drop] if drop else []), cwd=REPO, capture_output=True, text=True)
        s = json.loads(r.stdout)
        print(arm, 'without the 5' if drop else 'all 329', '| speak rule k', s['speak rule']['k'], 'm',
              s['speak rule']['m'], 'right', s['speak rule']['right'], '| greedy', s['greedy'], '| oracle',
              s['any try right (the oracle)'])
        h = s.get('held read')
        if h:
            print('   dev 80:200 at k', h['k'], 'm', h['m'], '|', h['at m'], '| top-1', h['top-1'],
                  '| path verifier next:', h['next step (path verifier) pre-registered'])
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
