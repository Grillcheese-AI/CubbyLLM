"""Generate system_eval.ipynb: order 0054's GPU part. Nick runs it; Claude Code runs the rest at home (the stack, the
VM, the stores, live Wikidata) and reads the outputs (Drive, `system_eval/run_0054/`). Re-run after editing:
    python notebooks/_build_system_eval.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "system_eval.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0054: the system evaluation, the GPU part

**Runtime:** an A100. **No model API is called** here or anywhere in this run; nothing needs a key.

The stack runs at home: the VM, the stores and live Wikidata. Its first pass wrote down every prompt it handed the
450M's programs seat. This notebook does the two jobs that need a GPU:
1. **the programs:** the 450M (programs adapter, greedy) writes a program for each prompt. The stack's second pass at
   home replays them.
2. **arm M:** the 450M talk adapter alone, closed book, answers every question of the five benchmarks.

**Step 1 is the preflight:** 50 per benchmark. It prints the time per question and the projected time, then
**stops**. Run step 2 only on Nick's go. Both steps resume from Drive. **Disconnect when done.**
"""),
    code(r"""
import hashlib, json, os, subprocess, sys, time
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
TOKENIZER  = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
CKPT_PROG  = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'     # the programs adapter's base
ADAPT_PROG = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'             # the serving programs seat
CKPT_TALK  = f'{DRIVE}/runs/base450m/base450m_final.pt'               # the talk adapter's base
ADAPT_TALK = f'{DRIVE}/system_eval/talk_v2_2'                          # the serving talk seat
RUN        = f'{DRIVE}/system_eval/run_0054'                           # prompts.jsonl, questions.jsonl in; outputs out
for p in (TOKENIZER, CKPT_PROG, CKPT_TALK, f'{ADAPT_PROG}/emitter_lora.safetensors', f'{ADAPT_TALK}/talk_lora.safetensors',
          f'{RUN}/prompts.jsonl', f'{RUN}/questions.jsonl'):
    assert os.path.exists(p), p
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()[:16]
frozen = {'commit': subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip(),
          'programs_adapter': sha(f'{ADAPT_PROG}/emitter_lora.safetensors'), 'talk_adapter': sha(f'{ADAPT_TALK}/talk_lora.safetensors'),
          'date': time.strftime('%Y-%m-%d')}
json.dump(frozen, open(f'{RUN}/frozen_gpu.json', 'w'), indent=1)
print(frozen)
!nvidia-smi --query-gpu=name,memory.total --format=csv

def gpu(step, *extra, log):
    common = ['--tokenizer', TOKENIZER]
    if step == 'programs':
        args = ['--prompts', f'{RUN}/prompts.jsonl', '--out', f'{RUN}/programs.jsonl', '--ckpt', CKPT_PROG, '--adapter', ADAPT_PROG]
    else:
        args = ['--questions', f'{RUN}/questions.jsonl', '--out', f'{RUN}/m', '--ckpt', CKPT_TALK, '--adapter', ADAPT_TALK]
    cmd = [sys.executable, '-u', 'validation/system_eval/gpu.py', step, *args, *common, *map(str, extra)]
    with open(f'{RUN}/{log}', 'a', encoding='utf-8', buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        assert p.wait() == 0, f'{step} failed ({p.returncode})'
"""),
    md(r"""
## Step 1: the preflight (then stop)

The programs for the first 250 prompts and M on the first 50 questions of each benchmark. Both are kept: step 2
resumes past them. Read the printed `s_per` and `projected_min`, and send them to Claude Code / Cowork. **Stop here
until Nick says go.**
"""),
    code(r"""
gpu('programs', '--n', 250, log='preflight.log')
gpu('m', '--n', 50, log='preflight.log')
n_prompts = sum(1 for _ in open(f'{RUN}/prompts.jsonl'))
print('prompts in all:', n_prompts)
"""),
    md(r"""
## Step 2: the full sets (on Nick's go)

Every prompt's program, then M on every question. Each resumes where it stopped: if the runtime drops, run this cell
again.
"""),
    code(r"""
gpu('programs', log='full.log')
gpu('m', log='full.log')
print('programs:', sum(1 for _ in open(f'{RUN}/programs.jsonl')), 'of', sum(1 for _ in open(f'{RUN}/prompts.jsonl')))
"""),
    md(r"""
## Done

**Disconnect the runtime.** At home, Claude Code replays the programs through the stack (arm C), and reads C-VM from
C's records, M from `m/`, and the read.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
