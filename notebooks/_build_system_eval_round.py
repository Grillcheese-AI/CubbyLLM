"""Generate system_eval_round.ipynb: one programs round for order 0054, between two passes of the stack. Nick runs it.

A later pass of the stack at home (the gather seat) can hand its programs seat prompts Colab has not seen. This
notebook writes the 450M's program for each prompt of ``system_eval/<ROUND>/prompts.jsonl`` to ``programs.jsonl``
beside it, as the first round wrote the others: the checkout is pinned to the first round's commit and the programs
adapter's hash is compared with the first round's (``run_0054/frozen_gpu.json``), so the new programs come from the
same seat. No questions, no arm M, nothing written into ``run_0054``. Re-run after editing:
    python notebooks/_build_system_eval_round.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "system_eval_round.ipynb"
ROUND = "run_0055_after"                                  # the folder under system_eval/ on Drive


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _src(text.replace("__ROUND__", ROUND))}


CELLS = [
    md(r"""
# 0054: one programs round (between two passes)

**Runtime:** an A100, as the first round. **No model API is called; nothing needs a key.** A few minutes: loading
the 450M is most of it.

The stack's later pass at home handed its programs seat prompts Colab had never seen. This notebook writes the
450M's program for each of them (`prompts.jsonl` in, `programs.jsonl` out, same folder), with the first round's
code, base and adapter. Claude Code replays them at home.

**Runtime > Run all.** It checks itself, then disconnects.
"""),
    code(r"""
import hashlib, json, os, subprocess, sys, time
from google.colab import drive; drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/cubbyllm'
REPO  = '/content/CubbyLLM'
FIRST = json.load(open(f'{DRIVE}/system_eval/run_0054/frozen_gpu.json'))   # the first round: its commit, its adapter
ROUND = f'{DRIVE}/system_eval/__ROUND__'                                  # prompts.jsonl in; programs.jsonl out
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'           # the programs adapter's base
ADAPTER   = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'                  # the serving programs seat
for p in (TOKENIZER, CKPT, f'{ADAPTER}/emitter_lora.safetensors', f'{ROUND}/prompts.jsonl'):
    assert os.path.exists(p), p

def sh(*cmd):
    subprocess.run(cmd, check=True)

if not os.path.exists(REPO):
    sh('git', 'clone', '-q', 'https://github.com/Grillcheese-AI/CubbyLLM.git', REPO)
sh('git', '-C', REPO, 'fetch', '-q', 'origin')
sh('git', '-C', REPO, 'checkout', '-q', FIRST['commit'])                  # the code that wrote the first round's programs
sh(sys.executable, '-m', 'pip', '-q', 'install', 'tokenizers', 'safetensors', 'numpy')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()[:16]

commit  = subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
adapter = sha(f'{ADAPTER}/emitter_lora.safetensors')
card    = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
assert commit.startswith(FIRST['commit']), (commit, FIRST['commit'])
assert adapter == FIRST['programs_adapter'], ('not the first round\'s programs adapter', adapter, FIRST['programs_adapter'])
n_prompts = sum(1 for x in open(f'{ROUND}/prompts.jsonl', encoding='utf-8') if x.strip())
print('commit', commit[:7], '| adapter', adapter, '| card', card, '| prompts', n_prompts)
"""),
    code(r"""
cmd = [sys.executable, '-u', 'validation/system_eval/gpu.py', 'programs', '--prompts', f'{ROUND}/prompts.jsonl',
       '--out', f'{ROUND}/programs.jsonl', '--ckpt', CKPT, '--adapter', ADAPTER, '--tokenizer', TOKENIZER]
with open(f'{ROUND}/round.log', 'a', encoding='utf-8', buffering=1) as f:
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end=''); f.write(line)
    assert p.wait() == 0, f'programs failed ({p.returncode})'
rows = [json.loads(x) for x in open(f'{ROUND}/programs.jsonl', encoding='utf-8') if x.strip()]
want = {json.loads(x)['prompt'] for x in open(f'{ROUND}/prompts.jsonl', encoding='utf-8') if x.strip()}
assert {r['prompt'] for r in rows} == want, (len(rows), len(want))
for r in rows:
    print(repr(r['prompt'][-90:]), '->', repr(r['program'][:240]))
json.dump({'commit': commit[:7], 'programs_adapter': adapter, 'card': card, 'date': time.strftime('%Y-%m-%d'),
           'prompts': len(want), 'programs': len(rows)}, open(f'{ROUND}/frozen_gpu.json', 'w'), indent=1)
print('DONE:', len(rows), 'programs for', len(want), 'prompts')
"""),
    code(r"""
from google.colab import runtime
drive.flush_and_unmount()
runtime.unassign()
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
