"""Generate system_eval_round.ipynb: one programs round for order 0054, between two passes of the stack. Nick runs it.

A later pass of the stack at home (the gather seat) can hand its programs seat prompts Colab has not seen. This
notebook writes the 450M's program for each prompt of ``system_eval/<ROUND>/prompts.jsonl`` to ``programs.jsonl``
beside it, as the first round wrote the others: the checkout is pinned to the first round's commit and the programs
adapter's hash is compared with the first round's (``run_0054/frozen_gpu.json``), so the new programs come from the
same seat. No questions, no arm M, nothing written into ``run_0054``.

The card is part of the seat: decoding is greedy, and another card can change a token on a near tie. So the notebook
stops and disconnects on any card but ``CARD`` before it loads anything, and it compares its programs, prompt by
prompt, with those another card wrote for the same prompts (``OTHER``: the round of 2026-11-11 ran on an L4, no A100
being free). A round's folder is never reused for another card: the programs step skips a prompt that already has a
program, so a second card would write nothing there. Re-run after editing:
    python notebooks/_build_system_eval_round.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "system_eval_round.ipynb"
ROUND = "run_0055_after_a100"                             # this round's folder under system_eval/ on Drive
OTHER = "run_0055_after"                                  # the same prompts' programs from another card (an L4)
CARD = "A100"                                             # the first round's card


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text.replace("__CARD__", CARD))}


def code(text):
    text = text.replace("__ROUND__", ROUND).replace("__OTHER__", OTHER).replace("__CARD__", CARD)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0054: one programs round (between two passes)

**Runtime:** an __CARD__, as the first round. On any other card the notebook says so and disconnects at once, before
it loads anything. **No model API is called; nothing needs a key.** A few minutes: loading the 450M is most of it.

The stack's later pass at home handed its programs seat prompts Colab had never seen. This notebook writes the
450M's program for each of them (`prompts.jsonl` in, `programs.jsonl` out, same folder), with the first round's
code, base and adapter, and compares them with the programs another card wrote for the same prompts. Claude Code
replays them at home.

**Runtime > Run all.** It checks itself, then disconnects.
"""),
    code(r"""
import hashlib, json, os, shutil, subprocess, sys, time
from google.colab import drive, runtime
try:
    card = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
except Exception:
    card = 'no GPU'
if '__CARD__' not in card:                                  # the card is part of the seat: stop before anything loads
    print(f'This runtime is "{card}", not an __CARD__. Nothing was run. Runtime > Change runtime type > __CARD__ GPU, then Run all.')
    runtime.unassign()
    raise SystemExit
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/cubbyllm'
REPO  = '/content/CubbyLLM'
FIRST = json.load(open(f'{DRIVE}/system_eval/run_0054/frozen_gpu.json'))   # the first round: its commit, its adapter
ROUND = f'{DRIVE}/system_eval/__ROUND__'                                   # prompts.jsonl in; programs.jsonl out
OTHER = f'{DRIVE}/system_eval/__OTHER__'                                   # the same prompts, another card's programs
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'           # the programs adapter's base
ADAPTER   = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'                  # the serving programs seat
os.makedirs(ROUND, exist_ok=True)
if not os.path.exists(f'{ROUND}/prompts.jsonl'):
    shutil.copy(f'{OTHER}/prompts.jsonl', f'{ROUND}/prompts.jsonl')
for p in (TOKENIZER, CKPT, f'{ADAPTER}/emitter_lora.safetensors', f'{ROUND}/prompts.jsonl', f'{OTHER}/programs.jsonl'):
    assert os.path.exists(p), p

def sh(*cmd):
    subprocess.run(cmd, check=True)

def rows_of(p):
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()[:16]

if not os.path.exists(REPO):
    sh('git', 'clone', '-q', 'https://github.com/Grillcheese-AI/CubbyLLM.git', REPO)
sh('git', '-C', REPO, 'fetch', '-q', 'origin')
sh('git', '-C', REPO, 'checkout', '-q', FIRST['commit'])                  # the code that wrote the first round's programs
sh(sys.executable, '-m', 'pip', '-q', 'install', 'tokenizers', 'safetensors', 'numpy')
commit  = subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
adapter = sha(f'{ADAPTER}/emitter_lora.safetensors')
assert commit.startswith(FIRST['commit']), (commit, FIRST['commit'])
assert adapter == FIRST['programs_adapter'], ('not the first round\'s programs adapter', adapter, FIRST['programs_adapter'])
want = {r['prompt'] for r in rows_of(f'{ROUND}/prompts.jsonl')}
assert want == {r['prompt'] for r in rows_of(f'{OTHER}/prompts.jsonl')}, 'the two rounds do not hold the same prompts'
print('commit', commit[:7], '| adapter', adapter, '| card', card, '| prompts', len(want))
"""),
    code(r"""
cmd = [sys.executable, '-u', 'validation/system_eval/gpu.py', 'programs', '--prompts', f'{ROUND}/prompts.jsonl',
       '--out', f'{ROUND}/programs.jsonl', '--ckpt', CKPT, '--adapter', ADAPTER, '--tokenizer', TOKENIZER]
with open(f'{ROUND}/round.log', 'a', encoding='utf-8', buffering=1) as f:
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end=''); f.write(line)
    assert p.wait() == 0, f'programs failed ({p.returncode})'
rows = rows_of(f'{ROUND}/programs.jsonl')
assert {r['prompt'] for r in rows} == want, (len(rows), len(want))
other = {r['prompt']: r['program'] for r in rows_of(f'{OTHER}/programs.jsonl')}
other_card = json.load(open(f'{OTHER}/frozen_gpu.json')).get('card', 'not recorded')
same = [r['program'] == other.get(r['prompt']) for r in rows]
for r, s in zip(rows, same):
    print('SAME' if s else 'DIFFERENT', '|', repr(r['prompt'][-70:]), '->', repr(r['program'][:160]))
json.dump({'commit': commit[:7], 'programs_adapter': adapter, 'card': card, 'date': time.strftime('%Y-%m-%d'),
           'prompts': len(want), 'programs': len(rows), 'compared_with': '__OTHER__', 'compared_card': other_card,
           'same_programs': sum(same), 'different_programs': len(same) - sum(same)},
          open(f'{ROUND}/frozen_gpu.json', 'w'), indent=1)
print(f'DONE on {card}: {len(rows)} programs; {sum(same)} the same as on "{other_card}", {len(same) - sum(same)} different')
"""),
    code(r"""
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
