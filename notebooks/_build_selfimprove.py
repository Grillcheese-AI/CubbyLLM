"""Generate selfimprove.ipynb: order 0049, the self-improvement loop's T1 nights and T3's arms on the 450M, the trunk
frozen, only adapters learning, the VM (CubeLang, built here at a pinned commit) marking every attempt. Nick runs it;
Claude Code reads its outputs (Drive, `selfimprove/run_0049/`). Re-run after editing:
    python notebooks/_build_selfimprove.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "selfimprove.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0049: the self-improvement loop (T1, T3)

**Runtime:** an A100. Three steps, each its own cell, each resumable from Drive:
1. **the base read** (about 10 min): the base adapter on each arithmetic level. Its result is written into order 0049
   before night 1, and fixes the planted level. **Stop after it** and send the printout to Claude Code / Cowork.
2. **T1, the nights** (a few hours): the stream, the gap finder, practice, adapters, the gate, the router.
3. **T3, the arms** (about an hour): right answers only, against + own negatives, against + shared negatives.

It reads, from Drive:
- `runs/base450m_cont/base450m_cont_final.pt` (the base, frozen);
- `emitter/cubby450m_v12e_w_step_cont/` (the base adapter, never written);
- `emitter/emitter_sft_v12e_wd.jsonl` (0048's audited train records);
- `selfimprove/wikidata_properties_en_fr.json` (property labels, to mark the relation words).

It writes `selfimprove/run_0049/`. **Disconnect when a step ends and the next is not started.**
"""),
    code(r"""
import json, os, subprocess, sys, time
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy "protobuf>=7.35.1"
from google.colab import drive; drive.mount('/content/drive')
REPO, DRIVE = '/content/CubbyLLM', '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
ADAPTER   = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'      # the base adapter: read, never written
RECORDS   = f'{DRIVE}/emitter/emitter_sft_v12e_wd.jsonl'
PROPS     = f'{DRIVE}/selfimprove/wikidata_properties_en_fr.json'
OUT       = f'{DRIVE}/selfimprove/run_0049'
for p in (CKPT, TOKENIZER, RECORDS, PROPS, f'{ADAPTER}/emitter_lora.safetensors'):
    assert os.path.exists(p), p
os.makedirs(OUT, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md(r"""
## The VM: CubeLang at a pinned commit, built here

CubeLang's repository is public: no token. The branch `cowork/vm-contracts` at `29e81bd`, built in release; the build
time is printed and saved for the order.
"""),
    code(r"""
VM_COMMIT = '29e81bd'
t0 = time.time()
if not os.path.exists('/content/cubelang'):
    !git clone -q --branch cowork/vm-contracts https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
!cd /content/cubelang && git checkout -q {VM_COMMIT} && git log -1 --format='%h %cd %s'
if not os.path.exists(os.path.expanduser('~/.cargo/bin/cargo')):
    !curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
!cd /content/cubelang && ~/.cargo/bin/cargo build --release -q
os.environ['CUBELANG_EXE'] = '/content/cubelang/target/release/cubelang'
build_min = round((time.time() - t0) / 60, 1)
json.dump({'commit': VM_COMMIT, 'build_minutes': build_min}, open(f'{OUT}/vm_build.json', 'w'))
print('VM built in', build_min, 'min')
!$CUBELANG_EXE --version
!cd $REPO && python -c "from cubbyllm.bridges import reasoning_pb2; print('VM bridge ok (protobuf runtime matches)')"

def run(step, *extra, out=OUT):
    cmd = [sys.executable, '-u', 'validation/selfimprove/run.py', step, '--records', RECORDS, '--ckpt', CKPT,
           '--tokenizer', TOKENIZER, '--adapter', ADAPTER, '--props', PROPS, '--out', out, *map(str, extra)]
    with open(f'{out}/{step}.log', 'a', encoding='utf-8', buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        assert p.wait() == 0, f'{step} failed ({p.returncode})'
"""),
    md(r"""
## Step 1: the base read (then stop)

The base adapter, greedy, on each arithmetic level's held problems (train records kept out of the stream: the base
adapter was trained on these records, so the stream's own population is read) and on the val split beside them. The
planted level is one above the highest level it gets right at least half the time. **Stop here:** the result goes into
0049 before night 1.
"""),
    code(r"""
run('base-read')
br = json.load(open(f'{OUT}/base_read.json'))
print('base level', br['base_level'], '-> planted', br['planted'])
"""),
    md(r"""
## Step 2: T1, the nights

`LEVEL` is the planted level written into 0049 (10, from the base read of 2026-11-05). `RUN` names the run's folder:
`t1_2` is the second T1 (Cowork, 2026-11-05): the probe reads the held right rate (+5 points on up to 100 problems),
the noise family is value-computing step rows of level 2, and practice tries up to 32 where 8 find fewer than 10 right.
The stream rebuilds from the records and the seed; its held problems are checked against the base read's. Resumable
only from the start: if the runtime drops, move the run's folder aside and run it again.
"""),
    code(r"""
LEVEL = 10            # the planted level, written into 0049 (base level 9, held 0.52)
RUN = 't1_2'          # the second T1
RUN_OUT = f'{OUT}/{RUN}'
for f in ('requests.jsonl', 'nights.jsonl'):
    assert not os.path.exists(f'{RUN_OUT}/{f}'), f'{RUN_OUT}/{f} exists: move it aside first'
os.makedirs(RUN_OUT, exist_ok=True)
!cp {OUT}/base_read.json {RUN_OUT}/base_read.json
run('t1', '--level', LEVEL, out=RUN_OUT)
"""),
    md(r"""
## Step 3: T3, the arms

After the second T1, in the same session if time allows. The same planted gaps, from the base adapter: A (right programs only), B (+ the gap's own negatives, unlikelihood from
the first divergence), C (+ the other gaps' negatives too), at equal steps and rows a step.
"""),
    code(r"""
run('t3', '--level', LEVEL, out=RUN_OUT)
print(json.dumps({g: {a: {k: v[k] for k in ('held', 'transfer') } for a, v in r.get('arms', {}).items()}
                  for g, r in json.load(open(f'{RUN_OUT}/t3.json'))['gaps'].items()}, indent=1))
"""),
    md(r"""
## Step 4 (order 0053): the fast learner's cache

Independent of steps 2 and 3, and short (about 15 min on an A100). For 0049's wd/en and wd/fr problems, the base
adapter's greedy program up to its HOP1 word, the trunk's state there, and the top 32 property words by the trunk's own
score. The held problems' hash is checked against the base read's. Written to `selfimprove/run_0049/fastlearn/`; the VM
marks the candidates at home and the learner's days run there on CPU.
"""),
    code(r"""
FL_OUT = f'{OUT}/fastlearn'
os.makedirs(FL_OUT, exist_ok=True)
cmd = [sys.executable, '-u', 'validation/selfimprove/fastlearn_cache.py', '--records', RECORDS, '--ckpt', CKPT,
       '--tokenizer', TOKENIZER, '--adapter', ADAPTER, '--props', PROPS, '--out', FL_OUT]
with open(f'{FL_OUT}/cache.log', 'a', encoding='utf-8', buffering=1) as f:
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end=''); f.write(line)
    assert p.wait() == 0, f'the cache failed ({p.returncode})'
"""),
    md(r"""
## Done

**Disconnect the runtime.** At home Claude Code reads `run_0049/` into 0049's report. It also reads the saved adapters
on the battery lines that need the full stack (E8 reworded, R3's rows outside every gap).
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
