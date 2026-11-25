"""Generate emitter_pf_0082.ipynb: order 0082's arm P on Colab -- the Gate A adapter continued one epoch on S0's file
plus H-E18's program-first plan and chain records (`validation/sft/build_0082_mix.py`), S0's recipe; then the
generations of P and of S0 on the unseen-writer eval and the v12e_w val split's families other than arithmetic
(`train_emitter_torch.py --gen-only`), read at home by `validation/exp_he18_ab.py`. An optional last cell (off) runs
the served-loop reads on grilly2. Re-run after editing:
    python notebooks/_build_emitter_pf_0082.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "emitter_pf_0082.ipynb"
CUBELANG_COMMIT = "efa2559"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# Order 0082, arm P: the programs adapter with H-E18's plan and chain records

**What this trains.** S0 (`cubby450m_v12e_w_step_cont`, the served programs adapter) is the Gate A adapter
(`cubby450m_v12e_w_tg_cont`) continued one epoch on `emitter_sft_v12e_w_tg30_step.jsonl`: **4,691 steps of 32**, lr 2e-4,
rank 16 / alpha 32, up to 1,024 tokens (its adapter file and its log; `emitter_s1.ipynb`'s "batch 16" is not what ran). **P is the same start, the same recipe and the same base on S0's file plus the
program-first plan and chain train records** (`emitter_sft_0082_pf.jsonl`; counts and SHA-256s in
`validation/logs/build_0082_mix_2026-11-24.json`). No program-first arithmetic. S0 took 34.6 min on an A100; this mix
is about 1.5% longer (4,759 steps of 32).

**Batch 32 on any card.** Under 70 GB a batch of 32 is cut into chunks of at most 8,192 padded tokens
(`--micro-tokens`), each weighted by its share of the answer tokens: the same 32 rows a step, in the same order, and the
same gradient as the whole batch (`tests/validation/test_talk_micro_batches.py`). **The first run (2026-11-25, an L4)
trained at batch 8** (19,033 steps; batch 16 ran out of memory), 4 times S0's steps: it stays in
`cubby450m_0082_p` as a side arm, labelled. This notebook's P is `cubby450m_0082_p32`.

**Then the generations, P and S0, greedy** (`--gen-only`): the unseen-writer eval's plan and chain records (285 and
294, `pf_heldout_eval_w_slots.jsonl`) and the v12e_w val split's chain, kernel, plan and role-binding records. They are
read at home (`exp_he18_ab.py`, the real VM), once per arm. Nothing here is chosen on them. Both arms are decoded the
same way (`--gen-batch`: the records of one prompt length together, with 0078's CUDA-graph step; greedy, as one at a
time gives them on CPU).

**It resumes:** a finished step is skipped when its files exist. Disconnect when it prints `0082 done`.
"""),
    code(r"""
# --- setup: clone, deps, Drive ---
import os, sys, subprocess, time, json, math
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers safetensors numpy "protobuf>=7.35.1"
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
CARD = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('card:', CARD)
"""),
    code(r"""
# --- paths and S0's recipe ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
CKPT      = f'{DRIVE}/runs/base450m_cont/base450m_cont_final.pt'
TOKENIZER = f'{DRIVE}/token_cache_base/tokenizer/grillcheese_bbpe128k.json'
MIX       = f'{DRIVE}/emitter/emitter_sft_0082_pf.jsonl'
HELD      = f'{DRIVE}/emitter/pf_heldout_eval_w_slots.jsonl'
FROM      = f'{DRIVE}/emitter/cubby450m_v12e_w_tg_cont'               # the Gate A adapter, as S0 started from
S0        = f'{DRIVE}/emitter/cubby450m_v12e_w_step_cont'
P         = f'{DRIVE}/emitter/cubby450m_0082_p32'                   # batch 32, S0's; the batch-8 run stays in cubby450m_0082_p
GEN       = {'p32': P, 's0': f'{DRIVE}/emitter/exp_0082/s0'}           # S0's adapter folder is left as it is
EPOCHS, BATCH, LR, RANK, ALPHA, SEED, MAX_LEN = 1, 32, 2e-4, 16, 32, 0, 1024   # S0's: 4,691 steps of 32
import torch
MICRO = 0 if torch.cuda.get_device_properties(0).total_memory >= 70e9 else 8192   # 8 x 1,024 fit on the L4
print('micro-batch tokens:', MICRO or 'off (the whole batch at once)')
for p in (CKPT, TOKENIZER, MIX, HELD, f'{FROM}/emitter_lora.safetensors', f'{S0}/emitter_lora.safetensors'):
    assert os.path.exists(p), p
def stream(cmd, log):
    with open(log, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {CARD} {' '.join(cmd[2:])}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        p.wait()
    assert p.returncode == 0, f'exit {p.returncode}'
ROWS = sum(int(r.get('repeat', 1) or 1) for r in map(json.loads, open(MIX, encoding='utf-8')) if r['split'] == 'train')
STEPS = math.ceil(EPOCHS * ROWS / BATCH)
print('P:', ROWS, 'train rows ->', STEPS, 'steps')
"""),
    code(r"""
# --- arm P: train (S0's recipe, batch 32; only the data differs) ---
if os.path.exists(f'{P}/TRAINED'):
    print('trained already ->', P)
else:
    os.makedirs(P, exist_ok=True)
    stream([sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
            '--data', MIX, '--out', P, '--steps', str(STEPS), '--batch', str(BATCH), '--lr', str(LR),
            '--rank', str(RANK), '--alpha', str(ALPHA), '--max-len', str(MAX_LEN), '--seed', str(SEED),
            '--micro-tokens', str(MICRO), '--resume-adapter', FROM, '--tag', '_0082_p32', '--no-gen'], f'{P}/train.log')
    os.system(f'cp {REPO}/validation/logs/train_emitter_torch_0082_p32.* {P}/ 2>/dev/null')
    open(f'{P}/TRAINED', 'w').write(f"{time.strftime('%F %T')} {CARD}")
print('P ->', P)
"""),
    code(r"""
# --- the generations, P and S0: the unseen-writer eval (plan, chain) and the val split's other families ---
GEN_BATCH = 32        # records of one prompt length decoded together, CUDA graphs (one at a time: ~12 s a record on an L4)
for arm, out in GEN.items():
    adapter = P if arm == 'p32' else S0
    done = [f'{out}/val_generations_0082_{arm}.json', f'{out}/val_generations_0082_{arm}_pfheld.json']
    if all(os.path.exists(d) for d in done):
        print(arm, 'generated already'); continue
    os.makedirs(out, exist_ok=True)
    stream([sys.executable, '-u', 'validation/train_emitter_torch.py', '--ckpt', CKPT, '--tokenizer', TOKENIZER,
            '--data', MIX, '--out', out, '--gen-only', '--resume-adapter', adapter, '--gen-per-task', '0',
            '--gen-tasks', 'chain,kernel,plan,role_binding', '--gen-extra', HELD, '--gen-extra-per-task', '0',
            '--gen-batch', str(GEN_BATCH), '--tag', f'_0082_{arm}'], f'{out}/gen.log')
    open(f'{out}/CARD', 'w').write(CARD)
print('0082 done ->', GEN)
"""),
    md(r"""
### Optional: the served-loop reads on grilly2 (off)

The gen-3 held split (seed 7) and H-E15's live eight through the served loop, P against S0, with `CB_EMIT_TRIE` as
0081 leaves it (killed: off). 0081's bench arm is on master (`bf9b90d`) and the base's grilly2 export is on Drive
(`runs/base450m_cont/grilly`, 2026-11-24). **Off:** these reads run at home on the local card; turn it on only if
the benches' inputs are on Drive too.
"""),
    code(r"""
RUN_SERVED = False                 # True only once the four conditions above hold
TRIE = 'off'                       # CB_EMIT_TRIE as 0081 leaves it (killed: off)
EXPORT = f'{DRIVE}/runs/base450m_cont/grilly'        # the base's grilly2 export (config.json, model.safetensors)
if RUN_SERVED:
    import glob, shutil
    have = subprocess.run([sys.executable, 'validation/exp_r17_gen3_heldout.py', '--help'], cwd=REPO,
                          capture_output=True, text=True).stdout
    assert '--cubby450m' in have and os.path.exists(f'{REPO}/validation/exp_0081_live.py'), "0081's bench arm is not on master"
    assert os.path.exists(f'{EXPORT}/model.safetensors'), EXPORT
    # Vulkan on Colab's NVIDIA driver: two JSON files, nothing installed (grilly2's notebooks/colab.ipynb, step 1)
    def lib(n):
        return next((os.path.join(d, n) for d in ['/usr/lib64-nvidia', '/usr/lib/x86_64-linux-gnu'] if os.path.exists(os.path.join(d, n))), None)
    glx, egl = lib('libGLX_nvidia.so.0'), lib('libEGL_nvidia.so.0')
    assert glx and egl, (glx, egl)
    for path, val in (('/etc/vulkan/icd.d/nvidia_icd.json', {'file_format_version': '1.0.1', 'ICD': {'library_path': glx, 'api_version': '1.3.0'}}),
                      ('/usr/share/glvnd/egl_vendor.d/10_nvidia.json', {'file_format_version': '1.0.0', 'ICD': {'library_path': egl}})):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(val, open(path, 'w'))
    !pip -q install "grilly2[huggingface]==0.1.0" && grilly-doctor
    COMMIT = '__COMMIT__'                                   # the VM, as school_s2.ipynb builds it
    if not os.path.exists('/content/cubelang/target/release/cubelang'):
        !curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null
        !git clone -q https://github.com/Grillcheese-AI/cubelang.git /content/cubelang
        !cd /content/cubelang && git checkout -q {COMMIT} && $HOME/.cargo/bin/cargo build --release -q
    os.environ['CUBELANG_EXE'] = '/content/cubelang/target/release/cubelang'
    for arm in ('s0', 'p32'):
        adapter = P if arm == 'p32' else S0
        stream([sys.executable, '-u', 'validation/exp_r17_gen3_heldout.py', '--cubby450m', EXPORT, '--emitter-adapter',
                adapter, '--tokenizer', TOKENIZER, '--arms', TRIE, '--seed', '7', '--tag', f'_0082_{arm}'], f'{GEN[arm]}/served.log')
        stream([sys.executable, '-u', 'validation/exp_0081_live.py', '--cubby450m', EXPORT, '--emitter-adapter', adapter,
                '--tokenizer', TOKENIZER, '--arms', TRIE, '--out', f'{GEN[arm]}/live_0082_{arm}.json'], f'{GEN[arm]}/served.log')
        os.system(f'cp {REPO}/validation/logs/exp_0081_r17_* {GEN[arm]}/ 2>/dev/null')
    print('served reads done on', CARD)
""".replace("__COMMIT__", CUBELANG_COMMIT)),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
