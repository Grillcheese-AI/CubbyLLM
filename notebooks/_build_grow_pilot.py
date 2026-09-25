"""Generate grow_pilot.ipynb: the growth pilot of docs/research/2026-09-25-grow-450m.md
(§5). Four arms of ~1 A100-hour each, from the trained base450m: the 450M continued
(the control), width x2, depth x2 + FFN x2 with zeroed exits, and the same shape as
a plain copy. The notebook is thin; the logic is validation/grow_base.py (grow +
exactness check) and validation/train_base.py (CB_INIT_FROM). Re-run after editing:
    python notebooks/_build_grow_pilot.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "grow_pilot.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def arm_cell(arm):
    return code(f"""
run_arm('{arm}')
""")


INTRO = md(r"""
# Growth pilot: base450m → 1-1.5B, four arms × ~1 A100-hour

**What this is.** The pilot of `docs/research/2026-09-25-grow-450m.md` (§5, pre-registered).

**Why.** The trained base (576M; trunk 314M; 2.77B tokens) can be grown into a bigger model that starts **exactly** where the base is. Step 0 verified this locally, in torch and on grilly2 (`validation/logs/grow_step0_*`, `grow_parity_*`). This pilot asks the one question step 0 cannot: **does the grown model learn faster than the 450M itself, for the same GPU time?**

| Arm | Shape | Params | Start | Micro-batch |
|---|---|---:|---|---:|
| `control` | the 450M, continued | 576M | the base | 32 |
| `width2` | d2048, L32, 8 heads × 256, FFN 4096 (HyperCloning) | 1,779M | exact | 8 |
| `depth2_ffn2` | d1024, L64, FFN 4096; each 3-layer block followed by a copy whose exits are zero | 1,293M | exact | 8 |
| `depth2_ffn2_copy` | the same shape, plain copies (G_stack's operator) | 1,293M | +1.86 nats (step 0) | 8 |

**How each arm runs**
- It starts from the base on Drive. A grown arm is grown on this machine in about a minute (`grow_base.py`, with a one-window exactness check), so nothing is uploaded.
- It trains with `train_base.py` for the same wall-clock (`ARM_HOURS`, default 1.0). The script measures its own speed, sizes its step count to the hours, and ends with a 20% decay.
- Settings: the same LR as the base (6e-4, re-warmed over 50 steps), 131k tokens per step, seed 1 (seed 0 would redraw the base's own first batches), and validation every 40 steps on the base's held-out tails.
- The same seed and mix for every arm, but the exact sequences differ with the micro-batch size. That noise is expected to be small against the 0.03-nat bar.

**Pick and kill (pre-registered)**
- **Pick:** the grown arm with the lowest projected loss at 20 H100-hours.
- **Kill growth, for now:** no grown arm is ≥ 0.03 nats below the control at equal wall-clock at the end, and no grown arm's slope reaches that by 20 H100-hours. The 450M then keeps training on its own.

**Before you run**
1. `master` has the growth code: `cubbyllm/model/grow.py`, `HybridBackbone(ffn_mult=…)`, `validation/grow_base.py`, and `train_base.py` with `CB_INIT_FROM` and `CB_FFN_MULT`. The setup cell checks all four.
2. Drive has `cubbyllm/runs/base450m/base450m_final.pt` and the token cache the base used (`cubbyllm/token_cache_base/`, 14.7 GB, staged to local disk below).
3. An A100-80G with high RAM, for ~4.5 hours: 4 arms plus staging, growing and compiling. The arms' checkpoints stay on local disk (a width arm's optimizer state alone is ~21 GB). Only metrics, logs and the summary go to Drive, under `cubbyllm/runs/grow_pilot/`, so ~1 MB of Drive is used.
4. A lost session loses only the arm that was running: rerun its cell. Finished arms are marked on Drive and skipped.
""")

SETUP = code(r"""
# --- setup: clone, deps, Drive (once per session) ---
import os, sys, time, subprocess, signal, json, glob, shutil, re, math
SESSION_T0 = time.time()
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install tokenizers numpy
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
!free -g | head -2
import torch; print('torch', torch.__version__)
need = {
    'cubbyllm/model/grow.py': 'def grow_into',
    'cubbyllm/model/backbone/hybrid.py': 'ffn_mult',
    'validation/grow_base.py': 'grow_into',
    'validation/train_base.py': 'CB_INIT_FROM',
}
for path, mark in need.items():
    full = f'{REPO}/{path}'
    assert os.path.exists(full) and mark in open(full, encoding='utf-8').read(), \
        f'{path} on master lacks {mark!r}: push the growth code first'
print('growth code present on master')
""")

PATHS = code(r"""
# --- EDIT to your layout ---
DRIVE        = '/content/drive/MyDrive/cubbyllm'
BASE_CKPT    = f'{DRIVE}/runs/base450m/base450m_final.pt'
CORPUS_DRIVE = f'{DRIVE}/token_cache_base'           # make_base_cache.py output, tokenizer/ inside
CORPUS       = '/content/token_cache'                # LOCAL SSD
TOKENIZER    = f'{CORPUS}/tokenizer/grillcheese_bbpe128k.json'
RUN_DIR      = f'{DRIVE}/runs/grow_pilot'            # metrics, logs, summary (small)
WORK         = '/content/pilot'                      # grown checkpoints, arm checkpoints (large, local)
ARM_HOURS    = 1.0                                   # wall-clock per arm, compile included
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(WORK, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)
assert os.path.exists(BASE_CKPT), BASE_CKPT
assert glob.glob(f'{CORPUS_DRIVE}/*.u32'), f'no .u32 shards in {CORPUS_DRIVE}'
!rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
if not os.path.exists(TOKENIZER):
    os.makedirs(os.path.dirname(TOKENIZER), exist_ok=True)
    !cp {DRIVE}/grillcheese_bbpe128k.json {TOKENIZER}
assert os.path.exists(TOKENIZER), TOKENIZER
!du -sh {CORPUS}; df -h /content | tail -1
""")

CONFIG = code(r"""
# --- the arms, the shared config, and a runner that tees each arm's log to Drive ---
BASE = dict(
    CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, CB_DRIVE_DIR='',       # arm checkpoints stay on local disk
    CB_WINDOW='512', CB_ATTN_EVERY='3', CB_S='1024', CB_TOKENS_PER_STEP='131072',
    CB_LR='6e-4', CB_BETA2='0.95', CB_WD='0.1',                   # the base's own settings
    CB_WARMUP='50', CB_DECAY_FRAC='0.2', CB_SEED='1',             # 50: 300 would be most of an arm
    CB_EVAL='40', CB_GEN='0', CB_CKPT_MIN='100000', CB_STABLE_END='0',
)
SHAPE_450M = dict(CB_D='1024', CB_L='32', CB_HEADS='4', CB_FFN_MULT='2')
SHAPE_DEEP = dict(CB_D='1024', CB_L='64', CB_HEADS='4', CB_FFN_MULT='4')
ARMS = {
    'control':          dict(grow=None, shape=SHAPE_450M, micro='32'),
    'width2':           dict(grow=['--width', '2'],
                             shape=dict(CB_D='2048', CB_L='32', CB_HEADS='8', CB_FFN_MULT='2'), micro='8'),
    'depth2_ffn2':      dict(grow=['--depth', '2', '--ffn', '2'], shape=SHAPE_DEEP, micro='8'),
    'depth2_ffn2_copy': dict(grow=['--depth', '2', '--ffn', '2', '--exit', 'copy'], shape=SHAPE_DEEP, micro='8'),
}


def tee(cmd, env_extra, log):
    env = dict(os.environ, **env_extra)
    p = None
    try:
        with open(log, 'a', encoding='utf-8', buffering=1) as f:
            f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[2:])} "
                    + ' '.join(f'{k}={v}' for k, v in sorted(env_extra.items()) if k.startswith('CB_')) + '\n')
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            for line in p.stdout:
                print(line, end=''); f.write(line)
            p.wait()
        return p.returncode
    finally:
        if p and p.poll() is None:       # a stopped cell: let the script save first
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                p.terminate()


def run_arm(arm):
    a = ARMS[arm]
    done = f'{RUN_DIR}/{arm}.done'
    if os.path.exists(done):
        print(f'{arm}: finished {open(done).read().strip()}; delete {done} to run it again')
        return
    log = f'{RUN_DIR}/{arm}.log'
    init = BASE_CKPT
    if a['grow']:
        init = f'{WORK}/grown_{arm}.pt'
        if not os.path.exists(init):           # ~1 min on CPU, with a one-window exactness check
            rc = tee([sys.executable, '-u', 'validation/grow_base.py', '--ckpt', BASE_CKPT, '--cache', CORPUS,
                      *a['grow'], '--sources', 'wikipedia', '--windows', '1',
                      '--save', init, '--tag', f'colab_{arm}'], {}, log)
            assert rc == 0 and os.path.exists(init), f'growing {arm} failed: see {log}'
    local = f'{WORK}/{arm}'
    env = dict(BASE, **a['shape'], CB_MICRO=a['micro'], CB_INIT_FROM=init, CB_LOCAL_DIR=local,
               CB_NAME=f'grow_{arm}', CB_HOURS=f'{ARM_HOURS:.2f}', CB_WALL_H=f'{ARM_HOURS + 0.5:.2f}')
    t0 = time.time()
    rc = tee([sys.executable, '-u', 'validation/train_base.py'], env, log)
    m = f'{local}/grow_{arm}_metrics.jsonl'
    if os.path.exists(m):
        shutil.copy(m, f'{RUN_DIR}/{arm}_metrics.jsonl')
    for f in glob.glob(f'{local}/grow_{arm}_latest.pt*'):
        os.remove(f)                           # optimizer state (up to ~21 GB), not needed after the arm
    final = f'{local}/grow_{arm}_final.pt'
    if rc == 0 and os.path.exists(final):
        with open(done, 'w') as f:
            f.write(f"{time.strftime('%F %T')} ({(time.time() - t0) / 3600:.2f} h)")
        print(f'{arm}: done; weights kept on local disk at {final}')
    else:
        print(f'{arm}: did not finish (exit {rc}); rerun this cell to run it again')
""")

BENCH_MD = md(r"""
### Optional preflight: the grown shapes' speed and memory (~10 min each)

The micro-batches in `ARMS` are estimates: 8 for the grown shapes, from the base's measured 23.6 GB at micro 8 scaled by their size. If a grown arm runs out of memory, set its `micro` to the largest this bench fits, rerun the config cell, then rerun the arm. Tokens per step stay at 131k either way.
""")

BENCH = code(r"""
for arm in ('width2', 'depth2_ffn2'):
    tee([sys.executable, '-u', 'validation/train_base.py'],
        dict(BASE, **ARMS[arm]['shape'], CB_MODE='bench', CB_BENCH_MICRO='4,8,12,16'),
        f'{RUN_DIR}/bench_{arm}.log')
""")

ARMS_MD = md(r"""
### The four arms (~1 hour each)

One cell per arm. If a session ends mid-arm, rerun that arm's cell; finished arms are skipped. The control goes first: it is the bar the others are read against.
""")

ANALYSIS = code(r"""
# --- read the arms: equal wall-clock, equal FLOPs, projection, verdict (writes summary.json to Drive) ---
import matplotlib.pyplot as plt
TPS, H100_X, TARGET_H, BASE_VAL = 131072, 3.2, 20.0, 2.3615   # H100_X: the acceleration note's estimate
res = {}
for arm in ARMS:
    p = f'{RUN_DIR}/{arm}_metrics.jsonl'
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p)]
    tr = [r for r in rows if 'loss' in r]
    va = [r for r in rows if 'val' in r]
    sized = [r for r in rows if r.get('event') == 'sized']
    if not va:
        continue
    ds = (sized[-1]['decay_start'] if sized            # sized by the hours; else by CB_STEPS
          else int(va[-1]['step'] * (1 - float(BASE['CB_DECAY_FRAC']))))
    txt = open(f'{RUN_DIR}/{arm}.log', encoding='utf-8').read()
    mp = re.findall(r'params ([\d.]+)M \(trainable.*?\| ([\d.]+) GFLOP/token', txt)
    params, gflop = (float(mp[-1][0]), float(mp[-1][1])) if mp else (float('nan'), float('nan'))
    speeds = [r['tok_s'] for r in tr if r.get('phase') == 'stable']
    tok_s = sum(speeds) / len(speeds) if speeds else float('nan')
    hours = lambda s: max([r['hours'] for r in tr if r['step'] <= s] or [0.0])
    stable = [r for r in va if int(BASE['CB_WARMUP']) < r['step'] <= ds]
    half = stable[len(stable) // 2:]
    a = b = float('nan')
    if len(half) >= 2:                                  # val = a + b ln(tokens), last half of the stable phase
        xs = [math.log(r['step'] * TPS) for r in half]; ys = [r['val']['_mix'] for r in half]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
    final = va[-1]['val']['_mix']
    gain = final - (stable[-1]['val']['_mix'] if stable else final)
    t20 = tok_s * H100_X * TARGET_H * 3600
    res[arm] = dict(params_M=params, gflop_per_token=gflop,
                    tok_s=round(tok_s) if math.isfinite(tok_s) else 0, steps=va[-1]['step'],
                    tokens_B=va[-1]['step'] * TPS / 1e9, hours=hours(va[-1]['step']), final=final,
                    decay_gain=gain, slope_per_e=b, tokens_at_20_H100h_B=t20 / 1e9,
                    projected_20_H100h=a + b * math.log(t20) + gain,
                    per_source={k: v for k, v in va[-1]['val'].items() if not k.startswith('_')},
                    copy=va[-1]['val'].get('_copy'),
                    curve=[(r['step'] * TPS, hours(r['step']), r['val']['_mix']) for r in va])

print(f"{'arm':18s} {'params':>8s} {'tok/s':>8s} {'tokens':>7s} {'hours':>6s} {'final':>7s} "
      f"{'vs base':>8s} {'slope/e':>8s} {'proj 20 H100-h':>15s}")
for arm, r in res.items():
    print(f"{arm:18s} {r['params_M']:7.0f}M {r['tok_s']:8,d} {r['tokens_B']:6.3f}B {r['hours']:6.2f} "
          f"{r['final']:7.4f} {r['final'] - BASE_VAL:+8.4f} {r['slope_per_e']:+8.4f} {r['projected_20_H100h']:15.4f}")
srcs = sorted({k for r in res.values() for k in r['per_source']})
print('\nfinal held-out loss per source:')
print(f"{'source':16s}" + ''.join(f'{a:>18s}' for a in res))
for s in srcs:
    print(f'{s:16s}' + ''.join(f"{res[a]['per_source'].get(s, float('nan')):18.4f}" for a in res))

verdict = 'incomplete: the control has not finished'
if 'control' in res:
    c = res['control']
    grown = {k: v for k, v in res.items() if k != 'control'}
    now = [k for k, v in grown.items() if c['final'] - v['final'] >= 0.03]
    proj = [k for k, v in grown.items() if c['projected_20_H100h'] - v['projected_20_H100h'] >= 0.03]
    print()
    for k, v in grown.items():
        print(f"{k}: {c['final'] - v['final']:+.4f} nats below the control at equal wall-clock; "
              f"{c['projected_20_H100h'] - v['projected_20_H100h']:+.4f} by projection at 20 H100-hours")
    if now or proj:
        key = lambda k: grown[k]['projected_20_H100h'] if math.isfinite(grown[k]['projected_20_H100h']) else 9e9
        pick = min(grown, key=key)
        verdict = (f'GROW: pick {pick} (lowest projected loss at 20 H100-hours). '
                   f'>= 0.03 below the control now: {now or "none"}; by projection: {proj or "none"}')
    elif grown:
        verdict = 'KILL growth for now: no grown arm is 0.03 nats below the control, at the end or by projection'
print('\nverdict:', verdict)
with open(f'{RUN_DIR}/summary.json', 'w') as f:
    json.dump({'verdict': verdict, 'base_val_mix': BASE_VAL, 'arm_hours': ARM_HOURS, 'h100_x': H100_X,
               'arms': res}, f, indent=1)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
for arm, r in res.items():
    tok, hrs, val = zip(*r['curve'])
    ax[0].plot(hrs, val, label=arm)
    ax[1].plot([t * r['gflop_per_token'] * 1e9 for t in tok], val, label=arm)
    ax[2].plot([t / 1e9 for t in tok], val, label=arm)
for x, t in zip(ax, ('equal wall-clock (hours)', 'equal compute (FLOPs)', 'tokens (B)')):
    x.axhline(BASE_VAL, color='k', lw=.6, ls='--'); x.set(xlabel=t, ylabel='held-out mix loss'); x.legend(fontsize=8)
ax[1].set_xscale('log')
plt.tight_layout(); plt.show()
""")

READ_MD = md(r"""
### How to read

- **Every arm first rises.** The base ended its decay at an LR near zero, and each arm re-warms it to 6e-4, which kicks the loss up before it falls; that is the control too. The dashed line is the base's final held-out loss (2.3615). The question is which arm ends lowest after its own decay, not whether an arm beats the base within an hour.
- **Equal wall-clock** (left) is the pre-registered comparison: each arm had the same hour, so a bigger model sees fewer tokens. **Equal compute** (middle) shows whether a grown arm learns more per FLOP. The **projection** fits each arm's stable phase as a + b·ln(tokens), extends it to the tokens 20 H100-hours would give that arm, and adds its own decay gain. It is an indicator for choosing the long run, not a result.
- **The plain-copy arm** starts 1.86 nats worse (step 0). If it catches the zero-exit arm within the hour, G_stack's finding (copies learn faster than identities) holds here too, and the long run can use it.
- **Per source**: FineWeb-Edu, books, QA and code are what the talk and program adapters lean on. A grown arm that wins the mix but loses code is worth a second look.
- The summary, logs and metrics are in `cubbyllm/runs/grow_pilot/` on Drive (`summary.json`, `<arm>.log`, `<arm>_metrics.jsonl`). They can be read from there directly, without copying anything out of this session.
""")

CELLS = [INTRO, SETUP, PATHS, CONFIG, BENCH_MD, BENCH, ARMS_MD,
         arm_cell('control'), arm_cell('width2'), arm_cell('depth2_ffn2'), arm_cell('depth2_ffn2_copy'),
         ANALYSIS, READ_MD]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True, "machine_shape": "hm",
                             "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name} ({len(CELLS)} cells)")
