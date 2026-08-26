"""Generate the Colab validation notebooks from a clean spec.

One notebook per validation, consistent structure: intro -> setup -> config ->
run -> how-to-read. Editing the notebooks by hand is error-prone JSON; editing
THIS is not. Re-run `python notebooks/_build.py` after changing a cell.

Each notebook bakes in the lessons this repo paid for:
  * corpus staged to LOCAL SSD (Drive-FUSE reads bottleneck the GPU: 3k vs 100k
    tok/s, measured 2026-08-04),
  * real-corpus LR ~3e-4 WITH warmup (3e-3 cold diverges in 3 steps),
  * backbone recorded in the checkpoint so eval reconstructs the trained model.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def write(name, *cells):
    nb = {"cells": list(cells),
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  wrote {name} ({len(cells)} cells)")


# ── shared cells ────────────────────────────────────────────────────────────
SETUP = code("""
# --- setup: clone repo, deps, mount Drive (run once per session) ---
import os, subprocess, sys, time
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!pip -q install torch numpy sentencepiece
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

PATHS = code("""
# --- EDIT to your Drive locations ---
DRIVE     = '/content/drive/MyDrive/cubbyllm'
TOKENIZER = f'{DRIVE}/grillcheese_bbpe128k.json'   # your tokenizer .json/.model
CORPUS_DRIVE = f'{DRIVE}/token_cache'              # uint32 shards on Drive
CORPUS    = '/content/token_cache'                 # staged to LOCAL SSD

# Stage the corpus to local SSD. Drive-FUSE memmap reads bottleneck the GPU —
# this run measured 3,000 tok/s off Drive vs ~100,000 tok/s off local disk.
# Skip (comment out) if CORPUS is already populated this session.
!mkdir -p {CORPUS} && rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
!du -sh {CORPUS}
""")


def runner(extra=""):
    return code(f"""
# --- run a script, tee to a log, catch a hung child on interrupt ---
def run(script, env_extra, log_name):
    env = dict(os.environ, CUBBY_SPM=TOKENIZER, CB_CORPUS=CORPUS, **env_extra)
    os.makedirs(f'{{REPO}}/validation/logs', exist_ok=True)
    log = f'{{REPO}}/validation/logs/{{log_name}}'
    p = None
    try:
        with open(log, 'a', encoding='utf-8', buffering=1) as f:
            f.write(f"\\n=== {{time.strftime('%F %T')}} "
                    + ' '.join(f'{{k}}={{v}}' for k, v in sorted(env_extra.items())) + '\\n')
            p = subprocess.Popen([sys.executable, '-u', f'validation/{{script}}'],
                                 cwd=REPO, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end=''); f.write(line)
            p.wait()
    finally:
        if p and p.poll() is None:      # never leave a child holding the GPU
            p.terminate()
{extra}""")


# ── 1. H-D3 backbone A/B ────────────────────────────────────────────────────
write(
    "hd3_backbone_ab.ipynb",
    md("""
# H-D3 — Windowed-attention hybrid vs pure MinGRU (needle recall)

**Claim.** The backbone bake-off (H-D1) chose MinGRU on *bpc*, which is blind to
content-based retrieval. On an induction task the toy showed pure MinGRU fails at
distance while a windowed-attention hybrid succeeds *at bounded state*. This
notebook tests it at real scale: two runs identical except `CB_BACKBONE`, scored
on needle recall across lengths.

**Read the needle table, not the training log** — both arms look similar on bpc /
generations; the difference lives only in recall at distance. The windowed hybrid
should recall where the needle is within its window, and fall back to MinGRU
beyond it (the honest bounded-state ceiling).
"""),
    SETUP,
    PATHS,
    code("""
# --- shared config: identical for both arms except CB_BACKBONE ---
BASE = dict(
    CB_D='512', CB_L='8', CB_HEADS='8',          # 512/8=64 per head (even -> RoPE ok)
    CB_ATTN_EVERY='3', CB_WINDOW='512',          # hybrid-only; mingru ignores them
    CB_B='32', CB_S='1024', CB_STEPS='30000',    # 0.98B tok/arm, ~9.8 tok/param
    CB_LR='3e-4', CB_WARMUP='2000',              # real-corpus LR + warmup (3e-3 diverges)
    CB_EVAL='20', CB_GRAD_CKPT='1',
    CB_HEALTH='500', CB_GEN='500', CB_GEN_PROMPT='1',
)
"""),
    runner(),
    code("""
# --- ARM 1: pure MinGRU baseline (~2.6h) ---
run('train_colab.py',
    dict(BASE, CB_BACKBONE='mingru', CB_CKPT=f'{DRIVE}/ab_mingru.pt'),
    'ab_mingru.log')
"""),
    code("""
# --- ARM 2: windowed hybrid — one flag different (~2.6h) ---
run('train_colab.py',
    dict(BASE, CB_BACKBONE='hybrid', CB_CKPT=f'{DRIVE}/ab_hybrid.pt'),
    'ab_hybrid.log')
"""),
    code("""
# --- NEEDLE EVAL: both checkpoints, lengths spanning the window ---
for name in ('mingru', 'hybrid'):
    print(f'\\n########## NEEDLE: {name} ##########', flush=True)
    run('exp_needle_recall.py',
        dict(CB_CKPT=f'{DRIVE}/ab_{name}.pt', CB_LENGTHS='256,512,1024,4096'),
        f'needle_{name}.log')
"""),
    md("""
### How to read

Each arm prints **TRAINED** vs two controls (**shuffled** needle, **untrained**
model), PMI-scored, with an argmax-spread check. Compare TRAINED against its own
shuffled control — never against 100%.

- **Hybrid recalls where MinGRU is at its shuffled floor** (short lengths / shallow
  depths, needle within ~window of the query) → the deployable win. H-D3 goes
  from toy-VERIFIED to scale-VERIFIED, and the hybrid earns becoming the default.
- **Both fall to chance at 4096-deep** (needle far beyond the window) → the honest
  bounded-state ceiling, the limit the 1M-context claim must respect.
- **No separation anywhere** → the toy didn't transfer; pure MinGRU stands.
"""),
)

# ── 2. Induction probe (the toy behind H-D3) ────────────────────────────────
write(
    "induction_probe.ipynb",
    md("""
# Induction probe — pure recurrence vs hybrid, on the canonical task

The toy that motivated H-D3. Dense repeated-sequence induction (`[block | block]`,
predict the second half), scored on **solve-rate** and **steps-to-solve** over
multiple seeds — a single run is a coin flip (grokking), so it must be averaged.

Arms: `mingru` (0 attn), `whybrid` (windowed attn — bounded state, deployable),
`hybrid` (full attn — unbounded, reference only). Sweep `CB_S` across `2*window`
to find where the windowed arm falls back to MinGRU. No corpus/tokenizer needed.
"""),
    SETUP,
    runner(),
    code("""
# --- one distance point: T = CB_S/2, window = CB_WINDOW ---
# within window (T < window) -> whybrid solves; beyond -> it collapses to mingru.
run('exp_d3_induction.py',
    dict(CB_S='128', CB_WINDOW='96', CB_SEEDS='4', CB_STEPS='8000',
         CB_D='128', CB_L='6', CB_ATTN_EVERY='3', CB_LR='3e-3', CB_LOG='500'),
    'induction_T64.log')
"""),
    code("""
# --- beyond window: T=256 > window=96 -> whybrid must collapse to mingru ---
run('exp_d3_induction.py',
    dict(CB_S='512', CB_WINDOW='96', CB_SEEDS='4', CB_STEPS='8000',
         CB_D='128', CB_L='6', CB_ATTN_EVERY='3', CB_LR='3e-3', CB_LOG='500'),
    'induction_T256.log')
"""),
    md("""
### How to read

- **`solve rate`** — fraction of seeds reaching ≥90% second-half accuracy.
- **`median steps`** — how fast it groks (the discriminator when both solve).
- Within the window: `whybrid` and `hybrid` solve, `mingru` fails at distance.
- Beyond the window: `whybrid` collapses to `mingru` (a window can't reach past
  itself); only full `hybrid` still solves — the bounded-state limit, made visible.
- **HARNESS FAILURE** prints only if *nothing* trains (raise `CB_STEPS`/`CB_LR`).
  A run where `mingru`/`whybrid` solve and `attn` lags is normal — pure attention
  is the *hardest* arm here (it must learn the prev-token head from scratch).
"""),
)

# ── 3. Decode throughput (the O(1) inference claim) ─────────────────────────
write(
    "decode_throughput.ipynb",
    md("""
# Decode throughput — is the O(1)-state inference advantage real?

`CubbyModel.step()` carries a fixed-size state (MinGRU) or a windowed KV cache
(hybrid) — bounded either way, unlike a growing KV cache. This measures it.

- **Arm A** (always): incremental `step()` vs naive full-prefix recompute, same
  model. The `step()` ms/tok should stay ~flat as context grows; naive climbs.
- **Arm B** (optional): vs a real peer (Qwen2.5-1.5B). Expect to lose at short
  context (Python loop vs fused kernels); the *crossover* is the finding.
"""),
    SETUP,
    PATHS,
    runner(),
    code("""
# --- Arm A only (no download). Point CB_CKPT at a trained checkpoint. ---
run('exp_decode_throughput.py',
    dict(CB_CKPT=f'{DRIVE}/ab_hybrid.pt', CB_CTXS='512,2048,8192,32768'),
    'decode_hybrid.log')
"""),
    code("""
# --- Arm B: vs Qwen2.5-1.5B (downloads ~3GB, needs transformers) ---
!pip -q install transformers
run('exp_decode_throughput.py',
    dict(CB_CKPT=f'{DRIVE}/ab_hybrid.pt', CB_CTXS='512,2048,8192,32768',
         CB_PEER='Qwen/Qwen2.5-1.5B'),
    'decode_vs_peer.log')
"""),
    md("""
### How to read
Arm A: a flat `step()` ms/tok across contexts **is** the O(1) claim, measured;
the speedup over naive grows with context. Arm B: losing at short context is
expected — report the crossover length, not the headline.
"""),
)

# ── 4. FFN spectrum (the low-rank gate) ─────────────────────────────────────
write(
    "ffn_spectrum.ipynb",
    md("""
# FFN spectrum — is there learned low-rank structure to exploit?

SwiGLU is ~40% of the params, so a low-rank / hypernet FFN is the biggest
structural lever left. cubby-lm's `ffn_compression/NOTES.md` says don't run that
bake-off blind: gate on whether the trained FFN's singular values have a **sharp
knee** (low-rank viable) or a **fat spectrum** (skip). This measures it on an
existing checkpoint — minutes, no training. Each trained matrix is compared to
its own init, since a decaying curve looks compressible no matter what.
"""),
    SETUP,
    PATHS,
    runner(),
    code("""
run('exp_ffn_spectrum.py', dict(CB_CKPT=f'{DRIVE}/ab_mingru.pt'), 'ffn_spectrum.log')
"""),
    md("""
### How to read
`r90/R` = fraction of directions holding 90% of the energy, **trained vs its own
init**. Fat spectrum (trained ≈ init, near full-rank) → skip the low-rank arm,
spend on ternary (bytes, ~free) or MoE. Sharp knee well below init → low-rank FFN
is worth a matched-budget A/B. Ternary is a *bytes* play, unaffected either way.
"""),
)

# ── 5. Multi-horizon pilot arm (H-P6, Group P) ──────────────────────────────
# Paths cell shared by the Group P pilots: the full token cache if present,
# else the mirrored Wikipedia cache, else the pilot script tokenizes the jsonl.
P_PATHS = code("""
# --- EDIT to your Drive locations ---
!pip -q install tokenizers                          # the 128k BBPE is an HF tokenizer .json
DRIVE     = '/content/drive/MyDrive/cubbyllm'
TOKENIZER = f'{DRIVE}/grillcheese_bbpe128k.json'
CORPUS_DRIVE = f'{DRIVE}/token_cache'               # full uint32 cache, if you still have it
JSONL_DIR    = f'{DRIVE}/wikipedia'                 # domain-tagged jsonl (fallback corpus)
CACHE_MIRROR = f'{DRIVE}/token_cache_wiki'          # where the tokenized jsonl gets mirrored
CORPUS    = '/content/token_cache'                  # LOCAL SSD (Drive-FUSE random reads crawl)
import os, glob
os.makedirs(CORPUS, exist_ok=True)
if glob.glob(f'{CORPUS_DRIVE}/*.u32'):
    !rsync -a --info=progress2 {CORPUS_DRIVE}/ {CORPUS}/
    print('staged the full token cache')
elif glob.glob(f'{CACHE_MIRROR}/*.u32'):
    !rsync -a --info=progress2 {CACHE_MIRROR}/ {CORPUS}/
    print('staged the mirrored Wikipedia cache')
else:
    print('no cache on Drive — the script will tokenize', JSONL_DIR, 'once and mirror it to', CACHE_MIRROR)
!du -sh {CORPUS}
""")

write(
    "multihorizon_pilot.ipynb",
    md("""
# H-P6 — Multi-horizon prediction head: a pilot arm (Group P)

**Why this exists.** The Group P checkpoint arms (`validation/prospection/`,
`CUBBYLLM_HYPOTHESES.md` §9b) found that after ~1B tokens the hybrid's recurrent
layers have median time constants of 0.5–1.8 steps — the recurrence is a local
mixer, and every long-range dependency lives in the windowed attention. So the
"short / medium / long-term outcome" horizons of the branching-futures thread
map onto the mechanisms that exist: recurrence → attention window → episodic
store. This notebook is the **medium** piece: an auxiliary head that predicts,
from the trunk feature at each position, the discounted average of the *future*
tokens' embeddings at four time constants (mean horizons 1 / 8 / 50 / 500
tokens — successor-feature style), trained next to plain next-token CE.

**Two matched arms**, identical except `CB_MH_W`:

| arm | what | kill |
|---|---|---|
| `baseline` | next-token CE only | — |
| `multihorizon` | + the head at weight 0.1 | held-out CE worse than baseline by more than noise → the head is not free; skill only at horizon 1 → the trunk cannot express the medium-term future at this scale |

Both arms also get: a **linear probe** on frozen features (how much future
information the trunk carries *anyway*), the **H-P3 gate spectrum** (does an
explicit long-horizon objective lengthen any recurrent time constant?), and the
representation-health / copy-floor gates that caught the H-B6 Goodhart.

~15 min per arm on an A100 at the default shape (d=512, L=8, 2000 steps ≈ 65M
tokens). Corpus: the `token_cache` on Drive if present, else the domain-tagged
Wikipedia jsonl next to the checkpoints is tokenized once (~200M tokens, a few
minutes) and mirrored back to Drive for the next session.
"""),
    SETUP,
    P_PATHS,
    runner(),
    code("""
# --- shared config: identical for both arms except CB_MH_W ---
# A100: ~15 min/arm. T4: set CB_B='16', CB_S='512' and expect ~1h/arm.
BASE = dict(
    CB_D='512', CB_L='8', CB_HEADS='8', CB_BACKBONE='hybrid',
    CB_ATTN_EVERY='3', CB_WINDOW='512',
    CB_B='32', CB_S='1024', CB_STEPS='2000',          # 32k tok/step, 65M tokens/arm
    CB_LR='3e-4', CB_WARMUP='200',                    # real-corpus LR + warmup (3e-3 diverges)
    CB_EVAL='50', CB_HEALTH='500', CB_CKPT_EVERY='0',   # save ONCE at the end (see README lesson)
    CB_MH_GAMMAS='0,0.875,0.98,0.998',                # mean horizons 1 / 8 / 50 / 500 tokens
    CB_MH_PROBE_STEPS='300', CB_MH_EVAL_BATCHES='16',
    CB_JSONL_DIR=JSONL_DIR, CB_CACHE_MIRROR=CACHE_MIRROR,
)
LOGS = f'{REPO}/validation/logs'
"""),
    code("""
# --- ARM 1: baseline (next-token CE only) ---
run('prospection/exp_p6_multihorizon_pilot.py',
    dict(BASE, CB_MH_W='0', CB_MH_TAG='baseline',
         CB_CKPT=f'{DRIVE}/mh_baseline.pt', CB_MH_OUT=f'{LOGS}/exp_p6_baseline.json'),
    'exp_p6_baseline.log')
"""),
    code("""
# --- ARM 2: + multi-horizon head — one flag different ---
# 2026-08-26 run used 0.03 (a 0.1 attempt was stopped at ~500 steps; its head
# cosines at step 500 matched 0.03's within 0.01, so the weight barely moves the
# head's skill — the CE cost is what a higher weight would have to justify).
run('prospection/exp_p6_multihorizon_pilot.py',
    dict(BASE, CB_MH_W='0.03', CB_MH_TAG='multihorizon',
         CB_CKPT=f'{DRIVE}/mh_multihorizon.pt', CB_MH_OUT=f'{LOGS}/exp_p6_multihorizon.json'),
    'exp_p6_multihorizon.log')
"""),
    code("""
# --- COMPARE the two arms, and keep the evidence on Drive ---
import json, shutil, math
R = {a: json.load(open(f'{LOGS}/exp_p6_{a}.json')) for a in ('baseline', 'multihorizon')}
b, m = R['baseline'], R['multihorizon']
hz = [round(1/(1-g)) if g < 1 else 'inf' for g in b['gammas']]
print(f"tokens/arm {b['tokens']:,}  |  steps {b['steps']} / {m['steps']}")
print(f"held-out CE   baseline {b['heldout_ce']:.4f}  multihorizon {m['heldout_ce']:.4f}  "
      f"delta {m['heldout_ce']-b['heldout_ce']:+.4f} nats ({(m['heldout_ce']-b['heldout_ce'])/math.log(2):+.4f} bits/tok)")
print(f"health        baseline ret {b['retrieval']:.1%} copy f{b['copy_freq']:.0%}   "
      f"multihorizon ret {m['retrieval']:.1%} copy f{m['copy_freq']:.0%}")
row = lambda name, vals: print(f"{name:<34}" + ''.join('      n/a' if v is None else f'{v:9.3f}' for v in vals))
print('horizon (mean tokens)' + ''.join(f'{str(h):>9}' for h in hz))
row('trivial baseline (mean dir)', b['cos_mean_baseline'])
row('probe on frozen h — baseline', b['cos_probe'])
row('probe on frozen h — multihorizon', m['cos_probe'])
row('trained head — multihorizon', m['cos_head'])
for a, r in R.items():
    print(f"gate spectrum {a:<13}", ', '.join(f"L{i}: p50 {v['p50']:.1f} slow8 {v['slow8']:.2f}"
                                           for i, v in r['gate_spectrum'].items()))
os.makedirs(f'{DRIVE}/logs', exist_ok=True)
for f in glob.glob(f'{LOGS}/exp_p6_*'):
    shutil.copy2(f, f'{DRIVE}/logs/')
print('copied logs + json to', f'{DRIVE}/logs/')
"""),
    code("""
# --- OPTIONAL: the Group P checkpoint arms on the new checkpoints (P2 entropy budget, P3 spectrum) ---
# Check the step FIRST: the 2026-08-26 baseline file on Drive turned out to be
# the step-500 periodic save, not the final one (Drive sync race) — its P2/P3
# numbers described a different model than the JSON.
import torch
for a in ('baseline', 'multihorizon'):
    print(a, 'checkpoint step =', torch.load(f'{DRIVE}/mh_{a}.pt', map_location='cpu', mmap=True)['step'])
!cd {REPO} && python validation/prospection/make_text_sample.py {JSONL_DIR} /content/wiki_sample.txt
for a in ('baseline', 'multihorizon'):
    for script in ('exp_p2_choice_points.py', 'exp_p3_gate_spectrum.py'):
        run(f'prospection/{script}',
            dict(CB_CKPT=f'{DRIVE}/mh_{a}.pt', CB_TEXT='/content/wiki_sample.txt'),
            f"{script[:-3]}_mh_{a}.log")
"""),
    md("""
### How to read

Three numbers decide it, in this order:

1. **Δ held-out CE** (multihorizon − baseline). Within ±0.01 nats at 65M tokens
   is noise: the head is free, keep it in the 2B runbook as an arm. Clearly
   positive: the head taxes the trunk at this scale; try `CB_MH_W='0.03'` before
   dropping it. Negative: a regularization win — worth a second seed before
   believing it.
2. **Skill by horizon** = cosine minus the trivial baseline. Trained head *and*
   probe should be well above the trivial row at horizons 8 and 50; that is the
   medium-term future being expressible from the trunk. If only horizon 1 has
   skill, the successor-feature design is dead at this scale. Probe-multihorizon
   above probe-baseline means the objective changed the representation, not
   just added a readout.
3. **Gate spectrum.** If the multihorizon arm's recurrent `p50`/`slow8` move
   above the baseline's (~1–3 steps, ~0), an explicit long-horizon objective
   *does* lengthen the recurrence — the first evidence for the "made, not
   inherited" route of H-P3. If they don't move, the horizons live in attention
   as measured, and that is the design.

`ret` must stay ~95%+ and `copy f` must be non-zero by the end for either arm's
numbers to mean anything (the H-B6 gates). Record the verdict in
`CUBBYLLM_HYPOTHESES.md` H-P6 with the log links.
"""),
)

# ── 6. Chrono-init pilot arm (H-P7, Group P) ────────────────────────────────
write(
    "chrono_pilot.ipynb",
    md("""
# H-P7 — Chrono init on the recurrent gates: a pilot arm (Group P)

**Why this is the last open route.** H-P3 measured the trained hybrid's
recurrence at 0.5–2 steps per unit, shortening monotonically from init (2.8)
through step 500 (2.0), step 2000 (1.2–2.1) and ~1B tokens (0.5–1.8); the gate
caps τ at ~1000. H-P6 then closed the "make it with an objective" route — an
explicit 500-token prediction loss moved no recurrent time constant. What's left
is setting the spectrum at **init** (Tallec & Ollivier's chrono init, τ
log-uniform over decades) and asking whether training keeps it when attention is
there to do the long-range work instead.

**Two matched arms**, identical except `CB_CHRONO`:

| arm | init | kill |
|---|---|---|
| `baseline` | proj_d bias 1.0 → every unit τ ≈ 2.8 | — |
| `chrono` | τ log-uniform in [1, 500] on every recurrent layer | held-out CE worse beyond ~0.01 nats; **or** the trained spectrum collapses onto the baseline's (survival fails) |

Four readings, in order: Δ held-out CE · **survival** (gate spectrum at init vs
after training, per layer) · **beyond-window memory** (impulse response of the
recurrent state to one substituted token, at lags past the 512 window — only the
recurrence can carry it there; baseline reads 0.000) · **capability** (needle
recall at 256 / 512 / 1024 / 4096, TRAINED vs its own shuffled control).

~16 min per arm on an A100, plus a few minutes of needle eval per checkpoint.
"""),
    SETUP,
    P_PATHS,
    runner(),
    code("""
# --- shared config: identical for both arms except CB_CHRONO ---
BASE = dict(
    CB_D='512', CB_L='8', CB_HEADS='8', CB_BACKBONE='hybrid',
    CB_ATTN_EVERY='3', CB_WINDOW='512',
    CB_B='32', CB_S='1024', CB_STEPS='2000',          # 32k tok/step, 65M tokens/arm
    CB_LR='3e-4', CB_WARMUP='200',
    CB_EVAL='50', CB_HEALTH='500', CB_CKPT_EVERY='0',   # save ONCE at the end, staged + verified
    CB_MH_W='0', CB_MH_PROBE_STEPS='300', CB_MH_EVAL_BATCHES='16',
    CB_CHRONO_TMIN='1', CB_CHRONO_TMAX='500', CB_CHRONO_WSCALE='1.0',   # bias-only = the paper's form
    CB_IMPULSE_LAGS='1,2,4,8,16,32,64,128,256,512,768,1000',            # past 512 = recurrence only
    CB_JSONL_DIR=JSONL_DIR, CB_CACHE_MIRROR=CACHE_MIRROR,
)
LOGS = f'{REPO}/validation/logs'
"""),
    code("""
# --- ARM 1: baseline (default gate init) ---
run('prospection/exp_p7_chrono_pilot.py',
    dict(BASE, CB_CHRONO='0', CB_MH_TAG='baseline',
         CB_CKPT=f'{DRIVE}/chrono_baseline.pt', CB_MH_OUT=f'{LOGS}/exp_p7_baseline.json'),
    'exp_p7_baseline.log')
"""),
    code("""
# --- ARM 2: chrono init — one flag different ---
run('prospection/exp_p7_chrono_pilot.py',
    dict(BASE, CB_CHRONO='1', CB_MH_TAG='chrono',
         CB_CKPT=f'{DRIVE}/chrono_arm.pt', CB_MH_OUT=f'{LOGS}/exp_p7_chrono.json'),
    'exp_p7_chrono.log')
"""),
    code("""
# --- COMPARE: CE, survival, beyond-window memory; keep the evidence on Drive ---
import json, shutil, math, torch
R = {a: json.load(open(f'{LOGS}/exp_p7_{a}.json')) for a in ('baseline', 'chrono')}
b, c = R['baseline'], R['chrono']
print(f"tokens/arm {b['tokens']:,}  |  steps {b['steps']} / {c['steps']}")
print(f"held-out CE   baseline {b['heldout_ce']:.4f}  chrono {c['heldout_ce']:.4f}  "
      f"delta {c['heldout_ce']-b['heldout_ce']:+.4f} nats")
print(f"health        baseline ret {b['retrieval']:.1%} copy f{b['copy_freq']:.0%}   "
      f"chrono ret {c['retrieval']:.1%} copy f{c['copy_freq']:.0%}")
print('\\nSURVIVAL — gate spectrum per recurrent layer (p50 tau | fraction >=8 | fraction >=100):')
for a, r in R.items():
    for when, key in (('init', 'gate_spectrum_init'), ('end ', 'gate_spectrum')):
        print(f"  {a:<9}{when}  " + '  '.join(f"L{i}: {v['p50']:5.1f} | {v['slow8']:.2f} | {v['slow100']:.2f}"
                                          for i, v in r[key].items()))
w = b['config']['window']
print(f"\\nBEYOND-WINDOW MEMORY — impulse response |dh|/|h| vs lag (window {w}):")
lags = list(b['impulse'].keys())
print('  lag      ' + ''.join(f"{k:>7}" for k in lags))
for a, r in R.items():
    print(f"  {a:<9}" + ''.join(f"{r['impulse'][k]:7.3f}" for k in lags))
beyond = [k for k in lags if int(k) > w]
print(f"  past the window {beyond}: baseline {[round(b['impulse'][k],3) for k in beyond]}  "
      f"chrono {[round(c['impulse'][k],3) for k in beyond]}")
for a in ('baseline', 'chrono'):
    p = f'{DRIVE}/chrono_{"arm" if a == "chrono" else "baseline"}.pt'
    print(a, 'checkpoint step =', torch.load(p, map_location='cpu', mmap=True)['step'])
os.makedirs(f'{DRIVE}/logs', exist_ok=True)
for f in glob.glob(f'{LOGS}/exp_p7_*'):
    shutil.copy2(f, f'{DRIVE}/logs/')
print('copied logs + json to', f'{DRIVE}/logs/')
"""),
    code("""
# --- CAPABILITY: needle recall on both checkpoints, within and beyond the window ---
for a, ck in (('baseline', 'chrono_baseline'), ('chrono', 'chrono_arm')):
    print(f'\\n########## NEEDLE: {a} ##########', flush=True)
    run('exp_needle_recall.py',
        dict(CB_CKPT=f'{DRIVE}/{ck}.pt', CB_LENGTHS='256,512,1024,4096'),
        f'needle_chrono_{a}.log')
for f in glob.glob(f'{LOGS}/needle_chrono_*'):
    shutil.copy2(f, f'{DRIVE}/logs/')
"""),
    md("""
### How to read

1. **Δ held-out CE** within ±0.01 nats → chrono is free at this scale. Worse →
   the long time constants cost prediction; the route is closed unless (3) or (4)
   pays for it.
2. **Survival.** The chrono arm starts with ~22% of units at τ ≥ 100. If the
   `end` row still shows a meaningful fraction ≥ 8 and ≥ 100, training kept the
   spectrum; if it reads like the baseline (p50 1–2, 0.00, 0.00), gradient descent
   eroded chrono the way it erodes the default init — **the route is closed** and
   the H-P3 conclusion stands as final: beyond-window memory lives in the store.
3. **Beyond-window impulse response.** Baseline reads 0.000 past 512 (nothing but
   the recurrence can carry a perturbation there, and its τ is ~2). A surviving
   chrono spectrum shows a non-zero tail. This is the mechanism; it can be
   non-zero even when (4) doesn't move.
4. **Needle at 1024 / 4096** (needle beyond the window), TRAINED vs its own
   shuffled control — never vs 100%. Both arms should match inside the window
   (attention's job). Chrono earns a place in the 2B runbook only if recall past
   the window rises above the floor. Survival without recall is still a recorded
   fact: the gate *can* hold a long time constant but has nothing useful to put
   in it at this scale.

Record the verdict in `CUBBYLLM_HYPOTHESES.md` H-P7 with the log links.
"""),
)

print("done.")
