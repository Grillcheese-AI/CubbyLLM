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

print("done.")
