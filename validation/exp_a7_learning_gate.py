"""exp_a7_learning_gate — H-A7: can a promotion gate composed from existing
harnesses tell a forgetting trunk update from a clean one?

THE CLAIM UNDER TEST (H-A7, added 2026-08-28). Weight updates from live data
should enter the trunk only through a nightly promotion rule whose every term
is an existing measurement. The composed rule the oracle-competition scoring
proposed had six terms; this script is the honest first pass at it, and its
first finding is which terms can see a trunk delta at all:

  term (as proposed)                     | what it is here
  ---------------------------------------+------------------------------------------
  general-capability regression (H-G4)  | held-out CE on the pretraining mix AND on
                                         | the GSM8K test text (socratic_test_text)
  dated-key forgetting vs NLMS (H-G2)    | NOT APPLICABLE to a trunk delta: that
                                         | harness benchmarks memory METHODS. Replaced
                                         | by per-source held-out CE deltas — the
                                         | trunk's own forgetting signal (a narrow
                                         | fine-tune must not raise CE on the sources
                                         | it did not see).
  in-window needle recall (H-D4)         | optional (CB_A7_NEEDLE=1): ~13 CPU-hours
                                         | for the full protocol; off by default.
  binding-channel retrieval (H-B6)       | the b6b probe: fixed roles, 16 fillers,
                                         | argmax retrieval on the trunk's h.
  claimed-answer precision (CoT)         | NOT APPLICABLE: model-free surface (word
                                         | table + VM); a trunk delta cannot move it.
  routing precision (margin gate)        | NOT APPLICABLE: same reason.

METHOD. Base = a trained checkpoint (default hd5_mem21.pt, 151M hybrid+mem,
step 92k). Three deltas, evaluated on IDENTICAL held-out streams:
  C  the null delta (no update)                       — must PASS
  A  a deliberately-forgetting update: CB_A7_STEPS of  — must be REJECTED
     fine-tuning on ONE source only (CB_A7_SLICE,
     default nemotron_code), no replay
  B  the matched clean update: same steps / LR on the  — must PASS
     full weighted mixture (= replay)
Thresholds are pre-registered from the NOISE, not tuned. Every delta is
scored against the base on the IDENTICAL held-out windows, so the honest noise
is the PAIRED one: for each CE metric the per-window differences
(delta - base) give a mean d and a standard error SE, and the term fires iff
d > max(min_eps, 3 x SE). (A first smoke used the unpaired seed-to-seed
difference — 0.29 nats on two batches — which is the wrong noise for a paired
comparison and would have hidden real forgetting.) Retrieval has no pairing;
its epsilon comes from the base on two differently-seeded streams,
max(0.05, 3 x |seed-to-seed|), and it must not fall below the larger of the
collapse floor 0.50 (H-B6's collapsed arm read 0.115) and base - eps.

  promote iff  dCE_mix     <= max(0.01, 3 SE)
           and max_s dCE_s <= max(0.03, 3 SE_s)   (every source; names the worst)
           and dCE_gsm8k   <= max(0.02, 3 SE)
           and retrieval   >= max(0.50, base_retrieval - eps_ret)

KILL (H-A7). The gate is not a gate if it cannot separate A from B — if it
passes A, or rejects B or C, at these pre-registered epsilons. Second kill:
if the full gate pass costs more than a nightly budget it must be tiered.

Standalone; never imported by cubbyllm/. Device: CB_DEVICE (auto -> cuda if
present; bf16 autocast for the fine-tunes on cuda). The env defaults below are
the CPU shape (~9 s/step at B4/S512 on 12 cores, ~50 min end to end, and a
session teardown killed it twice at step ~100 on 2026-08-28); the recorded run
is the Colab one — `notebooks/a7_learning_gate.ipynb` sets the GPU shape
(B32, 200 steps = 3.3M tokens per delta, 32 mix batches) and copies the
log + json to Drive.

  CB_CKPT=I:\\CUBBY-TRAINED-MODELS\\hd5_mem21.pt CB_CORPUS=I:\\grillcheese_training_data\\token_cache \\
  CB_SOURCES="D:\\My Drive\\cubbyllm\\corpus_sources.json" \\
  CUBBY_SPM=E:\\datasets\\_pipeline\\tokenizer\\grillcheese_bbpe128k.json \\
  python validation/exp_a7_learning_gate.py | tee validation/logs/exp_a7_learning_gate.log
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "validation")
for p in (ROOT, VAL):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ── configuration (env, all defaults = the 2026-08-28 first run) ───────────
CKPT = os.environ.get("CB_CKPT", r"I:\CUBBY-TRAINED-MODELS\hd5_mem21.pt")
CORPUS = os.environ.get("CB_CORPUS", r"I:\grillcheese_training_data\token_cache")
SOURCES = os.environ.get("CB_SOURCES", r"D:\My Drive\cubbyllm\corpus_sources.json")
SPM = os.environ.get("CUBBY_SPM", r"E:\datasets\_pipeline\tokenizer\grillcheese_bbpe128k.json")
GSM8K = os.environ.get("CB_GSM8K", r"I:\grillcheese_training_data\socratic_test_text.jsonl")
SAVE_DIR = os.environ.get("CB_A7_SAVE_DIR", r"I:\CUBBY-TRAINED-MODELS")
TAG = os.environ.get("CB_A7_TAG", "")
STEPS = int(os.environ.get("CB_A7_STEPS", "120"))
LR = float(os.environ.get("CB_A7_LR", "1e-4"))
WARMUP = int(os.environ.get("CB_A7_WARMUP", "10"))
BATCH = int(os.environ.get("CB_B", "4"))
SEQ = int(os.environ.get("CB_S", "512"))
SLICE = os.environ.get("CB_A7_SLICE", "nemotron_code")
MIX_BATCHES = int(os.environ.get("CB_A7_EVAL_BATCHES", "16"))
SRC_BATCHES = int(os.environ.get("CB_A7_SRC_BATCHES", "8"))
GSM_BATCHES = int(os.environ.get("CB_A7_GSM_BATCHES", "8"))
N_BIND = int(os.environ.get("CB_BIND_N", "16"))
RETRIEVAL_COLLAPSE_FLOOR = float(os.environ.get("CB_A7_RET_FLOOR", "0.50"))
MIN_EPS = {"mix": 0.01, "src": 0.03, "gsm": 0.02, "ret": 0.05}   # pre-registered minimum epsilons
NOISE_MULT = 3.0
EVAL_SEED_A, EVAL_SEED_B = 777, 779                   # two held-out streams for the noise floor
SAVE_DELTAS = os.environ.get("CB_A7_SAVE", "1") == "1"
DEVICE = os.environ.get("CB_DEVICE", "auto")
# LR-ladder decider (2026-08-30 result: the replay arm regressed uniformly at
# 1e-4 with a fresh Adam — its own train loss rose). CB_A7_ARMS picks which
# fine-tune arms run (A = slice/no replay, B = mixture/replay); C always runs.
# CB_A7_RESUME_OPT=1 restores the checkpoint's Adam state instead of a fresh
# optimizer, so the update continues the run rather than perturbing it.
ARMS = [a.strip().upper() for a in os.environ.get("CB_A7_ARMS", "A,B").split(",") if a.strip()]
RESUME_OPT = os.environ.get("CB_A7_RESUME_OPT", "0") == "1"


# ── the gate, as a pure function (unit-pinned in test_a7_gate.py) ──────────
def noise_floor(evalA: dict, evalB: dict, min_eps: dict = MIN_EPS, mult: float = NOISE_MULT) -> dict:
    """The UNPAIRED noise: two evaluations of the SAME model on two
    differently-seeded held-out streams. Used for the retrieval epsilon
    (retrieval has no per-window pairing) and reported for the CE metrics as
    information only — the CE terms use the paired SE in `paired_term`."""
    src_noise = max(abs(evalA["ce_by_source"][s] - evalB["ce_by_source"][s])
                    for s in evalA["ce_by_source"])
    return {
        "ret": max(min_eps["ret"], mult * abs(evalA["retrieval_acc"] - evalB["retrieval_acc"])),
        "unpaired_raw": {"mix": abs(evalA["ce_mix"] - evalB["ce_mix"]), "src_max": src_noise,
                         "gsm": abs(evalA["ce_gsm8k"] - evalB["ce_gsm8k"]),
                         "ret": abs(evalA["retrieval_acc"] - evalB["retrieval_acc"])},
    }


def paired_term(delta_windows, base_windows, min_eps: float, mult: float = NOISE_MULT) -> dict:
    """One CE term: per-window differences on identical windows -> mean d,
    standard error, threshold max(min_eps, mult x SE), fired iff d > threshold."""
    diff = np.asarray(delta_windows, dtype=np.float64) - np.asarray(base_windows, dtype=np.float64)
    n = int(diff.size)
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    d = float(diff.mean()) if n else 0.0
    thr = max(min_eps, mult * se)
    return {"d": d, "se": se, "n": n, "threshold": thr, "fired": d > thr}


def gate_decision(delta: dict, base: dict, eps: dict,
                  retrieval_floor: float = RETRIEVAL_COLLAPSE_FLOOR) -> dict:
    """The promotion rule. `delta`/`base` are evaluate() dicts. Returns
    promote + the list of terms that fired + every margin, so a rejection
    names its reason."""
    mix = paired_term(delta["ce_mix_windows"], base["ce_mix_windows"], MIN_EPS["mix"])
    src = {s: paired_term(delta["ce_src_windows"][s], base["ce_src_windows"][s], MIN_EPS["src"])
           for s in base["ce_src_windows"]}
    worst_src = max(src, key=lambda s: src[s]["d"] - src[s]["threshold"])
    gsm = paired_term(delta["ce_gsm_windows"], base["ce_gsm_windows"], MIN_EPS["gsm"])
    ret_floor = max(retrieval_floor, base["retrieval_acc"] - eps["ret"])
    fired = []
    if mix["fired"]:
        fired.append("general_capability_mix")
    fired += [f"forgetting_source:{s}" for s in base["ce_src_windows"] if src[s]["fired"]]
    if gsm["fired"]:
        fired.append("general_capability_gsm8k")
    if delta["retrieval_acc"] < ret_floor:
        fired.append("binding_health")
    return {
        "promote": not fired,
        "fired": fired,
        "margins": {"d_ce_mix": mix["d"], "eps_mix": mix["threshold"], "se_mix": mix["se"],
                    "d_ce_source_worst": src[worst_src]["d"], "worst_source": worst_src,
                    "eps_src": src[worst_src]["threshold"],
                    "d_ce_by_source": {s: src[s]["d"] for s in src},
                    "eps_by_source": {s: src[s]["threshold"] for s in src},
                    "d_ce_gsm8k": gsm["d"], "eps_gsm": gsm["threshold"], "se_gsm": gsm["se"],
                    "retrieval_acc": delta["retrieval_acc"], "retrieval_floor": ret_floor,
                    "retrieval_base": base["retrieval_acc"], "eps_ret": eps["ret"]},
        "not_applicable": ["nyt_forgetting_vs_nlms (memory-method benchmark, replaced by per-source CE)",
                           "claimed_answer_precision (model-free surface)",
                           "routing_precision (model-free surface)"],
    }


def separation_verdict(decisions: dict) -> dict:
    """H-A7's kill: the gate must reject A (forgetting), pass B (clean), pass C (null)."""
    ok_a = not decisions["A"]["promote"]
    ok_b = decisions["B"]["promote"]
    ok_c = decisions["C"]["promote"]
    return {"rejects_forgetting_A": ok_a, "passes_clean_B": ok_b, "passes_null_C": ok_c,
            "gate_separates": ok_a and ok_b and ok_c}


# ── model / data plumbing (reuses the needle script's exact model build) ───
def load_base(dev):
    from exp_needle_recall import build as build_model
    ck = torch.load(CKPT, map_location="cpu", mmap=True)
    meta = dict(ck["meta"])
    model, d, L, V, kind = build_model(meta, dev)
    ps = list(model.parameters())
    assert len(ps) == len(ck["params"]), (len(ps), len(ck["params"]))
    base_params = [s.detach().clone() for s in ck["params"]]
    restore(ps, base_params)          # the model is built at random init; load the checkpoint INTO it
    opt_state = ck.get("opt") if RESUME_OPT else None
    return model, ps, base_params, meta, int(ck.get("step", -1)), d, V, opt_state


def restore(ps, base_params):
    with torch.no_grad():
        for p, s in zip(ps, base_params):
            p.copy_(s.to(p.device))


def source_specs():
    """The pretrain mixture from CB_SOURCES (corpus_sources.json, weights and
    all) when every named shard is present in CB_CORPUS; otherwise the shards
    that ARE present at weight 1.0 (Colab staging may hold a different cache)
    -- printed, so a run on a different mixture is never silent."""
    import glob
    present = {os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(CORPUS, "*.u32"))}
    if os.path.exists(SOURCES):
        specs = json.load(open(SOURCES, encoding="utf-8"))["sources"]
        missing = [x["name"] for x in specs if x["name"] not in present]
        if not missing:
            return [{"name": x["name"], "weight": float(x.get("weight", 1.0))} for x in specs]
        print(f"  !! {len(missing)} sources named in {SOURCES} have no shard in {CORPUS}: {missing}\n"
              f"     falling back to the {len(present)} shards present at weight 1.0", flush=True)
    else:
        print(f"  !! {SOURCES} not found; using the {len(present)} shards in {CORPUS} at weight 1.0", flush=True)
    if not present:
        raise SystemExit(f"no *.u32 shards in {CORPUS}")
    return [{"name": n, "weight": 1.0} for n in sorted(present)]


def pipeline(sources, seed):
    from cubbyllm.training import WeightedCorpusPipeline
    return WeightedCorpusPipeline(sources, SPM, CORPUS, seed=seed, cache_only=True).prepare()


def gsm8k_windows(dev):
    """Fixed, deterministic windows over the GSM8K test text (H-G4): the first
    GSM_BATCHES x BATCH windows of SEQ+1 tokens of the concatenated examples."""
    from cubbyllm.training.data import _load_tokenizer
    encode, _dec, eos, _V = _load_tokenizer(SPM)
    need = GSM_BATCHES * BATCH * (SEQ + 1) + 1
    ids: list[int] = []
    with open(GSM8K, encoding="utf-8") as f:
        for line in f:
            ids.extend(encode(json.loads(line)["text"]))
            if eos >= 0:
                ids.append(eos)
            if len(ids) >= need:
                break
    arr = np.asarray(ids[:need], dtype=np.int64)
    n = GSM_BATCHES * BATCH
    win = np.stack([arr[i * (SEQ + 1):(i + 1) * (SEQ + 1)] for i in range(n)])
    x = torch.tensor(win[:, :-1], device=dev).view(GSM_BATCHES, BATCH, SEQ)
    y = torch.tensor(win[:, 1:], device=dev).view(GSM_BATCHES, BATCH, SEQ)
    return list(zip(x, y))


@torch.no_grad()
def ce_over(model, batches, keep_h: int = 0):
    """-> (mean CE over all tokens, per-window CE list, kept hidden states).
    Per-window CEs are what make the paired comparison possible."""
    tot, ntok, hs, per_win = 0.0, 0, [], []
    for x, y in batches:
        h = model.features(x)
        logits = model.logits_from(h)
        tok_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                                 reduction="none").view(y.shape)
        per_win.extend(tok_ce.mean(dim=1).tolist())
        tot += float(tok_ce.sum())
        ntok += int(y.numel())
        if len(hs) < keep_h:
            hs.append(h.float().cpu())
    return tot / ntok, per_win, hs


@torch.no_grad()
def retrieval_probe(hs: list, d: int, seed: int) -> dict:
    """b6b's binding diagnostic verbatim: bind N_BIND fillers (hidden states at
    random positions) to fixed roles, superpose, unbind, argmax-retrieve."""
    from cubbyllm.model.binding.torch_ops import make_roles
    h = torch.cat(hs, dim=0)
    n_seq, S, _ = h.shape
    roles = make_roles(N_BIND, d, h.device)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, S, (n_seq, N_BIND), generator=g)
    f = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d))
    r = roles.unsqueeze(0)
    rec = (r * f).sum(dim=1, keepdim=True) * r
    rn, fn = F.normalize(rec, dim=-1), F.normalize(f, dim=-1)
    cos = float((rn * fn).sum(-1).mean())
    sim = torch.bmm(rn, fn.transpose(1, 2))
    acc = float((sim.argmax(-1) == torch.arange(N_BIND)).float().mean())
    ff = float(torch.bmm(fn, fn.transpose(1, 2))[:, ~torch.eye(N_BIND, dtype=bool)].mean())
    return {"retrieval_acc": acc, "retrieval_cos": cos, "ff_cos": ff}


def evaluate(model, sources, d, dev, seed: int, gsm) -> dict:
    """All gate-visible measurements on one model, on seed-fixed streams."""
    t0 = time.perf_counter()
    mix_batches = pipeline(sources, seed).batches(BATCH, SEQ)
    mix = [next(mix_batches) for _ in range(MIX_BATCHES)]
    ce_mix, mix_w, hs = ce_over(model, [(x.to(dev), y.to(dev)) for x, y in mix], keep_h=8)
    by_src, src_w = {}, {}
    for s in sources:
        b = pipeline([{"name": s["name"], "weight": 1.0}], seed + 1).batches(BATCH, SEQ)
        by_src[s["name"]], src_w[s["name"]], _ = ce_over(
            model, [tuple(t.to(dev) for t in next(b)) for _ in range(SRC_BATCHES)])
    ce_gsm, gsm_w, _ = ce_over(model, gsm)
    out = {"ce_mix": ce_mix, "ce_mix_windows": mix_w,
           "ce_by_source": by_src, "ce_src_windows": src_w,
           "ce_gsm8k": ce_gsm, "ce_gsm_windows": gsm_w,
           **retrieval_probe(hs, d, seed), "eval_seed": seed,
           "eval_wall_s": time.perf_counter() - t0}
    return out


def finetune(model, sources, steps: int, seed: int, dev, opt_state=None) -> dict:
    from cubbyllm.core.generation import SnapshotHardener
    from cubbyllm.training import TrainLoop
    pipe = pipeline(sources, seed)
    loop = TrainLoop(model, pipe, SnapshotHardener(), lr=LR, batch_size=BATCH, seq_len=SEQ,
                     device=dev, amp=(dev.type == "cuda"), warmup=WARMUP, total_steps=0)
    resumed = False
    if opt_state is not None:
        try:
            loop.opt.load_state_dict(opt_state)
            for g in loop.opt.param_groups:
                g["lr"] = LR
            resumed = True
        except Exception as e:                       # shape/order mismatch -> say so, run fresh
            print(f"    !! could not restore optimizer state ({str(e)[:120]}); running with a fresh Adam", flush=True)
    t0 = time.perf_counter()
    losses = []
    for i in range(1, steps + 1):
        losses.append(loop.step())
        if i % 10 == 0 or i == steps:
            print(f"    step {i:4d}/{steps}  loss {np.mean(losses[-10:]):.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return {"steps": steps, "lr": LR, "warmup": WARMUP, "tokens": steps * BATCH * SEQ,
            "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])),
            "wall_s": time.perf_counter() - t0, "sources": [s["name"] for s in sources],
            "optimizer": "resumed" if resumed else "fresh"}


def save_delta(path, ps, meta, step, info):
    if not SAVE_DELTAS:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"step": step, "meta": meta, "params": [p.detach().cpu() for p in ps], "a7": info}, path)
    return path


def fmt(dec: dict) -> str:
    m = dec["margins"]
    return (f"{'PROMOTE' if dec['promote'] else 'REJECT '} | dCE mix {m['d_ce_mix']:+.4f} (thr {m['eps_mix']:.4f}) "
            f"| worst src {m['worst_source']} {m['d_ce_source_worst']:+.4f} (thr {m['eps_src']:.4f}) "
            f"| dCE gsm8k {m['d_ce_gsm8k']:+.4f} (thr {m['eps_gsm']:.4f}) "
            f"| retrieval {m['retrieval_acc']:.3f} (floor {m['retrieval_floor']:.3f})"
            + (f" | fired: {dec['fired']}" if dec['fired'] else ""))


def main():
    torch.manual_seed(0)
    if DEVICE == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(DEVICE)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"python {platform.python_version()} | {platform.platform()} | torch {torch.__version__} "
          f"| numpy {np.__version__} | device {dev} ({gpu}) | threads {torch.get_num_threads()}")
    print(f"base {CKPT} | corpus {CORPUS} | sources {SOURCES} | gsm8k {GSM8K}")
    print(f"deltas: {STEPS} steps x B{BATCH} x S{SEQ} = {STEPS * BATCH * SEQ:,} tokens, lr {LR:g}, "
          f"warmup {WARMUP} | A = slice [{SLICE}] no replay | B = full mixture (replay)")
    print(f"eval: mix {MIX_BATCHES} batches, {SRC_BATCHES}/source, gsm8k {GSM_BATCHES}; "
          f"thresholds = max(min eps {MIN_EPS}, {NOISE_MULT}x paired SE); retrieval collapse floor {RETRIEVAL_COLLAPSE_FLOOR}\n")
    t_all = time.perf_counter()

    model, ps, base_params, meta, base_step, d, V, opt_state = load_base(dev)
    print(f"arms: {ARMS} | optimizer: {'resumed from checkpoint' if opt_state is not None else 'fresh Adam'}")
    sources = source_specs()
    names = [s["name"] for s in sources]
    if SLICE not in names:
        raise SystemExit(f"CB_A7_SLICE={SLICE!r} is not a staged source; choose one of {names}")
    slice_sources = [{"name": SLICE, "weight": 1.0}]
    print(f"base: step {base_step} | meta {meta} | {sum(p.numel() for p in ps):,} params")
    print(f"sources ({len(names)}): {names}\n")
    gsm = gsm8k_windows(dev)

    # 1. base on two streams -> noise floor; C (null) := base on stream A
    print("=== base evaluation (two held-out streams) ===", flush=True)
    base_a = evaluate(model, sources, d, dev, EVAL_SEED_A, gsm)
    print(f"  seed {EVAL_SEED_A}: ce_mix {base_a['ce_mix']:.4f} gsm8k {base_a['ce_gsm8k']:.4f} "
          f"retrieval {base_a['retrieval_acc']:.3f} ({base_a['eval_wall_s']:.0f}s)", flush=True)
    base_b = evaluate(model, sources, d, dev, EVAL_SEED_B, gsm)
    print(f"  seed {EVAL_SEED_B}: ce_mix {base_b['ce_mix']:.4f} gsm8k {base_b['ce_gsm8k']:.4f} "
          f"retrieval {base_b['retrieval_acc']:.3f} ({base_b['eval_wall_s']:.0f}s)", flush=True)
    eps = noise_floor(base_a, base_b)
    print(f"  per-source ce (seed {EVAL_SEED_A}): " +
          " ".join(f"{k}={v:.3f}" for k, v in base_a["ce_by_source"].items()))
    print(f"  unpaired seed-to-seed noise (information; CE terms use the paired SE): {eps['unpaired_raw']} "
          f"-> retrieval eps {eps['ret']:.3f}\n")

    ft_a = eval_a = path_a = ft_b = eval_b = path_b = None
    # 2. A: forgetting delta (slice only, no replay)
    if "A" in ARMS:
        print(f"=== delta A: {STEPS} steps on [{SLICE}] only (no replay) ===", flush=True)
        restore(ps, base_params)
        ft_a = finetune(model, slice_sources, STEPS, seed=11, dev=dev, opt_state=opt_state)
        eval_a = evaluate(model, sources, d, dev, EVAL_SEED_A, gsm)
        print(f"  A: ce_mix {eval_a['ce_mix']:.4f} gsm8k {eval_a['ce_gsm8k']:.4f} retrieval {eval_a['retrieval_acc']:.3f} "
              f"| train loss {ft_a['loss_first10']:.3f} -> {ft_a['loss_last10']:.3f} ({ft_a['wall_s']:.0f}s, "
              f"optimizer {ft_a['optimizer']})", flush=True)
        path_a = save_delta(os.path.join(SAVE_DIR, f"a7_forget_{SLICE}{TAG}.pt"), ps, meta, base_step + STEPS, ft_a)

    # 3. B: clean delta (full mixture = replay)
    if "B" in ARMS:
        print(f"\n=== delta B: {STEPS} steps on the full mixture (replay) ===", flush=True)
        restore(ps, base_params)
        ft_b = finetune(model, sources, STEPS, seed=11, dev=dev, opt_state=opt_state)
        eval_b = evaluate(model, sources, d, dev, EVAL_SEED_A, gsm)
        print(f"  B: ce_mix {eval_b['ce_mix']:.4f} gsm8k {eval_b['ce_gsm8k']:.4f} retrieval {eval_b['retrieval_acc']:.3f} "
              f"| train loss {ft_b['loss_first10']:.3f} -> {ft_b['loss_last10']:.3f} ({ft_b['wall_s']:.0f}s, "
              f"optimizer {ft_b['optimizer']})", flush=True)
        path_b = save_delta(os.path.join(SAVE_DIR, f"a7_replay{TAG}.pt"), ps, meta, base_step + STEPS, ft_b)

    # 4. the gate on whatever ran (C always)
    decisions = {"C": gate_decision(base_a, base_a, eps)}
    if eval_a is not None:
        decisions["A"] = gate_decision(eval_a, base_a, eps)
    if eval_b is not None:
        decisions["B"] = gate_decision(eval_b, base_a, eps)
    verdict = (separation_verdict(decisions) if {"A", "B"} <= set(decisions)
               else {"partial_run": True, "arms": sorted(decisions),
                     **{f"{'rejects' if k == 'A' else 'passes'}_{k}": (not decisions[k]["promote"]) if k == "A" else decisions[k]["promote"]
                        for k in decisions}})
    print("\n=== gate decisions (base = seed-777 evaluation; C = null delta) ===")
    for k, label in (("A", f"forgetting ({SLICE} only)"), ("B", "clean (replay)"), ("C", "null")):
        if k in decisions:
            print(f"  {k} {label:28s}: {fmt(decisions[k])}")
    if eval_a is not None and eval_b is not None:
        print("  per-source dCE, A vs B: " + " ".join(
            f"{s}={decisions['A']['margins']['d_ce_by_source'][s]:+.3f}/{decisions['B']['margins']['d_ce_by_source'][s]:+.3f}"
            for s in names))
    elif eval_b is not None:
        print("  per-source dCE, B: " + " ".join(
            f"{s}={decisions['B']['margins']['d_ce_by_source'][s]:+.3f}" for s in names))
    print(f"  not applicable to a trunk delta: {decisions['C']['not_applicable']}")
    print("\n=== verdict (H-A7 kill: reject A, pass B, pass C) ===")
    for k, v in verdict.items():
        print(f"  [{'PASS' if v is True else ('FAIL' if v is False else v)}] {k}")
    sep = verdict.get("gate_separates")
    print(f"\n  GATE {'SEPARATES' if sep else ('DOES NOT SEPARATE' if sep is False else 'PARTIAL RUN (arms ' + ','.join(ARMS) + ')')} | "
          f"total wall {(time.perf_counter() - t_all) / 60:.1f} min | "
          f"one full eval {base_a['eval_wall_s']:.0f}s (nightly budget check)")

    out = {"config": {k: v for k, v in dict(CKPT=CKPT, CORPUS=CORPUS, SOURCES=SOURCES, SPM=SPM, GSM8K=GSM8K,
                                            DEVICE=str(dev), GPU=gpu,
                                            STEPS=STEPS, LR=LR, WARMUP=WARMUP, BATCH=BATCH, SEQ=SEQ, SLICE=SLICE,
                                            MIX_BATCHES=MIX_BATCHES, SRC_BATCHES=SRC_BATCHES, GSM_BATCHES=GSM_BATCHES,
                                            N_BIND=N_BIND, RETRIEVAL_COLLAPSE_FLOOR=RETRIEVAL_COLLAPSE_FLOOR, MIN_EPS=MIN_EPS,
                                            NOISE_MULT=NOISE_MULT, EVAL_SEEDS=[EVAL_SEED_A, EVAL_SEED_B]).items()},
           "base": {"step": base_step, "meta": meta, "eval_seedA": base_a, "eval_seedB": base_b},
           "epsilon": eps,
           "arms": ARMS, "resume_opt": RESUME_OPT,
           "deltas": {"A": {"finetune": ft_a, "eval": eval_a, "saved": path_a},
                      "B": {"finetune": ft_b, "eval": eval_b, "saved": path_b},
                      "C": {"eval": base_a}},
           "decisions": decisions, "verdict": verdict,
           "total_wall_s": time.perf_counter() - t_all,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    logs = os.path.join(VAL, "logs")
    os.makedirs(logs, exist_ok=True)
    jp = os.path.join(logs, f"exp_a7_learning_gate{TAG}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {jp}")


if __name__ == "__main__":
    main()
