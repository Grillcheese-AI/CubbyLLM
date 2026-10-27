# 0045 — A rented card (H200 or B200, spot welcome): the grown trunk and its memory against the 450M, run to the end
status: open
written: 2026-10-27 by Cowork · approved: 2026-10-27 by Nick ("yes"; "could be b200 too"; "we could get more hours with a spot server") · hypothesis: H-E20

## Why
H-E20 asks whether the 450M should grow (depth ×2 + FFN ×2, 1.36B, the same function at step 0) with the
episodic read, on data that teaches following and acting. Round 1 (Colab's RTX PRO 6000, 20 h) could not
answer it: the grown arm saw 0.42B tokens (micro 2, no activation recompute, 16 reads, 7-9% MFU), never left
the growth's transient (held-out mix flat at 2.44 through the stable phase), and its read stayed on the floor
(needle beyond the window 15.9% against 16.8% for the shuffled control;
`validation/logs/needle_grow_mem_grown_2026-10-18.json`). Gate 1 still waits on a control arm that never ran.
Since then the read's backward is fixed (2026-10-20), recompute is on, and round 2 gave the read a reason to
learn (`facts_text_far`). A rented H200 (989 dense BF16 TFLOPS, 4.8 TB/s) or B200 (2,250, 8 TB/s) gives each
arm roughly 5-10× round 1's tokens in the same hours. Spot (interruptible) cards cost about half, so the run
must lose minutes, not hours, when a card is taken back.

**What it brings:** Cubby's language cortex grown ~3× without starting over, a memory beyond the window for
long conversations and agent tasks, and both seats' adapters retrained on the winner. **Over a conventional
model:** growth pays only for what is new (the grown model starts as the 450M exactly) instead of a retrain
from scratch, and the memory's cost per token stays fixed where a KV cache grows with the context.

## Build

### A. Inputs, built at home (`validation/pod/prepare_inputs.py`)
1. **A fresh slice** of the full token cache: `make_base_cache.py --src <full cache> --dst <token_cache_r3>
   --frac 0.3 --exclude-frac 0.2 --exclude-dir <token_cache_cont>` (blocks neither the first run nor the
   continuation took, ~5B tokens), then `make_decay_shards.py` (instruct, agent, program) and
   `make_fact_shard.py --far` (facts_text_far) into it, with their held-out tails. Print the epoch table for
   the most tokens any arm can see (a B200 control at `HOURS_PER_ARM = 14`); every row at most 2 epochs, else
   raise `--frac`.
2. **A manifest** (`pod_inputs.json`): each file's size and sha256 (the slice and its `_sizes/_align/_holdout/
   _subset.json`, the tokenizer, `base450m_cont_stable_end.pt`) and the commit to run.
3. **The upload** to a private Hugging Face dataset repo named by `CB_POD_REPO`, with `huggingface_hub`'s
   resumable large-folder upload; the token from `HF_TOKEN` in the environment. It prints the size and an
   estimated time on this connection before starting, and resumes after an interruption.

### B. The runner (`validation/pod/setup.sh`, `validation/pod/run_pod.py`, split under 300 lines)
- **`setup.sh`, on a fresh pod:** checks the card (H100, H200 or B200, at least 80 GB) and the stack (on a
  B200, torch ≥ 2.7 with CUDA ≥ 12.8; say what is missing, never downgrade silently); installs what the image
  lacks (tokenizers, numpy, huggingface_hub); clones the repo at the manifest's commit; starts `run_pod.py` in
  tmux; prints how to attach and where the metrics appear on the remote.
- **`run_pod.py`, in phases**, its state (`pod_state.json`) kept on the remote, so a new pod resumes at the
  phase and step the last one reached:
  0. **preflight:** card, versions, free disk (≥ 3× the inputs + 100 GB), the token, write access to the
     remote. Any failure exits at once with its reason (no card billed for nothing).
  1. **fetch:** the manifest's files to local NVMe, checked by sha256, resumable.
  2. **kernels:** `tests/model/backbone/test_{mingru,window_attn}_triton.py` on this card. A failing kernel is
     turned off for the run (`CB_MINGRU_TRITON=0` / `CB_ATTN_TRITON=0`) and recorded; it is not fixed on the meter.
  3. **speed:** per arm, `speed_check.py` and `train_base.py` `CB_MODE=bench` (the notebook's micros) →
     `bench.json`: the fastest variant and micro that fit under 92% of the memory, the grown arm with
     recompute on. The memory read on every 4th layer (16 reads, as pre-registered) unless the grown arm's
     measured MFU is under half the control's, then every 8th (8 reads). Decided here, written before any
     training, never changed after.
  4. **control:** `train_base.py` from `base450m_cont_stable_end.pt` with `grow_mem_continue.ipynb`'s round-2
     config (S 2048, 131,072 tokens a step, the decay mix with instruct, agent, program and facts_text_far,
     samples every 500 steps), `CB_HOURS = HOURS_PER_ARM`.
  5. **grown:** `grow_base.py`'s step 0 from the same checkpoint (exactness checked; no training if it is not
     exact), then the same config and the same `CB_HOURS`.
  6. **probes:** `exp_hypertoken_recall.py` on facts_text_far for both finals and for the 450M as it is;
     `exp_needle_forward.py` at 1024 / 2048 / 4096 for both finals.
  7. **readout:** gates 0-3 below, from the metrics, to `readout.json` and `readout.md`. Move the notebook's
     readout into `validation/grow_mem_readout.py` and call it from both.
  8. **adapters on the winner** (the grown final if gate 1 passes, else the control's): `train_emitter_torch.py`
     (the v12e slots, as for `cubby450m_v12e_w_step_cont`) and `train_talk_torch.py` (the talk_v2.2 mix), with
     their notebooks' settings, into the run folder. The serving gates run afterwards at home, not on the meter.
  9. **done:** the last upload, then stop the pod if `CB_POD_STOP` says how (the provider's own command). A
     card never sits billing idle.
- **Spot preemption.** Checkpoint every 15 min (`CB_CKPT_MIN=15`, not 45). After each checkpoint, upload it in
  the background: the newest slot, the metrics, logs and samples; stable_end and the finals once. The state file
  records the arm, the phase and the last step uploaded. A new pod: `setup.sh` → fetch → the state says arm X at
  step N → it pulls that slot and resumes. Never count on a SIGTERM's grace: the slot on the remote is the truth.
  At most 15 min of training plus the fetch is lost per preemption.
- **The same card model for both arms.** `bench.json` records it; an arm resumes only on the same model (a spot
  replacement of another kind stops with a message: equal hours on different cards compare nothing). Hours are
  training hours from the metrics, not wall clock.
- **`HOURS_PER_ARM`** from the environment: 7 for a 20 h on-demand rental, 12-14 with spot.

### C. The instruction-following gate (`validation/exp_if_follow.py`): gate 4's missing piece
About 300 prompts from the instruct shard's held-out chats, each carrying constraints a rule can check: a word or
sentence limit, a required or forbidden word, the language (EN, FR), a format (a list of N items, JSON with given
keys, no markdown), the answer first. No model judges. Deterministic; prints the rate per kind of constraint; runs
on a base plus a talk adapter, in torch or grilly2.

### D. `docs/POD.md`
Which cards (H200, B200; an H100 works, slower), the image (PyTorch on CUDA 12.8), the disk (≥ 250 GB), the
variables (`HF_TOKEN`, `CB_POD_REPO`, `HOURS_PER_ARM`, `CB_POD_STOP`), the one command, how to watch, what a spot
preemption looks like and how to resume, the phases with their hours, and the prices seen on 2026-10-27 (H200
~$3.1-5/h on demand, ~$2.3 spot; B200 ~$3.75-8/h, spot from ~$3.2), marked to recheck before renting.

## Acceptance (pre-registered)
Tests, on CPU here:
- **The runner end to end on a tiny shape** (`--smoke`: train_base's tiny environment, a few steps an arm, a
  local folder standing in for the remote): every phase runs and writes its outputs. Killed mid-arm (the process
  killed, not signalled) and started again in an empty local folder, it fetches from the remote and resumes at
  the last uploaded step, losing at most the steps since that checkpoint. A phase already done is not run again.
- A failing kernel test turns its flag off for the run and records it; another card model refuses to resume an arm.
- The readout gives gates 0-3 from fixture metrics exactly as the notebook's readout does.
- `exp_if_follow.py`: each rule passes hand-written good replies and fails bad ones.
- `prepare_inputs.py --dry`: the slice takes no block the first run or the continuation took; the epoch table.

On the rented card (Cowork reads; H-E20's gates as pre-registered, gate 2 in round 2's form):
0. **Equal compute:** the same tokens a step, the same card model, training hours within 0.5 h.
1. **Growth:** the grown arm's held-out mix at or below the control's + 0.02 at equal hours.
   **Kill:** above it by more than 0.05 (the 450M wants data before parameters; growth stays killed).
2. **Memory:** on facts_text_far, the grown final at least 10 points above both floors (the control's final, the
   450M as it is) with the in-window `_copy` unchanged; then the needle's beyond-window cells > 25% where the
   control sits at chance. **Flat:** serve without the read.
3. **The decay data:** each arm below its first reading on instruct, agent and program; the checked sources
   (fineweb_edu, books, nemotron_qa, nemotron_code) of the grown arm within 0.02 of the control's.
   **Kill:** both arms regress on the checked sources (cut the new shards' share before asking again).
4. **Serving, at home afterwards:** E6-E9 each at least talk_v2_2's with 0 spoken wrong; exp_r17 and the step
   loop at least the 450M's own; exp_if_follow above the 450M's. The gap to the stand-in, as measured.
Report beside them: tokens per arm, tok/s and MFU per arm on the card, preemptions and minutes lost, dollars spent.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- The runner moves no pre-registered number. `HOURS_PER_ARM` and the read count are its only knobs, both
  written down before training starts.
- No secret in a file, log, argument or commit: `HF_TOKEN` comes from the environment and is redacted from logs.
- No account, repo or bucket name in a committed file (`CB_POD_REPO` from the environment).
- Claude Code does not rent or start the card: Nick does, from `docs/POD.md` (Cowork can watch the run).
- Uploading Nick's own data to his private repo is not an external download. Fetching a public dataset onto the
  pod is out of scope.

## Out of scope
- The serving gates on the meter (they run at home on grilly2).
- A new shape, a new data source, a second growth step.
- Order 0032 (the trunk reads the worlds): its own run.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
