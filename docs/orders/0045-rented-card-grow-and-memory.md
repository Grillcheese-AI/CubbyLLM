# 0045 — A rented card (H200 or B200, spot welcome): the grown trunk and its memory against the 450M, run to the end
status: open (Cowork's check, 2026-10-31: the pre-rental items, by Monday; the rental is next Tuesday)
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
- **Commits:** `29d4409` in progress; this one: parts A-D, the readout module, the tests, the smoke.
- **Files:**
  - **A.** `validation/pod/prepare_inputs.py`: the slice (`make_base_cache.py --frac 0.3 --exclude-frac 0.2 --exclude-dir`), the
    decay shards, `facts_text_far` (`--far --mode text --doc 2048`), the epoch table for the most tokens an arm can see
    (`--card b200 --hours 14 --mfu 0.35`: 13.7B; every row at most 2 epochs, else "raise --frac"), the manifest
    (`pod_inputs.json`: size and sha256 of every file, the commit, the named roles), the staging folder (hard links), the
    upload (`upload_large_folder`, resumable; a 16 MB probe times the connection first; `--mbps` to skip it). `--dry` plans
    and prints, writes nothing.
  - **B.** `validation/pod/setup.sh` (the card, the stack -- a B200 on torch < 2.7 / CUDA < 12.8 is refused with what is
    missing, nothing downgraded -- the packages, the clone at the manifest's commit, tmux, how to watch); `run_pod.py`
    (the phases, the state file on the remote, the knobs written once: `HOURS_PER_ARM` and the smoke's steps are kept
    across pods); `pod_phases.py` (preflight, fetch, kernels, speed, the arms' environment: round 2's config and the tiny
    smoke's); `pod_train.py` (the arms with the `Watcher` uploading each slot as its sidecar turns valid, step 0 exact or
    no training, probes, readout, adapters, done); `pod_remote.py` (`FolderRemote` for the smoke, `HfRemote` for the
    private dataset repo; sha256-checked fetch; the token redacted from every line logged).
  - `validation/grow_mem_readout.py` (new): gates 0-3 from the metrics and the probes, one place; the notebook's readout
    cell now calls it (`notebooks/_build_grow_mem.py`, the `.ipynb` regenerated).
  - **C.** `validation/exp_if_follow.py`: ten rule kinds read off the held chats' prompts (words, sentences, forbid,
    require, language, list, json, no_markdown, case, first), 796 of the 3,186 held prompts carry one; ~300 drawn
    round-robin over the kinds; scored over the talk seat as the server runs it (`cubbyllm.emit.seats_450m` on grilly2)
    or over a replies file from any other path (`--dump` writes the prompts for it).
  - **D.** `docs/POD.md`.
  - `validation/train_base.py`: the in-context copy probe draws within a tiny vocabulary (the smoke's 40 tokens; a real
    vocabulary exactly as before). The file was over 300 lines already.
  - Tests: `tests/validation/test_pod_runner.py` (2), `test_grow_mem_readout.py` (3), `test_prepare_inputs.py` (3),
    `test_exp_if_follow.py` (3).
- **Tests:** 1134 passed, 1 failed (`test_every_module_declares_wiring`, the known red of order 0002), the suite without
  the runner's; the runner's two pass apart (about a minute on this CPU).
- **Measured** (against each acceptance line, on this CPU):
  - **The runner end to end on the tiny shape:** `run_pod.py --smoke` runs every phase and writes its outputs
    (`bench.json`, `kernels.json`, both arms' finals and metrics, `grow_step0.json` exact, `probes.json`, `readout.json` /
    `.md`, `adapters.json`, the logs) in about a minute. **Killed mid-arm** (the process killed once a slot is uploaded)
    and started in an empty folder with the same remote, it runs the preflight and the fetch again, pulls the newest slot,
    and `train_base` resumes from it ("resumed ... @ step N" in the arm's log, N the uploaded step); the phases done before
    (kernels, speed) are not run again (no log of theirs in the new folder); the state counts the preemption. A third start
    finds everything done and runs nothing.
  - **A failing kernel test** (a fake test file that fails) records `CB_MINGRU_TRITON: off (tests failed, exit 1)` and the
    arms' environment carries the flag at 0. **Another card model** (the state says B200, the pod says H200) refuses before
    fetching: "compares nothing". A B200 on torch 2.5.1 is refused with "torch >= 2.7".
  - **The readout** gives gates 0, 1, 3a, 3b from fixture metrics as the notebook's cell did (its formulas restated in the
    test), the growth kill at +0.05, gate 0 failing on unequal hours or cards; gate 2 from the probes' reports (exact on
    facts_text_far against both floors, `_copy`, the needle's beyond-window cells against the control at chance).
  - **`exp_if_follow.py`:** each of the ten rules passes a hand-written good reply and fails a bad one; the draw is over
    the kinds; the score counts per kind.
  - **`prepare_inputs.py --dry`** on a toy cache: the planned blocks overlap none the first run (`--exclude-frac`) or the
    continuation (`--exclude-dir`'s `_subset.json`) took; a shard out of unseen blocks takes what is left and says so;
    the table flags a shard drawn more than twice; nothing is written.
  - **On the real cache:** at `--frac 0.3` the slice is ~5B tokens; a B200 control at 14 h and 35% MFU draws ~13.7B, so
    the table will say "raise --frac" on the big shards. Run `--dry` on the full cache and set `--frac` (or `--tokens`
    from the bench's measured tok/s) before building; the build itself is hours of disk, not done here.
- **Not verified here:** anything on a card (the kernel tests pass or fail there; `speed_check.py` runs only on CUDA;
  the probes and the adapters are skipped in the smoke and recorded as such); the upload (needs `HF_TOKEN` and
  `CB_POD_REPO`; the code path is `upload_large_folder` as documented); the prices.
- **Found on the way:**
  - `train_base.py`'s copy probe assumed a real vocabulary (ids 1000-30000): fixed for the smoke, bit-identical otherwise.
  - A new pod must run the preflight and the fetch again even when the state says they are done (its own card, its own
    empty disk); the first version skipped them and resumed into an empty corpus.
  - The adapters phase needs the emitter's and the talk's data in the inputs (`--emitter-data`, `--talk-data`); the order
    listed only the slice, the tokenizer and the start checkpoint. Without them the phase records "skipped".
  - `exp_if_follow.py` runs the talk seat on grilly2 (the serving path) or scores a replies file; a torch path would mean a
    torch LoRA loader the repo has only inside the trainers. Say if you want it.
- **Questions:**
  1. The epoch table at 14 h on a B200 asks for a bigger slice than `--frac 0.3`: should `--frac` follow the table (0.6-0.8
     of the cache, 25-35 GB to upload) or should `HOURS_PER_ARM` fall to what 0.3 feeds at 2 epochs (about 7 h)?
  2. The read count rule (16, or 8 under half the control's MFU) and the 92% memory line are in `pod_phases.py` as the
     order wrote them; the bench runs `speed_check.py` first and then the trainer's bench with its chosen kernels.
  3. The probes run on the card after the arms (1-2 h on the meter). Fine, or at home from the finals?

## Cowork's check (2026-10-31)
Read: the report, `docs/POD.md`, the smoke's outputs as listed. The runner does what the order asks.
That includes the kill-and-resume on an empty folder, the refusal of a different card model, and the
flag falling to 0 when a kernel test fails. **The rental is next Tuesday** (Nick, 2026-10-31).
- **Question 1, slice or hours:** the order already decides it. Every row of the epoch table must be at
  most 2 epochs, "else raise `--frac`". Keep `HOURS_PER_ARM` at 12-14 for spot and raise `--frac` until
  the table holds for a B200 control at 14 h. That covers an H200 too.
  - The report puts that at 0.6-0.8 of the cache (25-35 GB). If the unseen blocks (the cache less the
    first run's 20% and the continuation's subset) cannot give that much, say so before building.
  - Upload 25-35 GB before Tuesday.
- **Question 2:** yes, the read-count rule and the 92% memory line as written.
- **Question 3, probes:** on the card, after the arms. It is 1-2 h ($3-16 at the listed prices) and spares
  moving the finals. Add one knob:
  - `CB_POD_DEADLINE` (the rental's end, from the environment): a phase after the arms starts only if
    its estimate fits before it;
  - a probe phase skipped this way is recorded, and the probes then run from the finals on Colab, with
    the same script and floors.

### Before Tuesday (by Monday evening)
1. **Slice size:** `prepare_inputs.py --dry` on the full cache. Set `--frac` from the table, and record
   the table in the report.
2. **Build the inputs,** with `--emitter-data` and `--talk-data` for the adapters phase (the report found
   them missing from the order's list).
3. **The repo and token:** Nick makes the private dataset repo and sets `HF_TOKEN` and `CB_POD_REPO` in
   his environment, never in a file. Claude Code then runs a round trip through `HfRemote`: a small
   folder up and down, with hashes compared. It never prints the token.
4. **Upload** the inputs. Start by Sunday so the upload is done by Monday.
5. **Before renting:** Nick reads `docs/POD.md`'s "Before renting" and "The one command". Cowork
   rechecks the card prices and availability on Monday.
6. **On Tuesday:** Nick rents the card and runs the one command. Claude Code does not start a paid run.

Status: open until items 1 to 4 are reported. Then `review` (ready to rent).

## Pre-rental item 1: the slice size (Claude Code, 2026-10-31)
`prepare_inputs.py --dry` on the full cache (66 GB, 13 shards; excluding the first run's `--frac 0.2` and the
continuation's subset). The first try ran out of memory in the overlap check (a set of every taken token); it now
compares block starts (equal-size blocks overlap iff their starts are under a block apart), pinned by a test.
- **At 14 h on a B200 (13.69B tokens an arm):**
  - `--frac 0.3`: 5.42B, eight shards over 2 epochs (fineweb_edu 5.01).
  - `--frac 0.8`: 10.41B, and ten shards are **out of unseen blocks**. Still over: fineweb_edu 2.51, nemotron_code
    2.03. More `--frac` adds nothing unseen. At `--frac 1.0` the plan takes whole shards, including the blocks the
    earlier runs read.
- **So the unseen blocks cannot give what 14 h needs.** This is the case your check asked to hear before building.
  Nothing is built.
- **What holds:**
  - **(a) `HOURS_PER_ARM=11`** with every unseen block (`--frac 0.9`). Every row is at most 2 epochs (fineweb_edu 1.97,
    nemotron_code 1.59; table in `validation/logs/pod_dry_slice_11h_2026-10-31.txt`). Upload: about 42 GB of shards (10.41B tokens as
    u32), plus the decay shards, `facts_text_far`, the start checkpoint (6.6 GB) and the adapter sets (0.3 GB): about
    50 GB, against the 25-35 planned.
  - **(b) 14 h, accepting fineweb_edu at 2.5 epochs** and nemotron_code at 2.0. The same upload as (a).
  - **(c) 14 h, letting the arms reread blocks the earlier runs saw** (`--frac 1.0`). The slice is the whole cache:
    17.7B tokens, about 71 GB as u32 plus the rest. That breaks "blocks neither run took".
- **My reading:** (a) keeps both rules as written. It costs 3 h an arm, and with the probes on the card it still fits
  a spot rental. The upload is larger than planned either way.
- **The question is Nick's or Cowork's:** (a), (b) or (c)? The build starts on the answer. With D: as the stage it takes
  about an hour here, then the upload.
