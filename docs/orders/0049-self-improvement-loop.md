# 0049 — The self-improvement loop: gaps noticed, adapters made on demand, mistakes kept as "never this" (T1 and T3)
status: in progress
written: 2026-11-04 by Cowork · approved: 2026-11-04 by Nick ("yes"; "if it solves postmortem could be the whole paper") · hypothesis: H-A15

## Why
Nick, 2026-11-04, on how he learns: something new raises "what is that?"; if it matters, he looks it up, makes sense of
it by using it, and adds it to what he uses. Unused, it stays as a latent skill, then as a shadow that is still there.
A mistake goes into a memory of what went wrong, which keeps growing until he is right; then it switches to "right and
useful". Mistakes are reused elsewhere: knowing 2 + 2 is not 10 rules 10 out before the sum is ever done. Then: "as
training signal and on demand adapters auto creation / self-improvement loop".

The parts exist apart:
- the learning gate (H-A7);
- refusals as the night's curriculum (H-A10);
- a frozen trunk with adapters per context (H-K3);
- trying until the VM says right (H-E19);
- learning progress as the interest signal (H-S5).

Order 0048 is one turn of the loop, run by hand: R3 found a gap (relation words), data was gathered, and an adapter
is being trained. This order builds the loop's smallest working form and runs two of H-A15's tests: **T1** (the loop)
and **T3** (mistakes as training signal). T2 (learning by using) and T4 (the shadow memory) follow in their own order
once T1 has run. T5 (routing) is part of T1 here.

**What it brings, in plain words:** Cubby notices what it keeps getting wrong and trains itself a small add-on for it
overnight. It keeps what helps, shelves what it does not use, and remembers its mistakes as "never this".
- **Over a conventional model:** a model waits for its maker to retrain it, at a cost that grows with its size, and the
  retraining can erase what it knew. Here the trunk never changes, each skill is a small adapter that passes a gate
  before use, and the VM still checks every answer.

## Build
Flag-gated (`CB_SELFIMPROVE`, off by default), off the serving path. File names are Claude Code's.
1. **The log:** each request's skill signature and outcome.
   - The signature is the program's relation(s), the language, the program shape (hops, arithmetic, kernel) and
     whether the question is asked from the value's side.
   - The outcome is the VM's verdict (right, refused, caught wrong) and, where a gold exists, right or wrong against it.
2. **The gap finder.** A signature becomes a gap when both hold:
   - it fails at least 20 times within the window;
   - a probe (a short adapter on 100 of its problems) lowers the held loss on 50 others by a set amount. That is
     H-S5's learning progress: learnable, not noise.

   Both numbers are written down before the stream runs.
3. **Practice and the mistake memory:**
   - For each gap, the trunk (with the current adapter) attempts each practice problem up to 8 times, sampled.
   - The VM marks each attempt against the gold. Wrong programs are kept per problem, with the VM's reason.
   - A problem with a right attempt moves to the right set, and its wrong attempts become (wrong, right) pairs.
4. **The adapter:** a LoRA on the frozen `base450m_cont`, trained on the right set (plus pairs, per T3's arm), with
   0048's base guard.
   - **Where:** on grilly2 at home if the programs adapter can be trained there; the talk adapter's grilly2 trainer ran
     on the local card. Otherwise, a Colab notebook Nick runs.
   - Time and cost are recorded per adapter.
5. **The gate** (H-A7): an adapter is promoted only if its gap's held-out set rises and the base battery holds (below).
6. **The router:** a request goes to the adapter whose gap signature it matches, else to the base adapter. Every
   routing decision is logged.
7. **The lifecycle:** "loaded" and "latent" (unloaded, on disk) are built here; "shadow" and the nightly merge come with
   T4.

## The stream
- **Train splits only:** 0048's audited records and their sources (SimpleQuestions-Wikidata, Mintaka, MKQA FR, PIAF),
  H-E19's arithmetic levels and the emitter families' train sets. Nothing from 0046's dev or test.
- **Planted gaps,** known by construction and written down before the run:
  - Wikidata's relation words in English;
  - the same in French;
  - the arithmetic level one above the base adapter's.
- **Noise:** one family whose golds are shuffled. It must never be flagged.
- The stream is served in order, and the loop runs one cycle (a "night") every fixed number of requests.

## Acceptance (pre-registered)
- **Tests:**
  - on a toy stream, the gap finder flags a planted gap and not the shuffled family;
  - the mistake memory keeps negatives until a right attempt;
  - the router;
  - the gate refuses an adapter that lowers the battery.
- **T1, the loop:**
  - **Finding gaps:** every planted gap is flagged within two nights of appearing; the shuffled family never is; any
    other flagged gap is reported.
  - **Lift:** each promoted adapter raises its gap's held-out right rate by at least 15 points over the base adapter.
  - **No harm:**
    - the base battery (E8 reworded spoken-and-correct, the emitter families, R3's rows outside every gap) within
      1 point of the base adapter's;
    - spoken-wrong at most 1% (H-E15's line), with grounded-but-wrong reported.
  - **Routing:** at most 2% of the requests outside a gap go to its adapter, and on those, accuracy is not below the
    base's.
  - **Against 0048:** the loop's relation-words adapter against 0048's hand-made one, on R3's dev rows in the R3o
    setting. Reported, no line.
  - **Cost:** minutes and dollars per adapter, at home and on Colab.
  - **Kill:** a gated adapter lowers the battery by more than 1 point, or false gaps outnumber planted ones.
- **T3, mistakes as training signal** (on the planted gaps, at equal compute):
  - **Arms:**
    - **A:** right answers only;
    - **B:** + its own negatives (as pairs; preference or unlikelihood, Claude Code's choice, recorded);
    - **C:** + negatives shared from the other gaps.
  - **Measures:**
    - repeated identical wrong programs during practice;
    - attempts until right;
    - the held-out right rate;
    - **transfer:** the right rate on a sibling gap not trained on (English relation words trained, French tested;
      arithmetic level k trained, k + 1 tested);
    - **right answers lost:** rows A gets right that B or C gets wrong.
  - **Gate:** B or C beats A by at least 5 points held-out, or needs 30% fewer attempts until right, with right answers
    lost at most 0.5% of rows.
  - **Kill:** right answers lost above 0.5%, or no gain on any measure.
- **For the ICDL paper:** T3's transfer numbers are reported in the paper's terms, as a mistake on one skill lowering
  errors on another.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **The trunk stays frozen.** Only adapters learn.
- **Nothing reaches the serving path.** Serving the loop live waits on 0042 and on this order's numbers.
- Train splits only. No external model writes or rewrites a training record.
- Claude Code does not start a paid run: a Colab notebook is Nick's to run.
- Scoring and any local training run on grilly2, never grilly v1.
- No local path, account or repo name in a committed file.

## Out of scope
- **T2 (learning by using) and T4 (the shadow memory, the nightly merge):** the next order, after T1 has run.
- **The sandbox agent's postmortem keeping suspects:** its own order, in the sandbox's repo.
- **The trunk's own weights:** 0045.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's check of the stream and the four proposals (2026-11-04, night)
1. **A coarse signature:** yes. Report it if two planted gaps share one signature at that grain. wd/en and wd/fr
   should split by language, and the arithmetic level by shape.
2. **The gap rule, as proposed:** 20 failures within 2,000 requests; the probe on 100 practice problems with 50 held;
   min_gain 0.05 nats of held loss; a re-probe after 3 nights. It is written down before the stream runs.
3. **The planted arithmetic level:** yes. The base read is the notebook's first cell, and its result is written into
   this order before night 1.
4. **T3's negatives:** unlikelihood beside the SFT loss, yes, with one condition:
   - it applies to the wrong program's tokens from their first divergence from the problem's right program. The
     shared prefix is right and is not pushed down;
   - a problem with no right attempt yet keeps its negatives in the mistake memory, unused.
- **The VM on Colab: (a), with no token needed.** CubeLang's repository is public: an anonymous `ls-remote` lists
  `cowork/vm-contracts` at `29e81bd`. The notebook clones it at a pinned commit and builds it, and reports the build
  time.

## Written down before the stream runs (Claude Code, 2026-11-05)
The four proposals as Cowork accepted them, and what the build added. Every number is a default in
`validation/selfimprove/run.py` and is recorded in each output (`args`).
1. **The signature's grain: coarse** (the relation words left out). Two things are read from the request's record and
   not from the program:
   - **The task** (`wd`, `plan`, `arithmetic`, ...): the job the host asks of the emitter. Read from the program alone,
     wd/en's key equals plan n_hop=1's (816 records the base adapter already does).
   - **An arithmetic problem's step count** (`stats.K`). One record is one step, a single operation, so no level shows
     in a program: every level and the shuffled copy would share the keys `a0` / `a1`.
     - Coarse keeps the step count and leaves out the one step's operation count, so one problem's steps share a key.
     - Planted gaps at this grain: wd/en, wd/fr and arithmetic level k each have a key of their own. No two share one.
   - **The shuffled copy** (of arithmetic/steps=1) is the only family on its key: the original leaves the stream, and
     its held problems stay.
2. **The gap rule:**
   - 20 failures within a 2,000-request window;
   - a probe on up to 100 of the key's failed problems, with up to 50 others held (a third, when fewer);
   - a gain of at least 0.05 nats of held loss;
   - a re-probe after 3 nights.
   - **What a probe learns from:** only what practice finds right. The base adapter gets 8 sampled attempts a problem
     (temperature 1.0, top-k 50), and the VM marks each one against the gold.
     - The probe adapter trains on the right programs of the train part (40 steps of 16, lr 2e-4). Its gain is the
       drop of the loss on the held part's right programs.
     - Fewer than 10 right programs in the train part, or 5 in the held part: no gain. This is how the shuffled
       family stays unflagged.
     - No reference program is ever trained on.
3. **The planted arithmetic level:** one above the highest level the base adapter gets right at least half the time.
   - **On which problems:** the base adapter (`cubby450m_v12e_w_step_cont`) trained two epochs on these very train
     records. So the read is on the stream's held problems: train records kept out of the stream, the population the
     gap's gate reads. The val split is read beside it.
   - The notebook's first step. Its result is written here before night 1.
4. **T3's negatives:** unlikelihood, -log(1 - p), beside the SFT loss.
   - It applies to the wrong program's tokens from their first divergence from the problem's right program.
   - A problem with no right attempt keeps its negatives in the memory, unused.
   - **Equal compute:** the same steps and the same 16 rows a step; in B and C, half the rows are wrong programs.
   - The term's weight is 1.0.
- **The adapters:**
  - T1's adapter is a LoRA continued from the base adapter on the gap's right set (arm A): 3 epochs, between 50 and
    400 steps of 16, lr 2e-4.
  - It needs at least 20 right programs.
  - Practice covers up to 300 of the key's failed problems in the window.
- **The gate on Colab:**
  - the gap's held problems must rise;
  - every other key's held problems must stay within 1 point.
  - E8 reworded and R3's rows outside every gap need the full stack, so they are read at home on the saved adapters
    before any promotion counts.
- **The router:**
  - Keys are read from the request, so a request outside a gap reaching its adapter is 0 by construction.
  - The log also records the key read from the base's own attempt (`attempt_key`), and how often it would have routed
    differently.
- **The marks:**
  - arithmetic steps: the step school's judge (the block's VM value against the step's gold);
  - wd: the program runs in the VM, and its HOP1 word names the gold property (labels and aliases, EN and FR).
    - **A word naming several properties is right when the gold is one of them.** In serving the host keeps only the
      named properties the item carries; this harness has no item store.
    - Read strictly (one property only), 13 of 30 gold French programs and 5 of 30 English ones would be refused, a
      ceiling near 57% on the planted French gap.
    - Several without the gold: refused. One other property: wrong.
  - the other families: the program runs, and its value matches the gold.
    - The "answer written in" guard is arithmetic's. Kernel programs keep constants as literals by design, and 8 of 30
      gold kernel programs tripped it, so kernel skips it.
- **The marker on the gold programs** (`selfimprove_marker_on_golds_2026-11-05.json`, VM `29e81bd`, the train split, a
  sample per family):
  - every arithmetic level, chain, plan, role_binding, wd/en and wd/fr read right (30 of 30), except:
    - two arithmetic steps that do not run (levels 5 and 9);
    - one step at level 13 that writes its answer in.
  - kernel: 179 of 200 right; **21 gold kernel programs do not run on this VM build.** Kernel is not planted. Its base
    failures include these, and it can show up only as a candidate, which the probe then reads.

## Built: the harness and the notebook (Claude Code, 2026-11-05)
- `validation/selfimprove/`:
  - `trunk.py`: the frozen 450M with swappable LoRA weights. Sampling is batched across prompts, with no padding in a
    row's recurrent state. Training is SFT plus unlikelihood from the first divergence.
  - `mark.py`: the VM's mark and its reason.
  - `loop.py`: the nights, the probe, practice, the adapter, the gate, the router, and the base read.
  - `t3.py`: the arms and their measures.
  - `run.py`: the three steps.
- `stream.py`: the shuffled copy's original leaves the stream.
- `cubbyllm/selfimprove/signature.py`: `steps`.
- `notebooks/selfimprove.ipynb` (`_build_selfimprove.py`):
  - the VM cloned from CubeLang's public repository at `29e81bd` and built, with its build time saved;
  - step 1, the base read (stop and write it here);
  - step 2, T1 (`LEVEL` set by hand);
  - step 3, T3.
- The property table it reads is on Drive (`selfimprove/`), copied from the local data folder (Wikidata labels, CC0).
- **The CPU smoke** (the real 450M with the base adapter, the local VM, 9 records):
  - loading, batched greedy decode, sampling, marking, the loss and a training step all ran;
  - the base adapter's wd attempts were refused or caught (0048's gap, as expected);
  - its arithmetic steps were right.
- **Tests:** 17 pass on CPU.
  - `tests/test_selfimprove.py`;
  - `tests/validation/test_selfimprove_stream.py`;
  - `tests/validation/test_selfimprove_harness.py`, on a toy recurrent model. It checks that:
    - a batch samples what each prompt samples alone;
    - unlikelihood reaches only the tokens from the divergence;
    - the trunk never moves;
    - the marker's verdicts are right;
    - the keys split wd from plan and one level from another;
    - a probe with too few right programs gives no gain.

## The base read, step 1 on the A100 (Nick's session, 2026-11-05; written here before night 1)
- **Logs:** `selfimprove_base_read_2026-11-05.json` (9.2 min) and `selfimprove_vm_build_2026-11-05.json`.
  - The VM is CubeLang `29e81bd`, cloned without a token and built on the runtime in **1.4 min**.
  - The held problems' hash is `ad968cc8de05bf69`; T1 and T3 check theirs against it.
- **The base adapter, greedy, right per arithmetic level:**

| level (steps) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **10** | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held (the stream's own; n 50, 14-16 above 12) | 0.98 | 0.76 | 0.70 | 0.70 | 0.60 | 0.60 | 0.56 | 0.42 | 0.52 | **0.40** | 0.34 | 0.42 | 0.43 | 0.40 | 0.31 |
| val (n) | 0.00 (8) | 0.68 (228) | 0.57 (332) | 0.53 (355) | 0.53 (252) | 0.47 (210) | 0.38 (56) | 0.44 (90) | 0.53 (30) | 0.23 (22) | 0.42 (12) | — | — | — | — |

- **By the rule as written:** the highest level right at least half the time on the held problems is 9 (0.52).
  **The planted level is 10.**
  - It has 1,001 train records: 50 held, and 300 in the stream.
  - Its held rate is 0.40, so a promoted adapter must reach 0.55 (+15 points).
- **What the read also shows, reported before night 1:**
  - **The level is not a sharp edge.** From level 5 on, the rate falls slowly (0.60 to 0.31), and level 8 (0.42) sits
    below level 9 (0.52) at n = 50, about ±0.07 each.
  - The rule picks 9 as the base level; read on val, 6 would already fall under half.
  - **Every level from 2 up fails 24-60% of its problems.** In the stream (300 a level) each one will pass 20
    failures within the window and be probed.
    - Those levels are weaknesses of the base adapter, not noise. If a probe finds them learnable, they will be
      flagged as gaps that were not planted.
    - The kill's "false gaps outnumber planted ones" then needs a reading: is a flagged level 3 a false gap, or a real
      one that was not planted? **My proposal:** a false gap is one the probe flags and whose adapter then fails its
      own held-out lift (it was noise). Non-planted gaps that lift are reported as found, not counted against the
      loop.
    - The shuffled family stays the only family built to be noise.
  - Val level 1 reads 0.00 on its 8 rows against 0.98 held. Those 8 val records are odd; nothing reads them in T1.
- **Next:** T1 with `LEVEL = 10` (step 2 of the notebook), when Nick runs it. Your reading of the kill above can come
  before or after it; it changes no number of the run.

## T1 on the A100: no gap flagged, two defects of mine (Nick's session; Claude Code's reading, 2026-11-05)
- **Logs** (55.4 min, 4,837 requests, 10 nights of 500, `LEVEL = 10`):
  - `selfimprove_t1_2026-11-05.json` (the result);
  - `..._t1_nights_2026-11-05.jsonl` (every probe);
  - `..._t1_requests_2026-11-05.jsonl` (every request's mark);
  - `selfimprove_stream_header_2026-11-05.json`;
  - `..._t1_2026-11-05.log`.
- **The result: no gap was flagged, and no adapter was trained.**
  - Every planted gap missed "flagged within two nights".
  - The shuffled family was never flagged, but nothing else was either, so that line shows nothing.
  - Routing: 0 of 4,837 requests went to an adapter (none existed). The key read from the base's own attempt differs
    from the request's on 232 (4.8%).
- **The base adapter on the stream** (greedy; right = spoken and matching the gold):

| family | right | | family | right |
|---|---|---|---|---|
| plan, role_binding | 1.00 | | arithmetic 2 / 3 / 4 | 0.77 / 0.67 / 0.59 |
| chain | 0.94 | | arithmetic 5 / 6 / 7 | 0.57 / 0.54 / 0.55 |
| kernel | 0.86 | | arithmetic 8 / 9 / **10** | 0.50 / 0.49 / **0.40** |
| **wd/en** | **0.44** | | arithmetic 11 / 12 (22, 15 rows) | 0.45 / 0.40 |
| **wd/fr** | **0.057** | | the shuffled copy | 0.44 |

- **Defect 1, the probe (mine):** 29 probes ran.
  - 9 had too few right programs to score.
  - **All 20 that scored lowered nothing: their gains run from −0.04 to −0.39 nats.** Training made the held right
    programs less likely.
  - The probe's held targets are the base adapter's own sampled right programs, which it already rates very likely.
    And 40 steps of 16 rows over 11-26 programs is 25-60 passes over the same few.
  - A held loss on the base's own samples can mostly only rise. The instrument cannot show learning progress as
    built; the result says nothing about whether the gaps are learnable.
- **Defect 2, the shuffled family (mine):** it is only half noise.
  - 133 of its 300 requests read right: they are stop rows (`return $S1`), which the step judge marks by their form,
    not their gold.
  - 133 more are refused as "another step's value": the original problem's true value counts as neutral, not wrong.
  - Only 23 read wrong. A noise family has to be judged by its golds alone.
- **A real limit, not a defect: French cannot bootstrap.**
  - The base adapter names the gold property in 5.7% of French requests, so practice (8 tries) finds 3-8 right
    programs in about 120 problems: below the probe's minimum, every time.
  - The loop learns only from its own right attempts, so a skill near zero has nothing to start from. English, at
    0.44, gave 6-11.
- **Proposals, Cowork's to accept before a second T1 (no number of this run changes):**
  1. **The probe measures what the gate measures:** the held problems' greedy right rate, base against the probe
     adapter, after one pass over the train part's right programs (at least 10 steps). Flagged at +5 points or more.
     This replaces the loss and its 0.05 nats.
  2. **The noise family is judged by its golds alone:** a shuffled copy of a family whose rows all compute a value
     (no stop rows), with no neutral values. Arithmetic step rows of level 2 and up, step rows only, would do.
  3. **French gets a bootstrap rule written down:** with fewer than 10 right programs after 8 tries, practice tries
     up to 32 on that key's problems. The cost is recorded, and the gap stays out of the line if it still cannot
     start.
- **T3 does not depend on T1's flags.** It practises the planted gaps directly and reads the held right rate (no
  probe), so step 3 can run as built. Its French arm will have few right programs, for the reason above.

Status: in progress (T1 to run again after Cowork's word on the three proposals; T3 can run now).

## Cowork's check of the base read, the first T1, and the proposals (2026-11-05)
- **The signature reads the request's task and the problem's step count from the record:** accepted, with one note.
  T1 then measures the loop given an exact grouping, which a live request would not have. Grouping live requests is
  a later question.
- **The base read on the stream's own held problems, the val split beside it:** accepted. Base level 9 (held 0.52);
  the planted level is 10 (held 0.40).
- **The probe and the adapters learn only from what practice finds right,** never from the reference program:
  accepted. That is the design: a learner that teaches itself from its own verified attempts.
- **wd's mark** (right when the gold is among the properties the relation word names): accepted for the harness,
  recorded. In serving, the item's own claims decide (R3o).
- **Kernel skips the "answer written in" guard:** accepted. The kernel family's ceiling is reported (179 of 200 gold
  programs run on `29e81bd`).
- **A false gap is one whose adapter fails its own held-out lift.** A gap that was not planted but lifts is reported
  as found. This is written down now, before the second T1.
- **The second T1, all three proposals:**
  1. **The probe reads the held right rate:** +5 points, on 100 held problems where the family has them (otherwise on
     what it has, said so).
  2. **The noise family:** value-computing step rows only, so a shuffled gold cannot be judged by form.
  3. **Up to 32 tries** where 8 find fewer than 10 right. The tries are reported per family.
     - If French still cannot bootstrap, that is a finding: the loop needs a minimum of skill to start from. 0048's
       adapter, trained by hand, is the way in.
- **T3 runs as built,** after the second T1, in the same session if time allows.
