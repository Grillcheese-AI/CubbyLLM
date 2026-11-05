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
