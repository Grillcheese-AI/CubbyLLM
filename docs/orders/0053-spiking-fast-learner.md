# 0053 — A spiking fast learner in the loop: STDP gated by the VM's dopamine, a day replayed on CPU
status: in progress
written: 2026-11-05 by Cowork · approved: 2026-11-05 by Nick ("yes write it up") · hypothesis: H-A16

## Why
Nick, 2026-11-05: the SNN "should be the one involved in the self learning loop like in cubemind", and "I wonder why
we dont train these".

Read the same day: **nothing has ever taught the spiking part what was right.**
- cubemind's trained LM builds the GIF neurons and the hormone ODE but does not wire them (`training/vsa_lm.py`).
- cubemind's live brain runs STDP that strengthens what fires together, whatever the answer was. Its hormones are
  computed after the SNN and go only to the speaking LLM (`model.py`, `brain/synapsis.py`).
- Here, the ported ODE's `modulate_tau` has no caller.
- A correlation rule alone is what H-A1 measured: the Hebbian layer forgot 94%.

**0049 supplies the missing teacher:** a VM verdict on every attempt. That is the third factor.
- STDP marks the synapses that produced an answer.
- Dopamine (the striatum's prediction error) decides whether the mark becomes a weight change: up for right, down for
  wrong.

0049's first T1 also found the limit this could lift. The loop learns only from its own right attempts, so wd/fr
(5.7% right) could not start. This learner learns from wrong verdicts too.

**What it brings, in plain words:** Cubby gets a quick reflex beside its slow memory. Each time the checker says right
or wrong, the connections that produced the answer get stronger or weaker on the spot, and overnight the lesson moves
into a permanent add-on.
- **Over a conventional model:** a conventional model learns only when it is retrained, from labelled examples, so
  between retrainings the same mistake comes back every time. Here the correction starts with the next question,
  needs no GPU, learns from wrong answers as well as right ones (so a skill near zero can still start), and the trunk
  never changes.

## Build
Flag-gated (`CB_FASTLEARN`, off by default), off the serving path. The learner is numpy only (no torch, no GPU).
File names are Claude Code's.

1. **The cache,** built once. Where is Claude Code's choice, recorded: at home on grilly2 in short chunks, or a cell
   in Nick's next Colab session.
   - **The problems:** 0049's wd/en and wd/fr keys, in 0049's stream order, with 0049's held problems (hash
     `ad968cc8de05bf69`). Their counts are reported.
   - **Per problem,** with the base adapter (`cubby450m_v12e_w_step_cont`) on the frozen trunk:
     - the greedy program up to its HOP1 word;
     - the trunk's final hidden state there (d = 1,024);
     - the top K = 32 property words by the trunk's own score at that slot, and those scores. How a multi-token word
       is scored is Claude Code's, recorded.
   - **Per candidate:** the program with that word, run in the VM (CubeLang `29e81bd`) and marked as 0049 marks wd
     (right, refused, wrong). If the rest of the program depends on the word, it is completed greedily after it;
     recorded.
   - **The ceiling,** per key: the share of problems with at least one right candidate among the 32. No arm can pass
     it.
2. **The learner:**
   - **Encoder:** a fixed random 8× up-projection of the cached state (8,192 neurons) drives GIF neurons ported clean
     from `cubemind/brain/gif_neuron.py` (port, don't link: multi-bit spikes and the adaptive threshold as there; numpy
     only, without its grilly v1 path).
     - The state is presented for T steps (Claude Code's value, recorded).
     - The threshold is set so that about 2% fire at rest (H-A5's sparsity).
     - The encoder does not learn here.
   - **Readout:** plastic weights from the neurons to the properties that appear among the cached candidates.
     - A candidate's bias is its property's; for a word naming several, the largest (recorded).
     - Its score is the trunk's own plus gain × bias.
   - **The trace:** each synapse from a firing neuron to the chosen candidate's property is marked, and the mark
     decays with time constant τ. With one decision per wd request it is the last choice's. It matters for multi-step
     programs (out of scope).
   - **The third factor:** the verdict through the striatum's reward (+1 right, 0 refused, −5 wrong).
     - δ = reward − expected, per skill signature, by the striatum's own update (`cubbyllm/reasoning/striatum.py`; a
       separate instance, never the serving one).
     - Δw = η · δ · trace.
   - **Bounds:** each neuron's outgoing weights are kept within a fixed norm. Growth without limit is H-A1's failure
     class.
   - **Choosing:** during the day, sampled from the scores (temperature 1.0, as 0049's practice). Held reads are
     greedy.
   - **Set before any held read:** gain, η, τ and the norm, on wd/en's day stream. They are recorded and kept for
     wd/fr.
3. **The arms,** on the same cached day, in the same order, with 3 seeds (the projection and the sampling):
   - **N:** none (the base adapter's choice);
   - **S:** the spiking learner, three-factor (the proposal);
   - **P:** plain STDP, no dopamine. The chosen candidate's synapses strengthen at every choice, whatever the verdict
     (cubemind's rule), with the same bounds;
   - **D:** dense, three-factor. The same projection without the threshold (every unit's value passed on), the same
     readout and rule. A size-matched control, as in H-A5.
4. **The days:**
   - **Day 1:** wd/en's stream, then wd/en's held read.
   - **Day 2:** wd/fr's stream with the same learner continuing, then wd/fr's held read, then wd/en's held read again
     (the retention read).
   - **The bootstrap read:** wd/fr's stream from a fresh learner, then its held read.
   - Held problems are never seen by a learner before its read.

## Acceptance (pre-registered)
- **Tests:**
  - the GIF port gives the spike counts recorded once from cubemind's neuron on fixed inputs (a fixture; no import of
    cubemind at test time);
  - with the flag off, or with every weight zero, the choice is the base's, bit for bit;
  - the trace reaches only the chosen candidate's synapses from firing neurons;
  - a right verdict raises the chosen candidate's bias, a wrong one lowers it, and with δ = 0 nothing moves;
  - the bounds hold over a long random stream;
  - a replay reproduces under its seed.
- **A, reported per key and arm:**
  - the held right rate (mean and spread over the 3 seeds), with held wrongs and refusals;
  - the day's curve (right rate per 50 requests);
  - the right programs found during the day, against 0049's practice at the same number of tries (from the second
    T1's logs where they exist, otherwise N's sampled tries);
  - the ceiling;
  - the cost: wall time of the cache and of each arm's day on CPU.
  - **With up to 100 held problems a key, a difference within the seeds' spread counts as none.**
- **A, the lines:**
  - **Line:** S raises the held right rate by at least 5 points over N on at least one key (0049's probe bar), with
    held wrongs not above N's, and wd/en within 1 point of its day-1 read after day 2.
  - **Kill, the three-factor claim:** S − P under 1 point on both keys. Then dopamine adds nothing, and the loop keeps
    its adapters alone.
  - **Kill, the learner:** S under N + 5 on both keys; or wd/en more than 1 point down after day 2 (H-A1's failure);
    or held wrongs above N's.
  - **The spiking part:** if D is within 1 point of S on both keys, the spikes add nothing. The rule survives without
    them, and H-A16 is narrowed, not killed.
  - **The bootstrap read,** reported against the same +5 line, with the number of right verdicts S had to learn from.
- **B, consolidation** (only if A's line is met; one Colab session Nick runs):
  - 0049's adapter recipe (arm A, right answers only) trains on the right programs S's day found, and again on those
    N's tries found at the same number of attempts.
  - The held read: the adapter from S's day alone, the same adapter with S's learner, and the adapter from N's day.
  - **Line:** removing the learner costs at most 1 point (the night absorbed the day). The S-day adapter's gain over
    the N-day one is reported: the learner as the way in for French.
  - **Kill:** removing it costs more than 1 point. Then the learner holds something the night did not absorb and would
    have to persist, which is H-A1's risk. The consolidation claim is killed.
- **C, the hormones** (only if A's line is met; CPU):
  - the ported ODE (`cubbyllm/affect/neurochem.py`, a separate instance) is driven by the day: reward on a right
    verdict, threat from a run of wrongs, novelty at a new key. The signal sizes are Claude Code's, recorded before
    the run;
  - its `modulate_threshold` and `modulate_tau` set the GIF neurons' threshold and leak, per request;
  - against S with fixed threshold and leak at the best of a 3 × 3 sweep made on wd/en's day (no held reads).
  - **Line:** at least 3 points held on at least one key over the best fixed setting, with A's other lines still
    met.
  - **Kill:** no gain. The hormones stay where they are (manner and routing caution), and the learner keeps fixed
    values.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **The trunk stays frozen.** The learner reads cached states only.
- **Nothing reaches the serving path.** A passed learner would go in the brain's fast route, its SNN slot
  (`cubbyllm/brain.py`), later, by its own order.
- Train splits only, as in 0049.
- Port, don't link: the GIF neuron is copied into CubbyLLM, and nothing imports cubemind.
- grilly2 only on the local card, in short chunks. Claude Code does not start a paid run; B is a Colab cell Nick runs.
- No local path in a committed file.

## Out of scope
- **Multi-step programs** (arithmetic, plan), where the trace has to span the steps until the verdict: the next
  order, from A's numbers.
- **Plasticity in the encoder** (STDP on the input side, neurogenesis).
- **Serving the learner live** (the SNN slot), and grouping live requests.
- **cubemind's own fixes** (the gate that never learns, `snn_ratio` through a sigmoid, the hormones computed after the
  SNN): noted for cubemind's repo, not built here.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Built: the learner, the cache, the days (Claude Code, 2026-11-05)
**The choices the order left open, recorded before any run:**
- **Where the cache is built: a cell in Nick's Colab session** (`selfimprove.ipynb` step 4, about 15 min on an A100).
  It uses the same torch trunk and base adapter as 0049. On this CPU a problem takes about 50 s (the smoke), which is
  about 10 hours for 700. The VM marks the candidates at home (`fastlearn_mark.py`, seconds).
- **The candidates:** the property labels of the problem's language whose property is a wd train gold in it (the
  task's vocabulary: 137 EN, 159 FR), not all 12,684 properties.
- **A multi-token word's score:** log p(prefix + word + '"') - log p(prefix), the word and its closing quote, exact
  whatever the BPE does at the boundary.
- **The rest of the program** after the word is the greedy program's own: a wd program's tail does not depend on the
  word.
- **The base's choice (N)** is the top candidate by the trunk's score; during a day N samples from those scores
  (temperature 1.0).
- **The encoder:** 8,192 neurons (8 x 1,024), a fixed Gaussian projection of the rms-normalised state, presented for
  T = 8 steps. GIF with L = 16, tau 10 and alpha 0.01 (cubemind's defaults). One global current offset is calibrated
  on day 1's states so that 2% of the neurons fire. The activity is each neuron's spike count.
- **The dense control (D):** the same projection and offset, every value passed on.
- **The striatum:** its own instance, keyed per skill signature (the signature as the proposer, one shape). Reward:
  +1 right, 0 refused or caught, -5 wrong.
- **The settings, set on wd/en's day stream before any held read:** a grid over gain (0.5, 1, 2, 4), eta (0.005,
  0.02, 0.08), tau (0.25, 1 choices) and the norm (1, 4), by S's day right rate over the 3 seeds. The best are kept
  for every arm and for wd/fr.
- **The GIF fixture:** recorded once from cubemind `f21d805` on three settings. cubemind's module imports grilly v1's
  bridge, so recording it initialised the local card once; no GPU work ran (the neuron is numpy). The port imports
  neither cubemind nor grilly.

**Files:**
- `cubbyllm/selfimprove/gif.py` (the port);
- `cubbyllm/selfimprove/fastlearn.py` (`CB_FASTLEARN`: the encoder, the readout, the trace, the rules, the bounds);
- `validation/selfimprove/fastlearn_cache.py` (GPU), `fastlearn_mark.py` and `fastlearn_day.py` (CPU);
- `notebooks/selfimprove.ipynb`, step 4.

**Tests:** 9 pass.
- `tests/test_fastlearn.py`:
  - the port gives cubemind's spike counts exactly;
  - with every weight zero the choice is the base's;
  - the trace reaches only the chosen candidate's synapses from firing neurons;
  - right raises the bias, wrong lowers it, and delta = 0 moves nothing;
  - plain STDP strengthens whatever the verdict;
  - the bounds hold over 400 random verdicts;
  - a replay reproduces under its seed;
  - the calibration reaches 2%.
- `tests/validation/test_fastlearn_day.py`: the days run on a synthetic cache where, in one cluster, the right
  candidate is the trunk's second. N's held read is 0.5, S's is above 0.9.

**The smoke** (CPU, the real 450M, base adapter, VM, 2 problems a language):
- the cache found the slot in all 4, with the held hash matching 0049's;
- the marks put a right candidate among the 32 for 1 of 2 in each language.

**Next:** the cache in Nick's session (step 4), then the marks and the days at home.

## Cowork's addendum (2026-11-06): the reversal read
Nick, 2026-11-06, from his note on how the brain represents opposites: when what was good becomes bad, the amygdala
remaps the cue onto the opposite valence's ensemble, and the remapping comes before the change in behaviour. A
three-factor rule should do the same. Plain STDP cannot.
- **The read** (reported; after A's day 1, the same cache, on CPU):
  - on a fixed subset of wd/en's keys, the cached marks are swapped: the candidate that was right is now wrong, and
    one named other candidate is now right (a world where a relation word changed meaning);
  - the learner continues from day 1.
- **Measured, per arm (S, P, D):**
  - verdicts until the swapped keys' right rate is back at its day-1 level;
  - how often the old answer is still chosen after the swap (perseveration);
  - the keys not swapped stay within 1 point.
- **Expected:** P never recovers, since it strengthens whatever it chose. There is no line: it is reported beside A, in
  H-A16's terms.

## Cowork's note (2026-11-11): a learning rule from the field watch, approved (Nick, 2026-11-11: "proposals: yes")
Nick: "there's a lot more we can borrow in it". **Approved by Nick on 2026-11-11 ("proposals: yes"); built when this order is taken, in its queue place.** **Was: proposed, for Nick's yes; not built, not measured.** As the
field watch read it.
- **Vector Symbolic Policy Gradient** (arXiv 2608.18404): an actor whose policy update is exactly an
  advantage-weighted bundling of hypervectors. The rule is local and uses bundling only, with a proof, and suits
  event-driven hardware.
- **For this order:** a candidate update rule for the learner, compared with the current one on this order's
  own tasks before anything changes.
- **What it brings, in plain words:** the fast learner improves from experience with the one operation it
  already has, adding vectors.
- **Over a conventional model:** a network's update needs gradients carried back through every layer; this one
  needs only what the unit itself sees.
