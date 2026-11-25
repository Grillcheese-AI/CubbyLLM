# 0083 - Faster answers: a checked plan remembered before the 450M writes one, and the 128k head at a 32k head's cost
status: in progress (Code B, 2026-11-25)
written: 2026-11-24 by Cowork · approved: 2026-11-24 by Nick ("we should also include the 128k == 32k in it") · hypothesis: H-A5 (an external episodic store beside the weights), H-C3 (route 5, order 0068)

## Why
Nick asked whether verified facts should go into the model so that it does not have to certify them again every
time. The check is not where the time goes (the 20-question stack bench of 2026-10-24,
`validation/logs/0037_stack_on_2026-10-24.json`):
- questions the host plans by its own rules: median **0.07 s**, the lookup and the VM check included;
- questions the 450M plans: median **2.74 s**, almost all of it the seat writing the plan.

What saves the time is remembering the plan, still checked, and two parts for that were built in September and never
put on the serve path:
- **The hippocampus** (`cubbyllm/reasoning/hippocampus.py`) keeps every certified chain as an episode (question, plan,
  chain, answer, provenance; 256 bits a code, ~50 ms to recall over 552k). `propose` gives the episode's own plan and
  its relation shape rebound to the entity the new question names. Memory proposes; `covers()`, the walk and the VM
  dispose, exactly as for an emitter's plan.
- **The striatum** (`cubbyllm/reasoning/striatum.py`) orders the proposers per question shape by the gate's reward
  (certified +1, refused 0, wrong -5).
- Today `sleep.py` writes both (`hippocampus.jsonl`, `striatum.json`), and nothing reads them: the ask loop calls the
  emitter for every question (`cubbyllm/reasoning/ask.py`, `_ask`).

Nick added the head: order 0068's route 5 picks each token of the trained 128k head from about 18,000 rows, in less
time than a full 32k head (decode step 21.603 ms against the 32k head's 21.686 and the full head's 22.759 at home;
the same on an L4), with the full head's token on 99.76% of held-out positions. It is wired into grilly2's
`generate` and the seats behind `CB_HEAD_SHORTLIST`, off by default. Before it is turned on where anyone talks to
Cubby, 0068 left the serving batteries to run.

Facts stay out of the weights (H-K7: about 1,000 sightings for exact recall on our own trunk; a fact the model
remembers is a latent source, and an answer resting only on it is held, `latent_only`, by design).

- **What it brings CubbyLLM:** a question Cubby has answered before comes back in about a tenth of a second instead of
  about three, still checked; every token the seats write costs what a 32k head's would, with the 128k vocabulary.
- **Over a conventional model:** a conventional model re-derives every answer from its weights each time, or caches
  text it cannot re-check. Here the remembered plan is re-run against the store and the VM, so a stale fact is caught
  the moment the store changes.
- **What it does for you:** ask Cubby something it has answered before, in the same words or others, and it answers
  almost at once, with the same proof behind it.

## Build
1. **Memory first in the ask loop** (`CB_ASK_MEMORY`, off by default):
   - `cubby_serve` loads the hippocampus and the striatum from a folder given on the command line (the files
     `sleep.py` writes), or starts them empty.
   - In `_ask`, after the host's own shapes (self, relation, scene, everyday, profile) and before the emitter: the
     striatum's order decides whether the hippocampus is tried first for this question's shape. Its candidates go
     through `learn_and_answer` exactly as an emitted plan does. The first certified one is spoken, with
     `how = "a remembered plan"` and the episode's id in the record; if none certifies, the emitter runs as today.
   - Each try's outcome goes back to the striatum (its reward rule, unchanged).
   - **A live write** (`CB_ASK_MEMORY_WRITE`, off by default): a certified answer becomes an episode at once, as
     `sleep.replay` would make it at night. `sleep` still retires an episode whose chain lost a fact.
2. **Route 5 on the served trunk:** `cubby_serve` attaches it (`grilly_seats.attach_route`) when `CB_HEAD_SHORTLIST=1`
   and an index file is given; a mismatched index is refused (that check exists). Greedy one-row decoding uses it; a
   decode with a processor (0081's trie) or sampling runs the full head, as grilly2 does today; count both.
3. **The bench** (`validation/exp_0083_repeat.py`): questions asked, then asked again; arms by flag, on the home card,
   the served stack as `cubby_serve.build` builds it.
   - **Sets:** the gen-3 held split, seed 7 (600, as 0081 reads it), asked twice in the same words; then H-E18's
     second writer's plan and chain wordings or r18's rephrased pairs where the dataset is present, for the
     other-words pass.
   - **Arms:** A today (all off); M `CB_ASK_MEMORY` with the live write; H `CB_HEAD_SHORTLIST`; M+H.
   - **The serving batteries** 0068 left (0050's list): the programs seat's greedy programs with the route against
     without, on every logged prompt; H-E8; H-E15's live eight.

## Acceptance (pre-registered)
- **Tests:** off is byte-identical to today (both flags); a recalled plan about another entity is refused by
  `covers()` and the emitter then runs; the striatum's order and reward on a scripted run; the live write produces
  the episode `sleep.replay` would; the route refused on an index over other head rows.
- **Numbers to report, per arm and pass:** answered, verified, correct, WRONG, refusals by reason; how many answers
  came from a remembered plan; the median and p90 time per question; for the route, how many decode steps used it and
  how many ran the full head (and why).
- **Gate, memory:**
  - the second asking in the same words gives the first asking's certified answer on every question it certified,
    with a median time at most 0.2 s;
  - the first asking's median time at most 0.05 s above arm A's (the cost of a recall that misses);
  - 0 wrong on every pass.
- **Gate, route 5:** the programs seat's greedy programs identical with and without the route on at least 99% of the
  logged prompts; H-E8 and the live eight unchanged; the decode step at most a full 32k head's (p50, home card).
- **Kill, memory:** any wrong answer from a remembered plan; memory stays off.
- **Kill, route 5:** the batteries under 99%; the switch stays off.
- **Reported, no gate:** the other-words pass (how often memory carries across a rewording).

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Memory proposes; the gate disposes. Nothing is spoken because it was remembered, and nothing in `plan_verify`, the
  walk or the VM changes.
- **Held-out discipline** (the hypothesis file, section 0): nothing is tuned on the sets read here; the striatum
  learns from the gate's outcomes, never from gold.
- **The index is the head's:** route 5's index is built over `base450m_cont_final`'s head rows. A new base (order
  0045's control) needs its index rebuilt (0068's builder) before the route serves on it.
- After 0081: build on its `Cubby450mEmitter` and `ask.py` changes, not beside them.
- The home card is shared: check the inbox before a long run.

## Out of scope
- Facts trained into the weights.
- Scoring only the allowed rows inside 0081's literals (0068's route 1): reported if 0081's trie is on, built later.
- Batches above 1 and sampled talk with the route (0068's open items).
- Turning either flag on where a person talks to Cubby: Nick's call, after the report.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
