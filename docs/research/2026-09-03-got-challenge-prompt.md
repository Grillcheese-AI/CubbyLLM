# Challenge prompt — "make Cubby's verified graph-of-thought better and more general" (2026-09-03)

**How to run it (owner's notes, not part of the prompt).** Four models on OpenRouter — Gemini 3.8 Flash,
Qwen 3.8 2.4T, GLM 5.3, MiniMax M3 — each gets the text below the rule **independently, once, in its own
fresh session** (the 2026-08-28 contest put everyone in one room; from round 2 on every answer was the
transcript reflected back, and the only independent signal was round 1). Default temperature. Score the four
answers with the rubric at the end of the prompt against the measured record; the previous scoring doc
(`docs/research/2026-08-28-oracle-competition-scored.md`) is the template. Convergence with our plan is not
validation. Anything quoted back must be verbatim; anything else is either fabricated or the model's own.

---

You are one of four models answering the same brief independently. We are a small research lab (Grillcheese
Research Lab) building **Cubby**, a language model that can only *speak* through programs a verifier accepts.
We have a concrete plan to add a graph-shaped thinking process on top of it. We do not want the plan
validated; we want it **made better and more general** by ideas we have not had. Everything below is either
built and measured or an explicit plan; nothing is aspirational marketing. Treat it as the ground truth you
are allowed to quote, and quote it verbatim if you quote it at all.

## 1. What exists (built and measured)

**The trunk speaks only through the VM.** A language model (a 2B trunk is training; a fine-tuned 2.6B
open model in GGUF stands in for it today) never answers a user directly. Each turn is a program in
**CubeLang**, a small typed language with a Rust VM. The chat program implements an `IAgent` contract
(`think` / `act` / `observe`): the host *offers* candidate lines through an `ASK`, the VM's `resume`
rejects anything that was not offered ("chosen, not invented"), and the spoken line is the VM's output. When
nothing verified is available the model says one verbatim line: "I am sorry, my training is not finished, I do
not have that information yet."

**Reasoning cortex (the chain path).** A question is parsed by a small grammar (95.4% coverage of our
corpus) into hops. A **greedy walk** retrieves one accepted fact per hop over a fact store (retrieval
threshold 0.5959; first candidate above it wins). The model then writes the CubeLang chain program (a
`CotChain` opening is prefilled), the VM executes it hop by hop and verifies the whole chain at a symbol
threshold (0.2202); a **ground check** and a **consistency gate** (a verified walk that disagrees with the
model's answer vetoes it) sit before speech. A failed verify bans the offending fact and re-walks once, within a
budget of three repairs. Measured: claimed-answer precision **0.994** on the verified path; an absent-role
control **341/341** (the verifier never certifies a role that is not there); "chasing" an unverified answer was
tried and **falsified**. Serve self-test on 25 held-out chains with the facts stripped from the prompt: gold
**0.76**; spoken lines **76% correct / 20% don't-know / 4% wrong**.

**Worlds.** Facts live in named worlds (a fact store each). Routing is by axiom centroids with a margin
gate; a world that lacks the answer asks the next world through the router (inter-world delegation,
measured). Plugins mount worlds and "cortices" on top of the trunk (MindForge style: the plugin imports us,
never the reverse).

**Embodied learning: cubby-man.** Cubby plays a 3D Pac-Man maze (the "cubbyverse"). He starts with two
facts. Every move is VM-mediated (the ASK offers the exits; refused moves become wall facts); pellets,
ghosts, stars and traps become discovery facts at arrival time through the same anti-poisoning gate as user
facts. He plans by BFS over *his own* facts, flees ghosts with a fear radius scaled by a **neurochemistry
ODE** (five bounded hormones; they modulate routing thresholds and tone, never facts), rests to recover
energy in safe cells, and lays ghost traps (one per level, cumulative). He keeps a persisted
**ProgramLibrary** of VM-certified moves with lineage (modify a parent before inventing; consolidate per
level into one `Moves` program with a function per combo; **retire, never delete**; every entry carries a
reasoning record: why, because, the situation numbers, the rationale). A **ToolForge** lets the trunk write
*arbitrary* CubeLang for live decisions (flee / safer-exit compare / where-chain), certified on the VM
before he acts, kept pass or fail. One round of game-generated, VM-verified SFT data took forge acceptance
from **0.75 / 0.00 / 0.50** (decision / compare / chain) to **1.00 / 1.00 / 1.00**, with **24/24** on a held-out
level. He thinks out loud: every step produces a first-person thought from the real decision data, verbalized
by the model only when the numbers and names survive.

**The harvest stores the tree, not the path.** Each answered question logs the per-hop query, fact, retrieval
score, VM verdict, the top-k runners-up at every hop, the banned facts with their replacements, the failure
reason, repairs used, and a **counterfactual neighbourhood**: four planted-fault classes (wrong entity, wrong
relation, inverted direction, wrong hop order) re-planted on the accepted triples, each recovered through the
VM and logged per hop. 800 questions per run, with a store-snapshot hash and git revision on every record.
Nothing walks this tree yet.

**Trunk-side prospection (measured on the real trunk).** The decode state **forks exactly at constant
cost** (a snapshot is ~6 MB because of attention KV caches). Next-token entropy localizes choice points on
toys, but 70–90% of Wikipedia steps sit above 1–3 nats, so "branch only at choice points" needs a relative
criterion. The recurrent layers are short-range mixers; all long-range work lives in the windowed-attention
layers and an episodic store. Chrono init was tried and killed (+0.444 nats).

**Training loop.** Stand-in SFT rounds v3→v6: verified programs (arithmetic, kernels, role binding,
chains, the game's families), chat, content awareness, emotion (GoEmotions + Plutchik), affect
(valence/arousal), history (dated events, NYT recall and dating scored within ±5 years, the era of a book
passage), all under an identity system prompt with a sampled hormonal state. The eval is VM-verified,
never self-reported.

## 2. The plan we want you to improve (condensed)

Graph-of-thought as an **orchestration layer over the verified graph**, not as free thought generation:

- A typed graph store (versioned, in-memory + JSON/SQLite): nodes `Observation, Fact, Question,
  Hypothesis, Plan, ProgramCandidate, ExecutionResult, Verification, Failure, Reflection, Skill,
  TrainingExample`; typed edges `supports, contradicts, depends_on, derived_from, tests, refines, mutates,
  retrieves, causes, supersedes, uses_skill`; every node carries provenance, parents, status, score,
  version.
- **The host constructs and validates the graph; the model only proposes** nodes and transformations (a
  hypothesis, a missing edge, a program mutation, a goal decomposition, a failure explanation, a skill
  abstraction, a ranking of candidates). The host owns identity/canonicalization, provenance, execution, VM
  certification, reward, contradiction detection, retirement, what the model sees, and what enters the SFT
  corpus.
- The existing trace events (sense, route, walk, emit, vm, gate, learn, explore, forge) wrap into nodes first —
  telemetry before behaviour change.
- The flat `Facts:` block becomes a **bounded graph neighbourhood** (goal, current state, top facts, typed
  edges, contradictions, candidate operations, provenance), and the model returns a small typed operation
  (a program), never a narrative.
- **Bounded expansion**: max 32 nodes, 64 edges, depth 3, 8 candidates, 8 VM runs per turn (an earlier
  live failure: above ~100 ASK candidates the VM saw altered duplicates; 16 was fine).
- A **verifier-aware beam**: expand → normalize → deduplicate → execute → score → retain top B; the VM is
  the scorer. For the chain path: a frontier of hop states, every candidate above the retrieval threshold is
  a child, each child certified per hop, branches reaching the same entity **merge**, ordered by verified
  depth then retrieval score, stopping at a verified full chain or the VM-call budget.
- Program identity in three layers: source hash, normalized-AST hash, behaviour signature on the eval suite.
- Priority `P(n) = w_g·goal_relevance + w_v·verifier_confidence + w_r·expected_reward + w_u·novelty −
  w_c·cost − w_k·contradiction_risk − w_d·redundancy`; a self-update is accepted only with a held-out
  generalization term (`ΔL_target + λ·ΔL_heldout − μ·regressions − ρ·complexity`).
- SFT records only at **verified boundaries**: the selected subgraph, the winner, the rejected candidates
  with reasons, the verifier result, the minimal fix, whether it survived a held-out test.
- Milestone GoT-1: the agent maintains a typed, provenance-preserving graph of observations, facts,
  candidate programs, executions, failures and verified skills; selects among bounded alternative
  derivations; produces an SFT record only after held-out validation. The think trace is rendered as the
  graph (nodes with verdicts, edges with relations, rejected branches kept).
- Measurement: the 800-question harvest set, greedy vs graph — gold match, don't-know rate, VM calls;
  kill if no gain at ≤3× the VM calls.

## 3. Invariants (an idea that breaks one is rejected, however clever)

1. The VM is the only truth gate. The model never scores or judges its own thoughts; there is no
   LLM-as-judge anywhere in the loop.
2. The model proposes, the host disposes: identity, provenance, execution, reward, retirement and what the
   model is shown are host decisions.
3. The don't-know contract: an unverified claim is never spoken.
4. Retire, never delete; every record carries a store-snapshot hash and a git revision.
5. Serve-time budget: a 2.6B model on a 12 GB consumer GPU, VM calls in milliseconds; training on one
   80 GB GPU for tens of minutes per round.
6. No self-play or self-judging shortcuts (a differentiable "make the trunk VSA-bindable" objective was
   already shown to Goodhart into collapse; we are allergic).
7. Nothing depends on another repository at runtime: reusable pieces are ported, never linked.

## 4. What we want from you

Give **five to eight ideas**, each in exactly this shape:

- **Name** (≤6 words).
- **Mechanism** (≤120 words): what it does, in terms of our nodes, edges, walks, VM calls, worlds,
  hormones, program lineage or SFT records.
- **What it changes** in the plan above (which bullet), or what it adds.
- **Why it is more general**: how the same controller serves at least one domain outside the maze — a
  user's factual question over worlds, a causal graph over history (our dated events carry cause and
  impact fields), multi-world disagreement, the real 2B trunk with exact state forks, or a task family you
  name.
- **The cheapest falsifying experiment on OUR substrate** (the 800-question harvest set, the maze with its
  held-out levels, or one stand-in SFT round): the metric, the baseline it must beat, the kill criterion,
  and the cost in VM calls, GPU-hours and days.
- **Prior art**: arXiv IDs or venue+year you are confident exist; otherwise write `no citation`.
- **Confidence** (low / medium / high) and **what would make you wrong**.

Then add:

- **Two things we are probably getting wrong** in the plan, each with the observation that would expose it.
- **One generality test**: a task family outside the maze where GoT-1 should run *unchanged*, and the
  first thing you expect to break.

## 5. Rules

- Do not restate the plan back to us. Restatement scores zero.
- Do not invent numbers, results, components or citations. The previous contest produced a
  "Kalman-estimator verifier", a "350× cost-per-query" figure and a "0%→95% needle recall" — none existed,
  and they cost their author the round. Say `unknown` when you do not know.
- Prefer ideas that reuse what exists (the harvest tree, the program lineage, the hormones, the worlds, the
  exact state fork) over new machinery.
- No marketing language, no product names for things that are not built.
- At most 2,500 words. Markdown. Use the headings `Idea 1` … `Idea N`, `Probably wrong`, `Generality test`.

## 6. How the four answers will be scored (per idea)

| criterion | points |
|---|---|
| novelty relative to §2 | 0–3 |
| generality beyond the maze (concrete, not asserted) | 0–3 |
| falsifiability (metric + baseline + kill criterion + cost, all four present) | 0–3 |
| cost realism against §3.5 | 0–2 |
| invariant fit (§3) | pass / fail (fail = 0 for the idea) |
| fabricated number, result or citation | −5 each |

The winning ideas get built and measured; the measured result, not the idea, is what we keep.
