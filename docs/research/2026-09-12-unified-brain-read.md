# 2026-09-12 — reading GrillCheese alpha's `unified_brain.py` for CubbyLLM

Read in full (1,678 lines) on request: what the alpha's brain orchestrator does, and what of it belongs in
CubbyLLM's eight layers — judged against the seven invariants (the VM is the only truth gate; the model
proposes, the host disposes; the don't-know contract; retire, never delete; the 12 GB serve budget; no
self-play; ported, never linked). Nick's framing: most of it integrates at the SNN level, not as host Python.

**What it is.** One `process(text, embedding)` call runs the input through: place/time cells over a 2-D
projection of the embedding (every 5th call), a thalamus (gate + 4 routes), an amygdala/limbic pass
(valence, arousal; "call this emotion X" teaching; "remember X" → a protected memory), memory retrieval (k=3,
a relevance floor of 0.3, an optional Phi-3 reranker scoring 1–5, off by default), experiential recall
(similar past interactions by cosine, quality-filtered), CNS (stress, consciousness level), endocrine
(hormone levels), basal ganglia (strategy + go/no-go from routes, urgency, inhibition), a specialist per
routed domain, a Hebbian emotion→embedding weight update, and a combined modulation vector (warmth, energy,
empathy, creativity, caution, detail, focus, stability) that shapes the response. `provide_feedback(quality)`
trains the strategy choice, stores significant experiences (significance = emotional intensity × outcome
surprise), and nudges the amygdala online. `consolidate()` runs in idle time. A temporal-knowledge system
answers "what if X hadn't happened" by removing an event and listing prevented effects, and "what happens
next" by pattern. Two prompts describe the AI's own state to the LLM.

**Already in CubbyLLM, in its validated form** (nothing to port; the mapping is the record):

| alpha | CubbyLLM |
|---|---|
| thalamus gate + routes | the MoWM router: centroid routing with a margin gate, route-vs-spawn (exp_m3 open set, 0.680 macro), `CubbyBridge` |
| specialist registry with maturation | the MoWM worlds; the taxonomy tag as the harvest's domain label (measured, unwired) |
| experiences + `recall_similar_experiences` | the hippocampus: episodes, codes, `recall`, `utility` (exp_r13/r14) |
| `consolidate()` in idle time | `Hippocampus.consolidate(keep)` — retire, never delete; facts in the world store |
| "remember X" as a protected memory | provenance `source=user_explicit`, retire-never-delete; the VM's `REMEMBER`/`STORE` |
| significance gate on what to store | the harvest's rule: a refusal that became a verified answer after learning is the record worth keeping |

**Worth bringing over as mechanisms** (each is a measurable lever, none touches the truth gate):

1. *Basal ganglia as proposer arbitration.* The alpha selects a strategy from routes and learns from feedback.
   CubbyLLM now has four proposers — the grammar, the hippocampus, the emitter, a frontier model in probes —
   tried in a fixed order. A selector that learns which proposer to try first per question shape, with the
   VM's verdict as the reward (deterministic, no self-play, no judge), is the same circuit with an honest
   reward. Go/no-go stays the disposer's.
2. *Time cells → temporal qualifiers on facts.* The alpha's time cells are a temporal code with nothing to
   bind. CubbyLLM's open hole is the qualifier the plan cannot bind ("as of 2022", "in 1863" — refused as
   unbound constraints; Wikidata's point-in-time qualifiers are dropped by the source). A fact that carries a
   valid-time interval, and a walk that binds the question's time constraint to it, turns those refusals
   into answers; `TEMPORAL_BIND` is the VM opcode reserved for it (a `trace_structural` stub today).
3. *Caution → budgets.* The alpha's modulation vector shapes tone. The one modulation that belongs in the
   reasoning layer is caution/complexity setting the hop budget and the tau floors — Nick's "more hops on
   complexity" is the same knob from the other side. A state variable, on the record, never a judgement.
4. *Emotion-biased retrieval → an affect component in the episode code.* Optional and measurable: bundle an
   affect code into the hippocampus's DG so recall can be cued by state as well as content. Nothing in the
   current benchmarks needs it; the chat layer might.
5. *Counterfactual by event removal.* The alpha walks a causal graph and lists prevented effects. Under the
   invariants that is a certified traversal of a causal store with provenance (the MoWM `causal_graph`,
   the prospection experiments' forks), spoken as a possibility branch, never as a fact. The "predict the
   future by pattern" half stays out.

**Must not come over as is:**

- The Phi-3 reranker (an LLM scoring relevance 1–5) is an LLM judge — the invariant it violates is the
  first one. It is off by default there and stays out here; the hippocampus ranks by Hamming distance and the
  disposer decides.
- The `[MY_STATE]` prompt instructs the model to assert feelings as facts ("NEVER say I don't experience
  emotions"). CubbyLLM's talk layer can report internal state as state — "caution is high because the last
  three chains were refused" — with the variables on the record; it should not be instructed to claim more
  than the variables say. Honesty is the product.
- The Hebbian emotion→embedding weights as a store of anything: the July measurements (exp_a_forgetting)
  are the reason the hippocampus is an explicit code store.

**What belongs at the SNN level** (Nick, 2026-09-12): the cells and the chemistry — place/time/circadian
cells, the amygdala's valence/arousal, the endocrine and CNS state, Hebbian association — are neuron-level
dynamics, and in CubbyLLM they live in the substrate (grilly / cubemind), where they can modulate the talk
adapter and the budgets above through the symbolic boundary: (symbol, similarity) and state variables cross;
raw activations do not. The reasoning layer sees their outputs as knobs, never as truth.
