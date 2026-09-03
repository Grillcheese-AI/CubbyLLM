# GoT challenge — scoring (2026-09-03)

Prompt: `docs/research/2026-09-03-got-challenge-prompt.md`. Each model answered once, alone. Scored per
idea with the prompt's §6 rubric against the measured record; every citation was checked by hand; a
citation that cannot be found is treated as fabricated until the model points to it. The plan being
improved is `standin/plan_suggestions.md`.

Recurring misread to watch for in all four answers: the prompt gives **claimed-answer precision 0.994** and
**gold 0.76** (the serve self-test). Kill criteria written as "gold match below 0.99" confuse the two and are
unusable as written; they are scored as "kill criterion present but wrong" (falsifiability 1–2, not 0).

## Qwen 3.7

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Harvest DAG mining (repair memo) | 2 | 2 | 2 | 2 | pass | — | **8** |
| 2 | Hormone-guided priority weights | 2 | 1 | 1 | 1 | pass | −5 (invented "endorphins") | 0 |
| 3 | Inter-world contradiction registry | 1 | 2 | 1 | 2 | pass | — | 6 |
| 4 | Deductive beam over hops | 0 (restates §2) | — | — | — | pass | — | 0 |
| 5 | Behaviour-signature dedup as a VM cache | 1 | 2 | 2 | 1 | pass | — | 6 |
| 6 | Cross-hop subgraph distillation into skills | 3 | 3 | 3 | 2 | pass | −5 ("Dendrite project", not found) | **11 raw / 6** |
| 7 | State-fork speculative pre-check | — | — | — | — | **fail** (§3.1) | −5 ("Dendrite" again) | 0 |

**Adopt (cheap, CPU-only, on the harvest set):**

- **Idea 6 — skill mining.** Recurring verified sub-paths (two or more hops shared by several questions)
  become `Skill` nodes offered as a single candidate the next time their input facts are present. Two
  conditions the answer left out, both required by the invariants: a skill is **re-verified on the VM at use**
  (a cached verdict is not a verdict), and it is invalidated by the store-snapshot hash. Experiment as
  written is sound once moved from "the 25 held-out chains" to the 800-question harvest: count sub-paths
  shared by ≥2 questions, replay with the shortcut, metric VM calls saved at unchanged gold, kill if
  <20% of mined skills reproduce their path or net VM savings <15%. Its own generality test predicts the
  failure mode (sparse repetition outside the maze) — that is the first number to get.
- **Idea 1 — repair memo.** The harvest's `banned_facts` (rejected fact → replacement) already are the
  `Failure → RepairProgram` edges; the addition is *using* them at inference: when a walk lands on a fact
  that was banned in an earlier question, inject its recorded replacement before spending a repair. Metric
  VM calls per answered question at unchanged gold; kill if <5% fewer calls. The 0.99 kill line is the
  misread above. Prior art checks out (GoT 2308.09687, LINC EMNLP 2023, DBS 2401.17686).
- **"Probably wrong" #1 — the priority score's terms have no common scale** (a verifier verdict in [0,1],
  a cosine, a pellet count, raw integers). The proposed test is exactly right and goes into the GoT-1 spec
  as a required measurement: ablate each term over the harvest set; if zeroing any single term changes
  neither gold nor don't-know, the remaining terms already decide and the score is decoration.
- **"Probably wrong" #2 — graph mass grows without bound** under retire-never-delete. Also right: the
  per-turn caps bound density, not storage. Required measurement: node count and walk latency after
  100 / 500 / 1,000 maze turns; the game's per-level consolidation is the existing answer for programs,
  nothing consolidates observations yet.

**Not adopted, with the reason:**

- Idea 4 is §2's frontier walk (per-hop certification, merge on shared entity, depth-then-score order,
  budget stop) restated with a beam width; the `B ∈ {1, 3, 5}` sweep is the experiment already planned and
  its kill rule (<2 points at >50% more VM calls) is a reasonable version of ours. Prior art is real (DBS
  2401.17686, PathFinder 2312.05180; "self-evaluation guided beam search", NeurIPS 2023, is the
  LLM-as-judge shape §3.1 forbids, cited here as prior art only).
- Idea 7 breaks §3.1: "select the best pre-verdict candidate" is either the model judging its own proposals
  (forbidden) or a VM run per candidate (then it is Idea 4 and saves nothing). It also puts the real
  trunk's exact-state fork on the stand-in, which has no such fork. The 6 MB snapshot figure is quoted
  correctly.
- Idea 5's behaviour-signature cache is circular: a behaviour signature is computed by executing the
  program on the eval suite, which is the VM call it claims to skip. The normalized-AST hash as a pre-VM
  cache is valid, cheap, and already in §2's identity layers; at millisecond VM calls the saving is not
  the bottleneck. SYNVER 2410.14835 not verified; Clover 2310.17807 checks out.
- Idea 3 is GLM's era-stamped contradiction idea from the 2026-08-28 contest plus the agenda's friction
  index, concretized as an `(entity, relation)` registry with an `inconclusive` verdict (compatible with the
  don't-know contract: inconclusive is never spoken as fact). Its experiment cannot run where it says: the
  800-question harvest is a single-store run, so there are no cross-world tuples in it; it waits for the
  multi-world harvest. MADAM-RAG 2504.13079 checks out.
- Idea 2 names the hormones as "dopamine, serotonin, oxytocin, cortisol, endorphins"; ours are dopamine,
  serotonin, cortisol, oxytocin, noradrenaline — the prompt did not list them, and the answer presented a
  guess as a fact (the same embellishment the 2026-08-28 scoring recorded for this model). The mechanism
  itself does not break §3 (the ODE already scales the routing threshold, and P(n) weights are exploration
  control, not facts); it becomes a free serve-time experiment once P(n) exists, after "probably wrong #1"
  has settled the scales. "arXiv:2003.xxxx variants" is a placeholder, not a citation.

**Generality test:** "Wikidata-derived triples about historical figures" is our home domain (the harvest
set *is* Wikidata-shaped triples), so it tests nothing outside the maze; but its prediction — skill mining
finds fewer than three shared sub-paths per 100 questions — is a concrete number and contradicts Idea 6's
premise. That contradiction is the experiment.

**Fabrications and filler:** "endorphins" as one of our hormones; the "Dendrite project" ("O(1) fork latency
via CoW KV-cache management", "pattern consolidation in agent-native search engines") cited twice and not
found anywhere; "arXiv:2003.xxxx"; "AutoProof" (unspecific). All other checked IDs resolve.

**Net from Qwen:** two buildable additions (skill mining with re-verification, repair memo), two required
measurements for the GoT-1 spec (score-term ablation, graph growth), one experiment design we keep (the
beam-width sweep's kill rule). Convergence with §2 elsewhere is the plan reflected back, not validation.

## Gemini 3.8 Flash

_(pending)_

## GLM 5.3

_(pending)_

## MiniMax M3

_(pending)_
