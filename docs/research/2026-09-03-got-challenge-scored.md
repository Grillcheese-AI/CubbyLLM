# GoT challenge — scoring (2026-09-03)

Prompt: `docs/research/2026-09-03-got-challenge-prompt.md`. Each model answered once, alone. Scored per
idea with the prompt's §6 rubric against the measured record. **Every arXiv ID was resolved** (the Hub's
paper index for existence, arXiv's export API for the title where it answered; arXiv returned 503 for the
GLM batch, whose seven IDs all exist on the Hub and whose titles match the claims from memory): an ID that
resolves to an unrelated paper is a fabricated citation, −5 per use. The plan being improved is
`standin/plan_suggestions.md`.

Recurring misread to watch for: the prompt gives **claimed-answer precision 0.994** and **gold 0.76**. Kill
criteria written as "gold match below 0.99" confuse the two and are unusable as written; they are scored as
"kill criterion present but wrong" (falsifiability 1–2, not 0).

## 0. Pre-checks run the same day (free, on `validation/logs/cot_harvest_v3cf.jsonl`, 800 questions)

Several ideas came with an offline pre-check on data we already log. They were run before scoring, and
they decide more than the ideas do. Facts of the run: 763 parsed, 517 verified, 513 correct; failure
reasons `retrieval_exhausted` 246, `unparseable` 37, verified-but-wrong 4; `tau_ret` 0.5959; 1,899
counterfactual plantings over the verified chains.

| pre-check | result | verdict |
|---|---|---|
| (a) hops with ≥2 candidates above `tau_ret` (Qwen 3.8, idea 2) | **170 / 858 hops = 19.8%**; top-2 gap < 0.05 in 4.4%; median gap 0.092 | above its 10% kill line: branching has room |
| (b) accepted-hop verification rate by distance above the threshold (Qwen 3.8, "probably wrong" 2) | <0.02: 93% · <0.05: 97% · <0.10: 87% · <0.20: 87% · ≥0.20: 86% | **falsified**: no cliff, the score does not predict the verdict near the threshold |
| (c) per-relation score ranges of verified vs failed hops (MiniMax, idea 7) | min verified 0.60–0.77 vs max failed 0.88–0.99 on every frequent relation | **killed**: the ranges overlap completely |
| (d) share of not-correct answers whose gold sat in a top-3 runner-up (Qwen 3.8, idea 1) | 17 / 250 = **7%**; within 0.05 of the accepted fact: 1 | below its own 30% kill line: killed before training |
| (e) verified 2-hop fact pairs shared by ≥2 questions (fact-path skill mining, all models) | **1 of 219** | killed on the QA side, as Qwen 3.7's generality test predicted |
| (f) does the three-repair budget bind? (GLM, idea 2) | all 246 failures used all 3 repairs and verified **0 hops**; verified chains used 0 (514) or 1 (3) repairs; P(verified · repairs = 3) = 0/246 | the stop table is a **step function**: a repair budget of 1 loses nothing on this set and cuts two wasted re-walks on 31% of questions — but saves retrieval work, not VM calls (exhausted walks never reach the VM) |
| (g) are planted inverted-direction faults ever merge collisions? (GLM, idea 5) | **1,899 / 1,899 plantings caught**, all by `symbol_mismatch` (wrong entity 598, wrong relation 233, inverted direction 662, wrong hop order 406) | **killed** by its own rule: the symbol check rejects every direction flip, the guard is dead code |
| (h) sibling rate: hop positions with ≥2 logged candidates (GLM, "probably wrong" 1) | **858 / 858 = 100%** logged; 19.8% above the threshold | the fear (out-degree 1) is answered by materializing the logged runners-up as sibling nodes |
| (i) fact overlap across verified chains (GLM, idea 4) | 17 of 671 facts (3%) used by ≥2 chains; 81 / 517 chains touch a shared fact | cross-question reuse is bounded at ~16% of chains; the within-conversation case is untested |
| (j) relation-path templates, leave-100-out, seen ≥3× (GLM, idea 6) | 65 / 100 held-out chains subsumed overall; **8 / 36 (22%)** of the ≥2-hop chains, by (`instance`,`component`) 18×, (`occupation`,`instance`) 6×, (`parent taxon`,`instance`,`component`) 6× | passes its 20% kill line only just; 1-hop "templates" are the grammar, not skills |
| (k) hop grammar on causal phrasings (GLM, generality test) | "What led to X?", "Did A cause B?", "What caused X?", "What happened because of X?" do **not** parse; "What is the cause of X?" and "What is the impact of X?" do | prediction confirmed: 4 of 6 causal forms starve the frontier before any walk |

What the numbers say together: the chain path does not fail by choosing the wrong candidate; it fails by
**finding nothing above the threshold** (246 of 283 failures, every one with zero hops verified), and the
retrieval score is not the signal that separates verified hops from failed ones. The verifier is not the
weak point either: it caught all 1,899 planted faults. So the frontier walk fixes a minority case (19.8% of
hops have an alternative at all), skill reuse has little to reuse on this corpus, merge guards protect
against a fault the symbol check already rejects, and the experiment the data points at is ours: **lower
`tau_ret` and let the VM decide per hop** — since (b) shows candidates just above the threshold verify as
well as candidates far above it, candidates just *below* it may too. Metric: verified and correct out of
800 vs 517 / 513, cost in VM calls; kill if the wrong-but-verified count leaves 4. Free win from (f): the
repair budget goes from 3 to 1.

## 1. Qwen 3.7

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Harvest DAG mining (repair memo) | 2 | 2 | 2 | 2 | pass | — | **8** |
| 2 | Hormone-guided priority weights | 2 | 1 | 1 | 1 | pass | −5 (invented "endorphins") | 0 |
| 3 | Inter-world contradiction registry | 1 | 2 | 1 | 2 | pass | — | 6 |
| 4 | Deductive beam over hops | 0 (restates §2) | — | — | — | pass | — | 0 |
| 5 | Behaviour-signature dedup as a VM cache | 1 | 2 | 2 | 1 | pass | — | 6 |
| 6 | Cross-hop subgraph distillation into skills | 3 | 3 | 3 | 2 | pass | −5 ("Dendrite project", not found) | 11 raw / 6 |
| 7 | State-fork speculative pre-check | — | — | — | — | **fail** (§3.1) | −5 ("Dendrite" again) | 0 |

Citations: GoT 2308.09687, LINC (EMNLP 2023), DBS 2401.17686, PathFinder 2312.05180, MADAM-RAG
2504.13079, Clover 2310.17807, SYNVER 2410.14835 all resolve to the papers claimed. "Dendrite project"
(cited twice for O(1) KV-cache forks and "pattern consolidation in agent-native search engines") is not
found anywhere; "arXiv:2003.xxxx" is a placeholder; "AutoProof" is unspecific.

- **Idea 6** was the strongest idea in the answer and pre-check (e) killed it on the QA side: one shared
  2-hop pair in 800 questions. Its own generality test predicted exactly this. The program-AST version (the
  game's library, where consolidation already exists) is still open — see Qwen 3.8 idea 5 and GLM idea 6.
- **Idea 1 (repair memo)** survives as a small addition: the harvest's `banned_facts` pairs are the
  `Failure → Repair` edges; use them at inference when a walk lands on a fact banned in an earlier question.
  Bound: only 4 of 800 verified chains were wrong, and (f) says no failure ever reached a verify-stage ban
  on this run, so the memo's ceiling is small.
- Idea 4 restates §2's frontier walk; its `B ∈ {1, 3, 5}` sweep with "kill if <2 points at >50% more VM
  calls" is the experiment already planned, and (a) says the sweep touches 19.8% of hops.
- Idea 7 breaks §3.1 ("select the best pre-verdict candidate" is the model judging, or a VM run per
  candidate, which is idea 4). Idea 5's behaviour-signature cache is circular (the signature costs the VM
  run it claims to skip); the AST-hash cache is valid and already in §2. Idea 3 is GLM's era-stamped
  contradiction idea from 2026-08-28 as an `(entity, relation)` registry; its experiment cannot run on a
  single-store harvest. Idea 2 states our hormone set as "dopamine, serotonin, oxytocin, cortisol,
  endorphins" (ours: noradrenaline, not endorphins — the same embellishment the 2026-08-28 scoring
  recorded for this model).
- **Both "probably wrong" points are adopted as required GoT-1 measurements**: (1) the priority score's
  terms have no common scale — ablate each over the harvest set; if zeroing any term changes neither gold
  nor don't-know, the score is decoration; (2) graph mass grows without bound under retire-never-delete —
  measure node count and walk latency at 100 / 500 / 1,000 maze turns.

## 2. MiniMax M3 (free)

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Provenance-bounded hindsight replay | 2 | 2 | 3 | 2 | pass | −10 (2009.08483 is galaxy kinematics; 2006.03388 is speech features) | 9 raw / 0 |
| 2 | Counterfactual neighbourhood as a gate | 1 | 2 | 2 | 2 | pass | −5 (2007.00734 is an ion-gate paper) | 2 |
| 3 | State-forked speculative beams | — | — | — | — | **fail** (§3.1: "the model's scoring is forked") | — | 0 |
| 4 | Hormones as priority multipliers | 2 | 2 | 2 | 2 | pass | −5 (2104.13542 is STORM, robot MPC) | 3 |
| 5 | Program-lineage skill crystallization | 2 | 3 | 2 | 2 | pass | −10 (1604.04656 is ROC estimation; 1503.06267 is structural-damage Bayes) | 9 raw / 0 |
| 6 | World-margin disagreement ledger | 2 | 2 | 3 | 2 | pass | −5 (2104.08756 is an Arakelov inequality) | 4 |
| 7 | Retrieval threshold per relation | 3 | 2 | 3 | 2 | pass | −5 (2104.08756 again) | 10 raw / 5 — **killed by (c)** |
| 8 | Failure-root clustering | 2 | 2 | 1 | 2 | pass | −5 (2004.07560 is magnetic black holes) | 2 |

Citations: 2211.17192 (speculative decoding), 2305.10601 (ToT), 1706.04599 (calibration) and 2106.03046
(counterfactual reasoning for language understanding) resolve as claimed. **Eight of the twelve IDs resolve
to unrelated papers** (listed in the table), one of them used twice for two different claims. Every number
quoted from the prompt is quoted correctly; "about 800·3 = 2,400 VM calls" is a labelled estimate.

- **Idea 7** was the best idea of the four answers on paper and pre-check (c) killed it in one pass: on
  every frequent relation the lowest verified score (0.60–0.77) sits far below the highest failed score
  (0.88–0.99), so no per-relation interval separates them.
- Ideas 1 and 2 contradict each other on the sign of counterfactual survival: idea 1 down-weights an edge
  whose perturbation still verifies (right: a verifier blind spot); idea 2 gives a *bonus* to walks that
  survive perturbation (wrong: it rewards the misses). Pre-check (g) makes both moot on this run: nothing
  survived.
- Idea 3 forks the trunk so that each copy scores its candidate — the model as scorer, §3.1 — and puts
  the real trunk's exact fork on the stand-in, which has none.
- Idea 6's metric is the one to keep for the multi-world experiment: on a planted-contradiction subset the
  **don't-know rate must rise** (correct abstention) while gold does not fall. Idea 8's maze metric
  ("revisits to trap") misreads the game (traps are Cubby's own mines).
- "Probably wrong" 1 (the expansion caps 32/64/3/8/8 were set by a plan document, not measured; derive
  per-world depth caps from harvest statistics) and 2 (a *learned* goal-relevance weight is a second
  self-judge; keep the weights host-set from a discrete menu) are both right and go into the spec.

## 3. Qwen 3.8 2.4T

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Runner-up contrastive SFT from the harvest | 3 | 3 | 3 | 2 | pass | — | 11 — **killed by (d)** |
| 2 | Branch on retrieval ambiguity | 2 | 3 | 3 | 2 | pass | — | **10** — (a) passed |
| 3 | Nodes that carry decode snapshots | 3 | 2 | 3 | 2 | pass | — | **10** |
| 4 | Planted-fault repair training | 3 | 2 | 3 | 2 | pass | — | **10** |
| 5 | Skill extraction by lineage unification | 2 | 3 | 3 | 2 | pass | — | 10 — QA side killed by (e); program side open |
| 6 | Hormones gate the search budget | 2 | 2 | 3 | 2 | pass | — | 9 |
| 7 | Cross-world contradiction arbitration | 3 | 2 | 3 | 2 | pass | — | **10** |

Citations: STaR 2203.14465, ToT 2305.10601, GoT 2308.09687, LATS 2310.04406, Reflexion 2303.11366,
Voyager 2305.16291 resolve as claimed; Mitchell et al. 1986, Doya 2002, FEVER 2018, AGM 1985 are real.
DreamCoder is PLDI 2021, not ICLR 2023 (a venue slip, the paper exists). No fabrication; every experiment
is an offline replay first, then a live run, with all four rubric elements present.

- **Idea 1** proposed its own pre-check and failed it: gold sits in a top-3 runner-up for 7% of the
  not-correct answers (kill line 30%). Failures are retrieval misses, "what would make me wrong" as
  written. Had it passed, the harvest schema's rule (organic near-misses are the calibration/test split)
  would have confined the mining to the train half.
- **Idea 2** is the frontier walk with the piece §2 lacked: a **task-defined choice-point criterion** (≥2
  candidates above the threshold, or a top-2 gap below a margin) instead of the token-entropy criterion
  H-P2 found unusable. Pre-check (a): 19.8% of hops qualify. The branch decision is the host's; the model
  still only writes programs. **Build this first** — bounded by §0: it touches the 19.8%, and none of the
  246 exhausted walks.
- **Idea 3** — fork handles on nodes so backtracking is a restore, not a re-decode — is measurable on day
  one on the stand-in (llama.cpp has state save/restore); kill at >100 ms restore; ~200 MB under the
  32-node cap. The first proposal that makes a retained rejected branch *resumable*, which the rendered
  think-trace needs.
- **Idea 4** turns the counterfactual harvest into repair records, verifier-labelled. With 4
  verified-but-wrong chains in 800 and every failure exhausted at the walk (f), the repair budget is where a
  gain would show, not accuracy — and (f) already sets the budget to 1.
- **Idea 7** adds the step the other contradiction ideas lacked: derive the *discriminating hop* and ask
  both worlds through the router; until resolved the don't-know contract holds. Its experiment needs the
  two-world harvest we do not have yet; MiniMax's abstention metric is the right score.
- Idea 5: the unification rate over the program library is the pre-check (kill <10%); (e) already says the
  fact-path side is empty. Idea 6 targets the *budget* rather than the priority weights, cleaner than the
  multiplier proposals.
- **"Probably wrong" 1 is the best single correction in the round**: merging branches on entity identity
  conflates provenance. Merge on verified-subgraph isomorphism *and* agreeing verdicts. Adopted into the
  plan's merge rule. **"Probably wrong" 2 was tested and falsified** by (b).
- Generality test: causal-chain QA over dated history, with the prediction that chains will be symbol-valid
  yet temporally inverted because neither the hop grammar nor the ground check carries a temporal
  predicate — checkable by planting 50 inverted chains through the VM. The fix order given (temporal guard
  in the ground check, then depth, grammar last) is right — except that GLM's test below shows the grammar
  breaks *first*.

## 4. GLM 5.3

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Relative-entropy fork gating | 2 | 2 | 3 | 2 | pass | — | 9 — not run (needs decode-entropy logging on the GPU) |
| 2 | Harvest-mined stopping rule | 2 | 2 | 3 | 2 | pass | — | 9 — (f): **adopted as "repair budget 1"**; its VM-call metric is 0 |
| 3 | Planted-fault contrastive SFT | 2 | 3 | 3 | 2 | pass | — | 10 |
| 4 | Dependency-tracked re-verification | 3 | 3 | 3 | 2 | pass | — | **11** — (i) bounds cross-question reuse; in-conversation case open |
| 5 | Direction-safe merge guard | 2 | 2 | 3 | 2 | pass | — | 9 — **killed by (g)**, as its own kill rule says |
| 6 | Verified-subgraph skill compilation (slot templates) | 2 | 3 | 3 | 2 | pass | — | 10 — (j): 22% on multi-hop, at its 20% line |
| 7 | Hormone-scaled search widths | 1 | 2 | 3 | 2 | pass | — | 8 — fourth arrival |

Citations: Holtzman 1904.09751 (ICLR 2020), Medusa 2401.10774, CALM 2207.07061 (NeurIPS 2022), DPO
2305.18290, "Let's Verify Step by Step" 2305.20050, DreamCoder 2006.08381 (PLDI 2021, the correct venue),
Pathak 1705.05363 (ICML 2017), Doyle 1979 and de Kleer 1986 — all real, all on topic. No fabrication;
every quote from the brief is verbatim; the cost lines are priced against §3.5 as asked.

- **Idea 2** is the cheapest win of the whole round once (f) is read: the three-repair budget binds on
  every failure, no failure ever verified a hop, and no verified chain needed more than one repair. The
  stop table it wanted to learn is a step function, so the rule is one line: **repair budget 3 → 1**. By its
  own metric (VM calls at fixed precision) it scores nothing — exhausted walks never reach the VM — but it
  removes two wasted re-walks on 31% of questions. Its caveat "if the budget rarely binds, nothing to
  save" was the wrong worry: it always binds.
- **Idea 4** (a truth-maintenance system over `Verification` nodes: dependency sets, invalidation by
  `supersedes`/`contradicts`, recompute only dependents) is the most general idea submitted by anyone and
  the only one that gives conversations continuity. (i) says cross-question reuse on the harvest is small
  (3% of facts shared); its experiment — answer, mutate one fact, re-ask, 100 cases — tests the case that
  matters and has not been run. The store-snapshot hash already exists as the invalidation key.
- **Idea 5** proposed its own replay and died on it: all 662 planted direction flips were caught by the
  symbol check. "A cheap kill is a feature", as written.
- **Idea 6** reframes skills as **relation templates with entity slots** rather than shared fact pairs —
  the framing that survives (e): 65% of held-out chains match a template seen ≥3×, 22% of the multi-hop
  ones. Whether slot-binding certification is cheaper than a re-walk is the open question; the 1-hop
  templates that dominate are the grammar restated.
- Idea 3 converges with Qwen 3.8's idea 4 (pairs labelled by the VM, DPO-shaped); two cautions: the harvest
  schema reserves the planted faults for calibration, so training needs its own planting; and (g) shows the
  VM label on every planted fault is "rejected", so the pairs teach the four mutations unless beam-rejected
  candidates are mixed in — the answer's own caveat.
- Idea 1 supplies a relative criterion for H-P2 (fork where entropy exceeds a multiple of its running
  median); its falsifier needs decode-entropy logging, which is GPU work. (a)–(b) already suggest
  divergence is retrieval-defined, which is its stated way of being wrong.
- **"Probably wrong" 1** — telemetry from a greedy walk gives a graph with out-degree 1 — is exactly right
  and (h) resolves it: every hop position has logged runners-up (100%), so siblings are materialized from
  the log before any behaviour change. **"Probably wrong" 2** — boundary-only SFT drifts the trunk toward
  silence because abstention is the safe optimum under the don't-know contract — is the sharpest
  observation of the round; the tripwire is the serve self-test's three numbers across rounds, **which
  have not been re-run since v3**. Adopted: the self-test runs every round from now on.
- Generality test: causal why-questions; prediction that the grammar breaks first, confirmed by (k) — four
  of six causal phrasings do not parse, and the failure would present as an inflated don't-know rate that
  invariant 3 makes look safe. Second break (the symbol verifier on free-text cause/impact fields) is
  untested.

## 5. Gemini 3.8 Flash

_(pending)_

## 6. Across the four answers

- **Convergence** (skill extraction ×4, hormones as search control ×4, contradiction handling ×3, fork-based
  branching ×4) is the plan reflected back, not evidence. Two of the four fork proposals put the model in
  the scorer's seat and fail §3.1; Qwen 3.8 and GLM kept the VM as the scorer.
- **The pre-checks moved more than the ideas did.** Four ideas that scored 9–11 on paper are dead on the
  real harvest (per-relation thresholds, near-miss contrastive SFT, fact-path skill mining, the merge
  guard), one "probably wrong" was falsified (the threshold cliff), and two predictions were confirmed
  (the causal grammar gap; siblings exist in the log). The data point at a different bottleneck: retrieval
  exhaustion on 31% of questions, which no branching, skill, threshold table or guard addresses.
- **Ranking by what survived:** GLM 5.3 and Qwen 3.8 (clean, every idea with an offline falsifier, one
  free win each: the repair budget and the merge rule), then Qwen 3.7 (one strong idea killed, two required
  measurements), then MiniMax M3 (eight fabricated citations; one metric and two critiques kept).
- **Build order that survives the numbers:** (1) repair budget 3 → 1 and the VM-gated lower `tau_ret`
  experiment (ours, from §0); (2) Qwen 3.8's ambiguity-triggered frontier walk with its merge rule
  (isomorphism + agreeing verdicts) and GLM's materialized siblings, measured on the 19.8% of hops it can
  touch; (3) fork handles on nodes, restore latency on day one; (4) GLM's dependency-tracked
  re-verification on the 100 mutate-and-re-ask cases; (5) slot-template skills only if slot certification
  beats a re-walk on the 22%; (6) repair / contrastive records after a training-only planting; (7) the
  multi-world contradiction experiment when a two-world harvest exists, scored by correct abstention.
  Required measurements from the "probably wrong" lists: the serve self-test every round (the don't-know
  tripwire), score-term ablation, graph growth vs walk latency, per-world depth caps from harvest
  statistics, host-set discrete weights, the 50 temporally inverted chains and a 50-question causal parse
  set before any causal-history world ships.
- **Fabrications, do not reuse:** Qwen 3.7's "endorphins" and "Dendrite project"; MiniMax's eight arXiv
  IDs (2009.08483, 2006.03388, 2007.00734, 2104.13542, 1604.04656, 1503.06267, 2104.08756 ×2,
  2004.07560), all resolving to unrelated papers. Qwen 3.8 and GLM 5.3: none.
