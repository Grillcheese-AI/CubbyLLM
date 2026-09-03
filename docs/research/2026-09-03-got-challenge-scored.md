# GoT challenge — scoring (2026-09-03)

Prompt: `docs/research/2026-09-03-got-challenge-prompt.md`. Each model answered once, alone. Scored per
idea with the prompt's §6 rubric against the measured record. **Every arXiv ID was resolved** (the Hub's
paper index first, then arXiv's export API for the rest): an ID that resolves to an unrelated paper is a
fabricated citation, −5 per use. The plan being improved is `standin/plan_suggestions.md`.

Recurring misread to watch for: the prompt gives **claimed-answer precision 0.994** and **gold 0.76**. Kill
criteria written as "gold match below 0.99" confuse the two and are unusable as written; they are scored as
"kill criterion present but wrong" (falsifiability 1–2, not 0).

## 0. Pre-checks run the same day (free, on `validation/logs/cot_harvest_v3cf.jsonl`, 800 questions)

Several ideas came with an offline pre-check on data we already log. They were run before scoring, and
they decide more than the ideas do. Facts of the run: 763 parsed, 517 verified, 513 correct; failure
reasons `retrieval_exhausted` 246, `unparseable` 37, verified-but-wrong 4; `tau_ret` 0.5959.

| pre-check | result | verdict |
|---|---|---|
| (a) hops with ≥2 candidates above `tau_ret` (Qwen 3.8, idea 2) | **170 / 858 hops = 19.8%**; top-2 gap < 0.05 in 4.4%; median gap 0.092 | above its 10% kill line: branching has room |
| (b) accepted-hop verification rate by distance above the threshold (Qwen 3.8, "probably wrong" 2) | <0.02: 93% · <0.05: 97% · <0.10: 87% · <0.20: 87% · ≥0.20: 86% | **falsified**: no cliff, the score does not predict the verdict near the threshold |
| (c) per-relation score ranges of verified vs failed hops (MiniMax, idea 7) | min verified 0.60–0.77 vs max failed 0.88–0.99 on every frequent relation | **killed**: the ranges overlap completely, a per-relation threshold cannot separate them |
| (d) share of not-correct answers whose gold sat in a top-3 runner-up (Qwen 3.8, idea 1) | 17 / 250 = **7%**; within 0.05 of the accepted fact: 1 | below its own 30% kill line: near-miss contrastive SFT killed before training |
| (e) verified 2-hop fact pairs shared by ≥2 questions (skill mining, all three models) | **1 of 219** | killed on the QA side, as Qwen 3.7's generality test predicted (<3 per 100) |

What the five numbers say together: the chain path does not fail by choosing the wrong candidate; it fails
by **finding nothing above the threshold** (246 of 283 failures), and the retrieval score is not the signal
that separates verified hops from failed ones. So the frontier walk fixes a minority case (19.8% of hops have
an alternative at all), skill reuse has nothing to reuse on this corpus, and the experiment the data points
at is ours, not theirs: **lower `tau_ret` and let the VM decide per hop** — since (b) shows candidates just
above the threshold verify as well as candidates far above it, candidates just *below* it may too. Metric:
verified and correct out of 800 vs 517 / 513, cost in VM calls; kill if the wrong-but-verified count leaves 4.

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
  game's library, where consolidation already exists) is still open — see Qwen 3.8 idea 5.
- **Idea 1 (repair memo)** survives as a small addition: the harvest's `banned_facts` pairs are the
  `Failure → Repair` edges; use them at inference when a walk lands on a fact banned in an earlier question.
  Bound: only 4 of 800 verified chains were wrong, so the memo's ceiling is the repair budget, not accuracy.
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
| 7 | Retrieval threshold per relation | 3 | 2 | 3 | 2 | pass | −5 (2104.08756 again) | 10 raw / 5 — **killed by pre-check (c)** |
| 8 | Failure-root clustering | 2 | 2 | 1 | 2 | pass | −5 (2004.07560 is magnetic black holes) | 2 |

Citations: 2211.17192 (speculative decoding), 2305.10601 (ToT), 1706.04599 (calibration) and 2106.03046
(counterfactual reasoning for language understanding) resolve as claimed. **Eight of the twelve IDs resolve
to unrelated papers** (listed in the table), one of them used twice for two different claims. Every number
quoted from the prompt is quoted correctly; "about 800·3 = 2,400 VM calls" is a labelled estimate.

- **Idea 7** was the best idea of the four answers on paper — per-relation thresholds learned from the
  verification history, host-side, versioned — and pre-check (c) killed it in one pass: on every frequent
  relation the lowest verified score (0.60–0.77) sits far below the highest failed score (0.88–0.99), so no
  per-relation interval separates them. The retrieval score is not where the verdict is decided.
- Ideas 1 and 2 contradict each other on the sign of counterfactual survival: idea 1 down-weights an edge
  whose perturbation still verifies (right: that is a verifier blind spot, an edge the VM never tested);
  idea 2 gives a *bonus* to walks that survive perturbation (wrong: it rewards the misses). Idea 1's
  "non-load-bearing edge" diagnostic is kept as a harvest statistic; its SFT use is the counterfactual
  harvest we already produce.
- Idea 3 forks the trunk so that each copy scores its candidate — the model as scorer, §3.1 — and puts
  the real trunk's exact fork on the stand-in, which has none.
- Idea 6's metric is the one to keep for the multi-world experiment: on a planted-contradiction subset the
  **don't-know rate must rise** (correct abstention) while gold does not fall. Idea 5 converges with the
  other two models on skill extraction; idea 8's maze metric ("revisits to trap") misreads the game (traps
  are Cubby's own mines).
- "Probably wrong" 1 (the expansion caps 32/64/3/8/8 were set by a plan document, not measured; derive
  per-world depth caps from harvest statistics) and 2 (a *learned* goal-relevance weight is a second
  self-judge; keep the weights host-set from a discrete menu) are both right and go into the spec.
- Generality test (dated cause chains) is coherent and names the dependency correctly — except that the
  dependency it names, idea 7, is dead; the temporal predicate Qwen 3.8 names below is the real one.

## 3. Qwen 3.8 2.4T

| # | idea | novelty | generality | falsifiability | cost | invariants | penalties | total |
|---|---|---|---|---|---|---|---|---|
| 1 | Runner-up contrastive SFT from the harvest | 3 | 3 | 3 | 2 | pass | — | 11 — **killed by pre-check (d)** |
| 2 | Branch on retrieval ambiguity | 2 | 3 | 3 | 2 | pass | — | **10** — pre-check (a) passed |
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
  not-correct answers (kill line 30%), within 0.05 of the accepted fact for one. Failures are retrieval
  misses, "what would make me wrong" as written. Note for later: the harvest schema reserves organic
  near-misses for the calibration/test split; had the pre-check passed, the mining would have had to stay on
  the train half.
- **Idea 2** is the frontier walk with the piece §2 lacked: a **task-defined choice-point criterion** (≥2
  candidates above the threshold, or a top-2 gap below a margin) instead of the token-entropy criterion
  H-P2 found unusable. Pre-check (a): 19.8% of hops qualify, 4.4% are near-ties. The branch decision is the
  host's, from retrieval scores; the model still only writes programs. **Build this first** — and bound the
  expectation by §0: it can only touch the 19.8%, and none of the 246 exhausted walks.
- **Idea 3** — fork handles on nodes so backtracking is a restore, not a re-decode — is measurable on day
  one on the stand-in (llama.cpp has state save/restore); kill at >100 ms restore. Under the 32-node cap
  that is ~200 MB. It is the first proposal that makes the retained rejected branch *resumable*, which the
  rendered think-trace needs.
- **Idea 4** turns the counterfactual harvest into repair records (input: chain + VM failure signature;
  target: the repair the runtime already performs), verifier-labelled. Pre-check to run: the share of live
  failures whose ban-and-replace led to verification. With only 4 verified-but-wrong chains in 800 and 246
  exhausted walks, the repair budget is where the gain would show, not accuracy.
- **Idea 7** adds the step the other two contradiction ideas lacked: derive the *discriminating hop* and ask
  both worlds through the router; until resolved the don't-know contract holds. Its experiment (two worlds
  over the history corpus with planted date/cause disagreements; measure the natural rate first, kill if
  <2%) is the multi-world harvest we do not have yet, and MiniMax's abstention metric is the right score.
- Idea 5: the unification rate over the existing program library is the pre-check (kill <10%); (e) already
  says the fact-path side is empty. Idea 6 targets the *budget* rather than the priority weights (hormones
  contract the beam under threat, widen it under curiosity), which is cleaner than the two multiplier
  proposals and keeps P(n) untouched; its pre-check on telemetry is the right first step.
- **"Probably wrong" 1 is the best single correction in the round**: merging branches on entity identity
  conflates provenance (a later-banned derivation rides inside the merged node; a canonicalization error
  fuses different entities). Merge on verified-subgraph isomorphism *and* agreeing verdicts. Adopted into
  the plan's merge rule. **"Probably wrong" 2 was tested and falsified** by pre-check (b): the hard threshold
  is not hiding a calibration cliff.
- Generality test: causal-chain QA over dated history, with the prediction that chains will be symbol-valid
  yet temporally inverted because neither the hop grammar nor the ground check carries a temporal predicate
  — checkable by planting 50 inverted chains through the VM. That is the first experiment of the causal
  history world, and the fix order given (temporal guard in the ground check, then depth, grammar last) is
  the right one.

## 4. Gemini 3.8 Flash

_(pending)_

## 5. Across the three answers

- **Convergence** (skill extraction ×3, hormones as search control ×3, contradiction nodes ×3, fork-based
  branching ×3) is the plan reflected back through three models, not evidence. Two of the three fork
  proposals put the model in the scorer's seat and fail §3.1; only Qwen 3.8 kept the VM as the scorer.
- **The pre-checks moved more than the ideas did.** Three of the four highest-scoring ideas on paper
  (per-relation thresholds, near-miss contrastive SFT, QA-path skill mining) are dead on the real harvest,
  and one "probably wrong" (the threshold cliff) was falsified. The data point at a different bottleneck:
  retrieval exhaustion (31% of questions) that no branching, skill or threshold table addresses.
- **Build order that survives the numbers:** (1) the VM-gated lower `tau_ret` experiment from §0 (ours);
  (2) Qwen 3.8's ambiguity-triggered frontier walk with the merge rule fixed by its "probably wrong" 1,
  measured on the 19.8% of hops it can touch; (3) fork handles on nodes, restore latency on day one; (4)
  planted-fault repair records after their pre-check; (5) the program-side skill unification rate on the
  game's library; (6) the multi-world contradiction experiment when a two-world harvest exists, scored by
  correct abstention. Required measurements from the "probably wrong" lists: score-term ablation, graph
  growth vs walk latency, per-world depth caps from harvest statistics, host-set discrete weights, and the
  50 temporally inverted chains before any causal-history world ships.
- **Fabrications, do not reuse:** Qwen 3.7's "endorphins" and "Dendrite project"; MiniMax's eight arXiv
  IDs (2009.08483, 2006.03388, 2007.00734, 2104.13542, 1604.04656, 1503.06267, 2104.08756 ×2,
  2004.07560), all resolving to unrelated papers. Qwen 3.8: none.
