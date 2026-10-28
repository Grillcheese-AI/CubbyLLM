# CubbyConsul board on the retrieval brief (2026-10-28): its result

Plan mode, 5 rounds, $1.51 of a $5 cap. Seats: librarian DeepSeek V4.1 Flash; proposers DeepSeek V4 Pro and GLM-5.3; prover Qwen 3.8 Max (rounds 1-2; its round 1 was empty, thinking used the whole turn), then Gemini 3.8 Flash; skeptics GPT-5.6 Terra and Grok 4.7 (round 5 lost to a network error); scribe Claude Sonnet 5.5. The librarian's dossier came out empty and the seats could not read the code root, so they worked from the brief (`2026-10-28-retrieval-brief.md`). Cowork's reading is in `docs/RETRIEVAL.md`.


## Plan
Run E0 first: a written denominator, a refusal-reason × seed cross-tab split by English and French, and quarantine of the 187 rows. Then test candidate generation under a frozen budget, disambiguation with underspecified first mentions scored apart from qualified ones, fact selection, memory, and passages. Only exact checked facts are spoken. [turn:5/proposer] [turn:5/prover] [turn:5/librarian]

## Steps
1. E0prep: verify QALD-9 and LC-QuAD 2.0 for QIDs and French; freeze the relation lexicon with a checksum; lock the FameTrap pairs. [turn:5/proposer] [turn:5/librarian]
2. E0: publish the refusal-reason × seed matrix with its denominator; report absent-locally shares. [turn:5/prover] [turn:5/proposer]
3. Freeze the K grid and the latency, memory and build ceilings before any comparison, with workload and machine stated. [turn:5/proposer-2] [turn:5/skeptic]
4. Candidate generation: FST and capsules; BM25 only as a supplement and never as a tie-breaker. [turn:5/proposer] [turn:5/proposer-2]
5. Disambiguation: an underspecified first mention that gets a spoken pick is a kill; a qualified mention is scored on correctness. [turn:5/prover] [turn:5/skeptic]
6. Memory resolves an explicit anaphor or confirmation to a referent; it never breaks a tie or supplies a value. [turn:5/proposer] [turn:5/prover]
7. Gold labels: LLM labels plus human adjudication; local-store membership is a contested condition (see Open questions). [turn:5/proposer] [turn:5/skeptic]
8. Passages: precision with a coverage figure; erroneous admissions reported separately. [turn:5/proposer] [turn:5/skeptic]
9. Dump trigger: line and sample fixed before counting. [turn:5/skeptic-2, as cited by others]

## Risks
- No verified public French QID set. [turn:5/librarian] [turn:5/proposer]
- Self-built sets are tuned or tautological. [turn:5/skeptic]
- Unfrozen or arbitrary resource ceilings. [turn:5/skeptic]
- Correlated LLM label errors. [turn:5/librarian] [turn:5/skeptic]
- Grammar coverage under 50% on held-out sets. [turn:5/proposer]

## Open questions
The board did not settle these:
- The public French set, the dump line, the budget numbers, and the E5 admission rule.
- Whether store membership is a valid gold condition. Librarian, prover, proposer and proposer-2 require it. Skeptic calls it circular and wants intent-QID, factual-triple and store-membership labels kept separate.
- Whether degree may build FameTrap. Proposer and prover use a degree or sitelink disparity of at least 100. Librarian wants name collision with no degree filter. Skeptic allows degree only for a stress slice. Proposer-2 keeps degree ≥100 but declines a 90% gate.
- Whether recency is allowed on a present-tense cue. Librarian and skeptic call it prevalence in disguise. Proposer-2 allows it only on an explicit time cue. Prover allows an interval filter only when exactly one value is valid.
- The E5 gate figures. Proposer gives precision pass ≥0.95 and kill <0.80, with coverage pass ≥0.10 and kill <0.05. Prover requires 0 false admissions and recall ≥60%. Skeptic says erroneous admissions are separately disqualifying.
- The dump trigger. Proposer says >15% of at least 1,000 queries; proposer-2 says >10% of seed misses; prover says ≥50% of failed mentions absent. Skeptic says none is attributable until E0 labels each miss's route.
- The resource ceiling figures (4 GB index, 50 ms per hop, 200 ms per question), which skeptic calls arbitrary precommitted hypotheses.
- Whether a baseline-relative spoken floor should gate at all. Skeptic says it pressures refusals into auto-resolution.
- BM25 handling and other contested points: the 90/10 split with Set X, French relation failures bundled into E0, and qFHRR for relations behind a margin test.
- French relation evidence, for which none exists on any route.

## Provenance
- Round-5 agreed points: 61% stays undefined until E0 publishes its denominator; the first-mention kill applies only to underspecified mentions; FameTrap is built by name collision, not id split; the relation lexicon is frozen before held-out data is touched; memory supplies a referent, never a value; E5 reports coverage and erroneous admissions beyond precision; the spoken floor is baseline-relative; QALD-9 and LC-QuAD 2.0 remain unverified for French QIDs. Attributions are those cited in the steps and in the brief.
- Skeptic-2 failed this round and gave only a partial answer. The dump-line step rests on others' citation of it.
- Per-seat positions on the disputes are listed under Open questions with their attributions.

## Sources (collected)

- [code:...]