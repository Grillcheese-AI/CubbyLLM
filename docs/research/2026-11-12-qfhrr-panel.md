# Panel: qFHRR, its weaknesses and trade-offs (2026-11-12)

The brief: `2026-11-12-qfhrr-brief.md` (beside this file). Asked independently, same prompt, temperature 0.3; budgets per model as in `validation/panel_qfhrr.py`. Model ids checked against OpenRouter's live list at run time. Cached under `standin/data/out/panel_cache/`.

**These are model opinions, not findings.** Nothing here is adopted until an order pre-registers it and its line is read. Cowork's own answer was sealed before this ran: `2026-11-12-qfhrr-cowork-sealed.md`.

Cost reported by OpenRouter: $2.65. Answers: 8 of 8.

## openai/gpt-6.1-sol-pro

*19542 tokens out, 317.6 s, finish `stop`*

## 1. Diagnosis

The central distinction is between **proposing something similar** and **establishing what the question asks**. qFHRR can do the former. Neither its score nor the VM’s verification of a retrieved fact establishes the latter.

### Entity names

The strongest evidence points to **tail-vocabulary coverage and a weak first-stage shortlist**. The preprint identifies document-frequency vocabulary selection as the problem. An unseen word retains hash identity but gains no teacher-derived meaning. Compact recall@100 is about 0.14, so dense reranking cannot recover most omitted documents. v4 improves performance but retains the roughly 2.3-times gap to dense potion (§4.1).

Other contributions are plausible but not isolated by these measurements:

- Bag-of-words encoding cannot distinguish names differing only in word order.
- Compact encoding may lose useful distinctions.
- The teacher may represent entity names poorly.

The brief does not measure their separate contributions.

**Structural:** a static unigram sum cannot resolve contextual identity or preserve order.

**Fixable without a serving model:** vocabulary selection, name-preserving exact indexes, lexical candidate generation, and shortlist construction. An unseen word is not wholly unrepresented; its representation lacks meaning beyond identity.

### Choosing between namesakes

The decisive failure is **using similarity as evidence for identity**, compounded by incorrect separation of name, relation and qualifier.

At margin 0.9, qFHRR still makes 11 forbidden bare-name picks, triggered by relation words. Restricting candidates to those carrying the relation does not cure this: it produces 21 bare picks at the same margin (§4.2). BM25 fails similarly. This is not evidence that another cosine threshold will solve the task.

B1q demonstrates the better framing: use explicit qualifier evidence, without ranking. Its remaining errors expose:

- relation-word leakage into cues, including “né” becoming Nebraska;
- incomplete treatment of geographical qualification;
- insufficient refusal when evidence is ambiguous (§4.2).

**Structural:** an uncued namesake cannot be identified from the question. No larger table or model can supply missing evidence legitimately.

**Fixable outside the table:** span roles, relation vocabulary, qualifier checking and refusal. qFHRR may propose interpretations, but must not choose the entity.

### Relations, including French

There are several distinct causes.

1. **Candidate-space framing.** Restricting candidates to an item’s properties raises SimpleQuestions recall@5 from 0.417 to 0.887 (§4.3). The store supplies a powerful prior.
2. **Availability mistaken for intent.** On 38 missing-property rows, qFHRR substitutes an available property on 36. The VM can verify that property’s fact while the system answers the wrong question (§4.3).
3. **Language coverage.** The English teacher leaves most French words with hash identity only (§4.2). The EN/FR relation results are consistent with this, although the sets do not isolate language as the only difference.
4. **Direction and question structure.** The asked-kind filter loses incoming “who” relations and item-valued counting relations. Candidate filtering can manufacture missing-property cases (§4.3).
5. **Training objective.** The property-label adapter greatly increases property emission but still performs poorly on French and exceeds the lexicon’s wrong rate everywhere. Naming a property is not equivalent to understanding the requested relation (§4.3).

**Structural:** the unigram sum loses order and cannot itself represent subject–relation direction or contextual scope.

**Fixable:** surface-language coverage, explicit direction handling, abstention and candidate construction. French semantic coverage could be changed offline, but the brief supplies no tested French-capable teacher.

## 2. Fixes

The rankings below are hypotheses about gain per cost, not measured gain estimates. Actual gains and resource requirements are unknown.

All new numerical pass lines below are **decision targets, not performance estimates**. For every arm:

- Freeze construction, calibration data, calibration date and decision lines before evaluation.
- Use fresh held-out splits; the reported development results are not fresh confirmation.
- Read held-out and test sets once per arm.
- Kill deployment if the complete layered-store lookup-and-check stage exceeds p95 50 ms.
- Recalibrate score gates after changing a table or corpus. Never reuse an old numerical threshold as if it transferred.

### Entity names

#### 1. Replace the qFHRR-only shortlist with exact/lexical candidate generation

**Change.** Keep exact-name lookup and typo fallback for mentions. For open-text entity retrieval, add a lexical document index and union its candidates with qFHRR candidates before reranking. Exact matching produces candidates, not a spoken namesake choice.

The mapping between the Wikidata name index and BEIR documents is not stated. Do not assume it exists; test the lexical document branch independently where necessary.

**Readout.** BEIR dbpedia-entity (§4.1), plus fresh Mintaka EN/FR mention-coverage sets. Measure candidate recall separately from disambiguation.

**Pass.** BEIR recall@100 at least 0.30 and nDCG@10 at least 0.224; exact-name gold coverage at least 87.4% EN and 99.6% FR. The last two targets preserve the reported index coverage, not chooser accuracy.

**Kill.** Any pass condition fails, or the latency budget fails.

**Cost.** CPU build hours, index memory and added serving latency: unknown. No GPU is required by this design.

**Hard rules approached.** Latency; no popularity-based namesake ranking. Lexical relevance must not become an identity decision.

#### 2. Make vocabulary selection entity-aware

**Change.** Add words from stored labels, aliases, titles and redirects rather than relying primarily on document frequency. Preserve the exact full-name index; expanding a word table does not replace it. Embed literal source strings offline, without asking an external model to author facts or training records.

**Readout.** BEIR, a fresh rare-name and unseen-word slice, and a fresh routing screen because the table has changed.

**Pass.** Against proposal 1, improve nDCG@10 by at least 0.03 and recall@100 by at least 0.05, with no routing-quality regression under the preregistered comparison.

**Kill.** Any required gain is missed, routing regresses, or deployment exceeds its latency budget.

**Cost.** Offline CPU/GPU hours, extra table memory and latency: unknown. Serving remains table lookup, not teacher inference.

**Hard rules approached.** Routing must remain non-neural; gates require recalibration. The offline teacher produces codes, never authoritative facts.

### Choosing between namesakes

#### 1. Repair B1q’s role boundary and make uncertain qualifiers refuse

**Change.** Give each span one role. Add the documented missing relation forms, including “né”, before alias normalization. If a span could be relation text rather than a qualifier, refuse. Initially disable location-based singleton picks that require hierarchical interpretation.

Do not substitute qFHRR when this check refuses.

**Readout.** Fresh fame sets with the same bare/qualified construction; ShadowLink; fresh Mintaka and WebQSP_EL samples. Add adversarial relation-word, accent-normalization and alias-collision cases.

**Pass.** Zero bare picks; zero wrong qualified picks; at least 70% qualified resolution; ShadowLink Top–Shadow gap at most 10 points; at most one wrong answer in each 50-row Mintaka/WebQSP_EL sample.

**Kill.** Any forbidden bare pick or other failed line.

**Cost.** CPU implementation and evaluation hours: unknown. No GPU. Lexicon memory and revised latency: unknown. Existing B1q p95 is already 40.6 ms at six or more candidates (§4.2), leaving little room within the complete-stage budget.

**Hard rules approached.** Refusal over guessing; no popularity ranking; latency. The deliberate risk is lost coverage, not permission to guess.

#### 2. Add bounded, declared qualifier reasoning

**Change.** Extend B1q to check location chains under explicitly approved symbolic rules, including the documented direct-versus-indirect location case. Apply the check to every remaining candidate. Preserve the proof path and refuse when exploration cannot finish within budget.

Do not assume every location relation is freely transitive. Which appropriate rules already exist is unknown.

**Readout.** The chooser sets above, plus a new set containing direct, indirect, conflicting and incomplete location statements.

**Pass.** Meet every chooser line above and improve qualified resolution over proposal 1.

**Kill.** Any wrong or bare pick, no resolution gain, an undeclared inference, or a latency failure.

**Cost.** CPU hours, additional index memory and fan-out latency: unknown. No GPU.

**Hard rules approached.** Only the VM certifies every used hop. A completed store proof still does not establish real-world uniqueness or completeness.

### Relations

#### 1. Separate requested-property identification from property availability

**Change.** Establish the requested property and direction through the grammar/lexicon before using the item’s available properties. If the requested property is absent, refuse; do not replace it with its nearest available neighbour.

An unanchored qFHRR score is insufficient authorization to answer. It may propose a property for further interpretation.

**Readout.** Every §4.3 relation set, the 38 missing-property rows, and a fresh missing-property set constructed without changing the question’s intent.

**Pass.** Zero missing-property handovers; no increase in wrong rate and no recall loss against the frozen grammar/lexicon system.

**Kill.** Any missing-property substitution or failed comparison. This is a safety intervention, not by itself a successful qFHRR replacement: promotion still requires the stated 0.85 SimpleQuestions and 0.80 QALD FR gates.

**Cost.** CPU hours and additional memory: unknown. No GPU. Added check latency: unknown.

**Hard rules approached.** The VM cannot repair misidentified intent merely by verifying an available fact.

#### 2. Expand literal French relation coverage and encode direction explicitly

**Change.** Extend the grammar and lexicon using stored EN/FR property labels. Add host-authored rules for incoming questions and counting over item-valued properties. Do not use asked-kind filtering to discard candidates unless direction and counting semantics support it.

Adding French spellings does not make the English teacher multilingual. This proposal improves literal interpretation, not unmeasured paraphrase understanding.

**Readout.** SimpleQuestions, QALD EN/FR and Mintaka EN/FR (§4.3), plus fresh inverse-relation and counting cases.

**Pass.** Recall@5 at least 0.85 on SimpleQuestions and 0.80 on QALD FR; zero missing-property handovers; no higher wrong rate than the frozen lexicon on any set.

**Kill.** Any line fails, including any dropped-gold error caused by the new direction rules.

**Cost.** CPU construction/evaluation hours, lexicon memory and latency: unknown. No GPU.

**Hard rules approached.** Rules and source labels must be traceable. Model-generated paraphrases must not become training records or facts.

#### 3. Train contextual relation interpretation in the programs seat

**Change.** Before growing the model, train the current model on question-to-property interpretation, including direction, ambiguity and absent-property refusal—not merely property-label production. Training records must be host-authored or deterministically constructed from checked store material. External model-written records are excluded.

Place this work in program generation, not routing or the latency-limited lookup stage.

**Readout.** The relation sets above, with held-out wording patterns, plus fresh ambiguous and missing-property cases.

**Pass.** Both recall gates; zero missing-property handovers; no higher wrong rate than the lexicon on any set; and at least the original three-point pooled improvement required in §4.3.

**Kill.** Any line fails. More frequent property emission is not a success criterion.

**Cost.** New CPU/GPU hours, adapter memory and serving latency: unknown. The existing reference is slightly over an A100 hour for about 216k LoRA rows, and about 2.4 seconds for current program generation (§5); neither is a measured cost for this proposal.

**Hard rules approached.** No neural routing, no model-written facts or external model-written records, and no substitution of model confidence for certification.

## 3. The trade-offs

“Memory unknown” means the relevant complete component footprint is not given. Component timings are not end-to-end guarantees.

| Job | Responsible component, in order | Measured latency and memory | Measured accuracy or coverage | Failure mode |
|---|---|---|---|---|
| **Finding a name** | Exact name index → typo fallback → candidate set. Lexical/qFHRR retrieval only for broader search. | Index latency and bytes unknown. Contains 12.4M names for 4.8M items (§4.1). | Exact-name gold coverage: 87.4% Mintaka EN, 99.6% FR. This is not disambiguation accuracy. | Refusal on no match; wrong candidate if approximate matching is treated as identification. |
| **Choosing between namesakes** | Span-role check → symbolic qualifier evidence → VM-checked proof or refusal. Model may propose a parse, never an uncued pick. | Existing B1q p50/p95: 1.4/6.2 ms at two candidates; 10.2/40.6 ms at six or more. Memory unknown (§4.2). | B1q: 76.4% qualified resolution, three wrong and two bare picks; therefore not acceptable as measured. | Wrong identity; intended safe outcome is refusal. Expansion can cause slowness. |
| **Choosing the relation** | Grammar/lexicon identifies intent and direction → availability check → VM. Contextual programs-seat model only after validation. qFHRR proposes, not authorizes. | Full lookup latency and memory unknown. Current model program generation about 2.4 s (§5). | qFHRR among own properties: recall@5 0.887 SimpleQ, 0.536 QALD FR; 36/38 missing-property handovers at margin zero (§4.3). | A certified fact answering the wrong question; otherwise refusal. |
| **Reading French** | Exact French names → French grammar/lexicon → validated reasoning model if needed → symbolic checks. | French-specific latency and memory unknown. | Exact-name coverage 99.6%; relation recall@5 0.536 QALD FR and 0.397 Mintaka FR for qFHRR at margin zero (§4). | Literal-only matching, wrong relation or refusal. |
| **Routing to a world** | qFHRR table → calibrated bundled-centroid/margin gate → existing world or spawn. No neural model. | Short-text encoding 6.8–25 µs on one CPU core; complete routing latency and table memory unknown (§3). | 0.341 versus teacher 0.358 on 15 domains; 97% of teacher quality on 42 domains. Open-set precision 92.4%; 86.9% of unseen domains sent to spawn (§3). | Wrong route or unnecessary spawn. |
| **Retrieval over a known corpus** | qFHRR compact retrieval → permitted reranking → exact stored-fact checks when answering. | Compact document code 80 bytes; dense form 20 KB. Full search latency and index overhead unknown (§3). | On 1M documents: hit@1 0.71, hit@20 0.92. Verified pipeline later answers 574/800 at precision 1.000 on consistently misspelled data (§3). | Missed retrieval or refusal. These results do not establish real-world correctness. |
| **The n-gram memory** | qFHRR-addressed Hamming retrieval → exact row/fact check. Do not trust the nearest row alone. | Scan uses 10,240 bits; total memory and latency unknown (§3). | 74.3% correct row, 4.0% different fact on 35 hand pairs; no fresh-list confirmation (§3). | Wrong fact if unchecked; otherwise refusal or scan slowness. |
| **Reading a host-placed fact** | Host selects stored facts → loci reader copies → symbolic check where certification is claimed. | Loci serving latency and total memory unknown; copy path has 1,027 trained parameters. grilly2 read deployment remains unconfirmed (§6). | Run 5: 0.946 exact beyond 800 tokens. Equal-token run 6: 0.804 versus 0.019 ordinary reading (§6). | Wrong copy, closed copy gate, overwritten/multiple-value loss, or refusal. |

## 4. Growing the model

**I do not know a justified larger size.** The brief gives no scaling curve, home-GPU memory limit, grown-model throughput or training-budget ceiling. A parameter recommendation would be invented.

The first comparison should retain the current model: about 576M total weights and 445M read per token (§5).

| Job | Is growth the lever? | Cheaper first step; required evidence |
|---|---|---|
| Finding a name | Not for literal lookup or missing shortlist candidates. Possibly for contextual mention parsing, not demonstrated here. | Exact/typo and lexical indexes. A grown parser must improve candidate coverage without introducing identity guesses. |
| Choosing namesakes | Possibly for extracting explicit qualifiers; never for uncued names. | Repair B1q. Any model-assisted version must meet all chooser lines, especially zero bare picks. |
| Choosing relations | The strongest plausible growth case: contextual direction, scope and interpretation. | Contextual training on the current model first. Growth must pass both relation gates, the wrong-rate comparison and missing-property refusal tests. |
| Reading French | Possibly, but size alone does not supply French competence. | French lexical coverage and appropriate current-model training. Require separate EN/FR evaluation; do not infer success from aggregate loss. |
| Routing | No. Neural routing is forbidden. | Improve the static vocabulary and calibrate its gate. |
| Known-corpus retrieval | Not the first lever. | Candidate recall, lexical combination and indexing before model growth. |
| N-gram memory | Not established as useful. | Fresh confirmation and exact checks before increasing capacity or model size. |
| Reading host facts | Not first. | Complete loci serving validation and fix addressing, multiplicity and gating. Current-model loci already shows a large measured advantage (§6). |

Before selecting a grown size, measure its memory residency, program latency and cost against a validated current-model arm. The frontier-quality-at-one-hundredth-cost target cannot be evaluated from the brief: the required comparative quality and cost measurements are unknown.

## 5. Loci tokens

### Entity names: help with recall, not discovery or meaning

Loci can retain and copy a host-supplied name without a word-table vocabulary miss. That does not find the entity in the store or interpret an unfamiliar name semantically. Name-span binding remains a recall mechanism, not a demonstrated training-cost lever (§6).

**Division:** exact index finds candidate items; qFHRR retrieves broader material; the host selects facts; loci reads their literal contents.

### Namesakes: unchanged by construction; harmful if treated as identity resolution

Two items with the same name receive the same word-keyed address (§6). Loci therefore cannot select between them. Combining their facts under that address risks conflation.

**Division:** resolve or refuse symbolically first. Only then construct a scene for the selected item. A host-supplied item-distinguishing address is a possible experiment, not a measured capability.

The CPU toy’s symmetric subject–relation key must not be assumed to encode their distinct roles safely.

### Relations and French: help only after interpretation

Loci can read the value once the host supplies the correct entity, relation and facts. It does not resolve paraphrases: different relation wording produces a different address. The reader also misses relations absent from its training shard, multiple values and closed copy gates (§6).

**Division:** grammar, lexicon or a validated contextual parser establishes the canonical relation and direction; the host supplies the matching facts; loci copies; the VM remains the sole certifier.

Do not treat CPU toy results as 450M results, or torch/grilly2 logit agreement as successful loci serving.

## 6. What I would not do, and open questions

### I would not

- Rescue the namesake ranker by raising its margin again.
- Use popularity, agreement between arms or a large score as identity evidence.
- Rerank a shortlist as the principal remedy for candidates it never retrieved.
- Filter by asked kind without handling incoming relations and counting.
- Allow an available near-property to replace an absent requested property.
- Treat certification as proof of correct question interpretation or real-world truth.
- Expand n-grams or binding variants merely because they are available; the reported variants cost more than they bought (§3).
- Buy a larger model or multilingual rebuild before testing cheaper index, lexical and task-framing changes.

### Open questions

- What are the deployed v4 table footprint, vocabulary composition and language coverage?
- How much entity loss comes from vocabulary coverage versus compact encoding, teacher quality and query structure?
- Does the exact name index map to the BEIR documents?
- What causes B1q’s other bare pick, and which remaining errors survive the documented repairs?
- Which location rules are approved, expressible and affordable in the VM?
- What are current grammar-plus-lexicon recall and end-to-end lookup timings?
- Which French relation wordings are already in the store and lexicon?
- Can contextual current-model training improve interpretation without increasing wrong answers?
- Do fresh n-gram pairs confirm the hand-list result?
- Does run-6 loci reading reproduce on grilly2, including absent facts and multiple values?
- What hardware and comparative-cost measurements would justify a grown model?

The priority is **better candidate generation for names, proof-based refusal for namesakes, and intent-before-availability for relations**. Growth is a later experiment for contextual interpretation, not a substitute for any of those boundaries.

## anthropic/claude-fable-5.2

*28468 tokens out, 305.6 s, finish `stop`*

# qFHRR: where it is weak and what to do about it

Numbers are cited by brief section. Figures I computed from the brief's numbers are marked "derived". My own guesses are marked "estimate". Every pass and kill line below is mine, written now, and none has been run.

## 1. Diagnosis

### 1.1 Entity names (§4.1)

**Not the teacher.** v4 gained 18% and kept the 2.3-times gap to dense potion. A better teacher moves the number without changing its shape.

**Vocabulary by document frequency is only part of it.** The preprint's reading (names are tail words the list misses) does not explain the whole gap. A missed word still gets its hash code, which is identity (§3). BM25 has identity only and scores about 0.313, above potion-8M dense at 0.224. So on this set identity is worth more than meaning, and qFHRR has identity for every word and still scores 0.046 to 0.096.

**The bag-of-words sum and the compact form are the cause I think the numbers point to most.** The identity of one rare word has to survive an IDF-weighted sum over the whole document and then an argmax per block down to 80 bytes. The compact shortlist's recall@100 of about 0.14 caps everything after it. This is an inference; the brief does not isolate it. Two things I don't know would settle it:
- what IDF weight an unseen word gets;
- what dense qFHRR scores searched exhaustively, without the compact shortlist. The 0.096 is a re-rank of a capped shortlist, so the 2.3-times gap mixes code quality with shortlist loss.

**Framing.** The system does not give this job to qFHRR. The exact name index already holds the gold for 87.4% of Mintaka EN mentions and 99.6% of FR.

**Structural or fixable.** Holding a rare word's identity inside one fixed-size sum is structural to a single-vector static table. An inverted index keeps each term apart and the sum does not. The weight given to hashed words is fixable inside the table. A side list of rare words is fixable too, but it is an exact index under another name.

### 1.2 Choosing between namesakes (§4.2)

**Use as a ranker is the first cause.** A rank-normalized margin always has a top candidate. Bare picks were 298, 73, 21 and 11 as the margin rose, and never 0. Restricting to candidates that carry the relation did not help: 35 bare picks at 0.75.

**Bag of words is the second cause.** The table cannot tell which role a word plays. At 0.9 the 11 picks came from relation words in the template ("pays", "né", "born"), and BM25 died the same way. The span rule fixed the roles, but outside the table.

**The errors are not controlled by the margin.** On Mintaka the wrong count stays at 8 from margin 0.75 (8 of 92) to 0.9 (8 of 83, 9.6% derived, against a 2% line). No margin gives both 0 wrong and 70% resolved on the qualified line: 0.25 gives 77.4% with 29 wrong, and 0.75 gives 0 wrong at 12.4%.

**English teacher and hash fallback** add to this in French: cues match only as written.

**Structural.** The rule requires evidence that holds for one candidate. A cosine cannot express "holds", and it cannot express "there is no cue". The role split is fixable, but once it is fixed the remaining job is a symbolic check, which is what B1q is.

B1q's own residue is not qFHRR's:
- the 3 wrong picks are a missing transitive step on "located in";
- one bare pick is a lexicon gap ("né") combined with accent folding;
- I don't know the cause of the second bare pick, of the 61 undecided or the 154 refused, or of the 2.3% and 2.2% wrong on Mintaka and WebQSP_EL.

### 1.3 Relations (§4.3)

Four separate causes show in the numbers.

- **Framing (the candidate set).** Among all 12,684 properties recall@5 is 0.417 on SimpleQuestions. Among the item's own (median 39) it is 0.887. The table discriminates a small set and not a large one. This is fixed by the framing already adopted.
- **English teacher and hash fallback.** QALD drops 21.9 points from EN to FR (0.755 to 0.536) and Mintaka drops 15.7 (0.554 to 0.397), both derived. French words match only as written, so inflection alone breaks a match. This is fixable inside the table.
- **Ranker over a closed set.** This is the missing-fact danger. When the gold property is absent, margin 0 hands over a near property on 36 of 38 rows and margin 0.25 on 16. The lexicon does so on 6, the 450M on 4 and the grammar on 0. A ranker over the item's own properties is a forced choice. The margin reduces the handovers and does not remove them. This is structural to ranking. It is also the worst failure in the brief, because the VM certifies the result.
- **Bag of words.** Order and direction are lost. The asked-kind filter fails on "who was born in Paris?" for the same reason: the direction of the answer is not represented. This is structural. I don't know its share of the errors.

Three further observations:
- QALD EN is itself 0.755, below the 0.80 gate set for QALD FR. Fixing French alone cannot pass that gate. The English ceiling on that set is also short, for reasons the brief does not state.
- Mintaka EN is 0.554 at candidate counts similar to QALD's. I don't know why.
- The keep rule failed on a pooled 2.7 points, but the sets disagree. On QALD FR at the same 15.1% wrong rate, qFHRR at 0.25 gives 0.349 against the lexicon's 0.217. On Mintaka EN it gives 0.223 against 0.331 at half the wrong rate. The rule's verdict stands. The disagreement says the two arms hit different rows, which matters for the order of work in deliverable 3.

## 2. Fixes

All lines are tuned on dev and read once on a held-out split. The fame set, dev seed 0, has been read by several arms and can only serve as dev.

### 2.1 Entity names

**N1. An exact token index over the 12.4M names, behind the exact and typo indexes.**
- What changes: a mention that misses as a whole string is looked up by its name tokens, and the items whose name contains all of them form the set. No qFHRR is involved.
- Read by: gold-in-set on Mintaka EN mentions (87.4% today), then B1q on the result.
- Pass: gold-in-set at least 92% EN; FR not below 99.6%; the share of mentions naming more than 20 items up by at most 3 points.
- Kill: gain under 2 points; any added wrong pick on the fame set; or lookup-and-check p95 over 50 ms.
- Cost: build under 1 CPU-hour (estimate). Memory unknown; estimate the same order as the present name index. Serving latency unknown.
- Near: the 50 ms rule. Larger sets must go to B1q or refusal, never to popularity.
- Caveat: I don't know why 12.6% of EN mentions miss. Audit 100 misses first. If they are not token-order or partial-name misses, drop N1.

**N2. Weight hashed words by the corpus's own document frequency.**
- What changes: at index time the document frequency of every word in the target corpus is counted, and an unseen word gets its IDF from that count. First run two diagnostics on a subsample: the share of query words that are out of vocabulary, and exhaustive dense recall@100.
- Read by: BEIR dbpedia-entity, v4 (compact recall@100 about 0.14; nDCG@10 0.046 compact and 0.096 re-ranked).
- Pass: compact recall@100 at least 0.30, and re-ranked nDCG@10 at least 0.15.
- Kill: recall@100 under 0.20. Then the loss is in the sum and the compact form, and qFHRR stays out of entity search for good.
- Cost: a count over 4.64M documents plus a re-encode, a few CPU-hours (estimate). Compact index 371 MB (derived: 80 bytes × 4.64M). Serving latency unchanged.
- Near: thresholds don't transfer; any gate is re-set.

**N3. A posting list of rare words beside the compact code.**
- What changes: words above an IDF cut also go into an inverted list. The shortlist is the union of the compact-code neighbours and the posting hits, followed by the dense re-rank.
- Read by: the same set as N2.
- Pass: re-ranked nDCG@10 at least 0.224, parity with dense potion-8M.
- Kill: under 0.18, or shortlist latency more than doubles.
- Cost: memory and latency unknown. Build is a few CPU-hours (estimate).
- Near: none.
- Why it ranks last: the system already has the exact index for names. N3 is worth doing only if open-corpus entity search is a job GRL actually wants.

### 2.2 Choosing between namesakes

**C1. Lexicon morphology and a folding rule.**
- What changes: the relation lexicon gains the inflected forms of its own wordings (French participles and agreements, by rule). An accent-folded match can make a cue only if the exact form has no reading as a relation or function word and the token has at least 4 letters.
- Read by: the fame set, bare line (2 picks today).
- Pass: 0 bare picks from a relation or function word, and resolved not below 0.764.
- Kill: any such bare pick, or resolved under 0.70.
- Cost: CPU minutes. No serving cost.
- Near: the bare-name kill line.
- Caveat: I don't know the cause of the second bare pick. This fix may leave it standing.

**C2. Closure on containment in the check.**
- What changes: a candidate holds a place qualifier if the qualifier is reachable through stored "located in" statements within a bounded depth. Each hop is a stored fact the VM can check. The closure is an index and not a new fact. In the Rosebank case both candidates then stand, and the answer becomes undecided instead of wrong.
- Read by: the fame set, qualified line (692 right, 3 wrong, 0.764).
- Pass: 0 wrong; resolved at least 0.70; p95 at 6 or more candidates at most 50 ms (40.6 today, so 9.4 ms of headroom).
- Kill: any wrong pick, resolved under 0.70, or p95 over 50 ms.
- Cost: precomputing ancestor sets, a few CPU-hours (estimate). Memory unknown.
- Near: the 50 ms rule, and "facts are never model-written". The closure must be stored as an index and labelled as one.

**C3. qFHRR as a proposer of the cue's value, never of the candidate.**
- Precondition: an audit of the 61 undecided and 154 refused shows cue words that are not exact names of a value item.
- What changes: qFHRR maps the cue to at most one value among the values present on the candidates' statements, past a margin set on dev. B1q's check then runs unchanged: exactly one candidate standing, or undecided.
- Read by: the fame set, both lines, plus Mintaka and WebQSP_EL.
- Pass: resolved at least 0.81, with 0 wrong and 0 added bare picks.
- Kill: one wrong pick or one added bare pick.
- Cost: microseconds per cue (§3: 6.8 to 25 µs per text). Memory: one code per value label.
- Near: "refusal is better than a wrong answer". A near value held by the wrong candidate is the namesake version of the missing-fact handover. The kill line is one error for that reason. Thresholds are set per language.

### 2.3 Relations

**R1. A missing-fact guard, so that absence leads to a refusal.**
- What changes: the candidate list becomes the item's own properties plus the properties that items of its class usually carry and this item does not. The second list is counted from the store. If the best match is one of those absent-but-expected properties, the host refuses with "fact missing". qFHRR runs only after the grammar and the lexicon, with a margin.
- Read by: the 38 missing-fact rows (dev), and a new ablated set: for rows where the item holds the gold, hide the gold statement and require a refusal. The new set gives hundreds of rows where today there are 38. Its limit is that hidden facts are not distributed like naturally missing ones.
- Pass: at most 4 handovers on the 38 (the 450M's count); at most 10% on the ablated set; SimpleQuestions recall@5 at least 0.85 at the same setting.
- Kill: more than 6 of 38 (the lexicon's count), or SimpleQuestions under 0.85.
- Cost: class-property counts, CPU hours (estimate). The candidate set grows by an unknown amount. Scoring stays in microseconds per candidate (estimate).
- Near: "refusal is better than a wrong answer", which it serves. The counts are statistics and must never be stored as facts.

**R2. A French table, in three steps. Stop at the first step that passes.**
- (a) A rule-based French light stemmer, so that inflected forms share a row. CPU only.
- (b) French words tied to English codes through the store's paired EN and FR property labels and aliases.
- (c) French vocabulary embedded by a multilingual teacher into the same space. I don't know which teacher, or whether the pretraining corpus contains French.
- Read by: QALD-9-plus FR and Mintaka FR among the item's own properties. English sets and routing are the regression guards.
- Pass: QALD FR recall@5 at least 0.70, within about 5 points of EN's 0.755; Mintaka FR at least 0.50; wrong rate at margin 0.25 not above 15.1%.
- Kill: QALD FR gain under 8 points; SimpleQuestions under 0.877; or 42-domain routing under 95% of the teacher.
- Cost: (a) and (b) are CPU minutes to hours (estimate). (c) is under 1 GPU-hour (estimate). Table memory grows with the added words; bytes per row are not stated. Serving latency is unchanged.
- Near: thresholds are re-set per table and language.
- Caveat: the 0.80 gate will not pass on this fix alone (§1.3).

**R3. A direction-aware asked-kind filter.**
- What changes: the kind is tested on the answer end. For an incoming property that is the subject items, and "how many" admits item-valued properties as a count.
- Read by: the per-kind rows (place 0.77 at 6% wrong, time 0.63 at no wrong) and the handover count (83 with the filter, 36 without).
- Pass: handovers at most 36; golds dropped on at most 1% of rows; pooled wrong rate at margin 0 down by at least 5 points.
- Kill: handovers above 36.
- Cost: CPU. No serving cost of note.
- Near: none.

## 3. The trade-offs

| Job | Who does it, in order | Latency | Memory | Accuracy | Failure mode |
|---|---|---|---|---|---|
| Finding a name | Exact name index, then typo index, then N1 if it passes. No qFHRR. | Unknown alone; inside the p95 50 ms budget | Unknown (12.4M names) | Gold in set 87.4% EN, 99.6% FR (§4.1). qFHRR on BEIR 0.046 to 0.096 against BM25 about 0.313 | A miss, which becomes a refusal. An over-large set (a fifth to a third of mentions name more than 20). |
| Choosing between namesakes | B1q (span rule and symbolic check), then refusal. qFHRR at most as a value proposer (C3). Never the 450M, never loci. | 1.4 / 6.2 ms at 2 candidates; 10.2 / 40.6 ms at 6 or more (§4.2) | Unknown | Resolved 0.764, 3 wrong, 2 bare picks; 2.3% and 2.2% wrong on Mintaka and WebQSP_EL | Mostly refusal (61 undecided, 154 refused). A wrong pick where a hierarchy step is missing. |
| Choosing the relation | Grammar, then lexicon, then qFHRR among the item's own properties with a margin and R1's guard, then the VM. The 450M is not a namer today. | qFHRR 6.8 to 25 µs per text (§3); candidate scoring unknown; 450M about 2.4 s | Unknown | qFHRR 0.887 at 9.0% wrong (margin 0) and 0.573 at 3.0% (0.25) on SimpleQuestions; lexicon 0.563 at 4.0%; grammar unknown | A wrong answer that is certified: 36 of 38 handovers at margin 0. |
| Reading French | Names: exact index. Cues: B1q with the fixed lexicon. Relations: lexicon, then the French table (R2). | As above | As above | Names 99.6%; relations 0.536 (QALD FR) and 0.397 (Mintaka FR); French routing unknown | Wrong lists (45.8% and 60.3% at margin 0), or refusals once a margin is set. |
| Routing to a world | The qFHRR table alone (hard rule) | 6.8 to 25 µs | 80 bytes or 20 KB per centroid; table size unknown | 0.95 of the teacher on 15 domains; 97% on 42; open set 92.4% precision, 86.9% of unseen domains to spawn | A wrong route, or a missed spawn (13.1% of unseen domains, derived). |
| Retrieval over a known corpus | qFHRR compact shortlist, then dense re-rank, then the VM re-verifies | Scan latency unknown | 80 MB per 1M documents compact, 20 GB dense (derived) | hit@1 0.71, hit@20 0.92 at 1M; in the pipeline 0.992 precision at 0.646 coverage, then 574 of 800 at 1.000 (misspelled corpus: chain fidelity only) | A miss, which becomes a refusal. The VM turns wrong retrievals into lost coverage. |
| The n-gram memory | qFHRR word codes with a Hamming scan. Every hit is then checked against the store. | Unknown | 1,280 bytes per 10,240-bit row (derived) | 74.3% reach the row, 4.0% a different fact (35 hand pairs, unconfirmed) | A wrong fact (4.0%) unless it is checked; otherwise a miss. |
| Reading a fact the host placed in context | The host resolves item and property symbolically. The 450M reads through loci tokens. The host compares the read with the store's value as an exact string. | Unknown (grilly2 port in progress) | Unknown | 0.946 exact (run 5); 0.804 against 0.019 (run 6); word-keyed toy 0.854 to 0.899 on unseen subjects, not run on the 450M; the switch is about 12% wrong on present facts (toy) | A wrong read, or "does not know". Misses on multi-valued facts and unaddressed relations. |

Two notes on the table.
- In the question loop the VM already holds the certified value. The loci read is how the model uses that value, not how the system knows it. The exact string comparison is cheap and turns a wrong read into a fallback to the store's own value.
- The table belongs where its error becomes a refusal (retrieval under the VM, routing with a margin gate). It should be kept away from jobs where its error becomes a certified wrong answer (relation handover, namesake choice).

## 4. Growing the model

**Where a larger model is not the lever.**
- Finding a name: an index does it.
- Choosing between namesakes: the rule asks for evidence that holds, and a model's preference is not that. A larger model also knows the famous bearer better, which is the popularity prior the rules forbid.
- Routing: a neural model is barred by rule.
- Retrieval and the n-gram memory: these are table and VM jobs.
- The loci read: run 5 reached 0.946 by training 1,027 parameters. The lever there is the copy path and its switch, not width or depth.

**Where it might be the lever.** The programs seat is the candidate: splitting a question into name, relation and cue, and reading relation wordings the lexicon lacks, above all in French and on Mintaka-style questions. The brief has no evidence that size helps there, and some against:
- with a dedicated adapter the 450M reached 0.800 on SimpleQuestions, below the table's 0.887;
- on QALD FR it reached 0.241, below the table's 0.536;
- its pooled wrong rate was 0.286, about twice the lexicon's 0.145.

The same model went from naming a property on 0.21 to 0.53 of questions to 0.90 to 0.98 by changing the adapter's data. At this size, data moved more than anything else measured.

**Cheaper levers to try first, in order.**
1. R1 and R2. They cost CPU.
2. A selection adapter. The 450M is given the item's own property list and trained to pick one or to answer "none of these". Records are built from the store and the existing question sets only, never by an external model. The cost is about one hour on an A100 by the brief's LoRA figure. Read it on the §4.3 sets and the ablated set.
3. Grow only if step 2 falls short on coverage of hard questions and not on safety.

**What size.** I don't know what size would suffice; nothing in the brief measures scaling. If a step is taken, I would take one doubling of the trunk and keep the 128k vocabulary. The two embedding tables are 262M of the 576M, so the trunk is about 314M (derived). A doubled trunk gives roughly 0.9B in all (estimate). I would expect the programs seat to go from 2.4 s to about 4 to 5 s on the home GPU (estimate). That works against the one-hundredth cost aim and has to be earned. I can't cost the pretraining; the brief gives only LoRA costs.

**What it would have to show,** against the 450M with the same adapter data and equal training tokens:
- relation selection at least 0.887 on SimpleQuestions and at least 0.80 on QALD FR;
- pooled wrong rate at most 0.145;
- at most 4 missing-fact handovers of 38, and at most 10% on the ablated set;
- the loci read not below 0.946;
- a gain of at least 5 points over the 450M's selection adapter on QALD and Mintaka.

Under 3 points of gain, do not grow.

## 5. Loci tokens

The two codes fail in opposite ways. qFHRR is soft: it has meaning, and its error is a near neighbour. Loci keys are hard: they carry identity, and their error is no address. That decides how they divide the work.

**Entity names: as it is for finding; a help for reading.**
- A loci key is an address inside a scene the host wrote. It is not a search structure over 4.8M items, so it does nothing for finding.
- For reading it covers what the word table lacks: every name has a key, and in the toy unseen subjects read 0.854 to 0.899 where hashed addresses read 0.000. That is not yet run on the 450M.
- Division: the exact index finds, and loci reads facts about tail names. qFHRR does neither.

**Namesakes: no help, and a harm if misused.**
- Two items with the same name get the same key, and one address holds one value. A scene holding facts of two namesakes collides.
- The choice must be made before the scene is written: B1q picks or refuses, and the host writes only the chosen item's facts.
- If two namesakes must share a scene, the host could bind each item's key with a per-item code. That needs the asker to point at the code and is untested.
- Loci must never be used to choose.

**Relations: a help as a strict reader; as it is for paraphrase.**
- A different wording is a different address. Loci cannot map "où est né" to "place of birth"; the brief lists this as a miss.
- The same hardness is a guard against the near-property handover: an address that was not written is not read. The guard is weakened by the switch's 12% error in the toy.
- Combination, in order:
  1. grammar, lexicon and guarded qFHRR map the question's wording to one property of the item;
  2. the host writes the fact at the address of the property's canonical label and gives the asker that same label;
  3. the trunk reads;
  4. the host compares the read with the store's value.
- Two cheap toy tests before any 450M run:
  - bind a role code to the subject and to the relation, so the key stops being symmetric and incoming and outgoing readings of the same words differ;
  - write the j-th value of a multi-valued property at the key bound with an index code.
- A relation key made from the qFHRR compact code (a legal operand, §3) in place of the label's embedding would make addresses paraphrase-tolerant in principle. I don't know whether a nearly equal key unbinds cleanly. Run a toy first.

## 6. What I would not do, and open questions

**I would not:**
- retry qFHRR, BM25 or any score as the namesake chooser with other margins or prototypes; the failure is the ranking itself;
- serve relations at margin 0 on the strength of 0.887; the same setting hands over 36 of 38 missing facts;
- reopen the "not kept" verdict on the rows already read; the cascade with R1 is a new arm with new lines on unread rows;
- chase BEIR dbpedia-entity with further teacher upgrades;
- add n-gram or binding variants to the table for word order (§3: each cost more than it bought);
- use the asked-kind filter as it stands (83 handovers against 36);
- let agreement between qFHRR and the 450M count as more than a filter;
- carry a threshold from the English table to a French one;
- break sets of more than 20 by any popularity signal;
- grow the model for names, namesakes or routing, or before the selection adapter is read.

**Open questions:**
1. What IDF does an unseen word get, and what share of dbpedia-entity query words are out of vocabulary?
2. What is exhaustive dense recall for qFHRR on that set, without the compact shortlist?
3. Why do 12.6% of Mintaka EN mentions miss the name index?
4. What are the causes of the 61 undecided, the 154 refused, the second bare pick, and the wrong rows on Mintaka and WebQSP_EL? Are those wrong rows cases where the gold is absent from the set and a lone other bearer is picked?
5. How are five relation candidates consumed downstream: does the VM run all five, or the first only? The cost of a "wrong list" depends on this.
6. What is the grammar's recall, and how much do the lexicon's hits overlap with qFHRR's?
7. Why is Mintaka EN 0.554 when QALD EN is 0.755 at similar candidate counts?
8. Does the pretraining corpus contain French, and which multilingual teacher would GRL accept?
9. Which held-out splits remain unread for each set?
10. Does the 50 ms budget include relation choice, or only name lookup and the namesake check?
11. Does the switch's 12% error on present facts hold on the 450M, and what is the loci read's latency on grilly2?
12. What is "v4" made of (teacher, vocabulary size)?

## google/gemini-4-argon

*32134 tokens out, 538.5 s, finish `stop`*

### 1. Diagnosis

#### 1.1 Entity names
* **What the numbers point to:**
  * **Vocabulary built by document frequency & hash fallback for unseen words:** On open-web entity search (BEIR dbpedia-entity, 4.64M documents), qFHRR v4 scores 0.046 nDCG@10 in compact form and 0.096 with dense re-ranking, against 0.224 for dense potion-8M and about 0.313 for BM25, capped by a compact shortlist recall@100 of about 0.14 (section 4.1). As the preprint notes, entity names live in the long tail that a 30k-to-60k word list built by document frequency misses (sections 3, 4.1). Unseen words receive a deterministic random block code (identity, not meaning) blended 0.5/0.5 at build time for known words or pure hash for unseen words (section 3). When summed with IDF weights across a multi-sentence document, a rare entity’s random hash is diluted by the sum of dozens of in-vocabulary content words.
  * **Something else (block-argmax quantization loss):** Moving from dense re-ranking (20 KB per document) to the compact shortlist (80 bytes, taking only the single argmax slot per block of 128) cuts v4 nDCG@10 in half (0.096 to 0.046) and caps recall@100 at 0.14 (sections 3, 4.1). Taking a single winning slot per block erases minor components of a document sum, dropping entity-name hashes before dense re-ranking ever runs.
  * **Framing of the task:** Treating entity-mention lookup as dense semantic bag-of-words retrieval is the wrong framing for the reasoning path. An exact name index (12.4M names of 4.8M Wikidata items, plus a typo index) already holds the gold item under its exact name for 87.4% of Mintaka EN mentions and 99.6% of FR mentions without touching qFHRR (section 4.1).
* **Structural vs. fixable inside a static word table:**
  * **Structural:** A static table of 30k to 60k words cannot hold 12.4M proper names, and an 80-byte block-argmax sum over a full document structurally loses low-frequency token signals.
  * **Fixable inside it:** IDF scaling for unseen hash codes and storing more than one active slot per block during corpus indexing are adjustable inside the table framework, though finding entity mentions in questions should remain in the exact symbolic index.

#### 1.2 Choosing between namesakes
* **What the numbers point to:**
  * **Its use as a ranker & the framing of the task:** The fame set (1,910 EN and FR questions) requires 0 spoken picks on bare names and 0 wrong at >= 70% resolution on qualified names (sections 2, 4.2). A cosine ranker comparing question words against bundled item prototypes always produces a top candidate and a score gap. Raising the rank-normalized margin from 0.25 to 0.9 fails to reach 0 bare picks (dropping from 298 to 11 bare picks—still a kill) while collapsing qualified resolution from 77.4% down to 1.1% (and 2.5% at 0.9 when restricted to candidates carrying the relation) (section 4.2). Disambiguation is an exact evidentiary check (does one candidate hold a statement matching the question’s cue?), not a continuous similarity ranking problem.
  * **Bag-of-words encoding (on both question and item prototype):** Bundling an item's description and all of its property values into a single prototype destroys statement structure, while summing unsegmented question words allows template and relation words (`"pays"`, `"trouve"`, `"né"`, `"born"`, `"country"`) to collide with incidental words in one candidate's description. Those relation words caused all 11 bare picks at margin 0.9 and killed BM25 for the exact same reason (section 4.2).
  * **The English teacher & hash fallback for unseen words:** Because the teacher is English, most French cue words and French item descriptions fall back to deterministic hashes and match only on identical surface strings (sections 3, 4.2).
* **Structural vs. fixable inside a static word table:**
  * **Structural:** Using a bag-of-words prototype ranker as a namesake chooser is structurally incompatible with the hard rule that bare names get 0 spoken picks and namesakes are chosen only on evidence that holds for one candidate (section 2).
  * **Fixable inside it:** Stripping relation and function words before scoring (as B1q does) and replacing the English teacher with a multilingual teacher are fixable, but B1q (0.764 resolved, 2 bare picks, 3 wrong; section 4.2) has already proved that symbolic role-splitting and statement checking is the right mechanism.

#### 1.3 Relations
* **What the numbers point to:**
  * **Framing of the task (global vs. local candidate sets, and ranker handover on missing facts):** Scoring a question against all 12,684 Wikidata properties (25,368 EN/FR codes) yields recall@5 of 0.417 on SimpleQuestions (56.0% wrong) and 0.361 on QALD FR (63.9% wrong) (section 4.3). Restricting candidates to the item’s own outgoing and incoming properties (median 39 on SimpleQuestions, 175–190 elsewhere) lifts SimpleQuestions EN to 0.887 (pass line 0.85), *but* creates a fatal ranker trap when the store is incomplete: on the 38 rows where the item does not carry the gold property today, qFHRR at margin 0 hands over a semantically near property that the item *does* carry on 36 of 38 rows (16 of 38 at margin 0.25, vs. 6 for the lexicon, 4 for the 450M, and 0 for the grammar) (section 4.3). Because the item genuinely holds that near property in the store, the VM verifies the hop and certifies a wrong answer (sections 2, 4.3).
  * **The English teacher & hash fallback for unseen words:** Even restricted to the item's own properties, qFHRR drops from 0.755 on QALD EN to 0.536 on QALD FR (failing the 0.80 gate with 45.8% wrong at margin 0, and 0.349 with 15.1% wrong at margin 0.25), and from 0.554 on Mintaka EN to 0.397 on Mintaka FR (section 4.3). French words hit the hash fallback, turning semantic retrieval into brittle exact-word matching.
  * **Bag-of-words encoding (loss of direction and 0.5/0.5 hash dilution):** Order and direction are lost, so qFHRR cannot distinguish an outgoing property from an incoming one. When the asked-kind filter (`who`, `where`, `when`, `how many`) is applied, it drops the gold on `"who"` questions asked from the value’s side (`"who was born in Paris?"`, where the gold is an incoming property of Paris) and on `"how many"` over item-valued properties, turning every dropped gold into a missing-fact row and driving qFHRR’s missing-fact handovers up from 36 to 83 (section 4.3). Additionally, blending 0.5 semantic code with 0.5 surface-word hash (section 3) penalizes non-overlapping relation paraphrases by half their cosine range.
* **Structural vs. fixable inside a static word table:**
  * **Structural:** Loss of syntactic direction (outgoing vs. incoming links) and the inability of a local-only ranker to know that the true asked property is missing from the item’s record are structural to scoring only an item's carried properties with a bag of words.
  * **Fixable inside it:** The French collapse (English-only teacher and vocabulary) and the 0.5/0.5 hash dilution on property codes are directly fixable inside the word table build.

---

### 2. Fixes

#### 2.1 Fixes for Weakness 1: Entity names

1. **Proposal 1.1: Canonicalized Token-Subset Keys in the Exact Name Index (Closing the 12.6% Mintaka EN Gap)**
   * **What changes:** Keep qFHRR off question entity-finding; rely on the 12.4M exact/typo index (section 4.1). Because Mintaka FR already reaches 99.6% while Mintaka EN sits at 87.4% (section 4.1), the 12.2-point English gap comes from English surface variations (leading articles, honorifics, possessive `'s`, and inverted token order). Add a normalized token-set key (lowercase, strip English stopwords/possessives, sort remaining tokens alphabetically) to the exact name index as a fallback tier when exact and typo lookups miss, feeding any multi-item hits into B1q.
   * **Harness and set:** Mintaka EN and FR mention-finding set (section 4.1), plus the 1,910-question fame set, Mintaka, and WebQSP_EL end-to-end entity linking sets (section 4.2) on a pre-registered dev/test split.
   * **Pass line and kill line:**
     * *Pass:* Mintaka EN gold-in-set recall rises from 87.4% to >= 93.0% `(estimate)`, fame set bare picks = 0, and Mintaka / WebQSP_EL wrong rate stays <= 2.0% (section 4.2).
     * *Kill:* Mintaka EN recall gain < 2.0 points `(estimate)`, or expanded candidate sets push B1q p95 latency over 50 ms (section 2).
   * **Cost:** Build: ~0.5 CPU hours `(estimate)` over 12.4M names. Memory: +300 to 500 MB `(estimate)` in the index. Serving latency: +0.5 to 1.5 ms `(estimate)` on exact-miss fallback; +5 to 10 ms `(estimate)` inside B1q when candidate sets grow to >= 6 items (section 4.2).
   * **Hard rule near:** Comes near **A refusal is better than a wrong answer** (larger candidate sets must be filtered by B1q, never guessed) and **lookup-and-check within p95 50 ms** (section 2).

2. **Proposal 1.2: Unseen-Word Hash Inverted Postings + High-IDF OOV Weighting for Open-Corpus Entity Search**
   * **What changes:** For open-corpus document retrieval (BEIR dbpedia-entity, 4.64M docs; section 4.1), stop letting unseen entity-name hashes get washed out in the 80-byte block-argmax sum. Give unseen words (and capitalized low-frequency domain words) a 1.0 hash weight with a boosted IDF floor, and maintain a compact inverted posting list keyed by the 80-byte deterministic hash codes of out-of-vocabulary words appearing in each document. Union the top-50 hash-posting hits with the top-50 compact qFHRR hits before running the 20 KB dense re-rank.
   * **Harness and set:** BEIR dbpedia-entity (4.64M documents) and the 1M-document known-corpus retrieval benchmark (sections 3, 4.1).
   * **Pass line and kill line:**
     * *Pass:* BEIR dbpedia-entity shortlist recall@100 rises from 0.14 to >= 0.45 `(estimate)` and dense-reranked nDCG@10 rises from 0.096 to >= 0.220 `(estimate)` (matching dense potion-8M's 0.224; section 4.1), while 1M-doc hit@1 stays >= 0.71 (section 3).
     * *Kill:* BEIR nDCG@10 < 0.140 `(estimate)` or shortlist + rerank latency exceeds 50 ms p95 (section 2).
   * **Cost:** Build: ~2 CPU hours `(estimate)` over 4.64M docs. Memory: +400 to 800 MB `(estimate)` for OOV hash postings. Serving latency: +1 to 4 ms `(estimate)` per query on 1 CPU core.
   * **Hard rule near:** Comes near **No neural model on MoWM's routing path** (stays 100% non-neural table/index lookup) and **p95 50 ms lookup-and-check** (section 2).

3. **Proposal 1.3: Top-2 Slot Compact Codes (160 Bytes/Doc) for Corpus Shortlisting**
   * **What changes:** On BEIR dbpedia-entity, dense re-ranking doubles compact nDCG@10 (0.096 vs. 0.046; section 4.1), showing that taking a single argmax per 128-slot block drops entity signals in multi-sentence documents. Store the top 2 slot indices per block (160 bytes per document instead of 80 bytes) in the corpus index.
   * **Harness and set:** BEIR dbpedia-entity (4.64M docs) and 1M-document retrieval benchmark (sections 3, 4.1).
   * **Pass line and kill line:**
     * *Pass:* BEIR shortlist recall@100 rises from 0.14 to >= 0.25 `(estimate)` and 1M-doc hit@1 rises from 0.71 to >= 0.74 `(estimate)`.
     * *Kill:* BEIR recall@100 < 0.18 `(estimate)` or scan latency breaches 50 ms p95 (section 2).
   * **Cost:** Build: ~1.5 CPU hours `(estimate)`. Memory: Doubles compact corpus index from 80 bytes/doc to 160 bytes/doc (+80 MB per 1M docs; +371 MB for 4.64M docs `(estimate)`). Serving latency: ~1.8x compact scan time `(estimate)`.
   * **Hard rule near:** Comes near **p95 50 ms lookup-and-check** (section 2).

#### 2.2 Fixes for Weakness 2: Choosing between namesakes

1. **Proposal 2.1: Inflected EN/FR Relation & Function-Word Stop-Lexicon + Short-Acronym Guard in B1q**
   * **What changes:** Do not use qFHRR as a namesake chooser; harden B1q (section 4.2). One of B1q's 2 bare picks occurred because the French past participle `"né"` was missing from the relation lexicon, got accent-folded to `"ne"`, and matched the alias `"NE"` for Nebraska (section 4.2). Expand B1q’s relation and function-word exclusion lexicon in EN and FR to include all inflected verb forms, past participles, and gender/number agreements (`né, née, nés, nées, mort, morte, situé, située, trouve, appelé, écrit`, etc.), and prohibit 2-letter lowercase/accent-folded question tokens from matching uppercase acronym aliases unless the token was capitalized in the question.
   * **Harness and set:** Fame set (1,910 questions EN and FR), ShadowLink, Mintaka, and WebQSP_EL (section 4.2), plus a fresh held-out dev/test split of bare and qualified namesake questions.
   * **Pass line and kill line:**
     * *Pass:* Fame set bare single picks = 0 (down from 2), qualified resolved >= 0.75 (gate is 0.70), and wrong rate on Mintaka and WebQSP_EL drops from 2.3% / 2.2% to <= 2.0% (at most 1 in 50 rows; section 4.2).
     * *Kill:* Bare picks > 0 on the fame set or qualified resolved < 0.70.
   * **Cost:** Build: ~2 engineering hours and < 5 CPU minutes `(estimate)`. Memory: < 5 MB `(estimate)`. Serving latency: +0.0 ms (same hash-table role check inside B1q’s 1.4 / 6.2 ms at 2 candidates and 10.2 / 40.6 ms at >= 6 candidates; section 4.2).
   * **Hard rule near:** Enforces **A refusal is better than a wrong answer** (0 bare picks) and **Pre-registration** (must be validated on an unseen split, not only seed 0; section 2).

2. **Proposal 2.2: Two-Hop Transitive Hierarchy Check in B1q (Eliminating the 3 Qualified Wrong Picks)**
   * **What changes:** All 3 of B1q's wrong picks on the fame set happened because the gold item's `"located in"` statement was one administrative level more specific than the question's discriminator (e.g., gold Rosebank $\rightarrow$ Cape Town $\rightarrow$ Western Cape, while another Rosebank stated Western Cape in 1 hop; section 4.2). Pre-materialize 1-step parent IDs for transitive containment properties (`located in`, `country`, `part of`) on every item record in the store. In B1q, count a candidate as holding the location qualifier if the cue value matches at 1 hop *or* 2 hops, without preferring 1 hop over 2 hops. Both Rosebanks then satisfy `"Western Cape"`, turning those 3 wrong picks into refusals (undecided) unless a second cue breaks the tie.
   * **Harness and set:** Fame set (1,910 questions), ShadowLink, Mintaka, and WebQSP_EL (section 4.2).
   * **Pass line and kill line:**
     * *Pass:* Fame set qualified wrong = 0 (down from 3), bare picks = 0, qualified resolved >= 0.70 (currently 0.764; section 4.2), ShadowLink Top-Shadow gap <= 10 points, and Mintaka / WebQSP_EL wrong rate <= 2.0%.
     * *Kill:* Qualified wrong > 0, qualified resolved < 0.70, or p95 latency > 50 ms.
   * **Cost:** Build: ~1 CPU hour `(estimate)` to pre-join 2-hop parent location IDs across the 4.8M items. Memory: +150 to 300 MB `(estimate)` on the layered store. Serving latency: +0.5 to 2.0 ms `(estimate)` (keeps >= 6 candidates around ~42 ms p95, inside the 50 ms budget; sections 2, 4.2).
   * **Hard rule near:** Enforces **A refusal is better than a wrong answer** and **Namesakes are never ranked by popularity**, while coming near **lookup-and-check within p95 50 ms** (section 2).

3. **Proposal 2.3: Inverted Statement Intersection in B1q for High-Collision Names (> 20 Candidates)**
   * **What changes:** A fifth to a third of mentions name more than 20 items (section 4.1), and B1q's p95 latency reaches 40.6 ms at 6 or more candidates (section 4.2). Instead of iterating over every candidate item’s statements when candidate count >= 6, intersect the candidate item-ID sorted array with the incoming/outgoing subject-ID sorted list of each cue’s value items.
   * **Harness and set:** Fame set, ShadowLink, Mintaka, WebQSP_EL, and a high-collision subset of mentions naming > 20 items (sections 4.1, 4.2).
   * **Pass line and kill line:**
     * *Pass:* Exact same accuracy as B1q (0 change in picks), while p95 latency at >= 6 candidates drops from 40.6 ms to <= 15.0 ms `(estimate)`.
     * *Kill:* Any accuracy discrepancy against B1q or p95 > 40.6 ms.
   * **Cost:** Build: ~1 CPU hour `(estimate)` to sort incoming/outgoing posting lists by item ID. Memory: +200 to 400 MB `(estimate)` if not already sorted in the incoming-link index. Serving latency: Cuts p95 by ~25 ms at >= 6 candidates `(estimate)`.
   * **Hard rule near:** Protects **lookup-and-check within p95 50 ms** without truncating candidate lists by popularity (section 2).

#### 2.3 Fixes for Weakness 3: Relations

1. **Proposal 3.1: Global-vs-Local Property Contrast Gate (Stopping the 36/38 Missing-Fact Handover Trap)**
   * **What changes:** When an item lacks the gold property in the store (38 rows), scoring *only* the item's own properties causes qFHRR to hand over a near property that the item *does* carry on 36 of 38 rows at margin 0 (16 of 38 at margin 0.25), which the VM then certifies as true (section 4.3). Change the candidate scoring rule: always score the question against the item's carried properties *and* against the global frozen label lexicon / top-50 global qFHRR property codebook (all 12,684 properties). If a non-carried global property matches the question in the frozen lexicon or outscores the item’s best carried property by a dev-calibrated margin, refuse immediately with `"missing fact for asked property"` rather than handing the near property to the VM.
   * **Harness and set:** Section 4.3 relation harness (SimpleQuestions EN, QALD EN, QALD FR, Mintaka EN, Mintaka FR), the 38 missing-fact rows, plus a new pre-registered missing-fact test set (built by withholding the gold property from 500 dev/test items).
   * **Pass line and kill line:**
     * *Pass:* Missing-fact handovers on the 38 rows drop from 36 (margin 0) / 16 (margin 0.25) to <= 4 (matching or beating the 450M’s 4 and lexicon’s 6; section 4.3), SimpleQuestions EN recall@5 stays >= 0.85, and pooled wrong rate drops below the lexicon's 14.5%.
     * *Kill:* Missing-fact handovers > 6 on the 38 rows, or recall@5 on present-fact rows drops by > 2.5 points `(estimate)`.
   * **Cost:** Build: ~2 CPU hours `(estimate)` to tune the global-vs-local refusal margin on dev. Memory: +2 MB compact / +50 MB dense `(estimate)` for the global property codebook/lexicon in RAM. Serving latency: +0.2 to 0.8 ms `(estimate)` per question.
   * **Hard rule near:** Directly protects **Only the VM certifies** and **A refusal is better than a wrong answer** (prevents tricking the VM with an existing but unasked property; section 2).

2. **Proposal 3.2: Multilingual (EN+FR) Teacher Distillation (qFHRR v5) + 0.15/0.85 Hash-to-Semantic Blend on the Property Codebook**
   * **What changes:** Distill a bilingual EN+FR teacher into a 60k word table built from balanced EN and FR document frequency plus every word in the 25,368 EN and FR Wikidata property labels and aliases (sections 3, 4.3). When building the 25,368 property codes, reduce the random hash blend weight from 0.50 to 0.15 `(estimate)` (0.85 semantic / 0.15 hash) so that paraphrases and French-English equivalents are not penalized half their cosine similarity when surface words differ.
   * **Harness and set:** Section 4.3 relation harness across all 5 sets (SimpleQuestions EN, QALD EN, QALD FR, Mintaka EN, Mintaka FR) and the section 3 routing harness (15-domain screen and 42 real domains) to confirm routing does not regress.
   * **Pass line and kill line:**
     * *Pass:* SimpleQuestions EN recall@5 >= 0.85, QALD FR recall@5 >= 0.80 (up from 0.536; section 4.3), and pooled across the 5 sets (with Proposal 3.1's margin gate) qFHRR beats the frozen label lexicon by >= 3.0 points (the pre-registered keep line that v4 missed at +2.7 points; section 4.3) at a wrong rate <= 14.5%, while 42-domain routing stays >= 95% of teacher (section 3).
     * *Kill:* QALD FR recall@5 < 0.72 `(estimate)` or pooled gain over the lexicon < 3.0 points at matched wrong rate.
   * **Cost:** Build: ~1 GPU hour on an A100 80 GB or ~4 CPU hours `(estimate)` to embed the 60k EN+FR vocabulary and 25,368 property codes. Memory: Unchanged (60k word table + 25,368 codes). Serving latency: Unchanged (6.8 to 25 µs per text encode; section 3).
   * **Hard rule near:** Respects **No neural model on MoWM's routing path** and **Scores are not thresholds** (all margins must be re-set on dev with date and data recorded; section 2).

3. **Proposal 3.3: Direction-Aware Asked-Kind Filter (Fixing Incoming `"Who"` and Item-Valued `"How Many"`)**
   * **What changes:** The asked-kind filter (`who`, `where`, `when`, `how many`) worked for `"place"` (0.77 recall, 6% wrong) and `"time"` (0.63 recall, 0% wrong), but dropped the gold on incoming `"who"` questions (`"who was born in Paris?"`) and item-valued `"how many"` questions, inflating missing-fact handovers from 36 to 83 (section 4.3). Fix the filter rule: (1) for `"who"`, keep outgoing properties whose object value is a person/agent *and* incoming properties whose subject is a person/agent; (2) for `"how many"`, keep numeric-literal properties *and* multi-edge item properties (since CubeLang counts the edges).
   * **Harness and set:** Section 4.3 relation harness across all 5 sets, stratified by question word (`who`, `where`, `when`, `how many`), plus the missing-fact handover counter.
   * **Pass line and kill line:**
     * *Pass:* Missing-fact handovers do not increase over the unfiltered baseline (<= 36 at margin 0, and <= 4 when combined with Proposal 3.1), while pooled wrong rate drops by >= 3.0 points `(estimate)` compared to unfiltered qFHRR.
     * *Kill:* Any gold property dropped on incoming `"who"` or item-valued `"how many"` dev rows, or missing-fact handovers > 36 at margin 0.
   * **Cost:** Build: ~1 CPU hour `(estimate)` to attach 1-byte subject-kind tags to incoming property edges in the store. Memory: +50 to 100 MB `(estimate)`. Serving latency: < 0.1 ms `(estimate)`.
   * **Hard rule near:** Protects **A refusal is better than a wrong answer** (eliminates filter-induced missing-fact handovers; section 2).

---

### 3. The trade-offs

| Job | Which part should do it (in stated order) | Latency (sections 3–6) | Memory (sections 3–6) | Accuracy (sections 3–6) | Failure mode |
|---|---|---|---|---|---|
| **Finding a name** | **1.** Exact symbolic name index (12.4M names of 4.8M items)<br>**2.** Typo index (and normalized token-set key) on exact misses (section 4.1) | Unknown standalone (fits inside the p95 50 ms lookup-and-check stage; section 2) | Unknown (stores 12.4M names of 4.8M Wikidata items + typo index; section 4.1) | Holds gold under exact name for 87.4% of Mintaka EN mentions and 99.6% of FR mentions (section 4.1). *(qFHRR on BEIR dbpedia-entity: 0.046 compact / 0.096 dense nDCG@10; section 4.1)* | **Refusal** (when the mention misses both exact and typo indices, or returns > 20 items with no resolving cue). |
| **Choosing between namesakes** | **1.** Exact symbolic check **B1q**: span role rule (excluding relation lexicon & function words) $\rightarrow$ exact name lookup of cues $\rightarrow$ 1-to-2-hop bidirectional statement check on candidates; 1 standing = pick, else undecided (section 4.2) | p50 / p95: 1.4 / 6.2 ms at 2 candidates; 10.2 / 40.6 ms at 6 or more candidates (section 4.2) | Unknown (reads relation lexicon, name index, and item statements in layered store; section 4.2) | Fame set (1,910 rows): 0.764 resolved (692 qualified right, 3 wrong, 2 bare picks, 61 undecided, 154 refused). Mintaka: 2.3% wrong. WebQSP_EL: 2.2% wrong (section 4.2) | **Refusal** (undecided when 0 or >= 2 candidates match), **wrong answer** (2.2%–2.3% on Mintaka/WebQSP_EL; 3/1,910 from 2-hop locations; 2/1,910 bare picks from missing FR `"né"`), or **slowness** (near 40.6 ms p95 when candidates >= 6). |
| **Choosing the relation** | **1.** Direction-aware asked-kind filter over item's own outgoing + incoming properties<br>**2.** Grammar + frozen label lexicon + rebuilt multilingual qFHRR (with global-vs-local missing-fact refusal gate) to shortlist top-5 properties<br>**3.** 450M (programs seat) to write CubeLang program<br>**4.** VM to certify hop (sections 4.3, 5) | qFHRR encode: 6.8 to 25 µs on 1 CPU core (section 3). 450M programs seat: ~2.4 s per program on grilly2 (section 5). Grammar/lexicon/VM latency: unknown | qFHRR property codebook: 25,368 codes (80 B compact or 20 KB dense per code; sections 3, 4.3). 450M: ~576M weights (445M read/token; section 5). Lexicon/grammar memory: unknown | Recall@5 (wrong rate) among item's own properties:<br>• **qFHRR m=0:** SimpleQ 0.887 (9.0%), QALD EN 0.755 (22.5%), QALD FR 0.536 (45.8%), Mintaka EN 0.554 (44.6%), Mintaka FR 0.397 (60.3%), 36/38 missing-fact handovers.<br>• **qFHRR m=0.25:** 0.573 (3.0%), 0.400 (9.5%), 0.349 (15.1%), 0.223 (14.1%), 0.223 (27.3%), 16/38 handovers.<br>• **Lexicon:** 0.563 (4.0%), 0.360 (17.5%), 0.217 (15.1%), 0.331 (28.1%), 0.190 (21.5%), 6/38 handovers.<br>• **450M programs:** 0.470 (6.3%), 0.315 (11.0%), 0.115 (9.6%), 0.182 (15.7%), 0.066 (7.4%), 4/38 handovers.<br>• **Grammar:** 0/38 handovers (section 4.3) | **Wrong answer** (when a missing-fact handover picks a near property that the item carries, which the VM then certifies; section 4.3), **refusal** (when gated out or unverified by VM), and **slowness** (~2.4 s when the 450M programs seat runs; section 5). |
| **Reading French** | **1.** Exact symbolic name index + B1q (with inflected FR relation lexicon) for names/namesakes<br>**2.** Rebuilt EN+FR qFHRR table for routing & property shortlist<br>**3.** 450M (or grown model) with candidate properties in prompt for CubeLang program writing (sections 4.1–5) | Same as component stages: 6.8–25 µs (qFHRR; section 3), 1.4–40.6 ms (B1q; section 4.2), ~2.4 s (450M programs seat; section 5) | Fits inside the 30k–60k qFHRR vocabulary (section 3) and the 450M's 128,000-token BBPE vocabulary (131M weights in, 131M out; section 5) | Name finding: 99.6% Mintaka FR (section 4.1). Relations (recall@5, wrong rate): QALD FR: qFHRR m=0 0.536 (45.8%), m=0.25 0.349 (15.1%), lexicon 0.217 (15.1%), 450M programs 0.115 (9.6%), 450M label-LoRA 0.241. Mintaka FR: qFHRR m=0 0.397 (60.3%), m=0.25 0.223 (27.3%), lexicon 0.190 (21.5%), 450M programs 0.066 (7.4%) (section 4.3) | **Refusal** (when FR words hit the hash fallback in EN-only qFHRR or miss the 450M's French program coverage) or **wrong answer** (unstripped FR `"né"` matching `"NE"` in B1q, or near-property handovers; sections 4.2, 4.3). |
| **Routing to a world** | **1.** qFHRR word table alone, with bundled domain centroids and dev-calibrated margin gate to spawn (no neural model allowed; sections 2, 3) | 6.8 to 25 µs for a short text on 1 CPU core (350x faster than teacher; section 3) | 30k–60k word table (20 KB dense or 80 B compact per entry) + bundled domain centroids (section 3) | 15-domain screen: 0.341 vs. 0.358 teacher (0.95 of teacher, null 0.063). 42 real domains: 97% of teacher quality (60k vocab). Open-set routing: 92.4% precision, 86.9% of never-seen domains sent to spawn (section 3) | **Wrong answer / misroute** (7.6% false-positive rate on open-set routing, or 13.1% of unseen domains routed to an existing world instead of spawned; section 3) or **refusal / spawn**. |
| **Retrieval over a known corpus** | **1.** qFHRR compact shortlist (80 B/doc) + exact token-hash index for OOV entities<br>**2.** qFHRR dense re-rank (20 KB/doc)<br>**3.** Lookup-first walk, plan checker, and symbolic VM re-verification (sections 3, 4.1) | 6.8 to 25 µs query encode on 1 CPU core (section 3); corpus scan + VM check latency: unknown (must stay <= p95 50 ms on layered store; section 2) | Compact: 80 bytes/doc (~80 MB for 1M docs). Dense: 20 KB/doc (~20 GB for 1M docs) (section 3) | 1M docs compact: hit@1 0.71, hit@20 0.92. Verified pipeline: 0.992 claimed-answer precision at 0.646 coverage (unchanged under 100k distractors); with lookup-first walk + plan checker: 574/800 (0.718) certified at 1.000 precision (section 3) | **Refusal** (with the VM and plan checker, precision is 1.000 and unrecovered chains are refused, leaving 0.282 uncertified; section 3). Without VM check: **wrong answer** on tail entities. |
| **The n-gram memory** | **1.** qFHRR word codes (paraphrase teacher refined by WordNet) addressing a Hamming scan over 10,240 bits<br>**2.** VM certification if used to assert a fact (sections 2, 3) | 6.8 to 25 µs word lookup (section 3); Hamming scan latency: unknown | 10,240 bits (1,280 bytes) per n-gram row (section 3); total table size: unknown | Trigram with 1 word paraphrased reaches its row 74.3% of the time and a different fact 4.0% of the time (on 35 hand pairs; not yet confirmed on a fresh list; section 3) | **Refusal** (21.7% miss rate) or **wrong answer** (4.0% collision onto a different fact if not checked by the VM; section 3). |
| **Reading a fact the host placed in the context** | **1.** Host retrieves verified statements from store and binds object token embeddings to a canonical loci address at input<br>**2.** Loci tokens (H-K10) unbind at the middle layer of the 450M and copy via the 1,027-parameter path into logits<br>**3.** VM certifies any reasoning chain (sections 2, 6) | Unknown for the reader/talk seat on grilly2 (port in progress; logits match torch within 1e-4; programs seat takes ~2.4 s; sections 5, 6) | 450M trunk (~576M weights total, 445M read/token) + 1,027 copy-path parameters (sections 5, 6) | On 450M: 0.946 exact > 800 tokens back (run 5, 10 min train); 0.804 exact at equal training tokens vs. 0.019 ordinary read, text loss 2.2398 vs. 2.2491 (run 6). Word-keyed CPU toy (width 256): 0.854–0.899 on unseen subjects (vs. 0.000 hashed), ~12% of present facts answered wrong by copy/trunk switch (section 6) | **Wrong answer** (~12% switch error on present facts in the CPU toy, symmetric `(s, r)` collisions, or multi-value address overwrite) or **refusal** (when copy gate stays shut or relation wording was not in training shard; section 6). |

---

### 4. Growing the model

#### 4.1 Where a larger model is NOT the lever
* **Routing to a world:** A hard rule prohibits any neural model on MoWM's routing path (section 2), and the 60k qFHRR table already reaches 97% of teacher quality on 42 real domains in 6.8 to 25 µs (section 3).
* **Finding a name & choosing between namesakes:** The 12.4M-name exact/typo index hits 87.4% EN and 99.6% FR (section 4.1), and B1q resolves 0.764 of the fame set in 1.4 to 40.6 ms (section 4.2). A larger neural model cannot memorize 4.8M Wikidata items, cannot write facts (section 2), and cannot run inside the p95 50 ms lookup-and-check budget when the 450M already takes 2.4 s on the home GPU (sections 2, 5).
* **Retrieval over a known corpus & the n-gram memory:** Both are sub-millisecond/millisecond vector and bit-scan lookups (section 3).
* **Single-fact loci copy over long contexts:** On the existing 450M (width 1,024), training just 1,027 copy-path parameters for 10 minutes already reads facts > 800 tokens back at 0.946 exact (section 6).

#### 4.2 Where a larger model IS a potential lever
* **Choosing the relation (via the programs seat) and compositional CubeLang synthesis:** In its programs seat, the 450M scores 0.470 recall@5 on SimpleQuestions EN, 0.315 on QALD EN, 0.115 on QALD FR, 0.182 on Mintaka EN, and 0.066 on Mintaka FR (section 4.3). Even after training a LoRA adapter on 163,907 EN and FR property-label records, it reaches only 0.800 on SimpleQuestions EN and 0.241 on QALD FR, with a pooled wrong rate of 0.286 (nearly double the frozen lexicon’s 0.145) (section 4.3).
* **Reading French on the reasoning path:** Because the 128,000-token BBPE input embedding and untied output head consume 262M weights (131M each; section 5), Cubby’s 32-layer recurrent/attention trunk holds only ~314M weights (~445M read per token including one embedding lookup; section 5). That trunk capacity is stretched thin across English syntax, French syntax, CubeLang formal grammar, and Wikidata relation semantics.
* **Loci token pointing, paraphrase projection, and copy/trunk switch gating:** In section 6, the word-keyed loci reader requires the trunk to point at the right subject and relation spans in a natural question, map diverse relation phrasings to a shared unitary key, and operate the copy/trunk switch (which answers about 12% of present facts wrong on the width-256 toy and misses when the gate stays shut; section 6).

#### 4.3 Cheaper levers to try BEFORE growing the model
1. **Put the item’s candidate property labels into the 450M’s input context instead of forcing LoRA weights to memorize all 12,684 properties.** The 163,907-record LoRA adapter failed (0.286 wrong rate, 0.241 QALD FR; section 4.3) because it tried to recall global property names out of 12,684 options from adapter weights alone. Once the host resolves the subject item, the item has only a median of 39 (SimpleQuestions) to 175–190 properties (section 4.3), and a rebuilt EN+FR qFHRR + lexicon narrows that to a top-10 list in < 1 ms `(estimate)`. Train a LoRA adapter (~1 hour on an A100 80 GB; section 5) where the prompt lists those 10 candidate properties (plus `"NONE / MISSING FACT"`) and the 450M either selects one to write into the CubeLang program or refuses.
2. **Rebuild qFHRR with an EN+FR teacher and add the Global-vs-Local Missing-Fact Gate (Proposals 3.1 and 3.2).** This fixes the French hash fallback at the table level for ~1 GPU hour `(estimate)` and 0 ms added serving latency.
3. **Move the word-keyed loci reader from the width-256 CPU toy to the existing width-1,024 450M** and train a learned relation-projection adapter $W_r$ on the 163,907 EN/FR property labels so synonymous wordings map to the same unitary address.

#### 4.4 If the cheaper levers fall short: target size and required proof
* **What size:** **~1.2B to 1.5B total weights (~1.0B to 1.3B read per token)** `(estimate)`, achieved by widening the trunk from width 1,024 to **width 1,536 or 2,048** at 32 layers while keeping the exact same 128k BBPE tokenizer.
  * *Why this size:* Widening from 1,024 to 2,048 increases the non-embedding trunk from ~314M weights to ~1.25B weights `(estimate)` (a 4x increase in reasoning and bilingual trunk capacity) and doubles the spectral dimension available for loci binding (section 6), while keeping embedding overhead to ~17% of total weights instead of 45% (section 5). Going beyond ~1.5B (e.g., to 7B) would push home-GPU program generation on grilly2 from 2.4 s (section 5) to > 15–25 s `(estimate)`, breaking interactive usability and threatening the hard rule of one hundredth of frontier cost or less (section 2).
* **What it would have to show (pre-registered gates):**
  1. **Relations:** SimpleQuestions EN recall@5 >= 0.90, QALD FR >= 0.80, Mintaka EN and FR >= +5.0 points over the lexicon + qFHRR baseline, pooled wrong rate <= 8.0%, and missing-fact handovers <= 4 of 38 (section 4.3).
  2. **Loci reader:** Present-fact copy/trunk switch wrong rate drops from ~12% (section 6) to <= 2.0%, and unseen-subject exact read stays >= 0.90.
  3. **Serving cost:** Program generation latency on grilly2 on the home GPU stays <= 4.5 s `(estimate)` (using grammar-constrained token masks for CubeLang).

---

### 5. Loci tokens

#### 5.1 Impact on each of the three weaknesses

1. **Entity names: Leave store-level finding as it is, but HELP in-context entity reading.**
   * Loci tokens encode facts that the host has *already retrieved* and placed into the context (section 6); they do not search the 4.8M Wikidata store or 4.64M BEIR documents, so store-level name finding remains the job of the 12.4M exact/typo index (section 4.1).
   * However, once the host places facts about an unseen or multi-token entity into the context, ordinary neural reading fails (0.019 exact in run 6; section 6). Because loci keys are built from the trunk’s subword embeddings and multi-piece names can be bound by position into a single unit ($\sum e(\text{piece}_i) \circledast S(i)$, up to ~55 pieces at width 1,024), there is no vocabulary miss, and never-seen subjects read back at 0.854 to 0.899 on the toy and 0.804 to 0.946 on the 450M (section 6).

2. **Choosing between namesakes: HURT if unresolved namesakes are placed in the same loci scene; leave as is if B1q resolves them first.**
   * By construction, **two items with the same name get the exact same key** (because $u(e(s))$ is computed solely from the subwords of the name string $s$), and **one address holds one value** (section 6). If the host places facts for several candidate items named `"Rosebank"` into the context keyed by the surface string `"Rosebank"`, their attribute values collide and overwrite one another at the same spectral addresses.
   * Worse, the copy/trunk switch answers about 12% of present facts wrong (section 6), which would fail the fame set’s 0-wrong qualified rule, the 0-bare-pick rule, and the 2.0% wrong limit on Mintaka and WebQSP_EL (section 4.2). B1q must resolve namesakes *before* the host writes the winning item's facts into loci tokens.

3. **Relations: LEAVE AS IS (or HURT) unless the host normalizes relation keys and breaks symmetry before binding.**
   * By construction, loci tokens share the core relation bottlenecks (section 6):
     1. *Wording mismatch:* A relation’s key comes from its words ($u(e(r))$), so a different wording or French phrasing of a relation produces a different address and misses when the question’s phrasing differs from the stored fact’s label.
     2. *Symmetry (commutativity):* Because unit-modulus spectral binding commutes, $\text{key}(s, r) = u(e(s)) \circledast u(e(r))$ is symmetric in subject and relation, so it cannot tell an outgoing link from an incoming link (`"who was born in Paris?"`; section 4.3).
     3. *Multi-valued relations and switch errors:* One address holds only one value (failing when an item has multiple values for a property), and the copy/trunk switch has a ~12% wrong rate on present facts (section 6).
   * Where loci tokens *help* on relations is strictly downstream of relation selection: once a canonical relation is matched, the copy path extracts the exact target value from context (0.804–0.946 exact; section 6) without the trunk inventing facts from its weights (upholding **Facts are never model-written**; section 2).

#### 5.2 How qFHRR and loci tokens should divide the work and combine
qFHRR and loci tokens use the exact same algebra (products of unit-modulus spectra; sections 3, 6) at different stages of the request. They should divide the work in a strict three-step handoff:
1. **Host Lookup & Symbolic Filtering (Exact Index + B1q + Multilingual qFHRR):**
   * The exact name index finds the mention's candidates; B1q symbolically resolves namesakes to a single Wikidata item; and the multilingual qFHRR table + frozen lexicon + direction-aware asked-kind filter maps the user’s wording onto a shortlist of canonical Wikidata properties on that item (or refuses via the global-vs-local missing-fact gate).
2. **Canonical Loci Addressing (The Bridge):**
   * When the host writes the resolved entity’s facts into the 450M context as loci tokens, do **not** key the loci address with raw surface relation words. Instead:
     * Key the relation slot using a **canonical property embedding** (or project qFHRR’s paraphrase-trained relation code to width 1,024 as $u(r)$), and train the asker’s relation pointer on the 163,907 EN/FR property-label records (section 4.3) to map surface question words onto that same canonical unitary $u(r)$.
     * Break binding commutativity by binding the subject with a fixed random role unitary $R_{\text{subj}}$ and the relation with a directional role unitary $R_{\text{out}}$ or $R_{\text{in}}$: $\text{key}(s, r, \text{dir}) = (u(e(s)) \circledast R_{\text{subj}}) \circledast (u(r) \circledast R_{\text{dir}})$ `(estimate: 0 extra parameters, exact algebraic symmetry breaking)`.
     * For multi-valued properties (where one address holds one value; section 6), either bind a value-index unitary $V_j$ for the $j$-th value or route multi-valued sets through the CubeLang VM program rather than single-slot loci copy.
3. **Model Read & VM Certification:**
   * The 450M unbinds the canonical loci address to copy the exact value (or assemble the CubeLang query), and **only the VM certifies** the final hop against the store (section 2).

---

### 6. What you would not do, and open questions

#### 6.1 What I would not do
1. **Do not put qFHRR, BM25, or any continuous similarity ranker back in the namesake-choosing seat.** Section 4.2 showed that ranking bundled item prototypes fails the 0-bare-pick rule at every margin up to 0.9 while crushing qualified resolution to 1.1%. Keep B1q as the sole namesake chooser.
2. **Do not try to add n-gram or word-order binding into the general qFHRR document/routing table.** Section 3 already measured n-gram and binding variants inside qFHRR and proved they cost more than they bought. Handle word roles and directionality symbolically in B1q and the store’s incoming/outgoing edge tags.
3. **Do not use qFHRR to find entity names in questions.** The 12.4M exact/typo index gets 87.4% EN and 99.6% FR (section 4.1), whereas qFHRR gets 0.046/0.096 nDCG@10 on open-web entities.
4. **Do not score candidate relations solely among the item’s carried properties without a global missing-fact contrast check.** Scoring only present properties guarantees that whenever the store is missing the asked fact (38 rows), the ranker hands over a near property that the item *does* carry (36/38 at margin 0; section 4.3), blinding the VM into certifying a wrong answer.
5. **Do not train another standalone LoRA adapter to memorize all 12,684 property labels without passing the item's candidate properties in the prompt.** That was already tried on 163,907 EN/FR records and produced a 0.286 pooled wrong rate (worse than the lexicon’s 0.145) and 0.241 on QALD FR (section 4.3).
6. **Do not allow the loci token copy/trunk switch to emit an uncertified final answer, rank namesakes by popularity, or place a neural model on MoWM's routing path.** Each violates a hard rule of section 2, and the loci switch answers ~12% of present facts wrong in the CPU toy (section 6).

#### 6.2 Open questions
1. **What caused B1q’s 2.3% wrong rate on Mintaka and 2.2% wrong rate on WebQSP_EL (section 4.2)?** Section 4.2 explains the 3 wrong picks and 2 bare picks on the fame set (2-hop location hierarchies and French `"né"` $\rightarrow$ `"ne"`), but does not state what caused the wrong picks that left Mintaka and WebQSP_EL 0.2 to 0.3 points above the 2.0% (1 in 50) gate. Are they also 2-hop hierarchy misses and unstripped relation words, or alias collisions on cue values?
2. **Why did B1q leave 61 undecided and 154 refused rows on the fame set (section 4.2)?** What share of those 215 unresolved rows failed because the question expressed the cue as an adjective or demoným (e.g., `"French"` instead of `"France"`, where an exact name lookup of the cue misses the country item) versus multiple candidates genuinely holding the exact same statement in Wikidata?
3. **What exact teacher and vocabulary size are in serving table "v4" (section 3)?** Section 3 notes v1 used `all-MiniLM-L6-v2`, later builds used `potion-retrieval-32M`, and serving uses `"v4"`, which is English-only. I don't know which teacher produced v4 or whether a multilingual static teacher (`potion-multilingual` or similar) is already on disk in GRL’s repositories for building v5.
4. **How does the word-keyed loci copy/trunk switch behave at width 1,024 on the 450M (section 6)?** The word-keyed address (`key(s, r) = u(e(s)) * u(e(r))`) and its ~12% present-fact switch error rate were measured on the width-256 CPU toy, not yet on the width-1,024 450M. I don't know whether moving from width 256 to width 1,024 reduces that 12% switch error or why the copy gate stays shut on misses.
5. **Where is the 2.4-second latency spent when the 450M programs seat runs on grilly2 (section 5)?** I don't know the average token length of a generated CubeLang program, the hardware model of the home GPU, or whether the 2.4 s is dominated by Vulkan kernel launch overhead or sequential token decoding. Knowing that profile will set the exact ceiling on how far Cubby can grow beyond 450M without slowing down the reasoning path.

## deepseek/deepseek-v4-pro-0813

*22990 tokens out, 316.6 s, finish `stop`*

## 1. Diagnosis

**Entity names.** The entity-search numbers in 4.1 point mainly to the vocabulary and the hash fallback, not to the ranker alone: v1 compact nDCG@10 is 0.040 and v4 is 0.046; recall@100 is about 0.14 and caps the pipeline; potion-8M dense is 0.224 and BM25 is about 0.313. The preprint’s own reading is that entity names are tail vocabulary that a document-frequency word list misses. Unseen words receive a hashed identity code, not a meaning code, so the table cannot recover a semantic match for a name it did not store. Bag-of-words order loss is a stated limit, but the measured gap is driven by missing name vocabulary and identity fallback.

This is partly structural to a static word table: any finite word list has an unseen-word fallback, and the open-web name tail is effectively unbounded. The specific 30k–60k document-frequency cutoff is fixable by changing vocabulary construction, but the fallback problem remains unless names are handled by a separate exact index.

**Choosing between namesakes.** The 4.2 table shows qFHRR as chooser is killed at every margin, with 298 bare single picks at margin 0.25 and still 11 at 0.9; those 11 come from the question template’s relation words. This points to **use of qFHRR as a ranker** and to a framing error: a cosine-best item past a margin still always selects something, and the score is similarity to templates, not evidence that determines one candidate. BM25 was killed for the same reason. B1q’s span rule and symbolic check are superior because they treat every question word as a role and only pick when one candidate has the cue as a statement value.

This is not structural to the word table; it is a deployment choice. The table can propose candidates, but the chooser must be symbolic and evidence-based under the hard rules. The English teacher and hash fallback contribute specific French errors, but they do not explain the EN bare-name kills.

**Relations.** The 4.3 numbers point to several causes:
- **French**: qFHRR at margin 0 falls from 0.887 SimpleQuestions EN to 0.536 QALD FR, and among all properties from 0.417 to 0.361 QALD FR. This is consistent with the English teacher and hash fallback described in 4.2: French words match only as written, not by meaning.
- **Missing-fact danger**: qFHRR at margin 0 hands over a near property on 36 of the 38 rows where the item does not carry the gold property; the lexicon hands over 6, the 450M 4, the grammar 0. This is a ranker failure: qFHRR returns the nearest on-record property, and the VM verifies it, so the wrong answer is certified. The cause is using qFHRR to choose when the gold is absent.
- **Word order**: bag-of-words is a stated limit and plausibly weakens property-label wording matches, though 4.3 does not isolate its contribution numerically.

Bag-of-words is structural to this word table. The English teacher and hash fallback are fixable inside it by rebuilding with a multilingual or French-capable teacher and vocabulary. Use as a final ranker is fixable in the harness, not structural.

## 2. Fixes

### 4.1 Entity names

**Proposal E1 — Exact name index first; qFHRR bypassed for named lookup.**  
Changes: For entity-bearing queries, resolve names through the exact name index and typo index already described in 4.1; only after a name resolves should retrieval proceed, and qFHRR should not be the first-stage ranker for open-web entity search.  
Harness and set: BEIR dbpedia-entity 4.64M-document set used in 4.1, same dev/test read.  
Pass line: nDCG@10 ≥ 0.25 on the stated BEIR set.  
Kill line: nDCG@10 ≤ 0.15.  
Cost: Exact index already exists; serving latency and memory are not stated in the brief. qFHRR serving is 6.8–25 µs for a short text, but this proposal uses it less, not more.  
Hard rule near: “Scores are not thresholds” — the line is set for this data/date and does not transfer.

**Proposal E2 — Store name/alias strings as qFHRR word or entity codes, but only as an exact-match auxiliary.**  
Changes: Add exact-index names, aliases, titles, and redirects to the offline vocabulary or entity-prototype store, so an exact name is present as a code; keep exact index authoritative for lookup. qFHRR is used only to re-rank an exact-index candidate set.  
Harness and set: BEIR dbpedia-entity, same set as 4.1.  
Pass line: nDCG@10 ≥ 0.15 and recall@100 ≥ 0.20.  
Kill line: nDCG@10 ≤ 0.08 or recall@100 ≤ 0.14.  
Cost: Vocabulary growth from 12.4M names; memory not stated. If each name were stored as an 80-byte compact code, 12.4M would be about 992 MB, but this is an estimate because the name index format is not stated.  
Hard rule near: “Scores are not thresholds.”

**Proposal E3 — Retire qFHRR from open-web entity search; use BM25 plus exact name index.**  
Changes: Use published BM25 as the retrieval baseline for dbpedia-entity, with exact name/typo lookup for named disambiguation; do not spend qFHRR cost on a task where it caps recall.  
Harness and set: BEIR dbpedia-entity, same set as 4.1.  
Pass line: nDCG@10 ≥ 0.30.  
Kill line: nDCG@10 ≤ 0.25.  
Cost: qFHRR serving cost removed; BM25 index memory and latency not stated.  
Hard rule near: “Scores are not thresholds.”

### 4.2 Choosing between namesakes

**Proposal N1 — Minimal B1q repair.**  
Changes: Add the missing French relation/function word “né” and other French participle/function words seen in the 2 bare picks and 3 wrong picks; allow one-level traversal for cases such as “located in” when the candidate’s statement is one level below the requested discriminator, only if the traversal is a stored fact and the VM re-verifies.  
Harness and set: fame set dev seed 0 from 4.2, plus the Mintaka and WebQSP_EL 50-row sets.  
Pass line: fame bare single picks = 0; fame qualified wrong = 0; fame resolved ≥ 0.70; Mintaka/WebQSP_EL ≤ 1 wrong in 50 rows.  
Kill line: any bare spoken pick, any qualified wrong, or fame resolved < 0.70.  
Cost: CPU-only lexicon/rules; B1q latency already measured at 1.4/6.2 ms p50/p95 for 2 candidates and 10.2/40.6 ms for 6 or more candidates. Added traversal cost not stated.  
Hard rule near: “A refusal is better than a wrong answer”; “namesakes are never ranked by popularity.” This proposal ranks nothing.

**Proposal N2 — qFHRR as recaller, never as chooser.**  
Changes: After B1q fails to find a cue value by exact name lookup, qFHRR may propose candidate cue items or candidate entities, but the final pick still requires B1q’s symbolic statement check. qFHRR’s score is never a spoken pick and never a margin.  
Harness and set: fame set and Mintaka/WebQSP_EL, same as 4.2.  
Pass line: fame qualified resolved rises at least 5 points over B1q with 0 new wrong and 0 bare picks; Mintaka/WebQSP_EL ≤ 1 wrong in 50 rows.  
Kill line: any new bare or wrong pick, or no resolved gain.  
Cost: qFHRR lookup 6.8–25 µs per short text; memory 20 KB dense or 80 bytes compact per document/prototype.  
Hard rule near: “Namesakes are never ranked by popularity”; “a refusal is better than a wrong answer.”

**Proposal N3 — French function/relation-word gating.**  
Changes: For French questions, do not fold past participles and function words to other words before role assignment; treat “né” and similar wordings as relation words, and refuse if a French relation word is absent from the lexicon rather than letting it become a cue.  
Harness and set: French fame rows and the 4.2 French condition.  
Pass line: no bare pick from a French fold; qualified wrong = 0; resolved ≥ 0.70.  
Kill line: any French bare pick or any qualified wrong.  
Cost: small lexical list; CPU only.  
Hard rule near: “A refusal is better than a wrong answer.”

### 4.3 Relations

**Proposal R1 — Symbolic label/alias gate over qFHRR recall.**  
Changes: qFHRR may recall candidate properties at margin 0, but a property is accepted only if its label/alias or asked-kind is present in the question’s relation words; otherwise the question is refused before the VM builds an answer. This directly reduces near-property handovers when the gold is missing.  
Harness and set: relation sets of 4.3 plus the 38 missing-fact rows.  
Pass line: SimpleQuestions recall@5 ≥ 0.85; QALD FR recall@5 ≥ 0.80; pooled wrong rate ≤ 14.5%; missing-fact handovers ≤ 6.  
Kill line: below either recall gate, wrong rate > 14.5%, or missing-fact handovers > 6.  
Cost: qFHRR 6.8–25 µs plus symbolic label/alias check; full relation-path latency not stated.  
Hard rule near: “Only the VM certifies” — this gate is not certification; the VM still verifies. Also “scores are not thresholds”: pass/kill lines are dev/data-specific.

**Proposal R2 — French property-label lexicon or multilingual qFHRR.**  
Changes: Add French property labels/aliases to the symbolic relation lexicon, or rebuild qFHRR with a French-capable teacher and French domain vocabulary so French relation words do not fall back to hash identity. qFHRR remains a proposer, not a final chooser.  
Harness and set: QALD FR relation set from 4.3, plus the EN sets as control.  
Pass line: QALD FR recall@5 ≥ 0.80 and QALD FR wrong rate ≤ 15.1%.  
Kill line: QALD FR recall@5 ≤ 0.70 or wrong rate > 15.1%.  
Cost: If only lexicon, CPU cost small; if retraining a multilingual teacher, GPU hours and memory are not stated. Serving remains 6.8–25 µs for a short qFHRR lookup.  
Hard rule near: “Scores are not thresholds.”

**Proposal R3 — Use grammar and lexicon as final chooser; qFHRR only recall under the R1 gate.**  
Changes: Keep the existing grammar and frozen label lexicon as the chooser; do not let qFHRR’s score pick the relation. qFHRR may add recall only when its proposed property passes the symbolic label/alias gate of R1.  
Harness and set: SimpleQuestions, QALD EN/FR, Mintaka EN/FR from 4.3.  
Pass line: pooled recall at least 0.45 with pooled wrong ≤ 14.5%, and missing-fact handovers ≤ 6.  
Kill line: pooled recall < 0.40, pooled wrong > 14.5%, or missing-fact handovers > 6.  
Cost: lexicon and grammar CPU; qFHRR 6.8–25 µs when used; memory 20 KB dense/80 bytes compact per code.  
Hard rule near: “Only the VM certifies”; “facts are never model-written” — qFHRR and the 450M may propose but only statements in the store certify.

## 3. The trade-offs

| Job | Which part should do it | Latency | Memory | Accuracy | Failure mode |
|---|---|---|---|---|---|
| Finding a name | Exact symbolic index and typo index; qFHRR not first-stage | Unknown | Unknown; 12.4M names of 4.8M items, index size not stated | Exact name holds gold for 87.4% of Mintaka EN mentions and 99.6% of FR mentions (4.1); returns a set, not a pick | Refusal on exact/typo miss; wrong answer not measured separately |
| Choosing between namesakes | B1q span role plus symbolic statement check; qFHRR not chooser | p50/p95 1.4/6.2 ms at 2 candidates; 10.2/40.6 ms at 6+ candidates (4.2) | Unknown | Fame: 692 right, 3 wrong, 2 bare, 0.764 resolved; Mintaka/WebQSP_EL wrong 2.3%/2.2% (4.2) | Wrong answer (3 qualified wrong, 2 bare) and refusal/undecided |
| Choosing the relation | Grammar and label lexicon first; qFHRR only recall with symbolic label/alias gate; VM certifies | qFHRR lookup 6.8–25 µs; full path unknown | qFHRR 20 KB dense or 80 B compact per code; lexicon memory unknown | qFHRR margin 0 recall@5: SimpleQ 0.887, QALD EN 0.755, QALD FR 0.536, Mintaka EN 0.554, Mintaka FR 0.397; frozen lexicon: 0.563 / 0.360 / 0.217 / 0.331 / 0.190 (4.3) | Wrong answer, especially near-property handover when gold missing (36 of 38 missing rows at margin 0) |
| Reading French | Exact index for French names; B1q for namesakes; French relation lexicon or multilingual table; qFHRR only if rebuilt multilingual | qFHRR lookup 6.8–25 µs; component latency otherwise unknown | Exact/lexicon memory unknown; qFHRR as above | Exact-name coverage FR 99.6% (4.1); qFHRR QALD FR recall@5 0.536 own-properties / 0.361 all; lexicon 0.217; 450M adapter 0.241 (4.3) | Wrong answer or refusal; qFHRR French matches only as written (4.2) |
| Routing to a world | qFHRR table; no neural on MoWM routing path | 6.8–25 µs per short text (3) | 20 KB dense or 80 B compact per document/code (3) | 15-domain: 0.341 vs teacher 0.358, null 0.063; 42 domains at 97% of teacher quality; open-set precision 92.4%, 86.9% unseen domains sent to spawn (3) | Wrong answer: wrong world |
| Retrieval over a known corpus | qFHRR compact table | 6.8–25 µs per short text lookup; full 1M scan not stated | 80 B compact per document (3) | 1M documents: hit@1 0.71, hit@20 0.92 (3) | Wrong answer: missed or wrong hit |
| The n-gram memory | qFHRR word codes plus Hamming scan over 10,240 bits | Unknown | Unknown | WordNet-refined paraphrase teacher: trigram with one paraphrased word reaches its row 74.3%; different fact 4.0%; 35 hand pairs; not confirmed on fresh list (3) | Wrong answer: wrong row |
| Reading a fact the host placed in the context | Loci tokens on the 450M; host writes facts, trunk reads them, VM certifies | Unknown; being ported to grilly2 (6) | Unknown | Run 5 exact 0.946; run 6 exact through place 0.804 vs ordinary 0.019; text mix held-out loss 2.2398 vs 2.2491; toy word-keyed addresses novel subjects 0.854–0.899, hashed 0.000 (6) | Wrong answer and refusal; misses on unseen relations, multiple values, copy gate shut (6) |

## 4. Growing the model

A larger model is a plausible lever mainly for **choosing the relation in English**, and only as a broad property-name recaller, because the 450M adapter names a property on 0.90–0.98 of questions and achieves recall@5 0.800 on SimpleQuestions, though with a wrong rate above the lexicon and 16 missing-fact handovers (4.3). It is not a demonstrated lever for French relation choice: the adapter gets 0.241 on QALD FR, while qFHRR gets 0.536 own-properties and the lexicon 0.217; the 450M programs seat gets 0.115 (4.3). It is also not a lever for the symbolic precision the hard rules require.

A larger model is **not** the first lever for finding names: the exact name index already gives 87.4% EN and 99.6% FR exact-name coverage (4.1). It is not the first lever for namesakes: qFHRR and BM25 were killed as choosers, and B1q is symbolic (4.2). It is not a lever for routing: hard rule forbids a neural model on MoWM’s routing path. It is not needed for known-corpus retrieval or the n-gram memory, where the static qFHRR table is cheap and measured (3). For reading a host-placed fact, the loci copy path trains 1,027 parameters in about 10 minutes and already reads at 0.946/0.804 exact (6), so model size is not the obvious first lever.

The brief states no target size. The current Cubby is 576M weights total and 445M read per token (5). A larger model would have to show, before adoption:
- on relation choice: SimpleQuestions recall@5 ≥ 0.85 and QALD FR ≥ 0.80, with pooled wrong rate no higher than the lexicon’s 14.5% and missing-fact handovers ≤ 6 (the lexicon’s level), or at minimum pass the 4.3 pre-reg rule of at least 3 pooled recall points at no higher wrong rate;
- on namesakes: fame bare picks 0, qualified wrong 0, resolved ≥ 0.70, ShadowLink gap ≤ 10, Mintaka/WebQSP_EL ≤ 1 wrong in 50 rows;
- on French relation reading: QALD FR recall@5 ≥ 0.80, not the current adapter 0.241 or programs-seat 0.115.

Cheaper levers to try first: exact index and typo index for names; B1q lexicon and rule repairs for namesakes; symbolic label/alias gate and French relation labels for relations; and the loci copy path for host-placed facts. Growing the model comes only after those fail.

## 5. Loci tokens

**Entity names.** Loci tokens help only as an exact identity/recall mechanism: every name has a key because the key comes from trunk subword embeddings, and a name’s pieces can be bound into one unit; but the key carries identity, not meaning, and two items with the same name get the same key (6). For the open-web entity-name weakness, loci tokens do not fix the semantic tail-vocabulary gap in qFHRR. They leave qFHRR’s entity-search weakness as it is. Division: the exact index finds and disambiguates names; loci tokens read a fact once the item is in context.

**Choosing between namesakes.** Loci tokens leave the chosen B1q mechanism as the chooser, and would be harmful if used as a namesake discriminator because same-name items get the same key and the relation key is symmetric (6). They do not produce question-specific evidence that holds for one candidate. Division: B1q picks on stored statements; loci may read candidate facts the host has placed in context after B1q has constrained the candidate.

**Relations.** Loci tokens do not fix the relation-choice weakness; they may help read a fact after the relation is chosen. The relation key comes from its words, so a different wording is a different address; the question loop misses when the relation was never addressed in the training shard, when several values are stored, or when the copy gate stays shut (6). For choosing among an item’s own properties, loci tokens leave qFHRR/grammar/lexicon as the appropriate proposer. Division: qFHRR/grammar/lexicon choose or propose the relation; the host writes the selected candidate facts into context; loci tokens read the fact; the VM certifies.

Overall qFHRR and loci tokens should not compete. qFHRR is a static word-level retrieval/routing/proposal mechanism over stored codes; loci tokens are an exact readback mechanism for host-written facts in context. A clean combination is: qFHRR or the exact index retrieves candidates; the host writes only store facts into the scene; loci tokens read the placed fact; the VM certifies.

## 6. What I would not do, and open questions

**I would not do:**
- Use qFHRR as the chooser for namesakes or as the final relation chooser; at most it is a recall/proposer behind a symbolic gate.
- Use any similarity score or margin to produce a spoken pick for a bare name; bare name lines allow zero picks, regardless of margin.
- Let qFHRR hand over a near property when the gold property is absent; the system should refuse rather than let the VM certify a wrong answer built from the wrong on-record property.
- Put a neural model on MoWM’s routing path.
- Let a model, table, or embedding write facts or training records; only statements in the store are facts.
- Treat the English qFHRR table as a French meaning reader.
- Set a threshold and then transfer it across encoders, tables, or corpora; any threshold is tied to a stated dev set and date.
- Grow the model before trying exact index, B1q repairs, French lexicons, or loci copy-path training.

**Open questions:**
- What is the exact name index’s serving latency and memory? Not stated in the brief.
- Why do B1q’s Mintaka and WebQSP_EL wrong rates remain 2.3% and 2.2%, missing the 2% line? The brief does not break down the remaining errors.
- What would a multilingual qFHRR cost in teacher GPU hours and memory, and would it preserve the measured English routing advantage? Not stated.
- Does adding all exact-index names to qFHRR improve entity recall@100 without exceeding the p95 50 ms lookup-and-check cost? Unknown.
- Will the loci word-keyed read hold at serving on grilly2 at the 450M scale? Section 6 says the run-6 read at home is next, so not yet confirmed.
- How should one-address loci handle multiple values for one relation? Section 6 says one address holds one value and lists this as a miss.
- For a larger model, what data shard would cover QALD FR relation wording without raising the wrong rate above the lexicon? Not specified.
- Is the n-gram memory result stable on a fresh list? Section 3 says it is not yet confirmed.

## qwen/qwen3.8-max-prime

*27763 tokens out, 301.2 s, finish `stop`*

## 1. Diagnosis

### Entity names

The numbers point first to **the vocabulary built by document frequency**. Section 4.1 says the compact shortlist’s recall@100 is about 0.14 and caps the BEIR dbpedia-entity pipeline. The preprint’s own reading is that entity names are tail vocabulary that a document-frequency word list misses (section 4.1).

The **hash fallback for unseen words** is a second cause. An unseen word gets identity, not meaning (section 3). Entity names are often rare or unseen, so they match only as written.

The **bag-of-words encoding** matters for multiword names because order and binding are lost, and the preprint says n-gram and binding variants cost more than they bought (section 3). But the exact name index already finds many names without qFHRR, so bag-of-words is not the main cause for finding names.

The **English teacher** is not the main cause here: the exact name index holds the gold under its exact name for 99.6% of French mentions (section 4.1).

The **use as a ranker** is secondary. qFHRR can shortlist, but the open-web entity task is a large-corpus ranking task, and qFHRR’s compact shortlist recall is the bottleneck (section 4.1).

**Structural to a static word table:** tail coverage from document-frequency vocabulary; hash fallback giving identity only; bag-of-words loss of name structure.  
**Fixable inside or beside it:** adding entity surface forms to the vocabulary; using the exact symbolic name index instead of qFHRR for finding; using exact-name overlap as a symbolic filter.

### Choosing between namesakes

The numbers point strongly to **the use of qFHRR as a ranker/chooser**. qFHRR was killed on the bare-name line at every margin; even at margin 0.9, the 11 picks came from relation words such as “pays”, “trouve”, “né”, “born”, “country” (section 4.2). A score that always picks something is unsafe for bare names.

The **bag-of-words encoding** is also causal: question cue words and relation words are not separated by role. The relation words themselves became evidence (section 4.2).

The **framing of the task** is causal: namesake choice is not “which item prototype is most similar?” It is “which candidate has a stored fact matching the cue?” B1q’s symbolic check performs much better than qFHRR ranking (section 4.2).

The **English teacher / hash fallback** matters for French cues. Most French words fall back to hash codes and match only as written (section 4.2). The “né” failure shows a lexicon gap (section 4.2).

**Structural:** a static word table cannot enforce refusal, cannot represent roles reliably, and cannot check stored evidence by itself.  
**Fixable inside the system, but not by qFHRR alone:** use B1q-style span roles, exact name lookup, symbolic statement checks, and VM certification. qFHRR may propose but must not pick.

### Relations

The numbers point first to **task framing / candidate set**. Among every property, qFHRR fails: recall@5 0.417 on SimpleQuestions, 0.361 on QALD FR (section 4.3). Among the item’s own properties, it passes SimpleQuestions EN at margin 0: recall@5 0.887 (section 4.3). So the usable task is “choose among the item’s stored properties”, not “choose among all properties”.

The **English teacher / hash fallback** is a major cause for French. qFHRR fails QALD FR: recall@5 0.536, wrong rate 45.8% (section 4.3). The table’s teacher is English, so French words often get hash identity (section 4.2).

The **bag-of-words encoding** matters because relation wording, direction, and asked kind are not fully captured. The asked-kind filter helps “place” and “time” but drops incoming “who” relations and “how many” cases (section 4.3).

The **use as a ranker** is dangerous because of missing facts. On 38 rows where the item does not carry the gold property, qFHRR at margin 0 hands over a near property on 36 rows; the VM can check the near property but cannot catch that it is the wrong answer (section 4.3).

**Structural:** a static word table cannot know that a fact is missing, cannot certify, and cannot by itself handle French semantics if its teacher is English.  
**Fixable inside the system:** restrict candidates to the item’s own properties; use grammar/lexicon; add French property labels; use refusal gates; let qFHRR propose only under symbolic control.

---

## 2. Fixes

All pass and kill lines below are written now. Dev sets set thresholds; thresholds do not transfer (section 2).

### Entity names

Ranked by expected gain per cost:

1. **Use the exact symbolic name index for finding names; do not use qFHRR as the finder.**
   - **What changes:** Entity mentions go to the exact name index and typo index. qFHRR may still route or retrieve, but it does not find entity names. Ambiguous bare names produce a set, not a spoken pick.
   - **Harness/set:** The Mintaka EN/FR mention coverage in section 4.1. For BEIR dbpedia-entity, this needs a new harness mapping the exact index to the BEIR documents.
   - **Pass line:** Keep at least 87.4% EN and 99.6% FR gold-under-exact-name on Mintaka (section 4.1). Zero spoken picks on bare ambiguous names.
   - **Kill line:** Below 87.4% EN or 99.6% FR; any underspecified bare spoken pick.
   - **Cost:** No GPU. Memory: I don’t know; the index has 12.4M names over 4.8M items (section 4.1). Latency: I don’t know; must meet p95 50 ms (section 2).
   - **Hard rules near:** refusal better than wrong answer; only VM certifies; p95 50 ms.

2. **Expand qFHRR’s vocabulary with selected entity surface forms.**
   - **What changes:** Add high-value labels, aliases, redirects, or domain entity names to the static vocabulary, with IDF weights. Unseen tail names still fall back to hash unless added.
   - **Harness/set:** BEIR dbpedia-entity compact shortlist from section 4.1; also the routing sets from section 3 to check no regression.
   - **Pass line:** Compact recall@100 on dbpedia-entity dev ≥ 0.20, against current about 0.14 (section 4.1). nDCG@10 ≥ 0.06, against v4 compact 0.046 (section 4.1). 42-domain routing stays within 1 point of the current 97% of teacher quality (section 3).
   - **Kill line:** Recall@100 < 0.16; nDCG@10 < 0.05; routing drops more than 2 points.
   - **Cost:** Offline teacher embeddings for each added word; CPU/GPU hours not stated, I don’t know. Memory at least 80 bytes per added compact code (estimate, from section 3), plus overhead unknown. Serving latency should remain near 6.8–25 µs for short text (section 3), but with a larger table I don’t know; must stay under p95 50 ms.
   - **Hard rules near:** scores are not thresholds; costs that stand.

3. **Add exact-name/alias overlap as a symbolic re-rank or filter on qFHRR shortlists.**
   - **What changes:** qFHRR makes a shortlist; a symbolic index checks whether the query’s exact name or alias appears in the candidate document or item. No neural model.
   - **Harness/set:** BEIR dbpedia-entity; needs document name/alias fields or a new index.
   - **Pass line:** nDCG@10 ≥ 0.10 or compact recall@100 ≥ 0.25 on dev. Current v4 compact nDCG@10 is 0.046 and recall about 0.14 (section 4.1).
   - **Kill line:** nDCG@10 < 0.06; recall@100 < 0.18.
   - **Cost:** CPU only. Memory: I don’t know; needs postings for names/aliases. Latency: I don’t know; must meet p95 50 ms.
   - **Hard rules near:** costs that stand; scores are not thresholds.

### Choosing between namesakes

Ranked by expected gain per cost:

1. **Keep qFHRR out of the chooser seat; use B1q symbolic evidence as the only picker, with VM certification.**
   - **What changes:** Each question word gets one role: name, relation, or cue. Relation words and function words are never cues. Cues are looked up as names; a candidate is kept only if a stored statement matches the cue. If one candidate remains, send it to the VM; otherwise refuse.
   - **Harness/set:** Fame dev set, ShadowLink, Mintaka, WebQSP_EL from section 4.2.
   - **Pass line:** Bare single picks = 0. Qualified wrong = 0. Qualified resolved ≥ 0.70 (section 4.2). Mintaka and WebQSP_EL wrong ≤ 1 per 50 rows, per the stated allowance (section 4.2).
   - **Kill line:** Any bare spoken pick; qualified wrong > 0 on dev; resolved < 0.70; wrong > 1 per 50 on Mintaka/WebQSP_EL.
   - **Cost:** B1q p50/p95: 1.4/6.2 ms at 2 candidates; 10.2/40.6 ms at 6 or more (section 4.2). Memory: I don’t know. No GPU.
   - **Hard rules near:** refusal better than wrong answer; namesakes never ranked by popularity; only VM certifies; p95 50 ms.

2. **Repair the relation/cue lexicon and role tags, especially French.**
   - **What changes:** Add missing relation forms such as “né”; mark past participles and function words correctly; ensure relation lexicon words are never cues; add French cue forms where needed.
   - **Harness/set:** Fame EN/FR dev; the “né” failure case; Mintaka FR if separated, otherwise a new FR split.
   - **Pass line:** Zero picks caused by relation or function words. French qualified resolved ≥ 0.70 with zero wrong on dev.
   - **Kill line:** Any pick caused by a relation/function word; any French qualified wrong on dev.
   - **Cost:** Human curation; CPU/GPU hours not stated, I don’t know. Serving latency should remain in the B1q range (section 4.2). Memory: I don’t know.
   - **Hard rules near:** refusal better than wrong answer; scores are not thresholds.

3. **Add VM-certified typed qualifier expansion for place-like cues.**
   - **What changes:** For place discriminators, allow a stored one-hop “located in” expansion if the VM rules certify it. This addresses the Rosebank one-level-below errors (section 4.2).
   - **Harness/set:** Fame dev, especially B1q’s 3 wrong picks (section 4.2).
   - **Pass line:** The 3 known one-level errors are fixed; no new wrong picks; overall qualified wrong remains 0 on dev; p95 ≤ 50 ms.
   - **Kill line:** Any new wrong pick; p95 > 50 ms.
   - **Cost:** Extra symbolic graph lookups. Latency: I don’t know, but current B1q p95 is 40.6 ms at 6+ candidates (section 4.2), so there is some budget. Memory: I don’t know.
   - **Hard rules near:** only VM certifies; refusal better than wrong answer; costs that stand.

### Relations

Ranked by expected gain per cost:

1. **Restrict relation choice to the item’s own properties and add a missing-fact refusal gate.**
   - **What changes:** Candidates are only outgoing and incoming properties of the resolved item. qFHRR or a lexicon may propose, but if the evidence is weak, if multiple near properties remain, or if the question’s asked kind is not satisfied, the system refuses instead of handing over a near property.
   - **Harness/set:** SimpleQuestions-Wikidata, QALD-9-plus EN/FR, Mintaka EN/FR, and the 38 missing-fact rows from section 4.3.
   - **Pass line:** SimpleQuestions recall@5 ≥ 0.85, near current 0.887 (section 4.3). Missing-fact handovers ≤ 6, matching the lexicon’s 6, against qFHRR margin 0’s 36 (section 4.3). Wrong rate ≤ 9.0% on SimpleQuestions, matching qFHRR margin 0 (section 4.3).
   - **Kill line:** SimpleQuestions recall@5 < 0.80; missing-fact handovers > 16, matching qFHRR margin 0.25 (section 4.3); wrong rate > 11.3%, the pooled qFHRR margin 0.25 wrong rate (section 4.3).
   - **Cost:** qFHRR lookup 6.8–25 µs (section 3); property lookup latency not stated, I don’t know; total must meet p95 50 ms. Memory: I don’t know.
   - **Hard rules near:** refusal better than wrong answer; scores are not thresholds; only VM certifies.

2. **Make the frozen label lexicon/grammar primary; use qFHRR only as a controlled fallback.**
   - **What changes:** The lexicon decides when it has a clean label match. qFHRR is used only among the item’s own properties, only with a dev-set margin, and only when the result does not raise the wrong rate.
   - **Harness/set:** The section 4.3 relation sets.
   - **Pass line:** SimpleQuestions recall@5 ≥ 0.85. Pooled wrong ≤ 11.3%, the qFHRR margin 0.25 pooled wrong rate (section 4.3). QALD FR wrong ≤ 15.1%, the frozen lexicon’s QALD FR wrong rate (section 4.3).
   - **Kill line:** SimpleQuestions recall@5 < 0.80; wrong above the lexicon on any set; QALD FR recall@5 < 0.50, below current qFHRR’s 0.536 (section 4.3).
   - **Cost:** CPU lexicon lookup plus qFHRR 6.8–25 µs (section 3). Memory: I don’t know. Latency must meet p95 50 ms.
   - **Hard rules near:** scores are not thresholds; refusal better than wrong answer.

3. **Add French property labels and aliases, either as a frozen lexicon or as French qFHRR vocabulary.**
   - **What changes:** Add French property labels, aliases, and common question phrasings. If qFHRR is changed, the new table must be treated as a new encoder with fresh thresholds.
   - **Harness/set:** QALD-9-plus FR and Mintaka FR from section 4.3.
   - **Pass line:** QALD FR recall@5 ≥ 0.70, against current qFHRR 0.536 (section 4.3). Wrong rate ≤ 15.1%, matching the frozen lexicon’s QALD FR wrong rate (section 4.3). Mintaka FR recall@5 ≥ 0.50, against current qFHRR 0.397 (section 4.3).
   - **Kill line:** QALD FR recall@5 < 0.60; wrong > 20%; Mintaka FR recall@5 < 0.45.
   - **Cost:** Offline curation or offline teacher embedding. CPU/GPU hours: I don’t know. Serving latency should remain qFHRR/lexicon range, but I don’t know for a larger table; must meet p95 50 ms. Memory: I don’t know.
   - **Hard rules near:** no neural model on routing path; scores are not thresholds; costs that stand.

---

## 3. The trade-offs

| Job | Which part should do it | Latency, memory, accuracy | Failure mode |
|---|---|---|---|
| Finding a name | Exact symbolic index/typo index first. qFHRR should not find names. | Latency: I don’t know. Memory: I don’t know; index has 12.4M names for 4.8M items (section 4.1). Accuracy: gold under exact name 87.4% EN, 99.6% FR (section 4.1). | Refusal if no match; wrong if typo resolves wrongly; slowness unknown. |
| Choosing between namesakes | Exact symbolic check/B1q, then VM. qFHRR may only filter, never pick. Order: exact candidates → B1q evidence check → VM certify → refuse if unresolved. | Latency: B1q p50/p95 1.4/6.2 ms at 2 candidates; 10.2/40.6 ms at 6+ (section 4.2). Memory: I don’t know. Accuracy: fame B1q 692 right, 3 wrong, resolved 0.764, 2 bare picks; Mintaka/WebQSP wrong about 2.3%/2.2% (section 4.2). | Wrong answer if lexicon/hierarchy error; refusal if undecided. |
| Choosing the relation | Combination: resolve item → restrict to item’s own properties → grammar/lexicon first → qFHRR fallback under dev margin → missing-fact/kind gate → VM certify. | Latency: qFHRR 6.8–25 µs (section 3); property/lexicon lookup unknown; must meet p95 50 ms. Memory: I don’t know. Accuracy: qFHRR own-property SimpleQ recall@5 0.887, wrong 9.0%; QALD FR 0.536, wrong 45.8%; lexicon SimpleQ 0.563, wrong 4.0% (section 4.3). | Wrong near property when fact missing; refusal if gate refuses. |
| Reading French | Exact index for names; French lexicon/table for relations. English qFHRR should not be the semantic reader. | Latency: I don’t know. Memory: I don’t know. Accuracy: French name finding 99.6% under exact name (section 4.1); French relation choice fails with current qFHRR: QALD FR recall@5 0.536, wrong 45.8% (section 4.3). | Wrong answer from hash identity; refusal if lexicon lacks French form. |
| Routing to a world | qFHRR table with margin gate. No neural model on routing path. | Latency: 6.8–25 µs (section 3). Memory: I don’t know. Accuracy: 0.95 of teacher on 15 domains; 97% of teacher on 42 domains; open-set precision 92.4%, spawn 86.9% (section 3). | Wrong route if margin/gate is misset; slowness unlikely. |
| Retrieval over a known corpus | qFHRR compact index. | Latency: 6.8–25 µs query encoding (section 3). Memory: 80 bytes per document, so 80 MB for 1M documents for codes alone; overhead unknown (section 3). Accuracy: hit@1 0.71, hit@20 0.92 on 1M documents (section 3). | Wrong or missed documents; weak on open-web entity retrieval. |
| The n-gram memory | qFHRR word-code Hamming scan. | Latency: I don’t know. Memory: I don’t know. Accuracy: paraphrased trigram reaches row 74.3%, different fact 4.0%, but only 35 hand pairs and not confirmed on a fresh list (section 3). | Wrong fact; failure to recall; needs fresh validation. |
| Reading a fact the host placed in the context | Loci tokens, with VM certification only if the fact is also a stored statement. Order: host writes fact → loci read → answer candidate → VM checks store if certification is required. | Latency: serving is being ported; I don’t know (section 6). Memory: copy path has 1,027 trained parameters (section 6); trunk is 576M weights (section 5). Accuracy: 0.946 exact for facts seen once >800 tokens back; run 6 exact 0.804 loci vs 0.019 ordinary (section 6). | Wrong answer if copy/switch errs: about 12% wrong on present facts in the CPU toy (section 6); refusal if it says it does not know; slowness unknown. |

---

## 4. Growing the model

A larger model is a possible lever for:

- **French relation understanding and relation paraphrase.** The current English qFHRR fails QALD FR (section 4.3). A larger multilingual model could propose better relation candidates, but only as a proposer, not a certifier.
- **Extracting cue roles for namesakes**, if its output is converted into symbolic checks. It must not rank namesakes by popularity or similarity.
- **Reading difficult context facts**, if loci scaling improves. But the current loci result is already strong with only 1,027 trained parameters (section 6), so size is not the first lever.

A larger model is **not** the main lever for:

- **Finding a name.** The exact index already gives 87.4% EN and 99.6% FR coverage (section 4.1).
- **Routing.** qFHRR is strong, and no neural model is allowed on the routing path (sections 2, 3).
- **Final namesake choice.** The rule requires evidence in the question that holds for one candidate, not popularity or similarity (section 2).
- **Known-corpus retrieval.** qFHRR already gives hit@1 0.71 and hit@20 0.92 on 1M documents (section 3).
- **The n-gram memory.** The issue is addressing and recall, not raw model size (section 3, section 6).

**Size:** The brief does not state the right size; I don’t know. The current Cubby is about 576M weights, 445M read per token (section 5). If growing, I would use the smallest model that passes the pre-registered gates. If a number must be named, a 1B-to-3B class model is a plausible next step, but that is an estimate, not a fact from the brief.

**What it would have to show:**

- Relation gates: SimpleQuestions-Wikidata recall@5 ≥ 0.85 and QALD-9-plus FR ≥ 0.80 (section 4.3).
- Wrong rate no higher than the frozen lexicon, and preferably lower. Current lexicon pooled wrong is 14.5%; qFHRR margin 0.25 pooled wrong is 11.3%; the 450M adapter was 28.6% pooled and not kept (section 4.3).
- Missing-fact handovers no worse than the best symbolic baseline; grammar had 0 on the 38 missing rows (section 4.3).
- No violation of the no-neural-routing rule (section 2).
- Latency compatible with the p95 50 ms lookup-and-check budget, or clearly moved out of that path (section 2). The programs seat currently takes about 2.4 s per program (section 5), so it cannot sit inside the 50 ms path.

**Cheaper lever to try first:** Fix B1q lexicon, add French property labels, enforce missing-fact refusal, and use the exact name index. These are cheaper and directly address the measured failure modes (sections 4.1–4.3).

---

## 5. Loci tokens

### Entity names

Loci tokens **leave the finding problem as it is**, and may hurt if used as the finder. A key comes from subword embeddings and carries identity, not meaning; two items with the same name get the same key (section 6). That does not solve namesakes.

**Division of work:** Exact symbolic index finds the name. Loci tokens may later read a host-written fact about the resolved entity. qFHRR may route or retrieve, but should not resolve names by itself.

### Choosing between namesakes

Loci tokens **hurt or leave it as it is** if used as the chooser. Same-name candidates get the same key, and relation keys are wording-specific and symmetric (section 6). That is not enough to choose between namesakes.

**Division of work:** B1q symbolic check chooses by stored evidence. Loci tokens may surface a discriminator fact in context, but the pick must come from symbolic evidence and VM certification. qFHRR should not rank; loci should not pick.

### Relations

Loci tokens **help for reading a relation fact once it has been retrieved or placed in context**, but they do not by themselves solve relation choice. The loci read is strong: 0.946 exact for facts seen once more than 800 tokens back, and 0.804 exact against 0.019 ordinary reading in run 6 (section 6). But a relation key depends on its words, so a different wording is a different address, and unseen relations can miss (section 6).

**Division of work:** Grammar/lexicon and qFHRR propose relation candidates among the item’s own properties. The store retrieves the fact. Loci tokens can read the retrieved/host-written fact in context. The VM certifies the final hop. If the store lacks the fact, refuse.

---

## 6. What I would not do, and open questions

### What I would not do

- I would not use qFHRR as a namesake chooser. It was killed on the bare line at every margin (section 4.2).
- I would not rank namesakes by popularity, sitelinks, degree, or page views (section 2).
- I would not let qFHRR, the 450M, or loci output count as a fact. Only stored statements are facts (section 2).
- I would not treat VM certification as real-world correctness. It means the answer follows from stored facts and rules, not that it is correct (section 2).
- I would not use model agreement as certification (section 2).
- I would not reuse thresholds across encoders, tables, or corpora (sections 2, 3).
- I would not put a neural model on the routing path (section 2).
- I would not grow the model before trying exact-index repair, B1q lexicon repair, French property labels, and missing-fact refusal gates (sections 4.1–4.3).
- I would not use qFHRR for open-web entity retrieval without a symbolic name index or exact-name filter; the compact shortlist recall is too low (section 4.1).
- I would not let relation choice hand over a near property when the item lacks the asked fact; the missing-fact rows show this can produce certified wrong answers (section 4.3).

### Open questions

- What is the exact memory and latency of the 12.4M-name exact index under production load? I don’t know.
- How much entity vocabulary can be added to qFHRR before memory or cache latency becomes a problem? I don’t know.
- Which entity surface forms should be added: all labels, only frequent labels, or only domain-specific labels? Not stated.
- Can the missing-fact refusal gate be made reliable without access to the gold property? The 38-row set shows the danger, but the production rule is not stated.
- Can French relation labels be covered well enough by a frozen lexicon, or is a French table/teacher required? Not stated.
- Will loci tokens work on the 450M in grilly2 with the same accuracy as the current runs? Serving is still being ported (section 6).
- Can loci handle multiple values at one address, or does the store need to present one value per address? The brief says it misses when the store holds several values (section 6).
- What grown-model size, if any, would pass the French relation gates at acceptable cost? Not stated.
- Can the n-gram memory result be reproduced on a fresh list? The current result is only 35 hand pairs (section 3).

## z-ai/glm-5.3-prime

*12895 tokens out, 84.0 s, finish `stop`*

# qFHRR: where it is weak, and what to do about it

## 1. Diagnosis

**Entity names (section 4.1).** The numbers point primarily at the **vocabulary built by document frequency**, with the **hash fallback** as its direct consequence. The evidence: the preprint's own reading is that entity names are the tail vocabulary a document-frequency word list misses; the compact shortlist's recall@100 of about 0.14 caps the pipeline, so the name often is not in the shortlist at all; and the dense re-rank moves nDCG@10 only from 0.081 to 0.096 (v1) and 0.046 to 0.096 (v4) — re-ranking cannot recover what recall lost. v4 gained 18% and kept a 2.3-times gap to dense potion, which is the signature of a vocabulary ceiling, not an encoding ceiling: a better teacher over the same head-of-distribution word list does not close it. BM25 at about 0.313 beats every qFHRR form because a name is an exact string, and a bag of IDF-weighted word codes is the wrong representation for it. Both causes are **structural to a static word table of fixed, frequency-ranked size**: the table holds the head; names are the tail. Partially fixable inside it — the build already admits "domain words," so the tail can be added at build time at a memory cost. The hash fallback is by design (identity, not meaning) and cannot give meaning to a word the teacher never embedded.

**Choosing between namesakes (section 4.2).** Three causes, in order of severity. First, the **ranker framing**: a cosine score always picks something. Margins only shrink the damage — 298 bare single picks at 0.25 down to 11 at 0.9, never zero — and the bare line allows zero, so this is fatal by rule, not by tuning. This is structural to any scorer used as a picker. Second, the **bag-of-words encoding**: at margin 0.9 the 11 surviving picks came from the question template's relation words ("pays," "trouve," "né," "born," "country") — the summed code cannot tell a cue from the question's own relation wording, because order and role are lost. Structural to the sum. Third, the **English teacher**: French cue words fall back to hash and match only as written. This one is fixable inside the table (a French build), though B1q's replacement shows the symbolic route is cheaper. Note that B1q's own residual errors are not qFHRR's: a lexicon gap ("né" not a relation word, folding to "ne" and naming Nebraska) and an ontology level (Rosebank's "located in" one level below the discriminator) — both fixable outside the table.

**Relations (section 4.3).** Four causes. First, the **English teacher**: QALD FR recall@5 is 0.536 against 0.755 for QALD EN at margin 0, against a 0.80 gate — the French gap is the teacher and the hash fallback. Fixable inside the table. Second, the **ranker**: among all 12,684 properties the wrong rate is 56.0%; among the item's own properties the same scorer reaches 0.887, where five candidates at random would score 0.149 — the candidate-set framing does the work, not the score. Third, and most dangerous, the **ranker again in interaction with the system**: on the 38 rows where the item lacks the gold property, qFHRR at margin 0 hands over a near property on 36 (16 at 0.25). The near property is on record, so the VM checks it and certifies the wrong answer it builds. Certification cannot catch this because "certified" means "follows from the facts used," never "correct." A proposer that always proposes, checked by a verifier that only checks what is on record, is a structural hole — fixable only by giving the proposer an abstain grounded in the question, or by a framing fix. Fourth, the **bag-of-words**: wording mismatch between question and property label costs recall (QALD EN 0.755 against SimpleQuestions 0.887 is consistent with this, though the preprint does not isolate it). The asked-kind filter's dropped gold on "who" asked from the value's side and "how many" over item-valued properties is a framing error in the filter, not a property of the table.

## 2. Fixes

### 2.1 Entity names

**F1. Keep the division of labor: the exact name index finds, qFHRR never does.** This is already the practice (4.1: 12.4M names, gold under the exact name for 87.4% of Mintaka EN and 99.6% of FR). What changes: nothing; it is recorded as the standing design so no one re-routes finding through the table. Harness: none needed; a held-out re-read of Mintaka mentions. Pass line, written now: gold under exact name ≥ 85% EN and ≥ 99% FR. Kill line: none — it is not an experiment. Cost: the index's memory and latency are not stated; I don't know. Rules at risk: none.

**F2. Grow the vocabulary with entity-name tokens in the build's domain-words slot.** Add name tokens from the 12.4M-name index (deduplicated, up to a stated memory budget), embedded once offline by the teacher, so names get meaning codes instead of hash identity. Harness: BEIR dbpedia-entity, compact form, read once. Pass line, written now: recall@100 ≥ 0.30 and nDCG@10 compact ≥ 0.10. Kill line: recall@100 < 0.20. Cost: one offline teacher pass over the added words (its GPU cost is not stated; I don't know), plus table memory growing with the added rows (the current table's size is not stated; I don't know). Rule it comes near: "scores are not thresholds" — margins must be re-set on dev, with data and date, for the new table. Ranked second.

**F3. Dense re-rank over a lexical shortlist.** Since the compact shortlist's recall@100 of ~0.14 caps the pipeline, re-rank a BM25-style shortlist with the dense qFHRR form (20 KB a document) instead. Harness: BEIR dbpedia-entity. Pass: nDCG@10 ≥ 0.15. Kill: < 0.10. Cost: dense store for 4.64M documents is roughly 93 GB (estimate, from 20 KB × 4.64M); re-rank latency is not measured — unknown. Rules: none broken; the dense form is the table's own vector, not a neural model at serve. Ranked third: large memory for a bounded gain, and it does not fix the recall cap by itself.

Ranked by gain per cost: F1 (free, already won), F2 (one offline pass at the measured ceiling), F3 (memory-heavy).

### 2.2 Choosing between namesakes

**F1. Extend the relation lexicon and function-word list to French.** This kills the "né"→Nebraska error class. Harness: the fame set, fresh seed, both lines. Pass, written now: bare single picks = 0; qualified wrong ≤ 3; resolved ≥ 0.70. Kill: any bare spoken pick. Cost: lexicon curation; CPU negligible; B1q latency unchanged. Rules: none at risk — B1q's undecided and refused counts stay as the preferred failure.

**F2. One-level containment closure for the qualifier check.** Compute administrative containment (one "located in" hop) from stored statements, so a candidate whose statement sits one level below the discriminator still holds it; the VM checks the closure as it checks any hop. Harness: the fame set, qualified line. Pass: the Rosebank-class wrong → 0; qualified wrong ≤ 2; resolved not below 0.70. Kill: wrong > 3. Cost: extra hops at check time. Rule it comes near: the p95 50 ms line — B1q is already at p95 40.6 ms with 6 or more candidates, and a fifth to a third of mentions name more than 20 items, where the cost is not measured. Measure at 20+ candidates before adopting.

**F3. qFHRR as a filter only, on B1q's undecided set.** A pre-registered high margin converts some undecided questions to picks. Pass, written now: converts ≥ 10% of undecided to correct picks with 0 wrong and 0 bare picks. Kill: any bare pick or any wrong. Cost: negligible (6.8–25 µs lookups). Rules it comes near: "agreement may only filter, never certify," and the bare-name rule. I expect this to be killed: at margin 0.9 among relation-carrying candidates qFHRR still made 21 bare picks (4.2). The line is written anyway; that is the point of pre-registration.

Ranked: F1, F2, F3.

### 2.3 Relations

**F1. A French word table.** A second static table: French vocabulary, a French-capable teacher (which teacher is an open question — none is named in the material), margins set on French dev with data and date. Harness: the section 4.3 tables; the QALD FR gate of 0.80. Pass, written now: recall@5 ≥ 0.80 on QALD FR at a wrong rate ≤ 15.1% (the lexicon's QALD FR wrong rate). Kill: below 0.80, or wrong rate above the lexicon's. Cost: one offline embedding pass (cost not stated); memory for a second table (size not stated). Rule it comes near: thresholds do not transfer — fresh margins, fresh dev, dated.

**F2. Direction-aware asked-kind filter.** Keep the kind filter's gains for "place" (recall 0.77, 6% wrong) and "time" (0.63, no wrong) but make it direction-aware: for "who" asked from the value's side, keep incoming properties; for "how many," keep item-valued properties. This is a framing fix, symbolic, host-side. Harness: the section 4.3 tables plus the 38 missing-fact rows as a guard set. Pass, written now: missing-fact handovers ≤ 16 (the margin-0.25 level) with recall ≥ 0.887 on SimpleQuestions and ≥ 0.755 on QALD EN. Kill: handovers > 36, or recall dropping more than 3 points on either set. Cost: CPU only. Rules: none.

**F3. A missing-fact guard set and an abstain for proposers.** Build a held-out version of the 38-row set (rows where the item lacks the gold property today), and require any relation proposer to be able to return "no property" — a refusal — rather than always ranking. qFHRR's margin 0.25 already cuts handovers from 36 to 16 at a recall cost; the guard set measures the trade on data that was not used to tune it. Pass, written now: handovers ≤ 16 of 38-equivalent on the held-out set. Kill: ≥ 36. Cost: curation; CPU. Rules: this serves "a refusal is better than a wrong answer." Note that qFHRR was not kept by the rule written before its run (it needed +3 points pooled at no higher wrong rate than the lexicon and delivered +2.7); any re-run must clear the original +3 line, not a relaxed one.

Ranked: F1 (the largest measured gap: 0.536 against a 0.80 gate), F2, F3.

## 3. The trade-offs

| Job | Who does it | Latency | Memory | Accuracy | Failure mode |
|---|---|---|---|---|---|
| Finding a name | Exact name index + typo index; qFHRR: never | Not stated — unknown | 12.4M names; size not stated — unknown | Gold under exact name 87.4% EN, 99.6% FR (4.1) | A miss (typo index catches exact misses); otherwise a set, not a wrong answer |
| Choosing between namesakes | B1q (span rule + symbolic check), host-side; qFHRR at most as a filter | p50/p95 1.4/6.2 ms at 2 candidates; 10.2/40.6 ms at 6+ (4.2); at 20+ unknown | The layered store | 2 bare picks, 3 wrong, resolved 0.764 (4.2) | Refusal (undecided 61, refused 154) — the preferred mode |
| Choosing the relation | Grammar + lexicon today (4.3, 5); qFHRR as proposer only where a pre-registered line passes (SimpleQuestions 0.887) | Lexicon: CPU, not stated; qFHRR: 6.8–25 µs a short text (3) | Store + lexicon | SimpleQ 0.887 / QALD EN 0.755 / QALD FR 0.536 at margin 0 (4.3); FR below the 0.80 gate | Wrong answer via near-property handover (36 of 38) — the danger |
| Reading French | Exact index for names (99.6%, 4.1); B1q for namesakes (EN and FR, 4.2); relations: neither the lexicon (0.217) nor qFHRR (0.536) passes (4.3) | As above | As above | As above | Names: a set; relations: wrong or refusal depending on stage |
| Routing to a world | qFHRR word table only (hard rule) | 6.8–25 µs on one CPU core (3) | Table size not stated — unknown | 0.95 of teacher on 15 domains; 42 domains at 97%; open-set 92.4% precision, 86.9% of unseen to spawn (3) | A wrong route, or a spawn (a refusal to route) |
| Retrieval over a known corpus | qFHRR compact form | Not stated — unknown | 80 bytes a document; 1M docs ≈ 80 MB (estimate) | hit@1 0.71, hit@20 0.92 on 1M docs (3); entity search 0.046 (4.1) | A miss (wrong) |
| The n-gram memory | qFHRR word codes, Hamming scan over 10,240 bits | Not stated — unknown | Not stated | 74.3% paraphrase row, 4.0% different fact, 35 hand pairs, unconfirmed (3) | A wrong row |
| Reading a fact the host placed in the context | Loci tokens on the 450M | Unknown on grilly2 (port in progress); the programs seat takes ~2.4 s a question (5), a different seat | The 450M, ~576M weights (5) | 0.946 exact, run 5; 0.804 vs 0.019 at equal tokens, run 6 (6) | Refusal (relation outside the shard, several values, gate shut); the switch answers ~12% of present facts wrong |

Order of combination, where more than one part is involved: exact index → B1q → relation proposer (grammar/lexicon, or qFHRR where its line passes) → VM certifies or refuses. Loci tokens enter only as the reader of facts the host has already written.

## 4. Growing the model

**Where a larger model is the lever: French relation naming, possibly.** The measured gap is real (QALD FR 0.536 against a 0.80 gate), but the evidence that size is the lever is weak: the 450M with an adapter trained on 163,907 property-label records, EN and FR, reached 0.800 on SimpleQuestions and only 0.241 on QALD FR, with a pooled wrong rate of 0.286 against the lexicon's 0.145 — it was not kept. So the 450M-plus-adapter already failed the French gate at a higher wrong rate. A larger model would have to show, on the section 4.3 harnesses, read once: QALD FR recall@5 ≥ 0.80 at a pooled wrong rate ≤ 0.145, and — if it ever touches the namesake path — zero bare picks on the fame set. What size, I don't know; nothing in the material sizes a multilingual model, and the constraints are a home GPU on grilly2 and rented A100s (a LoRA run over ~216k rows takes a little over an hour there). The cheaper lever to try first is the French word table (F1 above): it attacks the same measured gap for one offline pass and some memory, with no wrong-rate risk of the adapter's kind.

**Where it is not the lever:** routing (barred by rule, and the table is already at 97% of its best static teacher on 42 domains); entity finding (the exact index holds the gold for 87.4%/99.6% of mentions against qFHRR's 0.046 nDCG); namesake choosing (B1q resolves 0.764 with 3 wrong and refuses otherwise; a model used as a picker risks bare picks, and popularity ranking is barred); reading placed facts (the 450M already reads them at 0.946 exact; growth is not the bottleneck — the shard's relation coverage and the one-address-one-value limit are).

## 5. Loci tokens

**Entity names: leave it as it is.** A loci key comes from the trunk's subword embeddings, so every name has one — no vocabulary miss — but it carries identity, not meaning, so it does not give semantic search over names, which is what BEIR dbpedia-entity measures. And two items with the same name get the same key, so loci cannot choose between namesakes either. Division of work: the exact index finds, B1q chooses, and loci only reads back facts the host writes about the chosen item. Names bound into single input units (up to about 55 pieces) are a recall mechanism, not a training lever — proper names are 5.2% of the training mix's tokens.

**Namesakes: hurt, if asked to choose; keep them out.** Same name, same key. The choice must stay with the exact index and B1q; loci reads what the host placed, after the item is resolved.

**Relations: help one cause, leave another.** They help the missing-fact danger, which is the worst one: an address keyed by the question's subject and relation words holds nothing when the store lacks that relation, so the reader says it does not know — a refusal — where qFHRR's ranker hands over a near property on 36 of 38 rows and the VM then certifies the wrong answer. They do not help wording variation: a different wording of the same relation is a different address, and the reader misses relations its training shard never addressed; French is unmeasured. Combine them in this order: the grammar/lexicon (or qFHRR where its line passes) proposes the property; the host writes the candidate facts into the context; loci reads them back (0.804 exact through the place against 0.019 for ordinary reading, run 6); the VM certifies. Loci is the reader, never the chooser, and never the writer — the host writes the facts, which keeps the "facts are never model-written" rule intact.

## 6. What I would not do, and open questions

**Not do:** use any scorer — qFHRR, BM25, the 450M — as the namesake picker, at any margin; rank namesakes by popularity in any form; re-keep the 450M relation adapter (wrong rate 0.286 pooled against 0.145; not kept); put a neural model on MoWM's routing path, or serve a dense teacher there; chase order-aware encodings or binding variants for the table (the preprint states every n-gram and binding variant measured cost more than it bought); treat VM certification as correctness — the 36-of-38 near-property handovers are the standing counterexample; transfer margins between encoders, tables or corpora; grow the model for routing, finding, or namesakes; let loci produce facts from trunk weights.

**Open questions:** Which teacher for a French word table — none is named in the material. Whether the 74.3% paraphrase-row result holds on a fresh list (stated as not yet confirmed). B1q's latency at 20 or more candidates — only 2 and 6+ are measured, and a fifth to a third of mentions name more than 20 items, against the p95 50 ms line. The memory and latency of the exact name index — not stated. Whether the loci run-6 read reproduces on grilly2 — the port is in progress. Whether a direction-aware kind filter keeps the "who"-from-the-value-side gold — not measured. Whether the 38 missing-fact rows generalize — they are dev, seed 0, and a held-out guard set does not yet exist. And what "v4," the serving teacher, is relative to the stated teacher lineage — the preprint names it without tying it down.

## x-ai/grok-4.7

*32481 tokens out, 525.4 s, finish `stop`*

The numbers do not support fixing these three jobs inside the word table. They support taking name finding and namesake choice off it, and not letting it pick a relation.

## 1. Diagnosis

**Entity names.** Section 4.1 points to the vocabulary built by document frequency, not to the English teacher. The preprint’s own reading is that names are the tail a document-frequency list misses. The teacher change does not move the result enough to say otherwise: v4 gained 18% (compact nDCG@10 0.046 against v1’s 0.040; dense re-rank 0.096 against 0.081) and kept the 2.3-times gap to dense potion-8M (0.224) and a larger gap to the published BM25 of about 0.313. The compact shortlist’s recall@100 of about 0.14 caps the pipeline, so the dense re-rank cannot recover what the shortlist dropped. That cap is the compact code (section 3: argmax of each block, 80 bytes), not a serve-time threshold.

The hash fallback is not what these numbers isolate. An unseen word gets identity, not meaning (section 3), so an exact string can still match. Section 4.1 does not separate that identity match from the document-frequency miss, from bag-of-words order loss, or from block-argmax. I do not know which of those three makes the gap to BM25. Bag-of-words is a stated limit of the table (section 3); it is not a cause section 4.1 isolates.

This is structural to this static table. The served vocabulary is the top 30k to 60k words by document frequency, plus domain words (section 3). The name index holds 12.4M names of 4.8M items (section 4.1). A table of that size cannot cover that tail. Raising the teacher again is the change already measured, and it left the gap. Finding, as the system does it, is already off the table: the exact index holds the gold under its exact name for 87.4% of Mintaka EN mentions and 99.6% of FR, and returns a set. A fifth to a third of mentions name more than 20 items, so the set is not a pick.

**Choosing between namesakes.** Section 4.2 points to use as a ranker, to the framing of the task, and to bag-of-words. It does not point to the document-frequency vocabulary. Every margin fails the bare line (0 spoken picks allowed): 298 single picks at 0.25, 73 at 0.5, 21 at 0.75, 11 at 0.9, 6 on the earlier tiers. At 0.9 the 11 picks came from template relation words in the bag (“pays”, “trouve”, “né”, “born”, “country”). No margin is both 0 wrong on qualified names and at least 70% resolved: 0.25 resolves 77.4% with 29 wrong; 0.5 resolves 47.4% with 3 wrong; 0.75 is 0 wrong and 12.4% resolved; 0.9 is 0 wrong and 1.1% resolved. A score that always has a best item, kept or cut by a margin, cannot meet both lines. That is structural to using the table as a chooser. It is not fixable inside the table: the measured Pareto frontier misses the lines written for this task.

The English teacher is a real limit of the table on French, and section 4.2 states it: most French words fall back to their hash and match only as written. It is not the cause of the remaining misses once the chooser is the symbolic check. Those misses are a lexicon hole and a one-hop fact. B1q, which does not rank, still fails the written lines: 2 bare picks, and 3 wrong on qualified names, with resolved 0.764. One bare pick is “né” not in the relation lexicon, folded to “ne”, which named Nebraska. The 3 wrong are the gold’s “located in” one level below the discriminator (a Rosebank in Cape Town, which is in the Western Cape, while another Rosebank states the Western Cape directly). On Mintaka and WebQSP_EL, B1q halves the earlier wrong rate to 2.3% and 2.2% and still misses at most 1 wrong in 50. I do not know the types of those remaining wrongs. ShadowLink’s gap is not stated for B1q. I don’t know it.

**Relations.** Section 4.3 points to the framing and to use as a ranker. Among every property, the codebook of 12,684 properties (25,368 codes) is not usable: recall@5 0.417 on SimpleQuestions at 56.0% wrong, 0.361 on QALD FR at 63.9%. Among the item’s own properties it is a different task: qFHRR at margin 0 reaches 0.887 on SimpleQuestions (9.0% wrong; 0.893 where the item holds the gold) and fails QALD FR (0.536; 0.578), against gates of 0.85 and 0.80. The pre-registered keep rule fails it: at margin 0.25 it adds 2.7 points pooled (0.401 against 0.374) at a lower wrong rate (11.3% against 14.5%), short of at least 3 points.

The ranker failure is the missing fact, and it is structural to score-and-pick. On 38 rows the item does not carry the gold today. Margin 0 hands over a near property on 36 of them; margin 0.25 on 16; the lexicon on 6; the 450M on 4; the grammar on 0. The near property is on record, so the VM checks it and cannot catch the wrong answer it builds. No better word table fixes a gold that is not in the candidate set: the table will still have a nearest label. Restricting candidates to the item’s properties is required, and it sits outside the table, in the store.

Bag-of-words is not the EN SimpleQuestions failure. At margin 0 the table beats the frozen label lexicon there (0.887 against 0.563). The English teacher is the cause section 4 points to for French: QALD FR 0.536 against QALD EN 0.755, Mintaka FR 0.397 against Mintaka EN 0.554, and section 4.2’s hash fallback is the stated mechanism. The lexicon is also weak in French (QALD FR 0.217, Mintaka FR 0.190), so French is not only the teacher. The document-frequency vocabulary is not isolated by these tables.

The asked-kind cut is not a table fix. It helps “place” (recall 0.77, 6% wrong) and “time” (0.63, no wrong) and drops the gold on “who” asked from the value’s side and on “how many” over item-valued properties. Each dropped gold becomes a missing-fact row: handovers 83 instead of 36. The 450M adapter (163,907 EN and FR label records) is worse than the lexicon on wrong rate on every set (pooled 0.286 against 0.145), with 16 missing-fact handovers, and was not kept. Its serving adapter names a property on only 0.21 to 0.53 of questions.

Structural to a static word table used as a ranker: the fill-in when the gold is absent; the global codebook; English-only semantics in this build. Fixable inside a word table, in principle: a multilingual teacher. That rebuild would not remove the missing-fact handover, and section 4 does not show it would clear 0.80 on QALD FR. I don’t know whether it would.

## 2. Fixes

A proposal that lets a score pick, transfers a margin, puts a neural model on routing, or treats a model output as a fact is out. Lines below are written now. Held-out and test sets are read once per arm. Where a section 4 set is already read, the arm needs a list not yet read. I don’t know whether those lists exist.

### Entity names

**1. Name finding stays on the exact index and the typo index. qFHRR is not on that path.** What changes: a mention is looked up as a string; the result is a set; an exact miss goes to the typo index; nothing is ranked and nothing is spoken from this stage. Harness: a fresh mention list with gold items, read once. The Mintaka rates in section 4.1 are already read, so they are not this arm. Pass: the gold is in the returned set on at least 87.4% of EN mentions and 99.6% of FR mentions, the floors already measured, and every row returns a set or a miss, never a spoken pick. Kill: any spoken pick from this stage, or either language under its floor. Cost: no GPU and no new table; the index exists (12.4M names). Lookup latency and memory of the index alone are not stated. I don’t know them. The check that already uses name lookup, B1q, is 1.4 / 6.2 ms p50 / p95 at 2 candidates and 10.2 / 40.6 ms at 6 or more (section 4.2), inside the p95 50 ms rule. Near rules: a refusal is better than a wrong answer; a bare name gets no spoken pick; facts are store statements, not table hits. It does not break them.

**2. An exact miss never falls through to a qFHRR nearest name.** What changes: the typo index is the only recovery; a hash-code neighbor cannot become a candidate. Harness: a fresh misspelled-mention list. Typo-index accuracy is not stated. I don’t know it. Pass: gold in the set on a floor written from dev before the single read; I cannot copy 87.4% onto a typo set, and no numeric typo floor is in the brief, so the numeric floor has to be set on dev with its date before the run, not taken from section 4.1. Kill: any spoken pick, or a typo hit that is a different name and is treated as exact. Cost: no GPU; added latency not stated. Near rules: same as proposal 1, plus scores are not thresholds (a cosine neighbor is not an exact miss).

I do not have a third proposal with expected gain. A denser re-rank of the compact shortlist is already measured and capped (section 4.1). Growing the 30k–60k list toward 12.4M names is not a fix inside this table.

### Choosing between namesakes

**1. Relation words are closed before folding, including French participles.** What changes: the surface form is checked against the relation lexicon before any fold; “né” is a relation word; a fold that is a name (“ne” to Nebraska) is not a cue; function words stay non-cues, as B1q already requires. Harness: fame bare line. The reported fame set is dev, seed 0, already read. The known row can be inspected on dev. The arm is one held-out bare list, if one exists. I don’t know if it does. Pass: 0 spoken picks on bare questions. Kill: any spoken pick on a bare question, or any cue that is a relation word only after folding. Cost: a lexicon edit; no GPU; serving latency not increased on the numbers given. CPU hours are not stated. Near rules: the bare-name kill line; namesakes are not ranked; a pick rests on evidence that holds for one candidate. This is the highest gain per cost because the stated bare-pick mechanism is a missing lexicon entry, not a model gap.

**2. One-hop “located in” is a veto, never a positive pick.** What changes: if one candidate stands because it states the discriminator directly, and another candidate reaches that same value by one “located in” hop, the question is undecided. The hop cannot by itself make a candidate stand. That turns the stated Rosebank pattern from a wrong pick into a refusal, and it cannot create a new pick. Harness: fame qualified line, one held-out read, not a re-read of dev seed 0. Pass: 0 wrong, and resolved at least 0.70. Kill: any wrong pick, resolved under 0.70, any bare spoken pick, or p95 over 50 ms. Cost: extra lookups only on the one-candidate path. Added latency is not measured. Current B1q p95 is 40.6 ms at 6 or more, which is 9.4 ms under the 50 ms rule by subtraction of stated figures, so this proposal comes near the cost rule and is out if that p95 breaks. I don’t know how many now-right picks the veto would also refuse. Ranked second because the safety gain is real and the resolve loss is unknown.

**3. Undecided stays undecided. No score fallback.** What changes: the 61 undecided and 154 refused on the fame qualified table, and any new undecided from the veto, are not passed to qFHRR, BM25, or the 450M to lift resolved 0.764. Harness: the same held-out fame read as proposal 2, plus one read of Mintaka and WebQSP_EL under the tightened check. Pass: still 0 bare picks and 0 qualified wrong, resolved at least 0.70, and at most 1 wrong in 50 on Mintaka and on WebQSP_EL. Kill: any ranker added to clear the 2.3% and 2.2%, or more than 1 wrong in 50. Cost: none. Near rules: scores are not thresholds; namesakes are never ranked by popularity. This is a guard, not a recall gain. I don’t know if proposals 1 and 2 are enough to reach 1 wrong in 50. If they are not, the arm stops. It does not grow a chooser.

### Relations

**1. The picker is the grammar, then the lexicon, among the item’s own properties. Otherwise refuse. qFHRR does not pick.** What changes: nothing is added; the table stays not kept, which is what the pre-registered rule already said. A handover requires the grammar or, if the grammar is silent, the lexicon, and the property has to be on the item. Disagreement or absence is undecided, not a nearest label. Harness: the 38 missing-fact rows are already read, so a new arm needs a fresh missing-fact list plus one read of SimpleQuestions and QALD. Pass: 0 missing-fact handovers, matching the grammar on the 38, and no spoken property when the item lacks it. Kill: any handover on a row where the item lacks the gold, or a cosine margin brought back from section 4.3. Cost: no GPU and no added model; this is the current host path (section 5). Grammar latency is not stated. I don’t know it. Recall cost is real and accepted: margin-0 qFHRR’s 0.887 on SimpleQuestions is given up. Near rules: only the VM certifies; a refusal is better than a wrong answer; the VM cannot catch a near property that is on record. Ranked first because it removes the stated danger at no new serve cost.

**2. The asked-kind cut is place and time only, and an empty cut refuses.** What changes: “who” does not drop incoming properties; “how many” does not drop item-valued properties; place and time may narrow the lexicon’s candidates; if the narrow leaves none, the system refuses and does not fall through to a near property. Harness: one fresh read, split by asked kind. The section 4.3 kind numbers are already read. Pass: time has no wrong, place wrong rate at most 6%, and no dropped incoming gold on “who”. Kill: missing-fact handovers above the lexicon’s 6 on a missing-gold set, or handovers rising toward 83, or the cut applied to “who” or “how many”. Cost: symbolic; CPU hours and added latency not stated. It has to stay inside the p95 50 ms lookup-and-check stage. Near rules: the missing-fact rule; scores are not thresholds. Expected gain is the place and time help already measured, kept only if the new read still shows it. I do not treat 0.77 and 0.63 as already passed for this narrower cut.

**3. French property wordings in the frozen lexicon, taken from store labels, not from a model.** What changes: the lexicon gains FR wordings already in the store. No new teacher, no adapter. Harness: one read of QALD-9-plus FR and Mintaka FR. Pass, two lines, both required: the old keep rule, at least 3 points of recall@5 over the current lexicon (0.217 and 0.190) at no higher wrong rate than 15.1% and 21.5%, and missing-fact handovers not above 6. The system gate of 0.80 on QALD FR stays the system gate. I don’t know if a lexicon extension can reach 0.80; this arm does not get to lower it. Kill: wrong rate above the current lexicon, handovers above 6, a second read, or a model-written wording. Cost: no GPU; hours not stated. Near rules: facts are never model-written; pre-registration. Ranked third because the expected recall gain is unknown and the current FR lexicon is far from 0.80. It is still the cheap lever before any rebuild.

A multilingual rebuild of the table is not in these three. It costs an unstated GPU embed of a new vocabulary, it still ranks, and the missing-fact handover is already present in the English-capable table. The 450M label adapter was tried and not kept (section 4.3). Repeating it is out.

## 3. The trade-offs

Latency, memory, and accuracy are from sections 3 to 6 or marked unknown. “Computed” means arithmetic on stated figures, not a new measurement.

| Job | Who does it, in order | Latency, memory, accuracy | Failure mode |
|---|---|---|---|
| Finding a name | Exact name index, then the typo index on an exact miss. qFHRR does not run. The 450M does not run. Loci tokens do not run. The result is a set. | Index: 12.4M names, 4.8M items; gold under the exact name for 87.4% of Mintaka EN mentions and 99.6% of FR. A fifth to a third of mentions name more than 20 items. Typo accuracy, index bytes, and index latency: unknown. qFHRR on BEIR dbpedia-entity is the wrong instrument: nDCG@10 0.046 compact and 0.096 with dense re-rank (v4); compact recall@100 about 0.14. | Refusal if the set is empty or larger than one and someone asks for a spoken name. A spoken pick here would be a wrong answer and is a kill. Slowness: unknown. |
| Choosing between namesakes | The set from the index, then B1q: one role per word, cues looked up as names, a candidate stands only if a statement has that value, one standing is a pick, else undecided. Then the veto in section 2, if it passes its lines. No ranker. VM certifies later hops against the chosen item; I don’t know whether the VM also checks that the cue held for only that item. | B1q on the fame dev set: bare single picks 2 (line is 0); qualified right 692, wrong 3, undecided 61, refused 154, resolved 0.764. Mintaka and WebQSP_EL wrong rates 2.3% and 2.2%, line is at most 1 in 50. p50 / p95: 1.4 / 6.2 ms at 2 candidates, 10.2 / 40.6 ms at 6 or more. Memory: unknown. qFHRR as chooser: killed at every margin in section 4.2. | Wrong answer on the 3 located-in picks and the 2 bare picks, until those arms pass. Otherwise refusal. Slowness only if the veto pushes p95 over 50 ms; that is not measured. |
| Choosing the relation | Host path, as section 5 already orders it: the programs seat may write a program, but the host picks the relation with the grammar, then the lexicon, among the item’s own properties. The 450M’s relation string is not the pick. qFHRR does not pick. VM runs the hop or refuses. | Grammar missing-fact handovers: 0 on 38 rows. Grammar recall@5: unknown. Lexicon recall@5 (wrong rate): SimpleQ 0.563 (4.0%), QALD EN 0.360 (17.5%), QALD FR 0.217 (15.1%), Mintaka EN 0.331 (28.1%), Mintaka FR 0.190 (21.5%). Gates: 0.85 SimpleQ, 0.80 QALD FR. qFHRR margin 0: 0.887 (9.0%) and 0.536 (45.8%) on SimpleQ and QALD FR, not kept. 450M programs seat: 0.470 (6.3%) to 0.066 (7.4%) across the five sets in section 4.3. Program write about 2.4 s. Lookup latency of grammar plus lexicon: unknown. | Refusal when grammar and lexicon are silent or disagree. Wrong answer if the lexicon hands over a near property (6 of 38) and the VM certifies it. The 450M as decider would add slowness (about 2.4 s against a 50 ms lookup-and-check stage; about 48 times, computed from stated figures) and a higher wrong rate than the lexicon on the adapter’s pooled 0.286 against 0.145. |
| Reading French | Exact index for names (already stronger on FR than EN: 99.6% against 87.4%). FR wordings in the lexicon for relations and cues. qFHRR is not the French semantic matcher. The 450M is not the French relation picker on these numbers. Loci language is unknown. | qFHRR relations, margin 0: QALD FR 0.536 against EN 0.755; Mintaka FR 0.397 against EN 0.554. Lexicon FR: 0.217 and 0.190. 450M programs seat FR: 0.115 (9.6%) and 0.066 (7.4%). French hash fallback: section 4.2. Latency and memory of an FR lexicon: unknown. | qFHRR on French meaning: wrong answer, because a hash matches only as written, or a template word moves a score. Symbolic path: refusal when the FR wording is not in the lexicon. I don’t know a measured French slowness. |
| Routing to a world | The qFHRR table only. A neural model on this path is out (section 2). Loci tokens are not a router: they need a host-written scene after a world is chosen. | Encode 6.8 to 25 µs for a short text, one CPU core (section 3). Full decision latency beyond the encode: unknown. Table memory in bytes: unknown. 15-domain screen: 0.341 against the teacher’s 0.358, null 0.063, 0.95 of the teacher. 42 real domains: 97% of the teacher’s quality, 60k vocabulary. Open-set: 92.4% precision, 86.9% of never-seen domains sent to spawn. The metric name for 0.341 is not stated. | Wrong world on the precision gap (92.4% is not 100%). Spawn is the refusal for never-seen domains. Slowness is not the measured problem. The margin gate does not transfer to another table. |
| Retrieval over a known corpus | If the query has an exact name hit, the name index. Otherwise qFHRR compact over the known corpus. Either way the hit is a candidate, not a fact, until the store and the VM say so. | Known corpus: 1M documents, compact, hit@1 0.71, hit@20 0.92 (section 3). Compact size computed from stated figures: 1M × 80 bytes = 80 MB. Scan latency over the 1M: unknown. Encode 6.8 to 25 µs. Open-web entity search is a different job and stays weak (section 4.1). Verified pipeline, misspelled corpus, chain-recovery not real-world correctness: precision 0.992 at 0.646 coverage; later 574 of 800 (0.718) at precision 1.000. | Table alone: wrong document on the hit@1 miss (1 − 0.71). System: refusal when the VM cannot certify. Using the top hit as a fact would be a wrong answer and breaks the fact rule. |
| The n-gram memory | The memory addressed by qFHRR word codes, then a store check, then the VM. A row is not a fact because the codes found it. The 450M is not this memory. Loci tokens are a different reader (section 6). | One paraphrased word in a trigram: its row 74.3% of the time, a different fact 4.0%, 35 hand pairs, not confirmed on a fresh list (section 3). The other 21.7% is a remainder I computed, and the brief does not say what it is. Hamming scan over 10,240 bits. Corpus size, latency, and memory: unknown. | Wrong answer on the 4.0% different fact, if the row is spoken. After a store check, that case should be a refusal. Unconfirmed on a fresh list, so 74.3% is not a pass. |
| Reading a fact the host placed in the context | Store, then the host writes the scene, then loci tokens on the 450M. The trunk does not produce the fact from its weights. A loci read is still a proposal: the VM checks it against the store. qFHRR does not read this scene. Do not key the address by a bare name when the name set has more than one item. | Run 5: facts seen once, more than 800 tokens back, 0.946 exact, copy path 1,027 parameters, about 10 minutes, text unchanged. Run 6: 0.804 exact through the place against 0.019 read the ordinary way; held-out loss 2.2398 against 2.2491. Home serve: not run yet (section 6). Serve latency: unknown. The CPU toy at width 256 is not the 450M: unseen subjects 0.854 to 0.899 against hashed addresses 0.000; the copy/trunk switch wrong on about 12% of present facts. | Designed miss is a refusal: relation absent from the shard, several values at one address, or the copy gate shut. A wrong answer is what the toy’s switch does (about 12% of present facts); I don’t know if runs 5 and 6 miss as refusals or as wrong reads. Slowness at serve: unknown. |

## 4. Growing the model

On the jobs in the table, a larger model is not the lever the numbers point to.

It is not the lever for finding a name. The exact index already holds the gold at 87.4% EN and 99.6% FR. Dense potion-8M at 0.224 still loses to BM25 at about 0.313 on the open-web entity set (section 4.1). No size in the brief beats an exact string match at finding a known name.

It is not the lever for namesakes. A larger scorer would still be a score. Section 4.2 killed every margin of that shape, including BM25. A pick has to rest on evidence that holds for one candidate. Scale does not supply that check. Popularity in the weights would also come near the rule against ranking namesakes by popularity. I don’t know whether the 450M would fail the bare line. It should not be given the job.

It is not the lever for relations, on the evidence in hand. The 450M in the programs seat is below qFHRR margin 0 and, on SimpleQuestions, below the lexicon (0.470 against 0.563). The adapter trained on 163,907 property-label records has a wrong rate above the lexicon on every set and 16 missing-fact handovers. That is the same VM-invisible error. More weights that still name a near property when the gold is absent do not fix section 4.3. French relation recall at this size is 0.115 and 0.066. I don’t know if a larger model would clear 0.80 on QALD FR. The cheaper lever is the FR lexicon, and it has to be tried first.

It is not the lever for routing. A neural model on MoWM’s routing path is out. Growing the 450M does not change that.

It is not the lever for known-corpus retrieval or the n-gram memory. Those are table jobs with stated rates (hit@1 0.71; 74.3% / 4.0% on 35 pairs). No grown-model retrieval number is in the brief. I don’t know one.

It is not the lever for reading a host-placed fact. The measured gain is the copy path: 1,027 parameters, about 10 minutes, 0.804 against 0.019 (section 6). The miss when the shard never addressed the relation is a coverage hole in the shard, or a missing host mapping, not a shown size hole. Growing the trunk so its weights know more facts would fight the design: the host writes the facts, and facts are never model-written.

I don’t know a job in this brief for which size is the measured bottleneck. Program quality against size is not stated. I don’t know it. Latency of a grown model is not stated. The programs seat is already about 2.4 s.

What it would have to show, if a grow were still run later, on one pre-registered read, off the routing path: for relations, at least 3 points over the lexicon at no higher wrong rate, 0 missing-fact handovers, and the QALD FR gate of 0.80 if that arm claims the system gate; for French, those FR lines, not a transfer of an English margin. I don’t know what size would show that. No size should be picked from these pages.

Cheaper levers first, in order: the lexicon and fold fix, the located-in veto, grammar-then-lexicon with refusal, FR store wordings in the lexicon. A LoRA is the first model spend if a model arm is still open after those, not a grow: a little over an hour on an A100 80 GB for about 216k rows (section 5). The label adapter was already that class of spend and was not kept. One read, then stop. A full grow’s GPU hours are not stated. I don’t know them.

## 5. Loci tokens

**Entity names: leave the measured weakness as it is.** Loci keys come from the trunk’s subword embeddings, so a name has a key with no document-frequency miss (section 6). That is identity, not meaning, and it is a key for a fact the host already placed, not a search over 4.64M documents or over 12.4M names. Section 4.1’s nDCG gap is untouched. Name spans are a recall mechanism, not a training-cost lever: proper names are 5.2% of the training-mix tokens. Divide the work: the exact index finds the set; loci reads facts only after a single item has been chosen and the host has written that item’s facts. Do not combine qFHRR codes into the loci key for finding. A combination is not measured. I don’t know if it would help. Both codes are identity for an unseen name, so neither one chooses among namesakes.

**Namesakes: hurt if used to choose; leave as it is if the host writes only after a unique symbolic pick.** Two items with the same name get the same key (section 6). The asker builds the key from the subject tokens it points at. Nothing in section 6 splits Rosebank-in-Cape-Town from the other Rosebank. One address holds one value, and several values are a stated miss. Writing both items under the name would collide. The key is symmetric in subject and relation because binding commutes, so it also does not encode which side the discriminator is on. Divide: B1q picks or refuses; on a refusal the host writes no scene and the trunk says it does not know; on a pick the host writes that item’s facts under a key that is not the bare shared name. I don’t know whether item-id keys were tried. qFHRR must not be the fallback chooser in front of the write.

**Relations: leave the naming weakness as it is, and hurt if a near wording is used as the key.** A relation’s key comes from its words, so a different wording is a different address (section 6). That does not map a question onto one of 12,684 properties. The question-loop reader misses when the relation is outside the shard, whose relations are the wiki world’s with Wikidata labels mapped onto them. That mapping is a host or lexicon act, not a loci act. If the key is a near property, the trunk can read a value the host wrote for the wrong property, and the VM will not see the mismatch if it only checks that the value is on record. The toy’s copy/trunk switch, about 12% wrong on present facts, is a further wrong-answer path; it was not run on the 450M. Divide: grammar and lexicon map words to a property id, or refuse; the host writes under the canonical property, not under the question’s raw wording; loci reads the value; the VM checks the store. qFHRR does not choose the property in front of that write. Shared binding algebra (sections 3 and 6) is not a reason to blend the two codes here. No such blend is measured.

## 6. What I would not do, and open questions

I would not put the 450M, or a grown model, on the routing path. I would not revive qFHRR, BM25, or any margin as a namesake chooser, and I would not rank namesakes by sitelinks, degree, or page views. I would not transfer 0.25, 0.5, 0.75, or 0.9 to a new encoder, table, or corpus. I would not let margin-0 qFHRR hand over a near property. I would not spend the one test read on an AND of qFHRR and the lexicon: section 4.3 already measured +2.7 points pooled, short of the +3 rule, and a conjunction cannot beat the lexicon’s recall if the lexicon must also fire. I would not apply the asked-kind cut to “who” or “how many” and then let a ranker fill. I would not use a one-hop geographic expansion as a positive pick. I would not key loci by a bare shared name. I would not treat a loci read, a table hit, or agreement between arms as certification. I would not have a model write a fact, a lexicon wording, or a training record. I would not reopen n-gram and binding variants inside the table for these three jobs: section 3 says every such variant cost more than it bought, and section 4 has no number showing one of them helps. I would not chase BEIR dbpedia-entity nDCG inside the word table. I don’t know if open-web entity search is a job this system has to do; name finding for its own items is already an index job. I would not grow the model to memorize Wikidata or to close the three weaknesses. I would not re-read dev seed 0 to tune the veto.

I don’t know: typo-index accuracy, latency, or bytes; the name index’s bytes and lookup latency; whether a held-out fame list exists; the types of the Mintaka and WebQSP_EL wrongs that remain at 2.3% and 2.2%; B1q’s ShadowLink gap; grammar recall@5; whether the VM checks namesake uniqueness; full dense qFHRR nDCG on BEIR without the compact cap; the word table’s bytes, dtype, and 1M scan latency; loci serve latency on grilly2, and the language of runs 5 and 6; whether item-id keys were tried; Cubby’s French ability outside the relation tables; what the n-gram remainder after 74.3% and 4.0% is, or the fresh-list result; what “the earlier tiers” are beyond the cells in section 4.2; real-world correctness of the misspelled-corpus pipeline; any grown-model run. Where those are missing, the pass lines above stay unmet until a single pre-registered read, and the arm stops on the kill line.

## moonshotai/kimi-k3.1-pro

*13770 tokens out, 141.8 s, finish `stop`*

## 1. Diagnosis

### Entity names

The dominant cause is the **document-frequency vocabulary**, with the **hash fallback** turning every other unseen name into identity rather than meaning. The preprint says this directly, and the measurements support it: v4 improves nDCG@10 from 0.040 to 0.046 compact and from 0.081 to 0.096 with dense re-ranking, but remains 2.3 times behind dense potion; compact recall@100 is only about 0.14 and caps the whole pipeline (§4.1).

There are two secondary causes:

- **Compact shortlisting**, independently of vocabulary: dense re-ranking approximately doubles nDCG@10 but cannot recover documents absent from the shortlist.
- **Bag-of-words encoding**, which cannot represent a multiword name as an ordered unit. The numbers do not isolate its contribution.

Vocabulary expansion and shortlist design are fixable inside a static table. Permanent semantic coverage of an open-ended name inventory is not: whatever the vocabulary size, a later unseen word still receives only its hash code. The exact-name results—87.4% EN and 99.6% FR gold coverage under exact names—show why name finding should be symbolic rather than semantic (§4.1).

### Choosing between namesakes

The primary cause is **using a similarity score as a ranker**, compounded by **task framing**. A bare name shared by several items contains no evidence that holds for one candidate. No margin can manufacture that evidence. qFHRR produced spoken bare-name picks at every margin, including 11 at margin 0.9; BM25 failed for the same reason (§4.2).

The margin-0.9 diagnosis also identifies a fixable encoding/pipeline fault: template relation words such as “pays”, “né” and “born” were treated as cues. A bag of words does not know their roles. B1q’s “né”/Nebraska error shows that the replacement lexicon has the same role-coverage risk.

The structural limit is that a static word table has no statements, qualifier direction, candidate roles or symbolic proof obligation. Better prototypes or margins may change scores, but cannot certify a choice. Role masking, candidate generation and abstention are fixable around the table; the actual choice must remain symbolic. Popularity is not an option.

### Relations

There are three distinct causes:

1. **Candidate framing.** Among all 12,684 properties, recall@5 is 0.417 on SimpleQuestions and 0.361 on QALD FR. Restricting candidates to the item’s own properties raises these to 0.887 and 0.536 (§4.3). This is fixable and already demonstrated.
2. **The English teacher and vocabulary.** QALD drops from 0.755 EN to 0.536 FR; Mintaka drops from 0.554 EN to 0.397 FR. The explicit French hash-fallback explanation in §4.2 supports the same cause here. A French or multilingual static table is fixable inside the architecture.
3. **A ranker’s obligation to return something, especially when the fact is missing.** On 38 rows lacking the gold property, margin-0 qFHRR hands over a near property on 36. Because that property is really present, the VM checks it and cannot detect that it answers the wrong question (§4.3). This is not repairable by a better score alone; it requires abstention or independent lexical/symbolic support.

The asked-kind filter is also a framing error when made mandatory. It helps “place” and “time”, but discards incoming relations and item-valued counting relations, increasing missing-fact handovers from 36 to 83 (§4.3).

## 2. Fixes

Section 4’s sets have already been read. Their numbers below are baselines, not fresh test evidence. Every arm needs its thresholds fixed on named dev data with a date, followed by one read of a frozen held-out/test set. If no fresh split exists, the arm remains dev-only and cannot pass. Every serving arm must also keep lookup-and-check within p95 50 ms.

The brief does not state build hours or index memory for the new arms, so those costs are unknown rather than estimated.

### A. Entity names

**A1. Use lexical indexes for finding; remove qFHRR from the name-finding decision.**  
*Expected gain per cost: highest.*

- **Change:** For Wikidata-style questions, use the exact name index, then the typo index on exact misses. For open-web dbpedia-entity retrieval, build a BM25 shortlist and optionally retain dense re-ranking. qFHRR remains available for routing and semantic retrieval, not as evidence that a name denotes an item.
- **Reads:** A fresh BEIR dbpedia-entity run measuring nDCG@10 and shortlist recall; fresh Mintaka EN/FR mentions for exact-name and typo-index candidate coverage.
- **Pass:** BEIR nDCG@10 at least 0.30; fresh Mintaka exact-name gold coverage at least 87.4% EN and 99.6% FR, with typo lookup never removing an exact hit; p95 lookup-and-check at most 50 ms.
- **Kill:** Any of those lines fails, or the index cannot be built and served within the system cost target.
- **Cost:** No GPU is required by the proposal. CPU build hours, index memory and query latency are unknown. Existing qFHRR entity results and the published BM25 result of about 0.313 are the comparison points (§4.1).
- **Hard rules nearby:** Facts remain store statements; p95 50 ms; no neural model is added to MoWM routing.

**A2. Add the tail name vocabulary to the static table.**  
*Expected gain per cost: lower; worthwhile only as a bounded retrieval experiment.*

- **Change:** Add entity-name words and domain words absent from the document-frequency vocabulary, embed them offline, and rebuild compact and dense indexes. Do not treat added names as facts.
- **Reads:** Fresh BEIR dbpedia-entity, plus regression reads of the 42-domain routing screen and known-corpus retrieval.
- **Pass:** Compact recall@100 at least 0.28, dense-re-ranked nDCG@10 at least 0.192, and no reduction in the routing or known-corpus retrieval measures; p95 remains within budget.
- **Kill:** Either retrieval line fails, a regression appears, or table/index memory makes the arm unaffordable. Also kill if improvement comes only from dense re-ranking while compact recall remains capped.
- **Cost:** One offline teacher embedding per added vocabulary entry; GPU hours and added memory are unknown. Serving is still static lookup, but latency for the enlarged table is unknown. Current short-text lookup is 6.8–25 µs on one CPU core (§3).
- **Hard rules nearby:** Thresholds cannot move from v1/v4 to this table; the served routing path remains non-neural.

### B. Choosing between namesakes

**B1. Harden B1q; do not repair the qFHRR ranker.**  
*Expected gain per cost: highest.*

- **Change:** Complete the EN/FR relation and function-word lexicon, including “né” and its folded forms, so relation words cannot become cues. For administrative qualifiers, permit a candidate to hold the qualifier through an explicit stored “located in” chain, with every hop checked. Multiple surviving candidates remain undecided.
- **Reads:** Fresh fame-set bare and qualified rows, ShadowLink, fresh 50-row Mintaka and WebQSP_EL reads, and a new administrative-location set containing direct and indirect qualifiers.
- **Pass:** Zero spoken picks on bare names; zero wrong qualified picks with at least 70% resolved; ShadowLink Top–Shadow gap at most 10 points; at most one wrong answer in each 50-row Mintaka/WebQSP_EL read; p95 at most 50 ms.
- **Kill:** Any bare spoken pick, any wrong qualified pick, failure of any other line, or p95 above 50 ms.
- **Cost:** CPU-only; development hours and added lexicon memory are unknown. Current B1q costs are p50/p95 1.4/6.2 ms at two candidates and 10.2/40.6 ms at six or more, so the high-candidate case has little latency headroom (§4.2).
- **Hard rules nearby:** Refusal over a wrong answer; no popularity; only stored facts and VM-checked hops.

**B2. Test qFHRR only as a candidate proposer upstream of B1q.**  
*Expected gain per cost: secondary and strictly ablated.*

- **Change:** After role scrubbing, let qFHRR propose additional value items or cue interpretations. B1q must still find the qualifier in a candidate’s statements. Cosine order never selects the answer.
- **Reads:** The same fresh sets as B1, with hardened B1q alone as the paired control.
- **Pass:** Every B1 safety line holds, qualified resolution improves by at least 3 percentage points over hardened B1q, and p95 remains at most 50 ms.
- **Kill:** Any new wrong or bare pick, less than 3 points of added resolution, or a latency failure.
- **Cost:** No serving GPU. qFHRR lookup is 6.8–25 µs for a short text; prototype memory, extra symbolic checks and total latency are unknown (§3).
- **Hard rules nearby:** Agreement may only filter; scores require their own dev calibration; the VM, not qFHRR/B1q agreement, certifies.

### C. Relations

**C1. Build a French static table and French label lexicon, with separately calibrated abstention.**  
*Expected gain per cost: highest for the largest measured deficit.*

- **Change:** Build the French vocabulary from French document frequency plus property-label and domain words, using a suitable French or multilingual teacher if one is available. Which teacher to use is not stated. Keep candidates restricted to the item’s incoming and outgoing properties. Expand the deterministic French label lexicon; do not transfer EN margins.
- **Reads:** Fresh QALD-9-plus FR and Mintaka FR, EN regression sets, and the 38 missing-fact rows plus a fresh missing-fact set.
- **Pass:** QALD FR recall@5 at least 0.80; SimpleQuestions remains at least 0.85; wrong rate no higher than the frozen lexicon on each set; zero qFHRR-only near-property handovers on missing-fact rows.
- **Kill:** Any of those lines fails, or no affordable French table meets them.
- **Cost:** Offline GPU hours and new table/lexicon memory are unknown. There is no neural serving model; new-table lookup latency must be measured.
- **Hard rules nearby:** No neural model on routing; thresholds do not travel; refusal over a near-property answer.

**C2. Make relation selection proposal-based and direction-safe.**  
*Expected gain per cost: necessary safety work, with uncertain recall gain.*

- **Change:** Grammar and lexicon come first. qFHRR may add or reorder proposals among the item’s own properties, but cannot emit a property solely because it is semantically near. Use asked-kind information as a soft proposal feature, never a filter: retain incoming relations for value-side “who” questions and item-valued relations for counting. If no candidate has the required independent support, refuse.
- **Reads:** All five own-property relation sets, the missing-fact sets, and dedicated incoming-“who” and item-valued-“how many” sets.
- **Pass:** The combined selector reaches 0.85 on SimpleQuestions and 0.80 on QALD FR; adds at least 3 points pooled over the lexicon at no higher wrong rate; produces zero near-property handovers on missing-fact rows; p95 stays within budget.
- **Kill:** The original keep rule fails again, any missing-fact handover remains, or either task gate fails.
- **Cost:** No training GPU is inherent in the proposal. Added CPU hours, memory and end-to-end latency are unknown; the table component itself is microseconds for short texts.
- **Hard rules nearby:** The VM certifies only factual derivation, not semantic correctness; refusal is preferred; pre-registered thresholds remain encoder- and language-specific.

## 3. Trade-offs

| Job | Part and order | Measured latency and memory | Measured accuracy | Principal failure mode |
|---|---|---|---|---|
| Finding a name | Exact symbolic name index → typo index on exact miss. No qFHRR choice. | Latency and memory unknown. | Gold under exact name: 87.4% Mintaka EN, 99.6% FR. A fifth to a third of mentions name more than 20 items (§4.1). | Refusal/no candidate on a miss; a large candidate set defers rather than solves ambiguity. |
| Choosing between namesakes | Role/span rule → B1q symbolic qualifier check → one candidate or undecided. qFHRR, if retained, only proposes. | B1q p50/p95 1.4/6.2 ms at two candidates; 10.2/40.6 ms at six or more. Memory unknown. | Current B1q: 2 bare picks, 692 qualified right, 3 wrong, 0.764 resolved; Mintaka/WebQSP wrong rates 2.3%/2.2%, still above the 2% line (§4.2). | Wrong answer from lexicon or qualifier gaps; otherwise refusal/undecided. |
| Choosing the relation | Own-property candidate construction → grammar/lexicon → guarded qFHRR proposals → refusal without support → VM execution. | qFHRR short-text lookup 6.8–25 µs; total selector latency and memory unknown. | qFHRR margin 0: 0.887 SimpleQ, 0.536 QALD FR; lexicon 0.563/0.217. The 450M adapter reaches 0.800/0.241 but has pooled wrong rate 0.286 against 0.145 (§4.3). | Wrong answer on a present near property, which the VM cannot catch; safe alternative is refusal. |
| Reading French | Exact index for names; French static table/lexicon for routing and relation proposals; 450M only on the permitted reasoning path. | General French latency/memory unknown. | Exact-name coverage 99.6% FR. Relation recall: qFHRR 0.536 QALD FR and 0.397 Mintaka FR; 450M adapter 0.241 QALD FR (§§4.1–4.3). General French reading accuracy unknown. | Wrong relation from hash identity or weak adapter; refusal when support is absent. |
| Routing to a world | qFHRR table alone, bundled centroids and a dev-set margin gate. | 6.8–25 µs for a short text on one CPU core; centroid/table memory unknown. | 0.341 against teacher 0.358, null 0.063; 97% of teacher quality on 42 real domains; open-set precision 92.4%, with 86.9% of unseen domains sent to spawn (§3). | Wrong world, or spawn/refusal at the gate. |
| Retrieval over a known corpus | Compact qFHRR shortlist → dense re-rank where affordable; exact symbolic retrieval for names. | Compact 80 bytes/document; dense 20 KB/document. At 1M compact documents this is 80,000,000 bytes by arithmetic. End-to-end latency unknown. | Hit@1 0.71, hit@20 0.92 on 1M documents (§3). | Missed document leading to refusal, or wrong retrieved candidate; VM checks stored facts, not corpus truth. |
| The n-gram memory | qFHRR word codes → Hamming scan → retrieve a proposed row → symbolic fact check before certified use. | Scan is over 10,240 bits; latency and total memory unknown. | Paraphrased trigram reaches its row 74.3%, reaches a different fact 4.0%; only 35 hand pairs and not fresh-list confirmed (§3). | Wrong fact on the 4.0% path, or a miss/refusal. |
| Reading a fact the host placed in context | Host writes addressed facts → loci read/copy path in the 450M → VM checks any answer that must be certified. | Serving latency and added memory unknown; grilly2 port and home run-6 read are not complete. Copy training in run 5 took about 10 minutes for 1,027 parameters. | Run 5: 0.946 exact after one exposure more than 800 tokens back. Run 6: 0.804 through place against 0.019 ordinary reading. Copy/trunk switch gets about 12% of present facts wrong (§6). | Wrong copy/switch answer, refusal when the gate stays shut, and unknown serving slowness. |

## 4. Growing the model

A larger model is a plausible lever for only two areas:

- **Relation wording and French reasoning**, after the static French table and lexicon have been tried. It might understand more wordings, but it does not solve a missing fact and must not be allowed to turn a near property into an answer.
- **Host-context reading with loci**, if fresh measurements show that trunk capacity, rather than addressing, multi-value storage, relation coverage or the copy gate, is the limiting factor.

It is not the lever for finding names, namesake choice, routing, known-corpus lexical retrieval or n-gram addressing. Those need indexes, symbolic evidence, better candidate construction or better addressing. A neural model on MoWM routing is prohibited.

**Size:** I do not know what size would help; the brief contains no scaling result. A roughly 1B-parameter trunk would be a proposed experimental pilot, not a data-derived optimum. Its training and serving memory on the A100 80 GB or home GPU are unknown. The stated LoRA cost—a little over an hour for about 216k rows on an A100 80 GB—cannot safely be extrapolated to a larger trunk (§5).

A larger relation model would have to reach recall@5 at least 0.85 on SimpleQuestions and 0.80 on QALD FR, keep wrong rate no higher than the lexicon, and produce zero missing-fact near-property handovers on fresh rows. A larger loci model would need fresh host-context reads at least as good as the 0.946 run-5 result, no held-out text-loss regression, and measured improvement on unseen relations, multiple values and the copy/trunk switch. Agreement or fluent output would not count.

Try first, in order: B1q hardening; exact/BM25 name retrieval; French static table and lexicon; safe relation abstention; then the cheap loci copy-path, addressing and gate experiments. Only a persistent, measured trunk-capacity limit justifies the scale arm.

## 5. Loci tokens

- **Entity names: leave the weakness as it is.** Loci give every name a key derived from subword embeddings, so there is no vocabulary miss, but the key carries identity rather than meaning and does not find an item in a 4.64M-document corpus. Exact/BM25 indexes find; qFHRR routes or retrieves; loci may subsequently read host-supplied facts about the found item. Name spans remain a recall mechanism, not a training-cost lever: proper names are 5.2% of training tokens (§6).
- **Namesakes: hurt if used as the chooser.** Two items bearing the same name receive the same key, directly preserving the ambiguity. B1q must choose from question evidence or refuse. After a symbolic choice, loci can read that item’s host-provided scene, with VM verification; they must not convert a name-key match into a pick.
- **Relations: narrow help after canonical selection, not a relation chooser.** qFHRR and the lexicon can map question wording to a proposed canonical property; the host would then have to write and query compatible canonical addresses. A different relation wording otherwise means a different address. Binding is symmetric in subject and relation, unseen training relations miss, one address holds one value, and the copy gate can stay shut (§6). Loci therefore help read an already selected relation’s stored fact, but hurt if substituted for candidate restriction, French semantics, direction handling or abstention.

## 6. What I would not do, and open questions

I would not:

- restore qFHRR, BM25 or popularity as a namesake ranker;
- permit any spoken pick for an underspecified bare name;
- revive the all-property codebook without candidate restriction;
- use asked-kind as a hard filter;
- accept a qFHRR-only near property when the asked fact is absent;
- treat VM certification, model agreement or loci copying as real-world correctness;
- transfer margins between tables, languages or corpora, or keep reading test sets while tuning;
- put a neural model on MoWM routing;
- spend first on growing the model, further n-gram/binding variants, or name spans for training-cost savings;
- describe the verified pipeline’s 0.992/1.000 precision as real-world correctness; its corpus measures chain-recovery fidelity (§3).

Open questions:

1. What are the exact index’s memory, query latency and typo-index incremental coverage?
2. Can a BEIR BM25 implementation reproduce the published result under the same scoring, and what are its recall@100 and index cost?
3. Which French teacher and corpus are available, and what French vocabulary size and memory result?
4. Does symbolic “located in” closure remove B1q’s Rosebank errors without creating additional surviving candidates?
5. How frequent are missing-gold rows outside the recorded 38, and can independent support reduce handovers without sacrificing the recall gates?
6. Does the n-gram result survive a fresh list, and what are its scan latency and memory?
7. On the 450M, can loci handle multiple values, unseen relations and canonical French relation wordings, and can the present-fact switch error be reduced below 12%?
8. What does the completed grilly2 loci port cost in home serving latency and memory?
9. Is there any measured evidence of a trunk-capacity bottleneck that would justify a particular larger model size?
