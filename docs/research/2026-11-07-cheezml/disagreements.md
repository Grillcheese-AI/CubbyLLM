# CheezML v0.1: disagreements, choices, cuts and open questions (2026-11-07)

Panel: GPT-6.1 sol pro (GPT), Gemini 4 argon (Gemini), GLM 5.3 (GLM), Qwen 3.8 max (Qwen), Claude Opus 5.5, truncated (Opus), Kimi K3.1 pro (Kimi), DeepSeek V4 pro (DeepSeek). Grok 4.7 came in after the synthesis; it is checked in §5 and changes none of the choices.

## 1. Where the panel disagreed, and what this synthesis chose

1. **Decision Ontology IRIs.** The panel made six different guesses: Gemini `w3id.org/decision#`, GLM a googlecode SVN IRI, Qwen an undefined `decision:`, Opus `example.invalid`, Kimi `w3id.org/decision-ontology/`, DeepSeek `w3.org/2001/sw/hcls`; GPT left it open. **Chose:** mint `cheez:Question`, `cheez:Option`, `cheez:Decision`, `cheez:option`, `cheez:hasResult`, `cheez:chosenOption`, `cheez:isSatisfiedBy` and `cheez:isViolatedBy`, with `skos:closeMatch` to the published `dec:` terms (`example.com` namespace). The published ontology has no `is_satisfied_by` or `is_violated_by`.
2. **What the Decision is.** Gemini, Qwen and Kimi made it the certificate, Opus and DeepSeek a separate node, GLM the answer. **Chose:** the answer or the refusal itself, as the brief says ("the answer or refusal as the `Decision`").
3. **Module of the Certificate.** Core (GPT, GLM, Opus, DeepSeek) or cubby (Gemini, Qwen, Kimi). **Chose:** core, because certification is the VM's job, not one product's.
4. **Criterion direction.** Option→check (Gemini, Kimi) or check→option (Opus). **Chose:** the check run points to the option with `cheez:isSatisfiedBy`, which reads like the brief's "criteria (`is_satisfied_by` / `is_violated_by`)".
5. **Check results.** Check-run activities (GPT, Opus, Kimi, DeepSeek, and Gemini as an entity) or codes and booleans on the record (GLM, Qwen). **Chose:** `cheez:CheckRun` with an outcome and an end time. A check that never ran has no record (hard rule 1).
6. **Recorded time.** Six answers minted a `recordedTime`; Opus reused PROV. **Chose:** PROV reuse (`prov:generatedAtTime`, closed by `prov:invalidatedAtTime`, successor via `prov:wasRevisionOf`). The brief's "reuse wins" binds, and PROV's generation and invalidation already give append-only recorded time.
7. **Valid time.** An instant (GLM, Qwen) or an interval (GPT, Gemini, Opus, Kimi, DeepSeek). **Chose:** `time:Interval` with `time:inXSDDateTimeStamp` boundaries.
8. **Statement content.** RDF reification (Opus, Kimi) or left open (the others). **Chose:** left out. `rdf:` reification is outside the allowed namespaces, and the brief does not say how the store holds statements. Records carry the version, its source and the revision.
9. **Calibration.** Inline properties on the threshold (Gemini, GLM, Qwen, Kimi) or a calibration node (GPT, Opus, DeepSeek). **Chose:** a `cheez:Calibration` activity reached by `prov:wasGeneratedBy`, so one screen can set several thresholds and one shape checks it.
10. **Calibration scope.** Encoder (GPT, Gemini, Qwen, Opus), corpus (GPT, Qwen, Opus), deployment (GPT, Opus). **Chose:** screen, data, date, encoder and deployment, all required. The corpus is folded into `cheez:calibrationData`.
11. **Number type.** Decimal (GPT, GLM, Qwen, Opus, Kimi), double (Gemini) or float (DeepSeek). **Chose:** `xsd:decimal`, written as JSON strings so JSON-LD does not turn them into doubles.
12. **Routing record.** Flat scores (Gemini, GLM, Qwen, Kimi, DeepSeek) or one check run per comparison (GPT, Opus). **Chose:** flat scores, thresholds and a coded outcome. SHACL-SPARQL proves that the outcome follows from the values and that margin = best − runner-up, so there is no stored "passed" field.
13. **RoutingDecision class.** `prov:Activity` (Gemini, Opus, Kimi, DeepSeek), an entity (GPT), or a Decision-Ontology decision (Gemini, GLM, Qwen). **Chose:** `prov:Activity`. The decision terms stay with the certificate step.
14. **Floor and margin.** Dedicated properties (Gemini, Kimi, DeepSeek) or a generic threshold on checks (GPT, Opus). **Chose:** `mowm:floorThreshold` and `mowm:marginThreshold`, and each threshold names its parameter.
15. **"Kind" codes.** Minted `kind`, `checkCode`, `checkKind` or `helperStatus` (GPT, Gemini, GLM, Opus, Qwen), or `dcterms:type` (Kimi). **Chose:** `dcterms:type`, which is reuse. DCMI gives it the range `rdfs:Class`, so the code concepts are punned; this is accepted and listed in §3.
16. **Outcome property.** One generic property (GPT, Opus) or one per record type (Gemini, Qwen, Kimi, DeepSeek). **Chose:** one `cheez:outcome`, with a scheme per record type enforced by `sh:in`.
17. **Reason property.** A generic `cheez:reason` (GPT, Opus) or `refusalReason` and `retirementReason` (Gemini, GLM, Qwen, Kimi, DeepSeek). **Chose:** the majority, `cubby:refusalReason` and `cheez:retirementReason`.
18. **Proposers.** SKOS concepts (GPT, Qwen, Opus) or agents (Gemini, GLM, Kimi, DeepSeek). **Chose:** concepts. They are stable codes that `sh:in` can check, and agent instances can be added later.
19. **Dialogue acts.** SKOS concepts (six answers) or turn subclasses (DeepSeek). **Chose:** concepts with an "Act" suffix (`cubby:RefusalAct`), so they do not collide with the class `cubby:Refusal`.
20. **Conflict kinds.** A separate scheme under reason Conflict (GPT, Opus, Gemini, Kimi, DeepSeek) or folded into the reasons (Qwen). **Chose:** a separate scheme, required exactly when the reason is `cubby:Conflict`.
21. **Hash form.** Hash plus algorithm (GPT, Kimi), sha256 only (Gemini), an `alg:hex` string (Opus) or one hash property per kind of thing (DeepSeek). **Chose:** one `cheez:Artifact` class with `cheez:hash` and `cheez:hashAlgorithm`.
22. **Rejected helper.** A status property (GLM, Qwen, DeepSeek), a retirement activity (GPT, Gemini, Opus) or a type code (Kimi). **Chose:** `cubby:HelperRejection` ⊑ `cheez:Retirement`, which `prov:invalidated` the helper (the helper stays). Its reason is the code of the failed gate.
23. **Gate results.** Counts (Gemini) or one check run per gate (GPT, Opus). **Chose:** check runs. Escape attempts are an observed value that must be 0.
24. **Reviewer.** A role property (Gemini, GLM, Qwen) or a PROV qualified association (GPT, Opus, Kimi). **Chose:** the qualified association. The committed fact's `prov:hadPrimarySource` is the reviewer (GPT), because "a reviewer is a source".
25. **Suggestion and committed value.** Boolean flags (Gemini, GLM, Kimi, DeepSeek) or separate classes (GPT, Opus). **Chose:** `cubby:FieldSuggestion`, turned into a `cheez:FactVersion` by a `cubby:FieldCommit`.
26. **Commit to review link.** GPT and Kimi used `prov:used`. **Chose:** `prov:wasInformedBy`. A review is a `prov:Activity`, and PROV-O makes Activity and Entity disjoint.
27. **Region text.** A minted `regionText` (Gemini, Qwen, Kimi, DeepSeek) or `schema:text` (GPT, GLM, Opus). **Chose:** `schema:text`, which is reuse. The reader version is the minted `cubby:readerVersion` (five answers).
28. **Names.** By majority: `Turn` (4) over `DialogueTurn` (3), `DocumentRegion` (4) over `Region` (3), `zDepth` (4) over `z` (3), `parentWorld` (6), `proposer` (4) over `proposedBy` (2).
29. **World id type.** Integer (GLM, Qwen, Opus, Kimi, DeepSeek) or nonNegativeInteger (GPT, Gemini). **Chose:** `xsd:integer` with `sh:minInclusive 1000`.
30. **Spawn activity name.** `Spawn` (Opus, Kimi), `WorldSpawn` (Gemini), `SpawnEvent` (Qwen), `SpawnDecision` (DeepSeek), `WorldChange` (GPT). **Chose:** `mowm:WorldSpawn`, which keeps the outcome code `mowm:Spawn` unambiguous.
31. **Split.** Its own class (DeepSeek, GPT) or the same as a spawn (Gemini, Qwen, Opus, Kimi). **Chose:** its own class, `mowm:WorldSplit`. Router spawns and z+1 splits follow different rules: tau_spawn and counter 5000 for one, tau, z_max and counter 1000 for the other.
32. **Prose.** Rendered text from a template (Opus) or templated `schema:text` on the record (GLM). **Chose:** `cheez:RenderedText` only. Turns forbid `schema:text`, and certificates and refusals are closed shapes.
33. **Banning truth scores.** `sh:maxCount 0` on an invented property (GLM, Qwen, DeepSeek). **Chose:** `sh:closed` on certificates and refusals, so any added score, rationale or status property fails (negative n01b).

**Assumptions this synthesis adds (not in the brief):**
- Example 03's spawned world takes id 5000, assuming a bridge spawn goes through the router's counter.
- `cubby:ConflictCheck` and `cubby:DatePrecisionCheck` are inferred from the refusal reasons.
- The roles `FirstApprover` and `SecondApprover` are inferred from "a second approval".
- `Approved` is the only review outcome.
- The world tau is calibrated against the HYLA encoder.
- The refusal in example 02 comes from a VM run that returned.

## 2. Cuts (panel proposals, merged and de-duplicated)

- **Status fields.** No "PASSED", "Active" or "Certified" status, and no certification by agreement or convergence (all 7).
- **Rationales.** No prose rationale or explanation fields (all 7). Text is template output only.
- **Truth claims.** No truth score, "correct" or "verified fact" property (GPT, GLM, Opus, Kimi, DeepSeek).
- **Embedded payloads.** No embedded vectors, weights, programs or bytecode; record hashes and training runs instead (GPT, Gemini, Kimi, DeepSeek).
- **Emotion.** No new emotion vocabulary: link EmotionML and keep user inference apart from hormone state (GPT). `HormoneState` is deferred until something reads it (GLM, Qwen).
- **Container bridge.** No container or orchestrator bridge records, because a `container` compiles to nothing (Gemini, GPT).
- **Opcodes.** No opcode names or bytes in records; the synced table belongs to the compiler repositories (Kimi, cutting GLM's `Opcode` class).
- **Routing policies.** No single routing policy. Bridge centroid routing, fact-world best-member routing, router residual spawning and depth-limited splits stay distinct (GPT). Qwen's `PossibilityOracle` and `InterWorldQuery` classes are folded into the routing mode.
- **MoWM faults.** No child-id arithmetic, no spawn on every low-confidence step and no sentence encoder on the routing path (Kimi).
- **Program flags.** No `strictMode`, `compilesToday` or `specOnly` flags on programs (Qwen; GLM's flags). Compilation is a `cubelang check` run, not a declared field.
- **Deletion.** No physical deletion on pruning (Kimi, GLM, Gemini, GPT).
- **XML.** No XML serialisation to match the name (Kimi).
- **Strategy classes.** No class per strategy; strategies are SKOS concepts (DeepSeek).
- **Cut by this synthesis.** All 12 panel CubeLang programs, since none compiles; they will be written against the compiler. Also deferred until a record needs them: emotion, story worlds, conversation state, split questions, domain worlds, RO-Crate, nanopublication and STATO packaging, `TrainingRun`, encoder kinds, counter identity, Oja consolidation and a `HYLA` class.

## 3. Open questions (merged, de-duplicated)

1. Is there an authoritative Decision Ontology namespace other than `example.com/ns/dec#`, and what are the domains and ranges of `dec:has_result` and `dec:involves_choosing`? (all 7)
2. Does `router.py` pruning delete or retire? CheezML can only record retirement (`cheez:Pruned`). (GPT, Gemini, GLM, Qwen, Kimi, DeepSeek)
3. Does "clears" mean > or ≥ for the floor and for tau_margin? The SHACL rule accepts equality either way. (GPT)
4. What is compared with a world's tau, and in which direction? This is why example 04 records only the z_max check. (GPT, Opus)
5. Which counter issues a bridge-spawned world's id? How are the counters persisted, and kept from colliding once the world counter passes 5000? (Opus, GPT, Kimi)
6. How does the store hold statement content: named graphs, reification or a hashed payload? (GPT, Kimi)
7. Which hash algorithm, and which canonicalisation for RDF, programs and weights? (GPT, Qwen, Kimi)
8. How is the host's identity (`cheez:writtenBy`) authenticated? (GPT)
9. What is the convention for closing a version, and do turns and routing records also need two clocks? (GPT, GLM)
10. What is the second-approval amount and its currency, and is it a policy limit or a calibrated threshold? It is not modelled; example 06 shows one approval and makes no claim about the amount. (all 7)
11. What sends a field to review (a derived value, a failed check, the amount)? What review outcomes exist besides Approved, and what are the reviewers' job roles? (Opus)
12. How is number-in-region matched: normalisation such as "1,250.00" against "1250.00", and fields holding several numbers? The SHACL rule is a plain substring test. (GPT, DeepSeek)
13. For helpers: what is the rejection-rate window, what defines the scope, how is "nothing outside scope changed" compared, and how are escapes detected? (GPT, Kimi)
14. For an inter-world query: record every hop, or only the first and the answering world? (Gemini, Opus)
15. What is the ISO 24617-2 mapping for the six acts? (GLM, Qwen, Opus, Kimi)
16. Which IRIs link EmotionML exports, and what is the hormone dimension set? (Qwen, Kimi, DeepSeek)
17. How do RO-Crate, nanopublication and STATO packages link to CheezML records? (Qwen, Kimi)
18. Where will the context be published (for example `https://w3id.org/cheezml/context.jsonld`), and what is the `owl:versionIRI` policy? (GLM, Kimi)
19. What are the CubeLang host bindings for MoWM (scoring, allocator, world creation, result reporting), and the exact ASK syntax? The program-writing agent needs these. (GPT, Gemini, Kimi, DeepSeek)
20. Is "weighted bundling" a second merge strategy or a description of `BindComposeMerge`? (synthesis)
21. Which check kinds do certificates actually record? (synthesis)
22. Is punning SKOS concepts as `rdfs:Class` through `dcterms:type` acceptable, or should CheezML mint a `cheez:kind`? (synthesis)

## 4. Claims in the answers that the brief does not support

- **Compilation.** GLM ("should pass `cubelang check` in strict mode"), Qwen ("in the executing subset"), Gemini ("restricted strictly to the executing safe subset") and Kimi all claimed their programs fit the compiler. All 12 failed at their first lines. DeepSeek also invented `RETURN` statements and `compare(...) == LESS` result codes.
- **Decision Ontology namespaces.** Gemini, GLM, Kimi and DeepSeek asserted IRIs the brief does not give. Gemini added that "the historical W3C Decision Incubator Group namespace is no longer resolvable", which the brief does not say.
- **Mixed spawn rules.** Gemini, GLM, Qwen, Kimi and DeepSeek combined router residual spawning (tau_spawn) with the bridge's floor and margin rule. Their bridge routing examples or programs carry a residual norm and tau_spawn, which the brief gives only for `router.py`. GLM typed a bridge-spawned world as a z=1 sub-world.
- **Invented values.** Gemini set a 10,000.00 second-approval amount, made-up reviewer roles (AP clerk, finance supervisor) and sandbox counts (50/49). Its DomainWorld definition cites "order 0068", which the brief does not mention.
- **ASK semantics.** Qwen's programs ASK when the residual does not exceed tau_spawn. The brief gives ASK only for a program that cannot choose, such as two cited dates.
- **z_max refusals.** Qwen: "if zDepth equals zMax ... no record is written". The brief says only that `spawn_subworld` refuses.
- **EmotionML as RDF.** Qwen treated EmotionML as an RDF vocabulary (`emotionml:Emotion`), and DeepSeek used `emotionml:pad-dimensions`. The brief cites EmotionML 1.0 as an export format.
- **Undeclared terms.** DeepSeek's shapes use `schema:truthScore`, `rocrate:WorkflowRun` and `cube:hasError` (error text, which is prose). GLM reused `schema:sha256`. Neither claim was checked against schema.org here, and none is needed.
- **Oja consolidation.** Gemini tied Oja consolidation to a world's centroid vector. The brief says only "Oja consolidation applies".
- **Kimi's own shapes.** Kimi's records do not conform to its own shapes (tool check), so its claim of minimal valid shapes is unsupported.

## 5. Grok 4.7 (arrived after the synthesis)

- **Tool check:** 8 JSON-LD blocks parse (350 triples), 1 Turtle block with 5 shapes, 2 CubeLang programs. Six `cheez:` terms (`blobHash`, `calibratedBy`, `calibrationData`, `calibrationDate`, `boundEncoder`, `boundCorpus`) are missing from its own context, so they expand to non-IRIs. Its records do not conform to its own shapes (9 closed-shape violations and 1 datatype violation).
- **Its two programs fail `cubelang check`** at line 2 (storage fields need `name: type`). They come closest of the panel (`program … implements IAgent`, `storage`, `functions`), but they also use `if x clears y` and `if x is two or more`, which are not CubeLang conditions. That makes 14 of 14 panel programs failing.
- **Agrees with the choices above:** calibration as its own node, no truth score or status, no delete property, no guessed Decision Ontology IRI, no container-bridge records, no opcodes in records.
- **New open questions it adds** (to §3): whether `world.tau` and `tau_spawn` are the same threshold (sharpens 4); whether the world-id counters pre- or post-increment (sharpens 5); whether CubeLang `forget` deletes; how a missing `query` is tested so that ASK can be written (sharpens 19); and what the stated defaults 0.5 and 100 were calibrated on, since they have no screen.

