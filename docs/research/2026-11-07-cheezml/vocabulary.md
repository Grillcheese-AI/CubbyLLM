# CheezML v0.1 vocabulary (draft synthesis, 2026-11-07)

Merged from seven panel answers (GLM 5.3, Qwen 3.8 max, Claude Opus 5.5, GPT-6.1 sol pro, Kimi K3.1 pro, DeepSeek V4 pro, Gemini 4 argon) under the brief's hard rules. Grok 4.7's answer came later and changes none of the choices (`disagreements.md` §5). Only the terms the six example records need, plus the controlled code lists. Not adopted; checked with rdflib, pyld and pySHACL (see `validation.txt`).

## What it does for CubbyLLM, in plain words

- **One record format for every part of the system.** A chat answer, a MoWM routing choice and an invoice field are all written the same way, so one checker can read any of them. Conventional models keep logs in a different shape for each component, or keep none.
- **Records say what actually ran.** A check that never ran leaves no record, so nothing can claim a "pass" it never had. A typical model's log just reports what the model says happened.
- **A certificate is a receipt, not a grade.** It lists the facts, rules and checks an answer came from. It never claims the answer is "true", and it never gives a confidence score.
- **Every threshold remembers where it came from.** A cut-off is stored with the test that set it, the data, the encoder and the date. You can tell when a number is being reused somewhere it was never tested; ordinary models keep their cut-offs as bare numbers in config files.
- **Nothing is deleted.** Rejected helpers, corrected facts and pruned worlds stay, with the reason. The history can always be replayed.

## Namespaces and identity

| Prefix | Namespace | Notes |
|---|---|---|
| `cheez:` | `https://w3id.org/cheezml/core#` | module ontology IRI `https://w3id.org/cheezml/core`; versions via `owl:versionIRI` (e.g. `https://w3id.org/cheezml/core/v0.1`), never in the namespace |
| `cubby:` | `https://w3id.org/cheezml/cubby#` | same pattern |
| `mowm:` | `https://w3id.org/cheezml/mowm#` | same pattern |
| `cube:` | `https://w3id.org/cheezml/cubelang#` | same pattern |
| `prov:` `schema:` `time:` `skos:` `dcterms:` `xsd:` `sh:` | `http://www.w3.org/ns/prov#`, `https://schema.org/`, `http://www.w3.org/2006/time#`, `http://www.w3.org/2004/02/skos/core#`, `http://purl.org/dc/terms/`, `http://www.w3.org/2001/XMLSchema#`, `http://www.w3.org/ns/shacl#` | reused as published |

Records and instances (hosts, runs, worlds, reviewers) have `urn:uuid:` ids, never ids in a vocabulary namespace. A world's numeric id is the separate literal `mowm:worldId`.

"Panel support" counts how many of the 7 answers proposed an equivalent term (same role, any name). Datatypes: scores, thresholds and observed values are `xsd:decimal`; ids, depths and counts are `xsd:integer`; times are `xsd:dateTime` (PROV) or `xsd:dateTimeStamp` (OWL-Time).

## core (`cheez:`)

| Term | Kind | Definition | Domain | Range | Reuse / alignment | Support |
|---|---|---|---|---|---|---|
| `cheez:Record` | class | Root node of a document the host wrote from what ran. | | | marker class; deliberately not ⊑ `prov:Entity`, since activities are records too | 7 |
| `cheez:Host` | class | The host process that writes every record. | | | ⊑ `prov:SoftwareAgent` | 3 |
| `cheez:writtenBy` | property | The host that wrote this record. | `cheez:Record` | `cheez:Host` | no single PROV term covers both entities and activities | 4 |
| `cheez:Artifact` | class | External payload known only by its hash: program, vector, weights, data, document, encoder table. | | | ⊑ `prov:Entity` | 7 |
| `cheez:hash` | property | Content hash; the bytes are never embedded. | `cheez:Artifact` | `xsd:string` | minted: no standard hash property verified | 7 |
| `cheez:hashAlgorithm` | property | Algorithm of `cheez:hash` (not given in the brief). | `cheez:Artifact` | `xsd:string` | | 3 |
| `cheez:Rule` | class | A rule applied by a program run, by hash. | | | ⊑ `cheez:Artifact`, `prov:Plan` | 6 |
| `cheez:FactVersion` | class | One version of a stored statement, with both clocks; a correction closes it and never overwrites it. | | | ⊑ `prov:Entity`; recorded time = `prov:generatedAtTime`, closure = `prov:invalidatedAtTime`, successor `prov:wasRevisionOf` | 7 |
| `cheez:validTime` | property | Clock 1: when the statement held. | `cheez:FactVersion` | `time:Interval` | OWL-Time range | 7 |
| `cheez:Threshold` | class | A decision boundary with its calibration. Never a score. | | | ⊑ `prov:Entity`; `prov:wasGeneratedBy` its `cheez:Calibration`; parameter via `dcterms:type` | 7 |
| `cheez:thresholdValue` | property | The boundary value. | `cheez:Threshold` | `xsd:decimal` | | 7 |
| `cheez:Calibration` | class | The screen run that set one or more thresholds for one encoder, data set and deployment, on a date (`prov:endedAtTime`). | | | ⊑ `prov:Activity` | 5 |
| `cheez:screen` | property | The screen that set the threshold. | `cheez:Calibration` | `cheez:Artifact` | ⊑ `prov:used` | 7 |
| `cheez:calibrationData` | property | Data (corpus) the screen ran on. | `cheez:Calibration` | `cheez:Artifact` | ⊑ `prov:used` | 7 |
| `cheez:encoder` | property | Exact encoder an activity used; a threshold holds only for its calibration's encoder. | `prov:Activity` | `cheez:Artifact` | ⊑ `prov:used` | 5 |
| `cheez:deployment` | property | Deployment a calibration holds for. | `cheez:Calibration` | `prov:Entity` | | 2 |
| `cheez:CheckRun` | class | One execution of a check. It exists only if the check ran. Kind via `dcterms:type`. | | | ⊑ `prov:Activity` | 5 |
| `cheez:check` | property | A check that ran for this record. | `prov:Activity` or `cheez:Certificate` | `cheez:CheckRun` | | 4 |
| `cheez:outcome` | property | Observed, coded outcome of a check, run, review or routing decision. | `prov:Activity` or `prov:Entity` | `skos:Concept` | | 6 |
| `cheez:observedValue` | property | A measured score or count; never a threshold. | `cheez:CheckRun` | `xsd:decimal` | | 3 |
| `cheez:threshold` | property | The threshold a check compared against. | `cheez:CheckRun` | `cheez:Threshold` | | 2 |
| `cheez:Certificate` | class | VM-produced record that an answer follows from the facts and rules used, under the recorded checks. Not a claim of correctness, never a truth score. | | | ⊑ `prov:Entity` | 7 |
| `cheez:certifies` | property | The answer a certificate covers. | `cheez:Certificate` | `cubby:Answer` | | 7 |
| `cheez:Question` | class | The question a decision is about. | | | ⊑ `prov:Entity`; `skos:closeMatch` `dec:Question` | 6 |
| `cheez:Option` | class | A candidate: one proposer's program. | | | ⊑ `prov:Entity`; `skos:closeMatch` `dec:Option` | 6 |
| `cheez:Decision` | class | The answer or the refusal. | | | ⊑ `prov:Entity`; `skos:closeMatch` `dec:Decision` | 6 |
| `cheez:option` | property | An option considered for the question. | `cheez:Question` | `cheez:Option` | `skos:closeMatch` `dec:involves_choosing` (direction unverified) | 5 |
| `cheez:hasResult` | property | The decision reached on the question. | `cheez:Question` | `cheez:Decision` | `skos:closeMatch` `dec:has_result` (direction unverified) | 4 |
| `cheez:chosenOption` | property | The option the decision took. | `cheez:Decision` | `cheez:Option` | | 2 |
| `cheez:isSatisfiedBy` | property | This criterion (check run) is satisfied by the option. | `cheez:CheckRun` | `cheez:Option` | brief's `is_satisfied_by`; absent from the published Decision Ontology, so minted with no alignment | 6 |
| `cheez:isViolatedBy` | property | This criterion (check run) is violated by the option. | `cheez:CheckRun` | `cheez:Option` | brief's `is_violated_by`; same note | 6 |
| `cheez:Template` | class | A sentence template, by hash. | | | ⊑ `cheez:Artifact` | 3 |
| `cheez:RenderedText` | class | A readable sentence produced by a template from a coded record: the only place prose may appear. | | | ⊑ `prov:Entity`; text in `schema:text`, source in `prov:wasDerivedFrom` | 3 |
| `cheez:template` | property | The template that produced the text. | `cheez:RenderedText` | `cheez:Template` | | 2 |
| `cheez:Retirement` | class | Retiring, rejecting or pruning something. The thing stays and is `prov:invalidated` by this activity. Kind via `dcterms:type`. | | | ⊑ `prov:Activity` | 7 |
| `cheez:retirementReason` | property | Coded reason for a retirement. | `cheez:Retirement` | `skos:Concept` | | 7 |

The Decision Ontology as published puts its terms in `http://example.com/ns/dec#` (prefix `dec:`), a placeholder namespace. CheezML therefore mints its own terms and records the published ones only as `skos:closeMatch` targets.

## cubby (`cubby:`)

| Term | Kind | Definition | Domain | Range | Reuse / alignment | Support |
|---|---|---|---|---|---|---|
| `cubby:Turn` | class | One conversation turn, written by the host; conversation via `schema:isPartOf`. | | | ⊑ `prov:Entity` | 7 |
| `cubby:dialogueAct` | property | The turn's dialogue act. | `cubby:Turn` | `skos:Concept` (`cubby:DialogueActs`) | ISO 24617-2 mapping pending (IRIs not known) | 7 |
| `cubby:turnIndex` | property | Host-assigned position in the conversation. | `cubby:Turn` | `xsd:integer` | | 3 |
| `cubby:decision` | property | The answer or refusal the turn delivered. | `cubby:Turn` | `cheez:Decision` | | 3 |
| `cubby:Answer` | class | A delivered answer (value in `prov:value`); always has a certificate. | | | ⊑ `cheez:Decision` | 6 |
| `cubby:Refusal` | class | A refusal decision with a coded reason. | | | ⊑ `cheez:Decision` | 7 |
| `cubby:refusalReason` | property | Why the question was refused. | `cubby:Refusal` | `skos:Concept` (`cubby:RefusalReasons`) | | 7 |
| `cubby:conflictKind` | property | Kind of conflict; required exactly when the reason is `cubby:Conflict`. | `cubby:Refusal` | `skos:Concept` (`cubby:ConflictKinds`) | | 6 |
| `cubby:proposer` | property | The proposer behind an answer, refusal, option or helper. | `cubby:Answer`, `cubby:Refusal`, `cheez:Option`, `cubby:Helper` | `skos:Concept` (`cubby:Proposers`) | | 7 |
| `cubby:Helper` | class | A helper program proposed by Cubby's own proposers (order 0062). | | | ⊑ `cube:Program` | 7 |
| `cubby:scope` | property | The verified answers the helper must reproduce. | `cubby:Helper` | `prov:Collection` | | 2 |
| `cubby:SandboxRun` | class | A run of a helper in a locked-down VM session; its gates are `cheez:check` runs. | | | ⊑ `cube:ProgramRun` | 6 |
| `cubby:HelperRejection` | class | The host's rejection of a helper after its sandbox run; reason = the failed gate's kind. | | | ⊑ `cheez:Retirement` | 7 |
| `cubby:breakingCase` | property | The case that broke the helper. | `cubby:HelperRejection` | `prov:Entity` | | 7 |
| `cubby:DocumentRegion` | class | A region of a document: text as read (`schema:text`), reader version, document (`schema:isPartOf`). | | | ⊑ `prov:Entity` | 7 |
| `cubby:readerVersion` | property | Version of the reader that produced the region. | `cubby:DocumentRegion` | `xsd:string` | | 7 |
| `cubby:FieldSuggestion` | class | A value read or derived for a document field (`prov:value`); a suggestion until committed. | | | ⊑ `prov:Entity` | 7 |
| `cubby:region` | property | The region a suggestion or review refers to. | `cubby:FieldSuggestion`, `cubby:Review` | `cubby:DocumentRegion` | | 7 |
| `cubby:Review` | class | A human review of a suggestion; reviewer and role via `prov:qualifiedAssociation`. | | | ⊑ `prov:Activity` | 7 |
| `cubby:FieldCommit` | class | Host commit turning an approved suggestion into a `cheez:FactVersion` whose source is the reviewer. | | | ⊑ `prov:Activity` | 3 |

## mowm (`mowm:`)

| Term | Kind | Definition | Domain | Range | Reuse / alignment | Support |
|---|---|---|---|---|---|---|
| `mowm:World` | class | A world tree node (`world.py`). | | | ⊑ `prov:Entity` | 7 |
| `mowm:worldId` | property | Counter-issued numeric id: world counter from 1000, router counter from 5000. Never computed from the parent's id. | `mowm:World` | `xsd:integer` | | 7 |
| `mowm:zDepth` | property | Depth z on the Z axis. | `mowm:World` | `xsd:integer` | | 7 |
| `mowm:zMax` | property | Maximum depth; `spawn_subworld` refuses at z_max. | `mowm:World` | `xsd:integer` | | 7 |
| `mowm:parentWorld` | property | The world this sub-world was split from. | `mowm:World` | `mowm:World` | specialises `prov:wasDerivedFrom` | 7 |
| `mowm:tau` | property | The world's confidence threshold for spawning sub-worlds. | `mowm:World` | `cheez:Threshold` | | 5 |
| `mowm:mergeStrategy` | property | How the world's HYLA transition deltas are merged. | `mowm:World` | `skos:Concept` (`mowm:MergeStrategies`) | | 7 |
| `mowm:WordTableEncoder` | class | The word-table text encoder (FastWordEncoder), by the hash of its table; the only encoder allowed on the routing path. | | | ⊑ `cheez:Artifact` | 7 |
| `mowm:RoutingDecision` | class | One bridge routing decision (`bridges/cubby_bridge.py`): route or spawn. Mode via `dcterms:type`. | | | ⊑ `prov:Activity` | 7 |
| `mowm:queryVector` | property | The encoded query, by hash. | `mowm:RoutingDecision` | `cheez:Artifact` | | 2 |
| `mowm:worldCount` | property | Worlds in the pool when the decision ran. | `mowm:RoutingDecision` | `xsd:integer` | | 2 |
| `mowm:bestWorld` | property | The highest-scoring world. | `mowm:RoutingDecision` | `mowm:World` | | 2 |
| `mowm:bestScore` | property | Its score (a score, not a threshold). | `mowm:RoutingDecision` | `xsd:decimal` | | 7 |
| `mowm:runnerUpWorld` | property | The second-highest world. | `mowm:RoutingDecision` | `mowm:World` | | 2 |
| `mowm:runnerUpScore` | property | Its score. | `mowm:RoutingDecision` | `xsd:decimal` | | 2 |
| `mowm:margin` | property | Best score minus runner-up score. | `mowm:RoutingDecision` | `xsd:decimal` | | 6 |
| `mowm:floorThreshold` | property | The calibrated absolute floor compared with the best score. | `mowm:RoutingDecision` | `cheez:Threshold` | | 5 |
| `mowm:marginThreshold` | property | The calibrated `tau_margin` compared with the margin. | `mowm:RoutingDecision` | `cheez:Threshold` | | 4 |
| `mowm:routedTo` | property | The world routed to (outcome Route only). | `mowm:RoutingDecision` | `mowm:World` | | 5 |
| `mowm:WorldSpawn` | class | A spawn that ran and generated one new world (`prov:generated`); `prov:wasInformedBy` the decision. | | | ⊑ `prov:Activity` | 7 |
| `mowm:WorldSplit` | class | Creation of a sub-world at z+1 from a parent (`prov:used`) to handle a residual. | | | ⊑ `prov:Activity` | 7 |
| `mowm:residual` | property | The residual the child handles, by hash. | `mowm:WorldSplit` | `cheez:Artifact` | | 0 (residual norm: 6) |

## cubelang (`cube:`)

| Term | Kind | Definition | Domain | Range | Reuse / alignment | Support |
|---|---|---|---|---|---|---|
| `cube:Program` | class | A CubeLang program, by hash. | | | ⊑ `cheez:Artifact`, `prov:Plan` | 7 |
| `cube:ProgramRun` | class | One VM execution; outcome `cube:Return`, `cube:Error` or `cube:Ask` (suspended, resumed by re-execution over `run-proto`). | | | ⊑ `prov:Activity` | 7 |
| `cube:VM` | class | The CubeLang VM: the only agent whose runs can produce a certificate. | | | ⊑ `prov:SoftwareAgent` | 2 |

## Standard terms used directly (not minted)

| Term | Used for | Support |
|---|---|---|
| `prov:Entity`, `prov:Activity`, `prov:SoftwareAgent`, `prov:Person`, `prov:Collection`, `prov:Association` | base classes | 7 |
| `prov:used`, `prov:generated`, `prov:wasGeneratedBy`, `prov:wasInformedBy`, `prov:wasAssociatedWith`, `prov:wasDerivedFrom` | what ran, on what, producing what | 7 |
| `prov:hadPrimarySource` | source of a fact version (a source with `schema:version` = revision, or the reviewer) | 3 |
| `prov:generatedAtTime`, `prov:invalidatedAtTime`, `prov:wasRevisionOf` | clock 2 (recorded time), closing a version, its successor | 1 (6 minted a `recordedTime` instead) |
| `prov:startedAtTime`, `prov:endedAtTime` | run times; a calibration's date | 7 |
| `prov:invalidated` | a retirement retires the thing (the thing stays) | 4 |
| `prov:qualifiedAssociation`, `prov:agent`, `prov:hadRole` | reviewer by id and role | 3 |
| `prov:value` | answer value, field value | 3 |
| `time:Interval`, `time:Instant`, `time:hasBeginning`, `time:hasEnd`, `time:inXSDDateTimeStamp` | clock 1 (valid time) | 5 |
| `schema:text` | region text as read; rendered sentence | 3 |
| `schema:version` | revision of a source | 2 |
| `schema:isPartOf`, `schema:about`, `schema:name`, `schema:identifier`, `schema:DigitalDocument` | conversation, document, labels, external ids | 5 |
| `dcterms:type` | kind codes: check kind, routing mode, threshold parameter, retirement kind (note: DCMI gives it range `rdfs:Class`) | 1 |

## Controlled code lists (SKOS)

| Scheme | Codes | Used by | Support |
|---|---|---|---|
| `cubby:DialogueActs` | QuestionAct, CertifiedAnswerAct, RefusalAct, ClarifyingQuestionAct, FollowUpAct, SmallTalkAct | `cubby:dialogueAct` | 7 |
| `cubby:RefusalReasons` | NoPlan, RelationMissing, FactMissing, WalkFailed, NotCertified, Ambiguous, Conflict, DatesTooCoarse, UnreadableSource | `cubby:refusalReason` | 7 |
| `cubby:ConflictKinds` | ValueConflict, TypeConflict, TemporalConflict, LogicalConflict, RecordedVsDerivedConflict | `cubby:conflictKind` | 6 |
| `cubby:Proposers` | HostGrammar, EverydayTable, OvernightRuleLibrary, EpisodicMemory, LMProgramEmitter | `cubby:proposer` | 7 |
| `cheez:CheckKinds` | 20 kinds, see Turtle below | `dcterms:type` on `cheez:CheckRun`; `cheez:retirementReason` | 6 |
| `cheez:CheckOutcomes` | Satisfied, Violated | `cheez:outcome` on a check | 4 |
| `cheez:RetirementKinds` | Retired, Rejected, Pruned | `dcterms:type` on `cheez:Retirement` | 4 |
| `cube:RunOutcomes` | Return, Error, Ask | `cheez:outcome` on a program run | 6 |
| `mowm:RoutingOutcomes` | Route, Spawn | `cheez:outcome` on a routing decision | 7 |
| `mowm:RoutingModes` | CentroidRouting, BestMemberRouting | `dcterms:type` on a routing decision | 3 |
| `mowm:Parameters` | AbsoluteFloor, TauMargin, TauAnswer, TauSpawn, Tau, TauPruneSteps | `dcterms:type` on a threshold | 2 |
| `mowm:MergeStrategies` | BindComposeMerge | `mowm:mergeStrategy` | 7 |
| `cubby:ReviewOutcomes` | Approved | `cheez:outcome` on a review | 3 |
| `cubby:ReviewerRoles` | FirstApprover, SecondApprover | `prov:hadRole` | 4 |

Check kinds `cubby:ConflictCheck` and `cubby:DatePrecisionCheck` are inferred from the refusal reasons (conflict, dates too coarse); confirm them against the host's real check list. The brief names no other review outcome, reviewer role or merge strategy, so none is added.

```turtle
@prefix skos:  <http://www.w3.org/2004/02/skos/core#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix mowm:  <https://w3id.org/cheezml/mowm#> .
@prefix cube:  <https://w3id.org/cheezml/cubelang#> .
@prefix dec:   <http://example.com/ns/dec#> .

# Alignment to the Decision Ontology as published (placeholder namespace, so never used as main terms)
cheez:Question skos:closeMatch dec:Question .
cheez:Option   skos:closeMatch dec:Option .
cheez:Decision skos:closeMatch dec:Decision .
cheez:hasResult skos:closeMatch dec:has_result .
cheez:option    skos:closeMatch dec:involves_choosing .

cubby:DialogueActs a skos:ConceptScheme ; skos:prefLabel "Dialogue acts (order 0059)"@en .
cubby:QuestionAct           a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "question"@en .
cubby:CertifiedAnswerAct    a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "certified answer"@en .
cubby:RefusalAct            a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "refusal (with reason)"@en .
cubby:ClarifyingQuestionAct a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "clarifying question"@en .
cubby:FollowUpAct           a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "follow-up (with its rewrite)"@en .
cubby:SmallTalkAct          a skos:Concept ; skos:inScheme cubby:DialogueActs ; skos:prefLabel "small talk"@en .

cubby:RefusalReasons a skos:ConceptScheme ; skos:prefLabel "Refusal reasons in use"@en .
cubby:NoPlan           a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "no plan"@en .
cubby:RelationMissing  a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "relation missing"@en .
cubby:FactMissing      a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "fact missing"@en .
cubby:WalkFailed       a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "walk failed"@en .
cubby:NotCertified     a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "not certified"@en .
cubby:Ambiguous        a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "ambiguous"@en .
cubby:Conflict         a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "conflict (kind in cubby:conflictKind)"@en .
cubby:DatesTooCoarse   a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "dates too coarse"@en .
cubby:UnreadableSource a skos:Concept ; skos:inScheme cubby:RefusalReasons ; skos:prefLabel "unreadable source (order 0061)"@en .

cubby:ConflictKinds a skos:ConceptScheme ; skos:prefLabel "Kinds of conflict"@en .
cubby:ValueConflict             a skos:Concept ; skos:inScheme cubby:ConflictKinds ; skos:prefLabel "value"@en .
cubby:TypeConflict              a skos:Concept ; skos:inScheme cubby:ConflictKinds ; skos:prefLabel "type"@en .
cubby:TemporalConflict          a skos:Concept ; skos:inScheme cubby:ConflictKinds ; skos:prefLabel "temporal"@en .
cubby:LogicalConflict           a skos:Concept ; skos:inScheme cubby:ConflictKinds ; skos:prefLabel "logical"@en .
cubby:RecordedVsDerivedConflict a skos:Concept ; skos:inScheme cubby:ConflictKinds ; skos:prefLabel "recorded against derived"@en .

cubby:Proposers a skos:ConceptScheme ; skos:prefLabel "Proposers in use"@en .
cubby:HostGrammar          a skos:Concept ; skos:inScheme cubby:Proposers ; skos:prefLabel "host grammar"@en .
cubby:EverydayTable        a skos:Concept ; skos:inScheme cubby:Proposers ; skos:prefLabel "everyday table"@en .
cubby:OvernightRuleLibrary a skos:Concept ; skos:inScheme cubby:Proposers ; skos:prefLabel "overnight rule library"@en .
cubby:EpisodicMemory       a skos:Concept ; skos:inScheme cubby:Proposers ; skos:prefLabel "episodic memory of certified chains"@en .
cubby:LMProgramEmitter     a skos:Concept ; skos:inScheme cubby:Proposers ; skos:prefLabel "language model's program emitter"@en .

cheez:CheckKinds a skos:ConceptScheme ; skos:prefLabel "Check kinds"@en .
cubby:ConflictCheck       a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "No value, type, temporal, logical or recorded-against-derived conflict among the statements used (inferred from the refusal reasons)."@en .
cubby:DatePrecisionCheck  a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Dates used are fine enough for the question (inferred from 'dates too coarse')."@en .
cubby:EntityGuardCheck    a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A follow-up rewrite passes the entity guard (0059)."@en .
cubby:PrerequisiteCheck   a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact prerequisite check of a story-world action (0060)."@en .
cubby:ToneCheck           a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Soft (tone) check of a story-world action, labelled soft (0060)."@en .
cubby:NumberInRegionCheck a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Every number in a field appears in its region's text (0061)."@en .
cubby:FormatCheck         a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact format check of a document field (0061)."@en .
cubby:ArithmeticCheck     a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact arithmetic check (0061)."@en .
cubby:PurchaseOrderCheck  a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact purchase-order check (0061)."@en .
cubby:SupplierCheck       a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact supplier check (0061)."@en .
cubby:DuplicateCheck      a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Exact duplicate check (0061)."@en .
cubby:ReproductionCheck   a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A helper reproduces every verified answer in its scope (0062)."@en .
cubby:ScopeCheck          a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A helper changes nothing outside its scope (0062)."@en .
cubby:EscapeCheck         a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Escape attempts observed in the sandbox; must be 0 (0062)."@en .
cubby:RejectionRateCheck  a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A proposer's helper rejection rate over a window (0062)."@en .
cubby:NoNewWrongAnswerCheck a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A domain change introduces no new wrong answer (0058)."@en .
cubby:GainCheck           a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "A domain change shows a gain (0058)."@en .
mowm:ZMaxCheck            a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "The parent world's depth is below z_max, so spawn_subworld may split."@en .
mowm:AnswerSupportCheck   a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "Support is at or above tau_answer (possibility oracle; inter-world query otherwise)."@en .
mowm:ResidualCheck        a skos:Concept ; skos:inScheme cheez:CheckKinds ; skos:definition "The residual norm exceeds tau_spawn (router.py maybe_spawn)."@en .

cheez:CheckOutcomes a skos:ConceptScheme ; skos:prefLabel "Check outcomes"@en .
cheez:Satisfied a skos:Concept ; skos:inScheme cheez:CheckOutcomes ; skos:prefLabel "satisfied"@en .
cheez:Violated  a skos:Concept ; skos:inScheme cheez:CheckOutcomes ; skos:prefLabel "violated"@en .

cheez:RetirementKinds a skos:ConceptScheme ; skos:prefLabel "Retirement kinds"@en .
cheez:Retired  a skos:Concept ; skos:inScheme cheez:RetirementKinds ; skos:prefLabel "retired"@en .
cheez:Rejected a skos:Concept ; skos:inScheme cheez:RetirementKinds ; skos:prefLabel "rejected"@en .
cheez:Pruned   a skos:Concept ; skos:inScheme cheez:RetirementKinds ; skos:prefLabel "pruned"@en .

cube:RunOutcomes a skos:ConceptScheme ; skos:prefLabel "VM execution outcomes"@en .
cube:Return a skos:Concept ; skos:inScheme cube:RunOutcomes ; skos:prefLabel "return"@en .
cube:Error  a skos:Concept ; skos:inScheme cube:RunOutcomes ; skos:prefLabel "error"@en .
cube:Ask    a skos:Concept ; skos:inScheme cube:RunOutcomes ; skos:prefLabel "ASK (suspended; resumed by re-execution over run-proto)"@en .

mowm:RoutingOutcomes a skos:ConceptScheme ; skos:prefLabel "Routing outcomes"@en .
mowm:Route a skos:Concept ; skos:inScheme mowm:RoutingOutcomes ; skos:prefLabel "route"@en .
mowm:Spawn a skos:Concept ; skos:inScheme mowm:RoutingOutcomes ; skos:prefLabel "spawn"@en .

mowm:RoutingModes a skos:ConceptScheme ; skos:prefLabel "Routing modes"@en .
mowm:CentroidRouting   a skos:Concept ; skos:inScheme mowm:RoutingModes ; skos:prefLabel "by centroid"@en .
mowm:BestMemberRouting a skos:Concept ; skos:inScheme mowm:RoutingModes ; skos:prefLabel "by best member"@en .

mowm:Parameters a skos:ConceptScheme ; skos:prefLabel "Threshold parameters"@en .
mowm:AbsoluteFloor a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "absolute floor (bridge)"@en .
mowm:TauMargin     a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "tau_margin"@en .
mowm:TauAnswer     a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "tau_answer"@en .
mowm:TauSpawn      a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "tau_spawn"@en .
mowm:Tau           a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "tau (world spawn confidence)"@en .
mowm:TauPruneSteps a skos:Concept ; skos:inScheme mowm:Parameters ; skos:prefLabel "tau_prune_steps"@en .

mowm:MergeStrategies a skos:ConceptScheme ; skos:prefLabel "HYLA merge strategies"@en .
mowm:BindComposeMerge a skos:Concept ; skos:inScheme mowm:MergeStrategies ; skos:prefLabel "BindComposeMerge"@en .

cubby:ReviewOutcomes a skos:ConceptScheme ; skos:prefLabel "Review outcomes"@en .
cubby:Approved a skos:Concept ; skos:inScheme cubby:ReviewOutcomes ; skos:prefLabel "approved"@en .

cubby:ReviewerRoles a skos:ConceptScheme ; skos:prefLabel "Reviewer roles"@en .
cubby:FirstApprover  a skos:Concept ; skos:inScheme cubby:ReviewerRoles ; skos:prefLabel "first approver"@en .
cubby:SecondApprover a skos:Concept ; skos:inScheme cubby:ReviewerRoles ; skos:prefLabel "second approver (above the amount)"@en .
```

## Fixture conventions (examples)

- Ids are `urn:uuid:00000000-0000-0000-0000-000000000NNN`, where the first digit of NNN is the record number. The host is `...001`, the VM `...002` and the conversation `...003`.
- Values the brief does not give are strings starting with `PLACEHOLDER` (hashes, revisions, names, text). They never appear inside an IRI.
- Typed slots (scores, thresholds, ids, depths, times) cannot hold a placeholder string without failing validation. They therefore hold **illustrative values, not measurements**. Only these come from the brief: the counter starts (1000, 5000), FastWordEncoder, BindComposeMerge and the HYLA encoder. The 0.680 / 0.589 scores and the 92.4% / 86.9% rates are never used as thresholds.
- Each example embeds the full `@context` of `context.jsonld` inline.
