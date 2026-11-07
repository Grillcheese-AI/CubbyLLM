# Panel: CheezML, one record vocabulary (2026-11-07)

The brief: `2026-11-07-cheezml-brief.md` (beside this file). Asked independently, same prompt, temperature 0.3; budgets per model as in `validation/panel_cheezml.py`. Model ids checked against OpenRouter's live list at run time. Cached under `standin/data/out/panel_cache/`.

**These are model opinions, not findings.** Nothing here is adopted until the synthesis has expanded every JSON-LD, validated the records against the SHACL shapes and compiled every program.

Cost reported by OpenRouter: $2.41. Answers: 8 of 8.

## openai/gpt-6.1-sol-pro

*26949 tokens out, 170.0 s, finish `stop`*

## 1. Classes and properties

This is a vocabulary proposal, not a finding about an implementation.

Three limits affect the deliverables:

- The brief does not give the Decision Ontology’s namespace or the exact IRIs for its relations. I do not know them. I preserve its required structure below, but do not invent those IRIs.
- The brief does not give enough CubeLang grammar or a callable MoWM interface to write a reliably compilable integration. The programs in section 5 are explicitly **spec only: does not compile today**.
- A vocabulary and SHACL cannot establish that an operation actually ran. Host-controlled collection and validation of execution evidence remain necessary.

### 1.1 Namespaces and identity

| Prefix | Namespace |
|---|---|
| `cheez:` | `https://w3id.org/cheezml/core#` |
| `cubby:` | `https://w3id.org/cheezml/cubby#` |
| `mowm:` | `https://w3id.org/cheezml/mowm#` |
| `cube:` | `https://w3id.org/cheezml/cubelang#` |

Record identities are `urn:uuid:...`, including identities for world records. A world’s numeric implementation identifier is a separate property.

Use `owl:versionIRI` on each ontology module. No namespace contains a version number. The eventual redirect does not change these identifiers.

### 1.2 Reuse first

Use these terms directly rather than minting equivalents:

| Standard terms | Domain → range in this design | Use |
|---|---|---|
| `prov:Entity`, `prov:Activity`, `prov:SoftwareAgent`, `prov:Person` | Classes | Data, execution, software and people. |
| `prov:used` | Activity → Entity | An actual execution input. |
| `prov:wasGeneratedBy` | Entity → Activity | Actual producing execution. |
| `prov:wasDerivedFrom` | Entity → Entity | Derivation, not certification. |
| `prov:wasRevisionOf` | Entity → Entity | New version of an existing entity. |
| `prov:wasAssociatedWith` | Activity → Agent | Agent participating in execution. |
| `prov:wasAttributedTo` | Entity → Agent | Attribution, distinct from the host writing its record. |
| `prov:startedAtTime`, `prov:endedAtTime` | Activity → `xsd:dateTime` | Observed execution times. |
| `prov:hadPrimarySource` | Entity → Entity | Source used for a statement or reading. |
| `prov:qualifiedAssociation`, `prov:agent`, `prov:hadRole` | PROV qualified association pattern | Reviewer, proposer and other roles. |
| `schema:identifier`, `schema:version`, `schema:text` | Relevant entity → literal | External identifier, source revision and observed source text. |
| `schema:hasPart`, `schema:isPartOf` | Relevant entity → entity | Certificates, question parts, regions and package contents. |
| `schema:encodingFormat` | Artifact → literal | Artifact media type. |
| `time:Interval`, `time:Instant` | Classes | Temporal extent and boundary. |
| `time:hasBeginning`, `time:hasEnd` | Interval → Instant | Interval boundaries. |
| `time:inXSDDateTimeStamp` | Instant → `xsd:dateTimeStamp` | Timestamp with timezone. |
| `skos:Concept`, `skos:ConceptScheme` | Classes | Controlled codes and their schemes. |
| `skos:inScheme`, `skos:prefLabel`, `skos:notation` | SKOS resources → scheme or literal | Code membership and presentation. |
| `owl:versionIRI` | Ontology → IRI | Ontology release identity. |

Definitions, code labels and source-region text are not model-written rationales. Human-readable operational messages must be rendered from codes, values and a separately versioned template.

Use Workflow Run RO-Crate for evaluation packages, nanopublications for hypothesis verdicts—including killed results—and STATO for statistical descriptions. I do not know the exact profile terms required by the deployments.

EmotionML 1.0 is an export linked through an artifact, not a new CheezML emotion serialization.

### 1.3 New classes

A domain below is the intended use domain, not an instruction to infer that every resource is a record. `cheez:Record` is not a subclass of `prov:Entity`: activity records must not thereby become entities.

| Class | Module | Superclass/reuse | Definition |
|---|---|---|---|
| `cheez:Record` | core | `owl:Thing` | A host-written record of observed data or execution. |
| `cheez:Artifact` | core | `prov:Entity` | A hash-addressed external payload; its bytes are not embedded. |
| `cheez:Assertion` | core | `prov:Entity` | One immutable version of a stored statement with both temporal extents. |
| `cheez:Certificate` | core | `prov:Entity` | VM-produced evidence of derivability under recorded inputs, rules and checks. |
| `cheez:CheckRun` | core | `prov:Activity` | One actual execution of a named check. |
| `cheez:Measurement` | core | `prov:Entity` | An observed value for a named metric. |
| `cheez:Threshold` | core | `prov:Entity` | A decision boundary produced by a recorded calibration. |
| `cheez:Calibration` | core | `prov:Activity` | A screen execution establishing a deployment-specific threshold. |
| `cheez:LifecycleChange` | core | `prov:Activity` | An observed retirement, rejection, pruning or promotion event. |
| `cubby:Turn` | cubby | `prov:Entity` | A conversation turn with a coded dialogue act. |
| `cubby:ConversationState` | cubby | `prov:Entity` | A host-maintained conversation-state version. |
| `cubby:EmotionInference` | cubby | `prov:Entity` | An uncertain inference about a user’s emotion. |
| `cubby:HormoneState` | cubby | `prov:Entity` | Cubby’s own separately labelled dimension state. |
| `cubby:DocumentRegion` | cubby | `prov:Entity` | Source-region text associated with a reader version. |
| `cubby:FieldSuggestion` | cubby | `prov:Entity` | A proposed field value, not a committed fact. |
| `cubby:Review` | cubby | `prov:Activity` | A human review of a specified region and suggestion. |
| `cubby:Helper` | cubby | `cube:Program` | A candidate helper subject to sandbox and adoption checks. |
| `cubby:DomainWorld` | cubby | `prov:Entity` | A domain package containing ontology, open records, rules and questions. |
| `mowm:World` | mowm | `prov:Entity` | A recorded world-node version, not the numeric world identifier. |
| `mowm:RoutingDecision` | mowm | `prov:Entity` | The recorded result of a routing policy execution. |
| `mowm:WorldChange` | mowm | `prov:Activity` | An actual spawn, split, consolidation or pruning operation. |
| `cube:Program` | cubelang | `cheez:Artifact` | A hash-addressed program artifact. |
| `cube:VMRun` | cubelang | `prov:Activity` | An actual VM execution, including suspended executions. |

No new classes named Question, Option or Decision are proposed. Those belong to the Decision Ontology. Until its IRIs are supplied, examples identify those roles through controlled codes; that is an explicit incomplete binding, not an equivalent ontology.

### 1.4 New properties

`Resource` means `rdfs:Resource`; `Literal` means `rdfs:Literal`. Classes with `Record` metadata should also be explicitly typed `cheez:Record`.

| Property | Module | Domain → range | Definition and reuse |
|---|---|---|---|
| `cheez:writer` | core | Record → `prov:SoftwareAgent` | Host software that wrote the record; distinct from PROV attribution. |
| `cheez:kind` | core | Resource → `skos:Concept` | Controlled category; uses SKOS rather than free-text categories. |
| `cheez:outcome` | core | Resource → `skos:Concept` | Recorded execution or decision outcome, never an unobserved declaration. |
| `cheez:reason` | core | Resource → `skos:Concept` | Controlled reason code, not explanatory prose. |
| `cheez:subject` | core | Resource → Resource | Resource about which a measurement, decision or lifecycle event is recorded. |
| `cheez:evidence` | core | Resource → Resource | Supporting recorded evidence; does not itself confer certification. |
| `cheez:metric` | core | Measurement or Threshold → `skos:Concept` | Metric whose value or boundary is recorded. |
| `cheez:value` | core | Measurement, Threshold or FieldSuggestion → Literal | Literal value; has no truth-score interpretation. |
| `cheez:threshold` | core | Resource → Threshold | Threshold actually used by an execution or decision. |
| `cheez:screen` | core | Calibration → Artifact | Hash-addressed calibration screen. |
| `cheez:dataset` | core | Calibration → Artifact | Calibration data artifact. |
| `cheez:encoder` | core | Calibration or Activity → Artifact | Exact encoder artifact used. |
| `cheez:corpus` | core | Calibration → Artifact | Corpus identity/version artifact defining calibration scope. |
| `cheez:deployment` | core | Calibration → Resource | Deployment for which the calibration applies. |
| `cheez:hash` | core | Artifact → `xsd:string` | Content digest; algorithm is recorded separately. |
| `cheez:hashAlgorithm` | core | Artifact → `xsd:string` | Digest algorithm identifier. |
| `cheez:validTime` | core | Assertion → `time:Interval` | When this statement version held. |
| `cheez:recordedTime` | core | Assertion → `time:Interval` | When this version was current in the store. |
| `cheez:certifies` | core | Certificate → `prov:Entity` | Answer covered by this certificate, not a claim of correctness. |
| `cheez:check` | core | Resource → CheckRun | Executed check relevant to the resource. |
| `cheez:rule` | core | Activity → Artifact | Exact rule artifact applied; specialises `prov:used`. |
| `cheez:statement` | core | Activity → Assertion | Exact statement version read; specialises `prov:used`. |
| `cubby:proposer` | cubby | Turn or Program → Resource | Proposer responsible for the candidate program. |
| `cubby:question` | cubby | Turn or ConversationState → `prov:Entity` | Question associated with this record. |
| `cubby:program` | cubby | Turn or ConversationState → Program | Associated program. |
| `cubby:answer` | cubby | Turn or ConversationState → `prov:Entity` | Associated answer. |
| `cubby:openQuestion` | cubby | ConversationState → `prov:Entity` | Outstanding clarification. |
| `cubby:entity` | cubby | ConversationState → Resource | Entity retained by identifier. |
| `cubby:rewrite` | cubby | Turn → `prov:Entity` | Follow-up rewrite checked by the entity guard. |
| `cubby:region` | cubby | FieldSuggestion, Review or Assertion → DocumentRegion | Source region for extraction, review or commitment. |
| `cubby:reader` | cubby | DocumentRegion → Artifact | Exact reader implementation/version artifact. |
| `cubby:confidence` | cubby | EmotionInference → `xsd:decimal` | Inference confidence in [0,1], not factual certainty. |
| `cubby:dimensionSet` | cubby | EmotionInference or HormoneState → Resource | Explicitly identifies the dimension vocabulary. |
| `cubby:export` | cubby | EmotionInference or HormoneState → Artifact | Linked EmotionML export. |
| `cubby:scope` | cubby | Helper → Resource | Scope of required answer preservation. |
| `cubby:breakingCase` | cubby | LifecycleChange → Resource | Retained case that caused helper rejection. |
| `cubby:combinationProgram` | cubby | Resource → Program | Program combining split-question results. |
| `cubby:missingPart` | cubby | Resource → Resource | Explicitly identifies an unanswered part. |
| `mowm:worldId` | mowm | World → `xsd:integer` | Implementation numeric id, not an RDF identifier. |
| `mowm:z` | mowm | World → `xsd:nonNegativeInteger` | World depth. |
| `mowm:zMax` | mowm | World → `xsd:nonNegativeInteger` | Maximum permitted depth. |
| `mowm:parent` | mowm | World → World | Parent world-node version. |
| `mowm:world` | mowm | RoutingDecision → World | Candidate or selected world, distinguished by evidence/outcome. |
| `mowm:mergeStrategy` | mowm | World → Resource | Recorded transition-delta merge strategy. |
| `mowm:cleanupStrategy` | mowm | Resource → Resource | Recorded cleanup strategy. |
| `mowm:component` | mowm | World → Artifact | Exact component, including HYLA or weights artifact. |
| `cube:executionOutcome` | cubelang | VMRun → `skos:Concept` | Return, error or ASK/suspend outcome. |
| `cube:resumeOf` | cubelang | VMRun → VMRun | Re-execution continuing a suspended request. |

A weights artifact uses `prov:wasGeneratedBy` to identify its training run. A seed is only another recorded input; it does not replace that link or the weights hash.

### 1.5 Controlled codes and recording policies

Code IRIs live in the corresponding vocabulary module; they are SKOS individuals, not record identifiers.

Required schemes include:

- Dialogue acts: question, certified answer, refusal, clarifying question, follow-up, small talk. Consider ISO 24617-2 alignment; I do not know the exact mappings.
- Refusal reasons: no plan, relation missing, fact missing, walk failed, not certified, ambiguous, conflict, dates too coarse, unreadable source.
- Conflict kinds: value, type, temporal, logical, recorded against derived.
- Proposers: host grammar, everyday table, overnight rule library, certified-chain episodic memory, language-model program emitter.
- Execution outcomes: satisfied, violated, return, error, suspend.
- Lifecycle kinds: retire, reject, prune, promote.
- Routing outcomes: route, spawn, request spawn, inter-world query.
- Check kinds: entity guard; format; arithmetic; purchase order; supplier; duplicate; number-in-region; scope preservation; verified-answer reproduction; escape attempts; domain regression and gain.

Separate schemes distinguish check outcomes from routing outcomes. There is no `truthScore` property or code.

Further recording rules:

- A certificate must identify the VM run, exact statement versions, sources, revisions, rules and checks. Agreement may be recorded as filtering evidence only.
- A stored assertion has valid and recorded intervals. Closing an old recorded interval must itself retain the prior record and closure evidence; replacing its bytes is not acceptable.
- Story assertions belong to a separately identified store/world. Invented sources identify their originating turn. State-change evidence retains old value, new value, date and turn.
- Reviewers use qualified PROV associations containing role and identity; the review activity provides time, and `cubby:region` provides region. Above-amount approval requires a calibrated or otherwise explicitly sourced approval boundary and the required second review. The amount is not given here.
- A helper is not promoted merely because its sandbox started. Adoption needs recorded preservation and scope checks, plus proposer-window rejection-rate and zero-escape evidence.
- A domain publication records its comparison run and evidence of no new wrong answer and a gain.
- Pruning is represented as retirement with a pruning reason. Whether the current implementation deletes is unknown; deletion would not satisfy this design.

## 2. One JSON-LD 1.1 context

Compact prefixed properties remain usable without individual aliases. Scalar datatype declarations are explicit in records where needed.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "https://schema.org/",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  }
}
```

## 3. Six example records

These are **synthetic fixtures, not records of operations that ran**. Every UUID, hash, timestamp, source revision, numeric example value and execution result below is a placeholder. Code names illustrate the proposed vocabulary.

Apply section 2’s context to each document. Referenced records omitted for space must be supplied before production validation. In particular, an unresolved reference is not evidence that its check ran.

The host software placeholder is always:

`urn:uuid:00000000-0000-4000-8000-000000000001`

### 3.1 Certified answer turn and certificate

Decision Ontology roles are visible as codes. Their exact ontology binding remains unresolved.

```json
{
  "@graph": [
    {
      "@id": "urn:uuid:10000000-0000-4000-8000-000000000001",
      "@type": ["cheez:Record", "cubby:Turn"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:kind": {"@id": "cubby:CertifiedAnswer"},
      "cubby:question": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000002",
        "@type": "prov:Entity",
        "cheez:kind": {"@id": "cheez:QuestionRole"}
      },
      "cubby:program": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000003",
        "@type": "cube:Program",
        "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
        "cheez:hash": "PLACEHOLDER_PROGRAM_HASH",
        "cheez:kind": {"@id": "cheez:OptionRole"}
      },
      "cubby:proposer": {"@id": "cubby:HostGrammar"},
      "cubby:answer": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000004",
        "@type": "prov:Entity",
        "cheez:kind": {"@id": "cheez:DecisionRole"},
        "prov:wasGeneratedBy": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000006"}
      },
      "schema:hasPart": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000005"}
    },
    {
      "@id": "urn:uuid:10000000-0000-4000-8000-000000000005",
      "@type": ["cheez:Record", "cheez:Certificate"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:certifies": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000004"},
      "prov:wasGeneratedBy": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000006"},
      "cheez:check": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000007",
        "@type": "cheez:CheckRun",
        "cheez:kind": {"@id": "cheez:DerivabilityCheck"},
        "cheez:outcome": {"@id": "cheez:Satisfied"},
        "prov:endedAtTime": {"@value": "2026-11-07T00:00:01Z", "@type": "xsd:dateTime"}
      }
    },
    {
      "@id": "urn:uuid:10000000-0000-4000-8000-000000000006",
      "@type": "cube:VMRun",
      "cube:executionOutcome": {"@id": "cube:Return"},
      "prov:used": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000003"},
      "cheez:statement": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000008",
        "@type": "cheez:Assertion",
        "schema:version": "PLACEHOLDER_REVISION",
        "prov:hadPrimarySource": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000009"},
        "cheez:validTime": {
          "@type": "time:Interval",
          "time:hasBeginning": {
            "@type": "time:Instant",
            "time:inXSDDateTimeStamp": {"@value": "2026-11-01T00:00:00Z", "@type": "xsd:dateTimeStamp"}
          }
        },
        "cheez:recordedTime": {
          "@type": "time:Interval",
          "time:hasBeginning": {
            "@type": "time:Instant",
            "time:inXSDDateTimeStamp": {"@value": "2026-11-02T00:00:00Z", "@type": "xsd:dateTimeStamp"}
          }
        }
      },
      "cheez:rule": {
        "@id": "urn:uuid:10000000-0000-4000-8000-000000000010",
        "@type": "cheez:Artifact",
        "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
        "cheez:hash": "PLACEHOLDER_RULE_HASH"
      },
      "cheez:check": {"@id": "urn:uuid:10000000-0000-4000-8000-000000000007"}
    }
  ]
}
```

The final Decision Ontology binding must connect each criterion to the actual satisfying or violating check using its `is_satisfied_by` / `is_violated_by` relations. The fixture does not claim to implement that missing binding.

### 3.2 Refusal with reason

```json
{
  "@id": "urn:uuid:20000000-0000-4000-8000-000000000001",
  "@type": ["cheez:Record", "cubby:Turn"],
  "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
  "cheez:kind": {"@id": "cubby:Refusal"},
  "cheez:reason": {"@id": "cubby:FactMissing"},
  "cubby:proposer": {"@id": "cubby:EverydayTable"},
  "cubby:question": {"@id": "urn:uuid:20000000-0000-4000-8000-000000000002"},
  "cheez:evidence": {
    "@id": "urn:uuid:20000000-0000-4000-8000-000000000003",
    "@type": "cheez:CheckRun",
    "cheez:kind": {"@id": "cheez:RequiredFactCheck"},
    "cheez:outcome": {"@id": "cheez:Violated"},
    "prov:endedAtTime": {"@value": "2026-11-07T00:01:00Z", "@type": "xsd:dateTime"}
  }
}
```

### 3.3 MoWM routing decision ending in a spawn

This illustrates the Cubby bridge policy, not router residual spawning. One candidate means no runner-up margin test applies.

```json
{
  "@graph": [
    {
      "@id": "urn:uuid:30000000-0000-4000-8000-000000000001",
      "@type": ["cheez:Record", "mowm:RoutingDecision"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:kind": {"@id": "mowm:CubbyBridgeCentroidPolicy"},
      "cheez:outcome": {"@id": "mowm:Spawn"},
      "cheez:reason": {"@id": "mowm:AbsoluteFloorNotCleared"},
      "mowm:world": {"@id": "urn:uuid:30000000-0000-4000-8000-000000000002"},
      "cheez:evidence": {
        "@id": "urn:uuid:30000000-0000-4000-8000-000000000003",
        "@type": "cheez:Measurement",
        "cheez:metric": {"@id": "mowm:CentroidScore"},
        "cheez:value": {"@value": "0.20", "@type": "xsd:decimal"}
      },
      "cheez:threshold": {"@id": "urn:uuid:30000000-0000-4000-8000-000000000004"},
      "prov:wasGeneratedBy": {"@id": "urn:uuid:30000000-0000-4000-8000-000000000006"}
    },
    {
      "@id": "urn:uuid:30000000-0000-4000-8000-000000000004",
      "@type": "cheez:Threshold",
      "cheez:metric": {"@id": "mowm:CentroidScore"},
      "cheez:value": {"@value": "0.40", "@type": "xsd:decimal"},
      "prov:wasGeneratedBy": {
        "@id": "urn:uuid:30000000-0000-4000-8000-000000000005",
        "@type": "cheez:Calibration",
        "cheez:screen": {
          "@id": "urn:uuid:30000000-0000-4000-8000-000000000010",
          "@type": "cheez:Artifact",
          "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
          "cheez:hash": "PLACEHOLDER_SCREEN_HASH"
        },
        "cheez:dataset": {
          "@id": "urn:uuid:30000000-0000-4000-8000-000000000011",
          "@type": "cheez:Artifact",
          "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
          "cheez:hash": "PLACEHOLDER_DATA_HASH"
        },
        "cheez:encoder": {
          "@id": "urn:uuid:30000000-0000-4000-8000-000000000012",
          "@type": "cheez:Artifact",
          "cheez:kind": {"@id": "mowm:FastWordEncoder"},
          "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
          "cheez:hash": "PLACEHOLDER_WORD_TABLE_ENCODER_HASH"
        },
        "cheez:corpus": {
          "@id": "urn:uuid:30000000-0000-4000-8000-000000000013",
          "@type": "cheez:Artifact",
          "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
          "cheez:hash": "PLACEHOLDER_CORPUS_HASH"
        },
        "cheez:deployment": {"@id": "urn:uuid:30000000-0000-4000-8000-000000000014"},
        "prov:endedAtTime": {"@value": "2026-11-06T00:00:00Z", "@type": "xsd:dateTime"}
      }
    },
    {
      "@id": "urn:uuid:30000000-0000-4000-8000-000000000006",
      "@type": "mowm:WorldChange",
      "cheez:kind": {"@id": "mowm:Spawn"},
      "prov:endedAtTime": {"@value": "2026-11-07T00:02:00Z", "@type": "xsd:dateTime"}
    },
    {
      "@id": "urn:uuid:30000000-0000-4000-8000-000000000007",
      "@type": "mowm:World",
      "mowm:worldId": 5000,
      "prov:wasGeneratedBy": {"@id": "urn:uuid:30000000-0000-4000-8000-000000000006"}
    }
  ]
}
```

The fixture’s spawn outcome has a generated world. Without that execution evidence, the outcome would be only `mowm:RequestSpawn`.

### 3.4 World split: child at z+1

```json
{
  "@graph": [
    {
      "@id": "urn:uuid:40000000-0000-4000-8000-000000000001",
      "@type": ["cheez:Record", "mowm:WorldChange"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:kind": {"@id": "mowm:ResidualSplit"},
      "cheez:subject": {"@id": "urn:uuid:40000000-0000-4000-8000-000000000002"},
      "cheez:threshold": {"@id": "urn:uuid:40000000-0000-4000-8000-000000000005"},
      "cheez:evidence": {
        "@id": "urn:uuid:40000000-0000-4000-8000-000000000006",
        "@type": "cheez:CheckRun",
        "cheez:kind": {"@id": "mowm:DepthAndResidualSpawnCheck"},
        "cheez:outcome": {"@id": "cheez:Satisfied"}
      }
    },
    {
      "@id": "urn:uuid:40000000-0000-4000-8000-000000000002",
      "@type": "mowm:World",
      "mowm:worldId": 1000,
      "mowm:z": {"@value": "0", "@type": "xsd:nonNegativeInteger"},
      "mowm:zMax": {"@value": "2", "@type": "xsd:nonNegativeInteger"}
    },
    {
      "@id": "urn:uuid:40000000-0000-4000-8000-000000000003",
      "@type": "mowm:World",
      "mowm:worldId": 1007,
      "mowm:z": {"@value": "1", "@type": "xsd:nonNegativeInteger"},
      "mowm:zMax": {"@value": "2", "@type": "xsd:nonNegativeInteger"},
      "mowm:parent": {"@id": "urn:uuid:40000000-0000-4000-8000-000000000002"},
      "prov:wasGeneratedBy": {"@id": "urn:uuid:40000000-0000-4000-8000-000000000001"}
    }
  ]
}
```

The threshold reference must resolve to a fully calibrated threshold. Its value and calibration are not supplied by this fixture. The child’s numeric id is an allocator result, not `parent + 1`.

### 3.5 Helper rejected in its sandbox

```json
{
  "@id": "urn:uuid:50000000-0000-4000-8000-000000000001",
  "@type": ["cheez:Record", "cheez:LifecycleChange"],
  "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
  "cheez:kind": {"@id": "cheez:Reject"},
  "cheez:reason": {"@id": "cubby:VerifiedAnswerChanged"},
  "cheez:subject": {
    "@id": "urn:uuid:50000000-0000-4000-8000-000000000002",
    "@type": "cubby:Helper",
    "cheez:hashAlgorithm": "PLACEHOLDER_ALGORITHM",
    "cheez:hash": "PLACEHOLDER_HELPER_HASH",
    "cubby:proposer": {"@id": "cubby:OvernightRuleLibrary"},
    "cubby:scope": {"@id": "urn:uuid:50000000-0000-4000-8000-000000000003"}
  },
  "cubby:breakingCase": {"@id": "urn:uuid:50000000-0000-4000-8000-000000000004"},
  "cheez:evidence": {
    "@id": "urn:uuid:50000000-0000-4000-8000-000000000005",
    "@type": "cube:VMRun",
    "cheez:kind": {"@id": "cubby:LockedDownSandbox"},
    "cube:executionOutcome": {"@id": "cube:Return"},
    "cheez:check": {
      "@id": "urn:uuid:50000000-0000-4000-8000-000000000006",
      "@type": "cheez:CheckRun",
      "cheez:kind": {"@id": "cubby:VerifiedAnswerReproduction"},
      "cheez:outcome": {"@id": "cheez:Violated"},
      "prov:used": {"@id": "urn:uuid:50000000-0000-4000-8000-000000000004"}
    }
  }
}
```

No promotion is asserted. The candidate, sandbox evidence and breaking case remain retained.

### 3.6 Invoice field reviewed, then committed

The placeholder region contains the placeholder field number. Omitted exact-check records and reviewer-role associations must be resolved before adoption. This fixture does not assume an unknown second-approval amount.

```json
{
  "@graph": [
    {
      "@id": "urn:uuid:60000000-0000-4000-8000-000000000001",
      "@type": "cubby:DocumentRegion",
      "schema:text": "PLACEHOLDER INVOICE TEXT: total 125.00",
      "cubby:reader": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000002"}
    },
    {
      "@id": "urn:uuid:60000000-0000-4000-8000-000000000003",
      "@type": ["cheez:Record", "cubby:FieldSuggestion"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:kind": {"@id": "cubby:InvoiceTotal"},
      "cheez:value": {"@value": "125.00", "@type": "xsd:decimal"},
      "cubby:region": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000001"}
    },
    {
      "@id": "urn:uuid:60000000-0000-4000-8000-000000000004",
      "@type": ["cheez:Record", "cubby:Review"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "prov:used": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000003"},
      "cubby:region": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000001"},
      "prov:qualifiedAssociation": {
        "@type": "prov:Association",
        "prov:agent": {
          "@id": "urn:uuid:60000000-0000-4000-8000-000000000005",
          "@type": "prov:Person"
        },
        "prov:hadRole": {"@id": "cubby:InvoiceReviewer"}
      },
      "cheez:outcome": {"@id": "cubby:Approved"},
      "prov:endedAtTime": {"@value": "2026-11-07T00:05:00Z", "@type": "xsd:dateTime"}
    },
    {
      "@id": "urn:uuid:60000000-0000-4000-8000-000000000006",
      "@type": ["cheez:Record", "cheez:Assertion"],
      "cheez:writer": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000001"},
      "cheez:kind": {"@id": "cubby:InvoiceTotal"},
      "cheez:value": {"@value": "125.00", "@type": "xsd:decimal"},
      "prov:wasDerivedFrom": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000003"},
      "prov:hadPrimarySource": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000005"},
      "cubby:region": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000001"},
      "cheez:evidence": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000004"},
      "prov:wasGeneratedBy": {
        "@id": "urn:uuid:60000000-0000-4000-8000-000000000007",
        "@type": "prov:Activity",
        "cheez:kind": {"@id": "cubby:HostCommit"},
        "prov:used": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000004"},
        "cheez:check": {"@id": "urn:uuid:60000000-0000-4000-8000-000000000008"}
      },
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTimeStamp": {"@value": "2026-11-01T00:00:00Z", "@type": "xsd:dateTimeStamp"}
        }
      },
      "cheez:recordedTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTimeStamp": {"@value": "2026-11-07T00:06:00Z", "@type": "xsd:dateTimeStamp"}
        }
      }
    }
  ]
}
```

The commit-check reference must cover number-in-region, format, arithmetic, purchase order, supplier, duplicate and approval-policy checks as applicable. A review approval alone is insufficient.

## 4. SHACL shapes

The three principal targets are certificates, thresholds and assertion versions. Supporting shapes are nested constraints.

These shapes validate structure, not factual correctness, host authenticity, sandbox isolation or actual execution. Production validation must also reject unresolved evidence references and verify artifact hashes.

```turtle
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cube: <https://w3id.org/cheezml/cubelang#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cheez:ArtifactShape a sh:NodeShape ;
  sh:class cheez:Artifact ;
  sh:property [
    sh:path cheez:hash ;
    sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string
  ] ;
  sh:property [
    sh:path cheez:hashAlgorithm ;
    sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string
  ] .

cheez:CompletedCheckShape a sh:NodeShape ;
  sh:class cheez:CheckRun ;
  sh:property [
    sh:path cheez:kind ; sh:minCount 1 ; sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cheez:outcome ; sh:hasValue cheez:Satisfied ;
    sh:maxCount 1
  ] ;
  sh:property [
    sh:path prov:endedAtTime ;
    sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:dateTime
  ] .

cheez:CertificateShape a sh:NodeShape ;
  sh:targetClass cheez:Certificate ;
  sh:nodeKind sh:IRI ;
  sh:property [
    sh:path cheez:writer ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cheez:certifies ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:class prov:Entity
  ] ;
  sh:property [
    sh:path cheez:check ; sh:minCount 1 ;
    sh:node cheez:CompletedCheckShape
  ] ;
  sh:property [
    sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:node [
      sh:class cube:VMRun ;
      sh:property [
        sh:path cube:executionOutcome ; sh:hasValue cube:Return
      ] ;
      sh:property [
        sh:path cheez:statement ; sh:minCount 1 ;
        sh:class cheez:Assertion
      ] ;
      sh:property [
        sh:path cheez:rule ; sh:minCount 1 ;
        sh:node cheez:ArtifactShape
      ] ;
      sh:property [
        sh:path cheez:check ; sh:minCount 1 ;
        sh:node cheez:CompletedCheckShape
      ]
    ]
  ] .

cheez:ThresholdShape a sh:NodeShape ;
  sh:targetClass cheez:Threshold ;
  sh:property [
    sh:path cheez:metric ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cheez:value ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:datatype xsd:decimal
  ] ;
  sh:property [
    sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:node [
      sh:class cheez:Calibration ;
      sh:property [
        sh:path cheez:screen ; sh:minCount 1 ; sh:maxCount 1 ;
        sh:node cheez:ArtifactShape
      ] ;
      sh:property [
        sh:path cheez:dataset ; sh:minCount 1 ;
        sh:node cheez:ArtifactShape
      ] ;
      sh:property [
        sh:path cheez:encoder ; sh:minCount 1 ; sh:maxCount 1 ;
        sh:node cheez:ArtifactShape
      ] ;
      sh:property [
        sh:path cheez:corpus ; sh:minCount 1 ;
        sh:node cheez:ArtifactShape
      ] ;
      sh:property [
        sh:path cheez:deployment ; sh:minCount 1 ;
        sh:nodeKind sh:IRI
      ] ;
      sh:property [
        sh:path prov:endedAtTime ; sh:minCount 1 ; sh:maxCount 1 ;
        sh:datatype xsd:dateTime
      ]
    ]
  ] .

cheez:IntervalShape a sh:NodeShape ;
  sh:class time:Interval ;
  sh:property [
    sh:path time:hasBeginning ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:node [
      sh:class time:Instant ;
      sh:property [
        sh:path time:inXSDDateTimeStamp ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:datatype xsd:dateTimeStamp
      ]
    ]
  ] ;
  sh:property [
    sh:path time:hasEnd ; sh:maxCount 1 ;
    sh:node [
      sh:class time:Instant ;
      sh:property [
        sh:path time:inXSDDateTimeStamp ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:datatype xsd:dateTimeStamp
      ]
    ]
  ] .

cheez:AssertionShape a sh:NodeShape ;
  sh:targetClass cheez:Assertion ;
  sh:property [
    sh:path cheez:validTime ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:node cheez:IntervalShape
  ] ;
  sh:property [
    sh:path cheez:recordedTime ; sh:minCount 1 ; sh:maxCount 1 ;
    sh:node cheez:IntervalShape
  ] ;
  sh:property [
    sh:path prov:hadPrimarySource ; sh:minCount 1 ;
    sh:nodeKind sh:IRI
  ] .
```

Additional production constraints must check interval ordering, unchanged historical versions, answer/run consistency, source revisions, statement payloads, certificate-required criteria and calibration compatibility. The shapes above deliberately do not claim to enforce all those invariants.

## 5. CubeLang programs for routing and spawning

Both programs below are **spec only: does not compile today**.

The outer form follows the stated language form. Storage and function bodies are schematic because their exact grammar and the MoWM host bindings are not given. These are integration specifications, not claims that named host operations exist. No new opcode or assembly mnemonic is proposed.

### 5.1 Cubby bridge routing

```text
program MoWMBridgeRouting implements IAgent {
  storage {
    // Specification inputs:
    // deployment, encoder artifact, corpus artifact
    // current candidate worlds
    // calibrated absolute floor, margin and answer thresholds
  }

  route() {
    // HOST: encode text by FastWordEncoder word-table lookup.
    // No sentence model or other neural model on this routing path.
    //
    // HOST: verify calibration scope against deployment,
    // encoder artifact and corpus artifact.
    //
    // If there are no worlds:
    //   produce a spawn request.
    // Else:
    //   HOST: measure centroid scores.
    //   Select best world.
    //   Require best score to clear the calibrated absolute floor.
    //
    //   If two or more worlds:
    //     compute best score minus runner-up score;
    //     also require that margin to clear calibrated tau_margin.
    //
    //   If either applicable requirement fails:
    //     produce a spawn request.
    //   Else:
    //     route to the selected world.
    //     HOST: measure its best member.
    //     If that member falls short of calibrated tau_answer:
    //       perform the inter-world query policy.
    //
    // HOST: write measurements, thresholds, executed checks
    // and the actual result. A request is not a completed spawn.
  }
}
```

This is not the possibility oracle’s default routing policy. Fact-world routing defaults to best member; centroid calibration must not be transferred to it.

The HYLA/DSelect-k expert pipeline is recorded separately from this word-table bridge policy. Their integration boundary is not specified sufficiently here to combine them into one executable program.

### 5.2 Spawn controller

```text
program MoWMSpawnControl implements IAgent {
  storage {
    // Specification inputs:
    // spawn cause and the measurements that caused it
    // relevant calibrated threshold
    // parent world where applicable
    // current depth and maximum depth
  }

  spawn() {
    // Distinguish the causes:
    //
    // Cubby bridge:
    //   use the failed absolute-floor or margin decision.
    //
    // Router residual:
    //   require residual norm to exceed calibrated tau_spawn.
    //   Do not substitute the bridge's floor or a benchmark score.
    //
    // World residual split:
    //   require the world's recorded spawn-confidence policy;
    //   require current z to be below z_max.
    //   At z_max, record refusal to spawn.
    //
    // HOST: invoke the repository's applicable spawn operation.
    // Do not manufacture a numeric identifier.
    //
    // For a world-created child:
    //   use the global world allocator starting at 1000;
    //   record child depth as parent depth plus one.
    //
    // For a router-created world:
    //   use the router allocator starting at 5000.
    //
    // Only after the operation actually creates a world:
    //   record the generated world's UUID and returned numeric id;
    //   record parent, depth, components and actual execution evidence.
    //
    // On failure:
    //   retain the failed attempt and its coded reason.
    //
    // Do not spawn merely because an unspecified confidence is low.
    // Do not claim an unspecified deduplication or cooldown mechanism.
  }
}
```

A future executing implementation can use flat comparison/arithmetic/control-flow operands where the compiler supports them. It still needs a specified host protocol for scoring, allocator access, world creation and result reporting. I do not know that protocol.

ASK/suspend is suitable when caller input is required. It is not evidence that an absent MoWM binding executed. Resumption must be recorded as re-execution over `run-proto`, linked to the suspended run.

## 6. What I would cut, and open questions

### Cuts

1. **No generic prose rationale, truth score, convergence certificate or manually asserted active state.**
2. **No duplicate certification system.** Keep PROV-O, the Decision Ontology and actual VM evidence.
3. **No embedded vectors, trained weights, programs or binary exports.** Use hashed artifacts and producing runs.
4. **No new emotion vocabulary.** Link EmotionML; keep user inference and Cubby hormone state separate.
5. **No purported executable container bridge.** The described container emits no bytecode.
6. **No inferred universal routing policy.** Bridge centroid routing, fact-world best-member routing, router residual spawning and depth-limited world splitting remain distinct.
7. **No full pipeline-specific ontology initially.** Record components and strategy artifacts first; add specialised terms only when observed records require them.
8. **No claim that these examples or shapes are deployment-complete.** Unresolved references and bindings are explicit adoption blockers.

### Open questions

- What are the authoritative Decision Ontology IRIs, and which exact criteria must every certificate include?
- What is the host’s authoritative record-writing identity and how is it authenticated?
- How are statement contents represented: named graphs, another RDF statement pattern, or hashed payloads?
- How are interval closures appended without rewriting historical records? What interval boundary convention is used?
- Which digest algorithms and canonicalization rules apply to RDF, program and model artifacts?
- What precisely identifies encoder and corpus compatibility for calibration?
- Does “clears” mean strict comparison for bridge floor and margin? The brief does not specify the equality case.
- What is the exact world-level `tau` spawning predicate? It is described as a confidence threshold, not as the router’s residual-norm threshold.
- How does the host prevent overlapping allocator identities, or distinguish them across restarts and deployments?
- Does current pruning delete data? If so, what changes are required to retain retired records?
- What is the concrete callable CubeLang/host interface for MoWM operations?
- How are inter-world traversal order and termination recorded?
- What is the second-approval amount, its policy source, and the required reviewer independence?
- How are number-in-region matching, numeric normalization and unreadable-source handling specified?
- What are the helper rejection-rate window, adoption scope and escape-detection instrumentation?
- Which exact regression set, gain criterion and RO-Crate profile govern domain publication?

Until these are answered, CheezML can supply a common record vocabulary, but not a complete executable or certification contract.

## anthropic/claude-opus-5.5

*48000 tokens out, 386.8 s, finish `length`*

# CheezML: one panel member's answer

This answer is an opinion, not a finding. Nothing below has been run through a JSON-LD processor, a SHACL engine or `cubelang check`.

## 0. Conventions used throughout

**Namespaces.** The four modules use these namespaces:

- `https://w3id.org/cheezml/core#` (`cheez:`)
- `https://w3id.org/cheezml/cubby#` (`cubby:`)
- `https://w3id.org/cheezml/mowm#` (`mowm:`)
- `https://w3id.org/cheezml/cubelang#` (`cube:`)

Each module is an ontology with IRI `https://w3id.org/cheezml/<module>` (no `#`). Versions are carried by `owl:versionIRI`, for example `https://w3id.org/cheezml/core/2026-11-07`. That puts a date in a version IRI, never in the namespace. The later redirect to `ns.grillcheeseai.com` changes where the IRIs resolve, not the IRIs themselves.

**Records and agents.** Every record, and every agent instance (the host, the VM, the encoder, reviewers), has a `urn:uuid:` id.

**Placeholders.** In the examples, every uuid is a placeholder of the form `urn:uuid:00000000-0000-4000-8000-000000000NNN`. Every other value not given in the brief is the string `"PLACEHOLDER"`, or begins with it. Where a placeholder sits in a typed slot (decimal, integer, dateTime), it is deliberately ill-typed, so SHACL validation will flag it until a real value is filled in.

**Decision Ontology.** The brief names its terms `Question`, `Option`, `Decision`, `is_satisfied_by` and `is_violated_by`, but not its namespace IRI. I don't know it.

- I write `dec:` bound to `https://example.invalid/PLACEHOLDER-decision-ontology#`. This must be replaced before adoption.
- The brief also does not name the Decision Ontology's criterion class, or its properties linking a question to its options and a decision to its chosen option. I mint `cheez:option`, `cheez:question` and `cheez:chosenOption`, to be replaced by the Decision Ontology's own terms once confirmed.

**Other standard namespaces** (standard IRIs):

| Prefix | IRI |
|---|---|
| `prov:` | `http://www.w3.org/ns/prov#` |
| `time:` | `http://www.w3.org/2006/time#` |
| `skos:` | `http://www.w3.org/2004/02/skos/core#` |
| `rdf:` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` |
| `xsd:` | `http://www.w3.org/2001/XMLSchema#` |
| `sh:` | `http://www.w3.org/ns/shacl#` |
| `schema:` | `https://schema.org/` |
| `np:` | `http://www.nanopub.org/nschema#` |
| `emo:` | `http://www.w3.org/TR/emotion-voc/xml#` |

`emo:` is used only to name the EmotionML vocabulary set `pad-dimensions` by link. EmotionML itself is not RDF.

## 1. Classes and properties

### Design decisions behind the tables

**Records are written by the host from what ran.** Every record class is a subclass of `prov:Entity` or `prov:Activity`, and carries `prov:wasAttributedTo` (or `prov:wasAssociatedWith`) a `cheez:Host`. The vocabulary has no status property at all ("Active", "PASSED", "Certified").

- A check's result exists only as a `cheez:CheckRun` with an outcome. A check that never ran has no CheckRun, so it has no "passed".
- A world's liveness is implied by the absence of a retirement.
- A bridge that never started has no run record.

**No truth score and no rationale field are minted.** Readable sentences exist only as `cheez:RenderedText`, generated by a `cheez:Template` from coded values.

**Thresholds are first-class.** Every threshold is a `cheez:Threshold` with a mandatory calibration. A check run links the observed score (a literal) to a threshold (a node). Scores never sit in a threshold slot.

### 1.1 `core#` (`cheez:`)

| Term | Domain → Range | Definition | Reuses / specialises |
|---|---|---|---|
| `cheez:Record` | class | Anything the host writes as a record. | ⊑ `prov:Entity` |
| `cheez:Host` | class | The host process that writes every record. | ⊑ `prov:SoftwareAgent` |
| `cheez:FactVersion` | class | One version of a stored statement, with both clocks. | ⊑ `rdf:Statement`, `prov:Entity`; uses `rdf:subject`/`rdf:predicate`/`rdf:object` |
| `cheez:validTime` | FactVersion → `time:Interval` | When the statement held (order 0055 valid time). | minted; range OWL-Time |
| recorded time | FactVersion → `xsd:dateTime` | When the store learned it is `prov:generatedAtTime`. A correction closes the old version with `prov:invalidatedAtTime` and links the new one by `prov:wasRevisionOf`. | PROV-O, no new term |
| `cheez:datePrecision` | `time:Instant` → `time:TemporalUnit` | Granularity of a date (for "dates too coarse"). | specialises `time:unitType` usage; values `time:unitDay`, `time:unitMonth`, `time:unitYear` |
| `cheez:sourceRevision` | FactVersion → `xsd:string` | The revision of the primary source that was read. | minted; the source itself is `prov:hadPrimarySource` |
| `cheez:Rule` | class | A rule applied by a program run. | ⊑ `prov:Plan` |
| `cheez:BlobRef` | class | A reference to a vector, weights or other blob, by hash. | ⊑ `prov:Entity` |
| `cheez:contentHash` | BlobRef → `xsd:string` | Hash as `algorithm:hex`. | minted. schema.org may have a pending `sha256` on MediaObject; I have not verified it, so I don't reuse it. |
| `cheez:VectorRef` | class | A reference to an encoded vector. | ⊑ BlobRef |
| `cheez:encoding` | VectorRef → `skos:Concept` | Code form, e.g. `cheez:qFHRR-blockcode`. | minted |
| `cheez:blocks`, `cheez:slots` | VectorRef → `xsd:integer` | k and l (80 and 128 in MoWM today). | minted |
| `cheez:WeightsRef` | class | Trained weights, by hash. | ⊑ BlobRef |
| `cheez:TrainingRun` | class | The run that produced weights; required, since a seed alone does not reproduce weights. | ⊑ `prov:Activity` |
| `cheez:seed` | TrainingRun → `xsd:integer` | Recorded, never sufficient on its own. | minted |
| `cheez:Encoder` | class | A text or world encoder. | ⊑ `prov:SoftwareAgent` |
| `cheez:encoderKind` | Encoder → `skos:Concept` | `cheez:wordTableLookup` or `cheez:neural`. | minted scheme |
| `cheez:tableHash` | Encoder → `xsd:string` | Hash of the word table. | minted |
| `cheez:DatasetRef` | class | Data used by a calibration screen or evaluation, by hash. | ⊑ BlobRef |
| `cheez:Corpus` | class | The corpus a threshold was calibrated on. | ⊑ `prov:Entity` |
| `cheez:Threshold` | class | A cut-off with its calibration. | ⊑ `prov:Entity` |
| `cheez:thresholdValue` | Threshold → `xsd:decimal` | The cut-off. | minted |
| `cheez:comparator` | Threshold → `skos:Concept` | `cheez:ge`, `cheez:gt`, `cheez:le`, `cheez:lt`. | minted scheme |
| `cheez:calibratedBy` | Threshold → CalibrationScreen | The screen that set it. | specialises `prov:wasGeneratedBy` |
| `cheez:forEncoder`, `cheez:forCorpus`, `cheez:deployment` | Threshold → Encoder / Corpus / `prov:Entity` | Scope. Thresholds do not transfer outside it. | minted |
| `cheez:CalibrationScreen` | class | The screen run that set a threshold. Its data is `prov:used` a DatasetRef; its date is `prov:endedAtTime`. | ⊑ `prov:Activity` |
| `cheez:CheckRun` | class | One execution of a check. | ⊑ `prov:Activity` |
| `cheez:checkKind` | CheckRun → `skos:Concept` | What was checked. | minted scheme |
| `cheez:checkStrength` | CheckRun → `skos:Concept` | `cheez:hard` or `cheez:soft` (soft means tone, order 0060). | minted |
| `cheez:observedValue` | CheckRun → `xsd:decimal` | The score or quantity measured. | specialises `prov:value` |
| `cheez:threshold` | CheckRun → Threshold | The threshold compared against. | minted |
| `cheez:outcome` | Activity or Entity → `skos:Concept` | Coded outcome. Check outcomes are `cheez:satisfied`, `cheez:violated`, `cheez:errored`; other schemes are listed per module. | minted |
| `cheez:hasCheck` | Activity → CheckRun | Checks run as part of an activity. | specialises `prov:wasInformedBy` (inverse sense) |
| `cheez:Certificate` | class | States that an answer follows from the facts and rules used, under the recorded checks. It says nothing about truth. | ⊑ `prov:Entity`. `prov:Bundle` was considered; see §6. |
| `cheez:certifies` | Certificate → `cubby:Answer` | The answer covered. | specialises `prov:wasDerivedFrom` (inverse sense) |
| `cheez:decision` | Certificate or Refusal → `dec:Decision` | The decision step. | minted link to Decision Ontology |
| `cheez:question`, `cheez:option`, `cheez:chosenOption` | `dec:Decision` → `dec:Question`; `dec:Question` → `dec:Option`; `dec:Decision` → `dec:Option` | Decision structure. | placeholders for Decision Ontology properties |
| `dec:is_satisfied_by`, `dec:is_violated_by` | CheckRun (as criterion) → `dec:Option` | Criterion result for a proposer's option. | Decision Ontology, as named in 0054 |
| `cheez:AgreementFilter` | class | Agreement between models or agents, used only to filter candidates. It can never generate a Certificate. | ⊑ `prov:Activity` |
| `cheez:Retirement` | class | Retiring, rejecting or pruning something. The thing stays, with `prov:wasInvalidatedBy` this activity and `prov:invalidatedAtTime`. | ⊑ `prov:Activity` |
| `cheez:reason` | Activity or Entity → `skos:Concept` | Coded reason, from a scheme (below). | minted |
| `cheez:Template` | class | A sentence template. | ⊑ `prov:Plan` |
| `cheez:RenderedText` | class | A sentence produced by a template from codes and values: `schema:text`, `cheez:template`, `prov:wasDerivedFrom` the coded record. | ⊑ `prov:Entity` |
| `cheez:template` | RenderedText → Template | The template used. | specialises `prov:hadPlan` usage |

**Code schemes** (each a `skos:ConceptScheme`; members are `skos:Concept` with `skos:notation`):

- `cheez:CheckOutcomes`
- `cheez:Comparators`
- `cheez:CheckKinds`
- `cheez:RetirementReasons`: `retired`, `rejected`, `pruned`, `superseded`.

### 1.2 `cubby#` (`cubby:`)

| Term | Domain → Range | Definition | Reuses |
|---|---|---|---|
| `cubby:Conversation` | class | A conversation. | ⊑ `schema:Conversation` |
| `cubby:Turn` | class | One turn. | ⊑ `schema:Message`, `prov:Entity` |
| `cubby:inConversation` | Turn → Conversation | Membership. | specialises `schema:isPartOf` |
| `cubby:turnIndex` | Turn → `xsd:integer` | Position. | minted |
| `cubby:dialogueAct` | Turn → `skos:Concept` | One of `question`, `certifiedAnswer`, `refusal`, `clarifyingQuestion`, `followUp`, `smallTalk`. | minted scheme; `skos:closeMatch` to ISO 24617-2 once its IRIs are known (I don't know them) |
| `cubby:ConversationState` | class | Host-kept state, versioned by `prov:wasRevisionOf`. | ⊑ `prov:Entity` |
| `cubby:trackedEntity`, `cubby:lastQuestion`, `cubby:lastProgram`, `cubby:lastAnswer`, `cubby:openClarification` | ConversationState → entity / Turn / `cube:Program` / Answer / Turn | The 0059 state fields. | minted |
| `cubby:Rewrite` | class | A follow-up rewrite. `prov:used` the follow-up turn, `prov:generated` the rewritten question, and `cheez:hasCheck` an entity-guard CheckRun. | ⊑ `prov:Activity` |
| `cubby:Answer` | class | An answer value (`prov:value`). | ⊑ `schema:Answer`, `prov:Entity` |
| `cubby:answer` | Turn → Answer, or Turn → Refusal | Content of the turn. | minted |
| `cubby:proposer` | Answer, Refusal or `cube:Program` → `skos:Concept` | One of `hostGrammar`, `everydayTable`, `overnightRuleLibrary`, `episodicMemory`, `lmProgramEmitter`. | minted scheme |
| `cubby:Refusal` | class | A refusal. Its reason is `cheez:reason`, drawn from `cubby:RefusalReasons`. | ⊑ `prov:Entity` |
| `cubby:RefusalReasons` | scheme | `noPlan`, `relationMissing`, `factMissing`, `walkFailed`, `notCertified`, `ambiguous`, `conflict`, `datesTooCoarse`, `unreadableSource`. | SKOS |
| `cubby:conflictKind` | Refusal → `skos:Concept` | `value`, `type`, `temporal`, `logical`, `recordedVsDerived`. | minted scheme |
| `cubby:missing` | Refusal → IRI | The relation, fact or part that was missing. | minted |
| `cubby:EmotionReading` | class | An inference about a user's emotion. Never a fact. | ⊑ `prov:Entity` |
| `cubby:confidence` | EmotionReading → `xsd:decimal` [0, 1] | Confidence of the inference. | minted |
| `cubby:dimensionSet` | EmotionReading or HormoneState → IRI | `emo:pad-dimensions` for users; `cubby:hormone-dimensions` for Cubby. | EmotionML vocabulary by link |
| `cubby:dimension` | reading → node with `cubby:dimensionName` (`xsd:string`) and `prov:value` (`xsd:decimal` [0, 1]) | One dimension value. | minted |
| `cubby:emotionMLExport` | reading → BlobRef | The EmotionML 1.0 document, by hash. | minted |
| `cubby:HormoneState` | class | Cubby's own state, in a separately labelled dimension set. The brief does not give its dimensions. | ⊑ `prov:Entity` |
| `cubby:StoryWorld` | class | A story world. | ⊑ `prov:Entity` |
| `cubby:isolatedFrom` | StoryWorld → store | The real-world store it must not write to. | minted |
| `cubby:StateChange` | class | A dated change in a story world: `cubby:oldValue`, `cubby:newValue`, `cubby:atTurn`, `prov:atTime`. | ⊑ `prov:Activity` |
| `cubby:inventedAtTurn` | FactVersion → Turn | Source of an invented fact ("invented at turn N"). | specialises `prov:hadPrimarySource` |
| `cubby:Document` | class | A document read by the host. | ⊑ `schema:DigitalDocument` |
| `cubby:Region` | class | A region: `schema:text`, `prov:wasGeneratedBy` a `cubby:Reading` associated with a reader agent carrying `schema:softwareVersion`. | ⊑ `prov:Entity` |
| `cubby:Reading` | class | The reader's run over a document. | ⊑ `prov:Activity` |
| `cubby:Field` | class | An extracted field: `cubby:fieldName`, `prov:value`, `cubby:fromRegion`. | ⊑ `prov:Entity` |
| `cubby:Suggestion` | class | A derived value. Suggestion only; it is never committed as a fact without review. | ⊑ `prov:Entity` |
| `cubby:Review` | class | A human review: `prov:qualifiedAssociation` (agent and `prov:hadRole`), `prov:endedAtTime`, `prov:used` the region. | ⊑ `prov:Activity` |
| `cubby:ApprovalLimit` | class | An amount above which a second approval is needed. It is a policy limit, not a calibrated score threshold. Value: `cheez:thresholdValue`; origin: `prov:wasDerivedFrom` the order. | ⊑ `prov:Entity` |
| `cubby:approvalsRequired` | CheckRun → `xsd:integer` | Approvals the limit check demanded. | minted |
| `cubby:Helper` | class | A helper proposed by Cubby's proposers. | ⊑ `cube:Program` |
| `cubby:SandboxTrial` | class | A locked-down VM session testing a helper. | ⊑ `prov:Activity` |
| `cubby:scope` | Helper → `prov:Collection` | The verified answers in scope. | minted |
| `cubby:breakingCase` | Retirement → Answer | The case that broke a rejected helper. | minted |
| `cubby:Promotion` | class | The host adopting a helper. | ⊑ `prov:Activity` |
| `cubby:ProposerCheck` | class | Windowed checks on a proposer: `cubby:rejectionRate` (`xsd:decimal`), `cubby:window` (`time:Interval`), `cubby:escapeAttempts` (`xsd:integer`, must be 0). | ⊑ CheckRun |
| `cubby:SplitQuestion` | class | A question split into parts: `cubby:part`, `cubby:combinationProgram`, `cheez:outcome`, `cubby:missingPart`. | ⊑ `prov:Entity` |
| `cubby:Part` | class | One part: `cubby:partRoute` and its verdict as `cheez:outcome`. | ⊑ `prov:Entity` |
| `cubby:DomainWorld` | class | `cubby:ontologyModule`, `cubby:openRecords`, `cubby:rules`, `cubby:questionSet`. | ⊑ `prov:Entity` |
| `cubby:DomainChange` | class | A change to mapping, rules or shapes. Published only with `cubby:newWrongAnswers` = 0 and `cubby:gain` > 0. The gain's metric is not stated. | ⊑ `prov:Activity` |
| `cubby:EvaluationRun` | class | Linked to its Workflow Run RO-Crate by `cubby:roCrate`. | ⊑ `prov:Activity`; RO-Crate by link |
| `cubby:HypothesisVerdict` | class | A verdict published as a nanopublication, killed results included. Statistics use STATO term IRIs; I don't supply specific ones. | ⊑ `np:Nanopublication` |

### 1.3 `mowm#` (`mowm:`)

| Term | Domain → Range | Definition | Reuses |
|---|---|---|---|
| `mowm:World` | class | A world tree node. | ⊑ `prov:Entity` |
| `mowm:worldId` | World → `xsd:integer` | The id as issued by a counter. It is never computed from the parent's id. | minted |
| `mowm:idCounter` | World → `skos:Concept` | `mowm:worldCounter` (starts at 1000) or `mowm:routerCounter` (starts at 5000). | minted |
| `mowm:parentWorld` | World → World | Tree parent. | specialises `prov:wasDerivedFrom` |
| `mowm:z`, `mowm:zMax` | World → `xsd:integer` | Depth, and maximum depth. | minted |
| `mowm:tau` | World → Threshold | Confidence threshold for spawning sub-worlds. | minted |
| `mowm:HYLA` | class | A HYLA hypernetwork instance, with `mowm:weights` → WeightsRef. | ⊑ `prov:Entity` |
| `mowm:hyla` | World → HYLA | The HYLAs held by the world. | minted |
| `mowm:mergeStrategy` | World → `skos:Concept` | e.g. `mowm:BindComposeMerge`, `mowm:weightedBundling`. | minted scheme |
| `mowm:cleanupStrategy` | World → `skos:Concept` | Cleanup used. The brief does not list values. | minted scheme |
| `mowm:GlobalWorld` | class | Holds the axioms. | ⊑ World |
| `mowm:Axiom` | class | An axiom: `mowm:axiomDomain`, `mowm:axiomOrigin` (`seeded` or `discovered`), `mowm:discoveryStep` (`semanticMapping`, `symbolicRegression`, `axiomDistillation`), `mowm:vector`. | ⊑ `prov:Entity` |
| `mowm:RoutingDecision` | class | One routing decision. | ⊑ `prov:Activity` |
| `mowm:encoder` | RoutingDecision → Encoder | The text encoder used. It must be a word-table lookup. | minted |
| `mowm:vector` | → VectorRef | Query, axiom or centroid vector, by hash. | minted |
| `mowm:candidate` | RoutingDecision → node with `mowm:candidateWorld` and `cheez:observedValue` | Scored candidates. | minted |
| `mowm:worldCount` | RoutingDecision → `xsd:integer` | Worlds in the pool. | minted |
| `mowm:routingMode` | → `skos:Concept` | `mowm:centroid` or `mowm:bestMember`. | minted |
| outcome values | RoutingDecision | `mowm:route`, `mowm:spawn`. A Spawn's outcomes are `mowm:spawned`, `mowm:refusedAtZMax`. | via `cheez:outcome` |
| `mowm:routedTo` | RoutingDecision → World | The chosen world. | minted |
| `mowm:Spawn` | class | Creates a world: `prov:generated` the world, `prov:wasInformedBy` the decision or the residual check. | ⊑ `prov:Activity` |
| `mowm:residualNorm` | CheckRun | Recorded as `cheez:observedValue` with `checkKind` `mowm:residualCheck`. | no new term |
| `mowm:Prune` | class | Retirement of an idle world, with `mowm:idleSteps`. | ⊑ `cheez:Retirement` |
| `mowm:InterWorldQuery` | class | World-to-world passing: `mowm:hops` (an `rdf:List` of Worlds) and `mowm:memberComparisons`. | ⊑ `prov:Activity` |
| `mowm:FactWorld` | class | A world whose members are FactVersions, encoded by the word table: `mowm:member`, `mowm:topK`. | ⊑ World |

### 1.4 `cubelang#` (`cube:`)

| Term | Domain → Range | Definition | Reuses |
|---|---|---|---|
| `cube:Program` | class | Source program, with `cheez:contentHash` and `cube:interface`. | ⊑ `prov:Plan`, `schema:SoftwareSourceCode` |
| `cube:Bytecode` | class | Compiled output, by hash, with `cube:compilerVersion`. | ⊑ BlobRef; `prov:wasDerivedFrom` the Program |
| `cube:CompileCheck` | class | A `cubelang check` run. If it produced no bytecode, no Bytecode entity exists. | ⊑ CheckRun |
| `cube:VM` | class | The VM. The only agent that may be associated with a certifying run. | ⊑ `prov:SoftwareAgent` |
| `cube:ProgramRun` | class | One execution. Its outcome is one of `cube:returned`, `cube:errored`, `cube:asked`; it records `cube:strictMode`. | ⊑ `prov:Activity` |
| `cube:Ask` | class | A suspension: `cube:askKind`, `cube:choice` (the options offered). | ⊑ `prov:Entity` |
| `cube:resumes` | ProgramRun → Ask | The re-execution over `run-proto` that answered the ask. | specialises `prov:wasInformedBy` |
| `cube:sandboxed` | ProgramRun → `xsd:boolean` | The run was in a locked-down session. | minted |

## 2. The `@context`

```json
{
  "@context": {
    "@version": 1.1,
    "@protected": true,
    "id": "@id",
    "type": "@type",
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "https://schema.org/",
    "np": "http://www.nanopub.org/nschema#",
    "emo": "http://www.w3.org/TR/emotion-voc/xml#",
    "dec": "https://example.invalid/PLACEHOLDER-decision-ontology#",

    "wasGeneratedBy": {"@id": "prov:wasGeneratedBy", "@type": "@id"},
    "wasAttributedTo": {"@id": "prov:wasAttributedTo", "@type": "@id"},
    "wasAssociatedWith": {"@id": "prov:wasAssociatedWith", "@type": "@id"},
    "wasInformedBy": {"@id": "prov:wasInformedBy", "@type": "@id"},
    "wasDerivedFrom": {"@id": "prov:wasDerivedFrom", "@type": "@id"},
    "wasRevisionOf": {"@id": "prov:wasRevisionOf", "@type": "@id"},
    "wasInvalidatedBy": {"@id": "prov:wasInvalidatedBy", "@type": "@id"},
    "hadPrimarySource": {"@id": "prov:hadPrimarySource", "@type": "@id"},
    "used": {"@id": "prov:used", "@type": "@id", "@container": "@set"},
    "generated": {"@id": "prov:generated", "@type": "@id"},
    "qualifiedAssociation": {"@id": "prov:qualifiedAssociation", "@type": "@id"},
    "agent": {"@id": "prov:agent", "@type": "@id"},
    "hadRole": {"@id": "prov:hadRole", "@type": "@id"},
    "startedAtTime": {"@id": "prov:startedAtTime", "@type": "xsd:dateTime"},
    "endedAtTime": {"@id": "prov:endedAtTime", "@type": "xsd:dateTime"},
    "generatedAtTime": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
    "invalidatedAtTime": {"@id": "prov:invalidatedAtTime", "@type": "xsd:dateTime"},
    "atTime": {"@id": "prov:atTime", "@type": "xsd:dateTime"},
    "value": {"@id": "prov:value"},

    "hasBeginning": {"@id": "time:hasBeginning", "@type": "@id"},
    "hasEnd": {"@id": "time:hasEnd", "@type": "@id"},
    "inXSDDateTimeStamp": {"@id": "time:inXSDDateTimeStamp", "@type": "xsd:dateTimeStamp"},
    "datePrecision": {"@id": "cheez:datePrecision", "@type": "@id"},

    "subject": {"@id": "rdf:subject", "@type": "@id"},
    "predicate": {"@id": "rdf:predicate", "@type": "@id"},
    "object": {"@id": "rdf:object"},
    "text": {"@id": "schema:text"},
    "softwareVersion": {"@id": "schema:softwareVersion"},

    "validTime": {"@id": "cheez:validTime", "@type": "@id"},
    "sourceRevision": {"@id": "cheez:sourceRevision"},
    "contentHash": {"@id": "cheez:contentHash"},
    "encoding": {"@id": "cheez:encoding", "@type": "@id"},
    "blocks": {"@id": "cheez:blocks", "@type": "xsd:integer"},
    "slots": {"@id": "cheez:slots", "@type": "xsd:integer"},
    "seed": {"@id": "cheez:seed", "@type": "xsd:integer"},
    "encoderKind": {"@id": "cheez:encoderKind", "@type": "@id"},
    "tableHash": {"@id": "cheez:tableHash"},
    "thresholdValue": {"@id": "cheez:thresholdValue", "@type": "xsd:decimal"},
    "comparator": {"@id": "cheez:comparator", "@type": "@id"},
    "calibratedBy": {"@id": "cheez:calibratedBy", "@type": "@id"},
    "forEncoder": {"@id": "cheez:forEncoder", "@type": "@id"},
    "forCorpus": {"@id": "cheez:forCorpus", "@type": "@id"},
    "deployment": {"@id": "cheez:deployment", "@type": "@id"},
    "checkKind": {"@id": "cheez:checkKind", "@type": "@id"},
    "checkStrength": {"@id": "cheez:checkStrength", "@type": "@id"},
    "observedValue": {"@id": "cheez:observedValue", "@type": "xsd:decimal"},
    "threshold": {"@id": "cheez:threshold", "@type": "@id"},
    "outcome": {"@id": "cheez:outcome", "@type": "@id"},
    "hasCheck": {"@id": "cheez:hasCheck", "@type": "@id", "@container": "@set"},
    "certifies": {"@id": "cheez:certifies", "@type": "@id"},
    "decision": {"@id": "cheez:decision", "@type": "@id"},
    "question": {"@id": "cheez:question", "@type": "@id"},
    "option": {"@id": "cheez:option", "@type": "@id", "@container": "@set"},
    "chosenOption": {"@id": "cheez:chosenOption", "@type": "@id"},
    "isSatisfiedBy": {"@id": "dec:is_satisfied_by", "@type": "@id"},
    "isViolatedBy": {"@id": "dec:is_violated_by", "@type": "@id"},
    "reason": {"@id": "cheez:reason", "@type": "@id"},
    "template": {"@id": "cheez:template", "@type": "@id"},

    "inConversation": {"@id": "cubby:inConversation", "@type": "@id"},
    "turnIndex": {"@id": "cubby:turnIndex", "@type": "xsd:integer"},
    "dialogueAct": {"@id": "cubby:dialogueAct", "@type": "@id"},
    "answer": {"@id": "cubby:answer", "@type": "@id"},
    "proposer": {"@id": "cubby:proposer", "@type": "@id"},
    "conflictKind": {"@id": "cubby:conflictKind", "@type": "@id"},
    "missing": {"@id": "cubby:missing", "@type": "@id"},
    "fieldName": {"@id": "cubby:fieldName"},
    "fromRegion": {"@id": "cubby:fromRegion", "@type": "@id"},
    "approvalsRequired": {"@id": "cubby:approvalsRequired", "@type": "xsd:integer"},
    "scope": {"@id": "cubby:scope", "@type": "@id"},
    "breakingCase": {"@id": "cubby:breakingCase", "@type": "@id"},
    "escapeAttempts": {"@id": "cubby:escapeAttempts", "@type": "xsd:integer"},
    "rejectionRate": {"@id": "cubby:rejectionRate", "@type": "xsd:decimal"},
    "window": {"@id": "cubby:window", "@type": "@id"},

    "worldId": {"@id": "mowm:worldId", "@type": "xsd:integer"},
    "idCounter": {"@id": "mowm:idCounter", "@type": "@id"},
    "parentWorld": {"@id": "mowm:parentWorld", "@type": "@id"},
    "z": {"@id": "mowm:z", "@type": "xsd:integer"},
    "zMax": {"@id": "mowm:zMax", "@type": "xsd:integer"},
    "tau": {"@id": "mowm:tau", "@type": "@id"},
    "hyla": {"@id": "mowm:hyla", "@type": "@id", "@container": "@set"},
    "weights": {"@id": "mowm:weights", "@type": "@id"},
    "mergeStrategy": {"@id": "mowm:mergeStrategy", "@type": "@id"},
    "cleanupStrategy": {"@id": "mowm:cleanupStrategy", "@type": "@id"},
    "encoder": {"@id": "mowm:encoder", "@type": "@id"},
    "vector": {"@id": "mowm:vector", "@type": "@id"},
    "candidate": {"@id": "mowm:candidate", "@container": "@set"},
    "candidateWorld": {"@id": "mowm:candidateWorld", "@type": "@id"},
    "worldCount": {"@id": "mowm:worldCount", "@type": "xsd:integer"},
    "routingMode": {"@id": "mowm:routingMode", "@type": "@id"},
    "routedTo": {"@id": "mowm:routedTo", "@type": "@id"},

    "interface": {"@id": "cube:interface"},
    "compilerVersion": {"@id": "cube:compilerVersion"},
    "strictMode": {"@id": "cube:strictMode", "@type": "xsd:boolean"},
    "sandboxed": {"@id": "cube:sandboxed", "@type": "xsd:boolean"}
  }
}
```

**Notes on the context:**

- **Prefixes.** Every prefix ends in `#` or `/`, and `xsd:` is the W3C XMLSchema namespace.
- **Code values.** These are written as compact IRIs (`"cubby:factMissing"`) in `@id`-typed slots, so they expand.
- **`object`.** It is left untyped so that it can hold a literal. Where the object is an IRI, a record must write `{"id": ...}`.
- **Context URL.** The examples reference this context as `https://w3id.org/cheezml/context.jsonld`. That URL does not resolve yet. For your expansion test, inline the context above in place of the URL.

## 3. Example records

These are all placeholders except: the structure; the k = 80 / l = 128 block-code form; the counter starts; and the order numbers.

### 3.1 A certified answer turn with its certificate

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000101",
   "type": ["cubby:Turn", "schema:Message", "cheez:Record"],
   "inConversation": "urn:uuid:00000000-0000-4000-8000-000000000100",
   "turnIndex": "PLACEHOLDER", "dialogueAct": "cubby:certifiedAnswer",
   "answer": "urn:uuid:00000000-0000-4000-8000-000000000102",
   "wasAttributedTo": "urn:uuid:00000000-0000-4000-8000-000000000900"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000102",
   "type": ["cubby:Answer", "schema:Answer", "prov:Entity"],
   "value": "PLACEHOLDER", "proposer": "cubby:everydayTable",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000103"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000103",
   "type": ["cube:ProgramRun", "prov:Activity"],
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000901",
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000104",
            "urn:uuid:00000000-0000-4000-8000-000000000105",
            "urn:uuid:00000000-0000-4000-8000-000000000106"],
   "outcome": "cube:returned", "strictMode": true,
   "startedAtTime": "PLACEHOLDER", "endedAtTime": "PLACEHOLDER",
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000107"]},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000901", "type": ["cube:VM", "prov:SoftwareAgent"]},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000104",
   "type": ["cube:Bytecode", "cheez:BlobRef"], "contentHash": "sha256:PLACEHOLDER",
   "compilerVersion": "PLACEHOLDER",
   "wasDerivedFrom": "urn:uuid:00000000-0000-4000-8000-000000000108"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000108",
   "type": ["cube:Program", "prov:Plan"], "contentHash": "sha256:PLACEHOLDER",
   "proposer": "cubby:everydayTable"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000105",
   "type": ["cheez:FactVersion", "rdf:Statement", "prov:Entity"],
   "subject": "urn:uuid:00000000-0000-4000-8000-000000000110",
   "predicate": "PLACEHOLDER-relation-IRI", "object": "PLACEHOLDER",
   "hadPrimarySource": "urn:uuid:00000000-0000-4000-8000-000000000111",
   "sourceRevision": "PLACEHOLDER",
   "validTime": {"type": "time:Interval",
     "hasBeginning": {"type": "time:Instant", "inXSDDateTimeStamp": "PLACEHOLDER",
                      "datePrecision": "time:unitDay"}},
   "generatedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000106",
   "type": ["cheez:Rule", "prov:Plan"], "contentHash": "sha256:PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000107",
   "type": ["cheez:CheckRun", "prov:Activity"],
   "checkKind": "cheez:PLACEHOLDER-check", "checkStrength": "cheez:hard",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER",
   "isSatisfiedBy": "urn:uuid:00000000-0000-4000-8000-000000000113"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000112", "type": "dec:Question",
   "option": ["urn:uuid:00000000-0000-4000-8000-000000000113"]},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000113", "type": "dec:Option",
   "wasDerivedFrom": "urn:uuid:00000000-0000-4000-8000-000000000108"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000114", "type": "dec:Decision",
   "question": "urn:uuid:00000000-0000-4000-8000-000000000112",
   "chosenOption": "urn:uuid:00000000-0000-4000-8000-000000000113"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000115",
   "type": ["cheez:Certificate", "prov:Entity"],
   "certifies": "urn:uuid:00000000-0000-4000-8000-000000000102",
   "decision": "urn:uuid:00000000-0000-4000-8000-000000000114",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000103",
   "wasAttributedTo": "urn:uuid:00000000-0000-4000-8000-000000000900"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000116", "type": "cheez:RenderedText",
   "template": "urn:uuid:00000000-0000-4000-8000-000000000117",
   "wasDerivedFrom": "urn:uuid:00000000-0000-4000-8000-000000000102",
   "text": "PLACEHOLDER (template output)"}
 ]}
```

### 3.2 A refusal with its reason

There is no Certificate, because a certificate covers an answer. The decision step is still recorded.

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000201",
   "type": ["cubby:Turn", "schema:Message", "cheez:Record"],
   "inConversation": "urn:uuid:00000000-0000-4000-8000-000000000100",
   "turnIndex": "PLACEHOLDER", "dialogueAct": "cubby:refusal",
   "answer": "urn:uuid:00000000-0000-4000-8000-000000000202",
   "wasAttributedTo": "urn:uuid:00000000-0000-4000-8000-000000000900"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000202",
   "type": ["cubby:Refusal", "prov:Entity"],
   "reason": "cubby:factMissing",
   "missing": "PLACEHOLDER-relation-IRI",
   "proposer": "cubby:hostGrammar",
   "decision": "urn:uuid:00000000-0000-4000-8000-000000000205",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000203"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000203",
   "type": ["cube:ProgramRun", "prov:Activity"],
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000901",
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000206"],
   "outcome": "cube:returned", "endedAtTime": "PLACEHOLDER",
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000204"]},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000204",
   "type": ["cheez:CheckRun", "prov:Activity"],
   "checkKind": "cheez:factPresent", "checkStrength": "cheez:hard",
   "outcome": "cheez:violated", "endedAtTime": "PLACEHOLDER",
   "isViolatedBy": "urn:uuid:00000000-0000-4000-8000-000000000207"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000207", "type": "dec:Option"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000205", "type": "dec:Decision"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000208", "type": "cheez:RenderedText",
   "template": "urn:uuid:00000000-0000-4000-8000-000000000209",
   "wasDerivedFrom": "urn:uuid:00000000-0000-4000-8000-000000000202",
   "text": "PLACEHOLDER (template output from reason code and missing IRI)"}
 ]}
```

### 3.3 A MoWM routing decision that ends in a spawn

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000301",
   "type": ["mowm:RoutingDecision", "prov:Activity"],
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000900",
   "encoder": "urn:uuid:00000000-0000-4000-8000-000000000302",
   "vector": "urn:uuid:00000000-0000-4000-8000-000000000303",
   "routingMode": "mowm:centroid", "worldCount": "PLACEHOLDER",
   "candidate": [
     {"candidateWorld": "urn:uuid:00000000-0000-4000-8000-000000000310", "observedValue": "PLACEHOLDER"},
     {"candidateWorld": "urn:uuid:00000000-0000-4000-8000-000000000311", "observedValue": "PLACEHOLDER"}],
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000304",
                "urn:uuid:00000000-0000-4000-8000-000000000305"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000320"],
   "outcome": "mowm:spawn", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000302",
   "type": ["cheez:Encoder", "prov:SoftwareAgent"],
   "encoderKind": "cheez:wordTableLookup", "tableHash": "sha256:PLACEHOLDER",
   "softwareVersion": "PLACEHOLDER (FastWordEncoder)"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000303",
   "type": ["cheez:VectorRef"], "contentHash": "sha256:PLACEHOLDER",
   "encoding": "cheez:qFHRR-blockcode", "blocks": 80, "slots": 128},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000304",
   "type": ["cheez:CheckRun", "prov:Activity"], "checkKind": "mowm:floorCheck",
   "checkStrength": "cheez:hard", "observedValue": "PLACEHOLDER (best centroid score)",
   "threshold": "urn:uuid:00000000-0000-4000-8000-000000000306",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000305",
   "type": ["cheez:CheckRun", "prov:Activity"], "checkKind": "mowm:marginCheck",
   "checkStrength": "cheez:hard", "observedValue": "PLACEHOLDER (best minus runner-up)",
   "threshold": "urn:uuid:00000000-0000-4000-8000-000000000307",
   "outcome": "cheez:violated", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000306", "type": "cheez:Threshold",
   "thresholdValue": "PLACEHOLDER", "comparator": "cheez:ge",
   "calibratedBy": "urn:uuid:00000000-0000-4000-8000-000000000308",
   "forEncoder": "urn:uuid:00000000-0000-4000-8000-000000000302",
   "forCorpus": "urn:uuid:00000000-0000-4000-8000-000000000309",
   "deployment": "urn:uuid:00000000-0000-4000-8000-000000000910"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000307", "type": "cheez:Threshold",
   "thresholdValue": "PLACEHOLDER (tau_margin)", "comparator": "cheez:ge",
   "calibratedBy": "urn:uuid:00000000-0000-4000-8000-000000000308",
   "forEncoder": "urn:uuid:00000000-0000-4000-8000-000000000302",
   "forCorpus": "urn:uuid:00000000-0000-4000-8000-000000000309",
   "deployment": "urn:uuid:00000000-0000-4000-8000-000000000910"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000308",
   "type": ["cheez:CalibrationScreen", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000312"],
   "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000312",
   "type": ["cheez:DatasetRef"], "contentHash": "sha256:PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000320",
   "type": ["cube:Bytecode"], "contentHash": "sha256:PLACEHOLDER (MowmRoute, §5.1)"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000330",
   "type": ["mowm:Spawn", "prov:Activity"],
   "wasInformedBy": "urn:uuid:00000000-0000-4000-8000-000000000301",
   "generated": "urn:uuid:00000000-0000-4000-8000-000000000331",
   "outcome": "mowm:spawned", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000331",
   "type": ["mowm:World", "prov:Entity"],
   "worldId": "PLACEHOLDER (issued by counter)", "idCounter": "PLACEHOLDER (counter not stated for bridge spawns)",
   "z": "PLACEHOLDER", "zMax": "PLACEHOLDER",
   "generatedAtTime": "PLACEHOLDER"}
 ]}
```

### 3.4 A world split (a sub-world at z+1)

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000401",
   "type": ["mowm:World", "prov:Entity"],
   "worldId": "PLACEHOLDER", "idCounter": "mowm:worldCounter",
   "z": "PLACEHOLDER (z)", "zMax": "PLACEHOLDER",
   "tau": "urn:uuid:00000000-0000-4000-8000-000000000402",
   "hyla": ["urn:uuid:00000000-0000-4000-8000-000000000403"],
   "mergeStrategy": "mowm:BindComposeMerge", "cleanupStrategy": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000402", "type": "cheez:Threshold",
   "thresholdValue": "PLACEHOLDER", "comparator": "PLACEHOLDER (direction not stated)",
   "calibratedBy": "urn:uuid:00000000-0000-4000-8000-000000000408",
   "forEncoder": "PLACEHOLDER", "forCorpus": "PLACEHOLDER", "deployment": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000403", "type": "mowm:HYLA",
   "weights": "urn:uuid:00000000-0000-4000-8000-000000000404"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000404", "type": "cheez:WeightsRef",
   "contentHash": "sha256:PLACEHOLDER",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000405"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000405", "type": "cheez:TrainingRun",
   "seed": "PLACEHOLDER", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000406",
   "type": ["mowm:Spawn", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000401"],
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000409",
                "urn:uuid:00000000-0000-4000-8000-000000000410"],
   "generated": "urn:uuid:00000000-0000-4000-8000-000000000407",
   "outcome": "mowm:spawned", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000409", "type": "cheez:CheckRun",
   "checkKind": "mowm:subworldConfidenceCheck", "checkStrength": "cheez:hard",
   "observedValue": "PLACEHOLDER", "threshold": "urn:uuid:00000000-0000-4000-8000-000000000402",
   "outcome": "cheez:violated", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000410", "type": "cheez:CheckRun",
   "checkKind": "mowm:depthBelowZMax", "checkStrength": "cheez:hard",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000407",
   "type": ["mowm:World", "prov:Entity"],
   "parentWorld": "urn:uuid:00000000-0000-4000-8000-000000000401",
   "worldId": "PLACEHOLDER (next value of the global counter; not parent id + 1)",
   "idCounter": "mowm:worldCounter",
   "z": "PLACEHOLDER (parent z + 1)", "zMax": "PLACEHOLDER",
   "generatedAtTime": "PLACEHOLDER"}
 ]}
```

In example 3.4, the confidence check is recorded as violated, because the sub-world spawns below confidence `tau`. Whether `tau` is compared against a confidence score, and in which direction, is not stated in the brief. See §6.

### 3.5 A helper rejected in its sandbox (order 0062)

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000501",
   "type": ["cubby:Helper", "cube:Program", "prov:Plan"],
   "contentHash": "sha256:PLACEHOLDER", "proposer": "cubby:overnightRuleLibrary",
   "scope": "urn:uuid:00000000-0000-4000-8000-000000000502",
   "wasInvalidatedBy": "urn:uuid:00000000-0000-4000-8000-000000000510",
   "invalidatedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000502",
   "type": "prov:Collection"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000503",
   "type": ["cubby:SandboxTrial", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000501",
            "urn:uuid:00000000-0000-4000-8000-000000000502"],
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000901",
   "sandboxed": true,
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000504",
                "urn:uuid:00000000-0000-4000-8000-000000000505",
                "urn:uuid:00000000-0000-4000-8000-000000000506"],
   "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000504", "type": "cheez:CheckRun",
   "checkKind": "cubby:reproducesVerifiedInScope", "checkStrength": "cheez:hard",
   "outcome": "cheez:violated", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000505", "type": "cheez:CheckRun",
   "checkKind": "cubby:noChangeOutsideScope", "checkStrength": "cheez:hard",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000506", "type": "cheez:CheckRun",
   "checkKind": "cubby:escapeAttempts", "checkStrength": "cheez:hard",
   "observedValue": 0, "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000510",
   "type": ["cheez:Retirement", "prov:Activity"],
   "reason": "cheez:rejected",
   "wasInformedBy": "urn:uuid:00000000-0000-4000-8000-000000000503",
   "breakingCase": "urn:uuid:00000000-0000-4000-8000-000000000511",
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000900"}
 ]}
```

### 3.6 An invoice field sent to human review, then committed (order 0061)

```json
{"@context": "https://w3id.org/cheezml/context.jsonld",
 "@graph": [
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000601",
   "type": ["cubby:Region", "prov:Entity"], "text": "PLACEHOLDER (region text as read)",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000602"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000602",
   "type": ["cubby:Reading", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000600"],
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000603"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000603",
   "type": "prov:SoftwareAgent", "softwareVersion": "PLACEHOLDER (reader version)"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000604",
   "type": ["cubby:Field", "prov:Entity"], "fieldName": "PLACEHOLDER (e.g. total)",
   "value": "PLACEHOLDER", "fromRegion": "urn:uuid:00000000-0000-4000-8000-000000000601",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000605"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000605",
   "type": ["prov:Activity"],
   "hasCheck": ["urn:uuid:00000000-0000-4000-8000-000000000606",
                "urn:uuid:00000000-0000-4000-8000-000000000607",
                "urn:uuid:00000000-0000-4000-8000-000000000608"]},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000606", "type": "cheez:CheckRun",
   "checkKind": "cubby:numbersInRegionText", "checkStrength": "cheez:hard",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000607", "type": "cheez:CheckRun",
   "checkKind": "cubby:arithmetic", "checkStrength": "cheez:hard",
   "outcome": "cheez:satisfied", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000608", "type": "cheez:CheckRun",
   "checkKind": "cubby:approvalLimit", "checkStrength": "cheez:hard",
   "observedValue": "PLACEHOLDER (amount)",
   "threshold": "urn:uuid:00000000-0000-4000-8000-000000000609",
   "outcome": "cheez:violated", "approvalsRequired": 2, "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000609", "type": "cubby:ApprovalLimit",
   "thresholdValue": "PLACEHOLDER", "wasDerivedFrom": "PLACEHOLDER (order 0061 policy record)"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000610",
   "type": ["cubby:Review", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000601",
            "urn:uuid:00000000-0000-4000-8000-000000000604"],
   "qualifiedAssociation": {"type": "prov:Association",
     "agent": "urn:uuid:00000000-0000-4000-8000-000000000620", "hadRole": "cubby:firstApprover"},
   "outcome": "cubby:approved", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000611",
   "type": ["cubby:Review", "prov:Activity"],
   "used": ["urn:uuid:00000000-0000-4000-8000-000000000601",
            "urn:uuid:00000000-0000-4000-8000-000000000604"],
   "qualifiedAssociation": {"type": "prov:Association",
     "agent": "urn:uuid:00000000-0000-4000-8000-000000000621", "hadRole": "cubby:secondApprover"},
   "outcome": "cubby:approved", "endedAtTime": "PLACEHOLDER"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000612",
   "type": ["cheez:FactVersion", "rdf:Statement", "prov:Entity"],
   "subject": "urn:uuid:00000000-0000-4000-8000-000000000600",
   "predicate": "PLACEHOLDER-field-relation-IRI", "object": "PLACEHOLDER",
   "wasDerivedFrom": "urn:uuid:00000000-0000-4000-8000-000000000604",
   "hadPrimarySource": "urn:uuid:00000000-0000-4000-8000-000000000601",
   "wasGeneratedBy": "urn:uuid:00000000-0000-4000-8000-000000000613",
   "validTime": {"type": "time:Interval",
     "hasBeginning": {"type": "time:Instant", "inXSDDateTimeStamp": "PLACEHOLDER (invoice date)"}},
   "generatedAtTime": "PLACEHOLDER (commit time)"},
  {"id": "urn:uuid:00000000-0000-4000-8000-000000000613",
   "type": ["prov:Activity"],
   "wasInformedBy": "urn:uuid:00000000-0000-4000-8000-000000000611",
   "wasAssociatedWith": "urn:uuid:00000000-0000-4000-8000-000000000900"}
 ]}
```

**Notes on example 3.6:**

- **Commit link.** The commit activity names only the second review. Both reviews are linked to the field through `used`. A SHACL rule (not written here) should require two approved Reviews before a FactVersion derived from a field whose approval-limit check was violated.
- **Review trigger.** I treat the trigger for review as a violation of the approval-limit check. The brief does not list other review triggers.

## 4. SHACL shapes

I chose three classes: the Certificate, because "only the VM certifies"; the RoutingDecision, because it carries the word-table rule and the route/spawn rule; and the Threshold, because of the calibration rule.

```turtle
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix mowm:  <https://w3id.org/cheezml/mowm#> .
@prefix cube:  <https://w3id.org/cheezml/cubelang#> .
@prefix dec:   <https://example.invalid/PLACEHOLDER-decision-ontology#> .

cheez:Prefixes sh:declare
  [ sh:prefix "cheez" ; sh:namespace "https://w3id.org/cheezml/core#"^^xsd:anyURI ] ,
  [ sh:prefix "mowm"  ; sh:namespace "https://w3id.org/cheezml/mowm#"^^xsd:anyURI ] ,
  [ sh:prefix "prov"  ; sh:namespace "http://www.w3.org/ns/prov#"^^xsd:anyURI ] .

# --- Threshold ---
cheez:ThresholdShape a sh:NodeShape ;
  sh:targetClass cheez:Threshold ;
  sh:property [ sh:path cheez:thresholdValue ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ] ;
  sh:property [ sh:path cheez:comparator ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:in ( cheez:ge cheez:gt cheez:le cheez:lt ) ] ;
  sh:property [ sh:path cheez:calibratedBy ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:class cheez:CalibrationScreen ; sh:node cheez:CalibrationShape ] ;
  sh:property [ sh:path cheez:forEncoder ; sh:minCount 1 ; sh:maxCount 1 ; sh:class cheez:Encoder ] ;
  sh:property [ sh:path cheez:forCorpus ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:deployment ; sh:minCount 1 ; sh:maxCount 1 ] .

cheez:CalibrationShape a sh:NodeShape ;
  sh:property [ sh:path prov:used ; sh:minCount 1 ; sh:class cheez:DatasetRef ;
                sh:node [ sh:property [ sh:path cheez:contentHash ; sh:minCount 1 ;
                                        sh:datatype xsd:string ] ] ] ;
  sh:property [ sh:path prov:endedAtTime ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:dateTime ] .

# --- CheckRun (used by both shapes below) ---
cheez:CheckRunShape a sh:NodeShape ;
  sh:targetClass cheez:CheckRun ;
  sh:property [ sh:path cheez:checkKind ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:checkStrength ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:in ( cheez:hard cheez:soft ) ] ;
  sh:property [ sh:path cheez:outcome ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:in ( cheez:satisfied cheez:violated cheez:errored ) ] ;
  sh:property [ sh:path prov:endedAtTime ; sh:minCount 1 ; sh:datatype xsd:dateTime ] ;
  sh:property [ sh:path cheez:threshold ; sh:maxCount 1 ; sh:node cheez:ThresholdShape ] .

# --- Certificate ---
cheez:CertificateShape a sh:NodeShape ;
  sh:targetClass cheez:Certificate ;
  sh:property [ sh:path cheez:certifies ; sh:minCount 1 ; sh:maxCount 1 ; sh:class cubby:Answer ] ;
  sh:property [ sh:path cheez:decision ; sh:minCount 1 ; sh:maxCount 1 ; sh:class dec:Decision ] ;
  sh:property [ sh:path prov:wasAttributedTo ; sh:minCount 1 ; sh:class cheez:Host ] ;
  sh:property [ sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:class cube:ProgramRun ; sh:node cheez:CertifyingRunShape ] .

cheez:CertifyingRunShape a sh:NodeShape ;
  sh:not [ sh:class cheez:AgreementFilter ] ;
  sh:property [ sh:path cheez:outcome ; sh:hasValue cube:returned ] ;
  sh:property [ sh:path prov:wasAssociatedWith ; sh:minCount 1 ; sh:class cube:VM ] ;
  sh:property [ sh:path prov:used ; sh:qualifiedValueShape [ sh:class cube:Bytecode ] ;
                sh:qualifiedMinCount 1 ] ;
  sh:property [ sh:path cheez:hasCheck ; sh:minCount 1 ; sh:node cheez:CheckRunShape ] ;
  sh:sparql [ a sh:SPARQLConstraint ; sh:prefixes cheez:Prefixes ;
    sh:message "A certifying run has a hard check that was not satisfied." ;
    sh:select """SELECT $this WHERE { $this cheez:hasCheck ?c .
                 ?c cheez:checkStrength cheez:hard ;

## google/gemini-4-argon

*17029 tokens out, 218.9 s, finish `stop`*

## 1. The classes and properties

Standard vocabularies reused:
- **PROV-O (`prov:`):** `http://www.w3.org/ns/prov#`
- **Decision Ontology (`do:`):** `https://w3id.org/decision#` (The exact W3C incubator URI for the Decision Ontology is not stated in the brief; I don't know the project's canonical prefix URI and use `https://w3id.org/decision#` as a placeholder).
- **OWL-Time (`time:`):** `http://www.w3.org/2006/time#`
- **SKOS (`skos:`):** `http://www.w3.org/2004/02/skos/core#`
- **schema.org (`schema:`):** `https://schema.org/`
- **EmotionML (`emoml:`):** `http://www.w3.org/2009/10/emotionml#`
- **RO-Crate / STATO / Nanopub:** referenced by standard IRIs where evaluation runs and hypotheses are packaged.

### 1.1 Module `core#` (`https://w3id.org/cheezml/core#`, prefix `cheez:`)

#### Classes (`core#`)
| Term | Superclass / Reused Standard | One-line definition |
| :--- | :--- | :--- |
| `cheez:Record` | `prov:Entity` | A host-written immutable record of an entity, fact, or execution outcome. |
| `cheez:BitemporalStatement` | `cheez:Record`, `prov:Entity` | A stored fact carrying both valid-time and recorded-time intervals (order 0055). |
| `cheez:Retirement` | `prov:Activity` | An activity that closes a record, helper, or world without deleting it, recording the reason. |
| `cheez:RetirementReason` | `skos:Concept` | A controlled code explaining why an entity was retired, rejected, or pruned. |
| `cheez:CheckResult` | `prov:Entity` | The outcome of a concrete check executed by the host or VM against an option or field. |
| `cheez:CalibratedThreshold` | `prov:Entity` | A numeric threshold paired with the evaluation screen, dataset, encoder, and date that set it. |
| `cheez:HashAddressedBlob` | `prov:Entity` | A reference by cryptographic digest to an external vector, weight tensor, or binary blob. |
| `cheez:TrainingRun` | `prov:Activity` | The activity that produced a weight blob, linking data, seed, and resulting hash. |

#### Properties (`core#`)
| Term | Domain | Range | Reuses / Specialises | One-line definition |
| :--- | :--- | :--- | :--- | :--- |
| `cheez:validTime` | `cheez:BitemporalStatement` | `time:Interval` | `prov:atLocation` (temporal) | The interval during which the stated fact held true in the world. |
| `cheez:recordedTime` | `cheez:BitemporalStatement` | `time:Interval` | `prov:generatedAtTime` | The interval during which the store held this version of the fact as current. |
| `cheez:supersedes` | `cheez:BitemporalStatement` | `cheez:BitemporalStatement` | `prov:wasRevisionOf` | Links a correcting fact version to the prior version whose `recordedTime` it closed. |
| `cheez:retiredBy` | `prov:Entity` | `cheez:Retirement` | `prov:wasInvalidatedBy` | Links a retired, rejected, or pruned entity to the retirement activity. |
| `cheez:retirementReason` | `cheez:Retirement` | `cheez:RetirementReason` | `prov:used` | The SKOS concept code giving the reason for retirement, rejection, or pruning. |
| `cheez:thresholdValue` | `cheez:CalibratedThreshold` | `xsd:double` | `schema:value` | The numeric cutoff value established by calibration. |
| `cheez:calibratedByScreen` | `cheez:CalibratedThreshold` | `prov:Activity` | `prov:wasGeneratedBy` | The evaluation screen run (RO-Crate workflow run) that set this threshold. |
| `cheez:calibrationDataset` | `cheez:CalibratedThreshold` | `prov:Entity` | `prov:wasDerivedFrom` | The dataset or corpus over which the threshold was calibrated. |
| `cheez:calibrationEncoder` | `cheez:CalibratedThreshold` | `prov:SoftwareAgent` | `prov:wasAttributedTo` | The exact encoder (e.g., `FastWordEncoder` version) for which the threshold holds. |
| `cheez:calibrationDate` | `cheez:CalibratedThreshold` | `xsd:dateTime` | `prov:generatedAtTime` | The timestamp when the calibration screen completed. |
| `cheez:sha256` | `cheez:HashAddressedBlob` | `xsd:hexBinary` | — | The SHA-256 digest identifying an external vector, weight set, or blob. |
| `cheez:checkCode` | `cheez:CheckResult` | `skos:Concept` | — | Identifies the specific rule or prerequisite check that executed. |
| `cheez:isSoftCheck` | `cheez:CheckResult` | `xsd:boolean` | — | True if the check is a soft (tone) check in a story world rather than an exact check. |

---

### 1.2 Module `cubby#` (`https://w3id.org/cheezml/cubby#`, prefix `cubby:`)

#### Classes (`cubby#`)
| Term | Superclass / Reused Standard | One-line definition |
| :--- | :--- | :--- |
| `cubby:DialogueTurn` | `prov:Entity` | One turn in a host-managed conversation (order 0059). |
| `cubby:DialogueAct` | `skos:Concept` | A dialogue act concept aligned with ISO 24617-2 (`Question`, `CertifiedAnswer`, `Refusal`, `ClarifyingQuestion`, `FollowUp`, `SmallTalk`). |
| `cubby:Certificate` | `prov:Entity`, `do:Decision` | The PROV-O and Decision Ontology proof structure emitted by the VM for a certified answer (order 0054). |
| `cubby:Refusal` | `prov:Entity`, `do:Decision` | A decision record rejecting an answer, carrying structured refusal and conflict codes. |
| `cubby:RefusalReason` | `skos:Concept` | Controlled code for why a turn refused (`NoPlan`, `RelationMissing`, `FactMissing`, `WalkFailed`, `NotCertified`, `Ambiguous`, `DatesTooCoarse`, `UnreadableSource`). |
| `cubby:ConflictKind` | `skos:Concept` | Controlled code for a conflict type (`ValueConflict`, `TypeConflict`, `TemporalConflict`, `LogicalConflict`, `RecordedVsDerivedConflict`). |
| `cubby:Proposer` | `prov:SoftwareAgent` | One of the five system proposers (`HostGrammar`, `EverydayTable`, `OvernightRuleLibrary`, `EpisodicMemory`, `LMProgramEmitter`). |
| `cubby:EmotionInference` | `prov:Entity` | A non-factual inference of user emotion with a confidence, linked to an EmotionML 1.0 document (order 0020). |
| `cubby:HormoneState` | `prov:Entity` | Cubby's internal hormone state represented as a labelled dimension set distinct from user emotion. |
| `cubby:StoryWorld` | `prov:Entity` | An isolated narrative world store separated from the real-world store (order 0060). |
| `cubby:StoryStateChange` | `prov:Activity` | A dated state transition in a story world recording old value, new value, and turn. |
| `cubby:DocumentRegion` | `prov:Entity` | An extracted region of a document carrying verbatim text and the reader's version (order 0061). |
| `cubby:DocumentFieldRecord` | `cheez:BitemporalStatement` | A committed or suggested field extracted from a `cubby:DocumentRegion`. |
| `cubby:HumanReview` | `prov:Activity` | An approval or correction of a document field by a human reviewer acting as a source. |
| `cubby:Reviewer` | `prov:Agent` | A human reviewer recorded with role, identifier, timestamp, and reviewed region. |
| `cubby:SplitQuestionRecord` | `do:Question` | A decomposed question tracking sub-parts, routes, part verdicts, and the combination program (order 0063). |
| `cubby:DomainWorld` | `prov:Entity` | A bundled ontology module, open records, rules, and question set (order 0068/0058). |

#### Properties (`cubby#`)
| Term | Domain | Range | Reuses / Specialises | One-line definition |
| :--- | :--- | :--- | :--- | :--- |
| `cubby:dialogueAct` | `cubby:DialogueTurn` | `cubby:DialogueAct` | — | The dialogue act classification of the turn. |
| `cubby:turnIndex` | `cubby:DialogueTurn` | `xsd:nonNegativeInteger` | `schema:position` | The zero-based or one-based integer turn number in the conversation. |
| `cubby:hasCertificate` | `cubby:DialogueTurn` | `cubby:Certificate` | `prov:wasDerivedFrom` | Links a certified answer turn to its VM-generated certificate. |
| `cubby:hasRefusal` | `cubby:DialogueTurn` | `cubby:Refusal` | `prov:wasDerivedFrom` | Links a refusal turn to its structured refusal decision record. |
| `cubby:proposedBy` | `do:Option` | `cubby:Proposer` | `prov:wasAttributedTo` | Identifies which proposer emitted a candidate program option. |
| `cubby:readStatement` | `prov:Activity` | `cheez:BitemporalStatement` | `prov:used` | Links a VM execution activity to a specific revision of a statement it read. |
| `cubby:appliedRule` | `prov:Activity` | `prov:Entity` | `prov:used` | Links a VM execution activity to a rule applied during derivation. |
| `cubby:refusalReason` | `cubby:Refusal` | `cubby:RefusalReason` | — | The primary code explaining why the question was refused. |
| `cubby:conflictKind` | `cubby:Refusal` | `cubby:ConflictKind` | — | The specific kind of conflict detected, when applicable. |
| `cubby:rewriteText` | `cubby:DialogueTurn` | `xsd:string` | — | The rewritten query produced for a follow-up turn. |
| `cubby:passedEntityGuard` | `cubby:DialogueTurn` | `xsd:boolean` | — | Records whether a follow-up rewrite passed the host's entity guard. |
| `cubby:emotionMLRef` | `cubby:EmotionInference` | `xsd:anyURI` | — | Link to an EmotionML 1.0 representation using the `pad-dimensions` set in [0, 1]. |
| `cubby:inferenceConfidence` | `cubby:EmotionInference` | `xsd:double` | — | The confidence in [0, 1] attached to an inferred user emotion reading. |
| `cubby:inventedAtTurn` | `cheez:BitemporalStatement` | `xsd:nonNegativeInteger` | — | Marks a story-world fact as invented at turn N (`cubby:StoryWorld` only). |
| `cubby:oldValue` | `cubby:StoryStateChange` | ` rdfs:Resource` | `prov:used` | The value before a story-world state change. |
| `cubby:newValue` | `cubby:StoryStateChange` | `rdfs:Resource` | `prov:generated` | The value after a story-world state change. |
| `cubby:regionText` | `cubby:DocumentRegion` | `xsd:string` | `schema:text` | The exact string read from a document region. |
| `cubby:readerVersion` | `cubby:DocumentRegion` | `xsd:string` | `schema:softwareVersion` | The version identifier of the OCR or document reader that produced the region. |
| `cubby:fromRegion` | `cubby:DocumentFieldRecord` | `cubby:DocumentRegion` | `prov:wasDerivedFrom` | The document region from which a field value was extracted. |
| `cubby:isDerivedSuggestion` | `cubby:DocumentFieldRecord` | `xsd:boolean` | — | True if the value is a derived suggestion rather than a committed fact. |
| `cubby:reviewerRole` | `cubby:Reviewer` | `skos:Concept` | `prov:hadRole` | The organizational role of the human reviewer. |
| `cubby:secondApprovalRequiredAbove` | `cubby:DocumentFieldRecord` | `xsd:decimal` | — | The monetary amount threshold above which a second human approval was required. |
| `cubby:missingPart` | `cubby:SplitQuestionRecord` | `do:Question` | — | Identifies an unanswered sub-question in a partial split-question response. |

---

### 1.3 Module `mowm#` (`https://w3id.org/cheezml/mowm#`, prefix `mowm:`)

#### Classes (`mowm#`)
| Term | Superclass / Reused Standard | One-line definition |
| :--- | :--- | :--- |
| `mowm:World` | `prov:Entity` | A tree-node expert world holding HYLAs, axioms or facts, and a `z` depth (`world.py`). |
| `mowm:GlobalWorld` | `mowm:World` | The root world holding seeded and distilled axioms across domains. |
| `mowm:FactWorld` | `mowm:World` | A possibility-oracle world whose members are facts encoded by `FastWordEncoder`. |
| `mowm:RoutingDecision` | `do:Decision`, `prov:Activity` | A DSelect-k or bridge routing execution resulting in a route, inter-world pass, or spawn. |
| `mowm:WorldSpawn` | `prov:Activity` | The creation of a new router world (id >= 5000) or child sub-world at `z+1` (id >= 1000). |
| `mowm:HYLA` | `prov:Entity` | A hypernetwork instance that generates block-code transition functions. |
| `mowm:MergeStrategy` | `skos:Concept` | Controlled concept for merging HYLA transition deltas (e.g., `BindComposeMerge`, `WeightedBundling`). |
| `mowm:CleanupStrategy` | `skos:Concept` | Controlled concept for post-binding VSA cleanup (e.g., `TopKMembers`). |
| `mowm:BlockCodeVector` | `cheez:HashAddressedBlob` | A hash reference to a qFHRR integer-phase sparse block code ($k=80, l=128, D=10240$). |
| `mowm:Axiom` | `prov:Entity` | A seeded or discovered axiom represented as a block-code vector in a world. |
| `mowm:AxiomDiscoveryRun` | `prov:Activity` | The three-step discovery loop (semantic mapping, symbolic regression, axiom distillation). |

#### Properties (`mowm#`)
| Term | Domain | Range | Reuses / Specialises | One-line definition |
| :--- | :--- | :--- | :--- | :--- |
| `mowm:worldId` | `mowm:World` | `xsd:nonNegativeInteger` | `schema:identifier` | The counter-assigned integer ID (starting at 1000 for worlds, 5000 for router-spawned worlds). |
| `mowm:zDepth` | `mowm:World` | `xsd:nonNegativeInteger` | — | The recursion depth $z$ of the world along the Z axis. |
| `mowm:zMax` | `mowm:World` | `xsd:nonNegativeInteger` | — | The maximum recursion depth allowed before `spawn_subworld` refuses. |
| `mowm:parentWorld` | `mowm:World` | `mowm:World` | `prov:wasDerivedFrom` | The parent world at depth $z-1$ from whose residuals this sub-world was spawned. |
| `mowm:hasHYLA` | `mowm:World` | `mowm:HYLA` | — | Links a world to one of its HYLA transition hypernetworks. |
| `mowm:mergeStrategy` | `mowm:World` | `mowm:MergeStrategy` | — | The strategy used to merge transition deltas from multiple HYLAs in the world. |
| `mowm:centroidVector` | `mowm:World` | `mowm:BlockCodeVector` | — | Hash reference to the world's Oja-consolidated or bundled centroid block code. |
| `mowm:bestWorldScore` | `mowm:RoutingDecision` | `xsd:double` | — | The cosine similarity score of the highest-scoring candidate world. |
| `mowm:runnerUpScore` | `mowm:RoutingDecision` | `xsd:double` | — | The cosine similarity score of the second-highest candidate world. |
| `mowm:observedMargin` | `mowm:RoutingDecision` | `xsd:double` | — | The difference `bestWorldScore - runnerUpScore` when two or more worlds exist. |
| `mowm:residualNorm` | `mowm:RoutingDecision` | `xsd:double` | — | The norm of the residual vector compared against `tau_spawn` (`maybe_spawn`). |
| `mowm:floorThreshold` | `mowm:RoutingDecision` | `cheez:CalibratedThreshold` | `prov:used` | The calibrated absolute floor threshold used in the routing decision. |
| `mowm:marginThreshold` | `mowm:RoutingDecision` | `cheez:CalibratedThreshold` | `prov:used` | The calibrated `tau_margin` threshold used in the bridge routing decision. |
| `mowm:spawnThreshold` | `mowm:WorldSpawn` | `cheez:CalibratedThreshold` | `prov:used` | The calibrated `tau` or `tau_spawn` threshold that triggered the spawn. |
| `mowm:answerThreshold` | `mowm:RoutingDecision` | `cheez:CalibratedThreshold` | `prov:used` | The calibrated `tau_answer` threshold checked for member support. |
| `mowm:idleSteps` | `mowm:World` | `xsd:nonNegativeInteger` | — | The number of consecutive routing steps a world has remained idle (compared against `tau_prune_steps`). |
| `mowm:routedToWorld` | `mowm:RoutingDecision` | `mowm:World` | — | The existing world selected when floor and margin checks pass. |
| `mowm:triggeredSpawn` | `mowm:RoutingDecision` | `mowm:WorldSpawn` | `prov:wasInformedBy` | Links a routing decision that fell short of thresholds to the resulting world spawn activity. |
| `mowm:spawnedWorld` | `mowm:WorldSpawn` | `mowm:World` | `prov:generated` | Links a spawn activity to the newly created world record. |

---

### 1.4 Module `cubelang#` (`https://w3id.org/cheezml/cubelang#`, prefix `cube:`)

#### Classes (`cubelang#`)
| Term | Superclass / Reused Standard | One-line definition |
| :--- | :--- | :--- |
| `cube:Program` | `prov:Plan`, `prov:Entity` | A compiled CubeLang bytecode program identified by source and bytecode hash. |
| `cube:VMExecution` | `prov:Activity` | A concrete run of a `cube:Program` inside the CubeLang VM. |
| `cube:ExecutionOutcome` | `skos:Concept` | One of the three implemented VM execution outcomes: `ReturnOutcome`, `ErrorOutcome`, or `AskSuspendOutcome`. |
| `cube:AskSuspension` | `prov:Entity` | The state emitted when a program executes `ASK` and suspends over `run-proto`. |
| `cube:HelperCandidate` | `cube:Program` | A helper program proposed by one of Cubby's proposers for sandbox testing (order 0062). |
| `cube:SandboxTrial` | `cube:VMExecution` | A locked-down VM session testing a `cube:HelperCandidate` against in-scope and out-of-scope cases. |

#### Properties (`cubelang#`)
| Term | Domain | Range | Reuses / Specialises | One-line definition |
| :--- | :--- | :--- | :--- | :--- |
| `cube:bytecodeHash` | `cube:Program` | `cheez:HashAddressedBlob` | — | Hash reference to the compiled bytecode executed by the VM. |
| `cube:implementsInterface` | `cube:Program` | `xsd:string` | — | The declared CubeLang interface (e.g., `ISolver`, `IAgent`). |
| `cube:outcome` | `cube:VMExecution` | `cube:ExecutionOutcome` | — | Records whether the VM run returned, errored, or suspended via `ASK`. |
| `cube:breakingCase` | `cube:SandboxTrial` | `prov:Entity` | `prov:used` | The test case that a rejected helper failed to reproduce or altered out of scope. |
| `cube:inScopeVerifiedCount` | `cube:SandboxTrial` | `xsd:nonNegativeInteger` | — | Count of verified in-scope answers tested during the sandbox trial. |
| `cube:inScopeReproducedCount` | `cube:SandboxTrial` | `xsd:nonNegativeInteger` | — | Count of verified in-scope answers exactly reproduced by the helper. |
| `cube:outOfScopeChangedCount` | `cube:SandboxTrial` | `xsd:nonNegativeInteger` | — | Count of out-of-scope answers altered by the helper (must be 0 for promotion). |
| `cube:escapeAttempts` | `cube:SandboxTrial` | `xsd:nonNegativeInteger` | — | Number of sandbox escape attempts recorded during the trial (must be 0). |
| `cube:proposerRejectionRate` | `cube:SandboxTrial` | `xsd:double` | — | The proposer's helper rejection rate measured over its sliding window. |

---

## 2. One JSON-LD 1.1 `@context`

```json
{
  "@context": {
    "@version": 1.1,
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "schema": "https://schema.org/",
    "do": "https://w3id.org/decision#",
    "emoml": "http://www.w3.org/2009/10/emotionml#",

    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",

    "id": "@id",
    "type": "@type",

    "validTime": { "@id": "cheez:validTime", "@type": "@id" },
    "recordedTime": { "@id": "cheez:recordedTime", "@type": "@id" },
    "supersedes": { "@id": "cheez:supersedes", "@type": "@id" },
    "retiredBy": { "@id": "cheez:retiredBy", "@type": "@id" },
    "retirementReason": { "@id": "cheez:retirementReason", "@type": "@id" },
    "thresholdValue": { "@id": "cheez:thresholdValue", "@type": "xsd:double" },
    "calibratedByScreen": { "@id": "cheez:calibratedByScreen", "@type": "@id" },
    "calibrationDataset": { "@id": "cheez:calibrationDataset", "@type": "@id" },
    "calibrationEncoder": { "@id": "cheez:calibrationEncoder", "@type": "@id" },
    "calibrationDate": { "@id": "cheez:calibrationDate", "@type": "xsd:dateTime" },
    "sha256": { "@id": "cheez:sha256", "@type": "xsd:hexBinary" },
    "checkCode": { "@id": "cheez:checkCode", "@type": "@id" },
    "isSoftCheck": { "@id": "cheez:isSoftCheck", "@type": "xsd:boolean" },

    "dialogueAct": { "@id": "cubby:dialogueAct", "@type": "@id" },
    "turnIndex": { "@id": "cubby:turnIndex", "@type": "xsd:nonNegativeInteger" },
    "hasCertificate": { "@id": "cubby:hasCertificate", "@type": "@id" },
    "hasRefusal": { "@id": "cubby:hasRefusal", "@type": "@id" },
    "proposedBy": { "@id": "cubby:proposedBy", "@type": "@id" },
    "readStatement": { "@id": "cubby:readStatement", "@type": "@id" },
    "appliedRule": { "@id": "cubby:appliedRule", "@type": "@id" },
    "refusalReason": { "@id": "cubby:refusalReason", "@type": "@id" },
    "conflictKind": { "@id": "cubby:conflictKind", "@type": "@id" },
    "rewriteText": { "@id": "cubby:rewriteText", "@type": "xsd:string" },
    "passedEntityGuard": { "@id": "cubby:passedEntityGuard", "@type": "xsd:boolean" },
    "emotionMLRef": { "@id": "cubby:emotionMLRef", "@type": "@id" },
    "inferenceConfidence": { "@id": "cubby:inferenceConfidence", "@type": "xsd:double" },
    "inventedAtTurn": { "@id": "cubby:inventedAtTurn", "@type": "xsd:nonNegativeInteger" },
    "oldValue": { "@id": "cubby:oldValue", "@type": "@id" },
    "newValue": { "@id": "cubby:newValue", "@type": "@id" },
    "regionText": { "@id": "cubby:regionText", "@type": "xsd:string" },
    "readerVersion": { "@id": "cubby:readerVersion", "@type": "xsd:string" },
    "fromRegion": { "@id": "cubby:fromRegion", "@type": "@id" },
    "isDerivedSuggestion": { "@id": "cubby:isDerivedSuggestion", "@type": "xsd:boolean" },
    "reviewerRole": { "@id": "cubby:reviewerRole", "@type": "@id" },
    "secondApprovalRequiredAbove": { "@id": "cubby:secondApprovalRequiredAbove", "@type": "xsd:decimal" },
    "missingPart": { "@id": "cubby:missingPart", "@type": "@id" },

    "worldId": { "@id": "mowm:worldId", "@type": "xsd:nonNegativeInteger" },
    "zDepth": { "@id": "mowm:zDepth", "@type": "xsd:nonNegativeInteger" },
    "zMax": { "@id": "mowm:zMax", "@type": "xsd:nonNegativeInteger" },
    "parentWorld": { "@id": "mowm:parentWorld", "@type": "@id" },
    "hasHYLA": { "@id": "mowm:hasHYLA", "@type": "@id" },
    "mergeStrategy": { "@id": "mowm:mergeStrategy", "@type": "@id" },
    "centroidVector": { "@id": "mowm:centroidVector", "@type": "@id" },
    "bestWorldScore": { "@id": "mowm:bestWorldScore", "@type": "xsd:double" },
    "runnerUpScore": { "@id": "mowm:runnerUpScore", "@type": "xsd:double" },
    "observedMargin": { "@id": "mowm:observedMargin", "@type": "xsd:double" },
    "residualNorm": { "@id": "mowm:residualNorm", "@type": "xsd:double" },
    "floorThreshold": { "@id": "mowm:floorThreshold", "@type": "@id" },
    "marginThreshold": { "@id": "mowm:marginThreshold", "@type": "@id" },
    "spawnThreshold": { "@id": "mowm:spawnThreshold", "@type": "@id" },
    "answerThreshold": { "@id": "mowm:answerThreshold", "@type": "@id" },
    "idleSteps": { "@id": "mowm:idleSteps", "@type": "xsd:nonNegativeInteger" },
    "routedToWorld": { "@id": "mowm:routedToWorld", "@type": "@id" },
    "triggeredSpawn": { "@id": "mowm:triggeredSpawn", "@type": "@id" },
    "spawnedWorld": { "@id": "mowm:spawnedWorld", "@type": "@id" },

    "bytecodeHash": { "@id": "cube:bytecodeHash", "@type": "@id" },
    "implementsInterface": { "@id": "cube:implementsInterface", "@type": "xsd:string" },
    "outcome": { "@id": "cube:outcome", "@type": "@id" },
    "breakingCase": { "@id": "cube:breakingCase", "@type": "@id" },
    "inScopeVerifiedCount": { "@id": "cube:inScopeVerifiedCount", "@type": "xsd:nonNegativeInteger" },
    "inScopeReproducedCount": { "@id": "cube:inScopeReproducedCount", "@type": "xsd:nonNegativeInteger" },
    "outOfScopeChangedCount": { "@id": "cube:outOfScopeChangedCount", "@type": "xsd:nonNegativeInteger" },
    "escapeAttempts": { "@id": "cube:escapeAttempts", "@type": "xsd:nonNegativeInteger" },
    "proposerRejectionRate": { "@id": "cube:proposerRejectionRate", "@type": "xsd:double" }
  }
}
```

---

## 3. Six example records

*(Note: UUIDs, SHA-256 digests, timestamps, and specific calibration numbers not given in the brief are plainly marked as placeholders via `0000...` or comments in strings.)*

### 3.1 A certified answer turn with its certificate (order 0054, 0055, 0059)
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:11111111-0001-4000-8000-000000000001",
  "type": ["cubby:DialogueTurn", "prov:Entity"],
  "turnIndex": 2,
  "dialogueAct": "cubby:CertifiedAnswer",
  "prov:wasGeneratedBy": {
    "id": "urn:uuid:11111111-0001-4000-8000-000000000002",
    "type": ["cube:VMExecution", "prov:Activity"],
    "prov:StartedAtTime": "2026-11-07T10:00:00Z",
    "prov:endedAtTime": "2026-11-07T10:00:01Z",
    "outcome": "cube:ReturnOutcome",
    "prov:used": {
      "id": "urn:uuid:11111111-0001-4000-8000-000000000003",
      "type": ["cube:Program", "do:Option"],
      "implementsInterface": "ISolver",
      "proposedBy": "cubby:HostGrammar",
      "bytecodeHash": {
        "id": "urn:uuid:11111111-0001-4000-8000-000000000004",
        "type": "cheez:HashAddressedBlob",
        "sha256": "a1b2c3d4e5f60000000000000000000000000000000000000000000000000001"
      }
    },
    "readStatement": [
      {
        "id": "urn:uuid:11111111-0001-4000-8000-000000000005",
        "type": "cheez:BitemporalStatement",
        "prov:wasDerivedFrom": { "id": "urn:uuid:11111111-0001-4000-8000-000000000099" },
        "validTime": {
          "id": "urn:uuid:11111111-0001-4000-8000-000000000006",
          "type": "time:Interval",
          "time:hasBeginning": { "id": "urn:uuid:11111111-0001-4000-8000-000000000007", "time:inXSDDateTimeStamp": "2026-01-01T00:00:00Z" }
        },
        "recordedTime": {
          "id": "urn:uuid:11111111-0001-4000-8000-000000000008",
          "type": "time:Interval",
          "time:hasBeginning": { "id": "urn:uuid:11111111-0001-4000-8000-000000000009", "time:inXSDDateTimeStamp": "2026-01-02T08:30:00Z" }
        }
      }
    ],
    "appliedRule": [
      { "id": "urn:uuid:11111111-0001-4000-8000-000000000010", "type": "prov:Entity" }
    ]
  },
  "hasCertificate": {
    "id": "urn:uuid:11111111-0001-4000-8000-000000000011",
    "type": ["cubby:Certificate", "do:Decision", "prov:Entity"],
    "prov:wasGeneratedBy": { "id": "urn:uuid:11111111-0001-4000-8000-000000000002" },
    "do:initiates": {
      "id": "urn:uuid:11111111-0001-4000-8000-000000000012",
      "type": "do:Question"
    },
    "do:considers": [
      {
        "id": "urn:uuid:11111111-0001-4000-8000-000000000003",
        "do:is_satisfied_by": [
          {
            "id": "urn:uuid:11111111-0001-4000-8000-000000000013",
            "type": "cheez:CheckResult",
            "checkCode": "cubby:PrerequisiteCheck",
            "isSoftCheck": false,
            "prov:wasGeneratedBy": { "id": "urn:uuid:11111111-0001-4000-8000-000000000002" }
          }
        ]
      }
    ]
  }
}
```

### 3.2 A refusal with its reason (order 0054, 0059)
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:22222222-0002-4000-8000-000000000001",
  "type": ["cubby:DialogueTurn", "prov:Entity"],
  "turnIndex": 4,
  "dialogueAct": "cubby:Refusal",
  "hasRefusal": {
    "id": "urn:uuid:22222222-0002-4000-8000-000000000002",
    "type": ["cubby:Refusal", "do:Decision", "prov:Entity"],
    "refusalReason": "cubby:NotCertified",
    "conflictKind": "cubby:TemporalConflict",
    "prov:wasGeneratedBy": {
      "id": "urn:uuid:22222222-0002-4000-8000-000000000003",
      "type": ["cube:VMExecution", "prov:Activity"],
      "outcome": "cube:ReturnOutcome",
      "readStatement": [
        { "id": "urn:uuid:22222222-0002-4000-8000-000000000004", "type": "cheez:BitemporalStatement" },
        { "id": "urn:uuid:22222222-0002-4000-8000-000000000005", "type": "cheez:BitemporalStatement" }
      ]
    },
    "do:considers": [
      {
        "id": "urn:uuid:22222222-0002-4000-8000-000000000006",
        "type": ["cube:Program", "do:Option"],
        "proposedBy": "cubby:EverydayTable",
        "do:is_violated_by": [
          {
            "id": "urn:uuid:22222222-0002-4000-8000-000000000007",
            "type": "cheez:CheckResult",
            "checkCode": "cubby:TemporalConsistencyCheck",
            "isSoftCheck": false,
            "prov:wasGeneratedBy": { "id": "urn:uuid:22222222-0002-4000-8000-000000000003" }
          }
        ]
      }
    ]
  }
}
```

### 3.3 A MoWM routing decision that ends in a spawn (`bridges/cubby_bridge.py`, `router.py`)
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:33333333-0003-4000-8000-000000000001",
  "type": ["mowm:RoutingDecision", "do:Decision", "prov:Activity"],
  "prov:used": {
    "id": "urn:uuid:33333333-0003-4000-8000-000000000002",
    "type": "mowm:BlockCodeVector",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000302"
  },
  "bestWorldScore": 0.410,
  "runnerUpScore": 0.395,
  "observedMargin": 0.015,
  "residualNorm": 0.620,
  "floorThreshold": {
    "id": "urn:uuid:33333333-0003-4000-8000-000000000003",
    "type": "cheez:CalibratedThreshold",
    "thresholdValue": 0.450,
    "calibratedByScreen": { "id": "urn:uuid:33333333-0003-4000-8000-000000000090" },
    "calibrationDataset": { "id": "urn:uuid:33333333-0003-4000-8000-000000000091" },
    "calibrationEncoder": { "id": "mowm:FastWordEncoder_v1" },
    "calibrationDate": "2026-11-01T00:00:00Z"
  },
  "marginThreshold": {
    "id": "urn:uuid:33333333-0003-4000-8000-000000000004",
    "type": "cheez:CalibratedThreshold",
    "thresholdValue": 0.050,
    "calibratedByScreen": { "id": "urn:uuid:33333333-0003-4000-8000-000000000090" },
    "calibrationDataset": { "id": "urn:uuid:33333333-0003-4000-8000-000000000091" },
    "calibrationEncoder": { "id": "mowm:FastWordEncoder_v1" },
    "calibrationDate": "2026-11-01T00:00:00Z"
  },
  "triggeredSpawn": {
    "id": "urn:uuid:33333333-0003-4000-8000-000000000005",
    "type": "mowm:WorldSpawn",
    "spawnThreshold": {
      "id": "urn:uuid:33333333-0003-4000-8000-000000000006",
      "type": "cheez:CalibratedThreshold",
      "thresholdValue": 0.500,
      "calibratedByScreen": { "id": "urn:uuid:33333333-0003-4000-8000-000000000090" },
      "calibrationDataset": { "id": "urn:uuid:33333333-0003-4000-8000-000000000091" },
      "calibrationEncoder": { "id": "mowm:FastWordEncoder_v1" },
      "calibrationDate": "2026-11-01T00:00:00Z"
    },
    "spawnedWorld": {
      "id": "urn:uuid:33333333-0003-4000-8000-000000000007",
      "type": "mowm:World",
      "worldId": 5000,
      "zDepth": 0,
      "zMax": 3,
      "idleSteps": 0,
      "mergeStrategy": "mowm:BindComposeMerge",
      "centroidVector": {
        "id": "urn:uuid:33333333-0003-4000-8000-000000000008",
        "type": "mowm:BlockCodeVector",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000308"
      }
    }
  }
}
```

### 3.4 A world split: a sub-world at z+1 (`world.py`)
*(Note: World IDs come from one global counter starting at 1000, never `parent_id + 1`. Here parent is 1004 and the newly spawned child from the global counter is 1019 — placeholder counter values).*
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:44444444-0004-4000-8000-000000000001",
  "type": ["mowm:WorldSpawn", "prov:Activity"],
  "prov:used": {
    "id": "urn:uuid:44444444-0004-4000-8000-000000000002",
    "type": "mowm:World",
    "worldId": 1004,
    "zDepth": 1,
    "zMax": 4
  },
  "spawnThreshold": {
    "id": "urn:uuid:44444444-0004-4000-8000-000000000003",
    "type": "cheez:CalibratedThreshold",
    "thresholdValue": 0.500,
    "calibratedByScreen": { "id": "urn:uuid:44444444-0004-4000-8000-000000000090" },
    "calibrationDataset": { "id": "urn:uuid:44444444-0004-4000-8000-000000000091" },
    "calibrationEncoder": { "id": "mowm:HYLA_WorldEncoder_v1" },
    "calibrationDate": "2026-11-01T00:00:00Z"
  },
  "spawnedWorld": {
    "id": "urn:uuid:44444444-0004-4000-8000-000000000004",
    "type": "mowm:World",
    "worldId": 1019,
    "zDepth": 2,
    "zMax": 4,
    "parentWorld": { "id": "urn:uuid:44444444-0004-4000-8000-000000000002" },
    "mergeStrategy": "mowm:BindComposeMerge",
    "hasHYLA": {
      "id": "urn:uuid:44444444-0004-4000-8000-000000000005",
      "type": "mowm:HYLA",
      "prov:wasGeneratedBy": {
        "id": "urn:uuid:44444444-0004-4000-8000-000000000006",
        "type": "cheez:TrainingRun"
      },
      "prov:hadPrimarySource": {
        "id": "urn:uuid:44444444-0004-4000-8000-000000000007",
        "type": "cheez:HashAddressedBlob",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000407"
      }
    }
  }
}
```

### 3.5 A helper rejected in its sandbox (order 0062)
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:55555555-0005-4000-8000-000000000001",
  "type": ["cube:HelperCandidate", "cube:Program", "prov:Entity"],
  "implementsInterface": "ISolver",
  "proposedBy": "cubby:LMProgramEmitter",
  "bytecodeHash": {
    "id": "urn:uuid:55555555-0005-4000-8000-000000000002",
    "type": "cheez:HashAddressedBlob",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000502"
  },
  "retiredBy": {
    "id": "urn:uuid:55555555-0005-4000-8000-000000000003",
    "type": "cheez:Retirement",
    "prov:endedAtTime": "2026-11-07T11:15:00Z",
    "retirementReason": "cube:HelperFailedInScopeReproduction",
    "prov:wasInformedBy": {
      "id": "urn:uuid:55555555-0005-4000-8000-000000000004",
      "type": ["cube:SandboxTrial", "cube:VMExecution"],
      "outcome": "cube:ReturnOutcome",
      "inScopeVerifiedCount": 50,
      "inScopeReproducedCount": 49,
      "outOfScopeChangedCount": 0,
      "escapeAttempts": 0,
      "proposerRejectionRate": 0.12,
      "breakingCase": {
        "id": "urn:uuid:55555555-0005-4000-8000-000000000005",
        "type": "cubby:DialogueTurn"
      }
    }
  }
}
```

### 3.6 An invoice field sent to human review and then committed (order 0061, 0055)
```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "id": "urn:uuid:66666666-0006-4000-8000-000000000001",
  "type": ["cubby:DocumentFieldRecord", "cheez:BitemporalStatement"],
  "isDerivedSuggestion": false,
  "schema:value": 14500.00,
  "secondApprovalRequiredAbove": 10000.00,
  "fromRegion": {
    "id": "urn:uuid:66666666-0006-4000-8000-000000000002",
    "type": "cubby:DocumentRegion",
    "regionText": "TOTAL DUE: 14500.00 USD",
    "readerVersion": "CubbyDocReader-2.1.0"
  },
  "validTime": {
    "id": "urn:uuid:66666666-0006-4000-8000-000000000003",
    "type": "time:Interval",
    "time:hasBeginning": { "id": "urn:uuid:66666666-0006-4000-8000-000000000004", "time:inXSDDateTimeStamp": "2026-10-31T00:00:00Z" }
  },
  "recordedTime": {
    "id": "urn:uuid:66666666-0006-4000-8000-000000000005",
    "type": "time:Interval",
    "time:hasBeginning": { "id": "urn:uuid:66666666-0006-4000-8000-000000000006", "time:inXSDDateTimeStamp": "2026-11-07T14:22:00Z" }
  },
  "supersedes": {
    "id": "urn:uuid:66666666-0006-4000-8000-000000000007",
    "type": ["cubby:DocumentFieldRecord", "cheez:BitemporalStatement"],
    "isDerivedSuggestion": true
  },
  "prov:wasGeneratedBy": {
    "id": "urn:uuid:66666666-0006-4000-8000-000000000008",
    "type": "cubby:HumanReview",
    "prov:endedAtTime": "2026-11-07T14:22:00Z",
    "prov:used": [
      {
        "id": "urn:uuid:66666666-0006-4000-8000-000000000009",
        "type": "cheez:CheckResult",
        "checkCode": "cubby:ArithmeticCheck"
      },
      {
        "id": "urn:uuid:66666666-0006-4000-8000-000000000010",
        "type": "cheez:CheckResult",
        "checkCode": "cubby:PurchaseOrderCheck"
      }
    ],
    "prov:wasAssociatedWith": [
      {
        "id": "urn:uuid:66666666-0006-4000-8000-000000000011",
        "type": "cubby:Reviewer",
        "reviewerRole": "cubby:AccountsPayableClerk"
      },
      {
        "id": "urn:uuid:66666666-0006-4000-8000-000000000012",
        "type": "cubby:Reviewer",
        "reviewerRole": "cubby:FinanceSupervisor"
      }
    ]
  }
}
```

---

## 4. SHACL shapes for the three most important classes

The three classes selected are:
1. `cubby:Certificate` (enforces VM-only certification, proposer attribution, bitemporal fact inputs, and forbids prose rationales or truth scores).
2. `cheez:CalibratedThreshold` (enforces that every threshold carries the screen, dataset, encoder, and date that set it).
3. `mowm:RoutingDecision` (enforces calibrated thresholds on routing paths and ensures either a route or a spawn is recorded from execution).

```turtle
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix do:    <https://w3id.org/decision#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix mowm:  <https://w3id.org/cheezml/mowm#> .
@prefix cube:  <https://w3id.org/cheezml/cubelang#> .

# 1. Shape for cubby:Certificate
cubby:CertificateShape
    a sh:NodeShape ;
    sh:targetClass cubby:Certificate ;
    sh:property [
        sh:path prov:wasGeneratedBy ;
        sh:class cube:VMExecution ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "Only a CubeLang VMExecution can generate a Certificate." ;
    ] ;
    sh:property [
        sh:path do:initiates ;
        sh:class do:Question ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path do:considers ;
        sh:minCount 1 ;
        sh:node [
            a sh:NodeShape ;
            sh:property [
                sh:path cubby:proposedBy ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:in (
                    cubby:HostGrammar
                    cubby:EverydayTable
                    cubby:OvernightRuleLibrary
                    cubby:EpisodicMemory
                    cubby:LMProgramEmitter
                ) ;
            ] ;
            sh:property [
                sh:path do:is_satisfied_by ;
                sh:class cheez:CheckResult ;
                sh:minCount 1 ;
            ] ;
        ] ;
    ] .

# 2. Shape for cheez:CalibratedThreshold
cheez:CalibratedThresholdShape
    a sh:NodeShape ;
    sh:targetClass cheez:CalibratedThreshold ;
    sh:property [
        sh:path cheez:thresholdValue ;
        sh:datatype xsd:double ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path cheez:calibratedByScreen ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path cheez:calibrationDataset ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path cheez:calibrationEncoder ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path cheez:calibrationDate ;
        sh:datatype xsd:dateTime ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

# 3. Shape for mowm:RoutingDecision
mowm:RoutingDecisionShape
    a sh:NodeShape ;
    sh:targetClass mowm:RoutingDecision ;
    sh:property [
        sh:path mowm:bestWorldScore ;
        sh:datatype xsd:double ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path mowm:floorThreshold ;
        sh:class cheez:CalibratedThreshold ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path mowm:marginThreshold ;
        sh:class cheez:CalibratedThreshold ;
        sh:maxCount 1 ;
    ] ;
    sh:xone (
        [
            sh:property [
                sh:path mowm:routedToWorld ;
                sh:class mowm:World ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
            ]
        ]
        [
            sh:property [
                sh:path mowm:triggeredSpawn ;
                sh:class mowm:WorldSpawn ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
            ]
        ]
    ) .
```

---

## 5. CubeLang programs for MoWM routing and spawning

**Compiler disclosure:** The brief states the general declaration skeleton `program X implements I { storage {...} functions }` and lists the executing safe subset (`create`, `assign` of a literal, `add`, `sub`, `mul`, `div`, `sum`, `push`, `pop`, `query`, `store`, `recall`, `remember`, `forget`, `compare`, `if`/`else`, `while`, and `ASK`, with flat operands only), plus what does not parse or execute (`for`, `match`, intra-program `call`, early `return`, `throw`, `@hook`, `??`, destructuring, `event`). However, the brief does **not** specify the exact token grammar for storage variable types, function parameter signatures, or the exact syntax of `compare`. Because I do not know the unstated grammar productions of `cubelang check`, both programs below are restricted strictly to the executing safe subset's operations and flat-operand rules, and are explicitly marked **spec only: does not compile today** if any type or statement keyword differs from `cubelang check`'s parser grammar.

### 5.1 MoWM Bridge Router (`MoWMBridgeRouter`)
Implements the rule in Section 4 (`bridges/cubby_bridge.py`):
- Route only if `best_score` clears the calibrated `tau_floor` AND (when `world_count >= 2`) `best_score - runner_up_score` clears the calibrated `tau_margin`.
- Otherwise, flag for spawn (`decision_code = 2`).
- Uses only flat operands, `assign`, `sub`, `compare`, `if`/`else`, `store`, and a single final return.

```cubelang
// Label: spec only: does not compile today (exact storage/function type grammar not stated in brief)
program MoWMBridgeRouter implements ISolver {
    storage {
        world_count: int;
        best_world_id: int;
        best_score: float;
        runner_up_score: float;
        tau_floor: float;
        tau_margin: float;
        margin: float;
        clears_floor: bool;
        has_multiple_worlds: bool;
        clears_margin: bool;
        decision_code: int;
        routed_world_id: int;
    }

    fn route() -> int {
        // decision_code: 1 = route, 2 = spawn
        decision_code = 2;
        routed_world_id = 0;

        compare clears_floor best_score >= tau_floor;

        if clears_floor {
            compare has_multiple_worlds world_count >= 2;
            if has_multiple_worlds {
                sub margin best_score runner_up_score;
                compare clears_margin margin >= tau_margin;
                if clears_margin {
                    decision_code = 1;
                    routed_world_id = best_world_id;
                } else {
                    decision_code = 2;
                }
            } else {
                decision_code = 1;
                routed_world_id = best_world_id;
            }
        } else {
            decision_code = 2;
        }

        store decision_code;
        store routed_world_id;
        return decision_code;
    }
}
```

### 5.2 MoWM Spawner (`MoWMSpawner`)
Implements the two spawn rules in Section 4 (`router.py` and `world.py`):
- **Router-level spawn (`is_subworld == 0`):** spawns when `residual_norm > tau_spawn` (default 0.5), allocating the new world ID from the router counter starting at `5000`.
- **Sub-world split at $z+1$ (`is_subworld == 1`):** checks `parent_z < z_max` (refuses at `z_max` with `spawn_status = 0`) and `residual_norm > tau`, allocating the new world ID from the global world counter starting at `1000` (never `parent_id + 1`) and setting `child_z = parent_z + 1`.

```cubelang
// Label: spec only: does not compile today (exact storage/function type grammar not stated in brief)
program MoWMSpawner implements ISolver {
    storage {
        is_subworld: int;
        global_world_counter: int;
        router_world_counter: int;
        parent_z: int;
        z_max: int;
        residual_norm: float;
        tau_spawn: float;
        tau_subworld: float;
        can_split_z: bool;
        exceeds_tau: bool;
        spawn_status: int;
        spawned_world_id: int;
        spawned_z: int;
    }

    fn maybe_spawn() -> int {
        // spawn_status: 0 = refused/no_spawn, 1 = spawned_router_world, 2 = spawned_subworld
        spawn_status = 0;
        spawned_world_id = 0;
        spawned_z = 0;

        if is_subworld {
            compare can_split_z parent_z < z_max;
            if can_split_z {
                compare exceeds_tau residual_norm > tau_subworld;
                if exceeds_tau {
                    spawned_world_id = global_world_counter;
                    add global_world_counter global_world_counter 1;
                    add spawned_z parent_z 1;
                    spawn_status = 2;
                    remember spawned_world_id;
                } else {
                    spawn_status = 0;
                }
            } else {
                // Refuses at z_max
                spawn_status = 0;
            }
        } else {
            compare exceeds_tau residual_norm > tau_spawn;
            if exceeds_tau {
                spawned_world_id = router_world_counter;
                add router_world_counter router_world_counter 1;
                spawned_z = 0;
                spawn_status = 1;
                remember spawned_world_id;
            } else {
                spawn_status = 0;
            }
        }

        store global_world_counter;
        store router_world_counter;
        store spawned_world_id;
        store spawned_z;
        store spawn_status;
        return spawn_status;
    }
}
```

---

## 6. What I would cut, and open questions

### 6.1 What I would cut from prior drafts
1. **Cut the `container` bridge and orchestrator records/programs.** `container`, `vm.start_from_file`, `vm.wire`, `vm.bridge`, and `@hook` compile to no bytecode or fail to parse today, and hard rule 1 forbids declaring statuses ("Active") for things that never ran.
2. **Cut swarm convergence / multi-agent consensus records.** Hard rule 7 states that agreement never certifies; only the VM certifies. Recording multi-agent convergence as a certification or decision artifact violates this rule.
3. **Cut inline vectors, weight arrays, and seed-only provenance.** Replace every inline float array with `cheez:HashAddressedBlob` (`cheez:sha256`) and a link to the `cheez:TrainingRun`.
4. **Cut all prose rationale properties and boolean "passed/verified" flags.** Any human-readable explanation belongs in UI templates rendered from `cubby:RefusalReason`, `cubby:ConflictKind`, `cheez:checkCode`, and the values read.

### 6.2 Open questions
1. **Does router pruning (`tau_prune_steps`, default 100) retire or delete?** Section 4 notes that whether `router.py` pruning deletes or retires is an open question, whereas Section 2 has the hard rule "Retire, never delete." In CheezML I modelled pruning as a `cheez:Retirement` with reason `mowm:PrunedForIdleness`. Does `router.py` unload the HYLA weights from memory while keeping the `mowm:World` record retired in the store, or does `router.py` need a code change to emit a retirement event?
2. **Canonical URI for the Decision Ontology:** Order 0054 requires the Decision Ontology (`Question`, `Option`, `Decision`, `is_satisfied_by`, `is_violated_by`). The historical W3C Decision Incubator Group namespace is no longer resolvable. What exact namespace URI does order 0054 use in existing code?
3. **Inter-world query hops (`bridges/cubby_bridge.py`):** When the best member of the initially routed world falls short of `tau_answer` and the query passes world-to-world, should the host record an ordered list of every intermediate `mowm:World` probed, or only the initial `mowm:RoutingDecision` and the final answering world?
4. **Exact BNF grammar for `cubelang check`:** To write CubeLang programs that are guaranteed to pass `cubelang check` without the "spec only" caveat, what is the exact syntax for `storage` block declarations, `compare` statements, and `ASK` invocation in the 2026-11-07 compiler?

## deepseek/deepseek-v4-pro-0813

*19532 tokens out, 88.9 s, finish `stop`*

# CheezML – one record vocabulary for CubbyLLM, MoWM and CubeLang

## 1. Classes and properties

Terms are listed under their module. “Minted” means a CheezML term. Where a standard term does the same job, the standard term is used directly in the records and the “Standard term” column says which one; no CheezML term is minted for it. Domain and range are given for properties. For classes, domain/range is “—”.

A note on the Decision Ontology: the namespace is assumed to be `http://www.w3.org/2001/sw/hcls/DecisionOntology#`. If that is not the Decision Ontology namespace, replace it; the vocabulary does not redefine its terms.

### core

| Term | Module | Domain → Range | Definition | Standard term |
|---|---|---|---|---|
| `cheez:Record` | core | — | Any record written by the host from what ran. | `prov:Entity` |
| `cheez:FactRecord` | core | — | A stored fact with both valid time and recorded time. | `prov:Entity` |
| `cheez:ThresholdCalibration` | core | — | A threshold plus the screen, data and date that set it. | none |
| `cheez:hasCalibration` | core | `cheez:Record` → `cheez:ThresholdCalibration` | Attaches a calibration to a record that uses a threshold. | none |
| `cheez:hasValidTime` | core | `cheez:FactRecord` → `time:TemporalEntity` | When the fact held. | `time:hasTime` (specialised) |
| `cheez:hasRecordedTime` | core | `cheez:FactRecord` → `xsd:dateTime` | When the store learned the fact. | `prov:generatedAtTime` |
| `cheez:retiredBy` | core | `cheez:Record` → `cheez:Record` | The record that retires, rejects or prunes the subject. | `prov:wasInvalidatedBy` |
| `cheez:hasRetirementReason` | core | `cheez:Record` → `skos:Concept` | The coded reason a thing is retired, rejected or pruned. | none |

### cubby

| Term | Module | Domain → Range | Definition | Standard term |
|---|---|---|---|---|
| `cubby:DialogueTurn` | cubby | — | One turn in the conversation state kept by the host. | `prov:Entity` |
| `cubby:CertifiedAnswerTurn` | cubby | — | A turn whose answer carries a certificate. | `cubby:DialogueTurn` |
| `cubby:RefusalTurn` | cubby | — | A turn that refuses the question. | `cubby:DialogueTurn` |
| `cubby:ClarifyingQuestionTurn` | cubby | — | A turn that asks a clarifying question. | `cubby:DialogueTurn` |
| `cubby:FollowUpTurn` | cubby | — | A follow-up turn with its rewrite. | `cubby:DialogueTurn` |
| `cubby:SmallTalkTurn` | cubby | — | A small-talk turn. | `cubby:DialogueTurn` |
| `cubby:hasProposer` | cubby | `cubby:DialogueTurn` → `prov:Agent` | The proposer that produced the turn. | `prov:wasAttributedTo` |
| `cubby:hasDialogueAct` | cubby | `cubby:DialogueTurn` → `skos:Concept` | The dialogue act, drawn from the in-use set. | `skos:Concept` |
| `cubby:hasRefusalReason` | cubby | `cubby:RefusalTurn` → `skos:Concept` | One of the in-use refusal reasons. | none |
| `cubby:hasConflictKind` | cubby | `cubby:RefusalTurn` → `skos:Concept` | One of the in-use conflict kinds. | none |
| `cubby:hasCertificate` | cubby | `cubby:CertifiedAnswerTurn` → `cheez:Certificate` | The certificate attached to a certified answer. | none |
| `cubby:hasRewrite` | cubby | `cubby:FollowUpTurn` → `xsd:string` | The follow-up rewrite that passed the entity guard. | none |
| `cubby:EmotionReading` | cubby | — | An inference about a user’s emotion, with confidence. | `emotionml:emotion` (linked) |
| `cubby:hasEmotionReading` | cubby | `cubby:DialogueTurn` → `cubby:EmotionReading` | An emotion reading attached to a turn. | none |
| `cubby:hasPleasure` | cubby | `cubby:EmotionReading` → `xsd:float` | Pleasure value in the pad-dimensions set, in [0,1]. | none |
| `cubby:hasArousal` | cubby | `cubby:EmotionReading` → `xsd:float` | Arousal value in the pad-dimensions set, in [0,1]. | none |
| `cubby:hasDominance` | cubby | `cubby:EmotionReading` → `xsd:float` | Dominance value in the pad-dimensions set, in [0,1]. | none |
| `cubby:hasConfidence` | cubby | `cubby:EmotionReading` → `xsd:float` | Confidence of the emotion inference. | none |
| `cubby:hasDimensionSet` | cubby | `cubby:EmotionReading` → `xsd:string` | Dimension set used, e.g. `pad-dimensions` or `cubby-hormones`. | none |
| `cubby:HelperProposal` | cubby | — | A helper proposed by one of Cubby’s proposers. | `prov:Entity` |
| `cubby:HelperSandboxTest` | cubby | — | A helper test record from a locked-down VM session. | `prov:Activity` |
| `cubby:HelperRejectionRecord` | cubby | — | The record of a helper rejected after sandbox testing. | `prov:Entity` |
| `cubby:hasOrder` | cubby | `cubby:HelperRejectionRecord` → `xsd:string` | The order number under which the helper was proposed. | none |
| `cubby:hasBrokenCase` | cubby | `cubby:HelperRejectionRecord` → `prov:Entity` | The case that broke the helper. | none |
| `cubby:hasEscapeAttempts` | cubby | `cubby:HelperRejectionRecord` → `xsd:integer` | Escape attempts observed during sandbox testing; must be 0 for adoption. | none |
| `cubby:hasRejectionRate` | cubby | `cubby:HelperProposal` → `xsd:float` | Rejection rate over the evaluation window. | none |
| `cubby:DocumentFieldRecord` | cubby | — | A field record from an invoice or similar document. | `cheez:FactRecord` |
| `cubby:DocumentRegion` | cubby | — | A region of a document with text and reader version. | `prov:Entity` |
| `cubby:hasRegion` | cubby | `cubby:DocumentFieldRecord` → `cubby:DocumentRegion` | The document region containing the field. | none |
| `cubby:hasRegionText` | cubby | `cubby:DocumentRegion` → `xsd:string` | The text of the region. | none |
| `cubby:hasReaderVersion` | cubby | `cubby:DocumentRegion` → `xsd:string` | The reader version used for the region. | none |
| `cubby:hasExactCheck` | cubby | `cubby:DocumentFieldRecord` → `prov:Activity` | An exact check that ran on the field. | none |
| `cubby:hasDerivedSuggestion` | cubby | `cubby:DocumentFieldRecord` → `xsd:string` | A derived value, recorded only as a suggestion. | none |
| `cubby:hasReviewer` | cubby | `cubby:DocumentFieldRecord` → `prov:Agent` | A human reviewer who acted as a source. | `prov:wasAttributedTo` |
| `cubby:hasApprovalAmount` | cubby | `cubby:DocumentFieldRecord` → `xsd:decimal` | The amount above which a second approval is required. | none |
| `cubby:hasSecondApprover` | cubby | `cubby:DocumentFieldRecord` → `prov:Agent` | The second approver, if required. | none |
| `cubby:SplitQuestionRecord` | cubby | — | A split question, with parts, routes and part verdicts. | `prov:Entity` |
| `cubby:hasPart` | cubby | `cubby:SplitQuestionRecord` → `prov:Entity` | One part of a split question. | none |
| `cubby:hasRoute` | cubby | `cubby:SplitQuestionRecord` → `prov:Entity` | One route used for a part. | none |
| `cubby:hasPartVerdict` | cubby | `cubby:SplitQuestionRecord` → `skos:Concept` | The verdict for one part. | none |
| `cubby:hasCombinationProgram` | cubby | `cubby:SplitQuestionRecord` → `prov:Entity` | The program that combines part verdicts. | none |
| `cubby:StoryWorldChangeRecord` | cubby | — | A dated state change in a story world. | `prov:Entity` |
| `cubby:hasOldValue` | cubby | `cubby:StoryWorldChangeRecord` → `xsd:string` | The old value before the change. | none |
| `cubby:hasNewValue` | cubby | `cubby:StoryWorldChangeRecord` → `xsd:string` | The new value after the change. | none |
| `cubby:hasTurnNumber` | cubby | `cubby:StoryWorldChangeRecord` → `xsd:integer` | The turn at which the change occurred. | none |
| `cubby:hasInventedSource` | cubby | `cubby:StoryWorldChangeRecord` → `xsd:string` | Source marker, e.g. `invented at turn N`. | none |

### mowm

| Term | Module | Domain → Range | Definition | Standard term |
|---|---|---|---|---|
| `mowm:World` | mowm | — | A MoWM world tree node. | `prov:Entity` |
| `mowm:RouterDecision` | mowm | — | A routing decision made by DSelect-k. | `prov:Activity` |
| `mowm:SpawnDecision` | mowm | — | A routing decision that ends in a spawn. | `mowm:RouterDecision` |
| `mowm:PruneDecision` | mowm | — | A decision to prune a world. | `mowm:RouterDecision` |
| `mowm:WorldSplit` | mowm | — | The record of a sub-world created at z+1. | `prov:Entity` |
| `mowm:PossibilityOracleQuery` | mowm | — | A query handled by the possibility oracle. | `prov:Activity` |
| `mowm:hasWorldId` | mowm | `mowm:World` → `xsd:integer` | The world id from the global counter. | none |
| `mowm:hasZDepth` | mowm | `mowm:World` → `xsd:integer` | The world’s z depth. | none |
| `mowm:hasZMax` | mowm | `mowm:World` → `xsd:integer` | The maximum allowed z depth for this world. | none |
| `mowm:hasTau` | mowm | `mowm:World` → `xsd:float` | The confidence threshold for spawning sub-worlds. | none |
| `mowm:hasTauSpawn` | mowm | `mowm:SpawnDecision` → `xsd:float` | The residual-norm threshold for spawning. | none |
| `mowm:hasTauPruneSteps` | mowm | `mowm:World` → `xsd:integer` | Idle steps before pruning is considered. | none |
| `mowm:hasResidualNorm` | mowm | `mowm:SpawnDecision` → `xsd:float` | The residual norm observed at the time of the decision. | none |
| `mowm:hasCentroidScore` | mowm | `mowm:RouterDecision` → `xsd:float` | Best world centroid score. | none |
| `mowm:hasMargin` | mowm | `mowm:RouterDecision` → `xsd:float` | Margin over the runner-up centroid score. | none |
| `mowm:hasRouteFloor` | mowm | `mowm:RouterDecision` → `xsd:float` | The absolute centroid-score floor. | none |
| `mowm:hasSelectedWorld` | mowm | `mowm:RouterDecision` → `mowm:World` | The world selected for routing, when routing happened. | none |
| `mowm:hasSpawnedWorld` | mowm | `mowm:SpawnDecision` → `mowm:World` | The world created by the spawn. | none |
| `mowm:routeOutcome` | mowm | `mowm:RouterDecision` → `skos:Concept` | Outcome of routing: `route`, `spawn`, or `query-pass`. | none |
| `mowm:hasWeightsHash` | mowm | `mowm:World` → `xsd:string` | Hash of trained weights, never the weights themselves. | none |
| `mowm:hasVectorHash` | mowm | `mowm:World` → `xsd:string` | Hash of a sparse block code vector. | none |
| `mowm:hasWordTableHash` | mowm | `mowm:PossibilityOracleQuery` → `xsd:string` | Hash of the word-table encoder used. | none |
| `mowm:hasHYLACount` | mowm | `mowm:World` → `xsd:integer` | Number of HYLAs held by the world. | none |
| `mowm:hasMergeStrategy` | mowm | `mowm:World` → `xsd:string` | Merge strategy name, e.g. `BindComposeMerge`. | none |
| `mowm:hasCleanupStrategy` | mowm | `mowm:RouterDecision` → `xsd:string` | Cleanup strategy name. | none |
| `mowm:hasTopK` | mowm | `mowm:PossibilityOracleQuery` → `xsd:integer` | Top-k members used in prediction cleanup. | none |

### cube

| Term | Module | Domain → Range | Definition | Standard term |
|---|---|---|---|---|
| `cube:ProgramRecord` | cube | — | A compiled or executed CubeLang program. | `prov:Entity` |
| `cube:ExecutionOutcome` | cube | — | The outcome of executing a program: return, error, or ask. | `prov:Activity` |
| `cube:hasOpcode` | cube | `cube:ProgramRecord` → `xsd:hexBinary` | A real opcode byte from the synced set. | none |
| `cube:hasReturnValue` | cube | `cube:ExecutionOutcome` → `xsd:string` | Value returned by a completed program. | none |
| `cube:hasError` | cube | `cube:ExecutionOutcome` → `xsd:string` | Error text, if the program errored. | none |
| `cube:hasAsk` | cube | `cube:ExecutionOutcome` → `xsd:string` | The ask payload when a program suspends. | none |
| `cube:hasSafeSubset` | cube | `cube:ProgramRecord` → `xsd:boolean` | True if the program uses only the executing subset. | none |
| `cube:hasCompilationStatus` | cube | `cube:ProgramRecord` → `skos:Concept` | Whether it compiles today. | none |

Standard vocabularies reused directly and not minted above:

- `prov:Entity`, `prov:Activity`, `prov:Agent`, `prov:used`, `prov:wasGeneratedBy`, `prov:wasAttributedTo`, `prov:generatedAtTime`, `prov:wasInvalidatedBy`.
- `decision:Question`, `decision:Option`, `decision:Decision`, `decision:is_satisfied_by`, `decision:is_violated_by`.
- `schema:Person`, `schema:Organization`, `schema:identifier`, `schema:name`.
- `time:Instant`, `time:Interval`, `time:inXSDDateTimeStamp`.
- `skos:Concept`, `skos:prefLabel`, `skos:inScheme`.
- `rocrate:WorkflowRun` for reproducibility package references.
- `emotionml:pad-dimensions` for emotion dimension set linking.

## 2. JSON-LD 1.1 `@context`

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "https://schema.org/",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rocrate": "https://w3id.org/ro/crate/1.1/",
    "emotionml": "https://www.w3.org/TR/emotionml/",
    "decision": "http://www.w3.org/2001/sw/hcls/DecisionOntology#",
    "Record": "cheez:Record",
    "FactRecord": "cheez:FactRecord",
    "hasCalibration": "cheez:hasCalibration",
    "hasValidTime": "cheez:hasValidTime",
    "hasRecordedTime": "cheez:hasRecordedTime",
    "retiredBy": "cheez:retiredBy",
    "hasRetirementReason": "cheez:hasRetirementReason",
    "DialogueTurn": "cubby:DialogueTurn",
    "CertifiedAnswerTurn": "cubby:CertifiedAnswerTurn",
    "RefusalTurn": "cubby:RefusalTurn",
    "hasProposer": "cubby:hasProposer",
    "hasDialogueAct": "cubby:hasDialogueAct",
    "hasRefusalReason": "cubby:hasRefusalReason",
    "hasConflictKind": "cubby:hasConflictKind",
    "hasCertificate": "cubby:hasCertificate",
    "hasRewrite": "cubby:hasRewrite",
    "EmotionReading": "cubby:EmotionReading",
    "hasEmotionReading": "cubby:hasEmotionReading",
    "hasPleasure": "cubby:hasPleasure",
    "hasArousal": "cubby:hasArousal",
    "hasDominance": "cubby:hasDominance",
    "hasConfidence": "cubby:hasConfidence",
    "hasDimensionSet": "cubby:hasDimensionSet",
    "HelperProposal": "cubby:HelperProposal",
    "HelperSandboxTest": "cubby:HelperSandboxTest",
    "HelperRejectionRecord": "cubby:HelperRejectionRecord",
    "hasOrder": "cubby:hasOrder",
    "hasBrokenCase": "cubby:hasBrokenCase",
    "hasEscapeAttempts": "cubby:hasEscapeAttempts",
    "hasRejectionRate": "cubby:hasRejectionRate",
    "DocumentFieldRecord": "cubby:DocumentFieldRecord",
    "DocumentRegion": "cubby:DocumentRegion",
    "hasRegion": "cubby:hasRegion",
    "hasRegionText": "cubby:hasRegionText",
    "hasReaderVersion": "cubby:hasReaderVersion",
    "hasExactCheck": "cubby:hasExactCheck",
    "hasDerivedSuggestion": "cubby:hasDerivedSuggestion",
    "hasReviewer": "cubby:hasReviewer",
    "hasApprovalAmount": "cubby:hasApprovalAmount",
    "hasSecondApprover": "cubby:hasSecondApprover",
    "SplitQuestionRecord": "cubby:SplitQuestionRecord",
    "hasPart": "cubby:hasPart",
    "hasRoute": "cubby:hasRoute",
    "hasPartVerdict": "cubby:hasPartVerdict",
    "hasCombinationProgram": "cubby:hasCombinationProgram",
    "StoryWorldChangeRecord": "cubby:StoryWorldChangeRecord",
    "hasOldValue": "cubby:hasOldValue",
    "hasNewValue": "cubby:hasNewValue",
    "hasTurnNumber": "cubby:hasTurnNumber",
    "hasInventedSource": "cubby:hasInventedSource",
    "World": "mowm:World",
    "RouterDecision": "mowm:RouterDecision",
    "SpawnDecision": "mowm:SpawnDecision",
    "PruneDecision": "mowm:PruneDecision",
    "WorldSplit": "mowm:WorldSplit",
    "hasWorldId": "mowm:hasWorldId",
    "hasZDepth": "mowm:hasZDepth",
    "hasZMax": "mowm:hasZMax",
    "hasTau": "mowm:hasTau",
    "hasTauSpawn": "mowm:hasTauSpawn",
    "hasTauPruneSteps": "mowm:hasTauPruneSteps",
    "hasResidualNorm": "mowm:hasResidualNorm",
    "hasCentroidScore": "mowm:hasCentroidScore",
    "hasMargin": "mowm:hasMargin",
    "hasRouteFloor": "mowm:hasRouteFloor",
    "hasSelectedWorld": "mowm:hasSelectedWorld",
    "hasSpawnedWorld": "mowm:hasSpawnedWorld",
    "routeOutcome": "mowm:routeOutcome",
    "hasWeightsHash": "mowm:hasWeightsHash",
    "hasVectorHash": "mowm:hasVectorHash",
    "hasWordTableHash": "mowm:hasWordTableHash",
    "hasHYLACount": "mowm:hasHYLACount",
    "hasMergeStrategy": "mowm:hasMergeStrategy",
    "hasCleanupStrategy": "mowm:hasCleanupStrategy",
    "hasTopK": "mowm:hasTopK",
    "ProgramRecord": "cube:ProgramRecord",
    "ExecutionOutcome": "cube:ExecutionOutcome",
    "hasOpcode": "cube:hasOpcode",
    "hasReturnValue": "cube:hasReturnValue",
    "hasError": "cube:hasError",
    "hasAsk": "cube:hasAsk",
    "hasSafeSubset": "cube:hasSafeSubset",
    "hasCompilationStatus": "cube:hasCompilationStatus"
  }
}
```

## 3. Six example records

All records use the context above. Unspecified values are strings beginning `PLACEHOLDER:`.

### 1. Certified answer turn with certificate

```json
{
  "@id": "urn:uuid:PLACEHOLDER-certified-turn",
  "@type": ["cubby:CertifiedAnswerTurn", "prov:Entity"],
  "cubby:hasProposer": {"@id": "PLACEHOLDER:proposer-id"},
  "cubby:hasDialogueAct": {"@id": "cubby:certified-answer"},
  "prov:generatedAtTime": "PLACEHOLDER:completion-time",
  "cubby:hasCertificate": {
    "@id": "urn:uuid:PLACEHOLDER-certificate",
    "@type": "cheez:Certificate",
    "decision:question": {"@id": "urn:uuid:PLACEHOLDER-question"},
    "decision:option": {"@id": "PLACEHOLDER:proposer-program-id"},
    "prov:used": [
      {
        "@id": "urn:uuid:PLACEHOLDER-statement-1",
        "@type": "prov:Entity",
        "schema:identifier": "PLACEHOLDER:source-and-revision"
      },
      {
        "@id": "urn:uuid:PLACEHOLDER-statement-2",
        "@type": "prov:Entity",
        "schema:identifier": "PLACEHOLDER:source-and-revision"
      }
    ],
    "prov:used": {
      "@id": "urn:uuid:PLACEHOLDER-rules-applied",
      "@type": "prov:Entity"
    },
    "decision:is_satisfied_by": {"@id": "PLACEHOLDER:check-id-satisfied"},
    "decision:is_violated_by": {"@id": "PLACEHOLDER:check-id-violated"},
    "decision:Decision": {
      "@id": "urn:uuid:PLACEHOLDER-decision",
      "decision:option": {"@id": "PLACEHOLDER:proposer-program-id"}
    }
  },
  "prov:wasGeneratedBy": {"@id": "urn:uuid:PLACEHOLDER-program-run-activity"}
}
```

### 2. Refusal with reason

```json
{
  "@id": "urn:uuid:PLACEHOLDER-refusal-turn",
  "@type": ["cubby:RefusalTurn", "prov:Entity"],
  "cubby:hasProposer": {"@id": "PLACEHOLDER:proposer-id"},
  "cubby:hasDialogueAct": {"@id": "cubby:refusal"},
  "cubby:hasRefusalReason": {"@id": "cubby:fact-missing"},
  "cubby:hasConflictKind": {"@id": "cubby:logical-conflict"},
  "prov:generatedAtTime": "PLACEHOLDER:completion-time"
}
```

### 3. MoWM routing decision that ends in a spawn

```json
{
  "@id": "urn:uuid:PLACEHOLDER-spawn-decision",
  "@type": ["mowm:SpawnDecision", "mowm:RouterDecision", "prov:Activity"],
  "mowm:routeOutcome": {"@id": "mowm:spawn"},
  "mowm:hasCentroidScore": "PLACEHOLDER:best-centroid-score",
  "mowm:hasMargin": "PLACEHOLDER:runner-up-margin",
  "mowm:hasRouteFloor": "PLACEHOLDER:calibrated-absolute-floor",
  "mowm:hasTauSpawn": "PLACEHOLDER:tau-spawn-value",
  "mowm:hasResidualNorm": "PLACEHOLDER:residual-norm-value",
  "mowm:hasCalibration": {
    "@id": "urn:uuid:PLACEHOLDER-threshold-calibration",
    "@type": "cheez:ThresholdCalibration",
    "schema:identifier": "PLACEHOLDER:screen-name",
    "schema:name": "PLACEHOLDER:calibration-data-set",
    "prov:generatedAtTime": "PLACEHOLDER:calibration-date"
  },
  "mowm:hasSpawnedWorld": {
    "@id": "urn:uuid:PLACEHOLDER-spawned-world",
    "@type": "mowm:World",
    "mowm:hasWorldId": "PLACEHOLDER:router-counter-id-greater-than-5000",
    "mowm:hasZDepth": "PLACEHOLDER:z"
  }
}
```

### 4. World split: sub-world at z+1

```json
{
  "@id": "urn:uuid:PLACEHOLDER-world-split",
  "@type": ["mowm:WorldSplit", "prov:Entity"],
  "mowm:hasParentWorld": {
    "@id": "urn:uuid:PLACEHOLDER-parent-world",
    "@type": "mowm:World",
    "mowm:hasWorldId": "PLACEHOLDER:parent-id-from-1000-counter",
    "mowm:hasZDepth": "PLACEHOLDER:z"
  },
  "mowm:hasSpawnedWorld": {
    "@id": "urn:uuid:PLACEHOLDER-child-world",
    "@type": "mowm:World",
    "mowm:hasWorldId": "PLACEHOLDER:child-id-from-1000-counter",
    "mowm:hasZDepth": "PLACEHOLDER:z-plus-1",
    "mowm:hasZMax": "PLACEHOLDER:z-max",
    "mowm:hasTau": "PLACEHOLDER:tau",
    "mowm:hasHYLACount": "PLACEHOLDER:number-of-hylas",
    "mowm:hasMergeStrategy": "PLACEHOLDER:merge-strategy-name"
  }
}
```

### 5. Helper rejected in its sandbox (order 0062)

```json
{
  "@id": "urn:uuid:PLACEHOLDER-helper-rejection",
  "@type": ["cubby:HelperRejectionRecord", "prov:Entity"],
  "cubby:hasOrder": "0062",
  "cubby:hasProposer": {"@id": "PLACEHOLDER:helper-proposer-id"},
  "cubby:hasBrokenCase": {"@id": "PLACEHOLDER:case-that-broke-the-helper"},
  "cubby:hasEscapeAttempts": 0,
  "cubby:hasRejectionRate": "PLACEHOLDER:rejection-rate-over-window",
  "cubby:hasRetirementReason": {"@id": "cubby:reproduce-failure"},
  "cheez:retiredBy": {"@id": "urn:uuid:PLACEHOLDER-promotion-decision-record"},
  "prov:generatedAtTime": "PLACEHOLDER:rejection-time"
}
```

### 6. Invoice field sent to human review and then committed (order 0061)

```json
{
  "@id": "urn:uuid:PLACEHOLDER-invoice-field-record",
  "@type": ["cubby:DocumentFieldRecord", "cheez:FactRecord"],
  "cubby:hasRegion": {
    "@id": "urn:uuid:PLACEHOLDER-document-region",
    "@type": "cubby:DocumentRegion",
    "cubby:hasRegionText": "PLACEHOLDER:region-text",
    "cubby:hasReaderVersion": "PLACEHOLDER:reader-version"
  },
  "cubby:hasExactCheck": [
    {"@id": "PLACEHOLDER:format-check", "@type": "prov:Activity"},
    {"@id": "PLACEHOLDER:arithmetic-check", "@type": "prov:Activity"},
    {"@id": "PLACEHOLDER:purchase-order-check", "@type": "prov:Activity"},
    {"@id": "PLACEHOLDER:supplier-check", "@type": "prov:Activity"},
    {"@id": "PLACEHOLDER:duplicate-check", "@type": "prov:Activity"}
  ],
  "cubby:hasDerivedSuggestion": "PLACEHOLDER:derived-value-suggestion-only",
  "cubby:hasReviewer": {
    "@id": "PLACEHOLDER:human-reviewer-id",
    "@type": ["prov:Agent", "schema:Person"],
    "schema:identifier": "PLACEHOLDER:reviewer-id",
    "schema:name": "PLACEHOLDER:reviewer-name"
  },
  "cubby:hasApprovalAmount": "PLACEHOLDER:amount",
  "cubby:hasSecondApprover": {
    "@id": "PLACEHOLDER:second-approver-id",
    "@type": ["prov:Agent", "schema:Person"]
  },
  "cheez:hasValidTime": {
    "@type": "time:Interval",
    "time:inXSDDateTimeStamp": "PLACEHOLDER:valid-start"
  },
  "cheez:hasRecordedTime": "PLACEHOLDER:recorded-time",
  "prov:generatedAtTime": "PLACEHOLDER:commit-time"
}
```

## 4. SHACL shapes for the three most important classes

I judge these three most important: `cheez:FactRecord` (two-clock rule), `cubby:CertifiedAnswerTurn` (no truth score, certificate required), and `mowm:SpawnDecision` (threshold calibration and spawn record required).

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix mowm: <https://w3id.org/cheezml/mowm#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cheez:FactRecordShape a sh:NodeShape ;
  sh:targetClass cheez:FactRecord ;
  sh:property [
    sh:path cheez:hasValidTime ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cheez:hasRecordedTime ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:dateTime
  ] ;
  sh:property [
    sh:path prov:generatedAtTime ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:dateTime
  ] .

cubby:CertifiedAnswerTurnShape a sh:NodeShape ;
  sh:targetClass cubby:CertifiedAnswerTurn ;
  sh:property [
    sh:path cubby:hasCertificate ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cubby:hasProposer ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path cubby:hasDialogueAct ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:not [
    sh:property [
      sh:path schema:truthScore
    ]
  ] ;
  sh:not [
    sh:property [
      sh:path cubby:hasProseRationale
    ]
  ] .

mowm:SpawnDecisionShape a sh:NodeShape ;
  sh:targetClass mowm:SpawnDecision ;
  sh:property [
    sh:path mowm:hasTauSpawn ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:float
  ] ;
  sh:property [
    sh:path mowm:hasResidualNorm ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:float
  ] ;
  sh:property [
    sh:path mowm:hasCalibration ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path mowm:hasSpawnedWorld ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] ;
  sh:property [
    sh:path mowm:routeOutcome ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:nodeKind sh:IRI
  ] .
```

The shape `cubby:CertifiedAnswerTurnShape` uses `schema:truthScore` in a `sh:not` constraint; `schema` must be declared. The property `cubby:hasProseRationale` is explicitly not in the vocabulary. If a SHACL engine rejects unknown properties in `sh:not`, remove that triple; the prohibition is enforceable by the vocabulary design itself.

## 5. CubeLang programs for MoWM routing and spawning

Both programs below are **spec only: does not compile today**. The interface and ASK syntax are not fully pinned down by the brief, and the compiler may reject the declared interfaces. They use only the executing subset and ASK, but I do not claim they compile without your check.

### Routing program

```
program MoWMRouter implements IRouter {
  storage {
    float best_score;
    float second_score;
    float floor_value;
    float margin_value;
    int best_world;
  }
  functions {
    function route(float centroid_best, float centroid_second, float floor, float margin) -> int {
      create float diff;
      assign diff, centroid_best;
      sub diff, centroid_second;
      if compare(centroid_best, floor) == LESS or compare(diff, margin) == LESS {
        ASK "spawn";
        RETURN -1;
      }
      RETURN best_world;
    }
  }
}
```

### Spawning program

```
program MoWMSpawner implements ISpawner {
  storage {
    float residual;
    float tau_spawn;
    int z;
    int z_max;
  }
  functions {
    function spawn(float residual_norm, float tau_spawn_val, int current_z, int z_max_val) -> int {
      if compare(residual_norm, tau_spawn_val) == GREATER and compare(current_z, z_max_val) == LESS {
        ASK "spawn_subworld";
        RETURN 1;
      }
      RETURN 0;
    }
  }
}
```

The comparison result codes (`LESS`, `GREATER`) are not pinned down; they are placeholders for the compiler’s real result representation. The word-table lookup is not inside these programs: it happens in the host’s FastWordEncoder before MoWM receives encoded facts.

## 6. What I would cut, and open questions

**What I would cut**

- Cut any separate class for each dialogue act. Use `skos:Concept` values under `cubby:hasDialogueAct` instead; only the turn types remain as classes.
- Cut per-strategy classes (`BindComposeMerge`, cleanup strategies). Record the strategy name as a string property; the strategy is not an entity the host writes facts about.
- Cut all vector, weight and blob embedding fields. Only the hash and the training-run record are written; nothing embeds them.
- Cut a `truthScore` or `correct` property. It is forbidden; instead, certificate checks and refusal reasons are the only recorded outcomes.
- Cut a `status` property with values like `PASSED`, `Active`, or `Certified` based on agent agreement. Statuses are host-written from what ran, and agreement never certifies.
- Cut a `rationale` string field. Every readable sentence is produced by a template from codes and values; no prose model reasoning is stored.

**Open questions**

- The exact Decision Ontology namespace is not stated in the brief. I have used one; the panel must confirm it.
- The exact EmotionML link target for RDF is not specified; the context uses the HTML spec URL, but a proper namespace or annotation property is unknown.
- Whether pruning deletes or retires worlds is open. The record vocabulary must not assume deletion; until decided, preserve a retirement record.
- The CubeLang program syntax for `ASK` and interface declaration is not stated with enough detail for me to guarantee compilation. I have labelled the programs spec only.
- The statement “Every number in a field must appear in its region’s text” from order 0061 needs a representation for the found numbers; the current `DocumentFieldRecord` uses a derived-suggestion string, which may be too weak.
- The exact mapping of the in-use refusal reasons and conflict kinds to `skos:Concept` identifiers is not given; I have used placeholder concept names like `cubby:fact-missing`.
- The two-clock rule applies to facts; it is unclear whether every `DocumentFieldRecord` should also be a `cheez:FactRecord`. I have included it, but this is open.
- The helper proposer identity (which of Cubby’s proposers proposed the helper) and the sandbox session identifier are not modelled in detail; the example uses placeholder IRIs.

## qwen/qwen3.8-max-prime

*12013 tokens out, 98.7 s, finish `stop`*

# CheezML – One Record Vocabulary

## 1. Classes and Properties

All terms live under `https://w3id.org/cheezml/`. Records themselves are identified by `urn:uuid:` IRIs and are never placed in the vocabulary namespace.

### Module `core#` (prefix `cheez:`)

| Term | Kind | Domain → Range | Definition | Reuses / specialises |
|---|---|---|---|---|
| `cheez:Record` | Class | – | Base class for every host-written record. | `prov:Entity` |
| `cheez:validTime` | Property | `cheez:Record` → `time:Instant` | When the stated fact held in the world. | OWL-Time `time:Instant` |
| `cheez:recordedTime` | Property | `cheez:Record` → `time:Instant` | When the store learned the fact. | OWL-Time `time:Instant` |
| `cheez:supersededBy` | Property | `cheez:Record` → `cheez:Record` | Closes the old version on correction; the old record is retained. | PROV-O `prov:wasRevisionOf` (inverse direction) |
| `cheez:retired` | Property | `cheez:Record` → `xsd:boolean` | Marks a retired item; retired items are never deleted. | – (no standard fit) |
| `cheez:retirementReason` | Property | `cheez:Record` → `skos:Concept` | Coded reason for retirement, rejection or pruning. | SKOS `skos:Concept` |
| `cheez:hashRef` | Property | `cheez:Record` → `xsd:string` | SHA-256 (or stated algorithm) reference to a vector, weight blob or binary payload. Never the payload itself. | RO-Crate `rocrate:contentUrl` pattern |
| `cheez:Threshold` | Class | – | A threshold together with its full calibration provenance. | – |
| `cheez:calibrationScreen` | Property | `cheez:Threshold` → `xsd:string` | The screen (evaluation run) that set the threshold. | PROV-O `prov:wasGeneratedBy` |
| `cheez:calibrationData` | Property | `cheez:Threshold` → `cheez:hashRef` | Hash of the calibration dataset. | – |
| `cheez:calibrationDate` | Property | `cheez:Threshold` → `xsd:date` | Date the calibration was performed. | – |
| `cheez:value` | Property | `cheez:Threshold` → `xsd:decimal` | The numeric threshold value. | – |
| `cheez:encoder` | Property | `cheez:Threshold` → `xsd:string` | The encoder the threshold was calibrated for; thresholds do not transfer between encoders. | – |
| `cheez:corpus` | Property | `cheez:Threshold` → `xsd:string` | The corpus the threshold was calibrated on. | – |
| `cheez:writtenByHost` | Property | `cheez:Record` → `xsd:boolean` | Asserts the record was produced by the host from what ran. Always true for valid records. | PROV-O `prov:wasGeneratedBy` |

### Module `cubby#` (prefix `cubby:`)

| Term | Kind | Domain → Range | Definition | Reuses / specialises |
|---|---|---|---|---|
| `cubby:Turn` | Class | – | One turn in a conversation. | schema.org `schema:Action` |
| `cubby:dialogueAct` | Property | `cubby:Turn` → `skos:Concept` | The act type: question, certified-answer, refusal, clarifying-question, follow-up, small-talk. Reference set: ISO 24617-2. | SKOS concept from ISO 24617-2 |
| `cubby:CertifiedAnswer` | Class | – | An answer certified by the VM. "Certified" means the answer follows from the recorded facts and rules under the recorded checks; it does not mean "correct". | PROV-O `prov:Entity`; Decision Ontology `Decision` |
| `cubby:certificate` | Property | `cubby:CertifiedAnswer` → `cubby:Certificate` | Links the answer to its PROV-O / Decision-Ontology certificate. | PROV-O `prov:wasGeneratedBy` |
| `cubby:Certificate` | Class | – | The PROV-O certificate: the answer entity, the program-run activity, statements read (source, revision), rules applied, and the Decision-Ontology decision step. | PROV-O `prov:Activity`; Decision Ontology |
| `cubby:proposer` | Property | `cubby:Turn` → `skos:Concept` | Which proposer produced the answer: host-grammar, everyday-table, overnight-rule-library, episodic-memory, lm-program-emitter. | SKOS |
| `cubby:Refusal` | Class | – | A refusal turn. | Decision Ontology `Decision` |
| `cubby:refusalReason` | Property | `cubby:Refusal` → `skos:Concept` | Coded reason: no-plan, relation-missing, fact-missing, walk-failed, not-certified, ambiguous, value-conflict, type-conflict, temporal-conflict, logical-conflict, recorded-vs-derived, dates-too-coarse, unreadable-source. | SKOS |
| `cubby:EmotionReading` | Class | – | An inferred emotion reading for a user, with confidence. Never a fact. | EmotionML `emotionml:Emotion` |
| `cubby:confidence` | Property | `cubby:EmotionReading` → `xsd:decimal` | Confidence of the inference, in [0,1]. | – |
| `cubby:padDimensions` | Property | `cubby:EmotionReading` → `xsd:string` | The PAD dimension set (pleasure, arousal, dominance), values in [0,1]. | EmotionML dimension set |
| `cubby:HormoneState` | Class | – | Cubby's own internal hormone state; a separate labelled dimension set from user emotion. | EmotionML (by analogy; separate dimension set) |
| `cubby:StoryWorld` | Class | – | An isolated story world, separate from the real-world store. | – |
| `cubby:stateChange` | Property | `cubby:StoryWorld` → `cubby:StateChange` | A dated state change with old value, new value and turn. | – |
| `cubby:inventedAtTurn` | Property | `cubby:StoryWorld` → `xsd:integer` | Marks an invented fact with the turn number of invention. | – |
| `cubby:DocumentRegion` | Class | – | A region of a document with its text and the reader's version. | RO-Crate `rocrate:DataEntity` |
| `cubby:exactCheck` | Property | `cubby:DocumentRegion` → `skos:Concept` | An exact check: format, arithmetic, purchase-order, supplier, duplicate. | SKOS |
| `cubby:derivedValue` | Property | `cubby:DocumentRegion` → `xsd:string` | A derived value; always a suggestion only, never a committed fact. | – |
| `cubby:Reviewer` | Class | – | A human reviewer: role, id, time, region. | schema.org `schema:Person` |
| `cubby:Helper` | Class | – | A helper proposed by Cubby's proposers and tested in a locked-down VM session. | – |
| `cubby:helperStatus` | Property | `cubby:Helper` → `skos:Concept` | adopted, rejected, or retired. Rejected helpers are kept with the case that broke them. | SKOS |
| `cubby:breakingCase` | Property | `cubby:Helper` → `cheez:Record` | The case that caused rejection. | PROV-O `prov:wasInvalidatedBy` |
| `cubby:SplitQuestion` | Class | – | A question split into parts with routes, per-part verdicts, a combination program and its verdict. | – |

### Module `mowm#` (prefix `mowm:`)

| Term | Kind | Domain → Range | Definition | Reuses / specialises |
|---|---|---|---|---|
| `mowm:World` | Class | – | A tree node holding one or more HYLAs, with `z_max`, `tau`, and a world id from the global counter (starting 1000) or router counter (starting 5000). | – |
| `mowm:worldId` | Property | `mowm:World` → `xsd:integer` | The integer id assigned by the global or router counter. | – |
| `mowm:zDepth` | Property | `mowm:World` → `xsd:integer` | The z-axis depth of this world. | – |
| `mowm:zMax` | Property | `mowm:World` → `xsd:integer` | Maximum depth; `spawn_subworld` refuses at `z_max`. | – |
| `mowm:tau` | Property | `mowm:World` → `cheez:Threshold` | Confidence threshold for spawning sub-worlds, with full calibration. | – |
| `mowm:hyla` | Property | `mowm:World` → `mowm:HYLA` | A HYLA hypernetwork instance held by the world. | – |
| `mowm:HYLA` | Class | – | A hypernetwork generating block-code transition functions. | – |
| `mowm:mergeStrategy` | Property | `mowm:World` → `skos:Concept` | How transition deltas are merged (e.g. BindComposeMerge, weighted bundling). | SKOS |
| `mowm:RoutingDecision` | Class | – | A DSelect-k routing decision over the dynamic pool of worlds. | Decision Ontology `Decision` |
| `mowm:routedWorld` | Property | `mowm:RoutingDecision` → `mowm:World` | The world selected by routing. | – |
| `mowm:centroidScore` | Property | `mowm:RoutingDecision` → `xsd:decimal` | The best world's centroid score (a score, not a threshold). | – |
| `mowm:marginOverRunnerUp` | Property | `mowm:RoutingDecision` → `xsd:decimal` | Margin over the runner-up world. | – |
| `mowm:SpawnEvent` | Class | – | A spawn of a new sub-world or specialist world. | PROV-O `prov:Activity` |
| `mowm:spawnReason` | Property | `mowm:SpawnEvent` → `skos:Concept` | Coded reason: residual-norm-exceeds-tau, centroid-below-floor, margin-below-tau-margin, etc. | SKOS |
| `mowm:residualNorm` | Property | `mowm:SpawnEvent` → `xsd:decimal` | The residual norm that triggered the spawn. | – |
| `mowm:PruneEvent` | Class | – | Retirement of a world idle for `tau_prune_steps`. Whether pruning deletes or retires is an open question; the record uses `cheez:retired`. | – |
| `mowm:InterWorldQuery` | Class | – | A query passed world-to-world when the routed world's best member falls short of `tau_answer`. | – |
| `mowm:PossibilityOracle` | Class | – | The possibility oracle over fact worlds. | – |
| `mowm:wordTableEncoding` | Property | `mowm:RoutingDecision` → `xsd:boolean` | Asserts text was encoded by FastWordEncoder word-table lookup (~25 µs), not a sentence model. | – |
| `mowm:blockCodeParams` | Property | `mowm:World` → `xsd:string` | The block-code parameters: k=80 blocks, l=128 slots, D=10240, qFHRR integer-phase form. | – |

### Module `cubelang#` (prefix `cube:`)

| Term | Kind | Domain → Range | Definition | Reuses / specialises |
|---|---|---|---|---|
| `cube:Program` | Class | – | A CubeLang program. | – |
| `cube:implements` | Property | `cube:Program` → `xsd:string` | The interface the program implements (ISolver, IAgent, etc.). | – |
| `cube:Execution` | Class | – | One execution of a CubeLang program by the VM. | PROV-O `prov:Activity` |
| `cube:outcome` | Property | `cube:Execution` → `skos:Concept` | return, error, or ask. | SKOS |
| `cube:ASKEvent` | Class | – | A suspension where the program cannot choose and asks the caller, then resumes by re-execution over run-proto. | – |
| `cube:strictMode` | Property | `cube:Program` → `xsd:boolean` | Whether strict mode (the default) is active, rejecting non-executing constructs. | – |

---

## 2. JSON-LD 1.1 `@context`

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "schema": "https://schema.org/",
    "emotionml": "https://www.w3.org/TR/emotionml/#",
    "rocrate": "https://w3id.org/ro/crate#",

    "Record": "cheez:Record",
    "validTime": { "@id": "cheez:validTime", "@type": "time:Instant" },
    "recordedTime": { "@id": "cheez:recordedTime", "@type": "time:Instant" },
    "supersededBy": { "@id": "cheez:supersededBy", "@type": "@id" },
    "retired": { "@id": "cheez:retired", "@type": "xsd:boolean" },
    "retirementReason": { "@id": "cheez:retirementReason", "@type": "@id" },
    "hashRef": { "@id": "cheez:hashRef", "@type": "xsd:string" },
    "Threshold": "cheez:Threshold",
    "calibrationScreen": { "@id": "cheez:calibrationScreen", "@type": "xsd:string" },
    "calibrationData": { "@id": "cheez:calibrationData", "@type": "xsd:string" },
    "calibrationDate": { "@id": "cheez:calibrationDate", "@type": "xsd:date" },
    "value": { "@id": "cheez:value", "@type": "xsd:decimal" },
    "encoder": { "@id": "cheez:encoder", "@type": "xsd:string" },
    "corpus": { "@id": "cheez:corpus", "@type": "xsd:string" },
    "writtenByHost": { "@id": "cheez:writtenByHost", "@type": "xsd:boolean" },

    "Turn": "cubby:Turn",
    "dialogueAct": { "@id": "cubby:dialogueAct", "@type": "@id" },
    "CertifiedAnswer": "cubby:CertifiedAnswer",
    "certificate": { "@id": "cubby:certificate", "@type": "@id" },
    "Certificate": "cubby:Certificate",
    "proposer": { "@id": "cubby:proposer", "@type": "@id" },
    "Refusal": "cubby:Refusal",
    "refusalReason": { "@id": "cubby:refusalReason", "@type": "@id" },
    "EmotionReading": "cubby:EmotionReading",
    "confidence": { "@id": "cubby:confidence", "@type": "xsd:decimal" },
    "padDimensions": { "@id": "cubby:padDimensions", "@type": "xsd:string" },
    "HormoneState": "cubby:HormoneState",
    "StoryWorld": "cubby:StoryWorld",
    "stateChange": { "@id": "cubby:stateChange", "@type": "@id" },
    "inventedAtTurn": { "@id": "cubby:inventedAtTurn", "@type": "xsd:integer" },
    "DocumentRegion": "cubby:DocumentRegion",
    "exactCheck": { "@id": "cubby:exactCheck", "@type": "@id" },
    "derivedValue": { "@id": "cubby:derivedValue", "@type": "xsd:string" },
    "Reviewer": "cubby:Reviewer",
    "Helper": "cubby:Helper",
    "helperStatus": { "@id": "cubby:helperStatus", "@type": "@id" },
    "breakingCase": { "@id": "cubby:breakingCase", "@type": "@id" },
    "SplitQuestion": "cubby:SplitQuestion",

    "World": "mowm:World",
    "worldId": { "@id": "mowm:worldId", "@type": "xsd:integer" },
    "zDepth": { "@id": "mowm:zDepth", "@type": "xsd:integer" },
    "zMax": { "@id": "mowm:zMax", "@type": "xsd:integer" },
    "tau": { "@id": "mowm:tau", "@type": "@id" },
    "hyla": { "@id": "mowm:hyla", "@type": "@id" },
    "HYLA": "mowm:HYLA",
    "mergeStrategy": { "@id": "mowm:mergeStrategy", "@type": "@id" },
    "RoutingDecision": "mowm:RoutingDecision",
    "routedWorld": { "@id": "mowm:routedWorld", "@type": "@id" },
    "centroidScore": { "@id": "mowm:centroidScore", "@type": "xsd:decimal" },
    "marginOverRunnerUp": { "@id": "mowm:marginOverRunnerUp", "@type": "xsd:decimal" },
    "SpawnEvent": "mowm:SpawnEvent",
    "spawnReason": { "@id": "mowm:spawnReason", "@type": "@id" },
    "residualNorm": { "@id": "mowm:residualNorm", "@type": "xsd:decimal" },
    "PruneEvent": "mowm:PruneEvent",
    "InterWorldQuery": "mowm:InterWorldQuery",
    "PossibilityOracle": "mowm:PossibilityOracle",
    "wordTableEncoding": { "@id": "mowm:wordTableEncoding", "@type": "xsd:boolean" },
    "blockCodeParams": { "@id": "mowm:blockCodeParams", "@type": "xsd:string" },

    "Program": "cube:Program",
    "implements": { "@id": "cube:implements", "@type": "xsd:string" },
    "Execution": "cube:Execution",
    "outcome": { "@id": "cube:outcome", "@type": "@id" },
    "ASKEvent": "cube:ASKEvent",
    "strictMode": { "@id": "cube:strictMode", "@type": "xsd:boolean" }
  }
}
```

---

## 3. Six Example Records

All `urn:uuid:` values and bracketed `[PLACEHOLDER]` values are illustrative. No value below is a real measurement unless the brief states it.

### 3.1 Certified answer turn with its certificate

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:a1b2c3d4-0001-4000-8000-000000000001",
  "@type": "cubby:CertifiedAnswer",
  "cheez:validTime": "[PLACEHOLDER: ISO-8601 instant the fact held]",
  "cheez:recordedTime": "[PLACEHOLDER: ISO-8601 instant the store learned it]",
  "cheez:writtenByHost": true,
  "cubby:dialogueAct": "cubby:certified-answer",
  "cubby:proposer": "cubby:everyday-table",
  "cubby:certificate": {
    "@id": "urn:uuid:a1b2c3d4-0001-4000-8000-000000000002",
    "@type": "cubby:Certificate",
    "prov:wasGeneratedBy": {
      "@type": "prov:Activity",
      "prov:used": [
        {
          "@type": "prov:Entity",
          "skos:prefLabel": "[PLACEHOLDER: statement read]",
          "prov:atLocation": "[PLACEHOLDER: source and revision]"
        }
      ]
    },
    "decision:Question": {
      "@type": "decision:Question",
      "decision:hasOption": [
        {
          "@type": "decision:Option",
          "skos:prefLabel": "[PLACEHOLDER: proposer program]"
        }
      ],
      "decision:is_satisfied_by": "[PLACEHOLDER: criteria met]",
      "decision:Decision": "urn:uuid:a1b2c3d4-0001-4000-8000-000000000001"
    }
  }
}
```

Note: the certificate records that the answer follows from the facts and rules used under the recorded checks. It does not record a truth score.

### 3.2 Refusal with its reason

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:b2c3d4e5-0002-4000-8000-000000000001",
  "@type": "cubby:Refusal",
  "cheez:validTime": "[PLACEHOLDER]",
  "cheez:recordedTime": "[PLACEHOLDER]",
  "cheez:writtenByHost": true,
  "cubby:dialogueAct": "cubby:refusal",
  "cubby:refusalReason": "cubby:fact-missing",
  "cubby:proposer": "cubby:host-grammar"
}
```

No prose rationale field. The reason is a code from the closed set. Any human-readable sentence is produced by a template from this code.

### 3.3 MoWM routing decision ending in a spawn

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:c3d4e5f6-0003-4000-8000-000000000001",
  "@type": "mowm:RoutingDecision",
  "cheez:validTime": "[PLACEHOLDER]",
  "cheez:recordedTime": "[PLACEHOLDER]",
  "cheez:writtenByHost": true,
  "mowm:wordTableEncoding": true,
  "mowm:centroidScore": "[PLACEHOLDER: the observed centroid score]",
  "mowm:marginOverRunnerUp": "[PLACEHOLDER]",
  "mowm:routedWorld": null,
  "mowm:spawnEvent": {
    "@id": "urn:uuid:c3d4e5f6-0003-4000-8000-000000000002",
    "@type": "mowm:SpawnEvent",
    "mowm:spawnReason": "mowm:centroid-below-floor",
    "mowm:residualNorm": "[PLACEHOLDER]",
    "mowm:spawnedWorld": {
      "@type": "mowm:World",
      "mowm:worldId": "[PLACEHOLDER: from router counter starting 5000]",
      "mowm:zDepth": 0,
      "mowm:zMax": "[PLACEHOLDER]",
      "mowm:tau": {
        "@type": "cheez:Threshold",
        "cheez:value": "[PLACEHOLDER]",
        "cheez:calibrationScreen": "[PLACEHOLDER]",
        "cheez:calibrationData": "[PLACEHOLDER: hash]",
        "cheez:calibrationDate": "[PLACEHOLDER]",
        "cheez:encoder": "FastWordEncoder",
        "cheez:corpus": "[PLACEHOLDER]"
      },
      "mowm:blockCodeParams": "k=80, l=128, D=10240, qFHRR"
    }
  }
}
```

The spawn occurred because the centroid score did not clear the absolute floor, or the margin did not clear `tau_margin`. A spawned specialist is recoverable; a mis-route is not.

### 3.4 World split (sub-world at z+1)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:d4e5f6a7-0004-4000-8000-000000000001",
  "@type": "mowm:SpawnEvent",
  "cheez:validTime": "[PLACEHOLDER]",
  "cheez:recordedTime": "[PLACEHOLDER]",
  "cheez:writtenByHost": true,
  "mowm:spawnReason": "mowm:residual-norm-exceeds-tau",
  "mowm:residualNorm": "[PLACEHOLDER]",
  "mowm:parentWorld": {
    "@type": "mowm:World",
    "mowm:worldId": "[PLACEHOLDER: from global counter starting 1000]",
    "mowm:zDepth": "[PLACEHOLDER: z]",
    "mowm:zMax": "[PLACEHOLDER]"
  },
  "mowm:spawnedWorld": {
    "@type": "mowm:World",
    "mowm:worldId": "[PLACEHOLDER: next from global counter]",
    "mowm:zDepth": "[PLACEHOLDER: z+1]",
    "mowm:zMax": "[PLACEHOLDER]",
    "mowm:mergeStrategy": "mowm:BindComposeMerge",
    "mowm:hyla": {
      "@type": "mowm:HYLA",
      "cheez:hashRef": "[PLACEHOLDER: hash of HYLA weights]"
    }
  }
}
```

The child id comes from the global counter, not from parent id + 1. If `zDepth` equals `zMax`, `spawn_subworld` refuses and no record is written.

### 3.5 Helper rejected in its sandbox (order 0062)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:e5f6a7b8-0005-4000-8000-000000000001",
  "@type": "cubby:Helper",
  "cheez:validTime": "[PLACEHOLDER]",
  "cheez:recordedTime": "[PLACEHOLDER]",
  "cheez:writtenByHost": true,
  "cheez:retired": true,
  "cheez:retirementReason": "cubby:helper-rejected",
  "cubby:helperStatus": "cubby:rejected",
  "cubby:proposer": "[PLACEHOLDER: which of Cubby's proposers proposed it]",
  "cubby:breakingCase": {
    "@id": "urn:uuid:e5f6a7b8-0005-4000-8000-000000000002",
    "@type": "cheez:Record",
    "skos:prefLabel": "[PLACEHOLDER: the case that broke the helper]"
  },
  "cubby:sandboxChecks": {
    "cubby:rejectionRateOverWindow": "[PLACEHOLDER]",
    "cubby:escapeAttempts": 0
  }
}
```

The rejected helper is kept, not deleted, with the case that broke it. Escape attempts must be 0.

### 3.6 Invoice field sent to human review and then committed (order 0061)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:f6a7b8c9-0006-4000-8000-000000000001",
  "@type": "cubby:DocumentRegion",
  "cheez:validTime": "[PLACEHOLDER]",
  "cheez:recordedTime": "[PLACEHOLDER]",
  "cheez:writtenByHost": true,
  "cubby:regionText": "[PLACEHOLDER: the region text]",
  "cubby:readerVersion": "[PLACEHOLDER]",
  "cubby:fieldValue": "[PLACEHOLDER: the number, which must appear in regionText]",
  "cubby:exactCheck": ["cubby:format-check", "cubby:arithmetic-check"],
  "cubby:derivedValue": "[PLACEHOLDER: suggestion only, not committed]",
  "cubby:reviewStep": {
    "@type": "cubby:Reviewer",
    "schema:roleName": "[PLACEHOLDER]",
    "schema:identifier": "[PLACEHOLDER: reviewer id]",
    "cheez:recordedTime": "[PLACEHOLDER: review time]",
    "cubby:reviewedRegion": "urn:uuid:f6a7b8c9-0006-4000-8000-000000000001"
  },
  "cubby:secondApproval": "[PLACEHOLDER: required above threshold amount; null if below]",
  "cubby:committed": true
}
```

Every number in the field must appear in the region's text. Derived values are suggestions only. A second approval is needed above a stated amount.

---

## 4. SHACL Shapes (three most important classes)

I judge the three most important classes to be `cubby:CertifiedAnswer`, `mowm:RoutingDecision`, and `cubby:Helper`, because they carry the hardest invariants: certification semantics, the no-neural-model routing rule, and the retire-never-delete rule.

### 4.1 `cubby:CertifiedAnswer`

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cubby:CertifiedAnswerShape
  a sh:NodeShape ;
  sh:targetClass cubby:CertifiedAnswer ;
  sh:property [
    sh:path cheez:validTime ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:dateTime ;
  ] ;
  sh:property [
    sh:path cheez:recordedTime ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:datatype xsd:dateTime ;
  ] ;
  sh:property [
    sh:path cheez:writtenByHost ;
    sh:hasValue true ;
  ] ;
  sh:property [
    sh:path cubby:certificate ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:class cubby:Certificate ;
  ] ;
  sh:property [
    sh:path cubby:proposer ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
  ] ;
  sh:property [
    sh:path cubby:dialogueAct ;
    sh:hasValue cubby:certified-answer ;
  ] ;
  # No truth score may appear.
  sh:not [
    sh:path cubby:truthScore ;
    sh:minCount 1 ;
  ] .
```

### 4.2 `mowm:RoutingDecision`

```turtle
@prefix mowm: <https://w3id.org/cheezml/mowm#> .

mowm:RoutingDecisionShape
  a sh:NodeShape ;
  sh:targetClass mowm:RoutingDecision ;
  sh:property [
    sh:path mowm:wordTableEncoding ;
    sh:hasValue true ;
    sh:message "Routing must use word-table encoding, not a sentence model." ;
  ] ;
  sh:property [
    sh:path cheez:writtenByHost ;
    sh:hasValue true ;
  ] ;
  sh:property [
    sh:path cheez:validTime ;
    sh:minCount 1 ;
  ] ;
  sh:property [
    sh:path cheez:recordedTime ;
    sh:minCount 1 ;
  ] ;
  # If a spawn occurred, it must have a coded reason.
  sh:property [
    sh:path mowm:spawnEvent ;
    sh:optional true ;
    sh:node [
      sh:property [
        sh:path mowm:spawnReason ;
        sh:minCount 1 ;
      ] ;
    ] ;
  ] ;
  # Thresholds, if present, must carry calibration.
  sh:property [
    sh:path mowm:tau ;
    sh:optional true ;
    sh:node cheez:ThresholdShape ;
  ] .

cheez:ThresholdShape
  a sh:NodeShape ;
  sh:targetClass cheez:Threshold ;
  sh:property [
    sh:path cheez:value ;
    sh:minCount 1 ;
    sh:datatype xsd:decimal ;
  ] ;
  sh:property [
    sh:path cheez:calibrationScreen ;
    sh:minCount 1 ;
  ] ;
  sh:property [
    sh:path cheez:calibrationDate ;
    sh:minCount 1 ;
    sh:datatype xsd:date ;
  ] ;
  sh:property [
    sh:path cheez:encoder ;
    sh:minCount 1 ;
  ] ;
  sh:property [
    sh:path cheez:corpus ;
    sh:minCount 1 ;
  ] .
```

### 4.3 `cubby:Helper`

```turtle
cubby:HelperShape
  a sh:NodeShape ;
  sh:targetClass cubby:Helper ;
  sh:property [
    sh:path cheez:writtenByHost ;
    sh:hasValue true ;
  ] ;
  sh:property [
    sh:path cubby:helperStatus ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:in ( cubby:adopted cubby:rejected cubby:retired ) ;
  ] ;
  # Rejected helpers must carry the breaking case.
  sh:sparql [
    sh:select """
      SELECT $this WHERE {
        $this cubby:helperStatus cubby:rejected .
        FILTER NOT EXISTS { $this cubby:breakingCase ?case . }
      }
    """ ;
    sh:message "A rejected helper must record the case that broke it." ;
  ] ;
  # Rejected or retired helpers must not be deleted: retired flag must be true.
  sh:sparql [
    sh:select """
      SELECT $this WHERE {
        $this cubby:helperStatus ?s .
        FILTER(?s IN (cubby:rejected, cubby:retired))
        FILTER NOT EXISTS { $this cheez:retired true . }
      }
    """ ;
    sh:message "Rejected and retired helpers are kept, not deleted." ;
  ] ;
  # Escape attempts must be 0.
  sh:property [
    sh:path cubby:escapeAttempts ;
    sh:hasValue 0 ;
  ] .
```

---

## 5. CubeLang Programs for MoWM Routing and Spawning

Both programs use only the executing safe subset (create, assign of a literal, add, sub, mul, div, sum, push, pop, query, store, recall, remember, forget, compare, if/else, while) and ASK. Flat operands only. Strict mode. No `for`, `match`, intra-program `call`, early `return`, `throw`, map literals as operands, or method-call results as operands.

### 5.1 MoWM Routing Program

```
program MowmRoute implements ISolver {

  storage {
    centroid_score : decimal
    floor : decimal
    margin : decimal
    tau_margin : decimal
    world_count : int
    route_ok : int
    residual_norm : decimal
    tau_spawn : decimal
    decision : int
  }

  functions {

    fn route() {
      create centroid_score
      create floor
      create margin
      create tau_margin
      create world_count
      create route_ok
      create residual_norm
      create tau_spawn
      create decision

      assign centroid_score 0.0
      assign floor 0.0
      assign margin 0.0
      assign tau_margin 0.0
      assign world_count 0
      assign route_ok 0
      assign residual_norm 0.0
      assign tau_spawn 0.5
      assign decision 0

      query centroid_score
      query floor
      query margin
      query tau_margin
      query world_count
      query residual_norm

      compare centroid_score floor
      if route_ok == 1 {
        compare margin tau_margin
        if route_ok == 1 {
          assign decision 1
        } else {
          assign decision 2
        }
      } else {
        assign decision 2
      }

      compare decision 2
      if decision == 2 {
        compare residual_norm tau_spawn
        if residual_norm > tau_spawn {
          assign decision 3
        } else {
          ASK
        }
      }

      store decision
      remember decision
    }
  }
}
```

Notes: `decision` 1 = route to best world; 2 = spawn (centroid below floor or margin below `tau_margin`); 3 = spawn (residual norm exceeds `tau_spawn`). The ASK suspends when neither condition is met and the program cannot choose; the caller resolves and re-executes over `run-proto`. All threshold values (`floor`, `tau_margin`, `tau_spawn`) are loaded by `query` from the host's calibrated store; they are not hard-coded. The word-table encoding happens before this program runs, in the HYLA/FastWordEncoder layer; no sentence model appears here.

### 5.2 MoWM Spawning Program

```
program MowmSpawn implements ISolver {

  storage {
    parent_z : int
    z_max : int
    spawn_allowed : int
    new_world_id : int
    residual_norm : decimal
    tau_spawn : decimal
    merge_strategy : int
  }

  functions {

    fn spawn() {
      create parent_z
      create z_max
      create spawn_allowed
      create new_world_id
      create residual_norm
      create tau_spawn
      create merge_strategy

      assign parent_z 0
      assign z_max 0
      assign spawn_allowed 0
      assign new_world_id 0
      assign residual_norm 0.0
      assign tau_spawn 0.5
      assign merge_strategy 0

      query parent_z
      query z_max
      query residual_norm
      query tau_spawn

      compare parent_z z_max
      if parent_z < z_max {
        assign spawn_allowed 1
      } else {
        assign spawn_allowed 0
      }

      compare spawn_allowed 1
      if spawn_allowed == 1 {
        compare residual_norm tau_spawn
        if residual_norm > tau_spawn {
          query new_world_id
          assign merge_strategy 1
          store new_world_id
          store merge_strategy
          remember new_world_id
        } else {
          ASK
        }
      } else {
        assign spawn_allowed 0
        store spawn_allowed
      }
    }
  }
}
```

Notes: `spawn_subworld` refuses at `z_max`; the program records `spawn_allowed` 0 and stores nothing further. The new world id comes from the host's global counter (starting 1000) or router counter (starting 5000), loaded by `query`; it is never computed as parent id + 1. `merge_strategy` 1 stands for BindComposeMerge. The ASK fires when the residual norm does not exceed `tau_spawn` and the program cannot decide; the caller resolves.

Both programs are in the executing subset. They should be compiled with `cubelang check` before adoption.

---

## 6. What I Would Cut, and Open Questions

### What I would cut

- **`cubby:HormoneState` as a full class.** The brief says it is "a separate, labelled dimension set" but gives no properties beyond that. I would reduce it to a `skos:Concept` with a `cheez:hashRef` to the dimension data until the dimension set is specified. I don't know the dimensions.

- **`mowm:PossibilityOracle` and `mowm:InterWorldQuery` as separate classes.** They share most properties with `mowm:RoutingDecision`. I would make them subclasses or drop them and use `mowm:RoutingDecision` with a `skos:Concept` distinguishing the mode (centroid-over-fact-world, best-member, inter-world). This removes three classes and their overlapping properties.

- **`cube:strictMode`.** Strict mode is the default and the only mode that executes. Recording it on every program record is noise. I would drop the property and note in the vocabulary documentation that all executing programs are in strict mode.

- **`cubby:padDimensions` as a string.** It should be three typed decimal properties (`pleasure`, `arousal`, `dominance`), each in [0,1]. But the brief says only "pleasure, arousal and dominance as the pad-dimensions set" and "values in [0,1]". I don't know whether EmotionML's own dimension vocabulary should be reused directly or wrapped. I would cut the string property and defer until that is decided.

- **`cheez:corpus` on `cheez:Threshold`.** The brief says thresholds do not transfer between encoders or corpora, but it does not state what a "corpus" identifier looks like or where it is registered. I would keep the property but mark it optional until the corpus registry exists.

### Open questions

1. **Pruning: delete or retire?** The brief says "Whether pruning deletes or retires is an open question." The hard rule says "Retire, never delete." I have used `cheez:retired` in the `PruneEvent` description, but the brief explicitly flags this as unresolved. I don't know the answer.

2. **Decision Ontology namespace.** The brief references the Decision Ontology's `Question`, `Option`, `is_satisfied_by`, `is_violated_by`, and `Decision`, but does not give the namespace IRI. I used `decision:` as a placeholder prefix. I don't know the actual IRI.

3. **ISO 24617-2 mapping.** The brief says ISO 24617-2 is "the reference set to consider" for dialogue acts but does not state which specific act labels map to which CubbyLLM acts. I don't know the mapping.

4. **EmotionML namespace.** The brief cites "EmotionML 1.0 (W3C, 2014)" and I used `emotionml:` as a prefix, but I don't know whether a stable namespace IRI exists for EmotionML or whether the terms should be minted under `cubby:`.

5. **RO-Crate integration.** The brief says the reproducibility package uses "a Workflow Run RO-Crate for each evaluation run" and "each hypothesis verdict as a nanopublication." I don't know how the RO-Crate metadata record links to the CheezML record for the same run. The `rocrate:` prefix is declared in the context but I have not modelled the RO-Crate structure.

6. **Nanopublication and STATO.** The brief mentions nanopublications and STATO for statistics in the reproducibility package. I don't know the nanopublication schema or which STATO terms apply. I have not modelled them.

7. **Split questions (0063).** The brief lists parts, routes, per-part verdicts, combination program and its verdict, and partial answers naming the missing part. I declared `cubby:SplitQuestion` but did not model its internal structure because the brief does not give enough detail. I don't know the exact properties.

8. **Domain worlds (0058).** The brief says a domain world has "an ontology module, open records, rules and a question set" and that changes are published "only with no new wrong answer and a gain." I have not modelled this. I don't know the structure of the ontology module or how the gain is measured.

9. **The `container` problem.** The brief states that a `container` passes `check` but compiles to nothing, so `vm.start_from_file`, `vm.wire` and `vm.bridge` don't run. My CubeLang programs avoid `container`. But I don't know whether this is a bug to be fixed or a design constraint to work around permanently.

10. **Second-approval amount threshold (order 0061).** The brief says "a second approval is needed above an amount" but does not state the amount or where it is configured. I used a placeholder. I don't know the value.

11. **`cheez:hashRef` algorithm.** I assumed SHA-256 because the brief says "referenced by hash" without naming the algorithm. I don't know which hash function is used.

12. **Oja consolidation.** The brief mentions "Oja consolidation applies" to worlds but gives no further detail. I have not modelled it. I don't know what record it produces.

---

This is an opinion, not a finding. Every placeholder is marked. Every term I minted is one I could not find in the listed standards. Every program uses only the opcodes and constructs the brief says execute today.

## z-ai/glm-5.3-prime

*9575 tokens out, 55.5 s, finish `stop`*

# CheezML: one record vocabulary

Opinion of one panel member. Everything below uses only the facts in the brief; where the brief is silent I say so.

## 1. Classes and properties

Conventions: every class is `rdfs:Class`; every property gets domain and range. "Reuse" means the term is used directly or specialised (`rdfs:subClassOf` / `rdfs:subPropertyOf`). I mint a term only where no listed standard fits.

### Module `core#` (prefix `cheez:`)

| Term | Kind | Domain / Range | Definition | Standard reused |
|---|---|---|---|---|
| `Record` | Class | — | Anything the host writes about what ran; never hand-declared. | `prov:Entity` (subclass) |
| `recordId` | Property | `Record` / IRI | The record's own `urn:uuid:` id, outside the vocabulary namespace. | `prov:id` (subproperty) |
| `retiredAsOf` | Property | `Record` / `time:Instant` | Marks a record retired, never deleted. | OWL-Time |
| `retirementReason` | Property | `Record` / `cheez:ReasonCode` | Why it was retired/rejected/pruned. | — (minted; SKOS below) |
| `ReasonCode` | Class | — | A closed code for a reason; no prose rationale. | `skos:Concept` (subclass) |
| `Threshold` | Class | — | A numeric threshold together with its calibration. | — (minted) |
| `thresholdValue` | Property | `Threshold` / `xsd:decimal` | The number. | `schema:value` (reuse) |
| `calibratedOnScreen` | Property | `Threshold` / `cheez:Record` | The screen that set it. | `prov:used` (subproperty) |
| `calibratedOnData` | Property | `Threshold` / `cheez:HashRef` | The data it was calibrated on, by hash. | `prov:used` (subproperty) |
| `calibratedAt` | Property | `Threshold` / `time:Instant` | Calibration date. | `time:hasTime` / `prov:atTime` |
| `HashRef` | Class | — | Reference to a vector, weight set or blob by hash; never embedded. | `prov:Entity` with `schema:sha256`-style hash property; I reuse `schema:associatedMedia`/`schema:sha256` where the digest is a plain string. |
| `hashOf` | Property | `HashRef` / `xsd:string` | The digest string. | `schema:sha256` (reuse) |
| `trainingRun` | Property | `HashRef` / `prov:Activity` | The training run that produced the weights. | `prov:wasGeneratedBy` (subproperty) |
| `validTime` | Property | `cheez:Fact` / `time:Instant` or `time:Interval` | When a fact held (order 0055). | OWL-Time |
| `recordedTime` | Property | `cheez:Fact` / `time:Instant` | When the store learned it. | `prov:atTime` (subproperty) |
| `supersededBy` | Property | `cheez:Fact` / `cheez:Fact` | A correction closes the old version; never overwrites. | `prov:wasRevisionOf` (inverse direction; I reuse `dcterms:isReplacedBy`) |
| `Fact` | Class | — | A stored fact with two clocks. | `prov:Entity` (subclass) |
| `Certificate` | Class | — | Says the answer follows from the facts and rules used, under the recorded checks; never a truth score. | `prov:Entity` + Decision Ontology `Decision` (see cubby#) |
| `EmotionReading` | Class | — | A user emotion reading: an inference with confidence, never a fact; values in [0,1], pad-dimensions. | EmotionML 1.0 by link (`cheez:emotionml` → the W3C EmotionML doc); expressed with `<cheez:emotionmlRef>` |
| `emotionmlRef` | Property | `EmotionReading` / IRI | Link to the EmotionML 1.0 representation. | — (minted; EmotionML has no RDF terms, so link only) |
| `HormoneState` | Class | — | Cubby's own hormone state, a separate labelled dimension set. | EmotionML by link |
| `confidence` | Property | `EmotionReading` / `xsd:decimal` | Confidence of the inference. | `schema:confidence` does not exist; I mint this. (Not claimed for schema.org.) |

### Module `cubby#` (prefix `cubby:`)

| Term | Kind | Domain / Range | Definition | Standard reused |
|---|---|---|---|---|
| `Conversation` | Class | — | A conversation whose state is kept by the host. | `schema:Conversation` (schema.org has no such type; I do **not** claim it — minted, with `schema:sender`/`schema:recipient` reused for participants) |
| `DialogueTurn` | Class | — | One turn: question, certified answer, refusal, clarifying question, follow-up, or small talk. | `schema:Message` (subclass); ISO 24617-2 is the reference set to consider, linked, not imported |
| `Question` | Class | — | The last or current question. | Decision Ontology `Question` (reuse) |
| `CertifiedAnswer` | Class | — | An answer turn carrying a certificate. | DO `Decision` + `schema:Answer` (subclass of both) |
| `Refusal` | Class | — | A refusal turn with a reason code. | DO `Decision` (subclass) |
| `refusalReason` | Property | `Refusal` / `cheez:ReasonCode` | One of: no plan, relation missing, fact missing, walk failed, not certified, ambiguous; conflict kinds (value, type, temporal, logical, recorded-against-derived); dates too coarse; unreadable source. | `skos:related` to a SKOS concept scheme of codes |
| `ClarifyingQuestion` | Class | — | An open clarifying question. | `schema:Question` (subclass) |
| `FollowUp` | Class | — | A follow-up with its rewrite, which passed the entity guard. | `schema:Message` (subclass) |
| `SmallTalk` | Class | — | A small-talk turn. | `schema:Message` (subclass) |
| `entityGuardPassed` | Property | `FollowUp` / `xsd:boolean` | Written by the host from what ran. | — (minted) |
| `Proposer` | Class | — | One of: host grammar, everyday table, overnight rule library, episodic memory of certified chains, LM program emitter. | `prov:Agent` (subclass) |
| `proposedBy` | Property | `CertifiedAnswer` / `cubby:Proposer` | Every answer records its proposer. | `prov:wasAttributedTo` (subproperty) |
| `Helper` | Class | — | A helper proposed by Cubby's own proposers and tested in a locked-down VM session. | `prov:Agent` + `prov:Plan` (subclass of Plan) |
| `helperStatus` | Property | `Helper` / `cheez:ReasonCode` or SKOS code | Adopted or rejected; a rejected helper is kept with the case that broke it. | `prov:wasInvalidatedBy` for rejection |
| `escapeAttempts` | Property | `Helper` / `xsd:integer` | Must be 0. | — (minted) |
| `rejectionRate` | Property | `cubby:Proposer` / `xsd:decimal` | Over a window; a score, not a threshold. | — (minted) |
| `Document` | Class | — | A document with regions. | RO-Crate `File`/`Dataset` context; `schema:DigitalDocument` (subclass) |
| `Region` | Class | — | A region with its text and the reader's version. | `schema:TextDigitalDocument` fragment; minted (no standard region term) |
| `regionText` | Property | `Region` / `xsd:string` | The region's text. | `schema:text` (reuse) |
| `Field` | Class | — | An invoice field; every number must appear in its region's text. | — (minted) |
| `fieldValue` | Property | `Field` / `xsd:string` or `xsd:decimal` | The extracted value; derived values are suggestions only. | `schema:value` (reuse) |
| `checkCode` | Property | `Field` / `cheez:ReasonCode` | Format, arithmetic, purchase order, supplier, duplicate. | — (minted) |
| `Reviewer` | Class | — | A reviewer as a source: role, id, time, region. | `prov:Agent` (subclass) |
| `review` | Property | `Field` / `prov:Activity` | The human-review activity. | `prov:Activity` |
| `secondApprovalNeeded` | Property | `Field` / `xsd:boolean` | Above an amount. | — (minted) |
| `StoryWorld` | Class | — | Isolated from the real-world store. | — (minted) |
| `StoryAction` | Class | — | A program with exact prerequisite checks; soft (tone) checks labelled soft. | `prov:Plan` (subclass) |
| `softCheck` | Property | `StoryAction` / `cheez:ReasonCode` | Labels a check as soft. | — (minted) |
| `StateChange` | Class | — | Dated, with old value, new value and turn. | — (minted; `prov:Activity` subclass) |
| `oldValue` / `newValue` / `turn` | Properties | `StateChange` / `xsd:string`, `xsd:integer` | As named. | `prov:generated` reuse for new value; old/new minted |
| `inventedAtTurn` | Property | `cheez:Fact` / `xsd:integer` | Source "invented at turn N". | `prov:wasDerivedFrom` pointing at an `Invention` marker (minted) |
| `SplitQuestion` | Class | — | Parts, routes, each part's verdict, the combination program and its verdict. | DO `Question` (subclass) |
| `missingPart` | Property | `CertifiedAnswer` / `cubby:SplitQuestion` | Partial answers name the missing part. | — (minted) |
| `DomainWorld` | Class | — | An ontology module, open records, rules and a question set. | `prov:Entity` (subclass) |
| `domainChange` | Property | `DomainWorld` / `prov:Activity` | Published only with no new wrong answer and a gain. | `prov:Activity` |

### Module `mowm#` (prefix `mowm:`)

| Term | Kind | Domain / Range | Definition | Standard reused |
|---|---|---|---|---|
| `World` | Class | — | A tree node (`world.py`) holding one or more HYLAs. | `prov:Entity` (subclass) |
| `worldId` | Property | `World` / `xsd:integer` | From the global counter starting at 1000 (router-spawned: 5000). Ids are recorded, never computed as parent+1. | — (minted) |
| `zDepth` | Property | `World` / `xsd:integer` | Position on the Z axis. | — (minted) |
| `zMax` | Property | `World` / `xsd:integer` | Maximum depth; `spawn_subworld` refuses at `z_max`. | — (minted) |
| `tau` | Property | `World` / `cheez:Threshold` | Confidence threshold for spawning sub-worlds, with calibration. | — (minted; the Threshold class carries the calibration) |
| `SubWorld` | Class | — | A world at z+1 handling residuals. | `mowm:World` (subclass) |
| `Hyla` | Class | — | A hypernetwork generating block-code transition functions. | `prov:Entity` (subclass) |
| `mergeStrategy` | Property | `World` / `xsd:string` | e.g. `BindComposeMerge`, weighted bundling. | — (minted) |
| `RoutingDecision` | Class | — | Route or spawn, written by the host from what ran. | DO `Decision` (subclass) |
| `routedTo` | Property | `RoutingDecision` / `mowm:World` | The best world, when routing. | `prov:used` (subproperty) |
| `spawned` | Property | `RoutingDecision` / `mowm:SubWorld` | The spawned specialist, when spawning. | `prov:wasGeneratedBy` inverse; minted |
| `centroidScore` | Property | `RoutingDecision` / `xsd:decimal` | A score, not a threshold. | — (minted) |
| `margin` | Property | `RoutingDecision` / `xsd:decimal` | Margin over the runner-up. | — (minted) |
| `tauMargin` / `tauSpawn` / `tauAnswer` / `tauPruneSteps` | Properties | `RoutingDecision` or `World` / `cheez:Threshold` | Calibrated per deployment; `tau_spawn` default 0.5, `tau_prune_steps` default 100. | — (minted) |
| `residualNorm` | Property | `RoutingDecision` / `xsd:decimal` | The residual that triggered a spawn. | — (minted) |
| `prunedWorld` | Property | `mowm:World` / `cheez:ReasonCode` | Pruning; whether it deletes or retires is an open question (I record retire-only per the hard rules, and flag it). | `cheez:retiredAsOf` |
| `WordTableEncoding` | Class | — | Text encoded by word-table lookup (FastWordEncoder, ~25 µs); no neural model on the routing path. | — (minted) |
| `BlockCode` | Class | — | Sparse block codes, k=80 × l=128 (D=10,240), qFHRR integer-phase. | — (minted) |
| `PossibilityOracle` | Class | — | Fact worlds, members encoded by the word table; routing by best member (default) or centroid. | — (minted) |
| `support` | Property | `PossibilityOracle` / `xsd:decimal` | At or above `tau_answer`. | — (minted) |
| `InterWorldQuery` | Class | — | Query passes world to world when the routed world's best member falls short of `tau_answer`. | `prov:Activity` (subclass) |

### Module `cubelang#` (prefix `cube:`)

| Term | Kind | Domain / Range | Definition | Standard reused |
|---|---|---|---|---|
| `Program` | Class | — | A CubeLang program, typed, `program X implements I`. | `prov:Plan` (subclass) |
| `programText` | Property | `Program` / `xsd:string` | Source text. | `schema:text` (reuse) |
| `compilesToday` | Property | `Program` / `xsd:boolean` | Whether it uses only the executing subset and ASK. | — (minted) |
| `specOnly` | Property | `Program` / `xsd:boolean` | Labelled "spec only: does not compile today". | — (minted) |
| `Run` | Class | — | An execution of a program. | `prov:Activity` (subclass) |
| `runProto` | Property | `Run` / `cube:Program` | Re-execution over `run-proto` after a suspend. | — (minted) |
| `Suspend` | Class | — | The ASK outcome: the program cannot choose and asks the caller. | — (minted) |
| `Opcode` | Class | — | One of the real opcodes; byte values synced across three repositories and never changed. | — (minted) |
| `opcodeByte` | Property | `Opcode` / `xsd:string` | e.g. `0x0c` for QUERY. | — (minted) |

## 2. The `@context`

```json
{
  "@context": {
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "https://schema.org/",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcterms": "http://purl.org/dc/terms/",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "do": "http://decision-ontology.googlecode.com/svn/trunk/DecisionOntology.owl#",
    "cheezml": "https://w3id.org/cheezml/",
    "recordId": "cheez:recordId",
    "retiredAsOf": {"@id": "cheez:retiredAsOf", "@type": "time:Instant"},
    "retirementReason": "cheez:retirementReason",
    "thresholdValue": "cheez:thresholdValue",
    "calibratedOnScreen": "cheez:calibratedOnScreen",
    "calibratedOnData": "cheez:calibratedOnData",
    "calibratedAt": {"@id": "cheez:calibratedAt", "@type": "time:Instant"},
    "hashOf": "cheez:hashOf",
    "trainingRun": "cheez:trainingRun",
    "validTime": "time:hasTime",
    "recordedTime": {"@id": "cheez:recordedTime", "@type": "time:Instant"},
    "supersededBy": "dcterms:isReplacedBy",
    "confidence": "cheez:confidence",
    "emotionmlRef": "cheez:emotionmlRef",
    "refusalReason": "cubby:refusalReason",
    "proposedBy": "cubby:proposedBy",
    "escapeAttempts": "cubby:escapeAttempts",
    "rejectionRate": "cubby:rejectionRate",
    "regionText": "schema:text",
    "fieldValue": "schema:value",
    "checkCode": "cubby:checkCode",
    "secondApprovalNeeded": "cubby:secondApprovalNeeded",
    "softCheck": "cubby:softCheck",
    "oldValue": "cubby:oldValue",
    "newValue": "cubby:newValue",
    "turn": "cubby:turn",
    "inventedAtTurn": "cubby:inventedAtTurn",
    "missingPart": "cubby:missingPart",
    "worldId": "mowm:worldId",
    "zDepth": "mowm:zDepth",
    "zMax": "mowm:zMax",
    "tau": "mowm:tau",
    "routedTo": "mowm:routedTo",
    "spawned": "mowm:spawned",
    "centroidScore": "mowm:centroidScore",
    "margin": "mowm:margin",
    "residualNorm": "mowm:residualNorm",
    "mergeStrategy": "mowm:mergeStrategy",
    "support": "mowm:support",
    "programText": "schema:text",
    "compilesToday": "cube:compilesToday",
    "specOnly": "cube:specOnly",
    "runProto": "cube:runProto",
    "opcodeByte": "cube:opcodeByte",
    "generatedAt": {"@id": "prov:generatedAtTime", "@type": "time:Instant"},
    "used": "prov:used",
    "wasGeneratedBy": "prov:wasGeneratedBy",
    "wasAttributedTo": "prov:wasAttributedTo",
    "wasInvalidatedBy": "prov:wasInvalidatedBy",
    "atTime": {"@id": "prov:atTime", "@type": "time:Instant"},
    "agent": "prov:agent",
    "text": "schema:text",
    "value": "schema:value",
    "sha256": "schema:sha256",
    "name": "schema:name"
  }
}
```

Note: `do:` — I do not know the canonical Decision Ontology IRI; the brief names the Decision Ontology but not its namespace. I flag this rather than invent one; if the panel knows it, substitute it. All prefixes end in `#` or `/`; `xsd` uses `http://www.w3.org/2001/XMLSchema#`.

## 3. Six example records

Values not given in the brief are marked `"PLACEHOLDER"`.

### 3.1 Certified answer turn with certificate

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:11111111-1111-4111-8111-111111111111",
  "@type": ["cubby:CertifiedAnswer", "prov:Entity", "do:Decision"],
  "recordId": "urn:uuid:11111111-1111-4111-8111-111111111111",
  "text": "PLACEHOLDER: template-rendered answer sentence",
  "generatedAt": "PLACEHOLDER: 2026-11-07T00:00:00Z",
  "proposedBy": {"@id": "urn:uuid:22222222-2222-4222-8222-222222222222", "@type": "cubby:Proposer", "name": "PLACEHOLDER: overnight rule library"},
  "schema:about": {"@id": "urn:uuid:33333333-3333-4333-8333-333333333333", "@type": "do:Question"},
  "cubby:certificate": {
    "@id": "urn:uuid:44444444-4444-4444-8444-444444444444",
    "@type": "cheez:Certificate",
    "wasGeneratedBy": {
      "@id": "urn:uuid:55555555-5555-4555-8555-555555555555",
      "@type": "prov:Activity",
      "name": "PLACEHOLDER: program run",
      "used": [
        {"@id": "urn:uuid:66666666-6666-4666-8666-666666666666", "@type": "cheez:Fact",
         "name": "PLACEHOLDER: statement read",
         "cubby:sourceRevision": "PLACEHOLDER",
         "validTime": "PLACEHOLDER", "recordedTime": "PLACEHOLDER"},
        {"@id": "urn:uuid:77777777-7777-4777-8777-777777777777", "@type": "prov:Entity",
         "name": "PLACEHOLDER: rule applied"}
      ]
    },
    "do:is_satisfied_by": {"@id": "urn:uuid:88888888-8888-4888-8888-888888888888", "@type": "do:Option",
      "name": "PLACEHOLDER: proposer program"},
    "do:is_violated_by": []
  }
}
```

No truth score anywhere. The certificate's checks are recorded as activities that ran; a check that never ran is absent.

### 3.2 Refusal with reason

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:aaaaaaa1-0000-4000-8000-000000000001",
  "@type": ["cubby:Refusal", "do:Decision"],
  "recordId": "urn:uuid:aaaaaaa1-0000-4000-8000-000000000001",
  "generatedAt": "PLACEHOLDER",
  "refusalReason": {"@id": "https://w3id.org/cheezml/cubby#reasonFactMissing", "@type": "cheez:ReasonCode"},
  "text": "PLACEHOLDER: template sentence from the code"
}
```

The reason is a code, not prose; the readable sentence is template-produced.

### 3.3 MoWM routing decision ending in spawn

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:bbbbbbb1-0000-4000-8000-000000000002",
  "@type": ["mowm:RoutingDecision", "do:Decision"],
  "recordId": "urn:uuid:bbbbbbb1-0000-4000-8000-000000000002",
  "generatedAt": "PLACEHOLDER",
  "centroidScore": "PLACEHOLDER: 0.41",
  "margin": "PLACEHOLDER: 0.03",
  "residualNorm": "PLACEHOLDER: 0.62",
  "mowm:tauSpawn": {
    "@type": "cheez:Threshold", "thresholdValue": 0.5,
    "calibratedOnScreen": "PLACEHOLDER: screen id",
    "calibratedOnData": {"@type": "cheez:HashRef", "hashOf": "PLACEHOLDER: sha256"},
    "calibratedAt": "PLACEHOLDER"
  },
  "spawned": {
    "@id": "urn:uuid:ccccccc1-0000-4000-8000-000000000003",
    "@type": ["mowm:SubWorld", "mowm:World"],
    "worldId": "PLACEHOLDER: 5000-series id from the router counter, recorded not computed",
    "zDepth": "PLACEHOLDER: 1",
    "zMax": "PLACEHOLDER",
    "mergeStrategy": "PLACEHOLDER: BindComposeMerge"
  }
}
```

Spawn, not route: the floor and margin were not both cleared. The encoding used is the word table; no sentence model appears on this path.

### 3.4 World split (sub-world at z+1)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:ddddddd1-0000-4000-8000-000000000004",
  "@type": ["mowm:SubWorld", "mowm:World"],
  "recordId": "urn:uuid:ddddddd1-0000-4000-8000-000000000004",
  "worldId": "PLACEHOLDER: id from the 1000 counter, recorded",
  "zDepth": 1,
  "zMax": "PLACEHOLDER",
  "tau": {"@type": "cheez:Threshold", "thresholdValue": "PLACEHOLDER",
          "calibratedOnScreen": "PLACEHOLDER", "calibratedOnData": "PLACEHOLDER", "calibratedAt": "PLACEHOLDER"},
  "mergeStrategy": "PLACEHOLDER",
  "mowm:parentWorld": {"@id": "urn:uuid:eeeeeee1-0000-4000-8000-000000000005", "zDepth": 0},
  "mowm:handlesResidualOf": {"@id": "urn:uuid:eeeeeee1-0000-4000-8000-000000000005"}
}
```

The parent link is a reference, not parent-id-plus-one.

### 3.5 Helper rejected in its sandbox (order 0062)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:fffffff1-0000-4000-8000-000000000006",
  "@type": "cubby:Helper",
  "recordId": "urn:uuid:fffffff1-0000-4000-8000-000000000006",
  "name": "PLACEHOLDER: helper name",
  "proposedBy": {"@id": "urn:uuid:00000000-0000-4000-8000-000000000007", "@type": "cubby:Proposer"},
  "escapeAttempts": 0,
  "wasInvalidatedBy": {
    "@id": "urn:uuid:11111112-0000-4000-8000-000000000008",
    "@type": "prov:Activity",
    "name": "PLACEHOLDER: sandbox VM session",
    "used": {"@id": "urn:uuid:22222223-0000-4000-8000-000000000009",
             "name": "PLACEHOLDER: the verified answer case that broke it"}
  },
  "retirementReason": {"@id": "https://w3id.org/cheezml/cubby#reasonFailedReproduction"},
  "retiredAsOf": "PLACEHOLDER"
}
```

Kept, not deleted, with the breaking case. The rejection rate over the window is recorded on the proposer, not the helper.

### 3.6 Invoice field: review then commit (order 0061)

```json
{
  "@context": "https://w3id.org/cheezml/context.jsonld",
  "@id": "urn:uuid:33333334-0000-4000-8000-000000000010",
  "@type": "cubby:Field",
  "recordId": "urn:uuid:33333334-0000-4000-8000-000000000010",
  "name": "PLACEHOLDER: invoice total",
  "fieldValue": "PLACEHOLDER: 1250.00",
  "cubby:region": {
    "@id": "urn:uuid:44444445-0000-4000-8000-000000000011",
    "@type": "cubby:Region",
    "regionText": "PLACEHOLDER: region text containing the number",
    "cubby:readerVersion": "PLACEHOLDER"
  },
  "checkCode": [{"@id": "https://w3id.org/cheezml/cubby#checkArithmetic"},
                {"@id": "https://w3id.org/cheezml/cubby#checkDuplicate"}],
  "cubby:derivedSuggestion": false,
  "review": {
    "@id": "urn:uuid:55555556-0000-4000-8000-000000000012",
    "@type": "prov:Activity",
    "agent": {"@id": "urn:uuid:66666667-0000-4000-8000-000000000013", "@type": "cubby:Reviewer",
              "cubby:role": "PLACEHOLDER", "cubby:reviewerId": "PLACEHOLDER",
              "cubby:reviewedRegion": "PLACEHOLDER: region id"},
    "atTime": "PLACEHOLDER"
  },
  "secondApprovalNeeded": "PLACEHOLDER: true if above the amount",
  "cubby:committedAt": {"@type": "time:Instant", "time:inXSDDateTimeStamp": "PLACEHOLDER"},
  "validTime": "PLACEHOLDER", "recordedTime": "PLACEHOLDER"
}
```

## 4. SHACL shapes

Three classes I judge most important: `cheez:Certificate` (the system's core claim), `mowm:RoutingDecision` (where thresholds must stay calibrated), `cubby:Helper` (where retire-never-delete bites).

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix mowm: <https://w3id.org/cheezml/mowm#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cheez:CertificateShape a sh:NodeShape ;
  sh:targetClass cheez:Certificate ;
  sh:property [
    sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:node prov:ActivityShape ] ;
  sh:property [
    sh:path prov:used ; sh:minCount 1 ] ;
  sh:property [
    sh:path cheez:truthScore ; sh:maxCount 0 ] ;
  sh:property [
    sh:path prov:generatedAtTime ; sh:minCount 1 ; sh:datatype xsd:dateTime ] .

mowm:RoutingDecisionShape a sh:NodeShape ;
  sh:targetClass mowm:RoutingDecision ;
  sh:property [
    sh:path mowm:tauSpawn ; sh:minCount 1 ;
    sh:property [
      sh:path [ sh:alternativePath ( cheez:calibratedOnScreen cheez:calibratedOnData cheez:calibratedAt ) ] ;
      sh:minCount 3 ] ] ;
  sh:property [
    sh:path mowm:spawned ; sh:minCount 0 ; sh:maxCount 1 ;
    sh:property [ sh:path mowm:worldId ; sh:datatype xsd:integer ; sh:minCount 1 ] ] ;
  sh:property [
    sh:path mowm:routedTo ; sh:minCount 0 ; sh:maxCount 1 ] ;
  sh:xor ( [ sh:path mowm:routedTo ; sh:minCount 1 ]
           [ sh:path mowm:spawned  ; sh:minCount 1 ] ) ;
  sh:property [
    sh:path mowm:neuralEncoder ; sh:maxCount 0 ] .

cubby:HelperShape a sh:NodeShape ;
  sh:targetClass cubby:Helper ;
  sh:property [ sh:path cubby:escapeAttempts ; sh:minCount 1 ; sh:datatype xsd:integer ; sh:maxInclusive 0 ] ;
  sh:property [ sh:path prov:wasInvalidatedBy ; sh:minCount 0 ; sh:maxCount 1 ] ;
  sh:property [
    sh:path cheez:retirementReason ; sh:minCount 0 ;
    sh:severity sh:Warning ;
    sh:message "A rejected helper must keep its reason." ] .
```

The `sh:maxCount 0` on `cheez:truthScore` and `mowm:neuralEncoder` encodes two hard rules negatively: those properties should not exist in the vocabulary at all, and the shapes catch any draft that reintroduces them.

## 5. CubeLang programs

Both use only the executing subset (create, assign of a literal, add, sub, mul, div, sum, push, pop, query, store, recall, remember, forget, compare, if/else, while; flat operands; no `for`, `match`, `call`, `return`, `throw`, map literals, or method-call results as operands). No opcodes are written by hand; the compiler emits them.

### 5.1 MoWM router

```text
program Router implements IAgent {
  storage {
    num bestScore;
    num runnerUp;
    num margin;
    num floor;
    num tauMargin;
    num residualNorm;
    num tauSpawn;
    num worldCount;
    num i;
    num score;
    num spawnFlag;
  }
  functions {
    create bestScore = 0.0;
    create runnerUp = 0.0;
    create margin = 0.0;
    create floor = PLACEHOLDER_FLOOR;
    create tauMargin = PLACEHOLDER_TAU_MARGIN;
    create residualNorm = PLACEHOLDER_RESIDUAL;
    create tauSpawn = PLACEHOLDER_TAU_SPAWN;
    create worldCount = PLACEHOLDER_WORLD_COUNT;
    create i = 0;
    create score = 0.0;
    create spawnFlag = 0;
    while compare i < worldCount {
      score = PLACEHOLDER_SCORE_OF_I;
      if compare score > bestScore {
        runnerUp = bestScore;
        bestScore = score;
      } else {
        if compare score > runnerUp {
          runnerUp = score;
        } else {
        }
      }
      i = add i 1;
    }
    margin = sub bestScore runnerUp;
    if compare bestScore < floor {
      spawnFlag = 1;
    } else {
      if compare worldCount < 2 {
        spawnFlag = 0;
      } else {
        if compare margin < tauMargin {
          spawnFlag = 1;
        } else {
          spawnFlag = 0;
        }
      }
    }
    if compare residualNorm > tauSpawn {
      spawnFlag = 1;
    } else {
    }
    if compare spawnFlag == 1 {
      remember spawnFlag;
    } else {
      remember spawnFlag;
    }
  }
}
```

`PLACEHOLDER_*` are host-supplied literals at run time; the word-table scores arrive as numbers, so no sentence model is on the path. Note the empty `else` branches: the executing subset has no early return, so branches that do nothing are written explicitly. The final `remember` in both branches is redundant but keeps the program within the subset; the host reads the stored value either way.

### 5.2 MoWM spawner

```text
program Spawner implements IAgent {
  storage {
    num z;
    num zMax;
    num residualNorm;
    num tauSpawn;
    num spawnFlag;
    num newWorldId;
  }
  functions {
    create z = PLACEHOLDER_Z;
    create zMax = PLACEHOLDER_Z_MAX;
    create residualNorm = PLACEHOLDER_RESIDUAL;
    create tauSpawn = PLACEHOLDER_TAU_SPAWN;
    create spawnFlag = 0;
    create newWorldId = 0;
    if compare z >= zMax {
      spawnFlag = 0;
    } else {
      if compare residualNorm > tauSpawn {
        spawnFlag = 1;
      } else {
        spawnFlag = 0;
      }
    }
    if compare spawnFlag == 1 {
      newWorldId = PLACEHOLDER_NEXT_WORLD_ID;
      store newWorldId;
      remember spawnFlag;
    } else {
      remember spawnFlag;
    }
  }
}
```

`PLACEHOLDER_NEXT_WORLD_ID` is the id the host took from the router counter (5000-series) or the world counter (1000-series); the program never computes parent+1. `spawn_subworld`'s refusal at `z_max` is the `z >= zMax` branch.

Both programs should pass `cubelang check` in strict mode; the panel will compile them before adoption, as stated.

## 6. What I would cut, and open questions

**Cut:**
- The `do:` namespace guess above — cut or replace with the real Decision Ontology IRI once known.
- `cheez:truthScore` and `mowm:neuralEncoder` exist only as SHACL prohibitions, not as vocabulary terms; cut them from the published vocabulary entirely.
- Any property whose only job is prose (a "rationale" string). Everything readable is template output from codes and values; the templates themselves live outside CheezML.
- Per-record threshold numbers without calibration objects: I would make the SHACL require the calibration triple, so a bare number fails validation rather than being cut later.
- The `HormoneState` class if no consumer reads it today; the brief says it is separate and labelled, but if nothing consumes it yet, defer it.

**Open questions:**
1. Does pruning in MoWM delete or retire? The brief says it is open. CheezML as designed can only record retirement; if deletion happens, the vocabulary cannot express it, and that gap should be decided before adoption.
2. What is the canonical Decision Ontology namespace IRI?
3. Where do the refusal-reason codes live — as IRIs in `cubby#` (as my examples assume) or as a SKOS concept scheme published separately? The brief lists the reasons but not their home.
4. How are the two clocks (order 0055) recorded for non-fact records — do turns and routing decisions also carry valid/recorded time, or only facts?
5. The brief gives no byte values for most opcodes and I have not invented any; if records should cite opcode bytes (e.g. for a program's disassembly), the synced table must be published as data first.
6. Is the `@context` served from `https://w3id.org/cheezml/context.jsonld` (the redirect to `ns.grillcheeseai.com` is stated but not its path layout)? I don't know.
7. For invoice fields, what counts as "the amount" above which a second approval is needed — is that itself a calibrated threshold? The brief says only "above an amount".

## x-ai/grok-4.7

*66622 tokens out, 899.5 s, finish `stop`*

This is an opinion, not a finding. I do not know the Decision Ontology IRI, the RO-Crate profile IRI, the STATO term IRIs, the ISO 24617-2 concept IRIs, the EmotionML document URL, the nanopublication namespace, the host URI, the hash algorithm, or the `owl:versionIRI` to publish. Those are not filled in below. Namespace IRIs have no version segment. Redirect paths on `ns.grillcheeseai.com` are not stated.

## 1. The classes and properties

Reuse is only where a named standard has an unambiguous RDF term. Decision Ontology names are kept as names, not IRIs. EmotionML is by link, not by minted EmotionML properties. No term below is a truth score, a prose rationale, an embedded vector, a sentence-model encoder, a delete operation, or a status declared without a run.

Stated figures that are not terms and are not defaults: centroid macro 0.680 against max-over-axioms 0.589; about 40× fewer comparisons; 92.4% precision and 86.9% of never-seen domains sent to spawn at a calibrated threshold whose cutoff is not stated; about 60× fewer member comparisons on inter-world query; FastWordEncoder about 25 µs; `tau_spawn` default 0.5; `tau_prune_steps` default 100; a fact-world member sits about 1/√n from a centroid of n quasi-orthogonal facts; Global World about 350 seeded axioms across 14 domains. I do not know the screen, data, or date behind the defaults, so the defaults are not usable thresholds.

### core (`cheez:`, `https://w3id.org/cheezml/core#`)

**cheez:HostRecord** — class. Subclass of `prov:Entity`. A record the host wrote from what ran. Reuses `prov:Entity`.

**cheez:FactVersion** — class. Subclass of `cheez:HostRecord`. One stored fact version. A correction closes it and writes another; it is not overwritten. Reuses `prov:Entity`. `prov:wasRevisionOf` links the newer version to the older one when a correction ran. I do not know a standard validity property, so valid time is not `prov:generatedAtTime`.

**cheez:validTime** — property. Domain `cheez:HostRecord`. Range `time:Interval`. When it held. The interval reuses OWL-Time (`time:hasBeginning`, `time:hasEnd`, `time:Instant`, `time:inXSDDateTime`). I do not know whether an open hold omits `time:hasEnd`. Recorded time reuses `prov:generatedAtTime` (range `xsd:dateTime`): when the store learned it. I do not know whether order 0055 applies to every host record or only to facts. I apply both clocks to host records in this opinion.

**cheez:revisionId** — property. Domain `prov:Entity`. Range `xsd:string`. Revision of a statement that was read. Identifier scheme not stated. I do not reuse `schema:version`; its domain is CreativeWork, which is not stated for statements. Source reuses `prov:hadPrimarySource` (entity to entity).

**cheez:Threshold** — class. A cutoff. Not a score. Does not transfer across encoders or corpora. No standard term fits.

**cheez:thresholdValue** — property. Domain `cheez:Threshold`. Range `xsd:decimal`. The cutoff number.

**cheez:Calibration** — class. The screen, data, and date that set a threshold. No standard term fits.

**cheez:calibratedBy** — property. Domain `cheez:Threshold`. Range `cheez:Calibration`. Required on every threshold.

**cheez:calibrationScreen** — property. Domain `cheez:Calibration`. Range `xsd:string`. The screen that set it. Screen-id scheme not stated.

**cheez:calibrationData** — property. Domain `cheez:Calibration`. Range `prov:Entity`. The data used, referenced, not embedded.

**cheez:calibrationDate** — property. Domain `cheez:Calibration`. Range `xsd:date`. The date. I do not know a schema.org or OWL-Time property whose meaning is this calibration date, so this is minted.

**cheez:Encoder** — class. An encoder a threshold is bound to.

**cheez:boundEncoder** — property. Domain `cheez:Calibration`. Range `cheez:Encoder`. The encoder the threshold must not leave.

**cheez:boundCorpus** — property. Domain `cheez:Calibration`. Range `prov:Entity`. The corpus the threshold must not leave.

**cheez:Score** — class. A number that was measured. Not a cutoff. No calibration. No standard term fits.

**cheez:scoreValue** — property. Domain `cheez:Score`. Range `xsd:decimal`. The measured number.

**cheez:blobHash** — property. Domain `prov:Entity`. Range `xsd:string`. Hash of a vector, weight set, or blob. Bytes are not in the record. Algorithm not stated.

**cheez:TrainingRecord** — class. Subclass of `prov:Activity`. A training run the host recorded. Weights are `cheez:blobHash` on the weights entity. Reuses `prov:Activity`. A seed is not a property: a seed alone does not reproduce trained weights. I do not know the seed’s fields, so none are minted.

**cheez:trainingRun** — property. Domain `prov:Entity`. Range `cheez:TrainingRecord`. The run that produced the hashed weights.

**cheez:hasReason** — property. Domain `cheez:HostRecord`. Range `skos:Concept`. A code. Not a sentence. A readable sentence is a template over codes and values, outside the record. Reuses `skos:Concept` as the range. Does not reuse `skos:note`.

**cheez:disposition** — property. Domain `cheez:HostRecord`. Range `skos:Concept`. Only for an outcome that ran. Codes: `cheez:Retired`, `cheez:Rejected`, `cheez:Pruned`. No Active, no Passed, no Certified-by-agreement. There is no delete property.

**cheez:templateId** — property. Domain `cheez:HostRecord`. Range `xsd:string`. Identifies the template that may render a sentence. Template language not stated.

**cheez:Filter** — class. A filter only. It does not certify.

**cheez:AgreementFilter** — class. Subclass of `cheez:Filter`. Agreement between models or agents. May only filter. Does not certify. No standard term fits.

**cheez:EvaluationRun** — class. Subclass of `prov:Activity`. An evaluation run packaged as a Workflow Run RO-Crate. Profile IRI not stated, so no RO-Crate term is aliased. `cheez:workflowRunCrate` (domain `cheez:EvaluationRun`, range `prov:Entity`) holds the crate id.

**cheez:HypothesisVerdict** — class. Subclass of `cheez:HostRecord`. A hypothesis verdict, killed results included. `cheez:killed` (domain this class, range `xsd:boolean`) is present only if that kill-or-not check ran. Statistics are STATO; term IRIs are not stated, so no STATO term is aliased. Nanopublication namespace is not stated, so it is not aliased.

**cheez:orderOfRule** is not minted. Order numbers are sources of these definitions, not fields the host measured.

### cubby (`cubby:`, `https://w3id.org/cheezml/cubby#`)

**cubby:Turn** — class. Subclass of `cheez:HostRecord`. One host-kept conversation step. Also reuses `schema:Message` as a second type. I do not know a dialogue-turn class in the named RDF vocabularies.

**cubby:dialogueAct** — property. Domain `cubby:Turn`. Range `skos:Concept`. One of: `cubby:QuestionAct`, `cubby:CertifiedAnswerAct`, `cubby:RefusalAct`, `cubby:ClarifyingQuestionAct`, `cubby:FollowUpAct`, `cubby:SmallTalkAct`. ISO 24617-2 is the reference set to consider. I do not know its concept IRIs, so no mapping is claimed. `cubby:CertifiedAnswerAct` means the VM certified under the certificate rules. It does not mean correct.

**cubby:ConversationState** — class. Subclass of `cheez:HostRecord`. Host-kept state. Properties, each domain this class: `cubby:entityId` (range `xsd:string`), `cubby:lastQuestion` (range `cubby:Turn`), `cubby:lastProgram` (range `cube:Program`), `cubby:lastAnswer` (range `cubby:Turn`), `cubby:openClarifyingQuestion` (range `cubby:Turn`). Absent if there is none.

**cubby:Certificate** — class. Subclass of `cheez:HostRecord`. Says the answer follows from the facts and rules used, under the recorded checks. Not correctness. Not a truth score. Written as PROV-O. The decision step is Decision Ontology `Question`, `Option`, `is_satisfied_by`, `is_violated_by`, and `Decision`. I do not know that ontology’s IRI, so I do not alias it and I do not mint substitutes.

**cubby:answer** — property. Domain `cubby:Certificate`. Range `prov:Entity`. The answer entity. The program run is a `prov:Activity` linked by `prov:wasGeneratedBy`. `prov:wasAttributedTo` is the host, not a model. `prov:wasAssociatedWith` on the run is the VM. VM URI not stated. Agreement is not this link.

**cubby:ReadStatement** — class. Subclass of `prov:Entity`. A statement read. Source: `prov:hadPrimarySource`. Revision: `cheez:revisionId`.

**cubby:AppliedRule** — class. Subclass of `prov:Entity`. A rule applied. Identity scheme not stated.

**cubby:proposer** — property. Domain `cubby:Certificate`. Range `skos:Concept`. Required on every answer. One of: `cubby:HostGrammar`, `cubby:EverydayTable`, `cubby:OvernightRuleLibrary`, `cubby:EpisodicCertifiedChains`, `cubby:ProgramEmitter`. The program emitter proposes. It does not write a stored fact or a training record. I do not know whether it is an external model.

**cubby:checksRan** — property. Domain `cubby:Certificate`. Range `cubby:Check`. A check node exists only if that check ran. No node means no “passed”.

**cubby:Check** — class. Subclass of `prov:Activity`. A check that ran. `cubby:soft` (domain this class, range `xsd:boolean`) is true only when the check is a labelled soft tone check. Exact checks are not marked soft. Satisfaction and violation are the Decision Ontology properties whose IRIs I do not know. I do not mint a passed flag.

**cubby:hasReason** values for refusal, as `skos:Concept`: `cubby:NoPlan`, `cubby:RelationMissing`, `cubby:FactMissing`, `cubby:WalkFailed`, `cubby:NotCertified`, `cubby:Ambiguous`, `cubby:DatesTooCoarse`, `cubby:UnreadableSource`. Conflict kinds, property `cubby:conflictKind` (domain `cubby:Turn`, range `skos:Concept`): `cubby:ValueConflict`, `cubby:TypeConflict`, `cubby:TemporalConflict`, `cubby:LogicalConflict`, `cubby:RecordedAgainstDerived`. I do not know the walk algorithm, so no walk steps are minted.

**cubby:rewriteText** — property. Domain `cubby:Turn`. Range `xsd:string`. The follow-up rewrite that was checked. Not a rationale. `cubby:entityGuard` (domain `cubby:Turn`, range `cubby:Check`) is present only if the guard ran.

**cubby:StoryWorld** — class. Subclass of `cheez:HostRecord`. Isolated from the real-world store. `cubby:inWorld` (domain `cheez:FactVersion`, range `cubby:StoryWorld`) places a story fact. `cubby:inventedAtTurn` (domain `cheez:FactVersion`, range `xsd:integer`) is the source “invented at turn N”. I do not know any other invented-fact source field.

**cubby:StateChange** — class. Subclass of `cheez:FactVersion`. A dated story change. `cubby:oldValue` and `cubby:newValue` (domain this class, range not stated; I do not assign a datatype). `cubby:atTurn` (domain this class, range `xsd:integer`).

**cubby:prerequisiteCheck** — property. Domain `cube:Program`. Range `cubby:Check`. An exact prerequisite that ran. Soft checks use `cubby:soft` true.

**cubby:DocumentRegion** — class. Subclass of `prov:Entity`. `cubby:regionText` (range `xsd:string`) is the region’s text, not a rationale. `cubby:readerVersion` (range `xsd:string`). Version scheme not stated.

**cubby:sourceRegion** — property. Domain `cheez:FactVersion`. Range `cubby:DocumentRegion`. The region a field was read from. The rule that every number in a field appears in that text is a check kind `cubby:NumberAppearsInRegion`, recorded only if that check ran.

Document exact-check codes, range of `cubby:checksRan` when the check ran: `cubby:FormatCheck`, `cubby:ArithmeticCheck`, `cubby:PurchaseOrderCheck`, `cubby:SupplierCheck`, `cubby:DuplicateCheck`. I do not know their procedures, so none are written out.

**cubby:DerivedSuggestion** — class. Subclass of `cheez:HostRecord`. A derived value. A suggestion only. Not a stored fact. No standard term fits.

**cubby:Reviewer** — class. Subclass of `prov:Agent`. A human source. `cubby:reviewerRole` (range `xsd:string`), `schema:identifier` for id (reused), `cubby:reviewerTime` (range `xsd:dateTime`), `cubby:reviewerRegion` (range `cubby:DocumentRegion`). I do not reuse `prov:hadPrimarySource` for the reviewer: that PROV property’s range is `prov:Entity`, and Agent is not stated to be an Entity. The region may be `prov:hadPrimarySource` of the field.

**cubby:reviewedBy** — property. Domain `cheez:FactVersion`. Range `prov:Activity`. The review that ran before the commit. Second approval is a second review, required above an amount. That amount is a `cheez:Threshold`. The amount, currency, and calibration are not stated. I do not know them.

**cubby:Helper** — class. Subclass of `cheez:HostRecord`. Proposed by Cubby’s proposers, tested in a locked-down VM session, kept if rejected. `cubby:proposedBy` (range `skos:Concept`, one of the proposers). `cubby:vmSession` (range `prov:Activity`). `cubby:breakingCase` (range `cheez:HostRecord`) if rejected. `cubby:promoted` (range `xsd:boolean`) only if the host’s promotion decision ran. Reason codes for a break: `cubby:DidNotReproduceVerifiedAnswer`, `cubby:ChangedOutsideScope`. `cubby:rejectionRate` (range `cheez:Score`) over `cubby:rateWindow` (range not stated). Window size not stated. `cubby:escapeAttempts` (range `xsd:integer`) only if that count ran. The adoption rule is escape attempts must be 0. That 0 is an order, not a calibrated score threshold, so it is not a `cheez:Threshold`.

**cubby:SplitQuestion** — class. Subclass of `cheez:HostRecord`. `cubby:part` (range `cheez:HostRecord`), `cubby:partRoute` (range not stated; I do not know the route object), `cubby:partVerdict` (range `cheez:HostRecord`), `cubby:combinationProgram` (range `cube:Program`), `cubby:combinationVerdict` (range `cheez:HostRecord`), `cubby:missingPart` (range `cheez:HostRecord`) when the answer is partial.

**cubby:DomainWorld** — class. Subclass of `cheez:HostRecord`. `cubby:ontologyModule`, `cubby:openRecord`, `cubby:rule`, `cubby:questionSet` (ranges `prov:Entity`; finer types not stated). **cubby:DomainChange** — class. Subclass of `cheez:HostRecord`. A change to a domain’s mapping, rules, or shapes. Published only with no new wrong answer and a gain. Gain’s unit is not stated, so no gain number is minted. `cubby:noNewWrongAnswer` (range `cubby:Check`) only if that check ran.

**cubby:EmotionReading** — class. Subclass of `cheez:HostRecord`. Not a subclass of `cheez:FactVersion`. An inference about a user, never a fact. `cubby:pleasure`, `cubby:arousal`, `cubby:dominance` (range `xsd:decimal`, values in [0, 1], set name `pad-dimensions`). `cubby:confidence` (range `xsd:decimal`). Confidence is not a truth score and not a threshold. `cubby:emotionmlExport` (range `xsd:string`, a `cheez:blobHash` of the EmotionML 1.0 export). Spec URL not stated, so I do not invent the link. These properties are local. They do not claim to be EmotionML RDF terms.

**cubby:HormoneState** — class. Subclass of `cheez:HostRecord`. Cubby’s own hormone state. A separate labelled dimension set. Not pad-dimensions. Not a user emotion. Dimension labels are not stated. I do not know them, so none are minted.

Possibility-oracle refusal by the host’s `contradicts` rule is `mowm:Contradicts`, not one of the dialogue refusal codes. I do not know whether those lists should merge.

### mowm (`mowm:`, `https://w3id.org/cheezml/mowm#`)

**mowm:World** — class. Subclass of `cheez:HostRecord`. A tree node. `mowm:worldId` (range `xsd:integer`) comes from a counter, not from parent id + 1. `mowm:idCounter` (range `skos:Concept`): `mowm:GlobalWorldCounter` starts at 1000; `mowm:RouterSpawnCounter` starts at 5000. I do not know which counter `spawn_subworld` uses. `mowm:parentWorld` (range `mowm:World`). `mowm:zDepth` and `mowm:zMax` (range `xsd:integer`). `mowm:depthRelation` value `mowm:ChildAtZPlusOne` when the child was created at z+1. `mowm:hyla` (range `prov:Entity`, one or more) with `cheez:blobHash` for weights. `mowm:mergeStrategy` (range `skos:Concept`). Named examples: `mowm:BindComposeMerge`, `mowm:WeightedBundling`. I do not know the full list. `mowm:tau` (range `cheez:Threshold`): confidence threshold for spawning sub-worlds. `spawn_subworld` refuses at `z_max`; reason code `mowm:AtZMax`. I do not know whether z greater than `z_max` can occur. Oja consolidation applies; parameters are not stated, so none are minted. `mowm:ojaApplied` (range `xsd:boolean`) only if it ran.

**mowm:BlockCodeForm** — class. The stated form: k = 80 blocks, l = 128 slots, D = 10,240, qFHRR integer-phase. The preprint title given is “Semantic Block Codes”. I do not know the citation, so none is attached. Vectors are `cheez:blobHash` only.

**mowm:RoutingDecision** — class. Subclass of `cheez:HostRecord`. The bridge decision that ran. `mowm:action` (range `skos:Concept`): `mowm:Route` or `mowm:Spawn`. `mowm:encoder` (range `cheez:Encoder`) is `mowm:FastWordEncoder`: word-table lookup, not a sentence model. There is no sentence-model term. `mowm:worldCount` (range `xsd:integer`). `mowm:bestCentroidScore`, `mowm:runnerUpScore`, `mowm:margin` (range `cheez:Score`). `mowm:floor`, `mowm:tauMargin` (range `cheez:Threshold`). Route only if the best centroid score clears the floor and, when `worldCount` is two or more, the margin clears `tau_margin`; otherwise spawn. I do not know whether “clears” is `>` or `>=`. Spawn reason codes from that rule: `mowm:FloorNotCleared`, `mowm:MarginNotCleared`. `mowm:spawnedWorld` (range `mowm:World`) when the action that ran was spawn. A bridge-spawned specialist is recoverable; a mis-route is not. That is a design statement, not a per-run measurement, so it is not a boolean on the record. I do not know whether a bridge spawn has a z depth.

**mowm:FastWordEncoder** — individual. Type `cheez:Encoder`. The routing-path text encoder. Not a sentence model.

**mowm:PipelineRun** — class. Subclass of `prov:Activity`. Stages that ran, in the stated order: Input, HYLA encoder, DSelect-k router, expert worlds along Z, bind and mixture, cleanup, CVL, HYLA decoder, output. Component names from `cubemind`, run on `grilly`: BlockCodes, HYLA, DSelectKGate, ContrastiveValueEstimator, DisARM, WorldEncoder, WorldManager. I do not know grilly’s URI, DisARM’s role, or whether WorldEncoder is FastWordEncoder. HYLA is a hypernetwork that generates block-code transition functions. I do not know whether HYLA is a neural model. It is not the bridge text encoder. No DSelect-k formula is stated, so none is minted.

**mowm:Axiom** — class. Subclass of `cheez:HostRecord`. `mowm:axiomOrigin` (range `skos:Concept`): `mowm:Seeded` or `mowm:Discovered`. Discovery steps, only if that step ran: semantic mapping, symbolic regression, axiom distillation. Selection by cosine similarity is `mowm:cosine` (range `cheez:Score`) when a selection ran. Composition by VSA bind is `mowm:vsaBind` (range `xsd:boolean`) when that composition ran. Counts are not constrained to 350 or 14.

**mowm:Router** — class. DSelect-k gating over a dynamic pool. `mowm:tauSpawn` and `mowm:tauPruneSteps` (range `cheez:Threshold`). Defaults 0.5 and 100 are not values of these properties until a calibration with screen, data, and date exists. I do not know that calibration. `mowm:maybeSpawn` when residual norm exceeds `tau_spawn`. “Exceeds” is stated as greater than. `mowm:residualNorm` (range `cheez:Score`). Idle prune uses `tau_prune_steps`. Whether MoWM code deletes or retires is not stated. This vocabulary has only `cheez:Pruned` (kept, with reason). I do not know the prune reason codes.

**mowm:InterWorldQuery** — class. Subclass of `prov:Activity`. Ran when the routed world’s best member fell short of `tau_answer` (`mowm:tauAnswer`, range `cheez:Threshold`). “Fell short” is below. I do not know the default of `tau_answer`.

**mowm:PossibilityQuery** — class. Subclass of `prov:Activity`. Fact-world members are facts encoded by the word table. `mowm:routingMode` (range `skos:Concept`): `mowm:BestMember` (stated default) or `mowm:Centroid`. Centroid routing does not carry over to fact worlds. Support at or above `tau_answer`. Prediction cleaned up to top-k. I do not know k. `mowm:topK` (range `xsd:integer`) only if the run recorded it. Refusal uses `mowm:Contradicts`.

**mowm:component** individuals are names only. I do not add capabilities beyond the sentences in the brief.

### cubelang (`cube:`, `https://w3id.org/cheezml/cubelang#`)

**cube:Program** — class. Subclass of `prov:Entity`. Source text in `cube:sourceText` (range `xsd:string`). `cube:interface` (range `xsd:string`). `ISolver` and `IAgent` are named interfaces. `implements IOrchestrator` fails unless that interface is declared. I do not know `IAgent`’s required functions. `cube:executesToday` (range `xsd:boolean`) only if a compile check ran. I did not run one, so I do not set it.

**cube:Run** — class. Subclass of `prov:Activity`. A VM execution that started. Outcomes that ran: `cube:Returned`, `cube:Error`, `cube:AskSuspend`. No Active. A container that compiles to nothing did not run `vm.start_from_file`, `vm.wire`, or `vm.bridge`. Those are not properties that can be set true by hand. I do not know their argument schemas.

**cube:Ask** — class. Subclass of `cheez:HostRecord`. The third outcome: the program cannot choose, suspends, and asks; resume is re-execution over `run-proto`. I do not know the `run-proto` schema. Six tests are stated; I do not know what they cover.

**cube:Opcode** — class. Only the stated bytes, as `cube:byteValue` (range `xsd:string`, the hex as given): CREATE `0x00`, ASSIGN `0x02`, ADD `0x03`, COMPARE `0x0b`, QUERY `0x0c`, STORE `0x0d`, COND `0x11`, JMP `0x14`, LABEL `0x15`, REMEMBER `0x21`, SUM `0x34`, RETURN `0x39`. I do not know bytes for sub, mul, div, push, pop, recall, forget, or while. I do not map `if`/`else` onto COND/JMP/LABEL; that mapping is not stated. `for`, `match`, intra-program `call`, and early `return` are not executed. Strict mode is the default and rejects `throw`. A map literal or a method-call result cannot be an operand. `container` compiles to nothing. `@hook` is a compile error. `??`, `revoke ... from`, `@pure`, and destructuring are not parsed. `event` is reserved. None of those are opcodes here.

## 2. One JSON-LD 1.1 `@context`

Proposal URL, not a stated one: `https://w3id.org/cheezml/context.jsonld`. I do not know the `owl:versionIRI`.

```json
{
  "@version": 1.1,
  "cheez": "https://w3id.org/cheezml/core#",
  "cubby": "https://w3id.org/cheezml/cubby#",
  "mowm": "https://w3id.org/cheezml/mowm#",
  "cube": "https://w3id.org/cheezml/cubelang#",
  "prov": "http://www.w3.org/ns/prov#",
  "skos": "http://www.w3.org/2004/02/skos/core#",
  "time": "http://www.w3.org/2006/time#",
  "schema": "https://schema.org/",
  "owl": "http://www.w3.org/2002/07/owl#",
  "xsd": "http://www.w3.org/2001/XMLSchema#",
  "cheez:validTime": { "@type": "@id" },
  "cheez:calibratedBy": { "@type": "@id" },
  "cheez:calibrationData": { "@type": "@id" },
  "cheez:boundEncoder": { "@type": "@id" },
  "cheez:boundCorpus": { "@type": "@id" },
  "cheez:trainingRun": { "@type": "@id" },
  "cheez:hasReason": { "@type": "@id" },
  "cheez:disposition": { "@type": "@id" },
  "cheez:workflowRunCrate": { "@type": "@id" },
  "cheez:thresholdValue": { "@type": "xsd:decimal" },
  "cheez:scoreValue": { "@type": "xsd:decimal" },
  "cheez:calibrationDate": { "@type": "xsd:date" },
  "cheez:revisionId": { "@type": "xsd:string" },
  "cheez:blobHash": { "@type": "xsd:string" },
  "cheez:calibrationScreen": { "@type": "xsd:string" },
  "cheez:killed": { "@type": "xsd:boolean" },
  "prov:wasGeneratedBy": { "@type": "@id" },
  "prov:used": { "@type": "@id" },
  "prov:wasAttributedTo": { "@type": "@id" },
  "prov:wasAssociatedWith": { "@type": "@id" },
  "prov:hadPrimarySource": { "@type": "@id" },
  "prov:wasRevisionOf": { "@type": "@id" },
  "prov:generatedAtTime": { "@type": "xsd:dateTime" },
  "time:hasBeginning": { "@type": "@id" },
  "time:hasEnd": { "@type": "@id" },
  "time:inXSDDateTime": { "@type": "xsd:dateTime" },
  "cubby:dialogueAct": { "@type": "@id" },
  "cubby:certificate": { "@type": "@id" },
  "cubby:proposer": { "@type": "@id" },
  "cubby:answer": { "@type": "@id" },
  "cubby:checksRan": { "@type": "@id" },
  "cubby:conflictKind": { "@type": "@id" },
  "cubby:entityGuard": { "@type": "@id" },
  "cubby:lastQuestion": { "@type": "@id" },
  "cubby:lastProgram": { "@type": "@id" },
  "cubby:lastAnswer": { "@type": "@id" },
  "cubby:openClarifyingQuestion": { "@type": "@id" },
  "cubby:inWorld": { "@type": "@id" },
  "cubby:sourceRegion": { "@type": "@id" },
  "cubby:reviewerRegion": { "@type": "@id" },
  "cubby:reviewedBy": { "@type": "@id" },
  "cubby:proposedBy": { "@type": "@id" },
  "cubby:vmSession": { "@type": "@id" },
  "cubby:breakingCase": { "@type": "@id" },
  "cubby:rejectionRate": { "@type": "@id" },
  "cubby:soft": { "@type": "xsd:boolean" },
  "cubby:promoted": { "@type": "xsd:boolean" },
  "cubby:inventedAtTurn": { "@type": "xsd:integer" },
  "cubby:atTurn": { "@type": "xsd:integer" },
  "cubby:escapeAttempts": { "@type": "xsd:integer" },
  "cubby:confidence": { "@type": "xsd:decimal" },
  "cubby:pleasure": { "@type": "xsd:decimal" },
  "cubby:arousal": { "@type": "xsd:decimal" },
  "cubby:dominance": { "@type": "xsd:decimal" },
  "cubby:reviewerTime": { "@type": "xsd:dateTime" },
  "mowm:action": { "@type": "@id" },
  "mowm:encoder": { "@type": "@id" },
  "mowm:bestCentroidScore": { "@type": "@id" },
  "mowm:runnerUpScore": { "@type": "@id" },
  "mowm:margin": { "@type": "@id" },
  "mowm:floor": { "@type": "@id" },
  "mowm:tauMargin": { "@type": "@id" },
  "mowm:tau": { "@type": "@id" },
  "mowm:tauSpawn": { "@type": "@id" },
  "mowm:tauPruneSteps": { "@type": "@id" },
  "mowm:tauAnswer": { "@type": "@id" },
  "mowm:spawnedWorld": { "@type": "@id" },
  "mowm:parentWorld": { "@type": "@id" },
  "mowm:idCounter": { "@type": "@id" },
  "mowm:hyla": { "@type": "@id" },
  "mowm:mergeStrategy": { "@type": "@id" },
  "mowm:depthRelation": { "@type": "@id" },
  "mowm:residualNorm": { "@type": "@id" },
  "mowm:worldId": { "@type": "xsd:integer" },
  "mowm:zDepth": { "@type": "xsd:integer" },
  "mowm:zMax": { "@type": "xsd:integer" },
  "mowm:worldCount": { "@type": "xsd:integer" },
  "mowm:topK": { "@type": "xsd:integer" },
  "cube:executesToday": { "@type": "xsd:boolean" }
}
```

## 3. Six example records

These are illustrations. They are not host-written records and must not be loaded as store facts. Every `urn:uuid` and every `«not stated»` is a placeholder. I do not know the real ids, times, numbers, proposer, refusal reason, or failing route clause. Decision Ontology triples are omitted because I do not know the IRI. Expand with the section 2 context; the short prefix block below is that context’s prefixes so a processor can expand offline.

Shared prefix block used by each example:

```json
{
  "@version": 1.1,
  "cheez": "https://w3id.org/cheezml/core#",
  "cubby": "https://w3id.org/cheezml/cubby#",
  "mowm": "https://w3id.org/cheezml/mowm#",
  "cube": "https://w3id.org/cheezml/cubelang#",
  "prov": "http://www.w3.org/ns/prov#",
  "time": "http://www.w3.org/2006/time#",
  "schema": "https://schema.org/",
  "xsd": "http://www.w3.org/2001/XMLSchema#"
}
```

Host placeholder for all six: `urn:uuid:00000000-0000-4000-8000-0000000000aa`. VM placeholder: `urn:uuid:00000000-0000-4000-8000-0000000000bb`.

### 3.1 Certified answer turn and certificate

Proposer below is a placeholder individual, not a finding that the grammar proposed it. No answer sentence. No truth score. No agreement. Check identity is a placeholder; satisfaction is a Decision Ontology triple and is omitted. I do not set passed.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "schema": "https://schema.org/"
  },
  "@graph": [
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000001",
      "@type": ["cubby:Turn", "cheez:HostRecord", "prov:Entity", "schema:Message"],
      "cubby:dialogueAct": { "@id": "cubby:CertifiedAnswerAct" },
      "cubby:entityId": "«not stated»",
      "cubby:certificate": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000002" },
      "cheez:templateId": "«not stated»",
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000002",
      "@type": ["cubby:Certificate", "cheez:HostRecord", "prov:Entity"],
      "cubby:answer": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000003" },
      "cubby:proposer": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000c1" },
      "prov:wasGeneratedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000004" },
      "prov:used": [
        { "@id": "urn:uuid:00000000-0000-4000-8000-000000000005" },
        { "@id": "urn:uuid:00000000-0000-4000-8000-000000000007" }
      ],
      "cubby:checksRan": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000009" },
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000003",
      "@type": "prov:Entity",
      "prov:wasGeneratedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000004" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000004",
      "@type": "prov:Activity",
      "prov:wasAssociatedWith": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000bb" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000005",
      "@type": ["cubby:ReadStatement", "prov:Entity"],
      "prov:hadPrimarySource": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000006" },
      "cheez:revisionId": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000007",
      "@type": ["cubby:AppliedRule", "prov:Entity"]
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000009",
      "@type": ["cubby:Check", "prov:Activity"]
    }
  ]
}
```

`urn:uuid:00000000-0000-4000-8000-0000000000c1` stands for one of the five proposers. Which one is not stated.

### 3.2 Refusal with its reason

The reason code is not stated. The property is present because a refusal has a reason. The value is a placeholder, not a choice among `cubby:NoPlan`, `cubby:RelationMissing`, `cubby:FactMissing`, `cubby:WalkFailed`, `cubby:NotCertified`, `cubby:Ambiguous`, `cubby:DatesTooCoarse`, `cubby:UnreadableSource`, or a conflict kind. No certificate. No check node. No sentence.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "schema": "https://schema.org/"
  },
  "@id": "urn:uuid:00000000-0000-4000-8000-000000000010",
  "@type": ["cubby:Turn", "cheez:HostRecord", "prov:Entity", "schema:Message"],
  "cubby:dialogueAct": { "@id": "cubby:RefusalAct" },
  "cheez:hasReason": "«not stated»",
  "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
  "prov:generatedAtTime": "«not stated»",
  "cheez:validTime": {
    "@type": "time:Interval",
    "time:hasBeginning": {
      "@type": "time:Instant",
      "time:inXSDDateTime": "«not stated»"
    }
  }
}
```

### 3.3 MoWM routing decision that ends in a spawn

Action is spawn because that is the example. Which clause failed is not stated, so the reason value is `«not stated»`, not a claim that the floor failed. Scores and thresholds are different objects. Threshold numbers, screen, data, date, and corpus are not stated. Encoder is FastWordEncoder because that is the bridge rule, not because a calibration was supplied. I do not use 0.680, 0.589, or 0.5. Spawned world id is not stated and is not parent + 1. I do not know the counter for a bridge spawn, so `mowm:idCounter` is omitted.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#"
  },
  "@graph": [
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000020",
      "@type": ["mowm:RoutingDecision", "cheez:HostRecord", "prov:Entity"],
      "mowm:action": { "@id": "mowm:Spawn" },
      "mowm:encoder": { "@id": "mowm:FastWordEncoder" },
      "mowm:worldCount": "«not stated»",
      "mowm:bestCentroidScore": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000021" },
      "mowm:runnerUpScore": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000022" },
      "mowm:margin": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000023" },
      "mowm:floor": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000024" },
      "mowm:tauMargin": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000025" },
      "mowm:spawnedWorld": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000029" },
      "cheez:hasReason": "«not stated»",
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000021",
      "@type": "cheez:Score",
      "cheez:scoreValue": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000022",
      "@type": "cheez:Score",
      "cheez:scoreValue": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000023",
      "@type": "cheez:Score",
      "cheez:scoreValue": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000024",
      "@type": "cheez:Threshold",
      "cheez:thresholdValue": "«not stated»",
      "cheez:calibratedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000026" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000025",
      "@type": "cheez:Threshold",
      "cheez:thresholdValue": "«not stated»",
      "cheez:calibratedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000027" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000026",
      "@type": "cheez:Calibration",
      "cheez:calibrationScreen": "«not stated»",
      "cheez:calibrationData": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000028" },
      "cheez:calibrationDate": "«not stated»",
      "cheez:boundEncoder": { "@id": "mowm:FastWordEncoder" },
      "cheez:boundCorpus": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000028" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000027",
      "@type": "cheez:Calibration",
      "cheez:calibrationScreen": "«not stated»",
      "cheez:calibrationData": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000028" },
      "cheez:calibrationDate": "«not stated»",
      "cheez:boundEncoder": { "@id": "mowm:FastWordEncoder" },
      "cheez:boundCorpus": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000028" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000028",
      "@type": "prov:Entity",
      "cheez:blobHash": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000029",
      "@type": ["mowm:World", "cheez:HostRecord", "prov:Entity"],
      "mowm:worldId": "«not stated»",
      "cheez:blobHash": "«not stated»"
    }
  ]
}
```

### 3.4 World split, sub-world at z+1

The child exists at z+1 relative to the parent. Numeric z, z_max, world ids, counter, residual, merge strategy, and HYLA count are not stated. Ids are omitted rather than set to adjacent integers. Tau is a threshold object. Its value and calibration are not stated. I do not record a passed spawn check.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#"
  },
  "@graph": [
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000030",
      "@type": ["mowm:World", "cheez:HostRecord", "prov:Entity"],
      "mowm:worldId": "«not stated»",
      "mowm:zDepth": "«not stated»",
      "mowm:zMax": "«not stated»",
      "mowm:tau": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000032" },
      "cheez:blobHash": "«not stated»",
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000031",
      "@type": ["mowm:World", "cheez:HostRecord", "prov:Entity"],
      "mowm:parentWorld": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000030" },
      "mowm:depthRelation": { "@id": "mowm:ChildAtZPlusOne" },
      "mowm:worldId": "«not stated»",
      "mowm:zDepth": "«not stated»",
      "cheez:blobHash": "«not stated»",
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000032",
      "@type": "cheez:Threshold",
      "cheez:thresholdValue": "«not stated»",
      "cheez:calibratedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000033" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000033",
      "@type": "cheez:Calibration",
      "cheez:calibrationScreen": "«not stated»",
      "cheez:calibrationData": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000034" },
      "cheez:calibrationDate": "«not stated»",
      "cheez:boundEncoder": { "@id": "mowm:FastWordEncoder" },
      "cheez:boundCorpus": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000034" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000034",
      "@type": "prov:Entity",
      "cheez:blobHash": "«not stated»"
    }
  ]
}
```

Binding the world-split tau to FastWordEncoder is a placeholder binding. I do not know the encoder for `world.tau`.

### 3.5 Helper rejected in its sandbox (order 0062)

The session ran. The host did not promote it. The helper is kept. The breaking case id is a placeholder. Which break code applies is not stated. Rejection rate, window, and escape-attempt count are not stated, so they are omitted. No compile-passed flag. No Active.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#"
  },
  "@graph": [
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000040",
      "@type": ["cubby:Helper", "cheez:HostRecord", "prov:Entity"],
      "cubby:proposedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000c1" },
      "cubby:vmSession": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000041" },
      "cubby:promoted": false,
      "cheez:disposition": { "@id": "cheez:Rejected" },
      "cheez:hasReason": "«not stated»",
      "cubby:breakingCase": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000042" },
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000041",
      "@type": "prov:Activity"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000042",
      "@type": ["cheez:HostRecord", "prov:Entity"]
    }
  ]
}
```

`cubby:promoted: false` means the promotion decision ran and did not promote. If that reading is wrong, I do not know a second way to record it.

### 3.6 Invoice field reviewed, then committed (order 0061)

Review happened, then the host committed. Reviewer role, id, time, and region text are not stated. Reader version is not stated. Which exact checks ran is not stated, so no check nodes and no passed flags. I do not know whether the field was derived, whether it sat above the second-approval amount, or whether a prior version existed. No second reviewer. Not a certificate. Not certified.

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "https://schema.org/",
    "time": "http://www.w3.org/2006/time#"
  },
  "@graph": [
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000050",
      "@type": ["cheez:FactVersion", "cheez:HostRecord", "prov:Entity"],
      "cubby:sourceRegion": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000051" },
      "cubby:reviewedBy": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000053" },
      "prov:hadPrimarySource": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000051" },
      "prov:wasAttributedTo": { "@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa" },
      "prov:generatedAtTime": "«not stated»",
      "cheez:validTime": {
        "@type": "time:Interval",
        "time:hasBeginning": {
          "@type": "time:Instant",
          "time:inXSDDateTime": "«not stated»"
        }
      }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000051",
      "@type": ["cubby:DocumentRegion", "prov:Entity"],
      "cubby:regionText": "«not stated»",
      "cubby:readerVersion": "«not stated»"
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000052",
      "@type": ["cubby:Reviewer", "prov:Agent"],
      "cubby:reviewerRole": "«not stated»",
      "schema:identifier": "«not stated»",
      "cubby:reviewerTime": "«not stated»",
      "cubby:reviewerRegion": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000051" }
    },
    {
      "@id": "urn:uuid:00000000-0000-4000-8000-000000000053",
      "@type": "prov:Activity",
      "prov:wasAssociatedWith": { "@id": "urn:uuid:00000000-0000-4000-8000-000000000052" }
    }
  ]
}
```

The field’s number is not stated. I do not invent one, so I also do not show it inside `cubby:regionText`.

## 4. SHACL shapes

The three classes I judge most important are `cubby:Certificate`, `cheez:Threshold`, and `mowm:RoutingDecision`. The recurring faults land there: certification by declaration or agreement, scores used as thresholds, and routing records that embed a model or a parent-plus-one id. `cheez:FactVersion` would be fourth for the two clocks. I do not emit a fourth shape.

Examples in section 3 are not expected to pass these shapes until `«not stated»` is replaced with real typed values. I do not encode “clears” as `>` or `>=`. I do not forbid numerically adjacent world ids; a counter may issue them. There is no parent-plus-one property to constrain.

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix cheez: <https://w3id.org/cheezml/core#> .
@prefix cubby: <https://w3id.org/cheezml/cubby#> .
@prefix mowm: <https://w3id.org/cheezml/mowm#> .

cheez:ThresholdShape a sh:NodeShape ;
  sh:targetClass cheez:Threshold ;
  sh:closed true ;
  sh:property [
    sh:path cheez:thresholdValue ;
    sh:datatype xsd:decimal ;
    sh:minCount 1 ;
    sh:maxCount 1
  ] ;
  sh:property [
    sh:path cheez:calibratedBy ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:node cheez:CalibrationShape
  ] .

cheez:CalibrationShape a sh:NodeShape ;
  sh:closed true ;
  sh:property [ sh:path cheez:calibrationScreen ; sh:datatype xsd:string ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:calibrationData ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:calibrationDate ; sh:datatype xsd:date ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:boundEncoder ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:boundCorpus ; sh:minCount 1 ; sh:maxCount 1 ] .

cheez:ScoreShape a sh:NodeShape ;
  sh:class cheez:Score ;
  sh:closed true ;
  sh:property [ sh:path cheez:scoreValue ; sh:datatype xsd:decimal ; sh:minCount 1 ; sh:maxCount 1 ] .

cubby:CertificateShape a sh:NodeShape ;
  sh:targetClass cubby:Certificate ;
  sh:closed true ;
  sh:property [ sh:path cubby:answer ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cubby:proposer ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ; sh:class prov:Activity ] ;
  sh:property [ sh:path prov:used ; sh:minCount 1 ] ;
  sh:property [ sh:path cubby:checksRan ; sh:minCount 1 ] ;
  sh:property [ sh:path prov:wasAttributedTo ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path prov:generatedAtTime ; sh:datatype xsd:dateTime ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:validTime ; sh:minCount 1 ; sh:maxCount 1 ] .

mowm:RoutingDecisionShape a sh:NodeShape ;
  sh:targetClass mowm:RoutingDecision ;
  sh:closed true ;
  sh:property [
    sh:path mowm:action ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:in ( mowm:Route mowm:Spawn )
  ] ;
  sh:property [
    sh:path mowm:encoder ;
    sh:hasValue mowm:FastWordEncoder ;
    sh:minCount 1 ;
    sh:maxCount 1
  ] ;
  sh:property [ sh:path mowm:worldCount ; sh:datatype xsd:integer ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path mowm:bestCentroidScore ; sh:minCount 1 ; sh:maxCount 1 ; sh:node cheez:ScoreShape ] ;
  sh:property [ sh:path mowm:floor ; sh:minCount 1 ; sh:maxCount 1 ; sh:class cheez:Threshold ] ;
  sh:property [ sh:path prov:wasAttributedTo ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path prov:generatedAtTime ; sh:datatype xsd:dateTime ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path cheez:validTime ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path mowm:runnerUpScore ; sh:maxCount 1 ; sh:node cheez:ScoreShape ] ;
  sh:property [ sh:path mowm:margin ; sh:maxCount 1 ; sh:node cheez:ScoreShape ] ;
  sh:property [ sh:path mowm:tauMargin ; sh:maxCount 1 ; sh:class cheez:Threshold ] ;
  sh:property [ sh:path mowm:spawnedWorld ; sh:maxCount 1 ; sh:class mowm:World ] ;
  sh:property [ sh:path cheez:hasReason ; sh:maxCount 1 ] ;
  sh:sparql [
    sh:message "A spawn record needs the spawned world and a reason code." ;
    sh:select """
      PREFIX mowm: <https://w3id.org/cheezml/mowm#>
      SELECT $this WHERE {
        $this mowm:action mowm:Spawn .
        FILTER NOT EXISTS { $this mowm:spawnedWorld ?w }
      }"""
  ] ;
  sh:sparql [
    sh:message "A spawn record needs a reason code." ;
    sh:select """
      PREFIX mowm: <https://w3id.org/cheezml/mowm#>
      PREFIX cheez: <https://w3id.org/cheezml/core#>
      SELECT $this WHERE {
        $this mowm:action mowm:Spawn .
        FILTER NOT EXISTS { $this cheez:hasReason ?r }
      }"""
  ] ;
  sh:sparql [
    sh:message "Two or more worlds require runner-up, margin, and tau_margin." ;
    sh:select """
      PREFIX mowm: <https://w3id.org/cheezml/mowm#>
      SELECT $this WHERE {
        $this mowm:worldCount ?n .
        FILTER (?n >= 2)
        FILTER NOT EXISTS { $this mowm:runnerUpScore ?a }
        FILTER NOT EXISTS { $this mowm:margin ?b }
        FILTER NOT EXISTS { $this mowm:tauMargin ?c }
      }"""
  ] .
```

The host URI is not stated, so the shapes require `prov:wasAttributedTo` and cannot pin its value. Closed shapes reject a rationale property, a truth-score property, a sentence-model property, and an embedded vector, because those terms are not in the allowed lists.

## 5. CubeLang programs

These are the executing subset in intent. They are not labelled “spec only: does not compile today”. I do not know function-body grammar, storage types, compare polarity, operand order, comment syntax, or `IAgent`’s required functions. I did not run `cubelang check`. There is no passed compile record. I do not claim bytecode. ASK is the right outcome when an input is missing, but I do not know the syntax that tests missing, so ASK is not written into the text.

Neither program names a model. `query` reads host-stored numbers. I do not know whether the language can point `query` at a model; the host must not store a sentence-model score there. Neither program contains a parent id, so neither can compute parent id + 1. Neither assigns 0.5, 0.680, or 0.589. Neither spawns because a score is merely low. The route program does not apply the inter-world `tau_answer` rule. The spawn program does not treat `world.tau` and `tau_spawn` as the same; I do not know whether they are. Counter advancement is not in the program; I do not know pre-increment versus post-increment. The host allocates the id and writes the CheezML record from what ran.

`MoWMRoute` stores `route` or `spawn` under the bridge rule only. It does not create a world.

```
program MoWMRoute implements IAgent {
  storage { worldCount bestScore runnerScore floor tauMargin margin action one two }
  functions {
    decide {
      query worldCount
      query bestScore
      query runnerScore
      query floor
      query tauMargin
      assign one 1
      assign two 2
      compare bestScore floor
      if bestScore clears floor {
        compare worldCount two
        if worldCount is two or more {
          sub margin bestScore runnerScore
          compare margin tauMargin
          if margin clears tauMargin {
            assign action "route"
          } else {
            assign action "spawn"
          }
        } else {
          assign action "route"
        }
      } else {
        assign action "spawn"
      }
      store action
      return
    }
  }
}
```

`MoWMSpawnSubWorld` is `spawn_subworld` plus the router residual test. Refuse at `z_max`. Otherwise spawn only if residual norm exceeds the queried `tauSpawn`. Child depth is z+1. The id is queried from storage the host already allocated.

```
program MoWMSpawnSubWorld implements IAgent {
  storage { z zMax residualNorm tauSpawn allocatedId childZ action one }
  functions {
    spawn {
      query z
      query zMax
      query residualNorm
      query tauSpawn
      query allocatedId
      assign one 1
      compare z zMax
      if z is zMax {
        assign action "refuse"
      } else {
        compare residualNorm tauSpawn
        if residualNorm exceeds tauSpawn {
          add childZ z one
          assign action "spawn"
          store childZ
          store allocatedId
        } else {
          assign action "no-spawn"
        }
      }
      store action
      return
    }
  }
}
```

`return` appears once, at the end of each function. I do not know whether that still counts as early `return`. If it does, these programs do not compile today, and that label would then apply. `forget` is not used. I do not know whether `forget` deletes.

## 6. What I would cut, and open questions

I would cut from this design, and from the prior drafts: any prose rationale; any truth score; any Active, Passed, or certified-by-agreement status; any seed-as-reproduction property; any embedded vector or weight; any sentence-model term on the routing path; any parent-id-plus-one property; any version segment in the namespace; any invented opcode or assembly listing; any container-bridge record that says the bridge ran; swarms, which are not stated here; a DSelect-k formula; study figures used as thresholds or as example values; a delete property; EmotionML properties that pretend to be that spec’s RDF; a guessed Decision Ontology, STATO, ISO, RO-Crate profile, or nanopublication IRI; hormone dimension names; a gain unit; a second-approval amount; a Lean proof or a “verified fact” flag. I would also cut `cubby:promoted: false` if a boolean false cannot be distinguished from a decision that never ran. I would not cut the hard rules.

Open questions, which I do not know the answers to:

1. Decision Ontology IRI, and whether `cubby:Check` should be replaced by it entirely.
2. Hash algorithm and hash IRI form.
3. Host URI, VM URI, and whether `cubby:ProgramEmitter` is an external model.
4. Screen-id scheme, and the screen, data, and date for the stated defaults 0.5 and 100.
5. Whether “clears” is strict, and how CubeLang `compare` binds into `if`.
6. Whether order 0055’s two clocks apply to every host record or only to facts, and how an open valid interval is written.
7. Which counter `spawn_subworld` uses, and whether bridge spawn has a z.
8. Whether `world.tau` and `tau_spawn` are the same threshold.
9. Whether prune in MoWM code deletes. The vocabulary only retires.
10. Top-k, `tau_answer`’s deployment value, and `IAgent`’s required functions.
11. Bytes and operand order for the executing operations whose bytes are not listed.
12. Whether a final `return` is an early `return`.
13. `run-proto` schema, and how a missing `query` is tested so ASK can be written.
14. Whether `forget` deletes.
15. `owl:versionIRI`, the context URL, and the redirect path on `ns.grillcheeseai.com`.
16. Workflow Run RO-Crate profile IRI; nanopublication namespace; EmotionML URL; ISO 24617-2 IRIs.
17. Whether `mowm:Contradicts` joins the dialogue refusal list.
18. Second-approval amount, currency, and calibration.
19. Gain’s unit for a domain change.
20. Hormone dimension labels.
21. Whether WorldEncoder is FastWordEncoder, and whether HYLA is a neural model. The bridge text encoder remains FastWordEncoder either way.
22. Reviewer-time versus the two clocks, and the field value type for story state changes.
23. Pre-increment versus post-increment of the world-id counters.

## moonshotai/kimi-k3.1-pro

*16526 tokens out, 67.5 s, finish `stop`*

## 1. Classes and properties

CheezML mints only the terms below. Standard properties keep their standard domains and ranges. `prov:Entity`, `prov:Activity`, `prov:Agent`, `do:Question`, `do:Option`, `do:Decision`, `do:is_satisfied_by`, `do:is_violated_by`, `time:Interval`, `skos:Concept`, RO-Crate, STATO and EmotionML are used directly where applicable.

The Decision Ontology namespace used below is provisional; I do not know its authoritative published IRI from the brief.

### `core#`

| Term | Domain → range, or parent | Definition | Reuse/specialisation |
|---|---|---|---|
| `cheez:HostRecord` | `rdfs:Class` | A record emitted by the host from an execution or stored state. | General superclass for host records. |
| `cheez:FactVersion` | `prov:Entity` | One valid-time/recorded-time version of a stored statement. | PROV-O entity with versioning. |
| `cheez:statement` | `FactVersion` → `rdf:Statement` | The reified statement held by this version. | RDF reification. |
| `cheez:validTime` | `FactVersion` → `time:Interval` | When the statement held in the world. | OWL-Time. |
| `cheez:recordedAt` | `prov:Entity` → `xsd:dateTime` | When the store learned the statement. | Transaction-time companion to OWL-Time. |
| `cheez:sourceRevision` | `prov:Entity` → literal | Source revision read by a run. | Companion to `prov:wasDerivedFrom`. |
| `cheez:ContentAddress` | `rdfs:Class` | Hash and algorithm identifying an external artifact. | Content-addressed external resource. |
| `cheez:hashAlgorithm` | `ContentAddress` → `skos:Concept` | Hash algorithm code. | SKOS-coded algorithm. |
| `cheez:hashValue` | `ContentAddress` → `xsd:string` | Hash bytes in the recorded encoding. | No suitable general PROV property. |
| `cheez:contentAddress` | `rdfs:Resource` → `ContentAddress` | References, rather than embeds, an artifact. | Used for vectors, weights and blobs. |
| `cheez:trainingRun` | `prov:Entity` → `prov:Activity` | Training run that produced referenced weights. | PROV-O activity. |
| `cheez:CalibratedThreshold` | `rdfs:Class` | A threshold together with its calibration provenance. | Prevents scores becoming uncalibrated thresholds. |
| `cheez:thresholdValue` | `CalibratedThreshold` → `xsd:decimal` | Numeric threshold value. | New. |
| `cheez:thresholdParameter` | `CalibratedThreshold` → `skos:Concept` | Parameter controlled by the threshold. | SKOS code. |
| `cheez:calibratedBy` | `CalibratedThreshold` → `prov:Activity` | Calibration screen or run. | PROV-O activity. |
| `cheez:calibrationData` | `CalibratedThreshold` → `prov:Entity` | Calibration corpus or dataset. | PROV-O entity. |
| `cheez:calibratedOn` | `CalibratedThreshold` → `xsd:date` | Calibration date. | New. |
| `cheez:scoreValue` | `prov:Entity` → `xsd:decimal` | A reported score, never implicitly a threshold. | STATO can describe the statistic. |
| `cheez:scoreMetric` | `prov:Entity` → `skos:Concept` | Code identifying what a score measures. | STATO/SKOS. |
| `cheez:softCheck` | `do:Criterion` or `prov:Activity` → `xsd:boolean` | Marks a tone check as soft rather than exact. | Decision Ontology criterion annotation. |
| `cheez:placeholders` | `HostRecord` → `xsd:boolean` | Marks example values as placeholders. | Documentation/tooling flag; not operational state. |

Corrections use `prov:wasRevisionOf`; closing an old version uses `prov:invalidatedAtTime`. Nothing is deleted from the record graph. Evaluation packages use Workflow Run RO-Crate; verdicts, including killed results, use nanopublications and STATO.

### `cubby#`

| Term | Domain → range, or parent | Definition | Reuse/specialisation |
|---|---|---|---|
| `cubby:Turn` | `prov:Activity` | One host-recorded conversation turn. | PROV-O activity. |
| `cubby:dialogueAct` | `Turn` → `skos:Concept` | Question, certified answer, refusal, clarification, follow-up or small-talk code. | ISO 24617-2 is the reference set, by link. |
| `cubby:conversation` | `Turn` → `prov:Entity` | Conversation whose state is held by the host. | PROV-O. |
| `cubby:turnIndex` | `Turn` → `xsd:integer` | Host-assigned turn number. | New. |
| `cubby:question` | `Turn` → `prov:Entity` | Question associated with the turn. | PROV-O. |
| `cubby:answer` | `Turn` → `prov:Entity` | Answer emitted by the turn. | PROV-O. |
| `cubby:program` | `Turn` or `prov:Activity` → `prov:Entity` | Program used or proposed. | PROV-O. |
| `cubby:openClarification` | `prov:Entity` → `do:Question` | Host state’s open clarifying question. | Decision Ontology. |
| `cubby:followUpOf` | `Turn` → `Turn` | Earlier turn rewritten by this follow-up. | PROV-O derivation between activities. |
| `cubby:rewrite` | `Turn` → `prov:Entity` | Guarded rewrite of the preceding question. | PROV-O. |
| `cubby:entityGuardRun` | `Turn` → `prov:Activity` | The entity-guard execution, if run. | PROV-O; outcome comes from the run. |
| `cubby:proposer` | `prov:Entity` → `prov:Agent` | Proposer that supplied the program. | Specialises provenance attribution. |
| `cubby:Certificate` | `do:Decision`, `prov:Entity` | Derivation certificate for an answer under recorded checks. | PROV-O plus Decision Ontology. |
| `cubby:certifies` | `Certificate` → `prov:Entity` | Answer whose derivation is certified; not a correctness claim. | Decision result. |
| `cubby:decisionQuestion` | `do:Decision` → `do:Question` | Question decided by the certificate or refusal. | Decision Ontology. |
| `cubby:option` | `do:Decision` → `do:Option` | A proposer’s candidate program. | Decision Ontology. |
| `cubby:criterion` | `do:Decision` → `do:Criterion` | Check used in the decision. | Decision Ontology. |
| `cubby:programRun` | `Certificate` → `prov:Activity` | Program execution covered by the certificate. | PROV-O. |
| `cubby:statementRead` | `Certificate` → `prov:Entity` | Statement read, retaining source and revision. | PROV-O. |
| `cubby:ruleApplied` | `Certificate` → `prov:Entity` | Rule applied by the run. | PROV-O. |
| `cubby:Refusal` | `do:Decision` | Host-recorded refusal decision. | Decision Ontology. |
| `cubby:refusalReason` | `Refusal` → `skos:Concept` | Refusal-reason code. | SKOS; no prose rationale. |
| `cubby:conflictKind` | `prov:Entity` → `skos:Concept` | Value, type, temporal, logical or recorded-against-derived conflict. | SKOS. |
| `cubby:derivedStatement` | `prov:Entity` → `prov:Entity` | Derived statement involved in a conflict. | PROV-O. |
| `cubby:SplitQuestion` | `do:Question` | Question decomposed into separately routed parts. | Decision Ontology. |
| `cubby:part` | `SplitQuestion` → `do:Question` | One split part. | Decision Ontology. |
| `cubby:partRoute` | `do:Question` → `prov:Activity` | Routing record for a part. | PROV-O. |
| `cubby:verdict` | `rdfs:Resource` → `skos:Concept` | Host-recorded verdict code. | SKOS. |
| `cubby:combinationProgram` | `SplitQuestion` → `prov:Entity` | Program combining part results. | PROV-O. |
| `cubby:missingPart` | `Turn` → `do:Question` | Missing part named by a partial answer. | Decision Ontology. |
| `cubby:EmotionReading` | `prov:Entity` | Confidence-bearing inference about a user’s emotion. | EmotionML 1.0 export by link. |
| `cubby:emotionSubject` | `EmotionReading` → `prov:Agent` | Subject of the inference. | PROV-O. |
| `cubby:confidence` | `EmotionReading` → `xsd:decimal` | Inference confidence in `[0,1]`. | New; never a fact score. |
| `cubby:dimensionSet` | `prov:Entity` → `skos:Concept` | `pad-dimensions` or Cubby’s separate hormone-state set. | EmotionML vocabulary by link. |
| `cubby:emotionExport` | `prov:Entity` → `schema:DigitalDocument` | External EmotionML representation. | EmotionML 1.0. |
| `cubby:StoryWorld` | `prov:Entity` | Store isolated from the real-world store. | PROV-O. |
| `cubby:isolatedStore` | `StoryWorld` → `prov:Entity` | Isolated statement store. | PROV-O. |
| `cubby:StateChange` | `prov:Activity` | Dated story-state change with old value, new value and turn. | PROV-O activity. |
| `cubby:oldValue` | `StateChange` → `rdf:Statement` | Previous value. | RDF reification. |
| `cubby:newValue` | `StateChange` → `rdf:Statement` | Replacement value. | RDF reification. |
| `cubby:inventedAtTurn` | `FactVersion` → `xsd:integer` | Turn at which a story fact was invented. | Replaces a prose source string. |
| `cubby:Region` | `prov:Entity` | Document region with text and reader version. | RO-Crate/PROV compatible entity. |
| `cubby:regionText` | `Region` → `xsd:string` | Exact text read from the region. | New. |
| `cubby:readerVersion` | `Region` → `xsd:string` | Reader version that produced the region. | New. |
| `cubby:Field` | `prov:Entity` | Candidate document field tied to a region. | PROV-O. |
| `cubby:region` | `Field` or `prov:Activity` → `Region` | Supporting document region. | PROV-O. |
| `cubby:exactCheck` | `Field` → `prov:Activity` | Format, arithmetic, PO, supplier, duplicate or number-in-region check. | PROV-O. |
| `cubby:suggestedOnly` | `Field` → `xsd:boolean` | Whether the value is merely a derived suggestion. | New. |
| `cubby:Review` | `prov:Activity` | Human review of a field. | PROV-O; reviewer is a source. |
| `cubby:reviewedField` | `Review` → `Field` | Field reviewed. | PROV-O. |
| `cubby:reviewOutcome` | `Review` → `skos:Concept` | Recorded review outcome code. | SKOS. |
| `cubby:Commit` | `prov:Activity` | Host commit following required checks and approvals. | PROV-O. |
| `cubby:committedField` | `Commit` → `Field` | Field committed. | PROV-O. |
| `cubby:Helper` | `schema:SoftwareSourceCode` | Proposed helper retained whether adopted or rejected. | schema.org software source. |
| `cubby:SandboxRun` | `prov:Activity` | Locked-down VM test of a helper. | PROV-O. |
| `cubby:testedHelper` | `SandboxRun` → `Helper` | Helper tested. | PROV-O. |
| `cubby:brokenBy` | `Helper` → `prov:Entity` | Verified case that caused rejection. | PROV-O. |
| `cubby:rejectionRate` | `prov:Activity` → `xsd:decimal` | Proposer rejection rate over the recorded window. | Score, not a threshold. |
| `cubby:escapeAttempts` | `SandboxRun` → `xsd:integer` | Observed escape attempts; adoption requires zero. | New. |
| `cubby:DomainWorld` | `prov:Entity` | Ontology module, open records, rules and question set. | PROV-O collection. |

Reviewer agent, role, time and region use `prov:wasAssociatedWith`, `prov:hadRole`, `prov:startedAtTime`, `prov:endedAtTime` and `cubby:region`. Amount and currency use `schema:price` and `schema:priceCurrency`.

### `mowm#`

| Term | Domain → range, or parent | Definition | Reuse/specialisation |
|---|---|---|---|
| `mowm:World` | `prov:Entity` | A tree node containing one or more HYLAs. | PROV-O. |
| `mowm:worldId` | `World` → `xsd:integer` | Host-allocated global or router-spawned world id. | New; never derived by adding one to a parent. |
| `mowm:z` | `World` → `xsd:integer` | Depth on the Z axis. | New. |
| `mowm:zMax` | `World` → `xsd:integer` | Maximum permitted depth. | New. |
| `mowm:parentWorld` | `World` → `World` | Parent at z−1. | Tree provenance. |
| `mowm:hyla` | `World` → `prov:Entity` | HYLA held by the world. | Component provenance. |
| `mowm:mergeStrategy` | `World` → `skos:Concept` | Transition-delta merge strategy code. | SKOS. |
| `mowm:cleanupStrategy` | `prov:Activity` → `skos:Concept` | Cleanup strategy used. | SKOS. |
| `mowm:RoutingDecision` | `prov:Activity` | Host record of DSelect-k bridge routing. | PROV-O. |
| `mowm:inputEncoding` | `RoutingDecision` → `prov:Entity` | Word-table encoding used for the input. | PROV-O; no sentence model. |
| `mowm:bestWorld` | `RoutingDecision` → `World` | Highest-scoring candidate world. | New. |
| `mowm:runnerUpWorld` | `RoutingDecision` → `World` | Second candidate, when one exists. | New. |
| `mowm:centroidScore` | `RoutingDecision` → `xsd:decimal` | Best centroid score. | Measurement only. |
| `mowm:margin` | `RoutingDecision` → `xsd:decimal` | Best-minus-runner-up score. | Measurement only. |
| `mowm:absoluteFloor` | `RoutingDecision` → `CalibratedThreshold` | Calibrated absolute floor. | Core threshold. |
| `mowm:marginThreshold` | `RoutingDecision` → `CalibratedThreshold` | Calibrated margin threshold. | Core threshold. |
| `mowm:routeOutcome` | `RoutingDecision` → `skos:Concept` | Route or spawn outcome code. | SKOS. |
| `mowm:Spawn` | `prov:Activity` | Creation of a specialist or residual sub-world. | PROV-O. |
| `mowm:spawnedWorld` | `Spawn` → `World` | World allocated by the host/router. | PROV-O. |
| `mowm:residualNorm` | `Spawn` or `RoutingDecision` → `xsd:decimal` | Measured residual norm. | Measurement only. |
| `mowm:spawnThreshold` | `Spawn` → `CalibratedThreshold` | Calibrated `tau_spawn`. | Core threshold. |
| `mowm:member` | `World` → `prov:Entity` | Axiom or fact member. | Collection membership. |
| `mowm:Prune` | `prov:Activity` | Host pruning record retaining the world and reason. | Retire-never-delete policy. |
| `mowm:prunedWorld` | `Prune` → `World` | World retired from the active pool. | PROV-O. |
| `mowm:idleSteps` | `Prune` → `xsd:integer` | Observed idle duration. | New. |

Vector or weight-bearing entities use `cheez:contentAddress`; trained weights also use `cheez:trainingRun`. Support, answer thresholds, pruning parameters and inter-world-query results follow the same measurement-versus-calibrated-threshold distinction.

### `cubelang#`

| Term | Domain → range, or parent | Definition | Reuse/specialisation |
|---|---|---|---|
| `cube:Program` | `schema:SoftwareSourceCode`, `prov:Entity` | CubeLang source identified by hash. | schema.org and PROV-O. |
| `cube:ProgramRun` | `prov:Activity` | One execution over the recorded input. | PROV-O. |
| `cube:executedProgram` | `ProgramRun` → `Program` | Program executed. | PROV-O. |
| `cube:bytecode` | `ProgramRun` → `prov:Entity` | External bytecode artifact referenced by hash. | No opcode vocabulary. |
| `cube:runOutcome` | `ProgramRun` → `skos:Concept` | Return, error or suspension outcome. | SKOS. |
| `cube:Suspension` | `prov:Entity` | ASK state produced by a run that cannot choose. | Third execution outcome. |
| `cube:askedQuestion` | `Suspension` → `do:Question` | Caller question causing suspension. | Decision Ontology. |
| `cube:resumedRun` | `Suspension` → `ProgramRun` | Re-execution over `run-proto`. | PROV-O. |
| `cube:compilerVersion` | `ProgramRun` → `xsd:string` | Compiler used for the run. | New. |

## 2. JSON-LD 1.1 `@context`

```json
{
  "@context": {
    "@version": 1.1,
    "cheez": "https://w3id.org/cheezml/core#",
    "cubby": "https://w3id.org/cheezml/cubby#",
    "mowm": "https://w3id.org/cheezml/mowm#",
    "cube": "https://w3id.org/cheezml/cubelang#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "schema": "https://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "do": "https://w3id.org/decision-ontology/",
    "id": "@id",
    "type": "@type",
    "value": "rdf:value",
    "recordedAt": {"@id": "cheez:recordedAt", "@type": "xsd:dateTime"},
    "calibratedOn": {"@id": "cheez:calibratedOn", "@type": "xsd:date"},
    "startedAt": {"@id": "prov:startedAtTime", "@type": "xsd:dateTime"},
    "endedAt": {"@id": "prov:endedAtTime", "@type": "xsd:dateTime"},
    "inXSDDateTimeStamp": {"@id": "time:inXSDDateTimeStamp", "@type": "xsd:dateTimeStamp"},
    "price": {"@id": "schema:price", "@type": "xsd:decimal"},
    "confidence": {"@id": "cubby:confidence", "@type": "xsd:decimal"},
    "turnIndex": {"@id": "cubby:turnIndex", "@type": "xsd:integer"},
    "worldId": {"@id": "mowm:worldId", "@type": "xsd:integer"},
    "z": {"@id": "mowm:z", "@type": "xsd:integer"},
    "zMax": {"@id": "mowm:zMax", "@type": "xsd:integer"},
    "escapeAttempts": {"@id": "cubby:escapeAttempts", "@type": "xsd:integer"},
    "thresholdValue": {"@id": "cheez:thresholdValue", "@type": "xsd:decimal"},
    "centroidScore": {"@id": "mowm:centroidScore", "@type": "xsd:decimal"},
    "margin": {"@id": "mowm:margin", "@type": "xsd:decimal"},
    "residualNorm": {"@id": "mowm:residualNorm", "@type": "xsd:decimal"}
  }
}
```

Every example below assumes this context. Every `@graph` is host-written. `cheez:placeholders: true` means identifiers, measurements, dates and texts not supplied by the brief are placeholders, not observations.

## 3. Six example records

### 3.1 Certified answer turn and certificate

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-turn-001",
      "type": "cubby:Turn",
      "cheez:placeholders": true,
      "cubby:conversation": {"id": "urn:uuid:PLACEHOLDER-conversation"},
      "turnIndex": 41,
      "cubby:dialogueAct": {"id": "cubby:act-certified-answer"},
      "cubby:question": {"id": "urn:uuid:PLACEHOLDER-question"},
      "cubby:answer": {"id": "urn:uuid:PLACEHOLDER-answer"},
      "cubby:program": {"id": "urn:uuid:PLACEHOLDER-program-host"},
      "prov:wasAssociatedWith": {"id": "urn:uuid:PLACEHOLDER-host-agent"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-answer",
      "type": "prov:Entity",
      "value": "PLACEHOLDER: rendered answer",
      "cubby:proposer": {"id": "cubby:proposer-host-grammar"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-question",
      "type": "do:Question",
      "value": "PLACEHOLDER: user question"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-certificate",
      "type": "cubby:Certificate",
      "cheez:placeholders": true,
      "cubby:certifies": {"id": "urn:uuid:PLACEHOLDER-answer"},
      "cubby:decisionQuestion": {"id": "urn:uuid:PLACEHOLDER-question"},
      "cubby:option": [
        {"id": "urn:uuid:PLACEHOLDER-option-host"},
        {"id": "urn:uuid:PLACEHOLDER-option-episodic"}
      ],
      "cubby:criterion": [
        {"id": "urn:uuid:PLACEHOLDER-check-types"},
        {"id": "urn:uuid:PLACEHOLDER-check-dates"}
      ],
      "cubby:programRun": {"id": "urn:uuid:PLACEHOLDER-program-run"},
      "cubby:statementRead": [{"id": "urn:uuid:PLACEHOLDER-statement-read"}],
      "cubby:ruleApplied": [{"id": "urn:uuid:PLACEHOLDER-rule"}],
      "prov:wasGeneratedBy": {"id": "urn:uuid:PLACEHOLDER-vm-certification-run"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-option-host",
      "type": "do:Option",
      "cubby:program": {"id": "urn:uuid:PLACEHOLDER-program-host"},
      "cubby:proposer": {"id": "cubby:proposer-host-grammar"},
      "do:is_satisfied_by": [
        {"id": "urn:uuid:PLACEHOLDER-check-types"},
        {"id": "urn:uuid:PLACEHOLDER-check-dates"}
      ]
    },
    {
      "id": "urn:uuid:PLACEHOLDER-option-episodic",
      "type": "do:Option",
      "cubby:program": {"id": "urn:uuid:PLACEHOLDER-program-episodic"},
      "cubby:proposer": {"id": "cubby:proposer-episodic-memory"},
      "do:is_violated_by": {"id": "urn:uuid:PLACEHOLDER-check-types"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-check-types",
      "type": "do:Criterion",
      "dcterms:type": {"id": "cubby:check-exact-types"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-check-dates",
      "type": "do:Criterion",
      "dcterms:type": {"id": "cubby:check-exact-dates"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-program-run",
      "type": ["prov:Activity", "cube:ProgramRun"],
      "cube:executedProgram": {"id": "urn:uuid:PLACEHOLDER-program-host"},
      "cube:runOutcome": {"id": "cube:outcome-return"},
      "startedAt": "2026-11-07T10:00:00Z",
      "endedAt": "2026-11-07T10:00:01Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-statement-read",
      "type": "prov:Entity",
      "prov:wasDerivedFrom": {"id": "urn:uuid:PLACEHOLDER-source"},
      "cheez:sourceRevision": "PLACEHOLDER: source-revision-17"
    }
  ]
}
```

The certificate states only that the answer follows from these inputs and rules under these checks. It carries no truth score.

### 3.2 Refusal and reason

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-turn-refusal",
      "type": "cubby:Turn",
      "cheez:placeholders": true,
      "turnIndex": 42,
      "cubby:dialogueAct": {"id": "cubby:act-refusal"},
      "cubby:question": {"id": "urn:uuid:PLACEHOLDER-question-refusal"},
      "prov:generated": {"id": "urn:uuid:PLACEHOLDER-refusal"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-refusal",
      "type": "cubby:Refusal",
      "cubby:decisionQuestion": {"id": "urn:uuid:PLACEHOLDER-question-refusal"},
      "cubby:refusalReason": {"id": "cubby:refusal-fact-missing"},
      "cubby:option": {"id": "urn:uuid:PLACEHOLDER-refusal-option"},
      "cubby:criterion": {"id": "urn:uuid:PLACEHOLDER-required-fact-check"},
      "prov:wasGeneratedBy": {"id": "urn:uuid:PLACEHOLDER-decision-run"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-refusal-option",
      "type": "do:Option",
      "cubby:program": {"id": "urn:uuid:PLACEHOLDER-program-everyday-table"},
      "cubby:proposer": {"id": "cubby:proposer-everyday-table"},
      "do:is_violated_by": {"id": "urn:uuid:PLACEHOLDER-required-fact-check"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-required-fact-check",
      "type": "do:Criterion",
      "dcterms:type": {"id": "cubby:check-required-fact-present"}
    }
  ]
}
```

A readable refusal can be templated from `cubby:refusal-fact-missing`; the record contains no model-written rationale.

### 3.3 MoWM routing decision ending in spawn

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-routing-1",
      "type": "mowm:RoutingDecision",
      "cheez:placeholders": true,
      "mowm:inputEncoding": {"id": "urn:uuid:PLACEHOLDER-fastword-encoding"},
      "mowm:bestWorld": {"id": "urn:uuid:PLACEHOLDER-world-1000"},
      "mowm:runnerUpWorld": {"id": "urn:uuid:PLACEHOLDER-world-1001"},
      "centroidScore": 0.71,
      "margin": 0.03,
      "mowm:absoluteFloor": {"id": "urn:uuid:PLACEHOLDER-floor"},
      "mowm:marginThreshold": {"id": "urn:uuid:PLACEHOLDER-margin-threshold"},
      "residualNorm": 0.63,
      "mowm:routeOutcome": {"id": "mowm:outcome-spawn"},
      "prov:generated": {"id": "urn:uuid:PLACEHOLDER-spawn-1"},
      "startedAt": "2026-11-07T11:00:00Z",
      "endedAt": "2026-11-07T11:00:01Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-floor",
      "type": "cheez:CalibratedThreshold",
      "thresholdValue": 0.65,
      "cheez:thresholdParameter": {"id": "mowm:param-absolute-floor"},
      "cheez:calibratedBy": {"id": "urn:uuid:PLACEHOLDER-screen-route"},
      "cheez:calibrationData": {"id": "urn:uuid:PLACEHOLDER-calibration-corpus"},
      "calibratedOn": "2026-11-07"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-margin-threshold",
      "type": "cheez:CalibratedThreshold",
      "thresholdValue": 0.08,
      "cheez:thresholdParameter": {"id": "mowm:param-tau-margin"},
      "cheez:calibratedBy": {"id": "urn:uuid:PLACEHOLDER-screen-route"},
      "cheez:calibrationData": {"id": "urn:uuid:PLACEHOLDER-calibration-corpus"},
      "calibratedOn": "2026-11-07"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-spawn-1",
      "type": "mowm:Spawn",
      "mowm:spawnedWorld": {"id": "urn:uuid:PLACEHOLDER-world-5000"},
      "residualNorm": 0.63,
      "mowm:spawnThreshold": {"id": "urn:uuid:PLACEHOLDER-spawn-tau"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-world-5000",
      "type": "mowm:World",
      "worldId": 5000,
      "z": 0,
      "zMax": 3,
      "mowm:mergeStrategy": {"id": "mowm:merge-bind-compose"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-spawn-tau",
      "type": "cheez:CalibratedThreshold",
      "thresholdValue": 0.5,
      "cheez:thresholdParameter": {"id": "mowm:param-tau-spawn"},
      "cheez:calibratedBy": {"id": "urn:uuid:PLACEHOLDER-screen-spawn"},
      "cheez:calibrationData": {"id": "urn:uuid:PLACEHOLDER-spawn-corpus"},
      "calibratedOn": "2026-11-07"
    }
  ]
}
```

The floor clears, but the margin does not. The host therefore spawns; it does not manufacture a route or infer the child id from the parent id.

### 3.4 World split at z+1

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-world-split",
      "type": "mowm:Spawn",
      "cheez:placeholders": true,
      "mowm:spawnedWorld": {"id": "urn:uuid:PLACEHOLDER-world-child"},
      "residualNorm": 0.57,
      "mowm:spawnThreshold": {"id": "urn:uuid:PLACEHOLDER-world-split-tau"},
      "prov:used": {"id": "urn:uuid:PLACEHOLDER-world-parent"},
      "endedAt": "2026-11-07T11:05:00Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-world-parent",
      "type": "mowm:World",
      "worldId": 1000,
      "z": 0,
      "zMax": 3,
      "mowm:hyla": {"id": "urn:uuid:PLACEHOLDER-hyla-parent"},
      "mowm:mergeStrategy": {"id": "mowm:merge-bind-compose"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-world-child",
      "type": "mowm:World",
      "worldId": 1001,
      "z": 1,
      "zMax": 3,
      "mowm:parentWorld": {"id": "urn:uuid:PLACEHOLDER-world-parent"},
      "mowm:hyla": {"id": "urn:uuid:PLACEHOLDER-hyla-child"},
      "mowm:mergeStrategy": {"id": "mowm:merge-weighted-bundle"},
      "cheez:contentAddress": {"id": "urn:uuid:PLACEHOLDER-child-weights-address"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-child-weights-address",
      "type": "cheez:ContentAddress",
      "cheez:hashAlgorithm": {"id": "cheez:hash-sha256"},
      "cheez:hashValue": "PLACEHOLDER: lowercase hexadecimal digest",
      "cheez:trainingRun": {"id": "urn:uuid:PLACEHOLDER-training-run"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-world-split-tau",
      "type": "cheez:CalibratedThreshold",
      "thresholdValue": 0.5,
      "cheez:thresholdParameter": {"id": "mowm:param-tau-spawn"},
      "cheez:calibratedBy": {"id": "urn:uuid:PLACEHOLDER-screen-world-split"},
      "cheez:calibrationData": {"id": "urn:uuid:PLACEHOLDER-world-split-data"},
      "calibratedOn": "2026-11-07"
    }
  ]
}
```

The parent remains. The child handles residuals at z=1; `spawn_subworld` would refuse if z already equalled `zMax`.

### 3.5 Helper rejected in its sandbox

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-helper",
      "type": ["cubby:Helper", "cube:Program"],
      "cheez:placeholders": true,
      "cubby:proposer": {"id": "cubby:proposer-overnight-rule-library"},
      "cheez:contentAddress": {"id": "urn:uuid:PLACEHOLDER-helper-source-hash"},
      "cubby:brokenBy": {"id": "urn:uuid:PLACEHOLDER-verified-case-9"},
      "dcterms:type": {"id": "cubby:helper-rejected"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-sandbox-run",
      "type": "cubby:SandboxRun",
      "cubby:testedHelper": {"id": "urn:uuid:PLACEHOLDER-helper"},
      "escapeAttempts": 1,
      "cubby:rejectionRate": 0.2,
      "prov:used": [
        {"id": "urn:uuid:PLACEHOLDER-verified-case-9"},
        {"id": "urn:uuid:PLACEHOLDER-proposer-window"}
      ],
      "startedAt": "2026-11-07T12:00:00Z",
      "endedAt": "2026-11-07T12:00:04Z",
      "prov:wasAssociatedWith": {"id": "urn:uuid:PLACEHOLDER-locked-down-vm"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-helper-source-hash",
      "type": "cheez:ContentAddress",
      "cheez:hashAlgorithm": {"id": "cheez:hash-sha256"},
      "cheez:hashValue": "PLACEHOLDER: source digest"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-verified-case-9",
      "type": "prov:Entity",
      "cubby:answer": {"id": "urn:uuid:PLACEHOLDER-existing-certified-answer"},
      "dcterms:type": {"id": "cubby:case-helper-output-differs"}
    }
  ]
}
```

This is an observed sandbox result, not a declared lifecycle status. The rejected helper and breaking case remain stored; no promotion activity is asserted.

### 3.6 Invoice field reviewed and committed

```json
{
  "@graph": [
    {
      "id": "urn:uuid:PLACEHOLDER-region-total",
      "type": "cubby:Region",
      "cheez:placeholders": true,
      "cubby:regionText": "PLACEHOLDER: Invoice total USD 120.00",
      "cubby:readerVersion": "PLACEHOLDER: reader-3"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-field-total",
      "type": "cubby:Field",
      "cubby:region": {"id": "urn:uuid:PLACEHOLDER-region-total"},
      "price": 120.00,
      "schema:priceCurrency": "USD",
      "cubby:suggestedOnly": false,
      "cubby:exactCheck": [
        {"id": "urn:uuid:PLACEHOLDER-number-in-region"},
        {"id": "urn:uuid:PLACEHOLDER-arithmetic-check"},
        {"id": "urn:uuid:PLACEHOLDER-supplier-check"},
        {"id": "urn:uuid:PLACEHOLDER-duplicate-check"}
      ]
    },
    {
      "id": "urn:uuid:PLACEHOLDER-number-in-region",
      "type": "prov:Activity",
      "dcterms:type": {"id": "cubby:check-number-in-region"},
      "cubby:verdict": {"id": "cubby:check-passed"},
      "endedAt": "2026-11-07T13:00:00Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-review-1",
      "type": "cubby:Review",
      "cubby:reviewedField": {"id": "urn:uuid:PLACEHOLDER-field-total"},
      "cubby:reviewOutcome": {"id": "cubby:review-accept"},
      "cubby:region": {"id": "urn:uuid:PLACEHOLDER-region-total"},
      "prov:wasAssociatedWith": {"id": "urn:uuid:PLACEHOLDER-reviewer-1"},
      "prov:qualifiedAssociation": {
        "type": "prov:Association",
        "prov:agent": {"id": "urn:uuid:PLACEHOLDER-reviewer-1"},
        "prov:hadRole": {"id": "cubby:role-invoice-reviewer"}
      },
      "endedAt": "2026-11-07T13:01:00Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-review-2",
      "type": "cubby:Review",
      "cubby:reviewedField": {"id": "urn:uuid:PLACEHOLDER-field-total"},
      "cubby:reviewOutcome": {"id": "cubby:review-accept"},
      "cubby:region": {"id": "urn:uuid:PLACEHOLDER-region-total"},
      "prov:wasAssociatedWith": {"id": "urn:uuid:PLACEHOLDER-reviewer-2"},
      "prov:qualifiedAssociation": {
        "type": "prov:Association",
        "prov:agent": {"id": "urn:uuid:PLACEHOLDER-reviewer-2"},
        "prov:hadRole": {"id": "cubby:role-second-approver"}
      },
      "endedAt": "2026-11-07T13:02:00Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-commit",
      "type": "cubby:Commit",
      "cubby:committedField": {"id": "urn:uuid:PLACEHOLDER-field-total"},
      "prov:used": [
        {"id": "urn:uuid:PLACEHOLDER-review-1"},
        {"id": "urn:uuid:PLACEHOLDER-review-2"},
        {"id": "urn:uuid:PLACEHOLDER-number-in-region"}
      ],
      "prov:generated": {"id": "urn:uuid:PLACEHOLDER-fact-version"},
      "endedAt": "2026-11-07T13:02:01Z"
    },
    {
      "id": "urn:uuid:PLACEHOLDER-fact-version",
      "type": "cheez:FactVersion",
      "cheez:statement": {
        "type": "rdf:Statement",
        "rdf:subject": {"id": "urn:uuid:PLACEHOLDER-invoice"},
        "rdf:predicate": {"id": "schema:totalPrice"},
        "rdf:object": {"value": "120.00", "type": "xsd:decimal"}
      },
      "cheez:validTime": {"id": "urn:uuid:PLACEHOLDER-valid-interval"},
      "recordedAt": "2026-11-07T13:02:01Z",
      "prov:wasDerivedFrom": {"id": "urn:uuid:PLACEHOLDER-region-total"}
    },
    {
      "id": "urn:uuid:PLACEHOLDER-valid-interval",
      "type": "time:Interval",
      "time:hasBeginning": {
        "type": "time:Instant",
        "inXSDDateTimeStamp": "2026-11-07T00:00:00Z"
      }
    }
  ]
}
```

Two reviews are shown because this example is intended to be above the second-approval amount; that amount is not stated and must not be inferred from the example.

## 4. SHACL shapes

The three most important constraints are certificate provenance, calibrated routing and bitemporal facts. These are minimum shapes, not a claim that they validate every operational rule.

```turtle
@prefix sh:     <http://www.w3.org/ns/shacl#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix prov:   <http://www.w3.org/ns/prov#> .
@prefix cheez:  <https://w3id.org/cheezml/core#> .
@prefix cubby:  <https://w3id.org/cheezml/cubby#> .
@prefix mowm:   <https://w3id.org/cheezml/mowm#> .
@prefix do:     <https://w3id.org/decision-ontology/> .

cheez:FactVersionShape
    a sh:NodeShape ;
    sh:targetClass cheez:FactVersion ;
    sh:property [
        sh:path cheez:statement ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class rdf:Statement
    ] ;
    sh:property [
        sh:path cheez:validTime ;
        sh:minCount 1 ; sh:maxCount 1
    ] ;
    sh:property [
        sh:path cheez:recordedAt ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:datatype xsd:dateTime
    ] .

cubby:CertificateShape
    a sh:NodeShape ;
    sh:targetClass cubby:Certificate ;
    sh:property [
        sh:path cubby:certifies ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class prov:Entity
    ] ;
    sh:property [
        sh:path cubby:decisionQuestion ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class do:Question
    ] ;
    sh:property [
        sh:path cubby:option ;
        sh:minCount 1 ;
        sh:class do:Option
    ] ;
    sh:property [
        sh:path cubby:criterion ;
        sh:minCount 1 ;
        sh:class do:Criterion
    ] ;
    sh:property [
        sh:path cubby:programRun ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class prov:Activity
    ] ;
    sh:property [
        sh:path cubby:statementRead ;
        sh:minCount 1 ;
        sh:class prov:Entity
    ] ;
    sh:property [
        sh:path cubby:ruleApplied ;
        sh:minCount 1 ;
        sh:class prov:Entity
    ] .

mowm:RoutingDecisionShape
    a sh:NodeShape ;
    sh:targetClass mowm:RoutingDecision ;
    sh:property [
        sh:path mowm:inputEncoding ;
        sh:minCount 1 ; sh:maxCount 1
    ] ;
    sh:property [
        sh:path mowm:bestWorld ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class mowm:World
    ] ;
    sh:property [
        sh:path mowm:centroidScore ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:datatype xsd:decimal
    ] ;
    sh:property [
        sh:path mowm:absoluteFloor ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class cheez:CalibratedThreshold
    ] ;
    sh:property [
        sh:path mowm:routeOutcome ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class <http://www.w3.org/2004/02/skos/core#Concept>
    ] ;
    sh:property [
        sh:path mowm:marginThreshold ;
        sh:minCount 1 ; sh:maxCount 1 ;
        sh:class cheez:CalibratedThreshold
    ] .
```

Additional host validation should require source and revision on every `cubby:statementRead`, calibration data/date/activity on every threshold, a runner-up and margin whenever at least two worlds exist, and a generated `mowm:Spawn` when the outcome is spawn. Those conditions are clearer as host validation rules than as a long SHACL-SPARQL shape.

## 5. CubeLang programs

These use only storage, literal assignment, `if`/`else` and flat operands. The interfaces are declared. Host bindings supply the boolean results of the actual encoder, router and residual calculations; CubeLang is not represented here as performing vector arithmetic. Outcome `1` means route, `2` means request spawn, and `0` means no action. The host executes the external router/spawner and writes the records.

```cubelang
interface IMoWMRouter {
}

program MoWMRouter implements IMoWMRouter {
    storage {
        floor_ok = 0;
        two_or_more = 0;
        margin_ok = 0;
        outcome = 0;
    }

    functions {
        route() {
            if floor_ok {
                if two_or_more {
                    if margin_ok {
                        outcome = 1;
                    } else {
                        outcome = 2;
                    }
                } else {
                    outcome = 1;
                }
            } else {
                outcome = 2;
            }
        }
    }
}
```

The spawner is invoked only after the router returns `2`, or when the residual-subworld path independently requests a split. It does not spawn merely because confidence was low.

```cubelang
interface IMoWMSpawner {
}

program MoWMSpawner implements IMoWMSpawner {
    storage {
        residual_exceeds_tau_spawn = 0;
        at_z_max = 0;
        outcome = 0;
    }

    functions {
        spawn() {
            if at_z_max {
                outcome = 0;
            } else {
                if residual_exceeds_tau_spawn {
                    outcome = 1;
                } else {
                    outcome = 0;
                }
            }
        }
    }
}
```

For spawning, `1` authorises the host to allocate a world from the appropriate counter; it does not compute an id. I do not know the exact surface syntax for host storage binding or ASK, so neither is invented here. These programs still require `cubelang check` before adoption.

## 6. Cuts and open questions

### Cut

- A generic `status` property, agent-declared `PASSED`, `Active` or `Certified`, and any certification-by-agreement record.
- All prose rationale, explanation and model-written justification fields.
- Embedded vectors, weights, source blobs and bytecode. Retain hashes, training runs and external artifact references.
- A universal uncalibrated `threshold` number. Retain measurements separately from calibrated thresholds.
- Claims that schema.org already contains CheezML’s operational terms.
- Duplicate JSON properties for PROV-O, Decision Ontology, OWL-Time and SKOS relations.
- Opcode names or byte values in instance records. The fixed opcode table belongs to the compiler repositories, not this vocabulary.
- An XML serialization merely to match the name.
- Automatic physical deletion on pruning. Even if `mowm` ultimately deletes working tensors, CheezML must retain the world’s retirement record and reason.
- Automatic spawning on every low-confidence result.
- Any child-id arithmetic.
- A sentence encoder on MoWM’s routing path.
- “Correct”, “verified fact” and truth-score properties on certificates.

### Open questions

1. What is the authoritative Decision Ontology namespace and its exact property signature for questions, options, decisions and criteria?
2. What namespace document and `owl:versionIRI` policy will `w3id.org/cheezml/` publish, and when will redirection to `ns.grillcheeseai.com` occur?
3. Which hash encodings and canonicalization rules are required for JSON, source, weights, vectors and bytecode?
4. How are the global world-id counter from 1000 and router counter from 5000 persisted, coordinated and protected against collision?
5. Does pruning delete working memory while retaining a CheezML tombstone, or retain the complete world artifact?
6. What are the exact `run-proto` representation and ASK surface syntax?
7. How are threshold calibration screens, datasets and dates identified when an older deployment record lacks them?
8. Which exact ISO 24617-2 mapping is wanted for small talk and follow-up, whose CubeLLM acts are more specific than the reference set?
9. What stable links identify EmotionML dimension sets and Cubby’s separate hormone-dimension set?
10. How are certificate criteria mapped to the VM’s checks, and what is the minimal certificate for a derivation reading no statements?
11. What second-approval amount applies to a document deployment, and how is its currency and calibration/policy provenance recorded?
12. Which fields define a helper’s adoption scope and its “nothing outside it changed” comparison?
13. What artifact identifiers distinguish HYLA transition functions, merge strategies, cleanup strategies and Oja consolidation runs?
14. Which parts of the stated vocabulary are required for the first release? The story, domain-publication, nanopublication and RO-Crate records need separate examples and shapes before they can be considered fully specified.
