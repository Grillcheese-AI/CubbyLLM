# System paper: notes to include
Working notes for the system paper ("From probabilistic chain of thoughts to Certified Execution"). These are points to
cover, not draft text; the prose is Nick's own. Numbers marked *pending* wait on orders 0054 and 0055.

## 1. The claim
- **The store is not the contribution.** In 2026, several open-source systems converged on the same store design:
  facts with two timelines, ontology rules run forward, and provenance per fact (TrustGraph, Semantica, CocoIndex,
  Utopia; section 2).
- **The contribution is the answer side:**
  - each answer is a program run on recorded facts;
  - it comes with a certificate (the statements read, their source and revision, the rules applied), or it is refused
    with a reason;
  - no LLM writes a stored fact, a training record or an answer.
- **A citation is not a certificate.** A citation tells the reader where to look, so a cited LLM answer can still be
  wrong. A certificate lets the reader check the answer.
- **The measured claim:** 0 wrong among spoken answers, always reported with the refusal rate and a 95% interval.
  Never accuracy alone.
- **For a non-scientific reader:** a model answers from what it absorbed. Cubby answers from records, shows the chain,
  or says it does not know.

## 2. Related work
### GRL's own prior work (cite)
Both are on Zenodo, in the Neuro-Symbolic Computing Society community (zenodo.org/communities/nscs); take the DOIs from
the record pages.
- **MoWM: Mixture of World Models via Hypernetwork-Generated VSA Transitions** (Cloutier, March 2026):
  - the world-model component;
  - a dated record of GRL's own sense of "world model", to cite when telling it apart from Utopia's "enterprise world
    model".
- **Learnable Reasoning Rules for Neuro-Vector-Symbolic Architectures via Hidden Markov Models** (HMM-VSA; Cloutier,
  March 2026, v2): learnable reasoning rules on hypervectors.

### Knowledge-graph platforms (closest in surface)
- **TrustGraph** (trustgraph-ai/trustgraph):
  - the graph is extracted by an LLM, and answers are written by an LLM over graph context;
  - no certification and no refusal path.
- **Semantica** (semantica-agi/semantica; MIT; Python):
  - a knowledge-graph toolkit with optional LLM extraction;
  - deterministic reasoning (forward chaining, Rete, Datalog), facts on two timelines, PROV-O lineage;
  - it settles conflicts by voting, source credibility or "most recent";
  - it publishes no accuracy figures, and its docs say to validate its rule engine before compliance use.
- **Utopia** (deeplethe/utopia; Apache-2.0; Rust; v0.1): the closest store:
  - a graph with two timelines, where corrections close the old version;
  - ontology axioms compiled into forward-chained rules, with derived facts marked;
  - PROV-O built in;
  - but answers come from an LLM with citations, a model judges doubtful duplicates, there is no refusal path and
    there are no accuracy figures;
  - it calls itself an "enterprise world model", so that use of the term should be told apart from MoWM's.
- **PULSE / PULSE-S** (deeplethe/pulse-spatial; Apache-2.0; pre-alpha; Python and Lean 4), by the same group as
  Utopia: the closest in spirit:
  - "executable knowledge graphs", computed deterministically, with no LLM;
  - one spatial statement is read in four modes: asserted, observed, normative (must or must not) and hypothetical;
  - Lean 4 proofs of deterministic evaluation, observations not overwriting assertions, scenario isolation and atomic
    failure. Its authors note that no theorem yet covers the full Python runtime;
  - cross-checked with zero mismatches against reference engines (GEOS, Apache Jena, PostGIS), and tested by mutation
    (ten single-field mutants, each detected);
  - **the difference:** its page shows no question answering from natural language and no public QA benchmark. Cubby
    goes from a question in plain language to an executed program, and is scored on public QA sets.

### Incremental indexing
- **CocoIndex** (cocoindex-io/cocoindex; Apache-2.0; Rust core):
  - keeps an index in sync, with lineage per row, and caches each step by its input and its code;
  - an indexing engine, not a QA system;
  - its knowledge-graph examples use LLM extraction and entity matching by embeddings.

### Question answering over Wikidata
- **GRASP and GRISP** (Walter and Bast):
  - GRASP won the WikiKGQA 2026 challenge;
  - in GRISP (arXiv 2604.21133), a small model writes a query skeleton and the entities are filled in by search. That
    is close to Cubby's slots (H-E15), so it is the baseline;
  - its published F1 on the query-log test goes beside Cubby's numbers, labelled as a different metric.
- **WikiKGQA 2026 / QAWiki** (v2026.3; Zenodo DOI 10.5281/zenodo.22661413; CC0):
  - a test of 150 questions in English and Spanish, with hand-crafted gold;
  - the record holds every team's submitted answers (GRASP/GRISP, P3, WikiProSP, ASTER, AMA-KBQA).
- **The Wikidata query-log benchmark** (Walter and Bast, SIGIR 2026; arXiv 2602.14594):
  - real user queries;
  - its questions are written by an LLM, so it serves as evaluation only, never training.

### Storage and reference engines
- **HDT and k-HDTCat** (ESWC 2024): the full Wikidata (19.2B triples) compressed on a 16 GB desktop in about two days.
  This is the cold tier.
- **QLever** (ad-freiburg): the reference SPARQL engine that Cubby's store is cross-checked against.

### Domain ontologies (regulated work)
- **FIBO** (finance) and **AUTO** (automotive), both from the EDM Council, MIT licence, OWL.
- The facts come from open records: GLEIF's legal-entity records (public domain), NHTSA and the EPA (licences being
  checked).
- FIBO is the EDM Council's trademark: name it as the source, never as a product claim.

## 3. Design points worth stating
- **One store, not one specialist per topic.** All seven models on the design panel rejected routing by topic: "you
  cannot route to a topic before you know the entity".
- **Two clocks per fact:**
  - valid time (when it held) and recorded time (when the store learned it);
  - corrections close the old version, never overwrite it;
  - benchmarks are pinned to the dump, so any past answer can be replayed exactly.
- **Derived facts are retired by their inputs.** A changed record retires every conclusion built on it, and only
  those are recomputed.
- **Merged items are followed.** Wikidata's redirects resolve old ids, including ids in benchmark gold answers.
- **Refusals carry a reason**, and each reason points to a fix:
  - not in the store;
  - the records disagree (value, type, temporal, logical, or recorded against derived);
  - dates too coarse;
  - the question cannot be expressed in CubeLang.
- **Conflicts are reported, never settled by guess.** There is no voting, no source credibility and no "most recent
  wins"; only the records' own ranks and dates decide.
- **No fame ranking.** Two entities with one name are told apart by a key, or Cubby asks.
- **Similarity only leads to a key.** Nothing is merged or answered by similarity alone.
- **External models build evaluation data only:** never stored facts, never training records.
- **A domain is a data job, not a training job:** an ontology module, open records, rules and a question set (0058).
- **For a non-scientific reader:** when an official record changes, every conclusion built on it is updated, and
  Cubby never keeps answering from the old version.

## 4. How results are reported
- **Right, wrong and refused, per benchmark,** with refusals broken down by reason and a 95% interval on the wrong
  rate.
- **Where a field uses F1** (QALD, the query-log benchmark), F1 goes beside Cubby's numbers, labelled as a different
  metric and setting.
- **The matched comparison (WikiKGQA 2026):**
  - same questions, same Wikidata snapshot (0055 builds from the challenge's own dump), same scoring;
  - the entrants' own submitted answers are rescored as right, wrong or refused;
  - no API is spent.
- **The frontier models:** published numbers only for now (system cards; the AA-Omniscience public split of 600). A
  run through OpenRouter comes later.
- **Frozen and recorded with every number:** the commit, the adapter hashes, the VM's commit, the store's snapshot and
  the hash of each build step.
- **Agreement with a reference engine, as counts:** rows compared and mismatches, against QLever. PULSE-S reports its
  cross-checks the same way.
- **Worth adding (not yet ordered):** a mutation test, as PULSE-S does. Seed small faults into the VM and the rules;
  each must be caught. This shows that "0 wrong" does not come from tests that cannot fail.
- **Spanish** is reported as Spanish, not hidden. Cubby serves English and French today.

## 5. Caveats and claims to avoid
- **Certified is not the same as correct.** A certificate proves the answer follows from the recorded facts and rules;
  if a record is wrong, the answer is wrong with a valid certificate.
- **Report wrong answers by cause.** In the preflight, the 8 wrongs on the Wikidata-family sets were mostly of three
  kinds:
  - a second true relation, scored against a single gold answer;
  - gold that contradicts the graph;
  - one failure of induction.
- **CLUTRR (827 of 1,146 answered, 0 wrong)** came from the rule library, with no language model. Present it as a
  rule-engine result.
- **The null control on CLUTRR is not a no-VM baseline.** Do not present it as one.
- **Store size:** the synthesis estimates the hot store at tens of times (10–75×) smaller. "A hundred times" was an
  overstatement. Quote only 0055's measured size.
- **No compression beyond information content.** Superposition does not beat entropy.
- **Coverage "can be improved easily"** should rest on the refusals-by-reason table: the share of refusals that are
  "not in the store" is the share that more data fixes. *Pending 0055.*
- **Store design:** don't claim novelty for two timelines, ontology rules or provenance on their own. Section 2 has the
  prior work.

## 6. Results still to come
- **Full-store coverage, and right, wrong and refused on every benchmark:** 0055, then 0054. *Pending.*
- **The WikiKGQA 2026 matched comparison:** 0054's addendum, after 0055's store. *Pending.*
- **The query-log understanding benchmark,** with GRISP's F1 beside it: 0054 and 0057. *Pending.*
- **Opposite pairs (polarity errors):** 0054. *Pending.*
- **Questions about two periods of time** (Allen's relations, with "dates too coarse" refusals): 0057. *Pending.*
- **Replaying past answers, and a store kept up to date matching a full rebuild:** 0055's addendum tests. *Pending.*
- **Domain worlds, finance then automotive,** 0 wrong per domain: 0058. *Pending.*
- **One certificate in PROV-O,** as a worked example: 0054's addendum. *Pending.*

## 7. Where and when to publish (to decide)
- **Zenodo, in the NSCS community:**
  - a DOI the same day, and versions under one concept DOI (v1, v2…), as with the HMM-VSA record;
  - not peer reviewed.
- **arXiv:** a first submission in a category may need an endorsement; check before counting on dates.
- **A peer-reviewed venue** for the full results, later.
- **One possible order:**
  1. v1 on Zenodo now: the method, the pre-registered pass lines and protocol (the orders' acceptance lines), and the
     preflight results. This dates the claim, and publishing the pass lines before the results makes "0 wrong" harder
     to dismiss.
  2. v2 with the full-store numbers from 0055 and 0054.

## 8. Keep out of the paper
- Private projects and internal-only sources.
- Local paths, drive letters and account details.
- Any claim of GeoSPARQL, OWL or SHACL conformance that has not been tested.
