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

### Answers with no LLM on the answer path (closest in claim; from the field watch, re-read 2026-11-09)
- **LP-LM** (Wu and Liu, arXiv 2502.09212, February 2025): prior art to cite for "no LLM on the answer path":
  - a question is parsed with a Prolog definite clause grammar into a Prolog term, which runs on a knowledge base of
    sentences held as Prolog terms. No LLM is used; LLMs appear only as baselines that hallucinate;
  - **the difference:** no certificate, no refusal reasons, and its abstract gives no public benchmark numbers.
- **OntoKG-EQ** (Nasir et al., arXiv 2609.08869, September 2026): answers rendered deterministically from a
  SHACL-checked, provenance-aware graph of equity-market data, with an explanation tracing each result to its
  sources. **The difference:** a fixed set of competency questions, not questions in plain language, and no refusal
  path stated.
- **ferrum-app-intelligence** (GitHub, AGPL): the same shape in a narrow field. A plain question becomes a typed read
  of a business metric by rules, and the answer carries lineage or is refused with a reason. No benchmarks. The watch
  flagged it as the one threat so far; read it in full before citing.
- **Open Ontologies** (fabio-rovai/open-ontologies, MIT): reasoning certificates (`asserted.tsv`, `derivations.tsv`
  and a digest) re-checked by `oo-cert`, a Lean 4 checker that shares no code with the engine. It certifies
  derivations from given assertions; it does not answer questions. 0055 takes the format (its addendum of
  2026-11-09).

### Incremental indexing
- **CocoIndex** (cocoindex-io/cocoindex; Apache-2.0; Rust core):
  - keeps an index in sync, with lineage per row, and caches each step by its input and its code;
  - an indexing engine, not a QA system;
  - its knowledge-graph examples use LLM extraction and entity matching by embeddings.

### Question answering over Wikidata
- **GRASP and GRISP** (Walter and Bast):
  - in the WikiKGQA 2026 challenge, GRASP had the best result in both without-mentions tracks (English 85.0, Spanish
    83.0, macro QALD F1), and ASTER (IBM) was first on English with mentions (0.88). Say it that way, not "GRASP won";
  - in GRISP (arXiv 2604.21133), a small model writes a query skeleton and the entities are filled in by search. That
    is close to Cubby's slots (H-E15), so it is the baseline;
  - its published F1 on the query-log test goes beside Cubby's numbers, labelled as a different metric.
- **WikiKGQA 2026 / QAWiki** (v2026.3; Zenodo DOI 10.5281/zenodo.22661413; CC0):
  - a test of 150 questions in English and Spanish, with hand-crafted gold;
  - the record holds every team's submitted answers (GRASP/GRISP, P3, WikiProSP, ASTER, AMA-KBQA);
  - the entrants' papers are in CEUR Vol-4284. Each writes one SPARQL query per question with an LLM, and none
    abstains. ASTER's paper reports 6 and 7 wrong of 100 on dev; GRASP's reports a Claude Code baseline at 85.0.
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
- **The frontier models:** published numbers only for now (system cards, on AA-Omniscience v1's public split of 600).
  A run through OpenRouter comes later.
- **AA-Omniscience has two versions; always name which.** v1.1 (October 2026: 3,000 questions, public set of 300,
  Apache 2.0) removed the saturated questions, and Artificial Analysis says its scores are not comparable with v1's.
  - Cubby is scored on both public sets: v1 to sit beside the system cards, v1.1 beside the current leaderboard.
  - AA's own metrics are computed from Cubby's counts: the Omniscience Index, 100 × (correct − incorrect) ÷ total;
    and the hallucination rate, incorrect ÷ (incorrect + partial + not attempted).
- **Refusals scored for their cause** (HopRefusalBench, once its data is out): a refusal with the wrong reason counts
  against Cubby.
- **Frozen and recorded with every number:** the commit, the adapter hashes, the VM's commit, the store's snapshot and
  the hash of each build step.
- **A reproducibility package with v2:** 0054's run as a Workflow Run RO-Crate, and each hypothesis verdict
  (killed ones included) as a nanopublication. It certifies that each number follows from its inputs, code and
  settings, not that a hypothesis is true.
- **Agreement with a reference engine, as counts:** rows compared and mismatches, against QLever. PULSE-S reports its
  cross-checks the same way.
- **Worth adding (partly ordered):** a mutation test, as PULSE-S does. 0064 mutates the records (every changed value
  must fail validation). Seeding small faults into the VM and the rules is not ordered yet. Each fault must be caught,
  which shows that "0 wrong" does not come from tests that cannot fail.
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
- **"No LLM on the answer path" is not new on its own** (LP-LM, OntoKG-EQ). The claim is the combination: questions in
  plain language, an executed program, a certificate or a refusal with its reason, and 0 wrong on public benchmarks.
- **AA's "hallucination rate" is not Cubby's wrong rate.** AA divides by the questions not answered correctly; Cubby
  divides by the answers spoken. Name the metric in every sentence that gives a number.

## 6. Results still to come
- **Full-store coverage, and right, wrong and refused on every benchmark:** 0055, then 0054. *Pending.*
- **The WikiKGQA 2026 matched comparison:** 0054's addendum, after 0055's store. *Pending.*
- **The query-log understanding benchmark,** with GRISP's F1 beside it: 0054 and 0057. *Pending.*
- **Opposite pairs (polarity errors):** 0054. *Pending.*
- **Questions about two periods of time** (Allen's relations, with "dates too coarse" refusals): 0057. *Pending.*
- **Replaying past answers, and a store kept up to date matching a full rebuild:** 0055's addendum tests. *Pending.*
- **Domain worlds, finance then automotive,** 0 wrong per domain: 0058. *Pending.*
- **One certificate in PROV-O,** as a worked example: 0054's addendum. *Pending.*
- **AA-Omniscience v1 and v1.1, refusal causes and as-of questions** (HopRefusalBench once released; the AIT Wikidata
  set): 0054's addendum of 2026-11-09. *Pending.*
- **The rules applied, re-checked by a separate checker:** 0055's addendum of 2026-11-09. *Pending.*

## 7. Where and when to publish
- **v1 is published** (2026-11-06): Zenodo, doi:10.5281/zenodo.24020073, preprint, CC BY 4.0.
  - Source: `2026-11-06-certified-execution-preprint-v1.md` in this folder; the uploaded PDF's checksum matches the
    final build.
  - **The DOI for all versions is doi:10.5281/zenodo.24020072** (given by Nick, 2026-11-10). It always opens the
    latest version, so the website and the READMEs link to it. A citation of v1's exact text uses …24020073.
  - v2 goes up as "New version" on that record, so every version stays linked under one concept DOI.
  - The record's description field is empty on Zenodo. A short abstract there helps search and the NSCS listing.
  - The record does not list the NSCS community yet; accept or add it from the community's side.
  - **For v2, the training-records sentence** (Nick, 2026-11-10: talk templates may be model-written, facts never):
    v1 says no external model writes training records. v2 says: no external model writes a fact or a reasoning
    record; the talk adapter's wording templates were drafted by external models and picked by a person.
  - **Also for v2 (found 2026-11-10):** talk_v2_2 already works this way. Its fact sentences were written by a
    frontier model and kept only when the host's checks passed (`build_ground_sft.py`), and its chat and web
    families are model-written text under the same guard. So v1's sentence holds for the emitter and the
    reasoning records, and not for the talk adapter's wording. v2 says it for both adapters: the wording of the
    talk adapter's training sentences came from external models and was kept only when every name and number
    in it was in the facts handed over; no external model wrote a fact or a reasoning record.
  - **For v2:** v1 cites "the public split of AA-Omniscience (600)". v2 names it as v1's public split and adds v1.1's.

The original options, kept for reference:
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
