# 0058 — Domain worlds: an industry ontology plus open records, answered exactly (finance and automotive first)
status: open (queue: after 0055)
written: 2026-11-06 by Cowork · approved: 2026-11-06 by Nick ("the more domain cubby can serve the better", with FIBO and AUTO) · hypothesis: the system paper's regulated-work claim; H-A13's rule checks

## Why
Nick wants Cubby to serve as many domains as possible. An industry ontology supplies a domain's classes and rules,
and open records supply its facts.
- **FIBO** (EDM Council and OMG): finance, MIT licence, quarterly releases. It covers legal entities and ownership,
  securities, loans, corporate actions, indices and market data.
- **AUTO** (EDM Council): automotive, MIT licence, built on schema.org's automotive extension.

Both are vocabularies, not data. Cubby answers from records and uses the ontology only to state what the records
mean and which rules follow.

**One pattern, so that adding a domain is a data job, not a training job.** A domain is:
- an ontology module;
- one or more open fact sources, with their licences;
- the module's rules, mapped to the rule engine from 0055's addendum (OWL 2 RL);
- a question set to test it.

It is built with 0055's machinery: the same store, name index and rule engine.

**What it brings, in plain words:** Cubby can answer "who really owns this company?" or "what was recalled on this
car?" from official records, step by step and sourced, and says "not on record" instead of guessing.
- **Over a conventional model:** a model answers from whatever it absorbed, often outdated or invented. Cubby follows
  the official records, shows the chain, and gains a domain without retraining.

## Build
0. **Licences**, checked and recorded for every ontology and every data source before anything is downloaded.
   Downloads over 1 GB wait for Nick's yes.
1. **The pattern:** a domain manifest naming the module, the sources, the rule mapping and the question set. A loader
   turns records into the store's statements, each with its source and date.
2. **Finance, first:**
   - **The module:** FIBO's legal entities (BE), ownership and control only, not all of FIBO.
   - **The records:** GLEIF's legal-entity records, public domain and updated daily. That is millions of entities by
     their LEI code, and the "who owns whom" records (direct and ultimate parent).
   - **The link:** Wikidata's LEI property (P1278) connects them to the main store.
   - **The rules:** ownership chains, and "ultimate parent" as a chain of recorded direct parents, checked against
     GLEIF's own ultimate-parent record. A disagreement between the two is reported, never resolved by guess.
3. **Automotive, second:**
   - **The module:** AUTO's vehicle classes.
   - **The records,** each licence checked first:
     - NHTSA's vehicle catalogue (vPIC: makes, models, VIN decoding);
     - NHTSA's recalls;
     - the EPA's fuel economy data.
   - **The link:** Wikidata's car models and manufacturers.
4. **The questions, per domain:** written by rule from held-out records (as in 0057), in EN and FR. A seeded sample of
   100 is read by hand.

## Acceptance (pre-registered)
- **Tests, per domain:**
  - the loaded counts equal the source's counts;
  - 1,000 random records round-trip exactly;
  - every derived fact carries its derivation;
  - two companies with one name are never separated by fame: the LEI decides, or Cubby asks.
- **Reported, per domain:**
  - right, wrong and refused on the question set, with refusals by reason;
  - the store's size;
  - how many rule violations the records contain.
- **Line:** 0 wrong on each domain's question set.
- **Kill:** any wrong answer that comes from a derived fact means the rule mapping is wrong for that domain. That
  domain's derivations go off until it is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model builds stored knowledge. The ontologies and the records are the only sources.
- **Facts keep their source and date.** A registry's record answers "as of" its date. For time-sensitive facts
  (ownership, recalls), the live source wins when there is one, as with Wikidata.
- **FIBO is a trademark of EDM Council:** named as the source, never as a product claim.
- No local path in a committed file.

## Out of scope
- **The rest of FIBO** (securities, derivatives, loans, market data) and the rest of AUTO: later domains, from these
  numbers.
- **Prices, market data and anything that changes by the minute:** not facts for a store.
- **Advice of any kind** (financial, legal, mechanical): Cubby answers what the records say.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## Cowork's addendum (2026-11-06): checks before and while loading a domain
Nick: "yes". The ideas come from Utopia and Semantica (both open source); only the ideas are taken.

**What it brings, in plain words:**
- Cubby checks an industry's rulebook for contradictions before trusting anything built from it.
- It shows which parts of the data it does not understand yet.
- When records disagree, it says how they disagree instead of quietly picking one.
- **Over a conventional model:** a model blends conflicting sources without saying so, and cannot tell you what it
  does not understand.

### 1. The ontology checked on its own first
- Before any record is loaded, the module (FIBO's legal entities, AUTO's vehicle classes) goes through the rule engine
  alone.
- Any inconsistency stops that domain until it is fixed, and is reported. For example, a class placed both under and
  disjoint from another.

### 2. Terms the module does not cover
- Every property or class in the records that the module does not cover is counted and listed, with its record count.
- None is mapped by guess. Each joins the mapping only with Nick's yes, and the mapping file records the approval.

### 3. Record checks written as SHACL shapes
- Each domain's record rules are written in the manifest as SHACL shapes, the W3C's standard format for data checks.
  Domain people can read and edit them. For example:
  - one LEI per company;
  - a recorded parent is itself a registered entity;
  - a recall names a make and a model year.
- **The candidate checker:** rudof. It is Rust, MIT or Apache-2.0, and usable as a library. It also checks ShEx, the
  format of Wikidata's entity schemas.
  - Its project marks versions 0.2.19 to 0.3.0 as unreliable, so the version is pinned and recorded.
  - Its licence is recorded.
- Violations are counted (the existing "rule violations" line) and never repaired by the build.

### 4. Kinds of conflict
A disagreement is reported with its kind:
- **value:** two values where one is allowed;
- **type:** the item is of a class the property does not allow;
- **temporal:** two periods that cannot both hold;
- **logical:** a rule violated;
- **recorded against derived:** for example, GLEIF's ultimate-parent record against the chain of direct parents.

Cubby does not answer a question that rests on a conflict. It says the records disagree, and how.
- Only the records' own signals decide (Wikidata's ranks, a record's validity dates), as before.
- Voting, source credibility and "the most recent source wins" are not used.

## Cowork's addendum (2026-11-07): test files, an enterprise demo, versions that earn their place, more industries
Nick: "yes". Sources: Microsoft's Ontology-Playground (MIT), EvoOntology (Renmin University, MIT, arXiv 2609.15779),
and the Industrial Ontologies Foundry (IOF, MIT). Only ideas and ontology files are taken; no code, and no model.

**What it brings, in plain words:**
- Every new industry plugs in the same way, already tested on dozens of examples.
- A company can describe its data the way it already does, and Cubby answers questions about it with a proof.
- A change to an industry's rules is kept only if it helps, so an update can't quietly make Cubby worse.
- **Over a conventional model:** business agents built on these schemas use an LLM to write the answer, so a wrong
  answer looks like a right one. Cubby keeps the schema and adds the certificate.

### 1. Test files for the loader
- **The source:** the Ontology-Playground catalogue, official and external entries only. Each of those states its
  licence; most community entries state none, so they are left out.
- **What's in it:** about 75 schemas. Not counting Brick, that is 651 classes, 890 relations and 3,050 data fields,
  with no instances, so they are fixtures, not knowledge.
- **The tests:**
  - every fixture loads;
  - every fixture passes the "ontology checked on its own first" step, or its violations are listed;
  - its class and property counts match the file.

### 2. An enterprise demo
- **The schema:** one official model, the complete Retail Supply Chain (step 6).
- **The records:** generated by rule from a fixed seed, with no model, and labelled as test data everywhere they
  appear.
- **The questions:** written by rule, as in 0057, including counts, filters and joins ("which orders from supplier X
  were late?").
- **Reported:** right, wrong and refused, with the line 0 wrong.
- **Purpose:** this is the demo for companies, not a benchmark result, and the paper says so.

### 3. Versions that earn their place
Any change to a domain's mapping, rules or SHACL shapes is a candidate version.
- **The comparison:** the candidate is run against its parent on the frozen domain question set, with the same store
  snapshot and the same programs.
- **The rule:** it is published only if it adds no wrong answer and gains right answers (or loses refusals).
  Otherwise the parent stays.
- **Recorded per version:** its hash and its before-and-after rates. EvoOntology's gate asks for a gain only;
  Cubby's also asks for 0 wrong.

### 4. More industries, after finance and automotive
None of these has open records, so each is an on-premises world, fed by a company's own records or by the demo
pattern of item 2.
- **IOF** (manufacturing; built on BFO). The released modules only:
  - Core, Maintenance, Supply Chain, Production Planning and Certification;
  - the released biopharmaceutical modules.

  Provisional modules wait until they are released.
- **Brick 1.4.4** (buildings and sensors; BSD-3), fed by a building's own sensor and equipment records.
- **OpenIM** (investments, aligned with FIBO; MIT), next to the finance world.

Licences are checked and recorded per module before anything is loaded. Utopia ships IOF Core as one of its packs,
and is cited as related work.
