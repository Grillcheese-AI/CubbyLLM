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

## Cowork's addendum (2026-11-11): three finance registers already built as worlds
Nick: "yes to all 4". Source: Tesseract Semantics' open research of August 2026. Three ontologies, each with its
SHACL shapes, its pipeline and its queries, built by script from public registers. Stated licence: CC BY 4.0 for
the ontology and documentation, MIT for the code. They ship no question answering: material for Cubby, not a
competitor. The figures below are their pages', read 2026-11-11.

| world | built from | their size | repository |
|---|---|---|---|
| US investment funds | the SEC's series and class dataset, Form N-CEN, GLEIF's ISIN-to-LEI file | 1.29 million triples; 14,841 funds, 43,344 share classes | `fabio-rovai/investment-fund-ontology` |
| US banks | the FDIC's BankFind register (27,836 institutions), GLEIF's full golden copy (3,403,760 LEI records), the Federal Reserve's MDRM dictionary | 1,123,634 triples | `fabio-rovai/bank-register-ontology` |
| EU insurers | EIOPA's register of insurance undertakings (33,924 rows), GLEIF, BaFin's German register | about 276,683 triples; 4,842 undertakings | `fabio-rovai/insurance-register-ontology` |

**What it brings, in plain words:** three finance worlds Cubby can load instead of building: "which funds does
this registrant run?", "is this bank's identifier still valid?", "where may this insurer sell?". Each answer
comes from the register's record, with its date.
- **Over a conventional model:** a model has no register behind its answer, and cannot say that an identifier
  lapsed last month.
- **For you:** a finance demo on real public records, sooner than building the worlds from FIBO and GLEIF alone.

### What is done with them
0. **Before any download:** each repository's licence file is read and recorded, and each register's own terms
   (the SEC, the FDIC, the Federal Reserve, EIOPA, BaFin, GLEIF). Their pages do not say which licence covers
   the shapes; that is settled here. Sizes are checked first: a download over 1 GB waits for Nick's yes (GLEIF's
   golden copy is the likely one).
1. **Each is a domain manifest of this order,** loaded by the same loader: the ontology checked on its own
   first, the records loaded with their source and date, the shapes run by the pinned checker, as the addenda
   above say. The funds world's FIBO alignment is read as they grade it. Where they say FIBO has no counterpart,
   none is invented.
2. **Their pipelines are read before anything of theirs runs.** Whether Cubby runs them, or loads the same
   registers with this order's own loader, is Claude Code's proposal in the report, before it is done. Any part
   of a world that was not built by rule from a register is left out.
3. **Their defect counts are the first test of the load.** Each page reports what its shapes found in the real
   records: in the bank world, 1,733 truncated identifiers held by active institutions and 283 lapsed
   registrations; in the insurer world, 4 identifiers that fail their check digits and 118 lapsed ones. Loaded
   from the same register date, Cubby's counts equal theirs. From a later date, the difference is reported per
   shape, and nothing is repaired.
4. **Their stop is taken:** each headline count is computed twice, once from the source rows and once by query
   on the loaded world, and the load's report is not written when the two disagree. It hardens this order's
   "loaded counts equal the source's counts".
5. **Questions:** by rule from held-out records, in EN and FR, 100 read by hand, as step 4 of the order. Each
   world adds its defect questions: "is this identifier valid?" is answered from the check, naming the rule that
   failed.
6. **Order of work:** the bank world first (it carries the full GLEIF copy that step 2 needs anyway), then
   funds, then insurers. Step 2's ownership chains (GLEIF's "who owns whom" records) stay as written and run on
   the same load.

### Acceptance (added to the order's)
- **Per world:** the licence record; the loaded counts equal the source's, both ways (item 4); the defect counts
  against theirs (item 3); right, wrong and refused on its questions. **Line:** 0 wrong.
- **Kill:** a licence that does not allow the use stops that world before any download.

Credit: Tesseract Semantics is named as the source of each ontology in its manifest and in the paper's related
work, as CC BY asks.

## Cowork's addendum (2026-11-11, second): five more from the field watch for the domain worlds, approved (Nick, 2026-11-11: "proposals: yes")
Nick: "there's a lot more we can borrow in it". **Approved by Nick on 2026-11-11 ("proposals: yes"); built when this order is taken, in its queue place.** **Was: proposed, for Nick's yes; nothing here is built or measured.**
The first three were read today; the last two are as the field watch read them.

**What it brings, in plain words:**
- A finance test with the answers already written, to grade Cubby on day one.
- Rules of law or policy as small checked programs that say "holds", "does not hold" or "cannot tell".
- A verdict that quotes the rule as it was worded on the day in question.
- **Over a conventional model:** a model paraphrases a rule from memory, in today's wording, with rounded
  numbers, and never says "cannot tell".
- **For you:** Cubby can check a record against a dated rule and show the exact sentence it applied.

### 1. A finance fixture with expected answers
Fluree's Financial Services Demo Kit: FIBO's business-entity ontologies with a credit extension, 75 real GLEIF
LEI records (public domain) among 171 entities, and 18 competency questions, each with its expected answer. A
21 KB zip; its own licence is in its README, to read first.
- **Use:** the finance world's first test. **Line:** 18 of 18 right or refused with a reason; 0 wrong.

### 2. Rules as typed programs with exact numbers and an explicit "cannot tell"
From MudraForge Decompiler (Apache 2.0): statutes become a typed tree; a VM evaluates it in exact rational
arithmetic and abstains on an ambiguous or contradictory provision. Its scores are its authors' own.
- **For this order's rule mapping:** a domain rule is a CubeLang program with exact arithmetic and three
  outcomes: holds, does not hold, cannot tell. No float decides a threshold.
- **The model that writes their tree is not taken:** here a rule is written by hand or mapped from the ontology.

### 3. Records checked against a dated law
From ccp-engine: 1,487,586 Portuguese public-procurement procedures checked against the public contracts code.
No model runs at run time. Each verdict cites the passage, its wording in force on that date, a literal quote
and the facts used; a refusal carries a typed reason; each verdict names the log head, the law's version and the
code's commit. Its licence file is to be read first.
- **Taken as patterns:** the wording in force on the record's date; refusal types (a fact missing, a rule
  missing, dates in conflict); the verdict naming the versions it was made with.
- **A candidate domain after finance and automotive:** public procurement, where the records are open.

### 4. A draft mapping from a company's own tables, approved term by term
From NVIDIA's Auto Ontology (a model drafts an enterprise ontology and people certify it). For the on-premises
worlds of this order.
- **Here:** the draft comes from profiling the tables by rule (keys, value ranges, joins that hold). A person
  approves each term, and the mapping records the approval, as the addendum of 2026-11-06 already asks.
- **Also taken:** a filter's value is checked against the data before a query runs.

### 5. An enterprise benchmark whose answers carry proofs
ENTLORE: 907 enterprise questions whose gold answers are computed by graph programs with proof certificates,
with a bank of questions that must be refused. Its code was not out when the field watch read it.
- **When released:** the benchmark for this order's enterprise demo.
