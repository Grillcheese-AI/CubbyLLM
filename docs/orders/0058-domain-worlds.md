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
