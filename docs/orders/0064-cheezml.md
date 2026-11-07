# 0064 - CheezML: one record vocabulary, written by the host from what ran
status: open (queue: after 0055; its record writers land with the orders that record: 0054, 0059, 0061, 0062)
written: 2026-11-07 by Cowork — approved: 2026-11-07 by Nick ("yes", to having the panel design one vocabulary from everything we have; the name CheezML; a subdomain namespace; cheese release names "in alphabetical order") — hypothesis: none (infrastructure); every acceptance line in the orders above is read off these records

## Why
Cubby writes many kinds of record: an answer and its certificate, a refusal and its reason, a MoWM routing choice, a
world split, a rejected helper, a reviewed invoice field. Each order so far has sketched its own JSON-LD, and Nick's
drafts (chat, roleplay, documents, science, skills, swarm, bridge, MoWM worlds) differ in shape and break the hard
rules in the same places: status fields declared by hand, prose rationales, bare thresholds, guessed IRIs. CheezML
is one vocabulary for all of them, so one checker reads any record, and a record can only say what actually ran.

**What it brings, in plain words:** every answer Cubby gives comes with a receipt you can check, in the same format
as every routing choice and every reviewed document behind it. Nothing in it is a claim the system makes about
itself.
- **Over a conventional model:** logs, if kept at all, are in a different shape per component and record what the
  model says it did. A CheezML record that disagrees with its own numbers fails validation.

## What exists (Cowork, 2026-11-07)
- `docs/research/2026-11-07-cheezml-brief.md`, the panel's eight answers (`…-cheezml-panel.md`) and the synthesis
  (`…-cheezml-synthesis.md`): only GPT-6.1 and Gemini passed every tool check; **0 of 14 panel CubeLang programs
  compile**; the Decision Ontology's published namespace is a placeholder (`http://example.com/ns/dec#`).
- **The v0.1 draft** in `docs/research/2026-11-07-cheezml/`: `vocabulary.md`, `context.jsonld`, `shapes.ttl` (33
  node shapes, 17 SPARQL rules), six example records that conform, ten broken records that fail with their expected
  message, `disagreements.md` (choices, cuts, open questions) and `tools/` to rebuild and re-check.
- **Namespaces:** `https://w3id.org/cheezml/` with modules `core#` (`cheez:`), `cubby#` (`cubby:`), `mowm#`
  (`mowm:`) and `cubelang#` (`cube:`). No version in a namespace; versions go in `owl:versionIRI`. Records and
  instances have `urn:uuid:` ids. Standard vocabularies are reused as published (PROV-O, schema.org, OWL-Time, SKOS,
  DCMI terms).

## Build
0. **Move the draft into place** (where `CLAUDE.md` puts schemas and data files), one Turtle file per module with
   its `owl:Ontology`, `owl:versionIRI`, labels and definitions from `vocabulary.md`, and the SKOS code lists. Split
   the shapes per module. `tools/validate.py` becomes a test: the six examples conform, the ten broken records fail
   with their messages, no IRI leaves the allowed namespaces.
1. **Settle the open questions the code can answer** (`disagreements.md` §3 and §5), each with the file and line it
   comes from, and a test where it is behaviour:
   - "clears": > or ≥, for the floor and for `tau_margin` (the rule accepts equality either way until then);
   - what a world's `tau` is compared with, and whether it is `tau_spawn`;
   - which counter issues a bridge-spawned world's id, whether the counters pre- or post-increment, how they persist;
   - whether router pruning deletes (CheezML records only retirement; if the code deletes, that is a finding for the
     mowm repo, not a vocabulary change);
   - the hash algorithm and its canonicalisation; how the store holds statement content;
   - the domain and range of `dec:has_result` and `dec:involves_choosing`, so the `skos:closeMatch` links point the
     right way.
2. **0054's decision step uses CheezML's decision terms** (`cheez:Question`, `cheez:Option`, `cheez:Decision`,
   `cheez:isSatisfiedBy`, `cheez:isViolatedBy`), linked to do-ont by `skos:closeMatch`, not do-ont's placeholder
   IRIs. Update 0054's addendum text to say so.
3. **The record writers**, behind a flag, as a small library the orders call:
   - **certificate and refusal**, from the VM run (0054): written by the host from the run log, never by the program
     or a model;
   - **routing and split**, from what the bridge returned. If the bridge does not return what the record needs (the
     runner-up score, the thresholds used), that is a change in the mowm repo: a separate job, only with Nick's yes;
   - **helper rejection** (with 0062), **field review and commit** (with 0061), **conversation turn** (with 0059).
4. **Thresholds without a screen.** Each threshold in use (`tau_margin`, the floor, `tau_answer`, `tau_spawn` 0.5,
   `tau_prune_steps` 100, a world's `tau`) is listed with the screen that set it. One with no screen is recorded with
   its code default as the source (file and commit) and a code saying it was not screened; never an invented screen.
   This needs one change to the calibration shape; the list goes in the report.
5. **CubeLang programs, written against the compiler:** the route-or-spawn decision and the split check first, with
   `qc_decision.cube` as the pattern. Each passes `cubelang check --strict` with the main checkout's release binary
   (read only, never rebuilt in place) and runs on the VM in a test. The panel's programs are not used. A missing
   feature goes on the cubelang repo's P3 list, only with Nick's yes.
6. **The publishing folder**, built from the Turtle: per module `.ttl`, `.jsonld` and an HTML page, plus
   `context.jsonld` and the shapes. Nothing is registered: Nick sets up `ns.grillcheeseai.com`, Cowork drafts the
   w3id.org pull request (redirects with content negotiation), and Nick submits it. Until the w3id path resolves,
   records embed the context or load it from the package, never from the URL.

## Names
- **Releases:** Nick's A–Z cheese list, in order. The first tagged CubbyLLM release is **CubbyLLM Asiago**, then
  Brie, Cheddar, Danablu, … to Zamorano. The name goes in release notes and tag messages. Tags, versions and IRIs
  stay technical (`v0.1.0`; `owl:versionIRI` as in the draft, such as `https://w3id.org/cheezml/core/v0.1`),
  and the change log says which release ships which CheezML version.
- **Protocols** (names in documents only, never in an IRI or an identifier): the first suggestions (Cheddar
  certificates, Emmental refusals and gaps, Brie conversation, Halloumi sandboxed helpers, Mozza MoWM worlds) collide
  with release names B, C, E and H. **Proposed instead, from cheeses outside the list:** Comté (certificates; a
  cheese with a certified origin), Maasdam (refusals and gaps; the holes), Mozza (MoWM world records), Paneer
  (sandboxed helpers; it keeps its shape under heat), Raclette (conversation; eaten together). **Nick picks**; until
  then documents use the technical names.

## Acceptance (pre-registered)
- **The draft's checks pass as tests:** 6 of 6 examples conform; 10 of 10 broken records fail with their expected
  message.
- **Records from real runs:** replaying 0054's latest run writes one certificate or one refusal per question, with no
  question missing a record, and 100% conform.
- **Records agree with what ran:** each written record is re-derived from the run log and compared field by field;
  0 differences.
- **Mutation test:** for each record kind, 100 copies with one recorded value changed (a margin, an outcome, a depth,
  a source, a reason code); every mutant fails validation, and the report lists any that pass.
- **Round trip:** expand, compact and expand again gives an isomorphic graph for every written record.
- **Hard line:** no record carries a truth score, a status declared by hand, a prose rationale, an embedded vector or
  weight, or a threshold without its calibration (closed shapes; tested by the broken records).
- **Kill:** a written record that conforms but disagrees with what ran. The writers stop until the cause is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model writes a record. The panel's answers were a design input, checked by tools; nothing from them is
  copied without passing the same checks.
- **CubeLang changes go to the cubelang repo, MoWM changes to the mowm repo,** each only with Nick's yes. The main
  cubelang checkout's `target/release` binary is never overwritten.
- No local path in a committed file. No URL in a record that does not resolve.

## Out of scope
- Emotion (EmotionML stays an export format), story worlds, split questions and domain worlds: each joins CheezML when
  its order needs a record, through the same checks.
- RO-Crate, nanopublication and STATO packaging (0054's addendum links to CheezML records; it does not redefine them).
- Records about containers and bridges until they compile to something (0062's addendum).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **The open questions, answered:** <with file and line>
- **Thresholds and their screens:** <the list; code defaults marked>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
