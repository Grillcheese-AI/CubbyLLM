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
- **Protocols** (names in documents only, never in an IRI or an identifier), taken from cheeses outside the
  release list so that no name means two things (Nick: "sounds good", 2026-11-07):
  - **Comté:** certificates (a cheese with a certified origin);
  - **Maasdam:** refusals and gaps (the holes);
  - **Mozza:** MoWM world records;
  - **Paneer:** sandboxed helpers (it keeps its shape under heat);
  - **Raclette:** conversation (eaten together).

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

## Cowork's note (2026-11-09): CheezML has its own repository, and the w3id request is open
Nick: "go ahead" (new public repo, GitHub Pages until ns.grillcheeseai.com exists, the w3id pull request opened
after his review).
- **The vocabulary now lives in Grillcheese-AI/cheezml** (public). It holds the source (`src/`: the term tables, the
  context, the shapes and the examples), the built site (`docs/`, on GitHub Pages at
  grillcheese-ai.github.io/cheezml) and the checks (`tools/`). Changes to the vocabulary are made there.
  `docs/research/2026-11-07-cheezml/` in this repo stays as the dated draft.
- **What that changes in this order:**
  - **Step 0:** CubbyLLM takes the vocabulary from that repo at a pinned commit (a submodule or a vendored copy;
    Claude Code's call). The tests run its `validate.py` and `check_site.py`.
  - **Step 6 is done:** one OWL module per namespace in Turtle, JSON-LD and RDF/XML, an HTML page with an anchor
    per term, the context, the shapes and a frozen v0.1 copy. Each module gives the same graph in all three formats,
    and every term the records use is declared and documented.
  - **The shapes moved to their own namespace,** `https://w3id.org/cheezml/shapes#` (they were in the module
    namespaces). Validation is unchanged: 6 examples conform, 10 broken records fail with their messages.
- **w3id.org:** pull request perma-id/w3id.org#7120 adds `ids/cheezml/`. The module and version IRIs are
  content-negotiated (303), and any other path goes to the file of that name (302). It passed w3id's checker and a
  test on a local Apache built with their scripts. Until it is merged, records keep the context inline.
- **Later:** when ns.grillcheeseai.com becomes the site's custom domain, a follow-up pull request points the
  redirects there.
- **Licences:** CC BY 4.0 for the vocabulary, shapes, examples and documentation; MIT for the scripts.

## Cowork's addendum (2026-11-10): records kept private, and what the VM must report
Nick: "it should also be encrypted so not everyone can see the answers". The VM's side is a new job in the cubelang
repo, `CUBELANG_FIXES.md` section P4: which facts were read, which checks ran, which rules were applied, error codes,
hashes of what ran, and no values in logs or over a network. This addendum is the host's side.

**What it brings, in plain words:** every answer still comes with a receipt anyone can check is genuine, but only
the people allowed to see it can read what it says.
- **Over a conventional model:** chat logs are usually stored in clear, readable by whoever can reach the server,
  and nothing proves they weren't edited.

### 1. From the run report to the record
Map each field of the P4 report to its CheezML term, in a table in this order's report:
- facts read → `cheez:FactVersion`, with `prov:hadPrimarySource` and the revision;
- checks → `cheez:CheckRun`, with `cheez:outcome`;
- rules → the derivations artefact, by hash;
- error codes → `cube:RunOutcomes`;
- the source and bytecode hashes, strict mode and the VM commit → the program run.

A field the VM doesn't report yet is left out of the record, never filled in.

### 2. Who may read what
This follows Nick's decisions for CubbyConsul (tenants are companies; viewers, members and admins; sources go to
the audit view only):
- **The conversation part** (the question, the answer, the turn) is readable by the tenant's members and viewers in
  the sessions they share.
- **The provenance part** (facts read, sources, revisions, rules, checks) is readable only through the audit view
  (owner, auditor). Regular users never see where Cubby learned something.
- **Nobody else.** Real records are never published. Only the vocabulary, the shapes and the made-up examples are
  public.

### 3. How
- **Two layers per record:**
  - a public envelope: the record's `urn:uuid`, its type, the recorded time, the key ids, a signature and the
    commitments below;
  - an encrypted body: the full JSON-LD record, split into the two parts above, each encrypted for its readers.
- **Standard formats only, no home-made cryptography:** JWE (RFC 7516) with AES-256-GCM for the bodies, JWS with
  Ed25519 for the signature, through maintained libraries (Python `cryptography` or `joserfc`).
- **Sign, then encrypt.** The host signs the plain record, so a reader who decrypts it can check that nobody changed
  it, and anyone can check the signature on the envelope.
- **Commitments for selective proof.** For the answer value and each fact read, the envelope holds a salted hash
  (HMAC-SHA-256, with a random salt per record). An owner can reveal one value and its salt to prove it matches the
  certificate, without revealing anything else. Plain hashes are not used: a short answer (a year, a yes) can be
  guessed from its hash.
- **Keys:**
  - one data key per tenant and part, wrapped by a master key that comes from the OS key store or a key service;
    never in a file, a log, an argument or a commit (the standing rule);
  - key ids in the envelope, so keys can be rotated by re-wrapping them, without re-encrypting every record;
  - destroying a tenant's key makes its records unreadable for good. That is never done without Nick's yes; it
    conflicts with "retire, never delete" and is his decision.
- **Validation:** SHACL runs on the decrypted record inside the host. The envelope gets its own shape. The new terms
  (envelope, key id, commitment, signature) are added in Grillcheese-AI/cheezml, through its checks.

### 4. Tests
- A record encrypted for tenant A cannot be decrypted with tenant B's key, and a member cannot open the provenance
  part.
- One changed byte in the envelope or the body fails verification.
- A revealed value with its salt matches its commitment; any other value fails.
- After a run over planted secret strings, none of them appears in any host or VM log, nor in the envelope.
- No key material in any file the repository tracks (extend `test_local_paths`' scan).

## Addendum (2026-11-10): CheezML v0.2 is published
**In plain words:** a receipt now also says which copy of the knowledge base it read, which date the question was
about, and which outside source it looked at.
- **Over a conventional model:** a model's answer never says which database release or which day's data it came
  from. A v0.2 record that leaves this out fails validation.

- **What changed** (cheezml repo, commit `0e14aa2`; `https://w3id.org/cheezml/core/v0.2`; v0.1 stays frozen):
  - `cheez:SourceRead`: a remote source read during a run, with its endpoint, its query and response by hash, its
    time and the source's licence. The run `prov:wasInformedBy` it.
  - Releases: a `cheez:Artifact` coded `cheez:OntologyRelease` or `cheez:DatasetRelease` must state its
    `schema:version`.
  - `cheez:asOf` and `cheez:storeSnapshot` on a run: the date the question was evaluated at, and the store state
    read, by hash.
- **What the record writers must do** (step 3):
  - every run that reads a fact version records its store snapshot and its as-of date. Without both, the run fails
    validation;
  - each live source read (order 0007) is written as a `cheez:SourceRead`;
  - each ontology or dataset release a domain world relies on (order 0058) is cited as a release with its version
    and hash.
- **Tests:** 7 examples conform and 13 broken records fail, up from 6 and 10. The first acceptance line uses these
  counts.
- **Domains:** the list and the rule for adding a module are in the cheezml repo's `DOMAINS.md`. Finance (`czfin`)
  is the first module, with order 0058.

## Cowork's note (2026-11-11): 0054's PROV-O certificate is this order's certificate record
Nick asked whether the PROV-O form of 0054's 2026-11-06 addendum is next. The ruling is in 0054: it is not built
as a form of its own. It is step 3's certificate and refusal writer, and 0054's round-trip line is met when this
order's acceptance is.
- **What the writer waits for** (0054, Cowork's read of the name statement as built):
  - item ids on the facts and on the name statement, taken from the world's decision, never matched by label;
  - the live source's revision or time of reading, for `cheez:SourceRead`;
  - a state that the worlds below the store can be cited at. An answer whose world names none gets no record,
    and the report counts those.
- **The name statement is part of the certificate record:** an entity derived from the label or alias statement
  and from the facts its steps rest on; the name step as a decision in this vocabulary's terms. If a term is
  missing for it, it is added in the cheezml repository, through its checks.
- **The queue place is unchanged:** after 0055, with 0054's writers.
