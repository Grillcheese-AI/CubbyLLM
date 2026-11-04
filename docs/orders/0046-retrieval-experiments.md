# 0046 — Retrieval, measured before it is built: experiments R0 to R8
status: open (R2 and R3 on dev: no arm passes, test unread; R1b killed on dev (below the API on Mintaka; the names are there, the namesake sets are too large for 20 unranked); R3o, then seed 1 with B1q and the fork, then R4)
written: 2026-10-28 by Cowork · approved: 2026-10-28 by Nick ("tested before implementing"; a plan doc and an experiments-only order) · hypothesis: none (each experiment below is pre-registered)

## Why
Retrieval decides which item a name means, and which value a relation gives, with tiers written one
scoreboard row at a time (0037, 0043), scored on those same rows. Nothing measures them on names they
were not written for. Nick, 2026-10-28: "the whole retrieval side needs to be rethought and
replanned, and tested before implementing." `docs/RETRIEVAL.md` is the plan, written from three
independent plans (Cowork's sealed one, CubbyConsul's board, Gemini 4 Argon). This order runs its
experiments. It builds nothing on the serving path.

**What it brings, in plain words:** we learn which way of finding things works on questions nobody
tuned Cubby on, in English and French, before any of it is built.
- **Over a conventional model:** a model's retrieval is judged by its average score. Here each stage
  is measured apart, every miss says why, and a design that speaks one wrong value is killed.

## Build
All of it lives in `validation/` (network fetchers in `cubby_serve/`), never in `cubbyllm/`. Data sits
off-repo under `$CUBBY_DATA_ROOT`. Only ids, counts and the fame set are committed.

1. **The sets**
   - **Builders** (`validation/retrieval_sets/`) for:
     - Mintaka (EN + FR);
     - QALD-9-plus (EN, FR; its Wikidata queries);
     - WebQSP_EL and GraphQ_EL;
     - SimpleQuestions-Wikidata;
     - ShadowLink (Top, Shadow, Neutral, Tail);
     - AmbER;
     - CronQuestions.
   - **One schema** per row: id, language, question, gold mention items, gold property when the set has
     one, gold answers, source, split.
   - **Wikipedia titles to Wikidata ids:** through Wikidata by title, cached.
   - **Licences:** confirm each in the report.
   - **Downloads:** ask Nick before any download over 1 GB.
   - **Splits:** dev and test, fixed by seed 0. Tune on dev only. Read test once, at the end of each
     experiment.
   - **Today's rows** (the 187 and the stack's 20) are regression only.
2. **The fame set** (`validation/retrieval_sets/fame_set.py`), built mechanically:
   - **Names:** every label that Wikidata5M gives to two or more items of one kind (P31, one `subclass
     of` step), where at least two of those items carry a relation from a fixed list (place of birth,
     place of death, date of birth, country, capital, occupation).
   - **Sample:** 500 names, drawn by seed 0.
   - **Bare-name questions,** EN and FR, from fixed templates ("Where was {name} born?" / "Où est né
     {name} ?"). Gold: no single item may be picked.
   - **Qualified variants:** the same question with one discriminator from one item's own facts (its
     occupation, a place it is in). Gold: that item. Spot-checked by hand: 50 rows, the check in the
     report.
   - **Freeze:** committed with its SHA-256 before R2 runs. Not changed after.
   - Built by name collision, not by sitelinks or degree.
3. **The relation lexicon:** property labels and aliases, EN and FR, from the property table that
   exists, frozen with its SHA-256 before any held-out row is read.
4. **The harness** (`validation/retrieval_bench/`): runs each arm of each stage on a set and writes
   `validation/logs/retrieval_R<n>_<date>.json`. Every row records its candidates (unordered within a
   kind), the stage that refused, and the reason.
5. **The experiments,** in this order: R0, then R1, then R2, R3 and R4 (in parallel), then R5 and R6,
   then R7, then R8. Each is described in `docs/RETRIEVAL.md`. The lines are below.

## Acceptance (pre-registered)
- **R0, the denominator**
  - Today's pipeline on every dev set. Each refusal is labelled by stage and route:
    - reading;
    - no candidate;
    - several candidates;
    - wrong candidate;
    - relation;
    - several values;
    - item absent from every local store (Wikidata5M, DBpedia's titles, EN/FR Wikipedia's titles).
  - English and French apart.
  - Also reported: whether the local Wikipedia text kept its links and redirects.
  - The dump line is decision 4 in `docs/RETRIEVAL.md`, fixed before R0 counts.
- **R1, finding**
  - Arms: the search API (today); exact; + normalized; + trigram codes; + BM25 as supplement only
    (fills the set when the others give fewer than K, never displaces).
  - Metric: recall@K, K in {5, 20, 100}, with names absent from the index's build split scored apart,
    plus a typo set (one or two letters changed in 20% of names).
  - **Gate:** the best local arm's recall@20 is at least 0.90 on Mintaka EN and 0.85 on Mintaka FR, and
    above the API's on every set. The trigram arm is kept only if it adds at least 5 points on the
    typo set.
  - **Kill:** no local arm beats the API at @20 on Mintaka.
- **R2, choosing** (on R1's candidates)
  - Arms: today's tiers; BM25 cue overlap with each item's facts and text; qFHRR context against one
    prototype per item; each plus the usage table between kinds (only if decision 1 allows it).
  - **Gate:**
    - fame set, bare names: 0 single picks;
    - qualified variants: 0 wrong, at least 70% resolved;
    - ShadowLink: Shadow's accuracy among spoken picks within 5 points of Top's;
    - on Mintaka and WebQSP_EL test: no more wrong picks than today's tiers (and at most 1 in 50), with
      more questions decided.
  - **Kill:** any bare-name single pick, or a Shadow-Top gap over 10 points.
- **R3, the relation**
  - Arms: today's grammar; the frozen exact lexicon over the spans left after the names; the qFHRR
    codebook behind a rank-normalized margin set on dev.
  - Metric: property recall@5 given the gold item, on SimpleQuestions-Wikidata, QALD-9-plus (EN, FR) and
    Mintaka (EN, FR).
  - **Gate:** at least 0.85 on SimpleQuestions-Wikidata and 0.80 on QALD-9-plus FR. The qFHRR arm is
    kept only if it adds at least 3 points at the same wrong rate.
  - Report the share of each set today's grammar parses.
- **R4, several values and time**
  - Arms: the source's own order (preferred rank, then single-value constraints, then the values with
    no stated end); + an interval filter for an explicit time; + "present tense means now", run only as
    Nick's decision 2 says.
  - Sets: CronQuestions (single-entity questions) and the scoreboard's 44 + 2 rows.
  - Wikidata5M holds no ranks or qualifiers. For these items, fetch the statements by id (rank, start,
    end, point in time) through the source's cache, and record the date of the fetch.
  - **Gate:** 0 wrong; at least 20 of the 46 rows spoken; at least 60% of CronQuestions single-entity
    dev questions spoken.
  - **Kill:** any wrong value, such as Valladolid or Toledo beside Madrid.
- **R5, the asker's memory** (synthetic dialogues, 200 per language)
  - **Gate:** referents resolved at least 95%; 0 values supplied by memory; 0 crossings between askers;
    ties untouched unless decision 3 allows it.
- **R6, passages**
  - A reader's facts from retrieved passages (500 sampled).
  - **Gate:** 0 erroneous admissions to the canonical store (latent until corroborated, as today);
    passage recall@10 reported.
- **R7, scale**
  - The name index and the shortlist at 5M, 20M and 75M names.
  - Reported: recall, p50 and p95 latency, memory and build time, with the machine and corpus stated.
  - **Gate:** p95 lookup at most 50 ms at 75M. Nick sets the memory line from the numbers.
- **R8, end to end**
  - The arms that passed, together, against today's pipeline on each held-out test set, EN and FR.
  - **Gate:** spoken at least today's.
  - **Kill:** any wrong spoken value.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim).
- **Nothing on the serving path changes.** No flag, no default, no tier.
- **No external model in any arm,** and none to build the stores. The fame set's questions come from
  templates, not from a model.
- **Nothing is ranked by fame within a kind:** sitelinks, degree, link counts and description length
  decide nothing, and candidates cross unordered.
- **No hand rule is written for a failing row.** A failure is reported as what it is.
- **Licences:** the sets stay off-repo when their licence asks it; logs hold ids and counts only.

## Out of scope
- Building any stage for serving: an order per stage, after its experiment passes and Cowork has
  read it.
- The 959 GB dump: only on decision 4's line.
- The trunk (0045).

## Report (R0 and R1; Claude Code, 2026-10-28)
- **Commits:** `c5e59ee` (the sets, the frozen lexicon, the fame set, the index, the harness); this one (R0, R1, the report).
- **Files:** `validation/retrieval_sets/` (download, the builders, `common`, `lexicon`, `fame_set`, `frozen.json`,
  `fame_set.jsonl`); `validation/retrieval_bench/` (`name_index`, `mentions`, `stores`, `r0_denominator`,
  `r1_finding`); `cubby_serve/sources/http.py` (`download`); tests `tests/validation/test_retrieval_bench.py` (8),
  `test_fame_set.py` (5).
- **Tests:** 1147 passed, 1 failed (`test_every_module_declares_wiring`, order 0002's known red).

### The sets (build items 1-3)
- **Downloaded** (366 MB, every file under the 1 GB line; `validation/logs/retrieval_sets_2026-10-27.json` has the
  counts). Licences: Mintaka CC BY 4.0; QALD-9-plus CC BY 4.0; WebQSP_EL and GraphQ_EL released with BLINK/ELQ (MIT;
  the questions are WebQSP's and GraphQuestions'); SimpleQuestions-Wikidata: no licence stated in the repository
  (SimpleQuestions itself CC BY 3.0); ShadowLink: none stated on its card; AmbER CC0; CronQuestions MIT; Wikidata5M
  CC0. All stay off-repo; logs hold ids and counts only.
- **Splits:** each set's own dev and test (QALD: train as dev; ShadowLink and AmbER: halves by seed 0, AmbER by
  name). ShadowLink's Neutral subset names no entity and is left out; 2,702 of its 2,712 titles map to an item
  (Wikipedia's API: Wikidata's title lookup answers 503 for any redirect title).
- **The relation lexicon** (34,920 EN and 22,059 FR wordings, 12,684 properties) frozen before any held-out row was
  read: SHA-256 `fffbd704...ba6e2` in `frozen.json`.
- **The fame set** (`fame_set.jsonl`, SHA-256 `c0525052...cc7e`): 500 names drawn by seed 0 from 34,773 eligible;
  1,910 rows (500 bare EN, 500 bare FR, 466 qualified EN, 444 qualified FR); relations country 222, place of birth
  142, occupation 117, place of death 19. Date of birth cannot be used: Wikidata5M holds item-to-item facts only.
  Labels are Wikidata's English ones as the hdc world holds them: Wikidata5M's entity file lists names unordered
  (Titian's first is "ditiano") and gives no label.
  - **Spot checks by hand, 50 rows each:** first, 45 hold by their words, 3 ambiguous (Rye High School in Rye; Andrea
    Maffei "(Italie)", both men Italian; Reid Avenue in Brooklyn), 2 doubtful; second, after two rules, 43 hold, 7
    ambiguous (missing citizenship; "Great Britain" against the UK; an occupation Wikidata5M lacks); third, fresh,
    after two more rules: **50 of 50 hold.** The rules (in the module): a label shared or held whole by another's
    value tells nothing apart; a place another item is not ruled out of (its places at or above it, or a homonym
    place holding it) tells nothing apart; nothing recorded rules nothing out; a person asked an occupation is told
    apart by birthplace, not citizenship. One residual class stays: an occupation the 2019 data lacks for the other
    namesake (Howard Storm, "the writer"); no rule over this data can see it.

### R0, the denominator (`retrieval_R0_2026-10-28.json`)
- **Pipeline:** the ask loop as the stack mounts it (wiki + hdc worlds, live Wikidata source, lexicon, everyday table,
  tense on, today's defaults). **No model in the programs seat:** the host's own reader only, so "reading" is the
  table's reach, not the 450M's.
- **2,650 dev questions** (300 a set and language; QALD FR 250), first stage that stopped each:

  | | EN (2,100) | FR (550) |
  |---|---|---|
  | reading | 1,739 | 533 |
  | no candidate | 214 | 9 |
  | several | 48 | 4 |
  | wrong candidate | 5 | 1 |
  | relation | 5 | 0 |
  | several values | 9 | 0 |
  | value missing / time / other | 15 | 0 |
  | spoken | 65 | 3 |

- **Every spoken answer read by hand (68):** 62 right (one, "How tall is Big Ben?", said "96" without its unit),
  **6 wrong**:
  - **5 from one defect in today's serving path:** the everyday table's restatement drops a leading "The" from the
    name -- `EverydayForms().restate("Who wrote The Last Dragon?")` gives the seed "Last Dragon" -- and the source's
    one-exact-hit tier then resolves a different item labelled exactly that: Freud's *The Interpretation of Dreams*
    (asked in EN and FR) became Antiphon's *Interpretation of Dreams*; *The Ides of March* (Wilder) became
    Manfredi's novel; *The Last Dragon* (De Mari) became McDermott's; "the skin" (the film) became the singer Skin.
  - **1 not what was asked:** "In which city did John F. Kennedy die?" was said "Parkland Memorial Hospital" (his
    place of death, a hospital, not a city).
  Order 0046 changes nothing on the serving path, so neither is fixed here; both need an order (question 3).
- **The links:** the local Wikipedia text keeps **no links and no redirects** (`wikipedia_dedup`: titles and plain
  text, 74,567 articles; `wiki_full`, 14 GB: plain text, no titles). Decision 5 applies.
- **The dump line** (decision 4's recommended line, written into the script before counting; Nick's decision is
  still open): 3,347 gold mentions in the sample (over the 1,000 minimum); 312 sit in route-labelled seed misses (no
  candidate, several, wrong candidate); **17 of them (5.5%) are items absent from every local store**, against the
  20% line. **Under the line:** the full dump would not fix today's seed misses. Of the 1,975 distinct gold items,
  Wikidata5M holds 1,805, DBpedia's titles 1,631, the routed Wikipedia's titles 29.

### R1, finding (`retrieval_R1_dev_2026-10-28.json`, `retrieval_R1_test_2026-10-28.json`)
- **Arms:** Wikidata's search (its own order, up to 100); the local index (`name_index`: 12.4M names of 4.8M items,
  Wikidata5M's names and the hdc world's EN/FR labels) exact, + normalized, + trigram (FTS5 blocking, reordered by
  the capsules' code), + BM25 as a supplement only. A set or a run of equal scores enters a shortlist whole or not
  at all (no tie broken). Mentions: the sets' own where marked, else the gold item's names located in the question
  (Mintaka FR, QALD, SimpleQuestions, CronQuestions). Samples: 1,000 Mintaka questions and 500 of every other set per
  split and language (seed 0); 6,373 dev mentions and 6,081 test, 20% with a typo variant.
- **recall@20, test** (dev within 3 points everywhere, in the dev log):

  | set | API | best local | items the index holds: API / local |
  |---|---|---|---|
  | Mintaka EN | 0.862 | **0.864** | 0.867 / 0.921 |
  | Mintaka FR | 0.971 | 0.738 | 0.980 / 0.787 |
  | WebQSP_EL | 0.884 | **0.925** | 0.885 / 0.939 |
  | GraphQ_EL | 0.820 | **0.852** | 0.860 / 0.925 |
  | ShadowLink | 0.673 | **0.707** | 0.656 / 0.810 |
  | AmbER | 0.950 | 0.874 | 0.950 / 0.912 |
  | QALD EN / FR | 1.000 / 1.000 | 0.885 / 0.529 | |
  | SimpleQuestions | 0.974 | 0.919 | |
  | CronQuestions | 0.996 | 0.745 | |

- **Against the lines:**
  - best local recall@20 >= 0.90 on Mintaka EN: **fail** (0.864 test, 0.877 dev);
  - >= 0.85 on Mintaka FR: **fail** (0.738, 0.759);
  - above the API on every set: **fail** (above on Mintaka EN, WebQSP_EL, GraphQ_EL, ShadowLink; below on the rest);
  - the trigram arm adds >= 5 points on the typo set: **pass**, by 33 to 76 points (the API finds 0 to 5% of typos);
  - BM25 as a supplement adds nothing at @20 on any set (it equals the trigram arm everywhere);
  - **kill** ("no local arm beats the API at @20 on Mintaka"): read per language, not killed (EN 0.864 against
    0.862); read on Mintaka as a whole, killed (the FR gap outweighs). Cowork's call (question 1).
- **What the misses are** (Mintaka dev): EN, 179 local misses: 77 items the 2019 index does not hold, 63 near but
  ranked past 20, 21 with no name near, 18 in a shared-name set over 20; FR, 267: **130 items the index holds under
  no French name** (the index has French labels for 1.2M of its 4.8M items), 68 near past 20, 54 not held. The two
  arms are complementary (EN: 140 found only locally, 124 only by the API).
- **A bias the reader should weigh:** where a set gives no mention words, they were located with the gold item's
  current Wikidata labels, which are exactly what the API searches; that favours the API on Mintaka FR, QALD,
  SimpleQuestions and CronQuestions. On the sets that mark their own mentions (Mintaka EN, WebQSP_EL, GraphQ_EL,
  ShadowLink, AmbER) the local index is above the API on four of five.

- **Not verified here:** the 450M in the programs seat (R0 runs the host's reader only); the fame set's 444 French
  qualified rows beyond the 50 checked (their French was read by hand only in those).
- **Questions:**
  1. **R1's kill line:** per language (not killed) or Mintaka as a whole (killed)? Either way the gate is not met; the
     measured lever is French names (130 of 267 French misses) and coverage past 2019 (77 of 179 English misses).
  2. **Decisions 1-4** are still open as far as the repo says. R0 is under the recommended dump line (5.5% against
     20%). **Decision 5 applies:** the local Wikipedia text kept no links or redirects; the redirect tables are a few
     GB, over the 1 GB line, so they wait for Nick's yes.
  3. **Today's pipeline speaks wrong answers on held-out names:** 5 of 68 from the dropped "The", 1 class mismatch.
     A fix order (keep the article for the seed; a "which city" answer checked against its class) is outside 0046.

## Cowork's check of R0 and R1 (2026-10-31)
Read: the report, both R1 logs, R0's log, and the fame set's construction and spot checks.
- **R0's six wrong answers** go to order 0047, the one exception to "nothing on the serving path": it
  only removes wrong answers.
- **Reading is where held-out questions stop:** 1,739 of 2,100 English and 533 of 550 French, before any
  name is looked up. R3 is therefore the most important experiment.
  - **R3 gains an arm:** the 450M in the programs seat. The model reads the question and writes the
    program, and the VM checks it, as Nick's design asks; the host's table is scaffolding.
  - **Report R3's coverage first:** the share of each set that each arm reads, then its recall.
- **The dump:**
  - 17 of 312 route-labelled seed misses (5.5%) are items absent from every local store. That is under
    the recommended line (20%), so the full claims extract is not justified by absent items.
  - R1 shows a different need, names. 130 of Mintaka FR's 267 misses are items the index holds under no
    French name, and 77 of the English 179 are items past 2019. That is R1b below, not the dump.
- **R1's kill line (question 1):** the line was written "on Mintaka", not per language. Reading it per
  language after seeing the numbers is the post-hoc choice this plan rules out, so Mintaka as a whole
  decides: **killed.** The local index as built (Wikidata5M's names, 2019) does not replace Wikidata's
  search.
  - **Not killed: the trigram arm.** It passes on typos by 33 to 76 points, where the API finds 0 to 5%.
    It stays as a second candidate source.
  - **Noted:** the mentions located with current Wikidata labels favour the API. R1b scores the sets
    that mark their own mentions first (Mintaka EN, WebQSP_EL, GraphQ_EL, ShadowLink, AmbER).
- **R1b, a new arm with its own lines** (written here, before it runs):
  - **Names:** EN and FR labels, aliases and Wikipedia titles for every item, from the current Wikidata
    dump, reading only its `id`, `labels`, `aliases` and `sitelinks` columns, never `claims`.
  - **Redirects:** the EN and FR Wikipedia redirect tables.
  - **How it is fetched:** on Colab, streamed, kept on Drive (`docs/RETRIEVAL.md`). Measure the
    columns' size on 10 files first, and report it before fetching the rest. **This needs Nick's yes**
    (decision 5 and this extract).
  - **Gate:** recall@20 at least 0.90 on Mintaka EN and 0.85 on Mintaka FR; above the API on every set
    that marks its own mentions; at most 5 points below the API on the others.
  - **Kill:** below the API on Mintaka EN or FR.
- **R2's candidates meanwhile:**
  - the union of the API's shortlist and the local arms', unordered (they are complementary: in English,
    140 items were found only locally and 124 only by the API);
  - a run of equal scores stays whole, as in R1.
- **Nick's decisions gate building only, not experiments:**
  - R2 runs with and without the usage table between kinds;
  - R4 with and without "present tense means now";
  - R5 with and without "a past choice settles a tie".

  Each arm's numbers are what Nick decides from.
- **The fame set** stands (50 of 50 on the third check; SHA-256 `c0525052...cc7e`). Its frozen
  construction is what R2 reads. Date of birth is out, as the report says.
- **Next:** 0047 first (wrong answers on the live path), then R2, R3 and R4. Then R1b when Nick says
  yes to the names. Status: open.

## Cowork's addition: the VM arm (2026-10-31, Nick: "shouldn't we use the VM for retrieval?" -- yes)
The exact stages will live in CubeLang's layered store (`docs/RETRIEVAL.md`, "Where each stage lives"), so
what is measured must be that store, not a stand-in.
- **R1b's local arm is the VM's knowledge segment.**
  - Names and aliases go to items as segment keys (`cubelang knowledge build`). The lookup is exact,
    then typo-tolerant through the segment's own index, one or two edits away.
  - The Python index of R1 stays as the reference arm, run on the same mentions.
- **R7 measures the segment:** at 5M, 20M and 75M names, report:
  - build time and file size;
  - memory when opened (plain and memory-mapped);
  - exact and typo lookup latency (p50, p95);
  - recall against R1's index.

  The gate stays p95 at most 50 ms at 75M.
- **CubeLang needs one addition for this:** a typo lookup over knowledge keys (`knowledge fuzzy KEY
  --edits N`, and the same in a program).
  - The history's fuzzy search does it over its term index today.
  - **Claude Code builds it, as part of this order** (Nick, 2026-10-31). Build it in the CubeLang checkout
    on the branch that holds the layered store, beside the main one, never over main's release binary.
  - Its tests sit beside the store's: exact and typo lookups answer as a scan of every key would.
  - R1b does not start without it.
- **R4's statements:** the logic is tested in Python first, as written. If it passes, the statements
  (rank, start, end, point in time) are stored as VM facts and checked there, in its build order.
- **Nothing here changes the serving path.** Serving through the VM waits on 0042.

## Cowork's addition: branch, check, decide (2026-10-31)
Nick, 2026-10-31: "we have a lot of reasoning already built, why aren't we using it with CoT and GoT within
the VM?", then "also: possibility worlds (H-E17), forkable recurrent state, counterfactual re-entry (H-P1 to
H-P5)".

Today the host's Python walk does every lookup and choice. The program the VM checks only carries the
objects the host found (`plan_verify.py`: "bind-only"). The thought graph records alternatives but decides
nothing (`graph.py`).

The parts to do it properly are built and measured:
- the VM's `QUERY`, its branches on the wire (each on its own clone of the store, a report of every lookup),
  `ASK` / suspend and its budgets (CubeLang `docs/BRANCH.md`, `docs/ASK_SUSPEND.md`);
- the possibility worlds (H-E17, passed at its screen 2026-09-29, not on the serving path);
- the forkable recurrent state and counterfactual re-entry (H-P1, H-P5, verified at toy scale).

This section tests them as one retrieval step, before anything is built.

**The step:** where a name gives several items (or the trunk's entropy marks a choice point, H-P2):
1. one branch per candidate item;
2. in each branch, the program for that reading runs in the VM: `QUERY` and the hops, verified;
3. each branch's reading is put to the possibility worlds: which world, and is it possible there? An
   impossible branch drops, an unknown one stays latent (never spoken), a possible one stands;
4. the graph over the branches decides:

   | Branches that verify | Outcome |
   |---|---|
   | none | refuse |
   | one | answer |
   | several, same value | say it once |
   | several, different values | answer each, at most three, or `ASK` |

Branches are never ranked. Similarity may only order which branch runs first under a budget.

**In plain words:** Cubby thinks each possible meaning through on its own, checks each one against what it
knows and what is possible, and only then decides what to say.
- **Over a conventional model:** a model folds the meanings into one guess in a single pass. Here each meaning
  is its own checked line of reasoning, and the record shows why one was said.

**Arms** (in R2, on the same candidates and sets; and in R4, where the branches are the values):
- **B1, VM branches.** The host opens the branches on the wire. A fixed program per relation runs `QUERY`
  against a test segment of the candidates' facts. The outcome is read from the branch reports. No model,
  no new opcode.
- **B2, B1 + the possibility worlds** (`bridges.possibility`, the store oracle, member routing as H-E17
  settled):
  - over Wikidata5M's fact worlds, the law worlds where they exist, and the fiction worlds;
  - in R4, the time worlds: a fact world sliced by when the fact held.
- **B3, B2 + the forked trunk.**
  - The 450M reads the question once. Its state is forked once per candidate (H-P1: exact, O(1) in the
    prefix) and re-entered with that candidate substituted (H-P5).
  - Each fork writes its branch's program in the programs seat, and the VM runs it.
  - B3a, optional: branch only where the trunk's entropy marks a choice point (H-P2). Report its AUROC
    against the gold ambiguous names.

**Lines** (R2's lines hold for every arm; these are added):
- **The possibility worlds never prune the gold item:** 0 on the fame set's qualified rows and on Mintaka's
  test. Kill: one pruned gold item.
- **The fame check, per arm:** Shadow's accuracy within 5 points of Top's. For B3, also the forked trunk's
  branch programs: correct as often on Shadow items as on Top items, within 5 points. Kill: over 10 points.
- **Fork against independent passes** (B3): every branch's state equal to an independent pass on the same
  substituted prefix (H-P1's line, now on real questions), and the cost per added branch reported.
- **Budget:** p95 per question with k branches, k in {1, 2, 3, 5}. On the wire, if it exceeds R7's per-hop
  line, the CubeLang task is the in-program fork, already designed (`VM::snapshot` / `restore`; opcode bytes
  0x3E and 0x3F free).
- **R4 with time worlds:** 0 wrong. A second value of a relation that may hold several is never refused as a
  contradiction: H-E17's 1.25% of true facts refused were that.

**Order:** B1 runs with R2's other arms. B2 follows once B1 has numbers. B3 needs the 450M in the programs
seat, which R3 sets up. Nothing here reaches the serving path.

## R1b: go (Nick, 2026-10-31: yes to the names extract and the redirect tables)
- **Claude Code writes the notebook, and Nick runs it on Colab.**
  - **What it reads:** the current Wikidata dump's `id`, `labels`, `aliases` and `sitelinks` columns only (never
    `claims`), keeping EN and FR labels and aliases, the enwiki and frwiki titles, and the item ids.
  - **What it fetches:** the EN and FR Wikipedia redirect tables, with the page tables they need to name a
    redirect's target.
  - **Where the extract goes:** Drive, as the master copy. The raw files are not kept.
- **First, 10 parquet files only.** Report the bytes read per file and the extract's size, and project the
  whole run from them. Nick says go on the projection before the rest runs.
- **Then the VM segment:** names and aliases to items, built from the extract (the VM arm above), copied to an
  SSD for the runs, and R1b scored with the lines already written.

## R1b: the notebook is ready (Claude Code, 2026-10-31)
- **Nick runs `notebooks/r1b_names.ipynb` on a CPU runtime.** The logic is in `validation/retrieval_sets/names_extract.py`
  and `wiki_redirects.py`, tested on canned inputs in `tests/validation/test_names_extract.py`. The notebook stops after
  the plan until `GO = True`. The extract is resumable.
- **The source, a change to note:** not the 959 GB copy (`philippesaade/wikidata`, whose "10 parquet files" the plan
  assumed) but the `wikidata-pq` tables (`permutans/wikidata-labels`, `-aliases` and `-links`, CC0). They are built from
  Wikidata's weekly dump of **2026-09-28** (the copy is from 2026-05-07) and pinned to that release. They are split by
  language and site, so only the EN, FR and `mul` files are read, and the claims are never touched.
  - `mul` is the default label. 10.3M items carry one and no `en` label, so it is read too.
- **The plan, measured from home** (footers and first row groups only):
  - **names** (`r1b_names_plan_2026-10-31.json`): 8 files, 1.36 GB read, 119.4M names (labels, aliases, titles), an
    extract of about 1.54 GB;
  - **redirects** (`r1b_redirects_plan_2026-10-31.json`): enwiki's `page` (2.43 GB) and `redirect` (0.19 GB) tables,
    dump of 2026-10-08; frwiki's (0.49 and 0.03 GB), dump of 2026-10-01. They are streamed and kept only as
    article-space pairs.
  - That is about 4.5 GB through Colab in all, and no raw file is stored. The 10-file stage does not apply at this size;
    the plan cell is the stop.
- **Then, at home:** the VM segment from the extract (names, aliases, titles and redirects to items, as knowledge keys;
  `cubelang knowledge build`), copied to the SSD; R1b scored against its lines; R7's sizes.
- **The source (Cowork, 2026-10-31):** Claude Code's change is accepted. The source is the `wikidata-pq` tables
  (`permutans/wikidata-labels`, `-aliases`, `-links`; CC0; Wikidata's dump of 2026-09-28): newer than the 959 GB copy,
  split by language, so only EN, FR and `mul` are read (1.36 GB, 119.4M names, an extract of about 1.5 GB).
  - They are a third party's derivation, so check them before scoring: 300 items drawn by seed 0. Their EN and FR
    labels, aliases and enwiki/frwiki titles are compared with live Wikidata (by id, through the source's cache).
    Report every difference.
  - Gate: at most 1% of items differ, beyond edits made after 2026-09-28 (Wikidata's revision dates say which).
    Otherwise the extract is rebuilt from Wikidata's own dump.

## R2 on dev: tiers, BM25, B1 (Claude Code, 2026-10-31)
- **Logs:** `retrieval_R2_dev_2026-10-31.json` (the full run: the fame set and the dev rows of Mintaka EN, WebQSP_EL
  and ShadowLink, 300 questions a set; 2,965 rows; 4.3 h, mostly fetching 43k item profiles once) and
  `retrieval_R2_dev_2026-10-31_el_meta.json` (the entity-linking rows again, after the fix below). The lines are read
  off each log by `validation/retrieval_bench/r2_gates.py`. **Test is not read**: R2 has no margin from dev to carry,
  and R2's other arms (qFHRR on grilly2, the usage table) are still to run. Test stays unread for R2's end.
- **A harness fault, fixed before reading:** today's source drops Wikimedia meta-pages (disambiguation, list,
  category) by their description before any tier (`wikidata_resolve`, 2026-09-14). R2's arms did not, so a
  disambiguation page could be picked ("Battle of Yorktown", "Ford Freeway", "Gravity Probe"). Every arm now drops
  them in `bearing` (test added). The fame rows cannot change: there the relation is known, and a meta-page carries
  none. Only the entity-linking rows were rerun.

**The lines, dev** (B1 runs only where the relation is known, i.e. on the fame set):

| Arm | Bare: single picks | Qualified: wrong, resolved | ShadowLink: Top, Shadow (gap) | Mintaka: wrong / decided | WebQSP_EL: wrong / decided |
|---|---|---|---|---|---|
| tiers (today's) | 6 | 2, 0.2% | 0%, 14.3% (14.3) | 25 / 105 | 13 / 87 |
| B1 | 2 | 0, 0% | not run | not run | not run |
| BM25, margin 0.25 | 66 | 11, 87.0% | 23.7%, 39.6% (15.9) | 58 / 236 | 31 / 161 |
| BM25, 0.5 | 64 | 4, 82.4% | 21.4%, 36.1% (14.7) | 45 / 215 | 25 / 151 |
| BM25, 0.75 | 63 | 2, 73.3% | 16.0%, 36.7% (20.7) | 34 / 195 | 20 / 145 |
| BM25, 0.9 | 63 | 2, 60.8% | 13.6%, 35.7% (22.1) | 31 / 185 | 19 / 144 |
| **Line** | **0 (kill: any)** | **0, >= 70%** | **gap <= 5 (kill: > 10)** | **<= tiers, <= 1 in 50, more decided** (on test) | same |

Rows: 1,000 bare and 910 qualified fame rows (EN and FR); 433 Mintaka EN, 323 WebQSP_EL and 299 ShadowLink mentions
(Top 103, Shadow 108, Tail 88). The gold item is among the candidates for 992 of the 1,055 entity-linking mentions.
ShadowLink's Tail, the rare unshared names, goes well: BM25 gets 76 to 80 of 88 right with none wrong, and tiers 71.
Accuracy is counted among spoken picks; "decided" is picks.

**1. Bare names, the kill line: every arm makes single picks.**
- **BM25: 63 to 66 at every margin. Killed as built.** At 0.9, all 63 come from the template's relation word, the only
  word a bare question leaves: "born" 28, "né" 15, "pays" / "trouve" 10, "country" 6, "mort" 4. It matched one
  namesake's description ("(born 1951)") or values, and that namesake was picked. The question's words name the
  relation, not the item.
- **B1: 2, both "Pilot Corporation" (EN and FR). By the line as written, killed.** In today's Wikidata the American
  company (Q7194412) is "Pilot Company", alias "Pilot Corp."; only the Japanese one (Q1356034) still bears "Pilot
  Corporation". So B1 had one branch, it verified, and B1 answered. The fame set's names are Wikidata5M's (2019), and
  the candidates' names are today's. I did not change the set or the line. Whether a name that today's data no
  longer shares counts against an arm is yours to decide.
- **Tiers (today's): 6, three names in both languages.** "Eleanor of Austria" and "Paz" are picked by the label over
  an alias. "Pilot Corporation" is picked as with B1.

**2. Qualified variants: B1 is never wrong and never resolves; BM25 resolves, never with 0 wrong.**
- **B1: 0 wrong, 0 resolved,** by construction. The fame names were drawn where two or more items carry the relation,
  so both branches verify, and the check alone cannot separate them. Outcomes: once 322 (the namesakes share the
  value, so saying it once is right whichever was meant), each 117, ask 471.
- **BM25:** resolves 61 to 87%, with 2 to 11 wrong; no margin reaches 0.
- **Tiers:** 2 picks, both wrong.

**3. ShadowLink: every arm's gap is over the kill line (10 points).** BM25 is at 14.7 to 22.1, tiers at 14.3, and
Shadow is the one ahead each time. Most of the wrongs are mentions that no name of the gold item equals: 34 of BM25's
37 at 0.9, and 23 of tiers' 26 ("Fest" for a Festschrift, "fixed" for a landline, "plug-in" for a plug-in electric
vehicle). Then any item that does bear the span competes, and if it is alone it is "the only one". The fix moved
tiers' Top from 11 wrong to 13: with the disambiguation page gone, the one real item that bears the span is left and
picked. This is a reading fault (which span is the name; R3), not a choosing one, but the line counts it here.

**4. Mintaka and WebQSP_EL: today's tiers is itself over 1 in 50 on dev,** at 25 wrong of 433 (5.8%) and 13 of 323
(4.0%), 27 and 15 before the fix. BM25 is more wrong than tiers at every margin. For 11 of tiers' 25 Mintaka
wrongs, and 9 of its 13 on WebQSP_EL, the mention is not a name of the gold item. The kinds of wrong:
- **Today's label tier** picks an obscure item whose label is exactly the mention: "Grammy" goes to an episode of a
  sitcom, "Harmonix" to a 1997 convention.
- **BM25** picks the item the question's other words describe, which is often the answer's kind, not the mention's:
  for "the second oldest university located in Oxford", the University of Oxford; for Nebraska's electoral votes,
  its university.
- **Gold labels that are wrong:** "Age of Empire" is linked to Hobsbawm's book (my earlier note had the game series
  as gold: wrong); "Chinese civilization" to the People's Republic; "Battle of Yorktown ... in 1871" to the 1862
  siege.
- **No relation is known on these rows,** so the kind step never runs there. Today's pipeline always has one.

**5. Not covered:** Mintaka FR marks no mention spans (R1 located them with today's labels, which you noted favours
the API), and WebQSP_EL and ShadowLink are English only. So R2's French side is the fame set alone.

**6. B1's cost** (the order's budget line; `_fame_b1`, the host's round trip on the wire, run-proto, this machine):

| Branches k | 1 | 2 | 3 | 4 | 5 | 6 or more |
|---|---|---|---|---|---|---|
| Questions | 2 | 301 | 219 | 157 | 116 | 1,115 |
| p50 / p95, ms | 7.0 / 7.0 | 14.2 / 18.0 | 21.5 / 26.5 | 28.8 / 36.1 | 36.0 / 42.3 | 101.5 / 239.2 |

- **About 7 ms a branch,** linear in k. A bare question on a small segment takes 0.4 ms, so the cost is most likely
  each branch's clone of the store. That is the case for the in-program fork (`VM::snapshot` / `restore`).
- **The median question has 7 branches** (at most 42): a fame name's namesakes, after the bearing and relation steps.
  Up to k = 5, p95 stays under 50 ms (the per-hop line R7 will set). Past that, it does not.

**What I read from it** (yours to decide; nothing changed on these lines):
- Choosing by the question's words alone fails the fame line for a structural reason: once the name is taken out,
  a bare question's only words name the relation.
- B1 never picks wrongly when the names are shared, and it cannot pick when they are qualified. The two do
  different jobs: B1 says which readings hold, and the words separate only among those.
- Three arms are possible next. None is registered, so none is run:
  - **(a) The relation's own words are not cues:** the relation lexicon's EN/FR labels and aliases for the
    question's property are taken out. It would be fitted after seeing the fame set, which has no held-out half, so
    it needs a fresh fame sample (seed 1, same construction) to be measured.
  - **(b) B1, then the words among the branches that verify:** the check first; BM25 or qFHRR only separates
    verified branches.
  - **(c) The fame set's names checked against today's:** a name that only one item bears today is reported apart,
    whatever the line decides about Pilot Corporation.
- **Next on R2:** the qFHRR arm on grilly2, with one prototype per item. Then test, once.

## R2's qFHRR arm, on grilly2 (Claude Code, 2026-10-31)
- **Log:** `retrieval_R2_dev_2026-10-31_qfhrr.json`. Every dev row, with tiers and BM25 rerun beside it; their numbers
  match the two logs above.
- **How it runs (`validation/retrieval_bench/r2_qfhrr.py`):**
  - **Same words as BM25:** the question's cue words and each item's description and values.
  - **qFHRR codes** from the serving word table (fastword v4, through `cubbyllm.worlds.fast_words`). One prototype per
    item and language.
  - **Pick rule:** the cosine, with the best item picked past a rank-normalized margin, as for BM25.
  - **On grilly2:** the cosine runs in grilly2's own interpreter (`r2_qfhrr_score.py`, through
    `uv run --project <grilly2>`), never grilly v1, which this repo's interpreter imports. It runs on `vulkan:0`
    (`grilly.vsa.similarity_batch` over `index_select`).
  - **Size:** 12,628 prototypes (0.52 GB on the card); 2,653 rows scored in 25 s. Rescoring 50 rows in numpy differs by
    at most 1.2e-6.

| qFHRR, margin | Bare: single picks | Qualified: wrong, resolved | ShadowLink: Top, Shadow (gap) | Mintaka: wrong / decided | WebQSP_EL: wrong / decided |
|---|---|---|---|---|---|
| 0.25 | 298 | 29, 77.4% | 17.6%, 28.9% (11.2) | 28 / 212 | 30 / 180 |
| 0.5 | 73 | 3, 47.4% | 15.4%, 26.7% (11.3) | 11 / 121 | 10 / 112 |
| 0.75 | 21 | 0, 12.4% | 16.7%, 11.1% (5.6) | 8 / 92 | 7 / 84 |
| 0.9 | 11 | 0, 1.1% | 0%, 11.1% (11.1) | 8 / 83 | 7 / 80 |
| tiers, for reference | 6 | 2, 0.2% | 0%, 14.3% (14.3) | 25 / 105 | 13 / 87 |

- **Bare names: killed at every margin, for BM25's reason.** At 0.9, the 11 picks come from the template's relation
  words: "pays" / "trouve" 4, "né" 2, and "born", "living", "country", "die" and "mort" once each.
- **Qualified:** 0 wrong only from 0.75, where it resolves 12.4% against the line's 70%.
- **ShadowLink:** the gap is over the kill line (10) at three margins. At 0.75 it is 5.6 points, just over the line,
  on few picks.
- **Mintaka and WebQSP_EL: the first arm better than today's tiers.**
  - At 0.5: fewer wrong (11 against 25, 10 against 13) and more decided (121 against 105, 112 against 87).
  - It is still over 1 in 50: 2.5% and 3.1% of rows.
  - From 0.75: 8 and 7 wrong (1.8% and 2.2%), but fewer decided than tiers.
- **French:** the table's teacher is English. Most French words fall back to their signal code, so in French the arm
  matches only the words as written.
- **Where R2 stands:**
  - Every registered arm has run on dev, except the usage table, which has no anchor counts. None passes, and no
    margin carries to test, so test is unread.
  - **Yours to decide:** whether R2 ends here (test read once, for the record), or goes on with an arm you register
    from (a) to (c) above. The qFHRR scorer would serve (b) as it is: B1 first, then the codes among the branches that
    verify.

## R3 on dev: the relation, given the item (Claude Code, 2026-11-03)
- **Log:** `retrieval_R3_dev_2026-11-03.json` (ids, properties, judgements; no question text). Harness:
  `validation/retrieval_bench/r3_relation.py`, with `r3_gold`, `r3_arms`, `r3_qfhrr` (+ `r3_qfhrr_score`, on grilly2)
  and `r3_450m` (on grilly2).
- **Rows:** 300 a set and language, by seed 0 (QALD FR has 250). The gold item gives every arm one thing only: where
  its name is. Its words are taken out of the question, and the 450M sees them as `[E1: ...]`. No arm reads the
  item's facts.
- **Gold properties, rules fixed in `r3_gold` before any row was scored:**
  - SimpleQuestions-Wikidata: the set's own property.
  - QALD-9-plus: the SPARQL's properties without P31 and P279 (class constraints). A row is scored only when exactly
    one property is left; 82 EN and 69 FR rows have several, and 18 and 15 have none.
  - Mintaka gives answers, not properties. Its gold is the properties that link a question item to an answer item in
    today's Wikidata, either way. 121 of 300 rows have such a link; the other 179 (counts, comparisons, yes/no) are
    counted apart.
- **The arms:**
  - **grammar:** today's host reader (the everyday table), its relation wording through the serving property table.
  - **lexicon:** the frozen lexicon over the words left once the names are out, longest wording first.
    **lexicon_content** is the same without wordings made only of function words ("in", "by", "de", "is a"); 36 EN
    and 20 FR such wordings are in the frozen lexicon. That rule is general and was written before scoring.
  - **qfhrr:** a qFHRR codebook of every property, one code per property and language: 25,368 codes (0.52 GB a
    language on the card, one language at a time), with the margin as in R2. 1,431 rows scored on `vulkan:0` in 17 s;
    numpy agrees to within 2.3e-6.
  - **450m:** the 450M in the programs seat (`base450m_cont` + `cubby450m_v12e_w_step_cont`, greedy) on grilly2. Its
    program's `HOP1` wording goes through the serving property table. About 2.4 s a question.
  - Every arm hands over at most 5 properties. A run of equal scores is taken whole, and a first run longer than 5 is
    "undecided".
- **Found on the way:** on `base450m`'s export, the programs adapter writes no program (PHP classes, URLs). H-E15
  records that it was trained on `base450m_cont`; the harness pins that base.

**Coverage first** (the share of each set's rows the arm reads):

| | SimpleQ EN | QALD EN | QALD FR | Mintaka EN | Mintaka FR |
|---|---|---|---|---|---|
| grammar | 16.7% | 10.3% | 4.4% | 3.7% | 2.0% |
| lexicon | 88.0% | 83.3% | 86.0% | 91.0% | 97.0% |
| lexicon_content | 80.7% | 65.0% | 54.0% | 77.0% | 75.0% |
| qfhrr, margin 0 | 97.7% | 97.7% | 99.2% | 99.0% | 100% |
| qfhrr, margin 0.25 | 10.3% | 16.3% | 31.6% | 15.0% | 24.7% |
| 450m | 53.3% | 42.3% | 28.8% | 27.7% | 21.0% |

**Recall@5, and the wrong rate** (a list without the gold), over the scored rows (300, 200, 166, 121, 121):

| | SimpleQ EN | QALD EN | QALD FR | Mintaka EN | Mintaka FR |
|---|---|---|---|---|---|
| grammar | 0.143 (2.3%) | 0.125 (2.0%) | 0.054 (1.2%) | 0.008 (1.7%) | 0.008 (1.7%) |
| lexicon | 0.367 (22.7%) | 0.260 (34.0%) | 0.211 (54.2%) | 0.198 (41.3%) | 0.157 (56.2%) |
| lexicon_content | 0.493 (23.0%) | 0.375 (28.5%) | 0.253 (35.5%) | 0.314 (47.9%) | 0.198 (57.9%) |
| qfhrr, margin 0 | 0.417 (56.0%) | 0.555 (43.0%) | 0.361 (63.9%) | 0.248 (75.2%) | 0.141 (86.0%) |
| lexicon_content + qfhrr, 0 | 0.557 (33.7%) | 0.500 (47.0%) | 0.343 (65.7%) | 0.314 (63.6%) | 0.207 (79.3%) |
| 450m | 0.350 (15.7%) | 0.290 (20.5%) | 0.121 (18.1%) | 0.124 (15.7%) | 0.041 (16.5%) |
| **gate** | **0.85** | | **0.80** | | |

- **No arm passes. The best are 0.557 on SimpleQuestions and 0.361 on QALD FR, against 0.85 and 0.80.** Taking, on
  each row, any of grammar, lexicon_content or the 450M that hits gives 0.587 on SimpleQuestions and 0.437 on QALD:
  no combination of these arms reaches the line either.
- **The qFHRR arm is not kept.** At no margin does lexicon_content + qfhrr have a pooled wrong rate at or below
  lexicon_content's (34.5%), so dev sets no margin. At margin 0 it adds 6.6 points pooled, at 18 more points of wrong.
- **Today's grammar reads little and reads it well:** 7.5% of rows, and among the rows it reads, 86% hold the gold on
  SimpleQuestions. In French it reads almost nothing (2 to 4%).
- **The 450M reads about half of SimpleQuestions (53%) and less elsewhere: it names many relations in words the
  property table does not hold.** On SimpleQuestions:
  - 132 of 300 programs bind a `HOP1` wording that names no property: "film" 15, "album" 12, "born in $E1" 10,
    "birth" 6, "game" 5. These are the wiki world's relation words, which it was trained on, not Wikidata's labels.
    8 rows got no fact program.
  - When its wording does name properties, it holds the gold in 105 of 160 rows (66%).
  - Most wrong lists are a near property: "death" for place of death (P20) 4, "language" for P364 3, "produced" for
    P162 3.
  - In French the share falls (29% read on QALD FR, 21% on Mintaka FR). French programs more often take 2 or more
    hops (81 of 166 on QALD FR).
- **SimpleQuestions' inverse rows** (73, asked from the object's side) are the 450M's worst: 7 hits, against 23 for
  lexicon_content.
- **Not run:** test. No arm passes on dev and no margin was set, so, as for R2, test stays unread until you say.
- **What this says about reading** (for your decision, not a rule written here):
  - The relation is the weakest stage measured so far.
  - The 450M's proposal is useful only once its wording reaches a property, a mapping the frozen lexicon does not
    give. A programs adapter trained on Wikidata's property labels (the frozen lexicon's wordings as targets) is the
    lever this run points to. That would be new training, so it needs its own order and your yes.
  - The exact lexicon's misses are the questions' own phrasings ("Who directed X" against "director"); grammar reads
    too little to matter.

## Cowork's check of R2, R3 and R1b's source check (2026-11-04)
Read: the three reports, both R2 tables, `r2_gates.py`'s lines against the order's, and R3's gold rules.

**R1b's source check passes** (300 of 300 identical; gate at most 1%). The extract stands as R1b's names. The sharded
segment (32 shards opened as one store, `knowledge batch`) is what R1b and R7 measure, and R7 reports it as served,
shards included. Next as you say: R1b scored on R1's own mentions.

**R2: test stays unread.** BM25 and qFHRR, as built, are killed on the bare-name line and are not run again as
choosers on their own. Three decisions, then one new arm:
1. **(c), yes, as a rule on the set:** a bare-name row whose name only one item bears in R1b's names (today's) is
   out of the bare-name line, for every arm, and reported apart with its count. The fame set tests shared names; a
   name no longer shared tests nothing. On seed 0 this takes out the Pilot Corporation rows and whatever else it
   finds. Report the line both ways.
2. **(a), yes, as one general rule, span accounting:** each word of the question serves one role: the name, the
   relation, or a cue. The relation lexicon's EN and FR labels and aliases for the question's property, and function
   words, are never cues. It is written here, before seed 1 exists.
3. **(b), changed: the qualifier is checked, not scored.** The construction defines a qualifier as "one discriminator
   from one item's own facts" (its occupation, a place it is in), so it can be verified like the relation.
   - **B1q:** in each branch, the cue words are looked up as names in R1b's segment (exact, then typo), giving a set
     of value items. The branch holds the qualifier if one of its item's statements has one of them as its value, in
     either direction. A branch stands if it verifies both the relation and the qualifier; the outcome table decides
     as before. No margin, nothing ranked.
   - **Without a relation** (ShadowLink, and entity-linking rows that have none), the same check runs on the context's
     cue words alone: a candidate stands if one of its statements holds a value the context names. One standing is a
     pick; otherwise undecided.
   - **Your (b), for comparison:** qFHRR among the branches that verify, past a margin set on seed 0. It ranks, so
     "branches are never ranked" stands unless its numbers make the case against B1q.
   - **In plain words:** asked about "Paz the singer", Cubby checks "singer" against what it knows of each Paz, and
     only the Paz it fits stands.
   - **Over a conventional model:** a model weighs word overlap, so "born" in the question can tip the choice. Here
     the detail is a fact that holds or does not.
4. **The fresh fame sample:** seed 1, the same construction, with (c) applied before it is frozen, and its SHA-256
   committed before any arm runs. Seed 0 is dev (the comparison arm's margin); seed 1 is read once.

**ShadowLink: the line is read where the gold bears the mention.** Most wrongs are mentions that no name of the gold
equals ("Fest", "fixed", "plug-in"): finding (R1b), not choosing. Which rows those are is decided by R1b's names
(labels, aliases, titles, redirects), never by an arm's output. The other rows are counted under R1b's misses, and
all rows are still reported.
- **Please label the cells:** tiers shows Top 0% beside "Top from 11 wrong to 13", and the text says Shadow is ahead
  each time, where qFHRR at 0.75 has Top ahead. Say whether the percentages are accuracy or wrong share among spoken
  picks, and give the counts (spoken, right, wrong) per group.

**Mintaka and WebQSP_EL: give the B arms their relation.** Today's pipeline always has a relation; these rows ran
without one, so the check never ran. Where R3's gold rules give one property (Mintaka's 121 linked rows, WebQSP's
SPARQL), run B1 and B1q with it, reported as "relation given". The other rows run the arms that need none. The line
is unchanged.

**B1's cost triggers the CubeLang task this order pre-registered:** p95 is 239 ms past 5 branches, and the fame set's
median is 7. Claude Code builds the in-program fork on `cowork/vm-contracts`: a branch shares the base store
read-only and keeps only its own exclusions and assumptions, or `VM::snapshot` / `restore` (bytes 0x3E and 0x3F),
whichever measures cheaper. Line: p95 at most 50 ms a question at k = 7, with the cost per added branch reported up
to k = 42.

**R3: the relation is chosen among the item's own properties first; the adapter waits.** The qFHRR arm is not kept,
as the report says.
- **R3o, a new arm:** the candidates are the properties the gold item carries, outgoing and incoming (for the inverse
  rows), from its statements fetched by id through the source's cache, with the fetch dated (R4 fetches the same).
  Each wording (lexicon_content, the 450M's `HOP1` wording, qFHRR) is scored only among them. It is the check step
  applied to relations: a property the item does not carry cannot verify.
  - **The 450M's wordings** ("film", "album", "born in $E1") are matched to the item's own property labels and
    aliases the same way. This may place most of its 132 unmapped programs without training.
  - **Reported apart:** the rows where the gold property is not among the item's properties today (a missing fact),
    with what the arm hands over there. That is where a near property would become a wrong answer, and R8's kill
    line reads it.
  - **Lines:** R3's gate as written (0.85 on SimpleQuestions-Wikidata, 0.80 on QALD-9-plus FR), read on the rows
    where the gold is held and on all rows; the wrong rate reported beside lexicon_content's.
  - **In plain words:** to work out what you are asking about something, Cubby considers only what it actually knows
    about that thing.
  - **Over a conventional model:** a model can name a relation the item does not have and answer anyway. Here the
    choice is bounded by the facts on record.
- **The programs adapter trained on Wikidata's property labels:** only if R3o fails, as its own order with Nick's yes.
- **`base450m` against `base450m_cont`:** please check which base the serving path loads with the programs adapter,
  and report it. If serving pairs the adapter with `base450m`, the programs seat writes nothing there today.
- Test stays unread until R3o has run on dev.

**Order of work:** R1b's scoring, then R3o (the weakest stage), then the seed-1 fame set with B1q and the fork task,
then R4. Nothing here reaches the serving path. Status: open.

## R1b on dev: the VM's names segment (Claude Code, 2026-11-04)
- **Logs:**
  - `r1b_names_check_2026-11-03.json`: the source check.
  - `retrieval_R1b_dev_2026-11-03.json`: the scoring, on R1's own 6,373 mentions and 1,204 typo variants, checked
    to be R1's by key and gold item. The API's ranks are R1's (2026-10-28), not asked again.
- **The source check passes:** 300 items drawn by seed 0 from 61,789,981, all 300 identical to live Wikidata. 580
  names were compared; 28 of the items had been edited after the dump, none in its names.
- **The segment** (`validation/retrieval_sets/names_segment.py`; manifest beside the shards, off-repo):
  - Every EN, FR and `mul` label, EN and FR alias, enwiki and frwiki title, and EN and FR redirect, as CubeLang
    knowledge keys. One fact is (name, item), with the name's kinds as its source.
  - **Counts:** 119.4M names and 12.5M redirects (1.32M section redirects left out, 15,221 with no item) gave
    92,390,110 facts. That is 5.65 GB in 32 shards on the SSD, built in 25 minutes.
  - **CubeLang** (local branch `cowork/vm-contracts`, `5779db3`): `knowledge build` holds a store in memory, which
    the whole does not fit, so it is built as shards. A directory of segments now opens as one store, each shard a
    layer read through its own index. `knowledge batch` answers many mentions with the same lookup and `fuzzy` a
    program's QUERY runs. Every CubeLang test passes (210 in the library, the rest in its suites).
- **The arms:**
  - `vm_exact`: the items under the mention's key.
  - `vm_edit1`: + those 1 edit away.
  - `vm_edit2`: + those 2 edits away.
  - Each distance is one unordered group, entering a shortlist whole or not at all (R1's rule). A group the typo
    lookup's limit (100 keys) may have cut is left out.

**Recall@20:**

| | API | R1's best local | vm_exact | vm_edit2 | vm_edit2 on typos (API) |
|---|---|---|---|---|---|
| Mintaka EN | 0.866 | 0.877 | 0.625 | 0.654 | 0.496 (0.036) |
| Mintaka FR | 0.981 | 0.759 | 0.775 | 0.777 | 0.559 (0.052) |
| WebQSP_EL | 0.874 | 0.931 | 0.657 | 0.662 | 0.485 (0.041) |
| GraphQ_EL | 0.816 | 0.863 | 0.729 | 0.738 | 0.480 (0.013) |
| ShadowLink | 0.707 | 0.731 | 0.534 | 0.542 | 0.440 (0.033) |
| AmbER | 0.958 | 0.886 | 0.872 | 0.872 | 0.610 (0.0) |
| QALD EN / FR | 0.990 / 0.996 | 0.915 / 0.681 | 0.664 / 0.721 | 0.664 / 0.729 | 0.453 / 0.476 |
| SimpleQ | 0.967 | 0.952 | 0.902 | 0.902 | 0.698 (0.052) |
| CronQuestions | 0.984 | 0.724 | 0.936 | 0.936 | 0.720 (0.0) |

- **Killed by its own line:** below the API on Mintaka EN and FR. No gate passes, except "within 5 points" on
  CronQuestions.
- **Why, measured. The names are there; the namesake sets are too large for 20 with no ranking:**

  | | gold under its exact name | within 2 edits | in an exact set of more than 20 items | under no name within 2 edits |
  |---|---|---|---|---|
  | Mintaka EN | 87.4% | 92.2% | 24.9% | 7.8% |
  | Mintaka FR | 99.6% | 99.8% | 22.1% | 0.2% |
  | WebQSP_EL | 91.8% | 93.3% | 26.2% | 6.7% |
  | QALD EN | 100% | 100% | 33.6% | 0% |
  | ShadowLink | 56.6% | 58.0% | 3.2% | 42.0% |

  - **The full name space brings many more namesakes than Wikidata5M:** 420 of Mintaka EN's 1,453 mentions name
    over 20 items (median 7). The API orders such a set by its own popularity signal, which this plan forbids
    within a kind.
  - At @100 the gap mostly closes: Mintaka EN 0.834 against 0.875, FR 0.966 against 0.985, WebQSP_EL 0.887 against
    0.879, GraphQ_EL 0.888 against 0.830.
  - ShadowLink's misses are names no label, alias or redirect spells. That is a reading question, not a coverage
    one.
- **Typos:** the typo lookup finds 44 to 72% of misspelt mentions where the API finds 0 to 5%.
- **Speed** (a first R7 reading, 92.4M facts, one process, the machine above):
  - The store opens in 5.0 s.
  - An exact lookup takes p50 43 µs, p95 76 µs.
  - A 2-edit typo lookup across the 32 shards takes **p50 435 ms, p95 589 ms**, over R7's 50 ms line by an order
    of magnitude.
- **For your decision** (nothing changed after the kill):
  - (a) Score finding as the candidate set, not a shortlist: recall at any size (Mintaka FR 99.8%, EN 92.2%), with
    the set's size reported. The choose stage then takes the whole set, the relation and the kind dropping what
    cannot carry it, as R2's arms already do. That would be a new line, so it is yours to write.
  - (b) Bound the set inside a kind by something other than fame.
  - (c) Move the typo lookup to one index, not 32 shards, before R7.

## R3o on dev: the relation among the item's own properties (Claude Code, 2026-11-04)
- **Log:** `retrieval_R3o_dev_2026-11-04.json`. Harness: `validation/retrieval_bench/r3o.py` and `r3o_props.py`.
  Rows, gold rules and the 450M's programs are R3's (the programs were read back, not regenerated).
- **The candidates:** for each item, its outgoing properties (from its statements, by id, through the source's cache)
  and its incoming ones (`SELECT DISTINCT ?p WHERE { ?s ?p wd:Q }` at query.wikidata.org).
  - 952 items, fetched 2026-11-03 and 2026-11-04. For 4 items the service could not answer (very common values);
    those are recorded as unavailable, never as "none".
  - A row's items are the ones its question names (located), else all of them. That keeps QALD's class constraints
    (Q5, "human") out.
  - **The sets are large**, so choosing 5 among them is not free. Median candidates: SimpleQuestions 39, QALD about
    190, Mintaka about 175. Five at random among them would score 0.149 on SimpleQuestions and 0.04-0.08 elsewhere.
- **Every arm scores only among the candidates.**
  - The 450M's `HOP1` wording goes through the serving property table (the whole wording), and through the frozen
    lexicon's wordings found inside it.
  - qFHRR scores against the candidates' codes only, on grilly2 (`index_select`; numpy agrees to within 1.4e-6).
  - `lexicon_content>450m` is the lexicon, and the 450M only where the lexicon reads nothing.
- **Held:** on 870 of 908 scored rows the items carry the gold property today. On 38 they do not (a missing fact):
  SimpleQuestions 2, QALD EN 17, QALD FR 12, Mintaka 1 and 6.

**Recall@5 (wrong rate), on all scored rows:**

| | SimpleQ EN | QALD EN | QALD FR | Mintaka EN | Mintaka FR |
|---|---|---|---|---|---|
| lexicon_content | 0.563 (4.0%) | 0.360 (17.5%) | 0.217 (15.1%) | 0.331 (28.1%) | 0.190 (21.5%) |
| 450m | 0.470 (6.3%) | 0.315 (11.0%) | 0.115 (9.6%) | 0.182 (15.7%) | 0.066 (7.4%) |
| lexicon_content>450m | 0.603 (6.3%) | 0.400 (21.0%) | 0.253 (19.9%) | 0.331 (28.9%) | 0.198 (25.6%) |
| grammar | 0.140 (0.3%) | 0.120 (1.0%) | 0.054 (0.6%) | 0.008 (0%) | 0.008 (0%) |
| **qfhrr, margin 0** | **0.887 (9.0%)** | 0.755 (22.5%) | **0.536 (45.8%)** | 0.554 (44.6%) | 0.397 (60.3%) |
| qfhrr, margin 0.1 | 0.747 (6.0%) | 0.580 (17.5%) | 0.476 (32.5%) | 0.421 (26.5%) | 0.314 (42.1%) |
| qfhrr, margin 0.25 | 0.573 (3.0%) | 0.400 (9.5%) | 0.349 (15.1%) | 0.223 (14.1%) | 0.223 (27.3%) |
| R3, for reference: best | 0.557 | 0.555 | 0.361 | 0.314 | 0.207 |
| **gate** | **0.85** | | **0.80** | | |

On the held rows the numbers are within 1-7 points of these: qfhrr at margin 0 holds 0.893 on SimpleQuestions and
0.578 on QALD FR.

- **No arm passes both lines.**
  - qFHRR at margin 0 passes SimpleQuestions: 0.887 on all rows, 0.893 on held ones, against chance's 0.149.
  - It fails QALD FR: 0.536, or 0.578 on held rows, against 0.80.
  - The French gap is R2's: the word table is English, so French words match only as written.
- **What the item's properties change, against R3:**
  - **The wrong rate collapses:** lexicon_content on SimpleQuestions goes from 23.0% to 4.0%. Most of R3's wrong
    lists named properties the item does not carry.
  - **qFHRR goes from not kept to the strongest arm** (0.417 to 0.887 on SimpleQuestions, 0.361 to 0.536 on QALD FR).
    Among an item's 40 or 200 properties, the codebook's ranking carries signal it could not carry among 12,684.
  - **The 450M's wordings** gain (SimpleQuestions recall 0.350 to 0.470, wrong 15.7% to 6.3%), but its coverage
    does not move (53%). The unmapped wordings ("film", "album", "birth") name no property even among the item's own,
    so the item's properties do not place most of the 132.
- **The missing-fact rows (38), what each arm hands over there:** qfhrr at margin 0 hands over a near property on 36,
  at 0.25 on 16; lexicon_content on 6; the 450M on 4; grammar on 0. These are the rows where a near property would
  become a wrong answer (R8's kill line).
- **Margins:** none was set. R3's rule (no higher pooled wrong rate than lexicon_content's) is not met by any margin.
  At margin 0.25 qfhrr's pooled wrong rate is 11.3% against lexicon_content's 14.5%, with recall 0.401 against
  0.374. That would be the reading if the rule applied to the candidate-bounded arm; it is yours to say.
- **Which base serving pairs with the programs adapter (your question):** nothing in the repo pins it. The server
  takes it on the command line (`python -m cubby_serve --cubby450m <export> --emitter-adapter <dir>`). The recorded
  live run (H-E15, 2026-10-17) passed `base450m_cont` for the programs seat and `base450m` only as `--talk-export`,
  so that pairing is right. Nothing refuses the wrong one, though: on `base450m` the programs seat writes no program.
  A guard (the adapter recording its base's hash, checked at load) would be a serving change, so it is an order.
- **Test not read.** Dev sets no margin and no arm passes both lines.

## Cowork's check of R1b and R3o (2026-11-04, later)
Read: both reports and their tables, against the lines.

**R1b: killed by its own line, and the record keeps it** (below the API on Mintaka EN and FR at @20).
- **What it found:** the names are there. The gold is under its exact name for 87.4% of Mintaka EN mentions and 99.6%
  of FR. But a fifth to a third of mentions name more than 20 items, and an unranked set enters a 20 shortlist whole or
  not at all. The API orders those sets by popularity, which this plan forbids within a kind.
- **(a) becomes a new arm, R1c: finding is the candidate set.** Its line is written here and read once, on Mintaka's
  test split (EN and FR), which no arm has read:
  - the gold within the set (exact, then up to 2 edits; names and redirects) at least as often as within the API's 20;
  - the set's size reported (p50, p95, max);
  - on the rows where R3's gold rules give one property, the set after the relation filter (the items that carry
    it): its size reported, and the gold kept whenever it carries the relation.

  The set's size is the cost choosing pays (one branch per item), so the fork task and the typo index below come
  before any serving.
- **(b) is choosing's job, not finding's:** the relation (B1), the qualifier (B1q) and the kind (B2) bound the set. No
  arm here.
- **(c), yes:** one typo index over all 32 shards, in CubeLang on `cowork/vm-contracts`, built with the fork task. The
  typo lookup runs only when the exact lookup finds nothing. Line: R7's, p95 at most 50 ms.
- **ShadowLink's 42% of names that nothing spells** are reading, as R2's check said, and its choosing line is read
  where the gold bears the mention.

**R3o: SimpleQuestions passes, QALD FR does not, so the arm does not pass.**
- **What it settles:** bounding the candidates by the item's own properties is right. The wrong rate collapses
  (lexicon_content on SimpleQuestions, 23% to 4%), and qFHRR goes from not kept to 0.887 (0.893 on held rows), against
  chance's 0.149.
- **The margin, by R3's rule as written:** the qFHRR arm is kept only if it adds at least 3 points at no higher wrong
  rate. At 0.25 it adds 2.7 (0.401 against 0.374) at a lower wrong rate (11.3% against 14.5%). Not met: no margin is
  set and qFHRR is not kept. The rule is not rewritten after the numbers.
- **The missing-fact rows are the danger:** at margin 0, qFHRR hands a near property over on 36 of 38. That is a wrong
  answer built on a checked fact, which the VM cannot catch, because the near property is on record.
- **French is now the common gap (R2, R3):** the word table's teacher is English, so French words match only as
  written. Recorded: a French-aware word table is the lever, after R4.

**R3k, a new arm: the kind the question asks bounds the properties further.**
- **The asked kind** comes from a closed list of question words per language, written here before any row is read:
  who / qui gives a person; where / où a place; when / quand / what year / en quelle année a time; how many / combien
  a quantity; anything else gives no kind.
- **Only the candidate properties whose values on this item are of that kind stay** (by the value's `instance of`, as
  0047's class check reads it). Each arm then scores as in R3o.
- **The margin rule, written now:** the smallest margin at which both the pooled wrong rate and the missing-fact
  handovers are at most lexicon_content's, on the same rows and candidates. Recall is read at that margin, against
  R3's gate as written.
- **In plain words:** "where was she born?" asks for a place, so Cubby only weighs what it knows about her that is a
  place.
- **Over a conventional model:** a model can answer a "where" with a date or a person when its guess slips. Here the
  question word bounds what can be said.

**The programs adapter on Wikidata's property labels:** the 450M's unmapped wordings name no property even among the
item's own, so the adapter stays the lever for the programs seat. It is new training, so it waits on Nick's yes, as
its own order. Adapters train in minutes on Colab, so it is cheap.

**The guard:** serving pairs the programs adapter with `base450m_cont` correctly today, but nothing refuses the wrong
base. The guard (the adapter records its base's hash, checked at load) goes with the adapter's order.

**Order of work:** the seed-1 fame set with B1q and the fork task; R1c on test; R3k; the typo index, with the fork;
then R4. Test stays unread for R2, R3o and R3k until their dev lines say otherwise. Status: open.

## The fork task: B1's cost on the layered store (Claude Code, 2026-11-04)
- **Log:** `retrieval_R2_dev_2026-11-04_fame_b1_layered.json`: B1 on the fame set (seed 0, 1,910 rows), the same harness,
  run on CubeLang's `cowork/vm-contracts` binary (`CUBELANG_EXE`).
- **Where the 7 ms a branch went:** R2's B1 ran on the main CubeLang binary, which copies the whole knowledge store
  into every request, so a branch cost grew with the store.
  - The layered store (`docs/STORE.md` there) is the fork this task describes: a request shares the frozen base
    read-only through a pointer and keeps only its own exclusions and assumptions in its top layer.
  - Measured on one request and a 2,000-fact store: a query costs 0.76 ms on the main binary and 0.38 ms on the
    layered one. A trivial program with no store costs 0.24 ms, the round trip.
- **The same 1,910 outcomes,** every row identical to `retrieval_R2_dev_2026-10-31_fame_b1.json`.

| branches k | main binary, p95 | layered store, p50 / p95 |
|---|---|---|
| 2 | 18.0 ms | 0.75 / 1.12 ms |
| 5 | 42.3 ms | 1.92 / 2.68 ms |
| **7** | in "6+": 239 ms | **2.57 / 3.28 ms** |
| 10 | | 3.74 / 4.62 ms |
| 20 | | 7.43 / 9.63 ms |
| 42 | | 15.3 / 15.5 ms |

- **The line is met:** p95 3.3 ms a question at k = 7, against 50 ms. The cost per added branch is 0.38 ms (the median
  of wall/k), flat up to k = 42.
- **No new opcode or snapshot was needed.** `VM::snapshot` / `restore` (bytes 0x3E and 0x3F) would only take the
  round trip out of each branch, at most 0.24 ms of the 0.38. Nothing at this scale asks for it; B3's fork of the
  trunk is a different cost.
- **What serving runs:** `cubbyllm.bridges.cubelang_client` finds the main CubeLang checkout's binary by default, the
  copying store. The layered store lives on `cowork/vm-contracts`, which has no remote. Merging it into CubeLang's
  main branch is a decision for you and Nick. Until then, the experiments set `CUBELANG_EXE`.

## R1c on test: finding is the candidate set (Claude Code, 2026-11-04)
- **Log:** `retrieval_R1c_test_2026-11-04.json`, read once, on Mintaka's test split (R1's own mentions: 1,475 EN, 1,159
  FR, checked by key and gold item; the API's ranks are R1's of 2026-10-28). Harness: `validation/retrieval_bench/r1c.py`.
- **The set:** the items under the mention's name in R1b's segment, exact. Only on an exact miss, the items up to 2 edits
  away (EN: 1,398 exact, 62 by the typo lookup, 15 found nothing; FR: all 1,159 exact).

| | gold in the set | gold in the API's 20 | line | set size p50 / p95 / max |
|---|---|---|---|---|
| Mintaka EN | **0.911** | 0.862 | holds | 9 / 100 / 880 |
| Mintaka FR | **0.994** | 0.971 | holds | 6 / 76 / 374 |

- **The relation filter,** on the rows where R3's gold rules give one property (EN 247 rows, FR 176): the set keeps the
  items that carry it, from their statements by id.
  - **The size falls to p50 1, p95 38 (EN) and 31 (FR),** max 140 and 96.
  - The gold is kept whenever it is in the set and carries the relation, by construction. Of the rows where the gold
    carries it (EN 98, FR 73), the 9 EN and 1 FR not kept are rows whose gold is not in the set: finding misses.
- **The line holds in both languages:** the set holds the gold more often than the API's 20 does.
- **What it costs choosing:** the set itself (p95 100 items in EN), or after the relation p95 38. On the layered store a
  branch costs 0.38 ms, so p95 is about 38 ms and 15 ms. One typo lookup took the cut at its 100-key limit.

## B1q on dev, seed 0 (Claude Code, 2026-11-04)
- **Logs:** `retrieval_R2_dev_2026-11-04_b1q.json` (the run reported here) and `retrieval_R2_dev_2026-11-04_b1q_first.json`
  (the first run, kept: two bugs in my cue code, below). Harness: `r2_choosing.py --b1-qfhrr 0.25 0.5 0.75 0.9`, with
  `r2_b1q.py` (the check) and `r2_b1q_prep.py` (cues, the relation given, the qualifier facts). Layered store
  (`CUBELANG_EXE`). Rules (a), (b) and (c) as written in your check; nothing tuned on these rows.
- **The first run's two bugs,** both in my span accounting, not in the rule:
  - the name took every occurrence of its words, so "Dhaka" in "Dhaka District" was no cue;
  - every sub-span of a cue run counted, so "terre" and "cap" alone found items.

  Fixed: the name is one span (the first occurrence of its longest name), and a cue run is looked up only by its
  longest span that names items. French questions name the item in French, so every label of the row's items is the
  name. Tests pin both. The first run had 5 wrong qualified picks, the fixed run 3.

**The fame set (seed 0, 1,910 rows), B1q against today's tiers and B1:**

| | bare: single picks | qualified: right | wrong | undecided | refused | resolved |
|---|---|---|---|---|---|---|
| tiers | 6 | 0 | 2 | 908 | 0 | 0.002 |
| B1 | 2 | 0 | 0 | 910 | 0 | 0 |
| **B1q** | **2** | **692** | **3** | 61 | 154 | **0.764** |
| B1 + qFHRR @0.75 (your (b)) | 35 | 173 | 0 | 737 | 0 | 0.190 |
| B1 + qFHRR @0.9 | 21 | 23 | 0 | 887 | 0 | 0.025 |

- **Rule (c), both ways:** 14 bare rows have a name one item bears today. B1q's 2 single picks:
  - *Pilot Corporation* (EN): one bearer today, the store has 1 item under the name. It is in "one bearer today", out of
    the line by (c).
  - *Tim Brown* (FR, "Où est né Tim Brown ?", relation P19): shared today. "né" stays a cue under rule (a) as written,
    because the lexicon's French wordings for P19 do not include it. Folded to "ne", it names Nebraska (alias "NE"),
    and one Tim Brown, Q2433818, was born in Nebraska, so that branch alone holds the "qualifier". **On the
    shared-today line, B1q makes 1 single pick, so the line fails by one row.** I have not changed (a) after the
    numbers. Whether a past participle like "né" counts as a relation word, or a cue must be longer than 2 letters,
    is yours to say.
- **The 3 wrong qualified picks are one case:** the gold no longer states the qualifier itself.
  - *Rosebank (Western Cape)*, EN and FR: the gold Q7368057 is `located in` Cape Town, and Cape Town is in the Western
    Cape. Another Rosebank, Q7368068, states the Western Cape directly, so B1q picks it.
  - *Signal Hill (Terre-Neuve-et-Labrador)*, FR: the gold Q245680 is in St. John's. Q61004535 states the province.
  - The fame set took the discriminator from the gold's statements when it was built, and today the gold's own
    `located in` is a level lower.
  - Checking the qualifier along the `located in` chain (0047's walk, `CB_CLASS_WALK`) would make both namesakes hold
    in both cases: undecided, not right. That is a rule change and not mine to make; the record keeps the 3 as wrong.
- **Your (b), qFHRR among the branches, has no margin:** the rule asks 0 bare single picks and 0 wrong qualified on
  seed 0. No margin tried gives 0 single picks (21 at 0.9). B1q resolves 4 times as much as qFHRR @0.75 does.

**ShadowLink, the cells labelled.** The percentage is right / (right + wrong), over spoken picks only; refused and
undecided rows are not in it. Counts are spoken (right, wrong):

| | rows | gold bears the mention | tiers | B1q |
|---|---|---|---|---|
| Top, bears | 29 | 29 | 2 (0, 2): 0% | 10 (8, 2): 80% |
| Shadow, bears | 38 | 37 | 2 (2, 0): 100% | 13 (11, 2): 85% |
| Tail, bears | 88 | 87 | 72 (71, 1): 99% | 36 (36, 0): 100% |
| Top, does not bear | 74 | 54 | 13 (0, 13) | 13 (0, 13) |
| Shadow, does not bear | 70 | 55 | 12 (0, 12) | 14 (2, 12) |

- **The line, where the gold bears the mention:** B1q's gap is Top 80% against Shadow 85%, 4.6 points, under the kill
  line (10). On all rows it is 34.8% against 48.1%, 13.4 points; there the wrongs are the rows no name of the gold
  spells, as your check said.
- **B1q speaks half as often as tiers on Tail** (36 against 72). It needs a cue that holds, and Tail rows rarely
  carry one. It is never wrong there.

**Mintaka and WebQSP_EL** (wrong per row; the line is at most 1 in 50):

| | Mintaka EN, 433 rows: right / wrong | wrong per row | WebQSP_EL, 323 rows: right / wrong | wrong per row |
|---|---|---|---|---|
| tiers | 80 / 25 | 5.8% | 74 / 13 | 4.0% |
| B1q, cues alone | 64 / 10 | 2.3% | 28 / 7 | 2.2% |
| B1, relation given | 13 / 5 | 1.2% | (no relation) | |
| B1q, relation given | 6 / 3 | 0.7% | (no relation) | |

- **Relation given:** for Mintaka, where R3's gold rules give one property (69 of the 433 rows). The other rows run
  the cue check alone.
- **WebQSP_EL has no Wikidata property:** its SPARQL is Freebase's, so there is nothing to give, and its rows run the
  cue check alone.
- **What the relation-given wrongs are:** Mintaka's complex questions, where the one property R3's rules find links
  the mention to the answer without being what is asked. Examples: "What city is called the Big Apple?" (P460, picks
  New York City, gold the nickname's item); "the longest Star Wars movie" (P8345); "the highest-grossing actor"
  (P106). The relation given is the rule's, not the question's.
- **B1q halves tiers' wrong rate on both sets and still misses the line** (2.3% and 2.2% against 2%).

**B1q's cost** (every row, the same layered store as the fork task): the check adds the cue lookups and the
qualifier's facts to each branch. p50 / p95 by branches: 2: 1.4 / 6.2 ms; 4: 2.8 / 21.9 ms; 6 or more: 10.2 / 40.6 ms.
That is under the 50 ms line, but near it past 6 branches. B1 alone, the same run: 6 or more 4.9 / 11.7 ms.

**Next:** seed 1, read once with every arm, after your word on "né" (a) and on the qualifier along `located in`.

## R3k on dev: the kind asked, the "one step" reading (Claude Code, 2026-11-04)
- **Log:** `retrieval_R3o_dev_2026-11-04_r3k_one.json`. Harness: `r3o.py --kind one` with `r3k_kind.py`. The question
  word opens the question (after an optional preposition), so a relative "qui" or the conjunction "ou" asks nothing.
  Tests pin the list.
- **The kind's classes:** the items Wikidata labels "person" / "personne" and "place" / "lieu" that have a `subclass
  of` of their own:
  - person: Q215627, Q5;
  - place: Q13526752, Q17334923, Q2221906, Q82794, Q98929991.
- **Rows asking a kind:** 513, of which 305 are scored (gold): person 189, quantity 36, place 64, time 16.
- **margin_r3k: none.** No qFHRR margin has a pooled wrong rate at or below lexicon_content's (9.9%): 28.9% at 0, 21.6%
  at 0.1, 13.4% at 0.25. The gate fails on QALD FR for every arm; at margin 0, SimpleQuestions held is 0.892 and QALD
  FR held 0.559 (R3o: 0.893 and 0.536).
- **Why: the kind drops the gold.** Of the rows whose gold the item carried before the kind, the kind keeps it out:

| kind | scored | gold held before | dropped by the kind | gold incoming only | gold outgoing |
|---|---|---|---|---|---|
| person | 189 | 182 | 76 | 51 | 25 |
| place | 64 | 64 | 55 | 0 | 55 |
| quantity | 36 | 36 | 20 | 9 | 11 |
| time | 16 | 16 | 0 | | |

  - **Person, 51 incoming:** "who was born in Paris?" asks from the value's side. The gold, P19, is an incoming property
    of Paris, and the rule as written keeps incoming properties only when no kind is asked.
  - **Place, 55 outgoing:** one step above `instance of` rarely reaches "place" (city, then human settlement, then
    ...). The chain reading is for this.
  - **Quantity, 11 outgoing:** "how many children" asks a count of an item-valued property (P40), not a quantity
    datatype.
  - **Time:** the datatype holds every time.
- **The chain reading runs next,** after PIAF's link queries finish (both use the query service). It tests the place
  rows. The incoming and count rows are the rule's to change, not the reading's.

## R3k on dev: the "chain" reading (Claude Code, 2026-11-04)
- **Log:** `retrieval_R3o_dev_2026-11-04_r3k_chain.json`. The values' `instance of` then any number of `subclass of`
  steps (one SPARQL query a row; 0 left unanswered). Same rows, classes and margin rule as the one-step reading.
- **The chain fixes "place":** the kind drops the gold on 5 place rows, not 55. In all, 99 golds are dropped (one step:
  151): person 74, quantity 20, place 5.
- **margin_r3k: none.** All scored rows (908), recall@5 / wrong rate / missing-fact handovers:

| | R3o (no kind) | R3k, chain |
|---|---|---|
| lexicon_content | 0.374 / 14.5% / 6 | 0.350 / 9.9% / 8 |
| qfhrr @0 | 0.684 / 30.3% / 36 | 0.609 / 28.9% / 83 |
| qfhrr @0.1 | 0.560 / 20.9% / 30 | 0.515 / 21.3% / 71 |
| qfhrr @0.25 | 0.401 / 11.3% / 16 | 0.389 / 13.2% / 51 |

- **The gate on held rows** (qfhrr @0): SimpleQuestions 0.880, QALD FR 0.564 (R3o 0.893, 0.578). It does not pass.
- **Where the kind works:** "place" (qfhrr @0, recall 0.77, 6% wrong) and "time" (0.63, no wrong). Read by kind, it
  helps where the asked kind is read off the item's own values.
- **Where it costs:**
  - "who" questions asked from the value's side ("who was born in Paris?"): the rule keeps incoming properties out when
    a kind is asked.
  - "how many" over item-valued properties.

  Each dropped gold makes a missing-fact row, where qFHRR hands a near property over (83 against 36). That is the
  danger R8's kill line reads.
- **Yours, if R3k is to be read again:**
  - keep incoming properties for "who" (the asker's kind is then the subject's);
  - let "how many" also keep item-valued properties (a count of values).

  Both are rule changes written after the numbers, so they would be read on test, not tuned here.

## The typo index (Claude Code, 2026-11-04)
- **Log:** `retrieval_typo_index_2026-11-04.json`. CubeLang `cowork/vm-contracts` `29e81bd` (`src/vm/typo_index.rs`,
  `docs/STORE.md` there).
- **Where the 589 ms went:** the names store's 32 shards are split by a hash of the key, so every shard holds every
  prefix. A 2-edit lookup walked the same region of 32 indexes, allocated a row for every byte it read, and read
  facts for every key in reach before cutting to the limit.
- **Built:**
  - one index of every shard's keys (keys only), built once beside the shards (`cubelang knowledge typo-index DIR`;
    76,813,854 keys, 846 MB, 130 s). It is used only over the shards it was built from, by name and size;
  - a banded row (5 cells at 2 edits);
  - facts read only up to the limit.

  The exact lookup is unchanged. A store opened from the directory (`knowledge batch`, run-proto's
  `knowledge_path`) uses the index without any change on CubbyLLM's side.

| R1's dev typo mentions, 2 edits | p50 | p95 |
|---|---|---|
| walking every shard (300) | 425 ms | 609 ms |
| banded row (the same 300) | 165 ms | 202 ms |
| **through the index** (the same 300; all 1,204) | **32 ms; 32 ms** | **39 ms; 38 ms** |

- **The line is met:** p95 38 ms against R7's 50 ms, with p99 100 ms. On the 300 compared, the answers are identical
  to the old binary's. The tests pin the index against the shards without it, at every distance and with a limit.
- **Served cost:** the typo lookup runs only on an exact miss (R1c's rule). On Mintaka's test, that is 62 of 1,475 EN
  mentions and none of FR.

## Cowork's check of the fork, R1c, B1q on seed 0, R3k and the typo index (2026-11-04, afternoon)
- **The fork: met** (0.38 ms a branch, p95 3.3 ms at k = 7, the outcomes identical). Merging `cowork/vm-contracts` is
  Nick's call.
- **The typo index: met** (p95 38 ms against 50, the same answers).
- **R1c on test: passes.** The gold is in the set at EN 0.911 against the API's 0.862, and FR 0.994 against 0.971.
  After the relation filter, the set's p50 is 1 and p95 38. From here on, finding is the candidate set; the API stays
  only as R1's reference.
- **B1q on seed 0:** resolved 0.764 (line 70%) but 3 wrong (line 0). Two rules fix that, both general, and both
  written before seed 1 is read:
  1. **Short names match only as written.** A name of 3 characters or fewer (a code, an abbreviation) matches a cue
     only with its case and accents kept, so "né" never matches "NE". A length floor on cues would drop short real
     words, and a list of participles would never end.
     - Span accounting also counts the words the relation reading used, whatever the lexicon holds. On the fame set
       those are the template's own relation words ("né", "born", "pays").
  2. **The qualifier is checked along the `located in` chain,** each step a VM-checked hop, up to the country. The 3
     wrongs become undecided, the safe direction.

  Seed 1 is then read once, with both rules frozen.
- **ShadowLink, where the gold bears the mention:** 4.6 points (line 5), so it passes on dev.
- **Mintaka and WebQSP_EL:** 2.3% and 2.2% wrong per row on dev, against the 2% line read on test (tiers: 5.8% and
  4.0%). The chain rule may move them; test decides.
- **R3k: three gaps in Cowork's rule, not in the code:**
  - **Incoming properties:** the kind is read on the items that point at this one ("who was born here?" asks for
    persons). An incoming property stays when its subjects are of the kind.
  - **The kind along the subclass chain:** a value's `instance of` reaches the kind through `subclass of` steps (a
    city up to a place), each VM-checked, the bound reported.
  - **"How many":** a quantity-valued property, or a property with several values on this item (their count).

  Re-read on dev with these, under the margin rule as written. These fixes come from dev, before test.
- **R4, both readings confirmed:**
  - the live scoreboard's 44 `ambiguous_hop` and 2 `time_mismatch` rows. If the latest run's rows are not the plan's
    44 + 2, keep the plan's by id and report the difference;
  - CronQuestions' `simple_entity` for the 60% line, with `simple_time` beside it and the complex types out.

Status: open.
