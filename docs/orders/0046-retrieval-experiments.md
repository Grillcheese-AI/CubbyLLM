# 0046 — Retrieval, measured before it is built: experiments R0 to R8
status: open (R2 on dev, 2026-10-31: no arm passes (tiers, BM25, qFHRR on grilly2, B1), test unread; R3, R4 next; R1b on the notebook)
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
