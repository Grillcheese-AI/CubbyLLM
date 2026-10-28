# 0046 — Retrieval, measured before it is built: experiments R0 to R8
status: review (R0 and R1 done; Claude Code, 2026-10-28)
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
