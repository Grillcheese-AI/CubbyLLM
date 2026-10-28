# 0043 — Namesakes at the source: 0037's rule on the Wikidata side, and namesakes that agree
status: open (Cowork's check, 2026-10-28: five small items, then the flag on)
written: 2026-10-27 by Cowork · approved: 2026-10-27 by Nick ("yes") · hypothesis: none

## Why
0038's scoreboard refuses about 25 everyday questions because the name means two things at the source
(`sources.wikidata_resolve`). The strict rule refuses two of a kind, and among different kinds lets only the
relation or the label decide, in narrow cases. From Cowork's sorting of the 76 refusals (the inbox, 2026-10-24,
item 3) and the last run (`everyday_webq_on_live_tense_names2_2026-10-24.json`):
- "uk" (4 times): also Uttarakhand's code, and the Ukrainian language's.
- Georgia (a country and a US state), Cyprus (a country and an island), Obama (a city in Japan and the president),
  the paintings named "King Henry VIII" and "Queen Isabella".
- Two people each: Ben Franklin, Anne Hathaway, Steve Nash.
- `ambiguous_entity` (7): London, Chicago, Toronto, Oregon, Iceland, George Wilson.

0037 settled this rule for the worlds and passed its audit (1 wrong pick in 50, at the line; 0037's "Cowork's
check"). Its constraints kept the source unchanged, so the same questions still refuse when the answer comes from
Wikidata. At the source the rule has more to go on: almost every item carries `instance of` (P31), and Wikidata
says which kinds a property applies to.

0037's audit also showed a cheap win: when every namesake gives the same value (two places named Petershausen, both
in Germany), the answer is right whichever one was meant.

**What it brings, in plain words:** Cubby answers "what currency do they use in the uk?" and "where was Obama
born?", says what else the name could mean, and answers for each when the question truly cannot tell (Georgia the
country, Georgia the state).
- **Over a conventional model:** a model picks the famous one silently, and blends two people of one name. Here one
  written rule decides, the same in the worlds and at the source, and it is measured against a hand-read audit.

## Build
1. **Diagnose first.** For each of the names above, record in the report why the source refused: which exact hits
   came back, their kinds, which carry the relation. Some may be search depth (`limit: 5`) or an alias, not the
   rule. Fix those as what they are.
2. **The kind a candidate is,** from its P31. A candidate with no P31 is read as in 0037 item 1. A candidate whose
   facts place it somewhere (`country`, `located in`, `located in the administrative territorial entity`)
   competes for a place relation (0037's KATA guard).
3. **Does the kind carry the relation,** from the source's own data, not a hand list:
   - the candidate's own claims carry the property: it does;
   - otherwise the property's subject-type constraint (P2302 with Q21503250) lists classes the candidate's kind
     falls under (one `subclass of` step is enough): it may;
   - otherwise it does not.
   Cache the property constraints; they change rarely.
4. **Cue words,** as 0037: the question's capitalised words and kind names only rule out.
5. **The rule and how it reads,** exactly 0037's item 4 and item 5: one fits and holds a value, so answer and
   offer at most two others by kind; two of a kind, or two kinds that both carry the relation, so answer each,
   labelled by one of its own facts (at most three, then list and ask); nothing tells them apart, so ask back. The
   asker's ledger and a qualifier still come first.
6. **Namesakes that agree,** in the worlds and at the source: when every candidate that fits gives the same value,
   say it once, with no offer line. "The same value" means the same item (QID), or the same literal for a date or
   a number.
7. **Behind `CB_SOURCE_NAMESAKES=1`,** off until Cowork's reading of the audit; then on by default.

## Acceptance (pre-registered)
- **Tests:** a canned search and claims (no network) for each case: one fits (Obama, birth place), two kinds both
  fit (Georgia, capital: each answered and labelled), two of a kind (Ben Franklin: never one picked), they agree
  (Petershausen: one answer), the subject-type constraint read from a cached property.
- **The audit,** read by hand by Cowork:
  - 50 shared names at the source: every one among the 187 and the stack's 20, then names drawn by seed 2 from the
    hdc world's namesakes and asked against the live source, until there are 50 (state the draw in the report);
  - for each: the hits with their kinds, the rule's outcome, the answer as spoken.
  - **The line:** at most 1 wrong pick in 50, and 0 values tied to the wrong item in any "answer each".
  - Below the line, strict stays at the source and the report says which guard missed.
- **The scoreboard** (`exp_everyday_webq.py`, live, tense on, the defaults of 0038): more spoken than 65, **0
  wrong**, every changed row read by hand. Report how many of the names above now answer.
- **The stack's 20:** 0 wrong; report what changed.
- **0014's bench and 0033's probe:** 0 wrong, 0 mixed.
- **Latency:** the added source calls per question (constraints are cached) and the decision time.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Nothing is ranked.** Fame, sitelinks, search order and how completely an item is described decide nothing.
- **The James Young rule stands:** two of a kind are never separated by which one is better described.
- Wikidata is reached only through the existing source module and its cache; `cubbyllm/` never touches the
  network.

## Out of scope
- Several values of ONE item (Switzerland's languages): order 0044.
- Reading the question better ("cleveland ohio"): order 0039.
- Which wins between the snapshot and the source for a time-sensitive value: decided, Wikidata wins (0037, item 3
  under Cowork's check).

## Report
- **Step 1, the diagnosis** (`validation/logs/_diag_0043_names.py`, live, 2026-10-27; each name with the question's relation):
  - **Two kinds that both carry the relation:** uk (Uttarakhand, a state of India, and the United Kingdom: both exact
    through the alias "UK", both carry currency / time zone / official language); Georgia (the U.S. state, the country,
    the former Province of Georgia; the given name and the university are exact too and carry no capital); Cyprus (the
    country and the island, both carry currency); Iceland (the country and the island, both carry country); Oregon (the
    state and a city in Illinois, both carry a capital through P1376 "capital of").
  - **Two or more of a kind:** Ben Franklin (Benjamin Franklin through the alias, and a 21st-century politician labelled
    Ben Franklin, both with a birthplace; an Australian politician without one), Anne Hathaway (the actress, the wife of
    Shakespeare, and Mary Bigelow Ingham through a pseudonym, all three with a birthplace), Steve Nash (five people, three
    with a citizenship), George Wilson (five people, four with an occupation), Jack London (three people with one).
  - **Search depth, not the rule, and no fix in reach:** Obama (the president is not among the search's hits at 5, 10 or
    20: Wikidata's search does not match him to "Obama"; the hits are a city in Japan, a genus of worms, a drag
    performer, a video game) and King Henry VIII (paintings only, at 20 hits too). Queen Isabella: the exact hits are a
    disambiguation page, a painting and a theatrical character; Isabella I of Castile matches only as "Queen Isabella of
    Spain". These stay refused; a wider search would not change them.
  - **Time zones (London, Chicago, Toronto):** no item carries P421 directly; the cities fit only through the
    property's constraint ("may"), hold nothing, and the loop still refuses them, now as asking back or as nothing held.
  - **The seven `ambiguous_entity` of the scoreboard (London, Chicago, Toronto, Oregon, Iceland, George Wilson):** the
    hdc world's namesakes, not the source's; 0037's rule, on since 2026-10-27, settles some (Oregon answers each).
- **Commits:** `869b8ae` in progress; this one: the rule, the loop's side, the tests, the audit, the measurements.
- **Files:**
  - `cubbyllm/sources/wikidata_namesakes.py` (new): the rule at the source -- `carries` (holds / may through the
    property's subject-type constraint P2302, one `subclass of` step; the constraints are the property's own cached
    claims), `PLACE` (the KATA guard), `decide` (cue words only rule out; a kind that may carry yields to another kind
    that holds; the James Young rule within a kind), `by_label` (the source's label-over-alias tier, kept, among
    different kinds only), `pinned` / `cued` / `namesake_decision` (per thread), `item_rows`.
  - `wikidata_resolve.py`: the hook (behind `CB_SOURCE_NAMESAKES=1`; the pin resolves without the ledger; the ledger
    still comes first; several that fit -> `last["each"]`). `wikidata.py`: the mixin (one import line; the file was over
    300 already).
  - `cubbyllm/reasoning/ask_source_namesakes.py` (new): one fits -> the offer line; several fit -> each walked apart
    with the source pinned, on a **scratch first world** (nothing of one item lands under the other's name: 0033's
    rule), labelled by one of its own facts (`label_for` over the item's claims, else its kind); namesakes that agree
    -> once, in the plainest words any was said in. `ask_namesakes.py`: the source's branch after the worlds'; the
    first-world guard no longer hides it. `ask.py`: `learn(store=)`. `ask_namesake_answer.py`: item 6 for the worlds
    too (two of a kind that give the same value are said once: Ann Lee).
  - `validation/audit_source_namesakes.py` (new). Tests: `tests/sources/test_wikidata_namesakes.py` (9),
    `tests/reasoning/test_ask_source_namesakes.py` (4), `test_ask_namesake_answer.py` (Ann Lee agrees).
- **Tests:** 1125 passed, 1 failed (`test_every_module_declares_wiring`, the known red of order 0002).
- **Measured** (`validation/logs/`, 2026-10-27, all live):
  - **The audit** (`audit_source_namesakes_2026-10-27.md`): **50 names, all from the scoreboard's 187 and the stack's 20**
    (62 are shared at the source under the strict rule, so no hdc draw was needed; the draw by seed 2 is in the script
    for when fewer are). Outcomes: 11 one fits (3 of them by the label over an alias: Spain, Cambodia, England), 4
    several fit (Georgia: Atlanta / Tbilisi / Savannah, each tied to its own item; uk: the rupee and the pound; Joseph
    P. Kennedy; Harry Truman), 1 agree (Iceland), 3 asking back (Gabriel Faure, uk's time zone, John Muir), 4 answered
    with no shared name left, 27 refused for other reasons (14 `ambiguous_hop`: several values of one item, 0044's;
    10 nothing found; 3 relations). **Wrong picks by my reading: 0. Values tied to the wrong item: 0.** Asked back where
    the question was obvious: 3 (the composer, the United Kingdom, the naturalist). Cowork's reading is the line.
  - **The scoreboard** (`everyday_webq_on_live_tense_0043_2026-10-27.json`, 187, live, tense on, the defaults): **82
    spoken** (the flag off the same day: 74, `everyday_webq_on_live_tense_0043_off_2026-10-27.json`; 2026-10-24: 65).
    Every changed row read by hand: 8 new answers (Georgia each; uk each; Joseph P. Kennedy each; Ben Franklin each,
    Boston for the right one; John Jay, three each; Eleanor Roosevelt and Jack London, each with the famous one's own
    value refused as several values; Vatican City, agreeing with itself), 27 unchanged answers now carrying an offer
    line, 3 single answers now answered each (Truman, MacArthur, Nixon). **0 wrong.** The 20 key mismatches are key
    wording (UTC-07:00 for Mountain Time, pound sterling for "UK", Italica for Hadrian, Research Medical Center for
    Kansas City) or pre-existing (Nina Dobrev's Canada, 0044's several values). Of the order's names: uk (the rupee and
    the pound, each), Georgia, Iceland, Ben Franklin answer; Steve Nash's and Anne Hathaway's questions are not among
    the 187 the table reads today; Cyprus's currency, Obama, Henry VIII and Queen Isabella still refuse (search, not
    the rule).
  - **The stack's 20** (`0043_stack_live_2026-10-27.json`): the same 11 spoken as 0037's live run, 0 wrong; three now
    carry an offer line (Ada Lovelace, Italie, World War II).
  - **0014's bench** (`exp_hdc_bench_2026-10-27_0043.json`): 192 correct, 0 wrong, 0 time-sensitive values said flat.
    **0033's probe** (`namesake_probe_2026-10-27_0043.json`): 0 mixed.
  - **Latency:** the decision p50 9.3 ms, max 17.4 ms with the cache warm (35 decisions); the first pass live, p50 543
    ms, max 2.8 s, is the network (a candidate's claims, the property's constraints, the kinds' labels, each fetched
    once). Source calls per question: p50 0, max 3 warm; p50 2.5, max 13 on the first pass.
- **Not verified here:** Cowork's reading of the audit and of the 8 new scoreboard answers.
- **Found on the way:**
  - **The two refinements** the first live pass forced (both in the module's docstring): a kind that only *may* carry
    the relation yields to a different kind that *holds* it (the novel Pride and Prejudice over three editions; the
    painting over two people named Mona Lisa; with "may" competing, both were refused); and the source's own
    label-over-alias tier stays, among different kinds (Spain over the Second Spanish Republic and New Spain; Germany
    over the German Reich and the Weimar Republic; Belize over British Honduras). Without it the scoreboard fell to 68
    and **"what is the capital city of germany now?" said "As of 2019, Bonn"**: the source refused (several fit), and
    the hdc snapshot's own value stood -- the snapshot holds Bonn as a capital of Germany (Q183) with no date. That
    stale value is still there behind the source-first rule; a snapshot with undated historical values is 0044's
    kind of question (which values still hold).
  - A label by the relation asked: "Jack London (occupation: writer): I do not know" labels by occupation and then
    refuses the occupation (several values). The label skips the relation when the wording matches exactly; here the
    plan's wording and the claim's label differ. Small; 0044 touches the same rows.
  - `each` answers "I do not know" for an item that fits but holds nothing (the rugby player, the researcher): true
    and tied to its item, but noisy; Cowork may prefer those dropped from the line when another item answered.
- **Questions:**
  1. The flag stays off until your reading; then on by default (item 7).
  2. "uk" answers each (the rupee and the pound) because both carry every relation asked and neither is labelled "uk".
     The asker ledger will settle it; or a cue word; or do you want a rule that a country's ISO code is the country?
  3. Is the offer line wanted on every one-fits answer at the source (27 rows gained one: "there is also a family name
     named Spain")? 0037 added it for the worlds; the source finds far more namesakes (family names, albums, chapters).

## Cowork's check (2026-10-28)
Read by hand: the audit (`audit_source_namesakes_2026-10-27.md`), the scoreboard with the flag on and off, the stack's
20, and the Wikidata items behind every "answer each".
- **The audit is under the line:** 0 wrong picks in 50, 0 values tied to the wrong item. Each value checked against
  its own item (Joseph Kennedy the scientist, Q18153879, holds the alias "Joseph P. Kennedy" and was born in Budapest;
  Savannah belongs to the Province of Georgia). The 3 asked back where the question was obvious are costs; item 1
  removes one of them (uk's time zone).
- **The scoreboard:** 82 spoken, 0 wrong values by my reading too. Two kinds of wording mislead: the labels (item 4),
  and "was a capital of ... at one point" said of current values (Juneau, Columbia, Dublin). The second is older than
  this order (2 rows on 2026-10-24, 5 with the flag off, 7 on) and goes to 0044, which reads which values still hold.
- **0037's Calhoun Township:** right. The placed township answers "United States of America"; every Calhoun
  Township is in the US. The Wikimedia disambiguation page it offers is not a namesake (item 5).
- **Q1:** yes, `CB_SOURCE_NAMESAKES` on by default once the items below are in and measured.
- **Q2:** neither the ledger alone nor an ISO rule. The United Kingdom's ISO code is GB; "UK" is Uttarakhand's code
  (IN-UK). The source's own names settle it: item 1.
- **Q3:** the trim in item 5 holds either way. Whether the trimmed line stays at the source is Nick's call (asked
  2026-10-28). Until he answers, keep the trimmed line.

### Items (behind the same flag)
1. **The initials tier.** It applies among different kinds only, after the label and before any other alias. An
   alias equal to the initials of the item's own label (its capitalised words; "of", "the" and "and" skipped) beats
   an alias that is not. "UK" is the initials of United Kingdom, and only a code for Uttarakhand, so the UK answers.
   The same holds for USA, UN, NYC and UAE. Two candidates that both match by initials go on to the rule. Never within
   one kind (the James Young rule).
2. **"Each", with items that hold nothing.** Answer the items that hold a value. Fold the others into one closing
   sentence instead of an "I do not know" each: "... The source holds no place of death for the other Harry Truman
   (rugby union player)." Do not drop them: answering only the better-described one is the pick the James Young
   rule forbids. When none holds a value: as today.
3. **"Each", where an item holds several values:** refuse the question as `ambiguous_hop`, as before this order,
   until 0044 can list that item's values. Today Eleanor Roosevelt gives the photographer's Oyster Bay and "I do not
   know" for the diplomat, and Jack London gives "writer: I do not know". Each value is tied to its own item, but the
   answer reads as if it were about the wrong person.
4. **Labels that tell them apart.** A label fact is never the item itself and never the value asked:
   "United Kingdom (country: United Kingdom)", "Georgia (country: Georgia)", "Jack London (occupation: merchant):
   merchant". When the kinds differ, label places by their kind: "Georgia, the U.S. state: ...; Georgia, the country:
   Tbilisi; the Province of Georgia: Savannah". Label people by their occupation and birth year: "John Jay (judge,
   born 1745)". Today "judge" and "diplomat" read like one man, because the famous John Jay was both. Labels still
   come from the item's own facts.
5. **The offer line names namesakes only, in the worlds and at the source.** Never offer:
   - name records: a family name or a given name;
   - Wikimedia internal items: disambiguation pages, lists;
   - reference entries: an encyclopedia article;
   - parts and editions of a work: a chapter, a custom map / level, a version, edition or translation;
   - a candidate that matched only through an alias while its own label is another name: German Reich and Italian
     Peninsula for Germany and Italy, Old Swiss Confederacy, domesticated turkey, Mohammed Haruna for "Cambodia".

   Read the classes with the same `subclass of` step as `carries`, not a list of labels. On the scoreboard this
   keeps 20 of the 32 lines (albums, rivers, taxa, towns, games, a political party) and removes 12.

### Acceptance for the items
- Tests (canned, no network): uk (currency, time zone, language) answers for the United Kingdom; two candidates
  that both match by initials go on to the rule; "each" with an item that holds nothing gives one folded sentence;
  "each" with an item that holds several values is refused; no label repeats the item or the value; a family name, a
  disambiguation page and an alias-only match are never offered.
- The audit re-run (the same 50): 0 wrong picks, 0 values on the wrong item; say what changed.
- The scoreboard: 0 wrong, every changed row read by hand, at least 80 spoken (item 3 takes back two).
- The stack's 20, 0014's bench and 0033's probe: as before.
- Then `CB_SOURCE_NAMESAKES` goes on by default. Report in one block here and set `done`; no review needed unless a
  wrong value appears.
