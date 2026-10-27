# 0044 — Several values for one relation: the source's preferred one, or every value that still holds
status: open
written: 2026-10-27 by Cowork · approved: 2026-10-27 by Nick ("yes") · hypothesis: none

## Why
51 of 0038's 187 everyday questions refuse as `ambiguous_hop` (`everyday_webq_on_live_tense_names2_2026-10-24.json`):
one item, one relation, several values, and nothing the source said orders them. 20 time zones, 14 languages, 8
forms of government, 2 capitals (Spain), 2 continents (the USA), 2 currencies (France, Panama), a country of
citizenship, the Iberian Peninsula's countries, Hitler's place of death.

Most of these are right as a set. Switzerland has four official languages, Panama uses the balboa and the US dollar,
Nina Dobrev holds two citizenships (0038 said only "Canada"). The walk's rule (`_latest_dated`) lets only a date
the source gave separate values. It treats every set as a choice, and a set of true values is not a choice.

Two cases must stay refusals:
- **A relation with one value at a time:** Spain's capital. Old capitals without an end date must never be listed
  beside Madrid.
- **The poisoned capital** (`ask_claims.branch_clarify`): a false value in the store must not come back as part of
  a "verified" set.

Wikidata itself says which case is which:
- a statement's **preferred rank** marks the current or best value;
- a property's **single-value** constraints (P2302 with Q19474404, or with Q52060874 for single-best-value) say it
  holds one value at a time.

The source module already drops deprecated statements. It does not read the preferred rank.

**What it brings, in plain words:** Cubby answers "what language do they speak in Switzerland?" with all four
languages, and Spain's capital with Madrid alone, because the source says which relations hold several values.
- **Over a conventional model:** a model lists from memory and can mix current and former values. Here each value is
  a checked fact, the ended ones are dropped by their dates, and whether to list or pick is read from the source's
  own data, not guessed.

## Build
1. **Preferred rank, kept with the fact.** `sources/wikidata.py` records a statement's rank beside its time
   qualifiers, in the same `times` record (`{"rank": "preferred"}`), and the cache keeps it.
2. **The source's order first.** At an ambiguous last hop, when exactly one candidate's fact carries the preferred
   rank, it is the answer. This is the source ordering its own values, as dates are (invariant 4). Several
   preferred, or none: go on.
3. **One value at a time, or several.** Read from the property's constraints (cached): single-value or
   single-best-value means one at a time, and the hop refuses as today. Otherwise the property may hold several.
4. **Every value that still holds.** For a property that may hold several:
   - drop the values whose end the source stated, and those the tense check refuses;
   - speak the rest as one answer, in a fixed order (by label), each value a verified fact the VM certified:
     "German, French, Italian and Romansh";
   - with more than five, say the first five and how many more;
   - only facts the gate holds as verified or corroborated count. A latent or attributed-only fact keeps the refusal.
5. **Where it applies:** the last hop only. Mid-chain splits keep `branch_clarify`, since the asker picks the path
   there.
6. **In the worlds** (no rank, no constraints): the source's constraints for the same property decide when its
   id is known. Otherwise the hop refuses as today.
7. **Behind `CB_EVERY_VALUE=1`,** off until Cowork's reading; then on by default.

## Acceptance (pre-registered)
- **Tests** (canned claims, no network):
  - Switzerland's official languages: all four, by label;
  - Spain's capital with an old capital undated: Madrid, by preferred rank;
  - a single-value property with two undated values and no preferred one: refused;
  - an ended value dropped from a set;
  - a latent fact in a set: refused;
  - a mid-chain split: unchanged.
- **The scoreboard** (`exp_everyday_webq.py`, live, tense on, 0038's defaults):
  - **Gate:** at least 20 of the 51 now spoken, with **0 wrong**, every changed row read by hand.
  - **Wrong** here means a listed value that does not hold for that item now, or a single value where the
    relation holds several and the question asked for all. Report as costs, not wrong: a true value the asker
    would not expect (the CFP franc for France), and two names for one thing (a zone and its UTC offset).
  - **Kill:** any wrong set, such as Valladolid or Toledo beside Madrid.
- **The stack's 20, 0014's bench, 0033's probe:** 0 wrong, 0 mixed; report what changed.
- **Nina Dobrev's citizenship** says both.
- **Latency:** the constraint lookups (cached) per question.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Nothing is ranked by the host.** Only the source orders its values: its dates, its preferred rank, its
  constraints. Fame, sitelinks and order of appearance decide nothing.
- Wikidata only through the source module and its cache; `cubbyllm/` never touches the network.

## Out of scope
- Namesakes (several items for one name): order 0043.
- Merging two names for one thing (Pacific Time Zone and UTC−08:00): note the cases; a later order if they matter.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
