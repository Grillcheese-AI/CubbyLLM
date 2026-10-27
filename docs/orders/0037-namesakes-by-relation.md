# 0037 — Namesakes: answer when the question's relation fits only one of them ("the capital of France")
status: open (2026-10-27: Cowork's check passes at the line; three small items left, below "Cowork's check")
written: 2026-10-23 by Cowork · approved: 2026-10-23 by Nick ("a capital is usually in a country so it should pick the country france (there is only one)") · hypothesis: none

## Why
Since 0036, a shared name is decided only by the asker or a qualifier, so "What is the capital of France?" asks
which France. Nick's point: a capital belongs to a country, and only one France is a country. The question's own
relation settles it.

0036's audit showed why the old tier failed (21 of 50 right). It counted an item with no known kind as one that
cannot carry the relation, yet most such items were plainly of the chosen item's kind by their own facts (a
citizenship and an occupation, a cast).

The world's namesakes for the three common questions:
- **France:**
  - Q142, the country (197 facts);
  - Q5478259, a constituency of the European Parliament (a kind with no capital);
  - Q49644, one fact: "country: France" (nothing says what it is).
- **Japan:** besides Q17, a manga (two of them), an album, a band, works with a country of origin and a genre, and a
  place with only "country" and "located in".
- **Italy:** besides Q38, places with only "country" and "located in" (the towns named Italy), and one item with only
  "part of".

**What it brings, in plain words:** Cubby says "Paris" to the obvious question and mentions what else the name could
mean. When the question truly cannot tell (two people named Steve Mills), he answers for each, saying which is which.
- **Over a conventional model:** a model also says Paris, but it would blend two Steve Mills into one answer without
  noticing. Here every choice follows one written rule, measured against a hand-read audit.

## Build
1. **A namesake's kind from its own facts,** when it has no `instance of`.
   - Through the kinds table (0036), read the kinds whose members carry its relations.
   - Specific facts tell what it is: a citizenship and an occupation mean a person, a cast means a film.
   - Generic facts ("country", "located in") say only that it is somewhere.
2. **"Does a kind carry the relation?" as a share, not "any member".**
   - The kinds table keeps, per (kind, relation), how many members carry it.
   - A capital is common among sovereign states and rare among towns.
   - Claude Code proposes the threshold and states it in the report.
3. **Cue words from the question** (Nick: "a building cannot have a Nobel Prize, only a person").
   - The relation alone is often too wide: "award received" fits films (an Oscar) and buildings (an architecture
     prize). The question's other words narrow it: "Nobel" goes only with people, and with a few organisations for
     the Peace Prize.
   - **Built from the world, not assigned by hand:** a word-to-kinds table, made once with the kinds table. For each
     content word in the world's values, it records which kinds of subjects carry it (recorded, or read by item 1).
     This covers every topic with no hand list to keep up.
   - In this snapshot the 22 holders of the Nobel Prize in Physics have no recorded kind, so item 1's reading is what
     makes the table useful.
   - A cue word **only rules out**: a namesake whose kind never goes with the word does not compete. It never ranks.
4. **The rule.** A hop on a shared name reads one item when two things hold:
   - that item's kind carries the relation, and goes with the question's cue words;
   - no other namesake's kind does (whether the kind is recorded, or read by item 1).

   An item whose facts say nothing of its kind does not compete (France's Q49644). The audit decides whether that
   holds. Two of a kind: see item 5.
5. **How the answer reads** (Nick: answer, and say what else the name could mean, as Gemini does).
   - **One fits:** answer from it, then add one line naming the other namesakes by their kind, and offer them: "Marie
     Curie won … If you meant something else, there is also a building named Marie Curie. Want to know about it?"
     - Name at most two others.
     - An item with nothing known of its kind is not named.
   - **Several fit (two of a kind):** answer each one, labelled by one of its own facts that tells them apart (an
     occupation, a country, a date): "Steve Mills the baseball player was born in 1950; Steve Mills the politician in
     1962."
     - Give them in a fixed order, never by fame.
     - With more than three, list them by their labels and ask which.
   - **Never merged:** each value stays tied to its own item (0033's rule). A label is one of the item's facts, never
     invented.
   - **Asking back stays** only where no fact tells the items apart.
6. **Behind a flag** until the audit passes; then on by default for hops. The asker and a qualifier still come first.

## Acceptance (pre-registered)
- **The audit,** read by hand by Cowork, as in 0036:
  - 50 shared-name hops, drawn by seed 1 (not 0036's seed 0);
  - for each: its label, the items with their kinds (recorded or read), the rule's outcome (one item, each item
    labelled, or asking back), and the answer as spoken.
  - **In the "several fit" answers:** 0 values tied to the wrong item, and every label one of that item's own facts.
  - **A wrong pick** means one item was chosen when a person asking could as well mean another.
  - **The line:** at most 1 wrong pick in 50. Asking back where the question was obvious is a cost, not a wrong
    pick; report how many.
  - Below the line, strict stays, and the report says which guard missed.
- **The three common questions** answer: France's capital (Paris), Japan's currency (the yen), Italy's capital (Rome).
- **A cue-word probe:**
  - 20 shared names, drawn by seed 1, each asked with a cue word taken from one item's own values;
  - each question picks that item or asks back, with 0 wrong picks;
  - Cowork reads the list.
- **"What Nobel Prize did Marie Curie win?"** picks Q7186, the person (the other Marie Curie has only a country).
  The snapshot holds no Nobel fact for her (four facts), so the answer says it does not know rather than guessing.
  Choosing the right item and holding its fact are separate checks.
- **0033's probe:** 0 mixed. **0014's bench:** 0 wrong.
- **The stack's 20:** France's capital and Japan's currency answer again; the other 18 are as after 0036.
- **Latency:** the hop's decision time is reported.

## Constraints
- Rules in `CLAUDE.md` hold.
- **Nothing is ranked.** An item never wins for being famous or more completely described: Q142's 197 facts count
  for nothing, only its kind does.
- The online source's rule (`sources.wikidata_resolve`) is unchanged.

## Report
- **Commits:**
  - `05c87c6`: the rule, behind `CB_NAMESAKE_RELATION=1` (off);
  - `127b6b7`: the answers pass the loop's checks;
  - `60ec6b1`: the audit and its lessons;
  - `39b87f9`: the cue words;
  - this one: the report.
- **Files:**
  - `cubbyllm/worlds/hdc_kinds.py`: carries by share; `read` / `read_how`.
  - `hdc_cues.py`: the word-to-kinds table, `CUE_MIN`.
  - `hdc_relation.py`: `decide`, `others`, `sayable`.
  - `hdc_namesakes.py`: `hop_rows`, `decision`, `cued`.
  - `hdc_store.py`.
  - `cubbyllm/reasoning/ask_namesake_answer.py`: how the answer reads.
  - `ask_namesakes.py`: `seed_for`, `after_walk`, `also`.
  - `ask_memory.py`: `_checked`.
  - `ask.py`: two hooks.
  - `sleep.py`: `not_what_was_asked` is routed to `source_gaps`.
  - `validation/audit_namesake_relation.py`.
  - Tests: `tests/worlds/test_hdc_relation.py` (10) and `tests/reasoning/test_ask_namesake_answer.py` (9).
- **The thresholds (item 2), and why:**
  - **LIFT = 2, NMIN = 3.** A kind carries a relation when its members hold it at least twice as often as the
    world's typed items do, from at least 3 members.
    - The base comes from typed items only. Untyped people dominate the snapshot, and an all-items base made
      humans carry nothing.
  - **SPREAD = 50.** A relation more than 50 kinds carry ("country", "located in") says only that an item is
    somewhere, so it reads no kind.
  - **CUE_MIN = 100.** A cue word rules out only when 100 of the world's subjects use it ("nobel": 711; "flows":
    1).
- **What the audit changed in the rule** (the first run, read by hand before handing it over; details in the
  commits):
  1. **One that fits must hold a value,** or it decides nothing.
     - In 5 of 12 picks the item of the right kind held nothing, while an item of no kind did (Qu River,
       Petershausen, ...). It would have said "I don't know" instead of asking.
  2. **Two of a kind compete together,** whatever their facts.
     - **Thespis was 1 wrong pick:** two taxa, one picked because its facts had been read widely.
     - The guard that missed: "no other namesake's kind does" compared kinds per item, so a kind shared with the
       fit item did not count.
  3. **An item read widely is never named by its kind.**
     - Its relations go with no kind together ("read widely").
     - Demon Hunter, a band, was named "a film". That was a wrong statement in the offer line.
  4. **Cue words are the question's capitalised words,** with kind names in the table and rare words ignored.
     - "flows" in "What river flows through Cairo?" had ruled out the city.
  5. **Labels skip the relation asked,** and two others alike are said once.
- **Measured** (the final rule; logs in `validation/logs/`):
  - **The audit:** `audit_namesake_relation_2026-10-24.md` is the list for Cowork's reading, and its `.json`.
    - 50 hops, seed 1.
    - Outcomes: 6 one item; 1 each labelled (Thespis, by parent taxon); 43 asked back.
      - 4 of the 43 ask back because the one that fits holds nothing.
      - The rest: no item of a known kind fits, or two of a kind no other fact tells apart.
    - Wrong picks by my reading: 0. Cowork's reading is the line.
  - **The three common questions** (world level, and through the loop):
    - Paris (Q142);
    - the Japanese yen (Q17);
    - Rome (Q38).
  - **The cue-word probe** (20 names, seed 1, in the same files): 3 pick the target, 17 ask back (16 with the
    target among the fits), 0 wrong picks.
    - A cue word rules out other kinds, never another item of the same kind. So two people named alike stay a
      question.
  - **Marie Curie** (`0037_stack_marie_curie_2026-10-24.jsonl`):
    - **World level:** the cue "nobel" picks Q7186. The class guard then says it does not know: her one award in
      the snapshot is the Elliott Cresson Medal.
    - **On the real stack, the literal question never reaches the hop.** The 450M plans `["won"]` with the seed
      "marie", and the plan check refuses it (`plan_does_not_cover_question`). The everyday table does not read
      "What X did Y win?" either.
    - "What award did Marie Curie receive?" picks Q7186 (recorded human; the other Marie Curie has no kind) and
      says "Elliott Cresson Medal": true, and all the snapshot holds.
  - **0033's probe:** 0 mixed (`namesake_probe_2026-10-24_0037.json`).
  - **0014's bench:** 192 correct, 0 wrong, 0 time-sensitive values said flat.
    - The same with the rule off (`exp_hdc_bench_2026-10-24_0037.json`, `..._0037_off.json`): 0037 changes
      nothing on this bench.
    - The gain over 0036's 178 comes from the day's defaults (0038: the tense check, the table).
  - **The stack's 20** (`0037_stack_off_2026-10-24.json`, `0037_stack_on_2026-10-24.json`):
    - Off and on speak the same 10.
    - Under today's defaults France and Japan answer even with the rule off, from the source's cache through
      0038's table.
    - With the rule on, the hdc snapshot answers first:
      - "As of 2019, Paris. If you meant something else, there is also a constituency of the European Parliament
        named France. ...";
      - "As of 2019, Ottawa.";
      - "Japanese yen. If you meant ... a manga named Japan (2 of them) ...";
      - "Nile. If you meant ... an operating system named Cairo and a village in the united states named Cairo
        ...".
    - The other 16 are the same in both arms.
  - **Latency:** the decision takes p50 4.5 ms, max 40.9 ms (75 decisions on the real world).
- **Not verified here:** Cowork's reading of the audit and the probe (the line is at most 1 wrong pick in 50).
- **Questions:**
  1. **The default.** Item 6 turns the rule on for hops once the audit passes. It stays off until Cowork's reading.
     Should it come on then?
  2. **Snapshot or source.** With the rule on, the snapshot's "As of 2019, Paris" replaces the source's dated,
     flat "Paris". Should a dated source's value win over a snapshot's time-sensitive one?
  3. **Steve Mills.** The two Steve Mills hold only their citizenships, so asked their citizenship they are asked
     which. "Answer each" needs a fact other than the answer. Is that acceptable?
  4. **The source side.** Item 3 of your notes on the 76 refusals (namesakes at the Wikidata source) is outside
     this order: its constraints keep `sources.wikidata_resolve` unchanged. A new order?

## Cowork's check (2026-10-27)
- **The audit, read by hand** (`audit_namesake_relation_2026-10-24.md`, seed 1): **1 wrong pick in 50**, at the line, so
  the rule passes.
  - **The wrong pick: 14, KATA.** It said "Kose Rural Municipality" (Q3459361, a village in Estonia). Q5808376 is
    also a village, in Iran, and Q6325364 is a radio station in California (both checked on Wikidata). Someone
    asking where KATA is could as well mean the Iranian village.
  - **The guard that missed:** "an item whose facts say nothing of its kind does not compete". Q5808376's only fact
    is `country: Iran`. A fact that places an item somewhere says it *is* somewhere, so it can carry a place
    relation.
  - **The other five picks are right:** The Architect (Syracuse, the band), Alfred-Pellan (the other is an electoral
    district), Demon Hunter (Marvel), CBGB (the film), The Toxic Avenger (the film). Thespis, labelled by parent
    taxon: each value is tied to its own item.
  - **Asked back where the question was obvious** (a cost): about 8. In four the namesakes give the same answer:
    Petershausen (Germany both), Notojima (Japan both), Calhoun Township, Sumner Township. Chen's capital is the
    State of Chen, an item with no kind. Others: Qu River, Barreiros, Gnojno.
- **The cue-word probe:** 3 picks (Demon Hunter, Chen as a taxon, CBGB), 17 ask back, **0 wrong**.
- **Your questions:**
  1. **The default:** on, as item 6 says. That is the first item below.
  2. **Snapshot or source:** Nick's answer (2026-10-27): **Wikidata wins.** That is the third item below.
  3. **Steve Mills:** yes. Asking back stays where no fact tells the items apart (item 5).
  4. **The source side:** order 0043.
- **Left to do here, then `done` without another review:**
  1. **Item 6:** the rule on by default for hops (`CB_NAMESAKE_RELATION=0` turns it off).
  2. **The KATA guard:** an item whose facts place it somewhere (`country`, `located in`, `located in the
     administrative territorial entity`) competes for a place relation even with no kind. Re-run the audit on
     seed 1: KATA asks back, and nothing else changes. Re-run the probe: unchanged. Report both in one line here.
  3. **Wikidata wins (Nick):** for a time-sensitive relation, a value the live source gives is the answer, and the
     snapshot's "As of ..." value answers only when the source gives none (offline, not cached, or no such fact).
     The namesake rule still decides *which item* is meant; this decides only whose value is said. Re-run the
     stack's 20: France and Canada answer with the source's value ("Paris", "Ottawa"), not "As of 2019, ...";
     with the source off they still say "As of 2019, Paris". Report the two arms in one line here.
- **Carried to 0043:** when every namesake gives the same value, saying it once is right whichever one was meant
  (Petershausen and Notojima would answer).
