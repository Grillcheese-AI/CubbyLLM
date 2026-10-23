# 0037 — Namesakes: answer when the question's relation fits only one of them ("the capital of France")
status: open
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

**What it brings, in plain words:** Cubby says "Paris" to the obvious question, and asks back only when the question
truly cannot tell (two people named Steve Mills).
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
3. **The rule.** A hop on a shared name reads one item when two things hold:
   - that item's kind carries the relation;
   - no other namesake's kind carries it (whether the kind is recorded, or read by item 1).

   Two of a kind still asks back. An item whose facts say nothing of its kind does not compete (France's Q49644). The
   audit decides whether that holds.
4. **Behind a flag** until the audit passes; then on by default for hops. The asker and a qualifier still come first.

## Acceptance (pre-registered)
- **The audit,** read by hand by Cowork, as in 0036:
  - 50 shared-name hops, drawn by seed 1 (not 0036's seed 0);
  - for each: its label, the items with their kinds (recorded or read), and the rule's outcome (one item, or asking
    back).
  - **A wrong pick** means one item was chosen when a person asking could as well mean another.
  - **The line:** at most 1 wrong pick in 50. Asking back where the question was obvious is a cost, not a wrong
    pick; report how many.
  - Below the line, strict stays, and the report says which guard missed.
- **The three common questions** answer: France's capital (Paris), Japan's currency (the yen), Italy's capital (Rome).
- **0033's probe:** 0 mixed. **0014's bench:** 0 wrong.
- **The stack's 20:** France's capital and Japan's currency answer again; the other 18 are as after 0036.
- **Latency:** the hop's decision time is reported.

## Constraints
- Rules in `CLAUDE.md` hold.
- **Nothing is ranked.** An item never wins for being famous or more completely described: Q142's 197 facts count
  for nothing, only its kind does.
- The online source's rule (`sources.wikidata_resolve`) is unchanged.

## Report
<!-- Claude Code fills this in, then sets status: review -->
