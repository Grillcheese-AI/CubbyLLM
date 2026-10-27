# 0043 — Namesakes at the source: 0037's rule on the Wikidata side, and namesakes that agree
status: in progress (Claude Code, 2026-10-27)
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
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
