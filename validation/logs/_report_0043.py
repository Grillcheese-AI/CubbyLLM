"""Scratch (untracked): write order 0043's report, the queue line and the inbox reply."""
import io

REPORT = '''- **Commits:** `869b8ae` in progress; this one: the rule, the loop's side, the tests, the audit, the measurements.
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
'''

p = 'docs/orders/0043-namesakes-at-the-source.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('status: in progress (Claude Code, 2026-10-27)', 'status: review (Claude Code, 2026-10-27)')
if '- **Commits:** `869b8ae`' not in s:
    s += REPORT
io.open(p, 'w', encoding='utf-8').write(s)

p = 'CLAUDE.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("0029 (arm D: step 2); open, in this order: 0043 (namesakes at the source), 0044 (several values for one\n  relation), then 0039",
              "0029 (arm D: step 2); review: 0043 (namesakes at the source: the rule on the Wikidata side, `CB_SOURCE_NAMESAKES=1`);\n  open, in this order: 0044 (several values for one relation), then 0039")
io.open(p, 'w', encoding='utf-8').write(s)

p = 'docs/inbox.local.md'
s = io.open(p, encoding='utf-8').read()
note = ('  - `Claude Code:` 2026-10-27 (later) -- 0043 in review: the rule at the source behind `CB_SOURCE_NAMESAKES=1`; the audit\n'
        '    (50 names, all from the scoreboard and the stack) and the live scoreboard (82 spoken, 0 wrong by my reading) are in\n'
        '    the order\'s report, with two refinements the live pass forced and three questions for you.\n')
if '0043 in review' not in s:
    s += note
io.open(p, 'w', encoding='utf-8').write(s)
print("ok")
