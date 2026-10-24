# 0039 — A question reader: two small heads find the relation and the name; the lookup stays exact
status: proposed (queued after the inbox items of 2026-10-24: `mul` labels, the table's reading, namesakes at the source)
written: 2026-10-24 by Cowork · approved: 2026-10-24 by Nick ("yes") · hypothesis: none

## Why
0038's scoreboard (187 WebQuestions the everyday table reads; live source; tense check on): 47 spoken, 0 wrong. The
76 "couldn't find it" refusals, sorted (`validation/logs/everyday_webq_refusals_sorted_2026-10-24.json`), are almost
all **reading**, not data. 45 left extra words on the name ("cleveland ohio", "croatia 2012", "sweden yahoo"), and
none is a fact Wikidata lacks. Beyond the 187, the table does not read 2,029 of the 2,216 questions at all. A hand
table covers the wordings someone thought of. Nick's design note proposes multi-head embeddings on XLM-RoBERTa,
trained with LoRA on grilly. Two of its heads are exactly the reading job:
- the **intent head** says which relation the question asks for;
- the **POS/context head** says where the name is and which words only qualify it.

The facts stay where they are, looked up by exact keys. Only the reading is learned.

**What it brings, in plain words:** Cubby understands many more ways of asking a question, in English and French, in
milliseconds, and still answers only from a fact it looked up and checked.
- **Over a conventional model:** a model reads any wording but answers from memory, and the answer can be wrong.
  Here a reading mistake can only become a refusal or a checked lookup of the wrong relation. The gate measures the
  second case.
- **Speed:** an encoder classifies a question in milliseconds. The 450M takes about 6 s to write a plan, so for
  one-step questions the reader can do its job and leave the 450M the hard ones.

## Build
1. **Splits first.** Add the original WebQuestions split (train / test) to `webq_na_eu.csv`
   (`validation/builders/build_webq_bench.py`). The raw files carry it. Dev is a seeded 10% of train, and the seed
   goes in the log. **Test is held out**: no threshold, wording or label is ever tuned on it.
2. **Training data** (a builder in `validation/builders/`), from train only:
   - **Templates:** `relation_template_mapping.csv` (the hdc question templates) filled with names of hdc items that
     carry the relation. Exact labels: the relation, the name's span, any qualifier.
   - **The table's own parses** of train questions, kept only when the lookup's answer matches a listed answer.
   - **Distant supervision:** for train questions the table does not read, the (span, relation) pairs whose looked-up
     values match a listed answer. A question with two such readings is left out.
   - **Classes:** the relations the table reads (`everyday.EVERYDAY`), plus `other`. The values that change
     (`NOT_READ`: mayor, team) are `other`.
3. **The reader** (`cubbyllm/reasoning/reader.py`, flag `CB_READER=1`, off by default):
   - XLM-RoBERTa base as the shared encoder, with LoRA on grilly (`grilly.nn.LoRAModel`; Nick's note: the BERT export
     path through optimum-grilly is validated, XLM-R's is not yet).
   - The intent head is a classifier on the first token. The span head tags each token: name, qualifier, year,
     noise, other.
   - Same interface as `EverydayForms.restate()`: `(restated, relation, seed)` plus the qualifier, so the loop needs
     no new path. The qualifier picks the item through the existing "Paris, Texas" reading.
   - **It abstains below a confidence set on dev.** Abstaining means the question goes to the model as today.
   - Order in the loop: the table first, then the reader for what the table does not read.
4. **The scoreboard:** `exp_everyday_webq.py --reader on|off`, on the **test split**, with three arms:
   - the table alone;
   - the reader alone;
   - the table, then the reader.

   All three: the live source (or its cache), the tense check on, strict namesakes, no model in the seat.
   Optionally the 450M for the rest (report only).

## Acceptance (pre-registered)
- Tests:
  - the reader's restate on hand cases: "cleveland ohio" gives Cleveland with the qualifier Ohio; "croatia 2012"
    gives the year 2012;
  - abstention gives the old path;
  - the flag off changes nothing.
- **Gate:** on the test split, the table-then-reader arm speaks **at least 1.5 times** as many answers as the table
  alone, with **0 wrong**, every spoken answer read by hand.
- **Kill:** fewer than 1.2 times as many, or any wrong answer that traces to the reader (a wrong relation or a wrong
  name that the lookup turned into a checked fact).
- Report, no gate:
  - the reader's median time a question on grilly;
  - its accuracy on the relation and on the name, on dev;
  - how often it abstains;
  - a French check: the table's French wordings filled with test-split names.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **The facts stay exact keys.** The reader only reads the question. Nothing is stored or retrieved by vector
  similarity here.
- **A wrong reading must not become a spoken fact unchecked.** Say in the report how one is caught: the abstention
  threshold, and whether the relation needs a word of the question to support it (the table's words or the
  source's aliases).
- **Weights are local.** `cubbyllm/` never touches the network, and XLM-RoBERTa is a local encoder, not an LLM on the
  serving path.
  - **Download:** check the disk first. If it is not there, the base weights (about 1.1 GB, MIT licence) need Nick's
    yes in Cowork's chat before the download.
- Training runs in the background at low priority, around 0002's harness and 0035's run 5c2.

## Out of scope
- The semantic and emotion heads: for memory search and mood, a later order.
- Storing or retrieving facts by embeddings: facts stay exact keys (the inbox, 2026-10-24).
- Replacing the 450M, and multi-step questions.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
