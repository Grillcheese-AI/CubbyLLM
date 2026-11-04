# 0048 — The programs adapter names Wikidata's properties (EN and FR), and refuses the wrong base
status: open
written: 2026-11-04 by Cowork · approved: 2026-11-04 by Nick ("Yes, write the order") · hypothesis: H-E15 (the programs seat)

## Why
0046's R3 and R3o measured the 450M in the programs seat (`base450m_cont` + `cubby450m_v12e_w_step_cont`) reading
which property a question asks for, given the item (`retrieval_R3_dev_2026-11-03.json`,
`retrieval_R3o_dev_2026-11-04.json`):
- It writes a usable program for 53% of SimpleQuestions-Wikidata and 29% of QALD-9-plus FR.
- **132 of 300 programs bind a `HOP1` wording that names no property:** "film" 15, "album" 12, "born in $E1" 10,
  "birth" 6. These are the wiki world's relation words, which v12e taught it, not Wikidata's labels. Among the item's
  own properties (R3o) they still name nothing.
- **Where its wording does name a property, 66% hold the gold.** So the model reads the question; it names the
  relation in the wrong vocabulary.

Adapters train in minutes on Colab, so the fix is to teach the vocabulary: Wikidata's property labels as the targets.

The order also adds a guard. On `base450m`'s export the programs adapter writes no program, because it was trained on
`base450m_cont`. Serving takes the base on the command line and nothing refuses the wrong one.

**What it brings, in plain words:** Cubby's own model learns to name what you are asking about in the store's own
terms, in English and French, so every relation it names can be looked up and checked.
- **Over a conventional model:** a model answers in its own words, and nothing checks them. Here its words are the
  store's property names, so each one is looked up and verified, and a wrong one finds nothing to say.

## Build
1. **The data** (`validation/emitter_wd_data.py`; tests beside it):
   - **Questions from the sets' train splits only:** SimpleQuestions-Wikidata, QALD-9-plus (EN and FR) and Mintaka
     (EN and FR). No question or item from 0046's dev or test rows enters (filtered by question id and by item id).
   - **The gold property** by R3's rules (`r3_gold`): the set's own property; QALD's single property once P31 and P279
     are dropped; Mintaka's linking property where there is exactly one. Other rows are left out and counted.
   - **The record:** the question with its item as a slot (`[E1: ...]`, as H-E15), and the v12e program shape with
     `HOP1`'s relation written as the property's label in the question's language, from the frozen lexicon.
     - Inverse rows (asked from the value's side) use the format's own inverse form, if it has one. If it has none,
       say so; they are then left out and scored apart.
   - **v12e's records stay in the mix,** with their share reported, so the adapter keeps its other jobs (arithmetic,
     kernels, several hops).
   - **The manifest** records each set's licence, its counts, the share per set and language, and every file's
     sha256. A set without a clear licence is left out.
2. **The training:** `train_emitter_torch.py` with `cubby450m_v12e_w_step_cont`'s settings, on `base450m_cont`, from a
   Colab notebook (`notebooks/emitter_wd.ipynb`) that Nick runs. Its output is `cubby450m_wd_cont`, next to the current
   adapter, which it does not replace.
3. **The guard** (serving and every emitter loader):
   - An adapter records its base's sha256 in `emitter_lora.json` (and the talk adapter's file likewise).
   - At load, a mismatch refuses, with both hashes and both names in the message.
   - An adapter with no recorded hash loads with a warning. A one-off tool writes the hash for the known pairings
     (`cubby450m_v12e_w_step_cont` on `base450m_cont`), checked by loading and emitting one program.

## Acceptance (pre-registered)
- **Tests:**
  - the data builder: no dev or test question or item gets in; the label is in the question's language; the gold
    rules are R3's;
  - the guard: a matched base loads, a mismatched one refuses, a missing hash warns.
- **R3's harness, dev only, a new arm `450m_wd`,** in both settings: all properties (R3) and the item's own (R3o).
  - **Coverage first:** the share of rows whose program's `HOP1` names a property, per set, against the current
    adapter's (53.3%, 42.3%, 28.8%, 27.7%, 21.0%).
  - **Gate,** in the R3o setting: recall@5 at least 0.85 on SimpleQuestions-Wikidata and 0.80 on QALD-9-plus FR (R3's
    gate). Both its wrong rate and its missing-fact handovers are at most lexicon_content's, on the same rows.
  - **Kill:** coverage not above the current adapter's on SimpleQuestions or on QALD FR. The labels did not teach
    the wordings.
- **No regression on the adapter's own jobs** (H-E15's benchmark): E8 reworded spoken-and-correct at least the
  current adapter's, spoken-wrong at most 1%, 0 copied literals run. A regression rules the adapter out, whatever R3
  says.
- **Test** (R3's test rows) is read once, and only if dev passes.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model writes or rewrites a training record.
- Claude Code does not start the Colab run; Nick does.
- **The new adapter is measured, not served.** Serving keeps `cubby450m_v12e_w_step_cont` until Cowork has read the
  report.
- **The guard is the one serving change.** It only refuses a load that cannot work, as 0047's exception allowed.
- Scoring on the local card runs on grilly2, never grilly v1.
- No local path, account or repo name in a committed file.

## Out of scope
- A French-aware word table: after 0046's R4.
- R3k (the asked kind): 0046.
- 0045's adapters phase: unchanged. If this order passes, its data replaces v12e's in the next retrain, on the
  control's final.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
