# 0051 — Loci tokens in training: count what spans would save, run the queued places-and-route continuation, then decide spans
status: in progress
written: 2026-11-04 by Cowork · approved: 2026-11-04 by Nick ("what about using loci tokens in training?", then "yes") · hypothesis: H-K10 (with H-S12, H-C2)

## Why
- **Shown already:** a fact read back by its address on the 450M. Run 5 reads far facts at 0.946 exact, 800+ tokens
  back, by training the copy path's 1,027 parameters for about 10 minutes, with the text unchanged (orders 0001,
  0005). The learned route reaches 0.82 when the route loss teaches it.
- **Queued in H-K10 and not run:** the 450M continued on `spatial_here` + `facts` with `CB_SPATIAL=1
  CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1`, against the text controls at the same tokens. That covers H-S12's gates 1-2
  and H-K10's gate 4 in one run; spans (gate 3) come after.
- **The training-cost case is gate 3.** A fact-bearing span (a name, a fact) becomes one input unit,
  `sum_i e(piece_i) (x) S(i)`, up to about 55 pieces at d = 1,024 by the capacity law.
  - Fewer positions per document make every part of a training step cheaper, the head included.
  - These are spans the trunk should not learn to write from its weights: names and numbers come through slots and
    the copy path.
- **What it saves is not measured.** This order counts it first, for free, as H-C2 counted the hypertokens.

**What it brings, in plain words:** Cubby reads a whole fact as one unit instead of a dozen word pieces, so training
spends its compute on language, not on re-reading and memorizing facts.
- **Over a conventional model:** a model pays for every piece of every fact, in training and at each use, and stores
  the facts in its weights. Here a fact is one unit that points into the store.

## Build
**A. The span count** (CPU, no training):
- **Where:** the 13 held-out tails of the training mix, and a sample of 0045's slice.
- **The spans,** per source, as shares of tokens:
  - names: the longest exact matches in R1b's names segment, the host choosing the boundary as H-K10 says;
  - optionally, fact spans that the store's statements and the frozen lexicon recognize (name, relation word, value);
  - each span capped at 55 pieces, with no overlaps.
- **Reported:**
  - the share of tokens folded;
  - the positions left per document;
  - the share of loss positions the trunk would no longer predict (inside a span, nothing is predicted; the price,
    by design, is that the trunk stops learning to spell names);
  - the projected step-time saving from the bench's per-token costs, backbone and head apart (0050's measurement).

**B. The queued continuation,** as H-K10 writes it:
- **Two arms at the same tokens:** the link (`CB_SPATIAL=1`, `placed`, the route loss) and the text control.
- **The budget:** about 0.1-0.2B tokens an arm, as runs 1-4 had, so about an hour each on an A100.
- **Training:** on CUDA in torch. The link has no grilly2 form yet (`export_base.py` refuses a `spatial` meta).
- **Who runs it:** a Colab notebook that Nick runs; it can share the session he opens for 0045's bench.

**C. Spans as input units:** not built here. It is written as its own order from A's and B's numbers, and only if:
- A folds at least 10% of the mix's tokens; and
- B keeps the text within 0.02 of the control.

## Acceptance (pre-registered)
- **Tests:** the span finder on a toy text (longest match, the 55-piece cap, no overlaps, deterministic). With
  `CB_SPATIAL` off, every output is bit-identical.
- **A:**
  - the table per source, with the mix's whole;
  - **the line for C:** at least 10% of the mix's tokens folded. Under it, spans are recorded as not worth it for
    training cost (they may still be for recall).
- **B:** H-S12's gates 1-2 and H-K10's gate 4, as those entries write them.
  - **H-S12's kill applies:** gate 1 below 75%, or a text loss worse than the control's by more than 0.02.
  - **Reported beside them:** tokens per second and MFU for both arms, and the compute units spent.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **0045 is not changed.** Its control runs as pre-registered.
- Nothing reaches the serving path.
- Claude Code does not start a paid run; the Colab notebook is Nick's.
- grilly2 for anything on the local card, never grilly v1. No local path in a committed file.

## Out of scope
- **C (spans as input units):** its own order, from A and B.
- **The output side** (predicting a loci token): a vector cannot hold "70% Paris, 30% Lyon". The output stays word
  pieces, and 0050 handles its cost.
- **Porting the link to grilly2:** its own order, if B passes.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**

## A, the span count (Claude Code, 2026-11-04)
- **Log:** `span_count_2026-11-04.json` (`validation/span_count.py`, CPU). 4 windows of 1,024 tokens from each of the
  13 held-out tails, and the same from 0045's slice (training region, evenly spaced): 53,248 tokens each.
- **The spans:** the longest exact matches in R1b's names segment (`cubelang knowledge batch`).
  - Word-aligned, leftmost first, at most 8 words and 55 pieces, no overlaps.
  - A span starts and ends on a letter or digit (punctuation is never folded in).
  - Only spans of two or more pieces fold anything.
- **Two readings,** since nearly every common word is some item's alias:
  - **(a) every match:** an upper bound, mostly language words ("the following", "most often", "a file").
  - **(b) proper names:** first and last words capitalised as written. A single word opening a sentence or a line
    needs a second word, since every sentence starts with a capital.
  - What (b) folds: "British Compressed Gases Association", "Raspberry Pi OS", "Redundant Array of Independent
    Disks", "SSL3". Some title-case heading words remain ("Colour Coding", "Distributors"), so (b) is if anything a
    little high.
  - H-K10 says language words stay words, so (b) is the reading for the line.

| | (a) folded | (a) saved | (b) folded | (b) saved | (b) positions left / 1,024 |
|---|---|---|---|---|---|
| held-out tails, equal per source | 32.3% | 19.0% | 6.1% | 3.9% | 984 |
| held-out tails, the mix's weights | 35.3% | 20.7% | **5.2%** | 3.4% | |
| 0045's slice, equal per source | 34.2% | 20.1% | 7.4% | 4.7% | 976 |
| 0045's slice, the mix's weights | 35.2% | 20.7% | **6.9%** | 4.4% | |

- **Columns:**
  - "folded": the pieces inside spans;
  - "saved": positions saved, the sum of (pieces - 1). It is also the share of loss positions no longer predicted
    (inside a span nothing is predicted).
  - "The mix's weights": the stable phase's (`train_base`: size x QUALITY).
- **Per source, (b) folded** (held-out): wikipedia 11.3% and wiki_full 11.8% at the top; fineweb_edu 8.0%; books
  4.8%; nemotron_code 1.0%; wikibooks 1.6%.
- **The projected step-time saving:**
  - Every part of a step scales with positions: the trunk, and the head too when a span's inner positions carry no
    loss.
  - So the saving is at most the saved share: **about 3.4% to 4.4% under (b)**, and about 21% under (a).
  - 0050's A100 numbers will split it between trunk and head; they cannot raise it above the saved share.
- **The line for C** (at least 10% of the mix's tokens folded):
  - **not met under (b), 5.2% to 6.9%.**
  - Met only under (a), which folds language words, against H-K10's own boundary.
  - **Spans are recorded as not worth it for training cost by names alone.** They may still be for recall.
- **Not counted:** the optional fact spans (name, relation word, value). They would add the relation words and values
  next to the names, which (b) already bounds from below. I can count them if you want the upper side of (b).
- **Tests:** `tests/validation/test_span_count.py`, 6 passed:
  - the longest run wins;
  - no overlaps;
  - deterministic;
  - the 55-piece and 8-word caps;
  - punctuation and whitespace stay out;
  - a capitalised sentence opener is no name alone;
  - only runs of two or more pieces fold.
- **B:** the notebook comes next. `CB_SPATIAL` off is bit-identical by the link's own tests (`tests/model/test_spatial.py`).

## Cowork's check of A, and of B's proposal (2026-11-04, evening)
- **A: C's line is not met, and the record keeps it.** Proper names fold 5.2% of the mix's tokens (0045's slice: 6.9%),
  saving 3.4-4.4% of positions. A training step would get at most about 4% cheaper.
  - Spans as input units are therefore not a training-cost lever. They stay a recall mechanism (H-K10).
  - Fact spans are not counted. Beyond the names, they would mostly fold relation words, and H-K10 keeps language words
    as words.
- **B: run 6 as you propose,** both arms from `base450m_cont_stable_end.pt` at equal tokens, about an hour each on an
  A100:
  - **link:** run 4's flags (placed, writes binding, the route loss, the copy ring; tau 3.04, beta 0.115), with the
    trunk training;
  - **text:** `spatial_text_here` + `facts_far1k_text`. That is the far-fact text control no run has had.
  - **Gates:**
    - gate 4 as a real comparison: link at least 2x text, and at least 0.5;
    - gate 2: the text within 0.02 of the control;
    - gate 1's text clause reported, not gated (in-window villages make it unmeetable by construction, as runs 1-3
      showed).
  - It goes in `spatial_facts_450m.ipynb` and can share Nick's next A100 session with 0050's notebook.

Status: in progress (B).
