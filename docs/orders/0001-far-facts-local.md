# 0001 — Far facts: reproduce the 450M failure on a CPU toy, then fix it there
status: in progress
written: 2026-10-21 by Cowork · approved: 2026-10-21 by Nick · hypothesis: H-K10 (and H-S12)

## Why
The spatial link reads **villages** at 0.953 exact on the 450M (run 3) but **facts 800 tokens past
the window** at 0.021 (`validation/logs/hypertoken_recall_spatial_run3_link_2026-10-21.json`). The
binding bug is fixed: answer pieces now score 0.309 with the places and 0.300 without
(`diag_fact_read_link3_2026-10-21.json`; run 2 was 0.096 against 0.213). So nothing hurts the facts
any more, but the read also adds almost nothing to them. On a 3.4M toy, the same channel read
facts beyond the window at **0.94** (`validation/exp_k10_hypertoken.py`, `exp_k10_hypertoken_placed_2026-10-20.json`).
Between the toy that works and the 450M that does not, six things differ:

| | toy (works) | 450M facts (fails) |
|---|---|---|
| answer | 1 token of 509 names, vocab 1,024 | 1–5 sub-word pieces of a wiki name, vocab 128k |
| trunk | trained from scratch with the link | trained first, link added (`o = 0`, gate 0) |
| supervision | 8 answers in a 256-token document (~3% of tokens) | ~25 answer tokens in 1,024, on 4.6% of the mix (~0.1% of tokens) |
| depth after the read | a few layers | 16 of 32 layers, into an untied 128k head |
| width | 256 | 1,024 |
| villages on the same 450M | — | **work** (a few dozen names): depth and "added late" alone do not break a closed set |

A Colab run costs ~18 units and gives one number. This order finds the cause and tests the fix
on the CPU first, so the next GPU run tests a fix that has already worked somewhere.
What it brings: a trunk that answers a fact it saw once, 800+ tokens back, by its address — with no
KV cache, which a conventional model cannot do past its window.

## Build
**A0. Is the answer in the read at all? (no training, minutes)** On the run-3 checkpoint
(`spatial_link_3_final.pt`, see `CLAUDE.local.md`) and `facts_far1k_lead` held-out documents: at each
answer token, unbind the scene at its address (`SpatialLink.lookup`, before the gate and `o`) and
**clean it up** against the table the scene stored (check what `HybridEmbedding.embed` returns: the
stored vector includes the context part). Report the target's rank among the 128k (top-1 / top-5),
cos(read, target) against the best wrong token, and the same for `spatial_here`. This is the qFHRR
paper's decoding (argmax over a candidate codebook, `docs/papers/2026-08-07-qfhrr-semantic-word-tables.md`
§3, §5): there it gave 0.99 precision and a 17× bound-vs-absent margin. If the target is top-1 most of
the time, the information is in the read and the failure is the trunk's decoding, so B1 should work
even before training. If it is not, the crosstalk or the correlated learned embeddings are the
problem, and B4 is the lead.

**A. The ladder** — `validation/exp_k10_far_facts.py` (new, < 300 lines; reuse `exp_k10_hypertoken.py`'s
document builder and model setup by import, do not grow that file). Start from the configuration
that worked (`--query placed --gate-bias 0 --route-loss 1`, D 256, the 2026-10-20 budget) and move
**one** difference at a time toward the 450M:
1. `--vocab N --entities N` — answers drawn from a large vocabulary (try 1,024 → 8,192 → 32,768).
2. `--pieces k` — multi-token objects at the fact's address + `6 i` in z, laid out like
   `make_fact_shard.py --lead`: the token before the answer asks for piece 0, piece `j` asks for `j + 1`.
3. `--pretrain S` — train the trunk text-only for `S` steps, then add the link with `o = 0`, gate 0
   (what `CB_INIT_FROM` does), then continue.
4. `--sparse` — fewer questions per document and more filler, so answer tokens are ~0.1–0.5%.
5. `--at` and `--layers` — more layers after the read (e.g. L 8, read at 4 vs at 7).
Then the combination closest to the 450M. Report exact and per-piece accuracy (first piece, later
pieces) like `diag_fact_read.py`, and the filler language's held-out loss, at the same step counts.

**B. Candidate fixes**, each tested on the arm(s) that failed in A:
1. **Copy path** — behind a flag in the real model (`cubbyllm/model/spatial.py` + `assembly.py`;
   `CB_SPATIAL_COPY=1` / `meta["spatial"]["copy"]`, absent by default). At ASK tokens, the read's
   unbound vector `r` (the scene ⊛ `S(p)`⁻¹, before `o`) is scored against the token-embedding table
   and added to the logits, gated: `logits += σ(g(h)) · exp(τ) · (r · Eᵀ)`, with the gate's bias
   starting closed so a model with it added computes exactly what it did. Check what
   `HybridEmbedding.embed` returns — the scene stores the embedding *with* its context part — and
   score against what was actually stored. Decode (`step`) must match the forward.
2. **Read nearer the head** — `at` = L − 2 instead of L / 2.
3. **Denser supervision** — more questions per document (a data change, the cheapest if it is the cause).
4. **Identity codes in the scene** — the qFHRR paper's blend: write `normalize(0.5·e + 0.5·id(token))`
   bound to `S(p)`, where `id` is a fixed hashed random phasor code per token id, and clean up against the
   `id` table. Quasi-orthogonal by construction, whatever the trunk's learned embeddings look like (the
   paper lists capacity under learned, correlated codes as open). Flag-gated like B1.

**C. Only if B passes:** a `RUN = 4` configuration in `notebooks/_build_spatial.py`, using the winning fix,
for Cowork to run on Colab. Regenerate the notebook.

## Acceptance (pre-registered)
- **A reproduces it:** the starting configuration reads ≥ 0.90 again, and at least one ladder step
  (or the 450M-like combination) drops to ≤ 0.20. If **no** step drops it, stop and report that: it
  means the cause is scale or budget, not mechanism, and it changes what we run next.
- **B fixes it:** a fix lifts the failing arm to **≥ 0.80** exact at the same budget, with the filler
  language's held-out loss within 0.02 of the arm without it. Kill: below 0.50.
- **Tests** for the copy path: off = bit-identical forward and decode; added to a trained model with
  the gate closed = the same logits at step 0; with the gate open and one token written at an
  address, the ASK position's top logit is that token; decode = forward. Full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. CPU only, Nick's machine: keep each arm to roughly the 2026-10-20 budget
  (~25 min at D 256); arms can run in parallel processes; a CE over 128k is too slow on the CPU, so
  stop the vocabulary at 32,768 unless a smaller step already explains it.
- Same seeds, steps and budgets across arms, so they can be compared.
- Logs to `validation/logs/exp_k10_far_facts_<arm>_<date>.{json,log}`.

## Out of scope
- Any GPU or Colab run (Cowork runs it once C exists).
- Changing `CUBBYLLM_HYPOTHESES.md` (Cowork records the result from the report).
- Retuning the route loss or the learned query: the question here is the read, with the address given.

## Addendum — 2026-10-21, from the literature search (Cowork)
The search (brief: `docs/research/2026-10-21-far-facts-research-brief.md`) ranks the fixes as this order
does: a direct path from the read to the logits first (B1), then a later read (B2), then density (B3),
with the cleanup check (A0) as the diagnostic. It adds the items below. **Acceptance is unchanged**: new
ladder steps are judged by the A lines, new fixes by the B lines. A0 and steps 1–5 stand as written.

**A, step 6. `--gap N`: the distance from the facts to their questions**, from the toy's (~120 tokens past the
window) to the 450M's (~800). Boesch & Wee 2026 (arXiv 2609.16183) find that fixed-state recurrent cells
which solve 32-pair recall drop to chance when 4 pairs sit in a distractor haystack. They trace it to
interference under sparse supervision, not capacity: denser supervision added nothing, while a distance
curriculum gated on measured accuracy took the same architecture from 0.021 to 1.000. Our scene sums only
placed tokens (`SpatialLink.scene`) and does not decay, so the read itself should not care about distance.
Whether the trunk learns to *use* the read can still depend on it, and the step is cheap. Their cells are
linear-attention and SSM recurrences, not this channel, so the analogy is conjecture until step 6 measures it.

**A0, one note.** Kumar 2026 (arXiv 2606.24948, HRR/FHRR on FB15k-237) finds single-hop retrieval works and
places the failures in capacity under superposition, not the algebra or the cleanup. It also proves that a
softmax cleanup over FHRR phases is not phase-equivariant (its Lemma 4.1). Score A0 with the real dot
product / cosine against the table (argmax), as written, not with a softmax over phases.

**B3, one more arm: answer-token weight.** The CE at answer tokens is multiplied by w (try w = 5), as the
same-data alternative to more questions per document. Given Boesch & Wee's result, B3 is not presumed to be
the cheap win: measure it like the others.

**B5. Supervise the read** (with B1; flag `CB_SPATIAL_COPY_AUX=<λ>` / `meta["spatial"]["copy_aux"]`, absent by
default). At ASK positions, add a CE of the copy logits alone, `exp(τ)·(r·Eᵀ)`, against the next token, with
weight λ (try 0.1 and 1). Why: B1's gate starts closed, so the main loss reaches τ and the scored table only
through σ(g) ≈ 0. The copy path can then stay shut because nothing trains it. The auxiliary loss trains it
to decode the address first, so the gate has something to open onto. Arms: B1 alone vs. B1 + B5 at the same
budget. Also report cos(r, stored target) as a metric. (The search also proposed `1 − cos` as the loss; the
CE scores against the whole table, which is what decoding needs.) It moves the embedding table too: the
filler-loss line in Acceptance catches it if that costs the language.

**B7. Distance curriculum**, only if step 6 is a step that breaks it. Start the questions near their facts
and move them out each time measured exact accuracy passes a threshold (Boesch & Wee's accuracy-gated ramp;
their uniform ramp failed at their longest length).

**Two things not to build here:**
- `--pretrain` (step 3): the link's `o` starts at 0, but that is not a cold start. The gradient to `o` is
  δ·rᵀ, nonzero whenever the read is, and villages learned from the same init on the 450M. Add one arm
  with a small nonzero `o` (std 1e-3) only if step 3 is the one that breaks.
- **Entity-level storage** (one vector per object instead of per-piece addresses): if step 2 (`--pieces`)
  is the one that breaks and B1/B5 do not lift it, stop and report. It is a design change, not a flag,
  and Cowork writes it as its own order.

**Sources checked:** See, Liu & Manning 2017, pointer-generator networks (arXiv 1704.04368), for the gated
copy distribution B1 follows. Khandelwal et al. 2020, kNN-LM (arXiv 1911.00172); the search's kNN-LM link,
arXiv 2102.02557, is in fact Yogatama et al. 2021, "Adaptive Semiparametric Language Models" (a gated
memory LM, also relevant). Boesch & Wee 2026 (arXiv 2609.16183) and Kumar 2026 (arXiv 2606.24948), as above.

## Addendum 2 — 2026-10-21: the read is at capacity for a 128k-way decode (Cowork)
`validation/sim_hrr_ceiling.py` (`validation/logs/sim_hrr_ceiling_2026-10-21.json`) gives the read's
best case: random codes and a perfect argmax decoder. It also counts the real shards, and the two
regimes write about the same load but differ in what the decoder must choose among:

| | items in a scene (mean / p90) | distinct written tokens | ceiling, top-1 per piece |
|---|---|---|---|
| villages (`spatial_here`) | 32.4 / 51 | **40** | 1.000 (37 items, 40 candidates) |
| far facts (`facts_far1k_lead`) | 37.1 / 50 | **47,670** | **0.765** (37 items, 131,072 candidates) |
| far facts, scene split in 2 | ~19 | | 0.995 |
| far facts, scene split in 4 | ~9 | | 1.000 |

An answer has ~4.6 pieces (from the shard's ask counts) and exact needs every one, so even a perfect
decoder with ideal codes would sit near 0.3 exact. Learned, correlated embeddings and a trunk that has
to learn the decoding only lower that. The ceiling alone does not explain 0.021: the read adds ~0.01
per piece, so the trunk is not decoding it at all, which B1 is for. But it says B1 alone cannot pass the
0.80 line, and it predicts which ladder steps break the toy: 1 (`--vocab`) and 2 (`--pieces`). **Report
the ceiling next to every ladder and fix number** (`sim_hrr_ceiling.py --d 256 --items <the arm's mean
writes> --candidates <its vocabulary>`).

**A0, extended (still no training, still minutes).** Rebuild the scene offline from the checkpoint's
stored vectors and report per-piece top-1 and exact for each cell of
{stored: as written | random identity codes (B4's best case)} × {K = 1, 2, 4 sub-scenes, chosen by a hash
of the full place, so an object's pieces spread} × {candidates: the 128k vocabulary | the tokens written
in this document so far}. This tests every read-side fix below before anything trains.

**B8. Address-partitioned scene** (`CB_SPATIAL_PARTS=K` / `meta["spatial"]["parts"]`, absent = 1 =
bit-identical). K sub-scenes per document. A WRITE goes into, and an ASK reads from, sub-scene
`hash(place) mod K` only, so each read sees ~1/K of the crosstalk. The routing is by the address that
both sides already hold, so there is no router to learn. In MoE-moryLLM (S. Rathi, a MemoryLLM fork;
GitHub README, not peer-reviewed), oracle routing was the largest gain: +72.5% memory retention against
+18% for round-robin, while their learned (REINFORCE) router failed to train. Their forgetting is
eviction from a token pool, not superposition, but the lesson carries: partition the store, and route
by what the writer and the reader both know. Cost: K·d of scene state per layer, still fixed-size (no
KV cache), and one sub-scene unbound per read. Decode = forward; K = 1 = off. Test with B1: B1 + B8 is
the lead pair.

**B9. A document-local candidate set** (measure in A0; **do not build** without Nick). B1's copy logits
are scored only against the tokens written in this document so far, which puts the facts in the villages'
regime (~37 candidates). It means keeping a growing list of written token ids: small, ints not hidden
states, but not fixed-size, so it is Nick's call on the decode-state rule.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <the ladder table; the fixes table; against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
