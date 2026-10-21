# 0001 — Far facts: reproduce the 450M failure on a CPU toy, then fix it there
status: review
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

## Addendum 3 — 2026-10-21: after A0, Nick's call, and a second literature pass (Cowork)
**What A0 settled** (`f3033a9`, `validation/logs/diag_fact_cleanup_run3_2026-10-21.json`):
- The answer is in the read: 0.758 per piece over the 128k vocabulary, and 0.992 per piece / **0.965 whole
  answers** among the document's own writers. The trunk reads 0.021.
- The learned table behaves like random codes: cos to the target is 0.165 against 1/√37.5 = 0.163, and the
  per-piece score matches `sim_hrr_ceiling.py`'s 0.765. **B4 is dropped**: identity codes cannot beat random.
- The villages never copy: the trunk maps one closed set to another (a stored "pond" is answered " pond").
  The facts need a copy, and the trunk has never learned one. That is B1's job, and the candidate set
  decides its ceiling.

**Nick's call (B9): a fixed ring of written ids is allowed.** Each stream keeps the last **256** token ids
it WROTE, in a ring like the attention window's. Decode state stays fixed-size; a document writes at most 92.
In training, the candidates at position t are the ids written at positions ≤ t in the same document (reset
at `starts`), capped at the last 256.

**The lead fix is now B1 over the ring** (flags as in B1, plus `CB_SPATIAL_COPY=ring` or
`meta["spatial"]["copy"] = "ring"`). At ASK tokens it scores the read `r` by cosine against the stored
vectors of the ring's ids, and mixes the result as a pointer distribution. This is the form of the copy
paths that work in the literature: pointer-sentinel (Merity et al. 2016, arXiv 1609.07843), pointer-generator
(See et al. 2017), kNN-LM:
`p = (1 − λ)·p_LM + λ·p_copy`, with `p_copy` = softmax over the ring of `exp(τ)·cos(r, e_c)`, scattered onto
the candidates' ids (duplicate ids summed), and `λ = σ(g(h))`.
- **λ starts at 0.1, not closed.** kNN-Adapter (Huang et al. 2023, arXiv 2302.10879; init as the search
  reported it, not checked here) starts its mixing weights at 0.1–0.25. The copy acts only at ASK tokens,
  so an open start costs nothing anywhere else. The gradient to λ is `(p_copy(y) − p_LM(y)) / p(y)`, so the
  gate opens as soon as the copy is right more often than the trunk.
- **τ starts so that A0's median gap**, between the target and the best wrong writer, is about 2 nats.
- **Tests.** The step-0 identity holds at every non-ASK position, and with the flag off. The rest are
  as in Acceptance: decode = forward, and with one token written at an address, the ASK position's top
  token is that token.
- **Arms:** the vocabulary form as written in B1 is a second arm, and its ceiling is 0.457 whole answers.
  B8 (the split scene) is the fixed-state alternative. Measure it offline first in A0's K grid; build it
  only if B1 over the ring misses the B line.
- **B5 becomes conditional.** No paper in either pass trains the copy distribution with its own loss.
  Build it only if λ at ASK tokens stays below 0.05 at half budget.

**The ladder, narrowed.** With A0 in, A's job is to give B a toy arm that fails (≤ 0.20). Start with the
450M-like combination. Walk single steps back only if B's result needs explaining. Step 6 (`--gap`) stays
a cheap check. If it breaks, B7 advances the gap cap only while a running probe accuracy stays above 0.6,
never retreats, and evaluates at the maximum gap (the search's reading of Boesch & Wee; the paper's HTML
confirms the gate is success-based, not the number). Their setting learns keys from scratch and has no
given-address or pretrained variant (checked), so it does not override A0.

**Seeds.** Boesch & Wee find recall outcomes bimodal and count seeds that lock in: dense supervision
alone 1/10, the curriculum 7/10 (checked in the paper). One seed can pass a fluke. The failing toy arm
and the winning fix run on **3 seeds**, and a B line passes when **at least 2 of 3** meet it. Other arms
run on one seed.

**C, cheaper.** Since the 450M's read already carries the answer, RUN = 4 continues from
`spatial_link_3_final` with B1 over the ring added. Only λ's gate and τ are new, so give it a short budget,
with a probe at each eval. Cowork sets the budget from the toy's curve.

## Report
<!-- Claude Code fills this in, then sets status: review -->
**In one line:** the read already carries the far facts; the trunk never learned to copy them out. A copy over the
last 256 ids the document wrote, with a gate that sees whether the asked address matches anything, takes the
450M-like toy from 0.000 to 0.930 / 0.908 / 0.895 exact (3 seeds), the language unchanged. Added to run 3 with
nothing trained, it lifts the far facts from 0.000 to 0.258 exact. `RUN = 4` is written for Cowork.

- **Commits:** `42450b0` claimed · `f3033a9` A0 · `f8b51c1` B1: the copy over the ring + the toy ladder · `7aa1077` A0
  extended (the grid), run 3 + the copy at step 0, `train_base` flags / `init_copy` / the eval probe · `4f672f5` the
  read-aware gate (`copy_beta`) · `1c51616` the ladder + the first fix arm · `279e99d` B on 3 seeds + C (`RUN = 4`) ·
  `c4b11ba` the controls, the table arm, run 3 + the read-aware ring · and the commit carrying this report.
- **Files:** new `cubbyllm/model/spatial_copy.py` (the copy path), `tests/model/test_spatial_copy.py`,
  `validation/diag_fact_cleanup.py` (A0), `validation/diag_fact_cleanup_grid.py` (A0 extended),
  `validation/diag_copy_step0.py`, `validation/exp_k10_far_facts.py` + `validation/k10_far_docs.py` (the ladder; its
  documents are `exp_k10_hypertoken`'s exactly at that file's settings, checked at every start). Changed:
  `cubbyllm/model/spatial.py` (`copy=`, `add_copy`), `cubbyllm/model/assembly.py` (`_trunk`; the copy in `forward` /
  `step`; the ring in `init_state`), `validation/train_base.py` (`CB_SPATIAL_COPY`, `_TAU`, `_BETA`; `init_copy`;
  `CB_PROBE_SHARDS` / `fact_probe`), `notebooks/_build_spatial.py` + `spatial_facts_450m.ipynb` (`RUN = 4`).
- **Tests:** `tests/model/test_spatial_copy.py`, 11 tests (17 cases), both modes where it applies. Flag off: nothing is built,
  and forward and decode are unchanged. Copy added: every non-asking position is bit-identical, and the asking rows
  are a normalized log-mixture. Gate open with one token written: that token comes out on top, in forward and
  decode. Decode = forward with the copy open, with roles, a document restart, and a ring of 2 under 3 writes. The
  ring holds each document's last writers (checked by brute force). No candidate: the trunk's logits are kept. The
  gate and τ get gradient (this test caught a NaN in τ's gradient before any run used it). The read-aware gate
  opens where the address holds something, not where it holds nothing. `CB_INIT_FROM` adds the copy to a linked
  checkpoint, and the non-asking positions are the checkpoint's function. The meta keys build it. Full suite:
  406 passed, 5 skipped, 1 failed: the known wiring guard, order 0002.
- **Measured.** All toy numbers: D 256, 1,000 steps × 16 documents, CPU, teacher-forced. "exact" = every piece of
  the answer; "end" = the EOS after the name. Ceiling = `sim_hrr_ceiling.py`, per piece, random codes, ideal argmax.
  Logs: `validation/logs/*_2026-10-21.{json,log}` under the names given.

  *A0, run 3, no training* (`diag_fact_cleanup_run3`, `diag_fact_cleanup_grid_run3`, `…_grid_beta_run3`), 9,451 piece reads
  on 400 held-out `facts_far1k_lead` documents:

  | candidates → | 128k vocabulary: piece / whole answer | the document's writers: piece / whole answer |
  |---|---|---|
  | stored, K = 1 (as run 3 has it) | 0.758 / 0.457 | **0.992 / 0.965** |
  | stored, K = 2 / 4 sub-scenes (B8) | 0.954 / 0.859 · 0.999 / 0.996 | 1.000 / 0.999 · 1.000 / 1.000 |
  | random identity codes, K = 1 (B4) | 0.759 / 0.436 | 0.991 / 0.961 |

  cos to the stored piece 0.165 against the best wrong 0.138, and 1/√37.5 = 0.163: the learned table behaves like
  random codes. The writer gap is 0.096, so τ = 3.04. Present reads have target cos 0.164, absent reads a best of
  0.066 among the writers, so β = 0.115. The villages: 0.821 per piece, 0.947 among the writers. The answer token
  is never the stored one (` pond` against `pond`), and 2,757 of the 3,610 village questions answer "nothing".

  *A, the ladder:*

  | arm | exact | first / later / end | language | ceiling (vocabulary / written) |
  |---|---|---|---|---|
  | the starting configuration (`--query placed --gate-bias 0 --route-loss 1`) | 0.877 | – / – / 1.000 | 2.075 | 0.767 (16 items: HERE writes too) |
  | + roles (stopped at step 800 to free the CPU) | 0.943 at 800 | | 2.098 | 0.993 |
  | 1. vocabulary 8,192 (stopped at step 500) | **0.000** at 500 | | 2.192 | 0.583 |
  | 2. 1–5-piece names (+ roles) | 0.221 | 0.548 / 0.396 / 0.843 | 2.142 | 0.513 / 0.885 |
  | 6. gap 800 (+ roles, documents of 1,024) | 0.941 | – / – / 1.000 | 2.167 | 0.993 / 0.999 |
  | the combination: vocabulary 8,192, 4 facts of 1–5 pieces, 1 question a document, L 8 read at 4; seeds 0 / 1 / 2 | **0.000** at every eval on all three | 0 / 0 / 1.000 | 2.226 / 2.222 / 2.228 | 0.857 / 0.995 |

  Not run, following Addendum 3's narrowed ladder and the CPU budget Nick set: vocabulary 32,768 and `--sparse` alone
  (stopped at steps 0 and 100), `--pretrain`, and `--layers` / `--at` alone.

  *B, on the combination arm, same budget:*

  | fix | seed | exact | first / later / end | language (no copy) | λ at asks |
  |---|---|---|---|---|---|
  | ring, Addendum 3's gate | 0 | 0.922 | 0.992 / 0.964 / **0.469** | 2.226 (2.226) | 0.78 |
  | **ring + read-aware gate** | 0 | **0.930** | 0.990 / 0.966 / 0.953 | 2.222 (2.226) | 0.76 |
  | | 1 | **0.908** | 0.996 / 0.952 / 0.961 | 2.218 (2.222) | 0.77 |
  | | 2 | **0.895** | 0.988 / 0.950 / 0.967 | 2.223 (2.228) | 0.78 |
  | vocabulary (`table`) + read-aware gate | 0 | 0.357 | 0.812 / 0.400 / 0.998 | 2.224 (2.226) | 0.80 |

  The order's gate reads 0.980 at step 0 (the cleanup alone), dips to 0.543 at step 300, then ends at 0.922. Its gate
  sees only h and cannot tell an empty address: the last piece asks for the next one, and the copy keeps offering a
  written token (end 0.469). The read-aware gate has no dip, and its λ is 0.92 at a real next piece against 0.22
  at an empty address. On the dense diagnostic (8 facts of 1–5 pieces, 300 steps, ceiling 0.879 per piece) both gates
  are capacity-bound (0.605 / 0.591); the read-aware one fixes the end (0.673 → 0.883).

  The other fixes:
  - **B2 (read at L − 2):** not run; the copy acts at the output.
  - **B3:** not run, since B1 passed.
  - **B4:** dropped (Addendum 3).
  - **B5:** not built. λ at the asks was 0.73–0.78 at half budget, against its < 0.05 condition.
  - **B7:** only if step 6 breaks (it does not: 0.941, with roles at 0.943 by step 800).
  - **B8:** offline only (the grid above); not built, since the ring passed.

  *Run 3 with the copy added, nothing trained* (`diag_copy_step0_run3_ring`, `…_ringb`), 24 held-out documents, far
  facts first / exact: as trained 0.250 / 0.000; ring with the order's gate 0.525 / 0.058; **ring with the read-aware
  gate 0.800 / 0.258**. The villages stay at 0.982 in all three.

  *Against each acceptance line:*
  - **A reproduces it: split.**
    - The starting configuration as written reads 0.877, short of 0.90 by 0.023. The same seed without the route
      loss read 0.934 (2026-10-20). With roles, which stop HERE writing at the address it asks, it read 0.943 by
      step 800.
    - At least one step drops to ≤ 0.20: **yes**. The vocabulary step reads 0.000, and the combination reads 0.000
      on 3 seeds. `--pieces` alone reads 0.221.
  - **B fixes it: PASS** for the ring with the read-aware gate. Exact is ≥ 0.80 on 3 of 3 seeds (0.930 / 0.908 /
    0.895), and the language is within 0.02 on each (−0.004 / −0.004 / −0.005). The ring with the order's gate
    passes on its one seed (0.922). The vocabulary form, at 0.357, is below the 0.50 kill line.
  - **Tests:** as above.
  - **C:** `RUN = 4` is written. It continues `spatial_link_3_final.pt` with `CB_SPATIAL_COPY=ring`,
    `CB_SPATIAL_COPY_TAU=3.04`, `CB_SPATIAL_COPY_BETA=0.115`, and `CB_PROBE_SHARDS` at every eval. It is the link
    arm only (run 3 is the control). The readout compares it with run 3: facts ≥ 0.50, villages kept ≥ 0.90, text mix
    within 0.02. **`HOURS_PER_ARM = 0.5` is a placeholder for Cowork to set.** For that, the toy's read-aware arms
    sit at 0.85–0.91 from step 100 to 400 and end at 0.90–0.93.
- **Not verified here:** anything on a GPU.
  - The copy under `torch.compile`: it gathers the asking positions with `nonzero`, a shape that changes every
    batch, so graph breaks or recompiles are possible. `CB_COMPILE=0` is the fallback, noted in the notebook.
  - bf16 autocast through the mixture (CPU float32 only).
  - The eval probe's cost at 64 documents.
  - Run 4 itself.

  The 450M numbers here are CPU forwards of run 3's checkpoint on 24 documents.
- **Found on the way:**
  1. **The villages never tested copying.** A stored `pond` is answered ` pond` (a different token), and 76% of
     village questions answer "nothing". Their 0.953 is empty-read detection plus a closed-set map, so "villages
     work" said nothing about open-vocabulary recall.
  2. **The specified gate σ(g(h)) cannot see an empty address.** I added the read-aware gate as a flagged variant
     (`copy_beta`; absent = the order's gate). It goes beyond the order as written, and `RUN = 4` uses it.
     Formula: `λ = σ(g·rms(h) + b + exp(τ)·(m − β))`, with m the best candidate's cosine.
  3. **An import side effect in `exp_hypertoken_recall.py`.** Importing it sets `CB_NO_FLEX=1` in `os.environ`, and
     the backbone reads that on every forward. Any in-process caller during training would silently leave flex
     attention. `fact_probe` undoes it; the fix belongs in that file (move the default into `main()`), as its own
     order.
  4. **I deviated from B1's first sketch.** It was additive with `σ(g)·exp(τ)`, which cannot be exact at step 0, and
     its scalar τ would move about 1 over the 450M's whole run at lr 3e-4. Addendum 3's mixture superseded it
     before any run, and the mixture is what is built.
  5. **B8 (a partitioned scene) is a strong fixed-state alternative.** On run 3, K = 4 makes even the 128k-way decode
     0.996 whole answers, offline. Not built.
  6. **Two files over 300 lines grew.** `train_base.py` (1,050 → ~1,115) and `_build_spatial.py` (331 → ~370); the
     reasons are in the commits. `assembly.py` stays under 300 (296).
- **Questions:**
  1. Run 4 is configured with the read-aware gate. Keep it, or run Addendum 3's gate as written (drop
     `CB_SPATIAL_COPY_BETA`)? The toy says keep it: end 0.95–0.97 against 0.47, and no dip.
  2. Run 4's budget: Cowork's to set from the curves above.
  3. Should the `exp_hypertoken_recall` import side effect (found 3) become its own order?
