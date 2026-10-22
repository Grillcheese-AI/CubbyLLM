# 0006 — Free-text questions: addresses keyed by the words, asked through a pointer (toy)
status: review
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("we need to start working on the free text too";
route chosen: text-keyed addresses) · hypothesis: H-K10

## Why
A loci token is answered when its question carries the address (run 4: 0.761 → ~0.84 with the copy alone).
A free-text question carries no address. Today a fact's address is a **hash** of the subject's name
(`make_fact_shard.cell`), and a hash cannot be computed from the question's words: the learned route can
only memorize it for a closed set. On run 3 that worked for the villages, where the route alone read 0.748, and
not for the far facts, which read 0.014 over 160k subjects (`hypertoken_recall_spatial_run3_link_route_2026-10-21.json`).

**The argument:** make the address a function of the words, and both sides can compute it. A fact is written at
`key(s, r) = u(e(s)) ⊛ u(e(r))`, where `u(x) = unitary(W x)`, a fixed random projection to phases, and `e` is the
token embedding. Piece i of the object goes at `key ⊛ P_i`. A question contains the same subject and relation
words, so the trunk does not have to regenerate a key. It only has to **point** at its subject and relation
tokens, a few positions back, and the key is computed exactly from what it points at. That holds for subjects
it never saw in training: their keys come from their embeddings, not from memory. Fast-weight memories build
keys from (subject, relation) the same way (Schlag et al. 2021, arXiv 2011.07831); the pointer is a pointer
network's (Vinyals et al. 2015, arXiv 1506.03134).

- **What it brings:** facts the host wrote once, asked in free text, about entities the model has never
  seen, with a fixed-size decode state. The pointer reaches back `w` tokens, a ring of ids.
- **Advantage over conventional models:** KBLaM and RAG match free-text questions by learned similarity over
  every stored fact, at a cost that grows with the store. Here a question makes one exact key and reads it.
- **In plain words:** you ask in your own words, even about names the model has never met, and it finds the
  fact it was given, or says it doesn't know.

## Build
- `cubbyllm/model/spatial_key.py` (new, < 300 lines), flag-gated in `SpatialLink` (`key="text"`,
  `query="pointer"`, meta keys absent by default = today's behaviour, bit-identical):
  - **Text keys.** `u(x) = unitary(W x)`, with `W` fixed random at a scale chosen so different tokens' keys
    have |cos| < 0.1. Measure and report the distribution on the toy's table. Keys are built from embeddings
    with no gradient. Writes take their key's token ids (subject, relation) as side data: the host writes the
    facts, so it knows them. Pieces are offset by fixed random unitaries `P_i`.
  - **Pointer query.** At an asking token, two soft pointers (subject, relation) over the previous `w = 16`
    tokens are computed from `h` (a small attention). The key is `u(Σ a e) ⊛ u(Σ b e) ⊛ P_j`. A pointer loss
    (CE on the gold positions, which the builder knows, like the route loss) trains them. Decode keeps a ring
    of the last `w` ids, and decode = forward.
- The toy: `validation/exp_k10_free_text.py` (new; reuse `k10_far_docs` and `exp_k10_far_facts` by import).
  Use the passing configuration as the base: the combination arm with the copy over the ring and the
  read-aware gate. Questions come in **≥ 6 templates**: subject and relation in either order, with filler
  function tokens. **2 templates are held out** of training. **20% of subjects are never seen in training at
  all.** **20% of questions ask a fact the document does not hold**; their answer is a NONE token.
- Arms, at the same budget:
  - **U**, text keys with the address given (`query="placed"`): the keys work as addresses.
  - **H**, today's hashed addresses with the learned route and its route loss: the control.
  - **R**, text keys with the learned query `W rms(h)` and the route loss: can the trunk regenerate a key?
  - **P**, text keys with the pointer query.

  Report exact for each cell of {seen, unseen subjects} × {seen, held-out templates}. Also report how often an
  absent fact gets a non-NONE answer, the pointers' accuracy, and the filler language's loss.

## Acceptance (pre-registered)
- **U:** ≥ 0.90 exact (the base read 0.93 with hashed keys).
- **P:** ≥ **0.80** exact on unseen subjects × held-out templates, on 2 of 3 seeds. Absent facts answered
  with something other than NONE ≤ **5%**. Language within 0.02 of U.
- **H**, a prediction, not a pass line: ≤ 0.20 on unseen subjects (the hash wall).
- **Kill:** P below 0.50 on unseen subjects with seen templates. Then the pointer does not deliver the key,
  and the next route is a host-side parser.
- **Tests:** flags off = bit-identical forward and decode; text keys of the same ids are identical and of
  different ids near-orthogonal; a gold pointer gives the write's key exactly; decode = forward with the
  pointer; full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. CPU, the 2026-10-20 budget an arm; arms in parallel; same seeds across arms.
- Logs: `validation/logs/exp_k10_free_text_<arm>_<date>.{json,log}`.

- **If text keys of similar subjects collide** ("New York" / "New Jersey"): pattern separation before binding,
  as `cache.py`'s hippocampal cache does with grilly's DentateGyrus (Nick's `empirical_grilly_next` project (a checkout beside this one; path in `CLAUDE.local.md`)). It is a sparse expansion that pushes
  similar inputs apart. An idea to try only if the key-similarity report shows the problem.

## Out of scope
- The 450M (shard marks for multi-token subjects, a run): the next order, if P passes. What happens after NONE (look
  the fact up once, know it next time) is order 0007; here NONE only has to be right. Free-text *writes*:
  the host writes facts here. Relation paraphrase beyond the toy's templates.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:** `adf4dac` (text keys, the pointer, the toy), and the copy-alone check and arm A (this report's commit).
- **Files:** `cubbyllm/model/spatial_key.py` (new, 166 lines: `TextKeys`, `Pointer`), `SpatialLink` / `CubbyModel`
  plumbing (flagged: `meta["spatial"]["key"]="text"`, `query="pointer"`), `validation/k10_free_docs.py` (new),
  `validation/exp_k10_free_text.py` (new; arm A and the copy-alone, LM-alone and hard-switch metrics added after
  Cowork's question), `tests/model/test_spatial_key.py` (8 tests, 11 cases).
- **Tests:** flags off bit-identical, forward, decode and text, against HEAD's package with the same seeds
  ([log](../../validation/logs/spatial_key_off_identity_2026-10-21.log)). Keys of the same ids are identical,
  and keys of different ids sit at the random floor. A gold pointer gives the write's key exactly. Decode =
  forward with the pointer. `pytest tests/model/test_spatial_key.py`: 11 passed. `pytest tests`: 425 passed,
  5 skipped, 1 failed (the known wiring guard, order 0002).
- **Measured** (1,000 steps, d 256, 8 layers, held-out documents; logs `validation/logs/exp_k10_free_text_<arm>_s<seed>_2026-10-21.json`,
  the alone reruns `..._alone_2026-10-22.json`):

  | arm | seen/train | seen/held | unseen/train | **unseen/held** | exact | absent not NONE | language | pointer |
  |---|---|---|---|---|---|---|---|---|
  | U s0 | 0.868 | 0.869 | 0.844 | 0.805 | 0.848 | 0.063 | 2.261 | – |
  | A s0 (hashed, address given) | 0.899 | 0.864 | 0.839 | 0.816 | 0.857 | 0.068 | 2.271 | – |
  | H s0 | 0.000 | 0.005 | 0.000 | 0.000 | 0.001 | 0.170 | 2.270 | – |
  | R s0 | 0.013 | 0.000 | 0.010 | 0.000 | 0.006 | 0.107 | 2.266 | – |
  | P s0 | 0.872 | 0.869 | 0.854 | **0.816** | 0.855 | 0.087 | 2.264 | 0.986 |
  | P s1 | 0.879 | 0.660 | 0.899 | **0.694** | 0.783 | 0.099 | 2.260 | 0.895 |
  | P s2 | 0.885 | 0.755 | 0.870 | **0.785** | 0.826 | 0.053 | 2.267 | 0.945 |

  Keys' |cos| over the toy's table: mean 0.050, p90 0.102, p99 0.161, max 0.297, 89% of pairs below 0.1,
  unchanged by training. The mean is the random floor at d = 256 (√(2/πd) = 0.050). "|cos| < 0.1 for different
  tokens" cannot hold for every pair at this width. The p90 is at the line.

  **Cowork's check: the copy path alone on present facts** (its argmax over the answer's pieces, before the mix):

  | arm | mixed (exact) | copy alone | LM alone | hard switch (λ ≥ 0.5) |
  |---|---|---|---|---|
  | A s0 | 0.857 | **0.974** | 0.000 | – |
  | U s0 | 0.846 | **0.956** | 0.000 | – |
  | P s0 | 0.855 | **0.955** | 0.000 | 0.875 |

  The copy alone is near 0.96 in all three arms, so by Cowork's rule the trained language model is outvoting the
  copy, and the read is fine. The fix is the mixing switch, as in order 0005. A hard switch on today's gate
  recovers only 0.02 of the 0.10 gap (P: 0.875). λ on a present piece averages 0.90, but enough answer pieces fall
  under 0.5 that a whole answer is lost. The switch needs training of its own (a gate loss on present / absent,
  or 0005's copy-only training), not a threshold. (The switch metric came in with the P rerun; A and U ran
  before it.)

- **Acceptance:**
  - **U ≥ 0.90: fails** (0.848; rerun 0.846). A, today's hashed places with the address given, reads 0.857 on
    these documents. The 0.93 in the order was the base read on order 0001's documents, so U is at its ceiling
    here, and text keys cost nothing as addresses. The ceiling itself is the mix (copy alone 0.956–0.974).
  - **P ≥ 0.80 on unseen × held-out templates, 2 of 3 seeds: fails** (0.816 / 0.694 / 0.785: 1 of 3). Unseen
    subjects with seen templates read 0.854 / 0.899 / 0.870, so the subjects the model never saw are not the
    problem. Held-out templates are: the pointer's accuracy falls with them (0.986 / 0.895 / 0.945).
  - **Absent facts answered non-NONE ≤ 5%: fails** (0.087 / 0.099 / 0.053; U 0.063, A 0.068 too). That is the
    same mixing problem from the other side: λ on an empty piece averages 0.15–0.18, and the LM's guess gets through.
  - **Language within 0.02 of U: passes** (2.264 / 2.260 / 2.267 vs 2.261).
  - **H ≤ 0.20 on unseen subjects (a prediction): holds** (0.000; H is near 0 on seen subjects too. The learned
    route does not find hashed addresses in 1,000 steps on this toy, so the wall is not only the unseen one).
  - **Kill (P < 0.50, unseen subjects, seen templates): not killed** (0.854 / 0.899 / 0.870). The pointer
    delivers the key.
  - R reads 0.006: the trunk cannot regenerate a text key from the stream. The pointer is what makes text keys
    work.
- **Not verified here:** everything is the CPU toy (d 256, 8 layers, 1,000 steps, 2 threads; ~67 min an arm).
  Nothing ran on the 450M (out of scope). Three seeds for P, one for the other arms.
- **Found on the way:** a Gaussian `W` lifted the keys' crosstalk to mean |cos| 0.064 against the floor's 0.050.
  An orthogonal `W` sits at the floor. Binding commutes, so a key is symmetric in subject and relation (pinned in
  the tests). "New York / New Jersey" collisions did not show in the key report, so pattern separation was not
  tried. P's training curve dips at step 200 (pointer 0.681, unseen/held 0.184) before it recovers. The pointer
  loss is on then, and it goes to 0 once the pointers are right.
- **Questions:** (1) A next order for the switch: train the mix alone with the trunk frozen (0005's recipe) on
  this toy, and read U, P and absent again. If it lifts U to the copy's ~0.96 and absent under 5%, P's
  acceptance is a template-generalization question only. (2) Held-out templates: more template variety in
  training, or a pointer that sees relation words by identity (no positional bias)? Seed 1's 0.694 is the
  pointer's 0.895.
