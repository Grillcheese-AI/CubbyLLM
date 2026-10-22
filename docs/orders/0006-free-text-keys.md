# 0006 — Free-text questions: addresses keyed by the words, asked through a pointer (toy)
status: open
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

## Out of scope
- The 450M (shard marks for multi-token subjects, a run): the next order, if P passes. Free-text *writes*:
  the host writes facts here. Relation paraphrase beyond the toy's templates.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
