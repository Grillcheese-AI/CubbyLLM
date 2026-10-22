# 0007 — Don't know → look it up once from a verified source → know it next time, through the loci path
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("if it doesn't know it needs to find the fact from a
verified source online with the one-shot learning; next time it will know") · hypothesis: H-K10 (+ the ask loop)

## Why
The host already learns on a miss. `standin/ask.py`'s `AskLoop` hands a stalled (entity, relation) to
`cubbyllm.reasoning.learn.learn_and_answer`, which asks Wikidata, then the web fallback (`standin/web_source.py`:
two independent sites before a fact leaves the latent tier). The memory gate admits the fact with provenance
into the persistent world store, and next time the store has it (`standin/README.md`, "I don't know yet — but I
can look it up for you"). The loci path now recalls a fact written once, at 0.844 exact on the 450M with the
copy (`diag_swap_copy_run4_2026-10-21.json`). Its read-aware gate also tells an empty address from a held one,
and that is the "don't know" signal. Nothing connects the two yet.

- **What it brings:** a question Cubby can't answer triggers one verified lookup. From then on, the trunk
  itself answers it from the store through the loci path, with no retraining and the source attached.
- **Advantage over conventional models:** they either hallucinate past their cutoff or need retraining or
  fine-tuning to learn a fact; here one admitted fact is known the next time it's asked.
- **In plain words:** when it doesn't know, it checks a trusted source once, and from then on it knows, and
  can say where it learned it.

## Build
- A loci reader in the stand-in (`standin/loci_reader.py`, new, < 300 lines; `AskLoop(loci=...)` /
  `serve_api --loci <ckpt>`; absent = today's loop, unchanged). Given the parsed (entity, relation) and the
  world store's facts about that entity:
  1. Write those facts into a fresh scene in the layout the 450M was trained on (`make_fact_shard --lead`:
     the facts with the object pieces placed at their addresses, then the question, with the answer's pieces
     asking).
  2. Run the trunk with the copy, teacher-free (greedy).
  3. Return the answer, or **miss** when the read-aware gate stays shut (λ below its β point) or the answer is
     NONE / EOS at the first piece.
- In `AskLoop.ask`, with the reader on:
  - Try the loci reader first. A hit is spoken only if the host verifies it equals the store's value
    (`value_check`), with the store's provenance. A mismatch is a refusal, never spoken.
  - A miss goes to `learn_and_answer` as today. Once the gate admits the fact, ask the loci reader again and
    speak what it returns, under the same check.
  - Record which path answered, and whether a source was called, in the loop's record.
- Addresses: the host's parse gives (entity, relation), and the address is `make_fact_shard.address` with the
  shard's relation index (a Wikidata property → index map from the wiki world). Free text arrives with order
  0006's pointer; until then, the loop's existing question parser stands in.
- The checkpoint: run 5's final (order 0005) once it exists. Before that, run 3's trunk with run 4's copy, as
  `validation/diag_swap_copy.py` assembles it. Paths come from the environment (`CLAUDE.local.md`), never in a
  committed file.

- **Creativity stays out of this path (Nick, 2026-10-22).** The reader and the lookup run only for turns that
  `serve.needs_facts` routes to facts. A creative ask (a poem, a story, "imagine…"), an opinion, or a guess
  framed as one never triggers a miss, a lookup or a refusal. The copy already leaves every non-asking position
  bit-identical; this keeps the host's side the same.

## Acceptance (pre-registered)
- **The battery** (`validation/bench_loci_learn.py`): 100 single-hop questions.
  - Each (entity, relation) is stated on Wikidata, is absent from the world store, and has an entity outside
    the wiki world. Relations come from those the 450M's shard covers.
  - Each is asked twice: once, then again in a **new process** with the persisted store.
- **Ask 1:** the loci reader reports a miss on ≥ **95%**. The loop learns from Wikidata only (no web fallback,
  no prose reader, cached). **0 wrong answers spoken**.
- **Ask 2:** **0 source calls** for the facts learned in ask 1. The loci reader's answer equals the stored value
  on ≥ **0.80**. Every mismatch is refused: **0 wrong spoken**.
- **Off = unchanged:** without the reader, the loop's existing tests pass as they do today. Full `pytest tests`.
- **Creative regression:** 30 creative asks (EN/FR, from `serve`'s own test phrasings) route the same and
  produce the same reply with the reader on and off; none calls a source.
- Report the Wikidata calls made (cached; keep it under ~300) and the loop's wall time per question on the CPU.

## Constraints
- Rules in `CLAUDE.md` hold. No external LLM on the path: Wikidata only here. The web source and the prose
  reader stay off for this bench. CPU, Nick's machine. The network is Wikidata's public API, cached; no
  bulk downloads.

## Out of scope
- Free-text parsing by the trunk (0006). A persistent scene across sessions: capacity, and B8's split scene,
  come later. Consolidation into weights (the sleep cycle, H-A10).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
