# 0007 — Don't know → look it up once from a verified source → know it next time, through the loci path
status: done
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
  itself answers it from the store through the loci path, with no retraining. The source is kept for audit and
  never shown to users (0016).
- **Advantage over conventional models:** they either hallucinate past their cutoff or need retraining or
  fine-tuning to learn a fact; here one admitted fact is known the next time it's asked.
- **In plain words:** when it doesn't know, it checks a trusted source once, and from then on it knows.

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
    (`value_check`). The provenance goes into the record, for audit, not into the reply. A mismatch is a refusal,
    never spoken.
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
- **Commits:** `039278d` (the reader in the loop, the battery), and this report's commit (five fixes the battery
  found, below).
- **Files:**
  - `standin/loci_reader.py`: `LociReader` (run 5's 450M on CPU, the `make_fact_shard --lead` scene, greedy pieces
    under the copy's gate), `relation_of`, `through_loci`, `NoSource`.
  - `standin/ask.py` (`loci=`; `learn(src)`).
  - `standin/serve_api.py` (`--loci`).
  - `standin/question_forms.py` (the emitter binds a label it cannot name back).
  - `validation/bench_loci_learn.py`.
  - Tests: `standin/tests/test_loci_reader.py` (9).
- **Tests:**
  - A reading equal to the store's value is spoken with no source called; one that differs is refused.
  - A miss is looked up once and read again under the same check; the next time, the trunk answers with no lookup.
  - Without the reader the loop is what it was. The scene is the shard's layout, token for token and role for role.
  - The question's own word reaches the reader as its label.
  - Several values at one address are a miss, not a blend, and they go to the walk without a fetch.
  - The question asks at the address the facts were written at.
- **Measured** ([log](../../validation/logs/bench_loci_learn_2026-10-22.json); run 5's final checkpoint on CPU,
  2 threads; 100 questions drawn by seed from the hdc world's Wikidata5M facts whose entity is outside the wiki
  world, on relations the shard addressed, asked with the set's own templates):
  - **Ask 1:**
    - The reader reports a miss on **95/100** (line: ≥ 95%; the other 5 had their entity unresolved before a read).
    - **34 spoken, 0 wrong**: the reader, after the lookup, read the stored value.
    - Refused: 45 retrieval exhausted (Wikidata gave nothing for the relation), 8 ambiguous, 6 not covered by the
      plan, 5 entity unresolved, 2 reader misses.
    - The loop learned 766 facts for 47 entities.
    - **Wikidata HTTP calls: 203** on the first run, 0 since (cached; line ~300). Wall time 0.7 s a question.
  - **Ask 2, a new process with the persisted store:**
    - 47 questions learned in ask 1. The reader answered 34 and **equals the stored value on 34/47 = 0.723
      (line 0.80: fails)**. On the 38 whose relation holds one value, 34/38 = 0.89.
    - 9 hold several values (two occupations, two districts): the reader defers to the walk, which refuses them as
      ambiguous with no fetch. 3 never learned the asked relation in ask 1 (Wikidata gave others), and 1 reader miss.
    - **0 mismatches, 0 wrong spoken.**
    - **Source calls on the learned facts: 3** (line 0: fails narrowly). They are the 3 whose relation was not
      learned, so a lookup is what they need; none is for a fact ask 1 stored.
  - **Creative asks:** 30 (EN/FR); the reader read 0 times; routes talk 20 / reasoning 10, and 3 needed facts, with
    the reader on as off (serve's routing never calls the reader).
- **Not verified here:** the reader on GPU and on the 450M's training relations with dates (the hdc world has
  none, so the battery's relations are country of citizenship, located-in, occupation, educated at, and a few
  more). The serve stack with `--loci` live.
- **Found on the way** (each fixed, each with a test). The first full battery spoke 0 and matched 0 of 49:
  1. **The address hashes the name as written.** The loop's seed arrives normalized ("harpdog brown"), the facts
     keep their spelling, so the question read another address and copied stray pieces ("Mut", "2022",
     "enos" for Buenos Aires). The reader now asks with the store's own spelling.
  2. **One address holds one value.** Several values written at the same address superimpose ("lawyer" +
     "politician" read "lawyerician"). Several values are now a miss, and the walk chooses or refuses.
  3. **The gate shut, the trunk kept writing.** "Spain American Footballer", "Cubaano": pieces are taken only
     while the gate stays open.
  4. **The question's own word.** "Where was the administrative territorial entity of X?" handed the reader a
     wording no shard relation is named by. The emitter now binds the label whenever the forms cannot name the
     wording back, and `through_loci` resolves a wording through the loop's resolvers.
  5. **The battery drew its questions with SQL's unseeded RANDOM()**, so ask 2 asked other questions than ask 1.
     It is now seeded.
- **Questions:**
  1. The ≥ 0.80 line counts multi-valued facts the reader is built to defer. Count it on single-valued facts
     (0.89 here), or should the reader learn a convention for several values (write one, list the rest)?
  2. The 3 lookups in ask 2 are for relations ask 1 never got. Is a second lookup on a relation never learned
     within the "0 source calls" line's intent?

## Review (Cowork, 2026-10-22): done
Checked against `bench_loci_learn_2026-10-22.json`. **Done; one line fails.**
- Ask 1 passes: a miss on 95/100, 34 spoken, 0 wrong, 766 facts learned for 47 entities. The 203 Wikidata calls of
  the first run are in the report, not in this log (it is the cached rerun, 0 calls): keep the first run's count
  in a log of its own next time.
- Ask 2: **0 wrong spoken**, 0 mismatches, 0 source calls on a fact ask 1 stored. The reader equals the stored value
  on 34/47 = **0.723: the 0.80 line fails** (0.89 on the 38 single-valued facts).
- Question 1: the line stays failed as written. Several values at one address superimpose, so they need addresses
  of their own: each value at its own sub-address, read in order until a read comes back empty. Routed to order
  0009 (capacity), which owns how facts are written.
- Question 2: yes, a lookup for a relation ask 1 never got is within the line's intent: the line is about facts
  stored, and a relation never learned is still a miss.
- The five fixes the battery found are the real yield, the first above all: the address hashes the name as
  written, so the question must use the store's own spelling. Text keys (0006, 0022) must keep that rule.
