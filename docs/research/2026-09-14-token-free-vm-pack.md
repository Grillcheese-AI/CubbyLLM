# Competition pack — the token-free way to generalize with a VM

GrillCheese Research Lab · 2026-09-14 · answer against the measured record below, not against
general practice. Every number here was measured; where something is unmeasured it says so.

## The system

A small local model emits **programs**, not prose. A Rust VM (CubeLang, Turing complete)
executes them. A host mediates: it fetches from sources, gates every fact, and decides what may
be spoken.

- **Emitter**: LFM2.5-2.6B, SFT'd, Q4_K_M, llama.cpp. It ONLY emits programs.
- **Talk**: a second LFM adapter. Separate because when one model did both, "the grammar and
  abilities were degrading each turn". Measured precedent: identity fell at v5, forge decision
  1.00 → 0.17 at v6, arithmetic 0.825 → 0.750 → 0.675 over three rounds as chat grew.
- **Kill line: 0 wrong answers spoken.** A refusal is a result. The model proposes, the host
  disposes, the VM decides.
- **No external LLM on the serving path, ever.** External models build datasets and run
  competitions like this one. They never serve.
- SFT budget is not a constraint (10+ rounds available). Interference is.

## What the emitter emits today (v13f, 75,534 rows)

| task | rows | interface | shape |
|---|--:|---|---|
| `plan` | 37,588 | ISolve | retrieval calldata |
| `chain` | 28,529 | ISolve | post-retrieval verification frame |
| `arithmetic` | 6,148 | ISolver | register compute |
| `kernel` | 1,769 | ISolver + storage | stateful decision |
| `role_binding` | 1,500 | ISolver | event binding |

`plan` — relation is a **string argument** in a fixed role:

```cubelang
bind frame, SEED, "iridomyrmex bigi";
bind frame, HOP1, "parent taxon";
bind frame, HOP2, "instance";
return recover(frame, SEED);
```

`chain` — relation is baked into the **identifier**:

```cubelang
bind frame, H1_PARENT_TAXON, "Iridomyrmex";
bind frame, H2_INSTANCE, "Taxxon";
```

**Measured consequence: 146 distinct roles in v12e → 407 in v13f, 402 of them per-relation.**
The identifier space grows monotonically every data round. A relation never seen in training
cannot be emitted at all. `chain` is also work the host already does deterministically
(`build_chain_program`), and in the training data the facts are handed to the model in the
prompt — it is transcribing, not discovering.

**A verification asymmetry that probably explains how this happened:** `chain` rows carry
`vm_ok: True` / `gold_match: True`. `plan` rows carry `vm_ok: None`. The wrong-shaped task was
the measurable one, so it is the one that got optimised.

## What the VM actually is

Turing complete, ~680 KB Rust. Four engine-provided interfaces — `ISolve`, `ISolver`,
`ISolverLearn`, `IAgent` — resolved registry-first and tamper-proof; a program `implements` one,
it cannot define them. `ASK`/suspend/resume exists: a program can suspend, receive an answer,
and resume. `run-proto` (the production transport) is unconditionally strict.

**Its parse surface vastly exceeds its execute surface** (audited 2026-09-14, verified by
execution):

- 24 extended opcodes (`infer`, `score`, `predict`, `debate`, `forge`, `analogy`, …) compile
  cleanly and emit only a trace line. No value-level effect.
- `match (x) { arms }` **executes every arm unconditionally**. It looks like branching and
  produces plausible wrong answers.
- `assign` to an index or map target (`arr[0] = x`) compiles to **zero bytecode**, silently.
- A non-trivial expression as a call argument (`f(n - 1)`) silently becomes a placeholder:
  returns 0 instead of 4. **`--strict` does not catch this one.**
- `finally` blocks are never compiled, in any mode.
- `container` / `world` / `robot` / `event` / `extend` / top-level `struct` / `enum` are **never
  visited by the compiler at all** — not flagged, not warned, in any mode.
- **There is no type-checking pass anywhere.** Declared types are decoration; `number` and
  `quantity` do not lex as type keywords at all.
- Every permission attribute (`@external`, `@system`, `@once`, `@ratelimit`, `@restricted`,
  `@cron`) parses and is never read again. No access control, no once-guard, no scheduler.
- `recover(frame, ROLE)` on a role never bound **in an already-bound frame** returns the
  globally-nearest symbol at ~0.03 similarity — **not Null**. Only a never-bound frame reliably
  returns Null. τ_vm thresholds are 1.0 / 0.47 / 0.22 by hop count.

## Measured costs

0.7 ms per VM verification · 1.69 MB per session · 5,211 programs/s from 8 threads · p99 2.77 ms
· 19 ms spawn · 592 sessions/GB. Wiki world: 552,297 facts, 0 of which carry a time qualifier
until a live fetch restates them.

## Unused lever

llama.cpp supports **GBNF grammar-constrained decoding**. It is not currently used. With a
grammar, the model could not emit a malformed program, an unknown role, or the wrong arity — not
"rarely does", *cannot*.

---

# The questions

Answer all five. Be concrete and falsifiable. Mark unmeasured assumptions `[ASSUM]` and derived
arithmetic `[derived]`. Where you propose a mechanism, give the kill criterion that would retire
it. Brevity over breadth — a precise answer to four questions beats a vague answer to five.

## Q1 — The unit of generalization

An emitted program must work for a relation, tool, or task shape the model **never saw in
training**. `plan` puts the relation in a string argument; `chain` puts it in the identifier and
cannot generalize by construction.

Is "fixed roles + opaque string arguments" the right generalization unit, or is there a better
one? What exactly must the model produce for an unseen relation, and what must the host supply?
Be specific about where the string is *evaluated* (a table lookup is fine) versus *parsed*
(grammar over English is what we are trying to eliminate). Give the test that distinguishes
genuine generalization from memorisation of a large identifier table.

## Q2 — What replaces the scratchpad

Chain-of-thought partly works because intermediate tokens are a scratchpad the forward pass
attends over. A one-shot program has no such loop; `ASK`/suspend/resume gives
emit → execute → observe → emit.

Is that loop sufficient to replace what CoT does, or is something lost? Name what is lost
concretely. If something is lost, what is the **minimal** addition that recovers it without
reintroducing free-form token reasoning? Consider explicitly: search and backtracking
("that's wrong, let me reconsider") is the bulk of real reasoning traces and is control flow
over a hypothesis space, not straight-line computation.

## Q3 — Where enforcement lives

Given no type checker, decorative attributes, and a parse surface far larger than the execute
surface: should a GBNF grammar be the enforcement layer, and if so what should it admit?

Argue the risk as well as the benefit — a grammar that is the *only* validator is a single point
of failure, and it encodes a snapshot of the language that the language can drift away from.
What keeps grammar and engine in sync, mechanically? And is there a better place for enforcement
than either the grammar or the (absent) type checker?

## Q4 — Measuring generalization without fooling yourself

The lab already fell into one trap: the verifiable task got optimised and the correct-shaped task
did not, because only one of them carried a `vm_ok` column.

Design the evaluation that would have caught that at the round it happened, and that would catch
the next version of it. State: what is held fixed, what varies, the primary metric, the
secondary metrics, and the **number at which the approach is abandoned**. Assume 10+ SFT rounds
are affordable and that per-round evaluation is cheap.

## Q5 — The failure mode we will not see coming

Name the way this architecture fails that its designers are structurally unable to notice, given
that the kill line (0 wrong answers spoken) can read green while it happens. Then give the
**cheapest instrument** that detects it early.

Previous rounds produced two such findings that were both correct and both unanticipated. Do not
repeat them: (a) picking among several values of one relation is a claim decision, not a referent
decision; (b) a prior that decides feeds itself through unchallenged acceptances until the
verifier becomes a recommender. Find a third.

---

## Out of scope — do not propose

- Any external LLM on the serving path, including as a verifier, reranker or judge.
- More host-side string tiers, grammar over English, or anything requiring the VM to parse
  natural language. A string is *evaluated* when bound to a variable; it is never parsed.
- Popularity signals of any kind (sitelinks, pageviews, fact counts, corpus frequency) as a
  resolution prior — already unanimously rejected in a prior competition.
- Letting the emitter's own preference decide referents. It proposed; that was its say.
