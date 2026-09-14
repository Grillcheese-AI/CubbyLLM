# The token-free competition — how to generalize with a VM instead of tokens

GrillCheese Research Lab · 2026-09-14 · pack: `docs/research/2026-09-14-token-free-vm-pack.md`
· log: `validation/logs/exp_r21_competition_tokenfree.{json,md}`

## Contestants — 7 of 7 answered, $0.65, 823 s

| model | answer | completion | reasoning | cost | wall |
|---|--:|--:|--:|--:|--:|
| anthropic/claude-opus-5 | 14,950 ch | 9,351 | 4,054 | $0.2482 | 146 s |
| openai/gpt-5.6-terra-pro | 12,851 ch | 13,790 | 5,545 | $0.2031 | 101 s |
| z-ai/glm-5.3 | 11,837 ch | 4,424 | 2,226 | $0.0225 | 43 s |
| google/gemini-3.8-flash | 11,633 ch | 3,011 | 0 | $0.0129 | 27 s |
| moonshotai/kimi-k3 | 9,569 ch | 3,130 | 1,001 | $0.0533 | 103 s |
| qwen/qwen3.8-max-0902 | 9,498 ch | 16,000 | 13,831 | $0.0999 | 370 s |
| x-ai/grok-4.6 | 6,412 ch | 1,626 | 38 | $0.0145 | 34 s |

**First 7/7 in four rounds.** `--effort low` is now conclusively the fix, not the token cap:
rounds 1–2 were 4 of 12 for $1.19, this round is 7 of 7 for $0.65. deepseek-v4-pro was dropped
after failing at every setting tried.

---

## Q1 — The unit of generalization

**Unanimous, and it is the `plan` shape taken seriously.** A small closed set of host-defined
roles (`SEED`, `HOP1`, …) plus **opaque payloads the VM never parses**. The model emits *shape*,
never vocabulary. Nothing about a relation may appear in a program's structure.

Three things sharpen it beyond what we had:

**1. The free diagnostic.** Opus 5: *"The diagnostic is not accuracy, it is vocabulary growth: a
generalizing interface has a role/identifier count that is flat across data rounds. 146 → 407 is
the failure signature, and it is measurable every round for free."* One integer per build.

**2. The capability manifest.** The host supplies, per request, the admissible label strings plus
direction/arity/functional-or-multi-valued metadata — and that manifest is compiled into a
**per-request grammar**, so an unknown relation is not rare, it is *undecodable*. The identifier
space becomes bounded by the request instead of growing monotonically with the dataset. The
string is evaluated exactly once, as a key into the manifest index — a miss is a hard error,
never a nearest-neighbour fallback. That is "evaluated, not parsed" made mechanical.

**3. The decisive test — relabelling invariance.** Proposed independently by Opus 5 (its T2) and
gpt-5.6-terra-pro: rerun a relation-disjoint split with every relation label replaced by a fresh
opaque token (`r_88417`, `rel_2`) in both the manifest and the question, direction and arity
preserved. Genuine structural generalization is **invariant under bijective renaming**;
memorisation of a relation table collapses. A relation-disjoint split alone is necessary and not
sufficient — this is the one that separates the two. Kill criterion: relabelled accuracy below
80% of labelled accuracy after 3 targeted rounds means the string-argument form is not carrying
the generalization.

Opus 5 also adds a **Null control**: put a syntactically valid relation with zero facts in the
manifest; the required output is a program that yields Null and a refusal. *"Any spoken answer
here is a kill-line breach that current `recover` semantics would produce silently."*

---

## Q2 — What replaces the scratchpad, and why it argues against fixing `match`

`ASK`/suspend/resume recovers **observation-conditioned continuation** — the thing CoT gets from
seeing its own hop-1 result. Convergent across models, it does **not** recover:

- control flow over a hypothesis space (enumerate, test, refute, back up, try the sibling);
- **retraction** — CoT can say "that's wrong"; a `bind`ed frame has committed;
- re-representation (unit normalisation, "this is really a two-hop reversed").

The obvious fix is to put the search in the program. **Opus 5 argues that fix is unavailable in
this VM, and the argument is ours:** `match` executes every arm, indexed `assign` compiles to
zero bytecode, `f(n-1)` silently becomes a placeholder, `finally` never compiles. VM-level
branching is precisely the construct most likely to produce plausible wrong answers. Therefore:

> **All hypothesis-space control flow should be hoisted out of the program and into the host
> loop.** Programs stay branchless.

The minimal addition is three closed-arity, branchless opcodes over the existing ASK loop —
`propose(slot, [lit…])`, `refute(slot, cand, test_frame)`, `retract(frame, ROLE)` — with the
**host** owning the agenda, dedup, cycle detection, a node budget and termination. The model
contributes candidate generation and refutation tests; the host contributes scheduling and the
guarantee that exhaustion exits as a refusal rather than a guess. At 0.7 ms per verification, an
8-expansion budget is ~5.6 ms of VM time. gpt-5.6 reaches the same place from the other side: a
VM-owned bounded search primitive with explicit state, because otherwise the model must
reconstruct search state from transcript tokens, *"which reintroduces the same fragile
token-level control problem, merely across turns."*

**This cuts against the decision to implement real `match` arm selection.** If branching belongs
in the host agenda, `match` is investment in the wrong layer — and the honest alternative
(reject it at compile time) is both cheaper and more aligned. Flagged, not settled.

---

## Q3 — Where enforcement lives

**GBNF yes — as one of three layers, and the grammar must be _generated, never authored_.**

What it admits: the **execute surface only**. That excludes all 24 trace-only opcodes, `match`,
indexed assign, `finally`, every top-level form the compiler never visits, and every permission
attribute — Opus 5's phrasing is worth keeping: *"admitting them teaches the model to write
security theatre."* Arguments are restricted to **literals and bare variables**, which is the
only reliable defence against the `f(n-1)` placeholder that `--strict` misses. Relation and
entity literals come from the per-request manifest enumeration.

Note what that means: **the grammar fixes C11 on the emitter path even if the compiler is never
touched.** It cannot fix C11 for hand-written programs.

**The risk, named correctly: drift, not decoding.** A hand-written grammar is a snapshot of a
language that has already demonstrated it moves — three parse/execute divergences and zero type
checking. A grammar that is the only validator fails *silently and in the safe-looking
direction*: it keeps admitting a construct after the engine stops honouring it. Three mechanical
syncs, all CI:

1. **Generate the grammar from the engine's own opcode/interface registry** — the same
   registry-first structure that already makes `ISolve`/`ISolver` tamper-proof. A hand-edited
   grammar fails the build.
2. **A witness test per grammar token** — for every opcode admitted, a program whose *observable
   return value* changes when that opcode is removed. Any token with no witness is demoted
   automatically. **This is exactly the test whose absence let 24 trace-only opcodes into the
   language**, and it costs 0.7 ms each.
3. **Bidirectional differential fuzz** — sample from the grammar, it must compile and produce a
   non-empty effect under strict; mutate outside the grammar, it must be rejected.

**And the best single engineering idea in the competition: a bytecode verifier at the loader
boundary** (JVM-style), which Opus 5 and gpt-5.6 reach independently. It checks the compiled
artifact rather than the source, which is the only place the parse/execute gap is visible *as
data*:

> **Reject any program where a source statement maps to zero emitted instructions.**

That one check subsumes most of the drift audit's §C — indexed assign, `finally`, bare
expression statements, `throw`, trace-only opcodes — **including the ones nobody has found yet**,
and it keeps working as the language drifts. Add: reject placeholder instructions, reject
`recover` on a role with no dominating `bind`, require every frame reaching `return` to be fully
bound.

---

## Q5 — The third failure mode, and it is a cluster

Three models independently landed on the same one, from different directions.

> **The emitter becomes decorative.** The kill line reads green — every spoken answer is verified
> true against gated facts — while the emitted program contributes nothing to any answer. The
> host retrieves, the host builds the effective computation, and the VM executes a program whose
> output is ignored or coincides with what the host computed anyway. Every instrument measures
> **answer correctness**, and answer correctness is exactly what the host guarantees
> independently of the emitter. Ten more SFT rounds could improve emitter metrics while the
> emitter's causal contribution to spoken answers is zero. — glm-5.3

gpt-5.6 calls it causal irrelevance and names the concrete bypasses (host fallback, cached prior
retrieval, deterministic chain construction, an answer path reading request state rather than VM
output). The pack contains the seed and the models found it: `chain` is work the host already
does, and the training data hands the model the facts in the prompt.

**Three instruments, all cheap:**

- **Emitter ablation** — replace the emitter's output with a fixed trivial program on the same
  question set. If end-to-end accuracy is unchanged, *"every generalization claim about it is
  unfalsifiable theater."*
- **Fact-perturbation differential (liveness rate)** — execute each program twice, once against
  the fact base and once against a copy where the queried relation's value is swapped for
  another valid one. A live program's output must change. This also detects the compiler's
  silent-failure bugs in production rather than only in audit.
- **Shadow causal-ablation + provenance tainting** — replace the program result with a
  guaranteed-deny and require the shadow path to refuse; and make every spoken value carry the
  ID of the VM operation that produced it, so *a value with no authorized program-output lineage
  is unspeakable.*

Alarm: liveness below 95% `[ASSUM]`, or an ablation gap inside the eval set's noise floor for two
consecutive rounds.

### Three more, each tied to a measured fact in the pack

**The timeless snapshot (opus-5).** The gold labels, the verifier and the answer all descend from
the same stale snapshot — 552,297 facts, **0 with a time qualifier**. A fact true at snapshot
time and false today is `vm_ok: True`, `gold_match: True`, passes τ_vm, and is spoken. *"The kill
line reads green by construction — not because the system is careful, but because the only
oracle it has is the thing that is stale."* Every instrument we own is downstream of the
snapshot, so no internal measurement can surface it, and it worsens monotonically with snapshot
age while all dashboards stay flat. Instrument: sample 50 spoken facts/day, re-fetch live, and
keep a **per-predicate volatility table** — *that table is the missing time qualifier, learned
empirically for free*. High-volatility predicates become fetch-required; snapshot-only derivation
on one becomes a refusal.

**The similarity floor becomes the answer generator (kimi-k3).** `recover` never returns Null; it
returns the globally-nearest symbol at ~0.03. τ_vm at hop 3 is 0.22. As the store grows,
nearest-symbol density rises, so more live queries bind plausible-but-wrong symbols that clear
the threshold. The system doesn't start refusing (visible) — it starts *confabulating with
verified-looking bindings*. Instrument: log the similarity of every accepted bind (the field
already exists) and histogram weekly; alert on mass concentrated just above a τ boundary. Kill
criterion: *"a never-Null recover is incompatible with the kill line regardless of how well
everything else measures."*

**Taxonomic collapse (gemini-3.8-flash).** The emitter learns that generic high-degree relations
("language", "instance of") almost always resolve to something the host confirms — so instead of
refusing on an unseen relation it retreats up the ontology. *"What dialect does Clan X speak?"* →
`HOP1 "language"` → "Indo-European" → host verifies true → spoken. Kill line preserved, answer
true, completely non-responsive. Instrument: rolling median out-degree of resolved entities; trip
on a 3× spike between rounds.

**Coverage contraction (qwen).** The model internalises the host's refusal boundary from SFT and
the system's coverage silently contracts to the host's acceptance region.

---

## What to adopt

1. **Vocabulary growth as a per-round metric.** Flat identifier count or the interface isn't
   generalizing. Free.
2. **Relabelling invariance** as the generalization test; relation-disjoint split alone is not
   sufficient.
3. **The per-request capability manifest**, compiled into a per-request grammar.
4. **Generate the GBNF from the engine registry**; hand-edits fail the build.
5. **The witness test per grammar token** — the test whose absence created the 24 trace-only
   opcodes.
6. **A bytecode verifier at the loader**: reject any source statement that maps to zero emitted
   instructions. Subsumes most of §C, including what hasn't been found.
7. **Emitter ablation and liveness rate**, every round, before any claim about generalization.
8. **Per-predicate volatility probe** — 50 re-fetches/day rebuilds the missing time qualifier.
9. **Branchless programs; hypothesis search in the host agenda** (`propose`/`refute`/`retract`).

## What this changes about decisions already made

- **`match`**: the decision to implement real arm selection is now contested. If search belongs
  in the host agenda, VM-level branching is investment in the layer most likely to produce
  plausible wrong answers. Rejecting `match` at compile time is cheaper and better aligned.
- **`recover`**: restricting cleanup to the frame's own bound symbols is confirmed from two
  directions — Q1's Null control and Q5's confabulation mechanism. Two models call the current
  never-Null behaviour incompatible with the kill line outright.
- **C11**: a generated grammar that admits only literals and bare variables as arguments fixes it
  on the emitter path, which is the path that matters, without touching the compiler.

## The honest caveat, from the winner against itself

Opus 5 indicts its own Q1 answer: once the host supplies the manifest and the grammar enumerates
it, **the emitted program becomes near-determined by the manifest, and a pure copier passes the
unseen-relation test.** Correctness migrates into manifest construction while the relation-
disjoint split still reports generalization. Detector, cheap: inject *k* plausible decoy
relations into every manifest and compare selection accuracy against the 1/m chance baseline. If
accuracy ≈ 1/m under decoys, the model contributes nothing and the host is the system — *"which
is fine to know, and fatal to not know."*
