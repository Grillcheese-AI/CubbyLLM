# CoT pipeline — cubbyllm → cubelang → bridge → output, with verified answers

**Date:** 2026-08-07 · **Status:** design, approved to spec · **Hypotheses:** H-F2 (the bridge carries reasoning, not just routing) + H-B3 lineage (real unbind in the loop) · **Predecessor:** `2026-08-05-reasoning-bridge-slice-design.md` (proved the roundtrip; this proves *the right answer*)

## 1. Why this exists

The reasoning-bridge slice proved one Python→cubelang→Python roundtrip with real
unbind over a symbolic boundary. Nothing yet *uses* it to answer a question. This
cycle builds the chain-of-thought pipeline: a question comes in, facts are
retrieved by the fast table, a CubeLang program holds the reasoning chain in VSA
superposition and reads the answer back out with per-step confidence, and the
output is either a **verified answer with an auditable trace** or an **honest
"cannot verify" with the partial trace**. The trace — fact used, symbol
recovered, cosine similarity, per hop — *is* the chain of thought: grounded,
auditable, and incapable of silently hallucinating a hop, which is the property
free-form token CoT cannot offer.

## 2. Settled decisions (user-approved 2026-08-07)

1. **Program author: harness planner now, trunk later.** A deterministic planner
   composes each question's CubeLang program from retrieval structure. The
   trunk-generates-CubeLang path (the BBPE-128k tokenizer's atomic opcodes exist
   for exactly this) is a later SFT milestone, explicitly deferred (§8).
2. **Failure semantics: repair loop, then honest fail.** A weak hop (low recover
   similarity, or no matching fact) triggers re-retrieval with the validated
   query-expansion repair; after N=3 attempts the pipeline outputs
   `verified=false` with the partial trace. An answer is *claimed* only when
   every hop cleared its thresholds. No LM free-form fallback.
3. **M1 eval: multi-hop .pq exact-answer.** `E:\valid_scaling_law_with_facts.pq`
   (10k questions, exact answer strings, known chains), scored by hop count
   against retrieval-only and chase-only baselines (§7).
4. **Fact representation: triple frames (approach C).** Facts parse to
   (subject, relation, object); the program binds `relation → object` into a
   frame and each hop is `recover(frame, REL_i)` with cosine confidence. General
   non-templated text falls back to fact-ID pointer binding (approach B) in a
   later cycle — out of M1 scope.
5. **Inherited invariants (from the reasoning-bridge slice, unchanged):** the
   boundary is symbolic — roles/fillers/programs cross, raw hypervectors never
   do; cubelang is the canonical reasoning VM (`use vsa; recover(reg, role)`);
   transport is the existing verify-before-execute protobuf-over-stdio bridge;
   grilly stays the trunk substrate.

## 3. Architecture and data flow

```
question ──► Planner ──────► (relation chain, hop count)
                │
                ▼
        Retriever (v4 table) ──► top-k facts per hop ──► triple parse
                │  ▲                                        │
                │  └── repair: query expansion ◄── weak hop ┤
                ▼                                           ▼
        Program builder ──► CubeLang source ──► bridge (strict compile,
                                                protobuf, subprocess)
                                                    │
                                                    ▼
        cubelang VM: bind REL_i→obj_i into frame; recover(frame, REL_i)
        per hop + absent-role control; returns (symbol, similarity) list
                                                    │
                                                    ▼
        Verifier ──► all hops ≥ τ_vm AND symbols match walk?
                       yes ──► {answer, trace, verified: true}
                       no  ──► repair (≤3) ──► {partial trace, verified: false}
```

Division of labor, stated plainly: **retrieval walks the chain; the VM holds it,
verifies it, and reads the answer out.** The Python side picks candidate facts
hop by hop (that is what the fast table is for); the VM's frame keeps every hop
in one superposition, `recover` proves each hop survives the algebra with its
confidence attached, the final `recover(frame, REL_n)` is the answer readout —
from the VM, not from Python bookkeeping — and an absent-role control proves the
frame is not a yes-machine.

## 4. Components

### 4.1 Planner — `cubbyllm/reasoning/planner.py` (new; `__wiring__ = WIRED` into pipeline)

Pure Python + regex; no model, no numpy.

- `parse_question(q: str) -> QuestionPlan | None`. The .pq grammar is
  `What is the R_n of the R_{n-1} of ... of <tail>?`. Split the relation chain
  on `" of the "`; hop count = segments; the tail segment (which mixes the last
  relation and the seed entity — relations themselves contain "of", e.g.
  "country of citizenship") is resolved *by the hop-1 fact parse*, not guessed.
  Returns the ordered relation segments. Unparseable questions return `None`
  and the eval reports the excluded count — parse failures are data, not
  crashes.
- `parse_fact(f: str) -> Triple | None`. Facts are templated
  `<object> is the <relation> of <subject>`; regex
  `^(?P<obj>.+?) is the (?P<rel>.+?) of (?P<subj>.+)$` (first `is the` wins;
  greedy subject). Returns None on mismatch.
- `relation_matches(expected_segment, parsed_rel) -> bool`: word-overlap match
  (the question says "country of citizenship", the fact says "country of
  citizenship"; minor surface drift tolerated by ≥0.6 word Jaccard).

### 4.2 Retriever — injected callable; M1 implementation lives in the eval script

The package must not import mowm (port-don't-link rule). `pipeline.py` takes
`retrieve: Callable[[str, int], list[tuple[int, float, str]]]` — query text in,
`(fact_row, score, fact_text)` top-k out. The M1 implementation (in
`validation/exp_m3_cot_pipeline.py`) is the v4 FastWordEncoder over the .pq
fact store, dense cosine — the same machinery as exp_m3_injection. Hop queries:
hop 1 = the question text; hop i>1 = `<current entity> <expected relation
segment>`; repair re-queries with the retrieved-so-far texts appended (the
validated chase expansion).

### 4.3 Program builder — `cubbyllm/reasoning/programs.py` (new; WIRED)

`build_chain_program(triples: list[Triple], control_role: str) -> str` emits:

```
use vsa;
bind frame, REL_1, "<obj_1>";
bind frame, REL_2, "<obj_2>";
...
r1 = recover(frame, REL_1);
...
rn = recover(frame, REL_n);
ctrl = recover(frame, <control_role>);
return [r1, ..., rn, ctrl];
```

Roles are sanitized uppercase symbols derived from the relation segments
(deterministic, collision-checked within a program); fillers are the object
strings as symbols. The exact return-shape syntax follows what the existing
`programs/reasoning_bridge.cube` + bridge client already support — if the VM's
current surface returns one value per program, the builder emits one program
per recover and the client batches them (decided at plan time from the shipped
client's capability, not re-invented here). The control role is a fixed symbol
(`ABSENT_CTRL`) never bound; its recover must come back *below* τ_vm.

### 4.4 Pipeline — `cubbyllm/reasoning/pipeline.py` (new; WIRED)

`answer(question, retrieve, vm_client, taus, max_repairs=3) -> CoTResult`:

1. Plan the question; no plan → `verified=false, reason="unparseable"`.
2. Walk hops: retrieve top-k (k=3); take the best candidate whose triple parses
   AND whose relation matches the expected segment AND whose subject matches
   the current entity (normalized); found → advance entity to its object.
   No candidate → repair (expanded query, next candidates); exhausted →
   honest fail with partial trace.
3. Build the program from the walked triples; run through the bridge
   (verify-before-execute; a compile failure is a bug, not a data condition —
   it raises).
4. Verify: every `recover` similarity ≥ τ_vm and symbol == the walked object;
   control role similarity < τ_vm. Any violation → one full-walk repair pass
   (fresh candidates for the weakest hop), then honest fail.
5. Output `CoTResult`: `answer` (final object), `verified: bool`, `trace`
   (per hop: query, fact text, triple, retrieval score, recover similarity),
   `repairs_used`, `reason` on failure.

`CoTResult` is a plain dataclass in `cubbyllm/reasoning/` — it is the output
contract the user asked for: answer + auditable chain, or honest refusal.

### 4.5 Thresholds

Two taus, both **deployment-calibrated in the eval, never transferred** (house
rule): `tau_vm` (recover similarity floor) calibrated on a 200-question
calibration slice disjoint from the eval questions (Youden on
correct-vs-incorrect hop recoveries); `tau_ret` (retrieval score floor for
"found a candidate") from the same slice. The eval reports both and the
sensitivity of the headline number to ±20% tau perturbation (the τ-fragility
check).

## 5. Error handling

- Unparseable question / fact: counted, reported, `verified=false` — never a
  crash, never silently skipped.
- VM subprocess failure or compile rejection: raised — infrastructure, not data.
- Repair budget: 3 per question total (walk-level and verify-level combined);
  the trace records each repair.
- The claimed-answer invariant is enforced in code: `verified=true` is
  impossible with any hop below τ_vm (asserted in the pipeline, tested).

## 6. Testing (package)

Unit tests with a **fake VM client and fake retriever** (the `fake_world_model`
pattern): parser goldens for both grammars incl. multi-word relations and
unparseable rows; program-builder golden strings incl. role sanitization and
the control role; pipeline paths — happy 3-hop walk, weak-hop repair success,
repair exhaustion → honest fail, control-role violation → fail, claimed-answer
invariant. Live-VM smoke test (marked, skipped when the cubelang binary is
absent) running one real 2-hop program end-to-end. All modules declare
`__wiring__`; guard tests unchanged.

## 7. Eval and kill criterion — `validation/exp_m3_cot_pipeline.py`

Same sampling as exp_m3_injection (seed 0, 800 questions, hop split
385/256/159; store = the sample's unique facts; a second arm adds 100k DBpedia
distractors). Arms:

- **retrieval-only**: top-1 fact for the raw question; answer = its parsed
  object.
- **chase-only**: the validated n_hop-step chase; answer = final retrieved
  fact's object.
- **CoT pipeline**: the full walk + VM verify + readout.

Metric: normalized exact match (lowercase, collapse whitespace, strip
punctuation and leading articles) against the dataset answer, by hop count.
Also reported: claimed-answer precision (accuracy among `verified=true`),
coverage (fraction claimed), repair usage, per-question latency, and the
control-role pass rate.

**Kill criterion (M1 passes iff all four):**
1. CoT exact-match beats retrieval-only on the 2-hop AND 3-hop subsets.
2. Claimed-answer precision ≥ 0.90 (the trace means what it says).
3. Zero `verified=true` results with any hop below τ_vm (asserted).
4. Absent-role control below τ_vm in ≥ 95% of executed programs.

Falsification is informative: if chase-only matches the CoT pipeline on exact
match, the VM adds verification but not accuracy — that result is recorded,
and the pipeline's value claim narrows to auditability (which baseline
retrieval cannot provide either way).

## 8. Deferred (explicitly out of M1)

- **Trunk-SFT CubeLang generation** — the planner's replacement; needs the
  Qwen trajectory decode→re-encode + a training cycle of its own.
- **General-text triples** (approach B fact-ID fallback) — M1's parser covers
  the templated corpus only, and says so.
- **mowm worlds as the fact store** (retrieval via routed worlds +
  inter-world delegation) — the retriever callable is the seam; swapping it in
  is a later cycle.
- **Bidirectional context/parameter handles on the bridge** (the full H-F2
  shape) — unchanged from the slice's deferral.
- Latency engineering (subprocess pooling, batched programs) — measure first.

## 9. Risks

- **Grammar drift in the .pq**: some rows may deviate from the template; the
  parser's excluded-count is reported and the eval proceeds on the parseable
  set (row-0 spot checks suggest high regularity; the probe verified
  fact_to_inject ∈ facts for 10000/10000, answer ∈ last fact 10000/10000).
- **Frame capacity**: ≤3 bound pairs per frame is far inside the measured
  bundle-capacity envelope (H-B4 datum); the control role guards the readout
  regardless.
- **Two-algebra confusion**: impossible by construction — symbols only across
  the boundary (inherited invariant).
