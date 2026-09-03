I reviewed the README. You already have most of the substrate that a useful GoT system needs: VM-certified programs, typed world facts, retrieval and walks, persistent program lineage, external execution, regression probes, and a host-side consistency gate. GoT should therefore be added as a **structured orchestration layer over your existing verified graph**, not as unconstrained “thought generation.” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

## What you already have

Your current system is closer to a **verified state-transition graph** than a conventional CoT agent:

- `FactStore` supplies graph nodes and retrieved edges.
- Chain walking supplies a constrained path through facts.
- `CubbyBrain.trace` records sense → route → walk → emit → VM → gate → learn.
- `ProgramLibrary` stores generated programs, parents, edits, children, uses, rewards, and retirement status.
- CubeLang plus the Rust VM provides executable verification.
- The consistency gate can veto an emitter answer when a verified walk disagrees.
- The Pac-Man environment supplies an embodied state graph with observations, actions, hazards, rewards, and persistent discoveries.
- The v4/v5 results already demonstrate the important loop: game experience generated verified programs, and the emitter learned those program patterns. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

That means the missing component is not “more thoughts.” It is a **graph controller** that decides which states, hypotheses, programs, and evidence should be expanded, merged, scored, or retired.

## How I would incorporate GoT

Use GoT at the orchestration level:

```text
Input / perception
        ↓
Graph state construction
        ↓
Candidate thought or program nodes
        ↓
Dependency and contradiction edges
        ↓
Parallel expansion
        ↓
VM / environment / retrieval verification
        ↓
Merge, prune, revise, or branch
        ↓
Answer, action, or SFT record
```

A useful node taxonomy for Cubby would be:

```text
Observation
Fact
Question
Hypothesis
Plan
ProgramCandidate
ExecutionResult
Verification
Failure
Reflection
Skill
TrainingExample
```

Edges should be typed rather than merely “related”:

```text
supports
contradicts
depends_on
derived_from
tests
refines
mutates
retrieves
causes
supersedes
uses_skill
```

For example:

```text
Observation:
    ghost at cell (4, 2, 1)

Fact:
    ghost-near(4, 2, 1)

Hypothesis:
    flee-left is safer than flee-forward

ProgramCandidate:
    FLEE_COMPARE(...)

ExecutionResult:
    left_distance = 5
    forward_distance = 2

Verification:
    result = true

Action:
    move_left
```

This is materially different from having the model write a paragraph of reasoning. Every important node either points to an observation, an executable artifact, or a verifier result.

## The critical design choice

Do not make the model emit the complete graph. Have the **host construct and validate the graph**, while the model proposes candidate nodes or transformations.

The model should be allowed to propose:

- A new hypothesis.
- A missing edge.
- A program mutation.
- A decomposition of a goal.
- A candidate explanation of failure.
- A new skill abstraction.
- A ranking of existing candidate actions.

The host should control:

- Node identity and canonicalization.
- Fact provenance.
- Program execution.
- VM certification.
- Reward assignment.
- Contradiction detection.
- Retirement and versioning.
- Which graph regions are exposed to the model.
- What enters the SFT corpus.

This follows the strongest principle already present in your codebase: **the trunk supplies syntax and selection; it does not blindly supply truth**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

## GoT over the current reasoning cortex

Your current reasoning path is approximately:

```text
retrieve → format Facts block → emit CotChain → VM → ground check → consistency gate
```

The GoT version could be:

```text
retrieve
  → construct candidate evidence subgraphs
  → rank subgraphs
  → ask emitter for a chain/program candidate
  → execute candidate
  → attach result and confidence
  → compare against alternative subgraphs
  → merge verified derivations
  → gate answer
```

Instead of sending one flat `Facts:` block, send a bounded graph neighborhood:

```text
Goal:
  Where is the target relative to the current cell?

Nodes:
  cell(2,1,0)
  neighbor(2,1,0, forward, 2,2,0)
  neighbor(2,2,0, left, 1,2,0)
  target(1,2,0)

Edges:
  current --forward--> intermediate
  intermediate --left--> target

Candidate operations:
  [walk, count, compare, recall]
```

Then require the emitter to return a **small typed operation**, not an unrestricted narrative:

```text
program Where implements ISolve {
    ...
}
```

The host can expand multiple candidate subgraphs in parallel, execute each, and retain only those that satisfy the target predicate.

## What should be merged

GoT is most useful when paths converge. Your implementation should explicitly merge:

- Duplicate facts with different provenance.
- Different derivations producing the same verified result.
- Program candidates with equivalent normalized CubeLang.
- Mutations sharing a parent and identical behavioral signatures.
- Multiple observations of the same world state.
- Alternative plans whose verified subplans are reusable.

For program deduplication, use at least three identities:

```text
source_hash
normalized_ast_hash
behavior_signature
```

Two programs with different source but the same normalized AST should normally share a lineage node. Two syntactically different programs with the same behavior on the evaluation suite should remain separate candidates until broader tests distinguish them.

## Scoring and pruning

A simple GoT priority score could be:

\[
P(n) =
w_g G(n)
+ w_v V(n)
+ w_r R(n)
+ w_u U(n)
- w_c C(n)
- w_k K(n)
- w_d D(n)
\]

Where:

- \(G(n)\): relevance to the current goal.
- \(V(n)\): verifier confidence.
- \(R(n)\): expected reward or task progress.
- \(U(n)\): novelty or information gain.
- \(C(n)\): execution cost.
- \(K(n)\): contradiction or risk penalty.
- \(D(n)\): redundancy penalty.

For self-improvement, add a held-out generalization term:

\[
P_{\text{update}} =
\Delta L_{\text{target}}
+ \lambda \Delta L_{\text{heldout}}
- \mu \text{regressions}
- \rho \text{complexity}
\]

A candidate should not be accepted merely because it solves the current situation. It should improve a held-out task family or produce a reusable skill.

Your existing retirement mechanism is exactly the right basis: retire low-value patterns, preserve their source and traces, and never delete them. GoT can provide the evidence for retirement instead of relying only on aggregate usage counts. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

## GoT and self-SFT

The graph should produce SFT examples only at **verified boundaries**:

```text
goal graph
→ selected evidence subgraph
→ proposed action/program
→ environment or VM result
→ accepted/rejected decision
→ generalized skill
```

Good SFT records include:

- The selected subgraph, not the entire memory.
- The winning candidate.
- Rejected candidates and the reason for rejection.
- The verifier result.
- The minimal transformation that fixed the failure.
- Whether the improvement survived a held-out test.

For example:

```json
{
  "goal": "choose a safe exit",
  "evidence_graph": "...",
  "candidates": [
    {"program": "compare_left_forward", "verified": true, "score": 5},
    {"program": "compare_right_forward", "verified": false, "error": "wrong branch"}
  ],
  "chosen": "compare_left_forward",
  "result": "left",
  "generalized_skill": "select_exit_by_distance",
  "held_out_pass": true
}
```

This would extend your existing v4 path naturally: the game generated verified records, the trunk learned the resulting program families, and GoT would add the explicit candidate-selection and evidence-aggregation structure. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

## Recommended implementation order

### 1. Add a typed graph store

Do not start with a general-purpose graph database. A versioned in-memory graph serialized to JSON or SQLite is enough.

Minimum fields:

```python
Node:
    id
    kind
    payload
    provenance
    parent_ids
    status
    created_at
    score
    version

Edge:
    source
    relation
    target
    confidence
    provenance
```

### 2. Wrap existing traces

Convert existing events into graph nodes:

```text
sense       → Observation
route       → Decision
walk        → Derivation
emit        → ProgramCandidate
vm          → ExecutionResult
gate        → Verification
learn       → TrainingExample
explore     → WorldUpdate
forge       → ProgramCandidate
```

This gives you GoT telemetry without changing behavior first.

### 3. Replace the flat Facts block

Keep the old flat chain path as a fallback, but add a graph serializer that emits:

- Goal node.
- Current-state node.
- Top relevant facts.
- Typed edges.
- Contradictions.
- Candidate operations.
- Provenance and timestamps.

### 4. Add bounded graph expansion

Start with small limits:

```text
max_nodes = 32
max_edges = 64
max_depth = 3
max_candidates = 8
max_vm_runs = 8
```

The existing experience with large ASK candidate sets is a warning here: at level 3, the VM encountered altered/duplicate candidates when the operand set grew beyond roughly 100, while a 16-candidate cap remained manageable. Keep GoT expansion bounded and ranked rather than exposing the entire program library. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

### 5. Add a verifier-aware beam

You probably do not need full MCTS initially. Use a beam of graph states:

```text
expand → normalize → deduplicate → execute → score → retain top B
```

For each state, preserve:

- The graph delta.
- The selected action.
- Verification status.
- Cost.
- Reward.
- Failure classification.

### 6. Add graph-derived SFT

Only after the graph controller is producing stable verified trajectories should you add graph serialization and candidate ranking to the training data. Otherwise, the model may learn formatting artifacts before the topology is stable.

## My assessment

Your system is already unusually well positioned for GoT because it has the components most GoT implementations lack:

- A real executable language.
- A real VM.
- A constrained ASK/resume interface.
- Persistent program lineage.
- An embodied environment.
- Host-side routing and safety gates.
- Regression measurements.
- Explicit “retire, do not delete” behavior.
- A data-generation loop that has already improved v3 → v4 program acceptance. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/313891666/a0e40695-1f61-41ad-aa1b-54e78d649bd4/README.md?AWSAccessKeyId=ASIA2F3EMEYE3QYCZV6B&Signature=g%2BVNOCv5JfBaI5iLSwJ3G4rSKII%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEA0aCXVzLWVhc3QtMSJHMEUCIQCRHTXMQexmSuRM5FZyZBiTX0WjCrkaJpYbtoiUlGEWFAIgOR7581w7nRFo%2B4Zk8SR6i%2BwyZ1OHrMItrcRGn3UvA7Mq%2FAQI1f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDLuSYwLoo%2FSVh8Q0HSrQBDgDfDI3xx56FzxAe8RIlQ70Rj1sVkB78gr0Vr1HlNRGHP0gOVZiz%2Fd71o5Q2Aok329IgVDVNKN2w%2Fy3zq1QMeoHRBjZys%2FVO81qMCRP5674PpHXVh%2BKcUNaBs8eZ4%2FjMpqz%2BVZjkKwKLAJKGanmC98T%2Bt9hKrFjdF1fvc863LofQjh64rH52MIv%2BTSUXKXNzyoJ4aDXcIIuT84cW2nPVck%2F%2FzgiLpm50xyazxRrEdq9JDXF5kS3qdTS0w0nmAVYKBVjA%2BCUbEhH%2BJlqqfAfis0rOJ5lyKU0%2FL6nGPPbzlfekA0tEKGbd9plpsOuFCNegSsVaUEh8YjZ%2BFF7ap1TpjOiMA7VHTH%2BkhtEbqDwbkCMdRb03BUN8HEiGzf0nJM2dmTxYPBl2ksNzYTDs96yM1VfqYSwbUc8AoILXkl1tm5o91iHWo4zWi6iQx11ke1jHIz31PRFLDljMM9WI6CgfsSFqjLr9AvIXOU4m2GMbWvzHfRbTxISrs4EK5OuTmv5W1%2B1m93CWw8fDzkVKXbVfGMogcnst6%2B5MeSbMZCJdrPKWpY1gtGyDgmeDFUM6V%2FtYpprOcs0ZyfFgc%2Bw%2BVmD8otbeVL%2BhhY3Q49xKEIkFiT30lIX1AkIYl%2FXbvxPkI%2FarS6%2Bnor8lucK3gPBIo%2FdJsNwaEk3C88623NJn2lYrQHGpcL6fvc5FGytPKyeMwjj22y9L%2BbYjlWihMLFRXqONLbH2zxF%2FQfXaEO4X4O8IGRo7sXC%2BGaTuXlwMDIa0Klj1APg%2FZzFYKQfM9nN4DeRLrkwzfXj1AY6mAHaT1qheBbhTN77G8FVNnRfPkNsPpWh1iLWwzRkA%2BZgxyQw57XrBQpLBROz70KidJ%2BTBGLE64ohZNU3FQQtUuEYDJFfjk4lPAh6fhYvF3sp5SXe%2BPL098NS6oN305wdnnDOysh6hajThHWglkCLBXUGV9dlb2kqGr%2BT911%2FAXk9k%2B60PZVIdrgsnCcTQUIYWY%2FG1fmnl3IVQw%3D%3D&Expires=1788414112)

I would define the next milestone as:

> **GoT-1: the agent can maintain a typed, provenance-preserving graph of observations, facts, candidate programs, executions, failures, and verified skills; it can select among bounded alternative derivations and produce an SFT record only after held-out validation.**

That is more valuable than simply adding a GoT prompt. In your architecture, GoT should become the **cognitive workspace and experiment lineage**, while CubeLang/VM remains the **symbolic execution and truth gate**, and the emitter remains a **proposal generator plus language interface**.