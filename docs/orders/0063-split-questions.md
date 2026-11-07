# 0063 — Split questions: a planner splits a question across specialists, each part is certified, and so is the way they combine
status: open (queue: after 0056, which waits for 0049's first promoted adapter)
written: 2026-11-07 by Cowork · approved: 2026-11-07 by Nick ("yes", after his multi-agent swarm example) · hypothesis: MoWM routing (route or spawn), H-E15, with 0056's batched adapters

## Why
Nick's example is a swarm. A router splits a problem, two workers solve their parts in parallel, their outputs meet
at a join, and a reviewer marks the result certified because the workers converged. Most of that is already Cubby's
design: MoWM routes to specialist worlds, several proposers exist, every answer records its proposer, and 0056 serves
many adapters together. Two things are missing: splitting a question into parts, and certifying the combination.
Cubby's rule also changes one step: agreement can filter answers, but only the VM certifies.

**What it brings, in plain words:** Cubby can split a big question into parts, send each to the specialist that knows
that area, and give one answer, but only if every part and the way they're combined both check out. Otherwise it
tells you which part it couldn't do.
- **Over a conventional model:** multi-agent systems let agents vote or review each other, so several agents can
  agree on the same wrong answer.

## Build
0. **First, check what CubeLang runs today:** its spec lists the `parallel`, `joined` and `async` modifiers; record
   which ones are enforced. Missing features go to the cubelang repo as their own job, with Nick's yes.
1. **The planner** splits a question into parts, with typed slots linking them. For example, "the year X was founded"
   feeds "who led Y in that year". It also proposes a combination program. The disposer checks that the plan covers
   the question.
2. **Routing:** each part goes to a specialist world by MoWM's rule (bundled centroid with a margin gate: route, or
   spawn). The specialist's proposers write the part's program, and the parts run in parallel VM sessions.
3. **The combination is a program too.** The VM runs it over the certified values of the parts: arithmetic, a
   comparison, a join, or a relation between periods (0057's addendum). The whole is certified only if every part it
   uses is certified and the combination is certified.
4. **Partial answers:** if a part is refused, the reply says which parts are answered and which are not, with the
   reason ("I can tell you when it was founded, not who led it then"). The missing part is never guessed.
5. **Agreement filters, it never certifies.** When two specialists answer the same part and disagree, the part is
   refused. When they agree, an uncertified answer stays uncertified.
6. **The final sentence** uses certified values only, through the talk voice's name-and-number check.
7. **The record:** each part with its route, proposer, program and verdict, then the combination, in the record
   format 0059 settles.

## Acceptance (pre-registered)
- **A battery of 300 questions** of 2 to 4 parts, written by rule from the store, in EN and FR:
  - comparing two entities;
  - feeding a date into a second question;
  - aggregating over a list;
  - planted cases where one part cannot be answered.
- **Hard lines:**
  - 0 wrong;
  - every partial reply names its missing part correctly, checked against the planted design.
- **Reported:**
  - right, wrong and refused, and the partial answers;
  - parts routed and parts spawned;
  - time with the parts in parallel against in series.
- **Later:** the multi-construct queries of the query-log benchmark (0057), once that set is ready.
- **Kill:** any wrong answer that comes from a combination step. Combinations are then limited to the operators
  already tested, until it is fixed.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model on the serving path or in the data. No model judges another.
- **CubeLang changes go to the cubelang repo,** only with Nick's yes, and the main checkout's `target/release` binary
  is never overwritten.
- No local path in a committed file.

## Out of scope
- Agents that act in the world (tools, email, payments).
- Agents negotiating with each other.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
