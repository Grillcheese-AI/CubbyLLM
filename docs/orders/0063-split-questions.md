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

## Cowork's note (2026-11-07): containers and wiring compile to nothing today
Measured on Nick's `bridge_mesh.cube` with the main checkout's release binary (built 2026-09-29; see 0062's addendum
for the full list). It passes `check`, but `compile` reports "no programs found": `container`, `io`,
`vm.start_from_file`, `vm.wire` and `vm.bridge` produce no bytecode. Until a build from the current source shows
otherwise, the parallel parts of this order run as separate VM sessions driven by the host, not as wired containers.

Records about containers and bridges, such as Nick's JSON-LD draft for the bridge, are written by the host from what
actually ran (the observed status, rate-limit counts and audit log), never declared by hand. A bridge that never
started must not be recorded as "Active".

## Cowork's addendum (2026-11-11): the record of a task, and five outcomes to test
Nick: "yes". The shape comes from the Agent Task Management Dataset (Hugging Face, `Lelonthecodeur`, MIT), a set
written by a program. Only the shape of its record is taken. None of its rows is used and nothing was downloaded:
its traces are all marked "simulated", where Cubby's are the VM's own.

**What it brings, in plain words:** every split question leaves one record that says what was planned, what
actually ran, what proves each step, and how it ended.
- **Over a conventional model:** an agent's report of its own steps is text it wrote. Nothing shows a step ran.
- **For you:** when Cubby says "I checked that", there is a step in the record with its proof, or it does not
  say it.

### The record (step 7), as one shape
- **`plan`:** the parts as steps. Each step names the slot it fills and `depends_on`, the steps whose values it
  reads.
- **`trace`:** one entry for each step that ran: its route, its proposer, its program, its verdict, and
  `evidence`, which is the VM's certificate for that step. A step that did not run has no entry. No model
  writes any of it.
- **`checks`:** one for each value the final sentence uses, naming the certificate behind it.
- **`outcome`:** exactly one of five, with its reason:
  - **done:** every part and the combination are certified;
  - **failed:** a part ran and was refused. The reply names it (step 4);
  - **recovered:** a part failed on its first route or proposer and was certified on another. Both attempts stay
    in the trace;
  - **impossible:** the plan cannot cover the question: a part of a shape Cubby does not answer, or a fact no
    world holds;
  - **needs a clarification:** a name stands for several things, or the question leaves a choice open. Cubby
    asks.
- It is the record 0059 settles, not a second one: the plan is the plan, the trace is the activities, the
  outcome is the decision.

### What Cubby says it did comes from the trace
The final sentence (step 6) already says certified values only. The same holds for what it says it did: a
statement that a step was done, checked or looked up is spoken only when the trace holds that step with its
evidence. It is change h of 0010, applied to actions.

### Acceptance, added (set before the build)
- **The planted cases cover the five outcomes:** at least 30 of the 300 questions for each outcome other than
  done. For "recovered", the first specialist lacks the fact and a second holds it. For "needs a clarification",
  a planted namesake.
- **Hard lines:**
  - the recorded outcome is the planted one on every planted case;
  - 0 trace entries without a certificate;
  - 0 replies that say a step was done which the trace does not hold.
- **Reported:** the outcomes by kind, and for "recovered" the number of attempts.
- **Kill:** one reply that says a step was done which the trace does not hold. It is read before any other
  number.
