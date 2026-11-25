# 0086 - MATH, in stages: a symbolic world that checks answers, then the 450M's steps inside it
status: proposed (Cowork, 2026-11-25, on Nick's "plan for MATH nonetheless"; stage 0 opens on his yes)
written: 2026-11-25 by Cowork · approved: — · hypothesis: H-E23

## Why
MATH (Hendrycks et al., NeurIPS 2021; MIT; `EleutherAI/hendrycks_math` on Hugging Face) holds 7,500 train and 5,000
test problems in seven subjects (prealgebra, algebra, number theory, counting and probability, geometry, intermediate
algebra, precalculus) at five levels, each answer in `\boxed{}`. MATH-500 (`HuggingFaceH4/MATH-500`, OpenAI's split
from PRM800K) is the 500-problem subset of test that most papers report, with its subject and level.

A proposal of 2026-11-25 ran SymPy inside the VM (`module.load("python")`), and it is set aside, for three reasons:
- the VM would run Python, which deny-by-default and Nick's rule that fetching goes through the host both exclude;
- `sympify` on a model's text evaluates that text as code;
- equality by `simplify(a - b) == 0` can fail to prove two equal answers equal, and nothing in it says "undecided".
Its `solve` simplified the problem's text and its `verify` compared the problem with the answer, so it checked
nothing.

What Cubby has for this today: the step loop (every value the VM's, the numbers placed by the host), the vote and the
answer-kind rules (0078), the school, and certificates (0085). What it lacks: symbolic operations, and a sound way to
say whether two answers are the same. The plan keeps the rule the rest of Cubby follows: the model proposes, a world
computes, the gate disposes, and an answer the world cannot decide is refused.

- **What it brings CubbyLLM:** a path from grade-school arithmetic to competition mathematics in which no value is
  the model's and no answer is accepted without a check.
- **Over a conventional model:** a model's boxed answer is compared by string matching or by a judge model; here a
  world compares it exactly, or says it cannot decide.
- **What it does for you:** a mathematics answer from Cubby is computed and checked, or Cubby tells you it could not
  check it.

## Stages

### Stage 0 - the checker (no model; CPU)
- **Where:** a MoWM world (`algebra`) that the VM calls through its world calls (CubeLang P5), never Python inside the
  VM. MoWM owns the worlds (Nick, 2026-11-10).
- **The parser:** MATH's answer forms (integers, fractions, exact decimals, radicals, π, e, i, degrees, intervals
  with open or closed ends, tuples, sets, ± pairs, polynomials in a few variables) parsed by a grammar into an
  expression tree. Nothing is ever evaluated as text: no `eval`, no `sympify`, no shell.
- **Equality, three verdicts:**
  - **equal:** the canonical forms match (exact rationals, simple surds, sorted sums and products);
  - **equal at points:** a rational or polynomial expression that does not canonicalize agrees at 16 random rational
    points in exact arithmetic (a false agreement is bounded by the Schwartz-Zippel argument); reported apart, never
    counted as "equal";
  - **undecided:** anything else, and the answer is refused.
- **Read:**
  - (a) MATH train's 7,500 golds, each against fixed rewritings of itself (`\frac` and `/`, `\dfrac`, exact decimals,
    reordered sums, `\left` and `\right`, spacing, units written in text, degrees): **equal or equal at points on at
    least 98%**;
  - (b) a hand-built set of near misses (an open against a closed interval, 1/2 against 2, x + 1 against x - 1, a
    missing ± branch, √2 against 2, 0.333 against 1/3, a set missing one element): **equal on none**.
- **Kill:** any near miss called equal. Nothing downstream runs on a checker that can say "equal" wrongly.

### Stage 1 - the floor, with the arithmetic Cubby has (no training)
- The arm 0078 keeps (the step loop, the vote, the answer-kind rules), numbers placed by the host as today, answers
  compared by stage 0's checker.
- **Read:** a dev slice of 500 problems from MATH train (fixed by seed), then MATH-500 once. Right, wrong and refused,
  by subject and level, with "equal at points" counted apart.
- **No gate.** It measures how much of MATH is arithmetic in disguise (prealgebra, algebra at levels 1 and 2).
  Spoken-wrong above 1% is flagged.

### Stage 2 - symbolic steps (training; Colab)
- **The step vocabulary grows** by operations the world computes, each picked by the model and run by the world as
  `add` and `mul` are run by the VM today: solve a linear or quadratic equation for a named unknown, substitute,
  expand, factor, simplify a radical, gcd, lcm, mod, n choose k, factorial, floor, ceiling, evaluate a polynomial, sum
  a finite series.
- **Data, program first (H-E18):** symbolic programs sampled and run in the world, worded by a writer (a dataset model
  for the wording only, never the label), kept when a blind re-solve lands on the world's value; and MATH train's
  solutions converted where their steps map onto the vocabulary. MATH test never.
- **Gate:** on the dev slice, right at least stage 1 + 5 points with spoken-wrong at most 1%; then MATH-500 once.
- **Kill:** under stage 1 + 5 points on dev.

### Stage 3 - later: enumeration and coordinates
- **Counting and number theory:** a bounded answer (a count, a residue, the smallest n) is checked by enumeration in
  the world when the search is small (at most 10^6 cases), so it is verified, not trusted.
- **Geometry:** exact coordinates (a world of points, lengths and areas over surds), the way Seed-Geometry pairs a
  model with a symbolic engine.

### Out of reach for now
Proofs (a Lean checker) and olympiad problems are noted, not planned.

## Acceptance (pre-registered)
Each stage's read, gate and kill are written in its section above, and a stage opens only when the one before it has
passed its line. Every answer carries a certificate as 0085 writes them (each symbolic step's program, the world's
value, the checker's verdict).

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- **Held-out discipline** (the hypothesis file, section 0): MATH test (all 5,000, MATH-500 among them) is never trained
  or tuned on; the dev slice comes from MATH train; MATH-500 is read once per stage.
- **Contamination:** MATH's problems are widely on the web, so the base's corpus may hold some. Before any read,
  MATH-500 is checked for 10-word runs against the base's token cache, and the count is reported.
- **No text evaluated,** anywhere in the path.
- The VM stays the gate; the world computes and compares; the model proposes.
- Frontier models' MATH numbers enter the paper only as published, each checked against its source.

## Out of scope
- Proofs; a larger base (H-E20 is parked); judge models of any kind.

## Who
Stage 0 is a MoWM world and a CubeLang world call: decided when Nick opens it.

## Report
<!-- Claude Code fills this in, stage by stage -->
