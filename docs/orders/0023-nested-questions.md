# 0023 — Nested questions over the worlds: no loose match at a hop, a wording translated on a miss, a live "as of"
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes") · hypothesis: none (the ask loop's walk) · **after 0014**

## Why
Order 0014 mounted the hdc world (3.4M Wikidata5M facts) and passed every line: 177 of 200 one-hop questions
spoken, all correct (`validation/logs/exp_hdc_bench_2026-10-22.json`). Three things it found are left:
- **A nested prompt answered wrong.** In the smoke run, "the X of the Y of Z" was answered wrong through the
  paraphrase tier's word-overlap tolerance at hop 0. The bench then skipped the set's 88 nested prompts. A loose
  match at an inner hop carries its error into every hop after it, and nothing downstream can see it.
- **A wording the walk does not hold.** "Who is the owner of ...?": the wiki world says "owner", Wikidata "owned by";
  4 of the 7 retrieval-exhausted questions. A store-side "R of" serving "R" was tried and reverted, because "capital
  of" reads the opposite way from "capital".
- **"As of 2019" only.** A time-sensitive fact asked in the present is said "as of 2019" or refused; nothing
  refreshes it.

- **What it brings:** questions with several steps ("Who founded the company that owns X?") answered across the
  worlds, each step exact, and present-tense facts current.
- **Advantage over conventional models:** a chat model chains guesses, and an early slip becomes a confident wrong
  answer. Here every hop is an exact lookup the host can show, and a hop that does not match exactly stops the
  answer.
- **In plain words:** ask a question with several steps and Cubby follows each one exactly, or tells you where it
  got stuck.

## Build
- **A. Exact inner hops:** at every hop but the question's own wording, a relation must resolve exactly (a held
  label, an alias, or the forms); the word-overlap tolerance applies at most to the final hop's wording and never
  feeds another hop. The 88 nested prompts become a bench (`validation/exp_hdc_bench.py --nested`).
- **B. A wording translated on a miss:** when a walk on the held wording finds nothing, try the relation's other
  labels in both worlds (Wikidata labels and aliases for the property; the wiki world's relation names), checking
  the direction with the relation's subject and object types, so "capital of" never stands in for "capital".
- **C. A live "as of", optional:** a present-tense question on a time-sensitive relation fetches the current value
  through order 0007's lookup (Wikidata, cached with a time to live) and says it "as of" the fetch date. Off by
  default; offline, the snapshot rule stands.

## Acceptance (pre-registered)
- **A:** on the 88 nested prompts, **0 wrong spoken**; spoken-correct reported per depth. The one-hop bench is
  unchanged (177/200, 0 wrong).
- **B:** of the 7 retrieval-exhausted one-hop questions, the 4 "owner" cases are answered correctly; 0 wrong across
  the one-hop and nested benches; a test pins that "capital of" does not answer "capital".
- **C:** 50 time-sensitive present-tense questions: each said with a fetch date or refused, **0 flat without a date**;
  the Wikidata calls counted (cached; keep them under ~200).
- Off = unchanged; full `pytest tests` and `pytest standin/tests`.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Sources are internal: a fetched value is said in Cubby's words; the audit view keeps the fetch.

## Out of scope
- Questions that need arithmetic or comparison across hops (the emitter's programs).
- Mounting more worlds (order 0015).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
