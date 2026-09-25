# 2026-09-24 — graph of thought: branch at an ambiguous hop, speak only where the branches agree

**The ask** (Nick, 24 Sep): "graph of thought with branching also needs some love". Three parts, all built:
the branching walk, a thought-graph store, and branches in the panel.

## Why it needed love

Before today, an ambiguous hop was a blind refusal. When the store held two objects for one hop (two
citizenships, two daughters, a poisoned capital), `pipeline._walk` raised `_Ambiguous` and `answer` refused
on the spot, naming the candidates. Nothing tried them.

exp_r31 showed that was a waste. It branched **outside** the walk, with one fresh `answer()` per candidate and
the rivals banned. On CLUTRR's fork records, 32 of 53 blind refusals turned out to be only apparent: one
branch finished and the others died. That gave 32 correct answers and 0 wrong. The mechanism was never moved
into the loop.

## The rule

Every candidate object at the ambiguous hop becomes a branch:

- each branch walks to the end of the chain;
- a nested split branches again;
- the whole search is bounded (`branch=12` continuations);
- each finished branch is certified by the same `_check_chain` the greedy walk uses, so a branch clears exactly the greedy bar.

**It speaks only when every branch finished, every branch was certified, and they all name ONE answer.**
Then the ambiguity was in the path, never in the answer, and saying it is not a pick. Anything else stays an
`ambiguous_hop` refusal, which now carries its branches:

- two certified answers;
- a branch the VM rejected;
- a branch the store could not finish;
- the bound cutting exploration short.

One difference from r31 was deliberate: **a branch the store cannot finish blocks the others.** In an open
world a missing fact is a don't-know (invariant 3), not a "no". Speaking the surviving branch would be
answering for Marie's French citizenship when she also holds a Belgian one we know nothing more about.

`closed_world=True` treats such a branch as refuted instead. It is for stores that are declared complete (a
CLUTRR story), and it is never the default.

Two ways out of the refusal, and neither one is a guess:

- **The loop fetches the gap.** The entity an unfinished branch stalled on is the first thing
  `learn_and_answer` asks the source about. If the fetched fact lets the branches finish and agree, the
  loop speaks.
- **The asker picks the path.** `standin/ask.branch_clarify` offers a mid-chain split as choices ("marie
  has more than one country of citizenship. Which one do you mean?", each choice saying where it leads).
  The pick comes back as `choose`. That narrows the hop to a candidate the store already offered and can
  never add one. The rest of the chain is still walked and certified.
  - **A last-hop split is never offered.** A pick there would be the answer itself coming back "verified":
    the poisoned-capital case (`lyon` beside `paris`). The refusal names the values and asks nothing.

## Measured

`validation/exp_r32_branching_inpipe.py` ran on the same 65 CLUTRR fork records as r31, with the real
cubelang VM, chunk 2:

| arm | spoken | correct | WRONG | refused | VM calls |
|---|---:|---:|---:|---:|---:|
| refuse on sight (`branch=0`) | 12 | 12 | 0 | 53 | 48 |
| closed world | **44** | 44 | **0** | 21 | 439 |
| open world (the serve default) | 12 | 12 | **0** | 53 | 439 |
| open + one asker pick | +32 → **44** | 32 | **0** | 2 | 196 |

- **The closed world reproduces r31 exactly.** That is 12 + 32 recovered, and its 21 refusals split into
  18 divergent last-hop sets and 3 VM rejections, as in r31. So moving branching into the walk changed
  nothing it does.
- **The open world recovers none of them without help.** Each of the 32 has a branch the story cannot finish.
  Those refusals are correct under the don't-know contract, and they are also the loop's best fetch targets.
- **One pick closes the gap.** Of the 53 open-world refusals:
  - 34 splits were mid-chain and were offered as choices.
  - Every one was resolved in a single round, and 32 were then certified correct.
  - 2 were rejected by the VM at 5–6 hops.
  - 19 were last-hop splits and were not offered, as designed.
- **Cost.** Branching costs about 9× the VM calls, but only on a question that is ambiguous; the rest pay
  nothing. Wall time for all 65 records was under a second.

A kill-line check ran in every arm: **0 wrong answers spoken.**

## The thought graph

`cubbyllm/reasoning/graph.py` reads one ask's events as a graph of thoughts, schema 1:

- **Nodes** are typed events. Each has a label and a verdict: ok, refused, gap, held or profile.
- **Edges** carry the relations the event tree cannot hold:

| edge | meaning |
|---|---|
| `derived_from` | the tree itself |
| `supports` | a fact → the certified walk or branch that rests on it; a converged answer rests on *all* its branches' facts |
| `verifies` | the VM → a walk it certified |
| `alternative_of` / `agrees_with` / `contradicts` | between the branches of one walk |
| `learned_into` | a gate that admitted a fact → where the walk used it |
| `retries` / `concludes` | the walks of one question, and the walk the verdict came from |

Every record `AskLoop` keeps now carries its graph, so `ask_history.jsonl` stores how each answer came
about, not only the answer.

A fetch that handed over 150 facts keeps the facts that were walked plus 12 more. The rest are collapsed into
one counted node.

The graph is derived from the events and never edited.

The latent tier reads the same idea: a converged answer is held back if any of its branches rests on a
latent-only fact, not only the branch it shows.

## The panel

`dashboard/control_panel.html` now shows branches:

- **Branch nodes.** Each branch is a node under its walk:
  - green when certified;
  - red when rejected;
  - grey when it is a gap.
- **Links between branches.** Two certified branches are joined by a green line when they agree (why it
  spoke) and a red line when they contradict (why it refused).
- **The inspector.** It lists a walk's branches, and shows a branch's whole chain end to end.
- **The clarify buttons.** They send the pick back as `choose`.

## Found on the way

`events.remove_sink` removed sinks by `==`. A `MemorySink` is a list, and two sinks that had heard the same
events compared equal. Removing the thought graph's per-ask collector took the caller's sink out instead and
left the collector registered for good. It now removes by identity and is pinned by a test.

## Files

- `cubbyllm/reasoning/pipeline.py`: `_check_chain`, `_explore`, `_converged`, `answer(branch=, choose=, closed_world=)`
- `cubbyllm/reasoning/learn.py`: `choose` and `branch` passed through; a stalled branch is a fetch target;
  the latent hold covers every branch
- `cubbyllm/reasoning/events.py`: `branch` events; `remove_sink` by identity
- `cubbyllm/reasoning/graph.py` (new, WIRED via `standin/ask.py`)
- `standin/ask.py`: `parse_choose`, `branch_clarify`, `ask(choose=)`, `rec["branches"]`, `rec["graph"]`
- `standin/serve_api.py`: `POST /ask` takes `choose`
- `dashboard/control_panel.html`: branch kind, sibling links, clarify with `choose`
- Tests:
  - `tests/reasoning/test_branching.py` (10)
  - `tests/reasoning/test_graph.py` (5)
  - `standin/tests/test_ask_branching.py` (5)
- Measurement: `validation/exp_r32_branching_inpipe.py`, with logs at
  `validation/logs/exp_r32_branching_inpipe.{log,json}`
- Hypothesis: H-A11 in `CUBBYLLM_HYPOTHESES.md`

## Next

- **Live askers.** Measure how often a live `ambiguous_hop` is mid-chain, and how often the offered pick
  resolves it. That is H-A11's second kill clause.
- **Set answers.** A last-hop set could be spoken *as a set* ("France and Belgium"). That is only safe when
  the relation is known to be multi-valued (citizenship yes, capital no), so it waits for relation
  functionality to be a stored property rather than a guess.
- **The sleep cycle.** It can read `contradicts` edges from the history: a relation that keeps splitting at
  the last hop is a candidate for a functionality flag or a qualifier fetch.
