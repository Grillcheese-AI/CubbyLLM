# 2026-09-13 â€” every step listened to: the event stream and the three.js control panel

Nick, 2026-09-13: "one more thing that always gets overlooked, all process should be listened to at every
step so we can visually check what's happening in a three.js control panel with links between query and
such." Built the same afternoon, into the host rather than beside it.

## The events (`cubbyllm/reasoning/events.py`)

Every step the loop computes is already a value â€” the question, the plan, the disposer's verdict, each
hop's lookup and the fact it found, the sources asked and what the gate said of each fact, the VM call,
the answer or the refusal. `emit(kind, parent, **fields)` records that value as an event with an id and
its parent's id, so a listener can draw the run as a tree: a node per event, an edge to its parent.

The kinds and their links: `question` â†’ `plan` (a rewritten plan is a new `plan` under the old, with
`how` = the lever and `rewrote` = the translations) â†’ `walk` (verdict, answer, refusal) â†’ `hop` (query,
fact, how it was found, similarity, the fact's provenance when the store keeps one) â†’ `fact` (subject,
relation, object); `walk` â†’ `vm` (verified, program length, repairs) and â†’ `latent` (a verified chain
held back); `plan` â†’ `alias` (lever 6); `question` â†’ `fetch` (source, entity, how many facts) â†’ `gate`
(fact, status, clash, provenance, whether a latent hold was lifted); `question` â†’ `answer` (verified,
answer, reason, refusal, aliases, entities asked, facts fetched).

`learn_and_answer` emits all of them; the serving brain's own walk (`standin/serve.py`, `walk_facts`)
emits question â†’ walk â†’ hops â†’ answer through the same `emit_walk` / `emit_answer`. No sink registered =
a dict built and dropped: the loop's behaviour is byte-identical with or without a listener (pinned in
`validation/test_search_learn.py`: same result, same entities, and the tree's every parent is an earlier
event of the same run). Sinks: `JsonlSink(path)` (one line per event; `--events path` on `exp_r11` and
`exp_r18`), `MemorySink` (tests, replay), and the stand-in server.

## The panel (`dashboard/control_panel.html`)

A single self-contained page (three.js r128 from cdnjs, nothing else). A run's events become a tidy
tree per question â€” root at the top, hops and facts beneath, time running into the screen â€” with nodes
coloured by kind and by verdict (verified green, refused rose, held-latent amber), an edge to each
parent, labels that follow the nodes, and a replay scrubber that plays the events in the order they
happened. Click a node for every field it carries; a walk or answer also lists the chain the VM checked,
with provenance. "All questions" lays the run out as a galaxy â€” one node per question, coloured by its
final verdict â€” and a click opens its tree. "Load events .jsonl" takes any run's file.

Served by the stand-in server: `python standin/serve_api.py â€¦` then `/panel`; `/loop/events?since=N` is
the ring of the last 20,000 events, `/loop/stream` is server-sent events (the backlog, then live, a ping
every 15 s), and the panel's `Live` button subscribes to it â€” a question asked in chat or by a running
experiment in the same process appears as it happens. The same page is published as an artifact with a
sample run embedded (hdc, host planner: verified one-hops, a lever-4 rewrite, an ambiguous hop, a
coverage refusal, a two-hop chain).

Next: the emitter's own decode as an event (tokens, time), the VM's per-hop similarity from the resident
session, and the source fetches' timing â€” the panel then shows where the milliseconds go.
