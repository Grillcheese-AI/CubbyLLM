# 0092 - A first answer in seconds: the Wikidata lookups batched and in parallel, the seats warmed at boot, and each step's time on the record
status: open (Code B, after 0087, before CubeLang P7 and P8)
written: 2026-11-27 by Cowork · approved: 2026-11-27 by Nick ("yes") · hypothesis: none (serving speed)

## Why
Nick, 2026-11-27: "the model is very slow to serve." The served brain (cubby-stack's `cubby` preset:
`base450m_cont`, `cubby450m_v12e_w_step_cont`, `talk_v3a`, on the home card) took 13.9 s for "hello" (the first ask
after it started), 12.9 s for "who is marie curie" and 7.7 s for "who are you?" (`wall_s` in the ask history). Code
B's 0083 bench, the same model and card with the web off and the Wikidata answers already cached, took a median
2.44 s.

Measured by Cowork the same day (a throwaway cache and the source's own transport; nothing of the brain's touched):
- **Wikidata, a first lookup:** Marie Curie 17 calls in 8.41 s (4.95 s on the network, 3.40 s in the 0.2 s pause
  after each call, one call after another); Ada Lovelace 6 calls in 2.62 s; "you" 7 calls in 2.87 s. The same
  lookup from the cache: about 1 ms.
- **The VM:** 0.3 ms a run in the resident session (25 ms as a fresh process). Not the cost.
- **The Python:** the brain's interpreter is the only one on this machine with grilly2's `grilly.infer`; the system
  Python and CubbyLLM's own environment hold grilly 0.4.7, and llama_cpp 0.3.30 is the same build in all. Not the cost.
- **Not timed yet:** the seat's program (about 31 to 34 ms a step in 0083), the talk seat's paraphrase on profiles
  (written, then not kept: no prose on any of the three), and the first decode after boot. Step 1 times them.

- **What it brings CubbyLLM:** answers at conversation speed on the home card, with every step's time on the record.
- **Over a conventional model:** a hosted model's latency is its vendor's; here each step that costs time is named
  in the answer's own record, beside the lookups that answer cites.
- **What it does for you:** Cubby answers a "who is" question in a few seconds instead of ten or more.

## Build
1. **The stage times on the record, first** (no change to what is answered): each ask record gets `times`, in
   seconds per stage: the seat's decode (with its token count), each source fetch (calls, network, pause), the
   namesake pick, the walk and the VM, the talk paraphrase (tokens, kept or not), the web and its reader. Measured
   before anything else changes, on step 4's questions.
2. **Wikidata when serving** (`CB_WD_PARALLEL`, off by default): label and claim lookups batched (`wbgetentities`
   takes up to 50 ids a call); the independent calls of one ask in parallel, at most 4 in flight; no pause after a
   call. Benches and bulk runs keep the pause as today. The source's state made safe for its own threads: the cache
   keeps its whole-file rename, and the links and choices files take a lock. The facts are merged in the order the
   sequential source returns them.
3. **The seats warmed at boot** (`CB_WARM_BOOT`, off by default): after the model loads, each seat decodes one fixed
   short prompt, so the first question does not pay the first decode; the boot log says how long it took.
4. **The read:** 30 "who / what / where is" questions about entities the brain's cache has never fetched (drawn by
   seed from the wiki world's names, each checked against the cache before the draw is fixed), and 30 gen-3 rows
   (seed 7). Through `cli.boot` with the preset's arguments and the web as served; flags off, then on; each setting
   with its own empty cache folder; off and on alternated question by question so network drift falls on both.

## Acceptance (pre-registered)
- **Gate:** flags on, the 30 fresh questions' median `wall_s` at most 3 s and p90 at most 5 s; the first ask after
  boot at most 2 s above the median; and on all 60 questions the same outcome as flags off (the same verified answer,
  or the same refusal reason). The change is speed only.
- **Kill:** an outcome that differs from flags off, or Wikidata errors (429s or failed calls) above 1% of calls with
  the calls in parallel. The parallel part then stays off, and the report says why.
- **Reported, no gate:** the stage times before and after, per stage; calls per question; the share of time in the
  talk paraphrase and how often it is kept.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Nothing in `plan_verify`, the walk, the VM or the gate changes: only how fast the source answers and when the
  seats warm.
- **Wikidata's etiquette:** the source's User-Agent stays; at most 4 calls in flight; a 429 or a `maxlag` answer
  backs off and retries once, as the source retries today.
- **The brain's files:** the read never writes the brain's ask history (`--history ""`) and never empties or writes
  its Wikidata cache; each run has its own folder.
- **The home card:** say so in the inbox before a long run; Code A's 0082 reads come first.
- **Serving:** setting the two flags in cubby-stack's `cubby` preset (the built-in one in `stack.py`) is Nick's call,
  after the report.
- The audit's section 6 (no lock in `CubbyBrain`, the board's cache, request sizes) stays its own item; this order
  only makes the source's own state safe for its own parallel calls.

## Out of scope
- The web reader on the GPU (the LFM 2.6B runs on the CPU, and only on web fallbacks; none of the three asks above
  used it).
- When the talk seat paraphrases a profile (0079's ground): reported here, changed there.
- Batches above 1; route 5 and memory (both killed in 0083).

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
