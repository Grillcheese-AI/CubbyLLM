# CubbyLLM results, by date

These are the result tables the README carried until November 2026, moved here as they were written on their
dates. Some numbers were superseded later: `CUBBYLLM_HYPOTHESES.md` holds each claim with its kill line and its
latest result, and the work orders in `docs/orders/` hold the runs since October. Every row names its experiment;
its log is in `validation/logs/`. The README keeps the current headline numbers.

In the September tables the answers were proposed by the host's grammar, the LFM2.5-2.6B stand-in or a frontier
model probe, as each row says, not by GRL's own 450M; the October section measures the 450M itself.

## 5 October 2026: GRL's own model

Two lines of work on GRL's own model, each run as a hypothesis with a kill line in `CUBBYLLM_HYPOTHESES.md`
(H-E19, H-M1); every number has a log in `validation/logs/`.

**Arithmetic through the VM, one step at a time (H-E19, `cubbyllm/reasoning/step_loop.py`).** The emitter never
writes the whole program: each turn it proposes ONE operation on numbers the host placed (the question's, as
`[N1: 308]` slots, and earlier results written back as `[S1: 154 = N1 / N2]`), the VM runs it, and the host writes
the value back. Every intermediate number is the VM's, never the model's.

| measured on 236 held-out word problems (base450m + emitter adapter) | |
|---|---:|
| whole program in one pass → one VM-verified step at a time | **5.5% → 22.9%** correct (the 2.6B stand-in, one pass: 85.2%) |
| templates seen in training / never seen | 24.6% / 20.4% — the gain is not memorisation (n = 138 / 98) |
| speak only when the 450M loop and the 2.6B stand-in agree | **0.8% of questions answered wrong**, precision 0.964 |
| every legal next step ranked by the adapter (the host's "menu", ~135 options) | right step top-1 0.62, **top-3 0.86**; stops 0.99; the margin is calibrated (lowest quarter 0.41, highest 0.98) |
| what did not move it | self-vote over 5 samples (0.237), a VM-judged step school (0.229), a pointer head on frozen features, a listwise fine-tune (0.119 / 0.059); model-free pruning catches 3% of the real mistakes |
| pick each step from that menu instead of writing it (no training) | **0.275** (seen 0.290 / unseen 0.255) |
| branch to the top-3 where the model is unsure (cutoff set on a separate dev set) | paths ranked by mean log-prob per step (ranker chosen on dev) **0.309**; the right answer is among the finished paths for **0.496** — choosing between paths is now the bottleneck |

What it brings over a conventional LM: the model is never trusted with a number. It proposes, the VM computes,
the trace is inspectable step by step, and a question is answered only when two independent proposers agree.

**An n-gram memory in the hypervector space (H-M1).** Qwen's 2026 preview reads a 51B-row n-gram table from host
RAM by hashed address. Here an n-gram is bound from the qFHRR word codes the serving table already holds
(`docs/papers/2026-08-07-qfhrr-semantic-word-tables.md`; 60,147 words, 10,240 dimensions): the Rubik's-cube encoding
from `grilly`'s cubemind cache, role (position) XOR filler (the word), bundled by majority and recalled by a
Hamming scan. 597k corpus trigrams, one word swapped, match accepted when the original is the nearest key within a
distance cutoff:

| table (teacher) | paraphrase reaches the row | a different fact (france → germany) | random word |
|---|---:|---:|---:|
| shipped table (potion-retrieval) | 22.3% | 0% | 0% |
| paraphrase-MiniLM | 74.7% | 10.7% | 0% |
| paraphrase-MiniLM + Attract-Repel (WordNet, test words held out) | **74.3%** | **4.0%** | 0% |

What it brings over a hashed n-gram table: a reworded n-gram still reaches its memory row, where a hash reaches
nothing, and one distance cutoff sets the trade-off against reading a different fact; the exact address stays
available as an identity tag. Open: the paraphrase list is 35 hand pairs (a larger fresh list is next), the scan
is measured on CPU (~330 ms over 597k keys; the GPU scan is next), and the language-model test has not run.

## 11 September 2026: the reasoning stack

The reasoning pipeline (`cubbyllm/reasoning/`) is wired end to end and measured on two worlds:

```
question ──► plan ──► plan_verify (disposer) ──► lookup-first walk ──► CotChain program ──► CubeLang VM ──► answer / refusal
          └─ a refusal the store caused ──► source ──► gate ──► store (+provenance) ──► walk again   (learn.py, lever 3)
              │              │                        │                                        │
     grammar or emitter   relations known?     TripleIndex, exact hop 0,          per-hop similarity ≥ frame-size floor,
     (CotPlan: SEED,      plan covers the      paraphrase tier on hops ≥ 1        control role below the floor; verdict
      HOP1..HOPk)         question?            (Jaccard ≥ 0.6)                    over a resident process, ~2 ms/question
```

| measured on the 800-question eval (1,241 facts, 165 relations) | |
|---|---:|
| verified coverage, lookup-first + disposer | **0.718** (574/800) at **precision 1.000** — every wrong answer the pipeline ever spoke here was a pick between several true facts; an ambiguous hop is now a refusal (lever 3) |
| honest refusals with a named reason | 226 (152 no-such-edge or plan-coverage · 28 no seed fact · 9 ambiguous, candidates named · 37 unparseable) |
| misparsed plans the disposer stops before any walk | 90 / 92 |
| VM-verified 3-hop answer, wall, resident VM | **2.0 ms** (65.7 ms per-process; CoT is now faster than chase-only) |
| gen-2 emitter (`emitter_v12e`) vs gen 1 on the 79 held-out B questions | 10 vs 5 accepted + gold hop; 4/4 verified correct, 0 wrong — bar met |
| gen 2 outside the grammar, arms B + C (exp_r7 --exclude) | **18/116** by emitter plan alone (B 7/79, C 11/37) after coverage levers 1–3, 0 wrong; 14 more C composed through the grammar (GoT-1 slice, exp_r8), 0 wrong |

| measured on the wiki world (552,297 facts, 2,967 relations — the emitter has seen 165) | |
|---|---:|
| 300 canonical two-hop chains (exp_g4b) | 300/300 VM-verified |
| matched pairs, 200 chains × 4 surface forms (exp_r9) — out of basin | grammar **0** correct · emitter **228** correct · **0 wrong** on every form |
| canonical / have / relative / possessive, emitter | 110 / 80 / 127 / 21 of 200, 0 wrong; the possessive residue is the emitter folding a hop into the seed, refused correctly |
| canonical form | grammar 189/200 · emitter 110/200 · 0 wrong |
| SimpleQA, 4,326 free-text questions through the whole gate (exp_r10) | 3,858 plans → 3,825 refused with a reason + 33 walk failures; 0 verified, **0 wrong** (the store holds 3 of the answers) |
| search-and-learn (exp_r11): held-out facts, then Wikidata on 600 SimpleQA | held-out: 200 refused → 195 learned and verified, 0 wrong; poisoned source, either order: ambiguous refusals, 0 wrong. Wikidata: 1,715 facts learned with provenance; with the source resolving relation words to its labels (`born` → `date of birth`, lever 4) **4 VM-verified, 3 correct, 0 false facts** (the 4th: Wikidata's transliteration and granularity vs the gold's). A WordNet+WOLF synonym oracle (`reasoning/lexicon.py`, lever 5) adds the bilingual axis — a French question verifies against an English store — and changes nothing on English factoids. With the typed answer class, the near rule and the local property table (12,684 Wikidata properties, EN+FR labels, aliases and datatypes, built once): **6 verified, 5 correct, 1 near, 0 wrong, 0 API calls** on the same 600 — the wording step (`born` → `date of birth`, a two-sense wording narrowed by the question's ask type and the property's datatype) never touches the network; 845 s against 1,395 |
| the offline Source (`standin/wikitext.py`): regex frames over 4.7M local article openings, provenance per sentence, EN + FR | same 600 SimpleQA: 9 facts admitted, all re-read true against their sentences; **2 verified, 2 correct, 0 wrong**, 0 API calls. Run 1 stored one false fact (`British is the author of The Roar`) — two source hazards, fixed and pinned. Half the API source's verified count, offline, with the sentence on record |
| the ceiling probe: a frontier model (Gemini 3.8 Flash) in the emitter's seat, same disposer / walk / VM / kill line (`standin/openrouter.py`; the serving model never calls it) | 600 questions, $0.27: the model declines 403 outright, plans 195; **4 verified, 3 correct, 0 false facts** vs the stand-in emitter's 6 / 5 / 0 the same day. Run 2 spoke one wrong answer — the source picked among several items labelled `James Young`; an entity label shared by two items is now a refusal, never a pick (pinned). What the probe exposed is the host's: lever 6 (`learn.resolve_wordings`, the question's own word for a canonical label, `founded` → `inception`) and a coverage rule that is wording-bound and vocabulary-sensitive. The ceiling on free text is not the proposer |
| the hippocampal side cortex (`reasoning/hippocampus.py`, exp_r13): certified chains as episodes, 256-bit codes, recalled as plan candidates through the same gate | matched pairs, 100 seen chains × 3 out-of-basin forms: **272/300 correct, 0 wrong** (possessive 90/100 — the emitter's 21/200 hole was folding a hop the memory has no reason to fold); 100 unseen chains: 7 verified by rebinding a remembered shape to the new entity, 0 wrong (ceiling 11); 1,200 VM walks, no model in the loop, 37 s. Memory proposes, the host disposes, the VM decides; consolidation retires by utility and deletes nothing |
| the encyclopedia Source (`standin/encyclopedia.py`, exp_r16): 29 OCR'd volumes of a 2005 encyclopedia, entries by headword, frames over each entry's opening, provenance = volume + headword + sentence; no model, no network | volume 13: 718 entries, 204 persons, **577 facts** in 16 s; cross-checked against Wikidata where the name resolves to exactly one item: years **121/125 agree** (the 4 disagreements: a calendar-style year, one entity the search resolved to the father, one genuine), places 58 agree + 6 spellings + 13 the same place at another granularity or name + 2 genuine disagreements; 50 hand-read: 49 true, 1 partial place (fixed, pinned). The OCR's hazards — line-break hyphens, `bom`, captions and bylines run into names and places, a missed headword handing the next entry's birth sentence to the previous person — each a rule, each pinned. All 29 volumes: 22,213 entries, 3,601 persons, **10,790 facts in 16 s**; 60 hand-read: 57 true, 3 the rules above (fixed, pinned) |
| the gen-3 builder (`standin/data/build_gen3.py`): certified chains in, free-text questions out, the plan the host's — each (question, plan) certified by the serving gate itself before it becomes a record; no LLM | 16,626 chains × 50 wordings: **63,138 questions, 52,538 certified (83.2%)**, 105k records in 1,050 s. The first probe certified 49.6%, and its refusals were the coverage rule's own measurement on free text — five host rules later (inflections in the property table; lever 4 through the property's every wording, both ways; a *place* ask type; `covers()` v5, the mixed reading; the question decides an unheld tail's split) coverage refusals went 1,476 → 0. On the held split (entities no record trains on), the gen-2 emitter — which never saw a free-text wording — plans 599/600 and verifies **511 correct, 0 wrong** once lever 7 snaps its garbled seeds to the question's spelling; the 82 it still misses read the ask words as relations, which is what gen 3 trains |
| hdc through the gate (`validation/exp_r18_hdc_rephrased.py`): Wiki5M 1/2/3-hop questions, each in its template wording and an LLM's rephrasing, with the chain's facts — a matched-pair set at 10,000 with nothing missing from the store | 8,840 chains × 2 wordings, **0 wrong** on every row scored. The probe found three gate defects, each pinned: a relation ending in `of` (`instance of`) split at the wrong ` of `; the walk's overlap tier read an *inverse* relation where the table's exact alias named the right one (lever 4 now runs ahead of the walk); and the **split wording** (`covers()` v6: `Which country does X hold citizenship in?` covers *country of citizenship*, word by word, the residual rule untouched). Rephrased one-hop certified went **31.5% → 87.1%**, two-hop 27.8% → 55.1%, three-hop 11.0% → 35.1% (265 rephrasings that dropped the chain's last hop counted, not scored; 0 wrong of 17,415). The battery caught the place ask over-reaching twice: 2 wrong on exp_r9 (a class noun followed by `of` is a relation) and 1 on the emitter arm (`In which country is the location of formation of X?` — the asked class noun is a hop when the store holds it as a relation, exactly); the emitter arm's 11 wrong on its first run all became rules or exclusions — **0 wrong of 1,180** on the rerun. 26,216 gen-3 records with two- and three-hop free-text wordings; exp_r17's bar moved 511 → **531 at 0 wrong**, exp_r9 228/0 and exp_r11 7/5/2/0 unchanged |
| the loop, live (`standin/ask.py`, `serve_api --ask`, the panel's ask box): one natural question at a time, every step streamed to the control panel as it happens, links drawn between questions that share a fact or an entity | The first evening found **one wrong answer spoken** — `Who is the mother of Justin Trudeau?` → Pierre Trudeau, verified: the synonym oracle's *verb* sense of "mother" (beget, sire, father) reached lever 4 behind a property the table knew and the store lacked. Two rules, pinned: the first resolver that names a wording decides (a known, unheld relation is fetched, never paraphrased), and the lexicon words relations with noun senses only. Entity resolution went from "a shared label is refused" (which refused Marie Curie — a physicist, a book, a metro station, a ferry) to three deterministic tiers: the item a served claim pointed at, one exact hit, the question's next hop then label over alias; Bill Haslam's father's given name and Marie Curie's birthplace verify. The gate's snapshot hashed 552k texts per fact (0.35 s) — a running set hash now. exp_r11 on the same 600 SimpleQA with all of it in: **6 / 5 / 1 / 0** (one near traded for an mbiguous_relation refusal; 2,406 facts admitted against 1,328 as shared names now resolve) |
| LFM2.5-2.6B as a fact Source (`standin/lfm_source.py`, the **latent** tier: a model's fact is stored with provenance and never spoken until a second source attests) | Same 600 SimpleQA: 450 facts admitted latent, 0 verified, **7 answers held — all 7 wrong**; against Wikidata's cached facts, **agree 1, disagree 28** (dates of birth 0 / 22). The base model's parametric facts are nearly always wrong on this tail; the latent tier is what made that safe to measure — nothing it said was spoken |
| one VM per user — what isolation costs (`validation/exp_r20_vm_per_user.py`): a resident CubeLang process per session, so no user's frame, knowledge path or program can reach another's | 256 sessions on one machine: **1.69 MB per user marginal** (592 users per GB; summed RSS says 6.8 MB each — the executable's pages counted once per process and mapped once by the OS, a 4× over-count that would become a wrong capacity plan), **19 ms** median spawn + first call, **5,211 programs/s** from 8 threads (3.62× the 1,438/s serial baseline), p99 2.77 ms, 506 MB released on close. Flat from 16 to 256 sessions. Isolation is not the expensive part of serving — the shared emitter is |
| "Who is X?" — the retrieval program (`standin/ask.py`: SEED + ASK, no hop; the host runs it, the VM recovers each fact, the facts are the reply; a mounted talk adapter paraphrases them and the paraphrase is kept only when every name and number in it is in the facts) | *who is marie curie?* → nine facts bound, nine recovered by the VM, the grounded paraphrase spoken; *where is Lévis?* → country, administrative entity, waters, inception; *what is Ottawa?* → `ambiguous_entity` with the four Ottawas named. `standin/data/build_profile_sft.py`: **30,964 (question, program) records** so gen 3 writes the program itself (who 12,000 · what 12,000 · where 6,964) |

The reading: the grammar gets the templates; the emitter gets the shapes the grammar cannot parse; the disposer
and the VM keep the wrong count at zero on both; free text against a store that cannot answer it is refused, not
guessed. The loop closes: emitted plans the VM verified (`cot_harvest_r7*.jsonl`) were the first training records
that did not come from the grammar, and gen 2 was trained on them.

Full record: `docs/research/2026-09-11-plan-verify.md` (the disposer, the resident VM, the emitter's plans walked,
arm C, gen 2, matched pairs), with `2026-09-11-entry-diagnosis.md` and `2026-09-11-hop0-search-is-dead-code.md`.
