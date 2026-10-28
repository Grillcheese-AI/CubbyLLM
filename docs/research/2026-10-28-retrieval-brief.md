Redesign CubbyLLM's retrieval as one system that generalizes, and say how to test every design choice
offline before anything is built.

## What Cubby is (read CLAUDE.md, Architecture.md and docs/WORLDS.md in the code root)
CubbyLLM is a neurosymbolic stack in which the language model is the smallest, most replaceable layer. A
question becomes a program; a VM (CubeLang, Rust) checks every hop against stored facts; Cubby speaks only a
VM-verified answer and otherwise refuses with a reason. The kill line of every benchmark is 0 wrong answers.
Cubby serves English and French (US, UK, France, Quebec).

## What "retrieval" covers here
Everything between a question and a fact the VM can check:
- reading the question: which names, which relation;
- finding the candidate items for each name, and choosing the one the asker means;
- fetching the item's facts, then choosing among several values and dates;
- recalling what this asker said before (corrections, choices);
- finding the passage a reader turns into facts (books, encyclopedias, news).

## What exists (paths are in the code root)
- **The walk's retrieval** (`cubbyllm/reasoning/retriever.py`, `index.py`): an exact triple index first, then
  cosine search. It reports candidates and a reason for every hop.
- **Worlds** (`cubbyllm/worlds/`): a 4.3M-fact Wikidata5M world in SQLite (`hdc_store.py`, `hdc_world.py`), the
  wiki world, history books (FTS5), an encyclopedia, fiction worlds, a history graph. Names of several items:
  `hdc_namesakes.py`, `hdc_relation.py`, `hdc_kinds.py`, `hdc_cues.py`.
- **Live Wikidata** (`cubbyllm/sources/wikidata*.py`): candidates come from Wikidata's search API; which item a
  name means is settled by tiers in `wikidata_resolve.py` and `wikidata_namesakes.py`. The web source holds a
  page's fact until two independent sites agree.
- **Codes:**
  - the qFHRR word table (`worlds/fast_words.py`; paper in `docs/papers/`: 60,151 words, D=10,240, no model
    when serving);
  - capsules' letter-trigram binary codes (`worlds/capsules.py`: near-miss names);
  - the hippocampus' 256-bit SimHash episodes (`reasoning/hippocampus.py`: certified chains recalled as plan
    candidates).
- **The VM's datastore:** a layered knowledge store (exact keys and aliases in finite-state-transducer
  segments, shared across requests), a conversation history with word, prefix and typo search, and a
  hippocampal memory whose key vectors are hashed from the key string. So "Georgia" and "Georgie" share
  nothing there. A standalone sealed store (cubby-store) holds per-person histories.
- **The rule at the symbolic boundary:** grilly's block codes and the VM's bipolar codes never exchange raw
  hypervectors; only (symbol, similarity) crosses.

## Measured state
- **Entry is where it fails:** 61% of failures never produce a first-hop seed (`retriever.py`, exp_r4).
- **Everyday scoreboard** (187 WebQuestions-style questions, live Wikidata):
  - 82 spoken, 0 wrong;
  - refused: 44 `ambiguous_hop` (several values for one item), 39 `retrieval_exhausted`, 15
    `ambiguous_entity`, 5 relation reading, 2 time mismatches;
  - the 187 are only the questions a hand grammar parses, and English only.
- **Hand-written tiers decide which item a name means:**
  - label over alias, cue words, a placed-item guard, place qualifiers ("cleveland ohio");
  - orders 0037 and 0043 in `docs/orders/` show them;
  - they were written for rows of that scoreboard and audited on the same rows. Nothing measures them on
    names they were not written for.
- **The search API misses:** the president for "Obama" (at 5, 10 and 20 hits), Henry VIII.
- **Measured negatives** (do not propose these again without new evidence):
  - superposed associative memories (Hebbian, NLMS, SDM, DG-sparse at D=1024-2048) lose recall between 512
    and 1,024 associations (`hippocampus.py`'s docstring);
  - BEAGLE-style co-occurrence hypervectors scored below the surface-only baseline;
  - the word table is weakest on open-vocabulary entity names (0.081 nDCG@10 on BEIR dbpedia-entity, against
    BM25's ~0.31);
  - similarity thresholds do not travel between deployments (the paper, section 7).

## Rules that do not move
- **0 wrong:** refuse when not sure; every spoken value is a checked fact tied to its own item.
- **Never rank by fame between two of a kind** (the "James Young rule"): two people of one name are never
  separated by which one is better known or better described. Answer each, or ask.
- **No external model on the answering path,** and none to build the stored knowledge. External LLMs may help
  design and build eval data, nothing else.
- **`cubbyllm/` never touches the network.** Live sources sit outside it; the live Wikidata value wins over an
  offline snapshot for time-sensitive facts.
- **Facts stay exact keys with provenance.** Similarity may only lead to a key, never stand for a fact.
- **Prefer learning by storing** (data pushed into the datastore) over training runs. A trained part is
  acceptable only when its experiment shows it beats storing.
- **English and French both,** from the start.

## Building blocks Nick wants considered
- Kanerva's sparse distributed memory.
- Codebooks and clean-up memories.
- Binary codes with popcount, LSH or multi-index hashing (grilly is a Vulkan compute framework).
- FHRR / qFHRR with approximate nearest-neighbour indexes (Qdrant runs locally).
- Query by unbinding followed by clean-up.

Say where each fits and where it does not, given the negatives above.

## Data that can be pushed into the datastore
- Wikidata5M (local).
- DBpedia's 4.6M abstracts.
- English and French Wikipedia (about 40 GB of text; whether links survive is unknown).
- History books.
- A general encyclopedia.
- Dated historical events.
- Natural Questions, WebQuestions and SimpleQA (for evaluation).
- **Not local yet:** the full Wikidata dump (74.5M entities, 959 GB to stream, 40-80 GB kept). Downloading it
  is a decision; say what must be shown first.

## What the answer must contain
1. **The design:** the stages, what each stores and from which data, what is a rule (only the guarantees
   above) and what comes from data, and what crosses the boundary.
2. **For each design choice, the experiment that could prove it wrong:** data, metric, pass line and kill
   line, run offline, before building.
   - Held-out sets must be public, carry gold Wikidata ids, and include French; name them.
   - At least one set must test the fame rule: names shared by a popular and a rare entity.
   - Today's pipeline is the baseline on the same sets.
3. **The order of the experiments,** and what each one unlocks or kills.
4. **What not to build,** and why.
5. **Risks,** and the decisions Nick must make.
