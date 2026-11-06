# A terabyte of knowledge outside the weights: a fact store a small model can query exactly, at home

Nick, 2026-11-06: "the problem we need to solve is how can we store / compress 1TB of data into our custom cubby
database for easy retrieval so it has all the facts outside the weights"; "one of the things I thought was to use a
mixture of trunk per topic / specialist or similar". Put to the OpenRouter panel; every answer is an opinion until
it passes the experiment it names.

## What Cubby is
A neurosymbolic stack in which the language model is the smallest, most replaceable layer.
- A 450M-parameter recurrent/attention hybrid (our own) reads a question and writes a program in CubeLang, our typed
  three-address language. A VM (Rust) runs it and checks every hop against stored facts.
- Cubby speaks only a VM-certified answer and otherwise refuses with a reason. The kill line of every benchmark is
  0 wrong answers. It serves English and French (US, UK, France, Quebec).
- The model should not memorize facts at all. **The bottleneck now is coverage:** on SimpleQA, against the store we
  had, the gold was an entity of the store for 860 of 4,326 questions, and a stored fact answered 3.

## What exists
- A 4.3M-fact Wikidata5M world in SQLite; a small wiki world; history books in SQLite FTS5; a general encyclopedia;
  a history graph; fiction worlds.
- Live Wikidata through its public API: candidates from its search, which item a name means settled by hand-written
  tiers. For time-sensitive facts the live value wins over any snapshot.
- **The VM's datastore:** exact keys and aliases in finite-state-transducer segments shared across requests; a
  conversation history with word, prefix and typo search (p95 38 ms); a hippocampal memory whose keys are hashed
  from the key string.
- Codes: a qFHRR word table (60,151 words, D = 10,240), letter-trigram binary codes for near-miss names, 256-bit
  SimHash episodes.

## The data, about a terabyte
- **The full Wikidata JSON dump:** more than 100M items. Our own sizing note says 74.5M entities and 959 GB to
  stream, with 40-80 GB kept; recheck both.
- English and French Wikipedia (about 40 GB of text).
- DBpedia's 4.6M abstracts.
- History book collections, a general encyclopedia, dated historical events.
- Coming: international news (with GDELT) and more books.
- **Facts from text** are read by a Cubby-native reader (our own model or rules), never by an external model. A fact
  from one book is spoken hedged until a second independent source agrees.

## The machine
- **Serving, at home:** Intel i5-12400F (6 cores), 32 GB DDR4 (64 GB possible), AMD RX 6750 XT with 12 GB (Vulkan
  only, no CUDA; our compute framework is Vulkan), local NVMe SSD (say how much the design needs), 5 TB of Google
  Drive.
- **Builds** may run in a Colab A100 session (sessions of up to about a day, Drive for storage) or at home.
- **Latency targets** (proposed; argue with them): an exact fact lookup inside a VM hop under 1 ms p95 when hot and
  under 10 ms from SSD; resolving a name to its candidate items under 50 ms p95.

## Rules that do not move
- **0 wrong:** refuse when not sure; every spoken value is a checked fact tied to its own item.
- **Facts stay exact keys with provenance.** Similarity may only lead to a key, never stand for a fact.
- **Never rank by fame between two of a kind:** two people of one name are never separated by which one is better
  known. Answer each, or ask.
- **No external model on the answering path, and none to build the stored knowledge.** External models (you) may
  help design and build evaluation data, nothing else.
- The serving code never touches the network; live sources sit outside it.
- English and French both, from the start.
- **Prefer learning by storing** (data pushed into the store) over training runs. A trained part is acceptable only
  when its experiment shows it beats storing.

## Measured negatives (do not propose these again without new evidence)
- Superposed associative memories (Hebbian, NLMS, SDM, DG-sparse at D = 1,024-2,048) lose recall between 512 and
  1,024 associations.
- BEAGLE-style co-occurrence hypervectors scored below a surface-only baseline.
- The qFHRR word table is weak on open-vocabulary entity names: 0.081 nDCG@10 on BEIR dbpedia-entity, against
  BM25's ~0.31.
- Similarity thresholds do not travel between deployments.

## Nick's idea, to be judged on its merits
**A mixture of specialists:** a trunk (or an adapter on the shared trunk) per topic, each owning a shard of the
store, with a router choosing one.
- Say what it buys: topic-specific reading of questions, smaller indexes per shard, parallel builds, shards swapped
  in and out of RAM.
- Say what it costs: routing errors, questions that cross topics, and whether it puts facts back into weights.
- Compare it with one trunk over one sharded store. If it is wrong for this problem, say so.

## What the answer must contain
1. **What to keep from the terabyte, and in what form:**
   - Wikidata: truthy statements or full statements, qualifiers (point in time, start and end), ranks, references
     as provenance, labels, aliases and descriptions in EN and FR, sitelinks. Sizes for each choice.
   - Text: which facts a reader can extract without an external model, and which passages are kept for provenance.
2. **The physical layout:** encodings and indexes (for example dictionary-encoded ids, sorted permutations,
   succinct or compressed structures, FSTs for names, memory-mapped files).
   - Name existing systems that store Wikidata at full scale on one machine, with their measured sizes, build times
     and query latencies.
   - Say what to reuse rather than build: local, embeddable, licence named, ideally Rust since the VM is Rust.
3. **Name to item at 100M items, EN and FR:** aliases, typos, namesakes, within the rules above, and how it plugs
   into the VM's hops.
4. **Time and change:** dated values, the live-source rule, updates without a full rebuild.
5. **The mixture-of-specialists idea**, judged as above.
6. **The build:** streaming the dump, where it runs (home or Colab), how long, and how to prove the store is a
   faithful copy of its source.
7. **For each choice, the offline experiment that could prove it wrong,** before downloading the full dump: data,
   metric, pass line and kill line. Held-out sets must be public, carry gold Wikidata ids, and include French.
8. **What not to build, the risks, and the decisions Nick must make.**

Give numbers with units (GB, ms, hours). Name systems and papers precisely so we can check them, and mark anything
you are unsure of.
