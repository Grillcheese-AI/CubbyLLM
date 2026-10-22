# Everything Cubby is given goes into a world

Nick, 2026-10-22: "we need to store by worlds everything in datasets, not just use them as pretraining" and
"everything in datasets is stuff Cubby can see once and learn."

Pretraining teaches the trunk language. A fact learned into the weights needs ~100–1,000 sightings (H-K7), and it
comes back blurred and without a source. A fact stored in a world is learned in **one** sighting. It is recalled
exactly, by address (loci tokens, H-K10; run 5: 0.946 on facts seen once, 800+ tokens back), checked, and cited.
So every dataset that holds knowledge is also stored, as rows with provenance, in the world it belongs to.

The per-dataset list (names, local paths, sizes, routes) is `docs/worlds_manifest.local.tsv`. It is **git-ignored**:
this repo is public, and it names local collections. This file is the policy that list follows.

## Kinds of world
| world | what is true there | examples of sources |
|---|---|---|
| **reality**, split by era and year (H-F2) and by domain | claims about the real world | knowledge graphs, reference and scholarly books, encyclopedias, dated news, Wikipedia |
| **fiction**, one per work | true only inside that work ("in *X*, Y is Z") | novels, fiction collections, poetry, drama |
| **self** | who Cubby is | the identity corpus |
| **data** | rows of a table, with the table's own meaning | tabular sets |
| **quotes** | who is *recorded* as having said what | quote collections |

A fiction world never answers a question about reality. A question reaches it only when it names the work, or
asks inside it.

## Tiers (Nick's rule, 2026-10-22)
- **attributed:** one source says it. Spoken with its source ("According to …").
- **corroborated:** two independent sources agree. Different authors, or a book and Wikidata, count as
  independent. Wikipedia, DBpedia and Wikidata count as **one** family. Spoken as fact.
- **held:** sources disagree. Both are kept; nothing is spoken flat.
- **witness:** web text. It counts toward corroboration and is never spoken alone.
- **as of:** time-sensitive relations carry their source's date. They are spoken with it, or refreshed live.

## Routes
- **Triples** (knowledge graphs) load directly (order 0014's loader).
- **Text** goes into a passage index with its citation. Frames, then the Cubby-native reader, read facts from it
  (orders 0012, 0013). Procedural books (how-to, exam prep) stay mostly passages.
- **Events** (dated news, event records) go to the year's world.
- **Excluded from serving worlds:**
  - **evaluation sets**, so that a bench measures the loop, not a leak; each bench declares which worlds it
    may see;
  - **synthetic or LLM-generated** sets (unverified);
  - **dialogue and SFT** sets;
  - **build artifacts**.

## Rights
Many sources are commercial books. They stay off-repo. Passages are internal evidence. A user sees the citation
and at most a short quote. Nothing from them is committed, and no book title or local path appears in a committed
file.
