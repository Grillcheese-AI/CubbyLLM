# 2026-09-24 — the web behind Wikidata: search, read, corroborate

**The ask** (Nick, 24 Sep): "need to add websearch to it too to find answers". His choices:

- **Backend:** Brave and SearXNG behind one interface.
- **Readers:** structured data and sentence frames, plus the local LFM as a reader.
- **Trust:** two independent sites before a web fact counts.
- **Trigger:** automatic, and only after Wikidata has failed.

## The rule

The search engine brings links. It never brings answers: no answer box, summary or generated text is ever
read.

A page is a **witness**, not a fact. A triple read off the web goes into the store's latent tier, the same
hold an LFM-recalled fact gets. A chain resting on it is refused as `latent_only`, and the would-be answer
is kept on record. The hold lifts only when a **second, independent site** states the same triple.

- **What counts as a site.** A site is a registrable domain (eTLD+1, approximated).
- **One witness, not many.** The Wikipedia family and its mirrors (wikiwand, DBpedia, and similar) count
  once.
- **No witnesses at all.** Archives (web.archive.org, archive.today) only repeat another site, so they
  never count.
- **Across days.** The tally of which sites said what persists, so a second site found tomorrow completes
  a fact first seen today.

## Readers, most structured first

1. **Wikipedia → Wikidata.** When a hit is a Wikipedia article, the article names its Wikidata item, and
   that item's claims come through `WikidataSource`. Wikidata attests them, so they are not held.
   - This path is taken only when ONE item among the hits carries the relation the walk needs. The
     question picks the item, never the search ranking.
   - Two items that both carry the relation are recorded as ambiguous, and nothing is taken.
   - In practice this makes web search a better name resolver for Wikidata. Live, "Ada Lovelace" plus one
     Wikipedia hit gave item Q7259 and 71 attested claims.
2. **schema.org JSON-LD**, read only from nodes *named* as the entity, and never from the page's own
   metadata.
   - Live, Wikipedia's JSON-LD is an `Article` named "Ada Lovelace", written by "Contributors to Wikimedia
     projects". Read naively, that is the author of Ada Lovelace.
   - Page types are therefore skipped (`PAGE_TYPES`), and a test pins it.
3. **Sentence frames.** These are `wikitext.py`'s frames plus day-first and "born in X on DATE" dates. They
   run over snippets and page text, but only on sentences that *open* with the entity: "Albert Einstein's
   friend was born in Berlin" is not about Einstein.
4. **The local LFM** (`LfmReader`, the 2.6B base on the CPU). It reads a prose paragraph (at least 12
   words) for the one relation the walk needs. Every value it writes must pass two checks:
   - **Grounded:** the value occurs in the passage. A date may be spelled differently and still match.
   - **Of the relation's kind:** a date for a date relation, a name that does not start with a digit for
     a place.

## What the live smoke runs showed

These runs used real pages and a fixed list of results, since there is no search key yet.

- **The reader needs every one of those guards.**
  - Before paragraph selection, the first "mention" of Ada Lovelace on Wikipedia was the header beside
    "112 languages". The reader wrote that down as her place of birth.
  - It wrote "1815-1852" as a date of birth.
  - For passages that say nothing about her birth, it wrote "7 November 1867".

  The fixes: prose-only passages, hatnotes skipped, the kind check, and grounding. After them, every
  value it kept was right: 1815-12-10 for Ada Lovelace, 1867-11-07 for Marie Curie from nobelprize.org,
  where the frames agreed.
- **Everything stayed held.** With 3–4 fixed pages per question, no fact reached two independent sites.
  - That is the rule doing its job, not a failure.
  - Real results bring 8 hits with extra snippets. Many of those open with "Marie Curie (born November 7,
    1867 …)", which is where a second site will come from.
  - **Not measured yet:** how often that happens.

## The plan, and what it changed (later on 24 Sep)

Nick's Brave plan licenses the results for AI use **and for storage**, at 50 requests a second, with no
monthly cap. Three things follow.

- **Results are kept.** Every live search is archived in `searches.jsonl`, and only when the backend's
  licence permits it; SearXNG results are never archived.
- **Evidence is kept per site.** The tally stores, for each fact and each site, the URL, the sentence, the
  readers and the time. Any web fact can be traced back to the exact words that stated it.
- **Searching wider.** Each question runs two queries: the label (`"Marie Curie" date of birth`) and the
  way a person types it (`Marie Curie born`). Each asks for 20 results with extra snippets. The first six
  witness pages are fetched in parallel. A limiter keeps the key under the plan's rate.

## Corroboration, second version

The first live battery (below) showed three ways the two-site rule failed on honest data:

- **Spellings.** "London, England, UK" and "London, England" never agreed.
- **Grain.** "Warsaw" and "Poland" split the walk, although both are true.
- **Cut-off snippets.** "Syracuse, New Yor…" made Tom Cruise's birthplace two claims.

The claims step (`WebSource._claims`) is now:

1. **One spelling.** Comma-prefix spellings are one claim, and accents fold.
2. **Entailment.** A finer value vouches for a coarser one, never the reverse.
   - A date to the day vouches for its year.
   - "Syracuse, New York" vouches for New York.
   - Warsaw vouches for Poland when Wikidata places Warsaw in Poland. That check uses P131 and P17,
     intersected over every item with the name, so an ambiguous name vouches for nothing.
3. **One value.** When the attested values form one chain, the finest one is the claim and the only
   value the store gets, so the walk can speak it. When they do not (two different dates), every attested
   value goes to the store and the walk refuses on the split.
4. **Dissent.** One site's different value is not a claim, and it is recorded as dissent. The strict
   policy, `dissent_blocks=True`, lets it veto instead, and the battery measures both.

Also:

- **Snippets cut off by the engine** are read only up to where they are whole.
- **More derivatives count as one witness.** Alchetron, Kiddle and academic.ru are folded into
  Wikipedia; PeoplePill, FamousFix and Born Glorious are folded into Wikidata.
- **grokipedia.com is not a witness.** A page written by a language model is not somewhere this loop
  takes facts from.

## Scoring rule: a place of birth is not a city of birth (Nick)

The question asks for the *place* of birth. A place that contains the gold place ("Sweden" for Bromma) is
therefore not a wrong answer.

`exp_r33` scores such an answer as `coarse` and counts it as not wrong. It checks containment against the
wiki world's LOCATED_IN graph and against Wikidata (P131 and P17, intersected over every item with the
name).

The same check lets "Lugo, Italy" match a gold of "Lugo Emilia Romagna", a name whose comma the dataset
dropped. It matches only when ONE item named Lugo lies in both Italy and Emilia-Romagna. A question that
asks for the *city* would need its own rule.

## The live battery (`validation/exp_r33_web_heldout.py`)

**Setup:**

- **Questions:** 60 withheld wiki-world facts, 15 each for date of birth, place of birth, date of death and
  place of death. Each becomes a one-hop question over an EMPTY store.
- **Sources:** Wikidata's stand-in answers nothing, and the Wikipedia → Wikidata handoff is off, so this
  measures the web's own readers.
- **Search and checks:** Brave, the real cubelang VM, and the LFM reader off.

| run | spoken | correct | near | WRONG | held (of them right) | split | none |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1: first rule | 12 | 10 | 1 | **1** | 12 (7) | 8 | 28 |
| v2: spellings + entailment + one value | 16 | 12 | 2 | **2**¹ | 14 (9) | 2 | 28 |
| v3: + copy detection + live Wikipedia | 15 | 11 | 4 | **0** | 14 (11) | 1 | 30 |

¹ One of v2's two WRONGs was the scorer's fault: "Lugo, Italy" against a gold of "Lugo Emilia Romagna".
Re-scored, the other one stands.

**The WRONG that mattered:** Agha Hashar Kashmiri died on 28 April 1935 (v1, v2) versus the gold 1 April
1935.

- Four "sites" carried 28 April in the **same words**: an older Wikipedia lead that Wikipedia has since
  corrected. They were Alchetron (a mirror), Born Glorious (built from Wikidata, quoting the lead), a
  Medium post and a blog.
- v3 counts sites that copied one text as one witness, and lets the Wikipedia family speak through its
  **live** article. The fact is now held and never spoken.

**What held back:** 3 wrong values, each stated by one site only. Examples: a Japanese date misread by a
Japanese wiki mirror, and "Wilmington, DE" for Talleyville.

**What the web misses:**

- **Place of death is the gap: 14 of 15 were none.** Snippets rarely say "died in X", and this is where
  the LFM reader should be measured next.
- **Some names in the battery are mangled.** The dataset's de-camel-casing produced names like "Marshall
  Mc Luhan", and a quoted search for those finds little.

## Wiring

- `learn_and_answer(..., fallbacks=[web], max_fallback=1)`. When a round admits nothing, or the primary
  source is out of rounds, the loop moves to the next source down, for the entity the walk stalled on.
- A source may carry per-fact `provenance` and `held`, which is how the web names its sites and its hold.
  - These are read *after* the fetch. Reading them before let the first test find a one-site fact stored
    as attested, and the fix went in before anything shipped.
- `AskLoop(web=)` gives the web the loop's own Wikidata source and property table.
- `serve_api --web auto|brave|searxng|off`, with `--searxng URL` and `--web-reader auto|<gguf>|off`.
- The sleep cycle keeps a web fact durable only when two or more sites stated it. That is the same bar it
  has to clear to be spoken.

## Tests

`standin/tests/test_web_source.py` has 16 tests. They pin:

- **Corroboration:** one site is held, two sites attest, and one site with two pages is still one witness.
- **The tally** carries across days.
- **Readers:**
  - frames read only sentences about the entity;
  - JSON-LD skips the page's own node;
  - the reader may only write what the passage says;
  - values must be of the relation's kind;
  - dates are written one way.
- **robots.txt** is honoured.
- **The Wikipedia handoff** happens only when one item carries the relation.
- **The backends** parse their documented response shapes and never read an answer box.
- **The loop:**
  - one site never carries an answer, and two do;
  - the web is asked only after the primary source;
  - the AskLoop record names the sites.

## Not done

- **Live search.** Brave needs `BRAVE_SEARCH_API_KEY` in the environment. You add it yourself; a key is
  never entered for you.
  - The API is $5 per 1,000 requests, with a $5 monthly credit.
  - Brave's terms on storing results are not stated on its pricing page. The store keeps only the
    extracted facts and their URLs, and raw search responses are cached for a day.
- **The profile path** ("who is X") and the chat path's "I can look it up for you" offer do not use the
  web yet.
- **A web fact that was held at admission and lifted later** is not made durable by the sleep cycle. The
  cycle keeps only facts that were accepted as attested. The web's own tally still carries it.
- **Syndication.** One wire story republished on many sites looks like many witnesses. That is the known
  hole in "independent".
