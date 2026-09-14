# The news source — what a headline may become, and which feeds actually answer

GrillCheese Research Lab · 2026-09-14 · `standin/news_source.py`, `standin/tests/test_news_source.py`

The wiki world genuinely cannot answer "what happened in 2026?". Measured 2026-09-14: it holds
**0 facts whose subject is a year**, 181 of 552,297 whose object is a bare year, and its 76,897
`timeline event` facts are dangling leaves with no dates at all. The refusal was correct; the
gap is a missing source, not a missing feature.

## The design decision

> **A headline is not a fact about the world. It is a fact about what a publisher reported.**

So a feed item becomes `"<headline>" is the headline of <date>`: the **date is the subject**,
the **publisher is the provenance** (which this architecture already carries per fact), and the
day lands in `times` (which the date index already reads). Nothing asserts that a headline is
*true* — only that it was published, which is the part this source can actually witness.

Two consequences fall out, and both are load-bearing:

- **A headline never becomes a fact about an entity.** "Russia hits Ukrainian train" does not
  enter the store as a fact about Russia. An entity fact needs a source that states it
  structurally under provenance — Wikidata's job, not a newsroom's. This source answers WHEN;
  it refuses WHO.
- **Two feeds printing the same words are one fact, not a corroboration.** Agreement among
  newsrooms is syndication, not evidence, and the store must not let the second copy make the
  first look confirmed. The first feed to say it keeps the attribution.

A source that cannot lie about the world is a source that cannot make the loop lie about it.

## Which feeds answer (measured from a real machine, 2026-09-14)

| feed | result | items | dates held |
|---|---|---|---|
| BBC News `feeds.bbci.co.uk/news/rss.xml` | OK, 29 KB | 37 | 30× 09-13, 4× 09-14, stragglers |
| NYT HomePage `rss.nytimes.com/.../HomePage.xml` | OK, 43 KB | 20 | 14× 09-13, 5× 09-14 |
| Politico `rss.politico.com/politics-news.xml` | OK, 207 KB | 30 | spread over 09-02 → 09-10 |
| AP via `feedx.net/rss/ap.xml` | OK, 405 KB | 10 | all 09-09 — **stale** |
| CBC (`www.cbc.ca/webfeed`, `rss.cbc.ca`, both paths) | **refused** | — | timeout / connection closed |
| AP S3 mirror `associated-press.s3-website-…` | **empty** | 0 | `<items></items>`, 55 bytes |

**CBC is deliberately absent from the default set.** All three of its feed URLs refused us — a
publisher declining to be read by a robot. That is their call and it is not worked around.

**The AP S3 mirror is dead.** The bucket listing is real and names 14 category feeds
(`world-news`, `politics`, `technology`, `science`, `climate-and-environment`, …) — which is
exactly the topic menu the clarify bar wants, written by the *publisher* rather than ranked by
this loop — but every file is an empty `<items/>`. The menu idea survives; it needs a live
backend.

**The scope this establishes: RSS is a 24-to-48-hour window, not an almanac.** Four working
feeds reach back about twelve days at the outside and one is already nine days stale. RSS
answers "what happened today / this week". It does not answer "what happened in 2026".

## What does answer "any date": GDELT, by file, not by query

Both routes were tested from the same machine.

- **The query API is out.** `api.gdeltproject.org/api/v2/doc/doc` returned **HTTP 429 on every
  attempt**, including after a 20-second backoff — the IP is rate-limited, not the request.
- **The file route works and is not rate-limited.** `data.gdeltproject.org/gdeltv2/lastupdate.txt`
  returned the current 15-minute drop instantly (`20260914031500`, live), and
  `masterfilelist.txt` (127 MB) indexes **every 15-minute file back to 2015-02-18**. That is the
  almanac: immutable, dated files addressable by timestamp.

Pulled one drop and read it:

- **`export.CSV` (Events)** — 451 rows × 61 columns, 436 of them dated `20260914`. A `Day`
  field, CAMEO actor and event codes, geography, source URL. It is the closest thing to a dated
  event KG that needs no LLM — but `Actor1Name = FJIJUD`, `EventCode = 173` is machine-readable
  and human-hostile. The CAMEO codebook is a static table, so translating it is a one-time build,
  not a model call.
- **`gkg.csv` (Global Knowledge Graph)** — 449 rows × 27 columns per drop, 1.9 MB. Named persons,
  organizations, locations and themes per dated article, with character offsets: `lloyd geering`,
  `otago university`, `Otago, Otago, New Zealand#NZ#NZF7#-45.25#169.5`. This is the one that fits
  CubbyLLM's shape, because it gives **named entities per dated article**.

The caveat that decides how GKG may be admitted: those names are extracted by GDELT's own NLP,
so they are claims-by-extraction, not statements a publisher made. Provenance has to say so —
`gdelt-gkg` plus the source URL — and they are evidence that an entity was *written about* on a
date, never that a fact about it holds.

## Where this leaves the timeline question

Three tiers, each honest about what it is:

1. **Today and this week** — RSS, four feeds, headlines attributed to publishers. Built.
2. **Any date since 2015-02-18** — GDELT v2 files, by timestamp. Not built; the file route is
   confirmed reachable and the CAMEO codebook is the only missing piece.
3. **Before 2015** — still Wikidata's qualifiers (`P580`/`P582`/`P585`), which the store now
   keeps beside each fact.

And `temporal/historical` in the datasets folder does **not** fill tier 3. `train_augmented.jsonl`
has the right shape (`title`, `year_start`, `year_end`, `region`, `actors`), but
`epub_pdf_consolidated.json` (49,226 records) is OCR'd encyclopedia text with LLM enrichment, and
its first record is an ISBN page at confidence 0.22 carrying invented `precursor_events` dated
1450 and 1800. Model-generated speculation wearing a schema is the worst possible input to a
gate: it looks structured and has no provenance. **An LLM-derived event can be a question to go
verify. It can never be a fact admitted to the store.**
