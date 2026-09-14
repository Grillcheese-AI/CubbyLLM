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
| AP via `feedx.net/rss/ap.xml` | OK, 405 KB | 10 | all 09-09 — **stale, dropped** |
| AP front page via `rss.app/feeds/v1.1/…json` | OK, 36 KB, **JSON Feed** | 25 | 09-13 current |
| CBC (`www.cbc.ca/webfeed/rss/rss-topstories`) | OK, 23 KB — **once the User-Agent was right** | 20 | 04-15 → 09-13 |
| AP S3 mirror `associated-press.s3-website-…` | **empty** | 0 | `<items></items>`, 55 bytes |

**The CBC refusal was mine, not theirs.** All three URLs closed the connection — and then a
browser User-Agent got 24 KB in 0.1 s. It is User-Agent filtering at the edge, and the rule is
narrower than it looks: `CubbyLLM/0.1 (+https://github.com/...)` is refused and
`CubbyLLM/0.1` is served. The block is on UA strings carrying a URL or parentheses, not on bots.
So the source now sends the **bare product token** — honest about who is asking, and it works.
A browser UA also works and is deliberately not used: getting through a publisher's door by
claiming to be Chrome is lying to them about who is asking.

**The AP S3 mirror is dead.** The bucket listing is real and names 14 category feeds
(`world-news`, `politics`, `technology`, `science`, `climate-and-environment`, …) — which is
exactly the topic menu the clarify bar wants, written by the *publisher* rather than ranked by
this loop — but every file is an empty `<items/>`. The menu idea survives; it needs a live
backend.

## A topic label is the feed owner's claim

The rss.app feed above is titled **"Iran war"** and carries "Texas stakes its claim for No. 1 in
AP Top 25" and "A flaw in Georgia's election systems could expose voters' choices". It is in fact
AP's front page (Nick, 2026-09-14; the contents agree). Which settles how a topic feed may be
used:

> **A topic feed is a curated selection, not a filter.** The label is the feed owner's claim
> about what belongs together, so it is recorded as provenance and never as a fact that an item
> *is about* that topic.

Taking the label at face value would be adopting someone else's editorial judgement as a fact —
the same error the neutral-prior competition unanimously refused for sitelinks and PageRank, and
it arrives here wearing a friendlier hat. The feed is keyed by what it *contains*
(`ap-frontpage`), not by what it is called, and it replaces the nine-days-stale feedx mirror.

It also settles the wire format: rss.app serves **JSON Feed 1.1**, not RSS XML. Which format a
publisher chose says nothing about the facts, so neither does the source — XML and JSON arrive at
the same Triples, the same `times`, the same provenance, and an undated item is dropped either
way.

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


## What the publishers actually say about this use

Reading CBC's robots.txt to check the feed path turned up something more important than the path.
Measured 2026-09-14 across every host in Nick's feed list:

| publisher | AI crawlers told `Disallow: /` |
|---|---|
| BBC (`www.bbc.co.uk`) | 15, incl. **anthropic-ai**, claude-web, claudebot |
| NYT (`www.nytimes.com`) | 14, incl. **anthropic-ai**, claude-web, claudebot |
| The Verge | 14, incl. **anthropic-ai**, claude-web, claudebot |
| CBC | 10, incl. **anthropic-ai**, claude-web |
| TechCrunch | 10, incl. **anthropic-ai**, claudebot |
| HackerNoon | 11, incl. **anthropic-ai**, claudebot |
| feedx.net | 8, incl. claudebot |
| Ars Technica | 1 (amazonbot) |
| Global News, WIRED, arXiv, Reddit, ByteByteGo | **none** |
| Politico | no robots.txt served |

Two things are true at once and both need saying:

1. **None of those directives is addressed to CubbyLLM.** We are not GPTBot or claudebot, and
   every one of those hosts allows the feed path under `User-agent: *`. Read literally, the
   feeds are permitted.
2. **The intent is generic and unmistakable.** These publishers are saying they do not want
   their words used to build AI systems, and CubbyLLM is an AI system. A loop whose kill line is
   *0 wrong answers spoken* should not get cute about a "no" it understands perfectly well.

What the architecture can contribute is the honest middle: **provenance already travels with
every fact, so let it carry the publisher's stated position too.** `NewsSource.ai_optout(feed)`
reads robots.txt and returns the AI crawlers that host disallows — and the source does **not**
act on it. Dropping feeds on its own would be deciding a publishing-rights question by side
effect, and hiding it in a config would be worse. It is recorded and surfaced; a person decides.

Three details the measurement forced:

- **Read the front door, not the side door.** `feeds.bbci.co.uk` names no AI crawler while
  `www.bbc.co.uk` names fifteen; `rss.nytimes.com` serves no robots.txt at all while
  `www.nytimes.com` names fourteen. A feed subdomain is delivery infrastructure. Reading only it
  is a way of not hearing the answer.
- **`bbci.co.uk` is not a subdomain of `bbc.co.uk`.** Stripping labels never finds the BBC's
  policy, so the few known delivery-domain → front-door pairs are written down rather than
  guessed at, and a host not in that table is simply not claimed.
- **`[]` and `None` are different answers.** `[]` is "they published a policy and it names none
  of these"; `None` is "there was nothing to read". No robots.txt is not a yes and not a no.

The distinction that most likely decides this in practice is **serving vs training**, because
they are genuinely different uses:

- *Serving* — fetch a headline, speak it once with attribution, retain nothing. That is what a
  feed reader does, and RSS exists to be read by software.
- *Training* — bake headlines into a dataset that becomes model weights. That is what the block
  lists are about, and it is not undone by attribution.

This source does the first and stores headline + date + provenance only, never article bodies.
Whether the second is allowed for opted-out publishers is Nick's call, and the `ai_optout` field
is what makes it enforceable per fact rather than per good intention.

## The wider feed sweep (17 feeds, honest UA)

| working | note |
|---|---|
| CBC top / technology / world | 20 items each, current |
| Global News, Ars Technica, TechCrunch, The Verge, HackerNoon | current, 10–20 items |
| Reddit r/news, r/LocalLLaMA | 25 items; the r/ failures on the first pass were transient |
| ByteByteGo | 20 items but a month-wide span — newsletter cadence, not news |
| WIRED AI tag | current |

| not working | why |
|---|---|
| WIRED "ideas" | **stale by two years** (2024-01-08 → 2024-06-19) |
| feedx AP | nine days stale — already replaced by the AP front page feed |
| arXiv `cs.AI`, `math.QA` | 892 bytes, no items, on both `rss.` and `export.` hosts |
