# 0093 - News and specific events go to the web: searched, read, told as reported, and only what two sites state
status: open (Code B, after 0092, before CubeLang P7 and P8)
written: 2026-11-28 by Cowork · approved: 2026-11-28 by Nick ("we also need a websearch tool invocation when I'm asking about latest news or something specific") · hypothesis: H-F2 (temporal worlds) · builds on: the web source (order 0031's `cubbyllm/sources/web.py`), 0024 (news from feeds), 0092

## Why
Nick's example: "tell me about the million satellites elon musk wants to launch". Today the ask loop reaches the web
only as a fallback for one entity and one relation a walk stalled on (`WebSource.facts`: a triple, held until two
independent sites state it). A question about an event, a plan or this week's news has no such relation, so the loop
plans a profile of the nearest name or refuses. The served brain runs with the web on (Brave) and news off.

- **What it brings CubbyLLM:** questions about what is happening now get an answer, built from what at least two
  independent sites report, dated, and refused when they do not agree.
- **Over a conventional model:** a model's knowledge stops at its training date and it answers news from memory; here
  each claim is read from the web that day, kept only when two sites state it, and the record keeps every page.
- **What it does for you:** ask Cubby about the news or a specific announcement and it tells you what is reported,
  as of when, instead of guessing or refusing.

## Build (behind `CB_WEB_REPORT`, off by default; in `cubbyllm/` with the fetching in `cubby_serve/`)
1. **The route** (the host, before the programs seat): a question goes to the report path when
   - it carries a recency cue (latest, news, recent, today, this week / month / year, just, announced, a date in the
     last 12 months), or
   - it asks to be told about something ("tell me about", "what is going on with", "what's the story with") whose
     subject the stores hold no item for, or whose plan the gate refused with no relation the stores hold.
   Everything else takes today's path; the route's reason goes on the record.
2. **The search:** one search a question through the existing backend (Brave; SearXNG when configured), its query the
   question's content words; recent results asked for when the cue is recency. At most 5 pages read, robots.txt and
   the source's caps and cache as today.
3. **The read, deterministic first:** sentences that name the question's subject (its content words), from the page
   text and the snippets. A claim is a sentence's names, numbers and dates. A claim is **kept** when two independent
   sites (the source's rule: registrable domains, mirrors folded, archives not witnesses) state the same names and
   numbers; a claim on one site stays held and is never spoken. No model reads the pages in this order; the LFM reader
   stays where it is (the walk's fallback).
4. **The answer:** the kept claims go to the talk seat as its facts block, with the date read; the reply says what is
   reported and as of when ("As of November 28, reports say ..."), and the name-and-number guard holds it to the kept
   claims. Sources stay internal (`CLAUDE.md`): the reply names no site; the audit view shows every page and which
   claims each stated. Nothing kept is written to the store: news is read again when asked again (after the cache's
   TTL).
5. **When the sites do not agree, or nothing is kept:** Cubby says what it could not confirm, in its own words, with
   no claim from a single site.
6. **Tests:** the route on planted questions (cue, "tell me about" with and without a stored subject, an evergreen
   question that must not route); a claim on one site held; the same claim on a mirror of the first site held; two
   sites kept; the reply's guard on a planted extra number.

## Acceptance (pre-registered)
- **The read:** 50 news and event questions about the week of the read, written by Cowork before it, each with the
  facts Cowork checked by hand that day (Nick's satellite question among them); and 50 evergreen questions the stores
  answer today, which must not leave their path. Read once through `cli.boot` with the web as served.
- **Gate:**
  - 0 spoken claims that contradict the checked facts, or that no two sites state;
  - at least 50% of the 50 news questions answered with at least one kept claim;
  - the route: at most 1 of the 50 evergreen questions sent to the web, and their answers unchanged;
  - median time at most 8 s (0092's stage times on the record).
- **Kill:** one spoken claim that contradicts the checked facts. The flag stays off, and the report shows the pages.
- **Reported:** searches and pages per question, the Brave calls spent, claims kept and held, and the share of answers
  that said "could not confirm".

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim). `cubbyllm/` opens no
  socket: the search and page fetches come through what `cubby_serve/` hands it.
- **No external LLM on the answer path**, and no search engine's answer box or generated summary is read: links,
  snippets and pages only (the source's rule).
- Brave is metered: one search and at most 5 pages a question; the read's spend is reported.
- The keys stay in their git-ignored files.

## Out of scope
- A model that reads pages for claims (the 450M or the LFM): measured later, against this order's deterministic read.
- The RSS archive of 0024 as a second source of news; writing news into the store; following a story over days.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
