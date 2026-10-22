# 0024 — News from feeds: a dated archive, events by day, and news as the trigger that a fact may have changed
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("yes open it") · hypothesis: H-F2 (temporal worlds) · **with 0023's live "as of"**

## Why
Nick (2026-10-22): "what about getting the news from feeds?" Half of it exists. `standin/news_source.py` (2026-09-14)
reads RSS feeds into dated facts: a headline is a fact about what a publisher reported on a date, never a fact about
the world, the publisher sits in provenance, and two newsrooms agreeing is not corroboration ("it is syndication").
`ask.py` consults it only for a `when` question. What it lacks:
- **No memory.** The feed cache is overwritten every 15 minutes, and it was last filled on 2026-09-14: nothing older
  than a feed's current page can be asked about.
- **English only.** BBC, NYT, Politico and AP's front page through a third-party mirror; CBC refused robots and was
  rightly left alone. No French or Québec source.
- **Since order 0016** a reply may not name the publisher or quote a headline of 8+ words, and the source's own
  example reply ("the BBC's headlines included ...") does both.
- **Nothing connects news to the facts it changes.** "X named CEO" today leaves the worlds saying the old CEO, as of
  2019, until Wikidata catches up.

- **What it brings:** Cubby can say what was reported on a given day, in English or French, and notices when the news
  says a fact it holds may have changed.
- **Advantage over conventional models:** a chat model's knowledge stops at its training cut-off, or it searches and
  repeats whatever it finds. Here every news item is dated and kept, said as "reported", never as settled fact, and a
  reported change makes Cubby check the structured source before it answers.
- **In plain words:** Cubby keeps up with the news day by day, and when the news says something changed, it checks
  before telling you.

## Build
- **The archive** (`standin/news_archive.py`, a CLI with `--once`): poll the feeds, append each item once by its id
  or link (first seen, published date, feed, headline, summary), never overwrite, one file per day. Run hourly by the
  serve stack (Nick's choice, 2026-10-22); `--once` stays for a manual or catch-up poll.
- **The feeds** (a manifest in `standin/data/news_feeds.json`): today's four, plus French and Québec outlets. Verify
  each URL, record its terms, and respect a refusal (a robot block, an error, a terms page that forbids it). Replace
  the third-party AP mirror with an official feed or drop it.
  - **Le Devoir (Nick, 2026-10-22):** the front-page feed `https://www.ledevoir.com/rss/manchettes.xml`, the older
    `http://feeds2.feedburner.com/fluxdudevoir` (Le Devoir's "Fil de presse", current on 2026-10-22; drop it if it
    goes stale, as feedx.net's AP mirror did), and the section feeds listed at `https://www.ledevoir.com/flux-rss` (the page builds its list in the
    browser; read it from Nick's machine).
  - **La Presse (Nick, 2026-10-22):** `https://www.lapresse.ca/manchettes/rss` (front page) and
    `https://www.lapresse.ca/actualites/rss` (news). Cowork's cloud fetcher got a client error from both; from Nick's
    machine with the poller's user agent (`CubbyLLM/0.1`) both answer 200 with current items (2026-10-22: 8 and 10
    items). So they read from the serve stack; record their terms.
  - Checked the same way the same day: Le Devoir's `manchettes.xml` (36 items) and the FeedBurner "Fil de presse"
    (19 items) are both current.
  - Then other Québec and French outlets, each checked the same way.
  - **AI opt-outs are a refusal (Cowork, 2026-10-22; Nick to confirm per publisher).** La Presse, Le Devoir and
    Radio-Canada answer the poller (`CubbyLLM/0.1` is not on their lists), but each one's robots.txt blocks AI agents
    by name: La Presse blocks ClaudeBot, anthropic-ai, CCBot, PerplexityBot and Google-Extended; Le Devoir also blocks
    GPTBot, ChatGPT-User and OAI-SearchBot; Radio-Canada blocks those plus Perplexity-User, cohere-ai and DeepSeek.
    That is a publisher declining AI use, and a name not on the list does not change it, the same rule
    `news_source.py` applies to CBC. So the poller checks each publisher's own site (not the feed's host: the
    FeedBurner copy of Le Devoir carries no robots rule of its own) and **holds** a feed whose publisher blocks AI
    agents: not polled, not stored, until Nick decides for that publisher (a licence, or a written yes from it).
    The English feeds are checked the same way. Report which feeds run and which are held.
  - **Sources that welcome reuse** come first: government news releases under an open licence (the Government of
    Canada's, under the Open Government Licence - Canada; Québec's, once its terms are checked), and feeds whose
    terms allow it.
- **Events by day:** items about the same event (shared names and key words, within a day or two) grouped across
  outlets; who and what read by the frames (Cubby-native), as **reported** claims dated by their items. A group stays
  "reported" however many outlets carry it, per `news_source.py`'s rule.
- **Replies** in Cubby's own words: dated ("on 21 October, reports said ..."), no outlet named, no 8-word run from a
  headline or summary (the 0016 guard). The audit view shows the items.
- **News as a change trigger:** a reported role change about an entity a world holds (named, resigned, elected, died,
  acquired) marks that entity's time-sensitive relations as possibly stale. A present-tense question about them then
  goes through 0023's live "as of"; if the structured source has not caught up, the answer carries both: the fact as of
  its date, and that reports on a later date say it changed.

## Acceptance (pre-registered)
- **Archive:** 7 days unattended: items per feed per day, 0 duplicates, a re-poll inside the window adds 0, size on
  disk. Each feed's terms recorded; every refusal respected.
- **Grouping:** 100 random event groups judged by Cowork and Nick: items about the same event >= 0.90.
- **Bench:** 50 questions about the archive's week ("What happened with X on <date>?", "What was reported about Y
  this week?", EN and FR): **0 wrong spoken**; every answer dated and said as reported; replies naming an outlet 0;
  8-word runs from an item 0.
- **Change trigger:** 10 reported role changes against the worlds: the present-tense answer is never the stale value
  said flat; each carries the change as reported, or a refreshed value with its date.
- Off = unchanged (`news=None`); full `pytest tests` and `pytest standin/tests`.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- Headlines and summaries only: never fetch or store article bodies, never get around a paywall or a robot refusal.
- No external model reads the news: grouping and frames are Cubby's own.
- Feed terms and any licence questions go in the local rights notes, not in this repo.

## Out of scope
- Summarizing articles, opinion pieces as claims (items marked opinion or editorial are kept, never read as events).
- Training on news text.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
