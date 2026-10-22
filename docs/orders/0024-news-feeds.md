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
  serve stack, or by a Windows scheduled task Nick sets up (`--once`).
- **The feeds** (a manifest in `standin/data/news_feeds.json`): today's four, plus French and Québec outlets. Verify
  each URL, record its terms, and respect a refusal (a robot block, an error, a terms page that forbids it). Replace
  the third-party AP mirror with an official feed or drop it.
  - **Le Devoir (Nick, 2026-10-22):** the front-page feed `https://www.ledevoir.com/rss/manchettes.xml`, the older
    `http://feeds2.feedburner.com/fluxdudevoir` (check it is still updated: a stale mirror is dropped, as feedx.net's
    AP was), and the section feeds listed at `https://www.ledevoir.com/flux-rss` (the page builds its list in the
    browser; read it from Nick's machine). Then other Québec and French outlets, each checked the same way.
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
