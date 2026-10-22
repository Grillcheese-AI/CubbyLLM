# 0016 — Sources stay internal: replies never name them; an audit view shows the whole chain
status: open
written: 2026-10-22 by Cowork · approved: 2026-10-22 by Nick ("the WHERE it learned it should be for internal purposes
only, we don't want to leak sources to users"; "if we have an audit it's available but not to regular users") ·
**suggested before 0012 and 0007**, which start speaking book- and lookup-sourced facts

## Why
Every new source (books, the hdc world, the lookup) keeps its provenance, and it should. Today some reply paths can
surface it. For example, the self-profile once carried `source: wikidata (152)` into a reply (`standin/ask.py`, the
no-paraphrase note), and the docs promised Cubby "can say where it learned it". Nick's rule: sources are for
Cubby's own records and for an audit, never for regular users. `docs/WORLDS.md` and `CLAUDE.md` now state it.

- **What it brings:** Cubby's sources stay its own, while every answer can still be traced end to end by someone
  allowed to look.
- **Advantage over conventional models:** they cannot trace an answer at all. RAG systems trace it, but show their
  sources to everyone. Here the trace exists and is access-controlled.
- **In plain words:** users get answers, not your sources; you, or an auditor, can always see exactly where an answer
  came from.

## Build
- **An audience on every reply path** (serve, the ask loop, the talk seat, the API):
  - `user` is the **default**. This is Nick's rule, so it is not flag-gated off: the change is to what users see.
  - `audit` is unlocked by a token from a git-ignored config (`CUBBY_AUDIT_TOKEN`).
- **For `user`:**
  - No source names in reply text: Wikidata, Wikipedia, DBpedia, news outlets, sites, domains, URLs, book titles or
    authors, dataset names.
  - No quote of 8 or more consecutive words from a passage.
  - No `source:` or `provenance` lines (the self-profile says "152 facts learned", not where from).
  - Attributed facts hedged: EN "one account says…", FR "selon une source…".
  - The API's JSON for `user` drops the provenance fields (`learned`, the facts' sources, passages).
- **For `audit`:** everything, plus `GET /audit/<answer id>`. It returns the chain: the question's plan, the facts with
  their sources and passages, the gate's and VM's verdicts, the tiers, and the ledger hash. The control panel is
  audit-only.
- **A user's own documents** (things the user gave Cubby) may be cited back to that same user.
- **A leak guard at the mouth:** the last check before speaking scans the reply against the provenance of the facts
  behind it and the store's source names. A hit is rewritten without the name, or refused, never spoken.

## Acceptance (pre-registered)
- **Leak scan, `user` role:** every bench's replies plus 200 live-loop answers (store, books, lookup). **0**
  occurrences of any source name, author, title, URL or domain from the store's provenance and the worlds
  manifest. **0** quotes of 8 or more words from a passage.
- **Audit:** **100%** of spoken answers have a retrievable chain under `audit`. Without the token, every audit
  endpoint refuses (403), and the panel does not load.
- **Behaviour:** answers are otherwise unchanged. Spoken-and-correct and 0 wrong spoken hold on the existing benches.
- Tests per path, and full `pytest tests`.

## Constraints
- Rules in `CLAUDE.md` hold. The token lives in a git-ignored file, never in a command, a log or a commit.

## Out of scope
- User accounts beyond the two audiences. Encryption of the stores at rest.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:**
- **Not verified here:**
- **Found on the way:**
- **Questions:**
