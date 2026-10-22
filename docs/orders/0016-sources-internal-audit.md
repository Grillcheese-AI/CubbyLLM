# 0016 — Sources stay internal: replies never name them; an audit view shows the whole chain
status: review
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
  - No run of 8 or more consecutive words from **any** stored passage. Scan against the passage index, not only
    the facts behind the answer. Cubby paraphrases in its own words (Nick: like a student who learned in class).
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
- **Commits:** `6763f1f` (the audience, the token, the guard at every mouth, `/audit/<id>`), `a95868d` (the scan
  counts a profile as spoken, by sleep's rule, and asks until 200 answers), and this report's commit.
- **Files:**
  - `standin/audience.py` (new): `audience`, `token_of`, `for_user`, `LeakGuard`, `mouth`, `audit_chain`, `record_hash`.
  - `standin/ask.py`: every kept record goes through the mouth and is sealed with its hash.
  - `standin/serve_api.py`: the panel and the event streams are 403 for a user; `/audit/<id>` is under the token;
    `/ask` and `/turn` return `for_user`.
  - `standin/serve.py`: the talk seat's reply goes through the guard.
  - `standin/self_source.py`: the self-profile counts what it learned, never from whom.
  - `validation/exp_leak_scan.py`.
  - Tests: `standin/tests/test_audience.py` (5, the HTTP surface included); `test_self_source.py` and
    `test_serve.py` updated.
  - The token is read only from `CUBBY_AUDIT_TOKEN` or the git-ignored `standin/data/out/audit_token`; with
    neither, there is no audit audience at all. `user` is the default and not behind a flag (Nick's rule).
    `ask.py`, `serve_api.py` and `serve.py` were already over 300 lines and grew by 5 / 23 / 7 lines (reasons in
    `6763f1f`).
- **Tests:** `pytest tests`: 428 passed, 5 skipped, 1 failed (the known wiring guard, order 0002). `pytest
  standin/tests`: 613 passed, 1 skipped. The 25 failures and 1 collection error are all in `test_build_chat_sft`,
  `test_chat` and `test_goemotions_cube`; they need GOEMOTIONS / MULTI_PROMPT, missing at HEAD before this order.
- **Measured** ([log](../../validation/logs/exp_leak_scan_v2_2026-10-22.json), `user` role, the books on, the
  offline Wikidata cache, the CubeLang VM):
  - **Leak scan: passes.** 525 asked, 200 spoken: books 61 (the books bench's questions, `exp_books_bench_v4`),
    store 88, profiles 37, lookup 14.
    - Scanned against 182 source names (every witness the store's provenance names after the run, every cited
      book's author and title, the source list).
    - Replies with a source name **0**, with a URL or domain **0**, with an 8-word run any passage holds **0**.
    - User JSON with a provenance field **0**.
    - The books bench's own replies naming a source: 0 of 31 spoken.
  - **Audit: passes.** 200/200 spoken answers have a sealed chain under `audit`. Without the token every audit
    endpoint and the panel return 403 (test, HTTP).
  - **Behaviour:** the guard changed **0** replies in the scan, so answers are otherwise as they were. 0 wrong
    spoken does not hold on the books bench, for reasons of the books' own, not this order's: 2 flat as scored,
    both read in order 0012's report.
- **Not verified here:** the live serve stack with a browser (the panel's 403 and the audit page's token shim
  are pinned by HTTP tests, not driven in a browser). The scan's earlier run (138 of 359 spoken) counted a profile
  as unspoken; the rule now follows sleep's.
- **Found on the way:**
  - The self-profile said whom it had learned from ("fact learned: 152; source: wikidata (152)"); it now says only
    the count.
  - The help line named the mounted worlds; it now counts facts.
  - An EventSource cannot set a header, so the audit page passes the token as `?token=`; the access log masks it.
- **Questions:**
  1. The audit token is one shared secret. Should owner and auditor get separate tokens, so the access log tells
     them apart?
  2. A profile line whose only content is its source is dropped whole, and a profile left empty is refused
     (`leak_guard`). Is that refusal the wanted reply, or should it fall back to the "I don't know" line?
