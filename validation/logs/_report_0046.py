"""Scratch (untracked): order 0046's report after R0 and R1, status review."""
import io

REPORT = '''## Report (R0 and R1; Claude Code, 2026-10-28)
- **Commits:** `c5e59ee` (the sets, the frozen lexicon, the fame set, the index, the harness); this one (R0, R1, the report).
- **Files:** `validation/retrieval_sets/` (download, the builders, `common`, `lexicon`, `fame_set`, `frozen.json`,
  `fame_set.jsonl`); `validation/retrieval_bench/` (`name_index`, `mentions`, `stores`, `r0_denominator`,
  `r1_finding`); `cubby_serve/sources/http.py` (`download`); tests `tests/validation/test_retrieval_bench.py` (8),
  `test_fame_set.py` (5).
- **Tests:** 1147 passed, 1 failed (`test_every_module_declares_wiring`, order 0002's known red).

### The sets (build items 1-3)
- **Downloaded** (366 MB, every file under the 1 GB line; `validation/logs/retrieval_sets_2026-10-27.json` has the
  counts). Licences: Mintaka CC BY 4.0; QALD-9-plus CC BY 4.0; WebQSP_EL and GraphQ_EL released with BLINK/ELQ (MIT;
  the questions are WebQSP's and GraphQuestions'); SimpleQuestions-Wikidata: no licence stated in the repository
  (SimpleQuestions itself CC BY 3.0); ShadowLink: none stated on its card; AmbER CC0; CronQuestions MIT; Wikidata5M
  CC0. All stay off-repo; logs hold ids and counts only.
- **Splits:** each set's own dev and test (QALD: train as dev; ShadowLink and AmbER: halves by seed 0, AmbER by
  name). ShadowLink's Neutral subset names no entity and is left out; 2,702 of its 2,712 titles map to an item
  (Wikipedia's API: Wikidata's title lookup answers 503 for any redirect title).
- **The relation lexicon** (34,920 EN and 22,059 FR wordings, 12,684 properties) frozen before any held-out row was
  read: SHA-256 `fffbd704...ba6e2` in `frozen.json`.
- **The fame set** (`fame_set.jsonl`, SHA-256 `c0525052...cc7e`): 500 names drawn by seed 0 from 34,773 eligible;
  1,910 rows (500 bare EN, 500 bare FR, 466 qualified EN, 444 qualified FR); relations country 222, place of birth
  142, occupation 117, place of death 19. Date of birth cannot be used: Wikidata5M holds item-to-item facts only.
  Labels are Wikidata's English ones as the hdc world holds them: Wikidata5M's entity file lists names unordered
  (Titian's first is "ditiano") and gives no label.
  - **Spot checks by hand, 50 rows each:** first, 45 hold by their words, 3 ambiguous (Rye High School in Rye; Andrea
    Maffei "(Italie)", both men Italian; Reid Avenue in Brooklyn), 2 doubtful; second, after two rules, 43 hold, 7
    ambiguous (missing citizenship; "Great Britain" against the UK; an occupation Wikidata5M lacks); third, fresh,
    after two more rules: **50 of 50 hold.** The rules (in the module): a label shared or held whole by another's
    value tells nothing apart; a place another item is not ruled out of (its places at or above it, or a homonym
    place holding it) tells nothing apart; nothing recorded rules nothing out; a person asked an occupation is told
    apart by birthplace, not citizenship. One residual class stays: an occupation the 2019 data lacks for the other
    namesake (Howard Storm, "the writer"); no rule over this data can see it.

### R0, the denominator (`retrieval_R0_2026-10-28.json`)
- **Pipeline:** the ask loop as the stack mounts it (wiki + hdc worlds, live Wikidata source, lexicon, everyday table,
  tense on, today's defaults). **No model in the programs seat:** the host's own reader only, so "reading" is the
  table's reach, not the 450M's.
- **2,650 dev questions** (300 a set and language; QALD FR 250), first stage that stopped each:

  | | EN (2,100) | FR (550) |
  |---|---|---|
  | reading | 1,739 | 533 |
  | no candidate | 214 | 9 |
  | several | 48 | 4 |
  | wrong candidate | 5 | 1 |
  | relation | 5 | 0 |
  | several values | 9 | 0 |
  | value missing / time / other | 15 | 0 |
  | spoken | 65 | 3 |

- **Every spoken answer read by hand (68):** 62 right (one, "How tall is Big Ben?", said "96" without its unit),
  **6 wrong**:
  - **5 from one defect in today's serving path:** the everyday table's restatement drops a leading "The" from the
    name -- `EverydayForms().restate("Who wrote The Last Dragon?")` gives the seed "Last Dragon" -- and the source's
    one-exact-hit tier then resolves a different item labelled exactly that: Freud's *The Interpretation of Dreams*
    (asked in EN and FR) became Antiphon's *Interpretation of Dreams*; *The Ides of March* (Wilder) became
    Manfredi's novel; *The Last Dragon* (De Mari) became McDermott's; "the skin" (the film) became the singer Skin.
  - **1 not what was asked:** "In which city did John F. Kennedy die?" was said "Parkland Memorial Hospital" (his
    place of death, a hospital, not a city).
  Order 0046 changes nothing on the serving path, so neither is fixed here; both need an order (question 3).
- **The links:** the local Wikipedia text keeps **no links and no redirects** (`wikipedia_dedup`: titles and plain
  text, 74,567 articles; `wiki_full`, 14 GB: plain text, no titles). Decision 5 applies.
- **The dump line** (decision 4's recommended line, written into the script before counting; Nick's decision is
  still open): 3,347 gold mentions in the sample (over the 1,000 minimum); 312 sit in route-labelled seed misses (no
  candidate, several, wrong candidate); **17 of them (5.5%) are items absent from every local store**, against the
  20% line. **Under the line:** the full dump would not fix today's seed misses. Of the 1,975 distinct gold items,
  Wikidata5M holds 1,805, DBpedia's titles 1,631, the routed Wikipedia's titles 29.

### R1, finding (`retrieval_R1_dev_2026-10-28.json`, `retrieval_R1_test_2026-10-28.json`)
- **Arms:** Wikidata's search (its own order, up to 100); the local index (`name_index`: 12.4M names of 4.8M items,
  Wikidata5M's names and the hdc world's EN/FR labels) exact, + normalized, + trigram (FTS5 blocking, reordered by
  the capsules' code), + BM25 as a supplement only. A set or a run of equal scores enters a shortlist whole or not
  at all (no tie broken). Mentions: the sets' own where marked, else the gold item's names located in the question
  (Mintaka FR, QALD, SimpleQuestions, CronQuestions). Samples: 1,000 Mintaka questions and 500 of every other set per
  split and language (seed 0); 6,373 dev mentions and 6,081 test, 20% with a typo variant.
- **recall@20, test** (dev within 3 points everywhere, in the dev log):

  | set | API | best local | items the index holds: API / local |
  |---|---|---|---|
  | Mintaka EN | 0.862 | **0.864** | 0.867 / 0.921 |
  | Mintaka FR | 0.971 | 0.738 | 0.980 / 0.787 |
  | WebQSP_EL | 0.884 | **0.925** | 0.885 / 0.939 |
  | GraphQ_EL | 0.820 | **0.852** | 0.860 / 0.925 |
  | ShadowLink | 0.673 | **0.707** | 0.656 / 0.810 |
  | AmbER | 0.950 | 0.874 | 0.950 / 0.912 |
  | QALD EN / FR | 1.000 / 1.000 | 0.885 / 0.529 | |
  | SimpleQuestions | 0.974 | 0.919 | |
  | CronQuestions | 0.996 | 0.745 | |

- **Against the lines:**
  - best local recall@20 >= 0.90 on Mintaka EN: **fail** (0.864 test, 0.877 dev);
  - >= 0.85 on Mintaka FR: **fail** (0.738, 0.759);
  - above the API on every set: **fail** (above on Mintaka EN, WebQSP_EL, GraphQ_EL, ShadowLink; below on the rest);
  - the trigram arm adds >= 5 points on the typo set: **pass**, by 33 to 76 points (the API finds 0 to 5% of typos);
  - BM25 as a supplement adds nothing at @20 on any set (it equals the trigram arm everywhere);
  - **kill** ("no local arm beats the API at @20 on Mintaka"): read per language, not killed (EN 0.864 against
    0.862); read on Mintaka as a whole, killed (the FR gap outweighs). Cowork's call (question 1).
- **What the misses are** (Mintaka dev): EN, 179 local misses: 77 items the 2019 index does not hold, 63 near but
  ranked past 20, 21 with no name near, 18 in a shared-name set over 20; FR, 267: **130 items the index holds under
  no French name** (the index has French labels for 1.2M of its 4.8M items), 68 near past 20, 54 not held. The two
  arms are complementary (EN: 140 found only locally, 124 only by the API).
- **A bias the reader should weigh:** where a set gives no mention words, they were located with the gold item's
  current Wikidata labels, which are exactly what the API searches; that favours the API on Mintaka FR, QALD,
  SimpleQuestions and CronQuestions. On the sets that mark their own mentions (Mintaka EN, WebQSP_EL, GraphQ_EL,
  ShadowLink, AmbER) the local index is above the API on four of five.

- **Not verified here:** the 450M in the programs seat (R0 runs the host's reader only); the fame set's 444 French
  qualified rows beyond the 50 checked (their French was read by hand only in those).
- **Questions:**
  1. **R1's kill line:** per language (not killed) or Mintaka as a whole (killed)? Either way the gate is not met; the
     measured lever is French names (130 of 267 French misses) and coverage past 2019 (77 of 179 English misses).
  2. **Decisions 1-4** are still open as far as the repo says. R0 is under the recommended dump line (5.5% against
     20%). **Decision 5 applies:** the local Wikipedia text kept no links or redirects; the redirect tables are a few
     GB, over the 1 GB line, so they wait for Nick's yes.
  3. **Today's pipeline speaks wrong answers on held-out names:** 5 of 68 from the dropped "The", 1 class mismatch.
     A fix order (keep the article for the seed; a "which city" answer checked against its class) is outside 0046.
'''

p = "docs/orders/0046-retrieval-experiments.md"
s = io.open(p, encoding="utf-8").read()
start = s.index("## Report")
s = s[:start] + REPORT
s = s.replace("status: in progress (Claude Code, 2026-10-28)", "status: review (R0 and R1 done; Claude Code, 2026-10-28)")
io.open(p, "w", encoding="utf-8").write(s)

p = "CLAUDE.md"
s = io.open(p, encoding="utf-8").read()
s = s.replace("built, under its line), 0044; open, in this order: 0046 (retrieval experiments R0-R8, offline; review after R1), 0032 (the trunk",
              "built, under its line), 0044; review: 0046 (R0 and R1 measured; R2 waits on Cowork's reading); open, in this order: 0032 (the trunk")
io.open(p, "w", encoding="utf-8").write(s)

p = "docs/inbox.local.md"
s = io.open(p, encoding="utf-8").read()
note = ("\n  - `Claude Code:` 2026-10-28 -- 0046 in review after R0 and R1 (the report is in the order). R1's gate is not met"
        " (the French names are the gap); R0 is under the dump line (5.5%). **Today's pipeline spoke 6 wrong answers in 2,650"
        " held-out questions:** 5 because the everyday table drops a leading \"The\" from the name, 1 hospital for \"which city\"."
        " Nothing on the serving path changed (0046's constraint); a fix needs an order. Decisions 1-4 still open; decision 5"
        " applies (no links in the local Wikipedia text).\n")
if "0046 in review after R0 and R1" not in s:
    s += note
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
