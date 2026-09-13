# 2026-09-13 — the encyclopedia as a Source: frames over OCR'd entries, cross-checked before they feed anything

Nick: "we could easily get Q/A extracted from these encyclopedia records". The corpus is a 2005 general
encyclopedia in 29 OCR'd text volumes (130 MB, off-repo, ~4.5 MB a volume, ~1,850 lines each: the OCR kept
page breaks and little else). The rule in force since 2026-09-12: outside testing and training no external
LLM is involved, and an LLM's only job is building the dataset. So the first question is not "what can a
model read out of this" but "what can a *frame* read out of this, and how often is it true" — because a fact
that enters the world store is walked and spoken, and the kill line is 0 wrong.

## The source (`standin/encyclopedia.py`)

The shape is `standin/wikitext.py`'s: entries, frames over each entry's opening, `Triple`s with the sentence
they came from as provenance, `relations(text)` naming the source's own relations for lever 4, and — new
today, the narrowing rule's need — `kind(label)` naming each relation's value kind. No model, no network.

An entry opens `GOETHE, gu'ta, Johann Wolfgang von (1749- 1832), the greatest of all German poets …` or
`GUINNESS, Alec, gin'is (1914-2000), British actor …` or `GOSPORT, gos'port, a borough in south central
England, is in Hampshire …`. So: the headword is one to five all-caps words at a line start or after a sentence
end, followed by a comma; the given names are the comma part of the headword line whose tokens are all
capitalised words or particles (the pronunciation is the lowercase part, and comes before or after); the
parenthesis is the birth-death years; a person's name is given names + surname (`Johann Wolfgang von Goethe`,
`Alec Guinness`, `Curtis Guild` with the `Jr.` dropped). A person whose given names cannot be read is nobody:
no name, no facts. A place is its headword.

Frames, persons: `born|bom (in|at PLACE,)? on DATE` → `date of birth`; `born (on DATE,)? in|at PLACE` → `place
of birth`; the same two with `died`. `DATE` is the encyclopedia's `Aug. 28, 1749` / `April 3, 1934`; `PLACE`
is capitalised tokens, `St.`/`Mt.`/`Ft.` the only periods allowed, particles inside (`Pinar del Rio`,
`Frankfurt am Main`), a named kind of place allowed before (`the free city of Frankfurt`), and the *immediate*
place returned (`Vienna`, not `Vienna, Austria`). The headword's years are `date of birth` / `date of death` as
bare years when no full date is read. Frames, places: a container the entry names *as one* — `in Fulton
county`, `is in Shizuoka prefecture`, `is in Hampshire,` — in Wikidata's label form (`Fulton County`); never
the opener's `a city in Japan`, which is a country as often as a region.

## The OCR's hazards, each a rule, each pinned (`standin/tests/test_encyclopedia.py`, 9)

The OCR is rough: `bom` for `born`, `tire` for `the`, `fives` for `lives`, `un- til` at every line break,
captions and contributor bylines run into the text, page running heads run into headwords, and — the one
that matters most — a headword the pattern misses, so the next entry's sentences follow the previous
person's. Volume 13, first pass, showed each of these; the rules:

- **line-break hyphens** are removed (`Lon- don`, `Ethi- opia`, `Gold- schmidt`; `-` + space + lowercase);
  `bom` reads as `born` and nothing else does.
- **a person frame counts only in a sentence about the subject**: one opening with `He`, `She`, `Born`, a
  word of the name, or the headword line's own first token. `His father, Johann Caspar Goethe, a lawyer, was
  born in Frankfurt` states nothing about the son.
- **a subject sentence dated to another year than the headword's** — `He was born in Cumberland county, Pa.,
  on March 5, 1794` after `GRIEG, Nordahl (1902-1943)`, the next entry's headword missed — is a misread or a
  run-in; the source disagrees with itself, and **neither** the sentence's fact nor the headword's year is
  stated. A full date that disagrees with the headword year is the same case.
- **captions and bylines**: `Ulysses Simpson Historical Collection` is not given names (more than three
  non-particle tokens: nameless, factless); `He died in Justus J. Schifferes Coauthor of …` names no place
  (an initial follows); `New York GRISWOLD`, `New HARBIN` name no place (an all-caps token: the next entry's
  headword ran in); `GORDON GORDON` and `GOMBERT-GOMEZ GOMBERT` are `GORDON` and `GOMBERT` (the running
  head); `N. Y. GOTTSCHED` is `GOTTSCHED` (a byline's state); `H. M.` is a cross-reference, not an entry.
- **a place entry is one whose first sentence says so** (`a city …`, `a town …`); a bird `found in Fulton
  county` is located nowhere.

## exp_r16, volume 13 (`validation/exp_r16_encyclopedia.py`, `--wikidata`)

718 entries found, 204 persons (14 unnamed), 653 names indexed, **577 facts** in 16 s: 175 birth dates,
161 death dates, 121 birthplaces, 77 death places, 43 containers. Then the check that decides whether any of
it may feed a dataset — two checks, neither a model:

**Against Wikidata**, where the name resolves to exactly one item (the entity-ambiguity rule of 2026-09-12:
80 resolved, 114 ambiguous and checked against neither item, 36 unresolved; cached, 26 live calls): **years
121 agree, 4 disagree**; places 58 agree, 6 the same name in another spelling (`Beach Island` / `Beech
Island`, `Fredrikshald` / `Frederikshald`, `Newton` / `Newtown`), 15 disagree. The four year disagreements,
read: Richard Gridley `1711` vs `1710-01-03` (Old Style / New Style — January 1710/11, both right);
William Frederick Halsey `1882-1959` vs `1853 / 1920` (Wikidata's search resolved the *father* — the
encyclopedia is right about the son; an entity-resolution miss, not an OCR one); Arthur Garfield Hays
`1885` vs `1881-12-12` — the one genuine disagreement, and an `1`→`5` misread is the likeliest cause. The
fifteen place disagreements, read: thirteen are the same place at another granularity or under another
name (`Gleiwitz` / `Gliwice`, `Pinne` / `Pniewy`, `Scutari` / `Üsküdar`, `Istanbul` / `Constantinople`,
`Pangbourne` vs `Berkshire`, `Westernville` vs `Oneida County`, `London` vs `Clapham`, `Judithenkirch` vs
`Königsberg`, …) and two are genuine (`Marshalltown` vs `Grinnell` for J. B. Grinnell's death; `New York`
vs `Istanbul` for A. D. F. Hamlin's — where Wikidata is the doubtful one). **Hand-read, 50 seeded facts:
49 true as stated, 1 partial** (`Pinar` for `Pinar del Rio` — the particle rule above, from this). Two
malformed names in the sample (`George Gordon Gordon`, `Johann Christoph Y. Gottsched`) were the running
head and the byline rules above, from this.

So the digits are reliable and the frames are precise; what is lost is *reach* — 114 of 230 names are
shared by two or more Wikidata items and cannot be cross-checked (they can still be stated, with their
provenance), and the headword pattern finds 718 entries where the volume has more (a headword after a
byline or a caption, or after a sentence the OCR did not end with a period, is not found; its sentences
then belong to nobody, by the subject rule, rather than to the wrong person).

## All 29 volumes (`_all`)

22,213 entries, 3,601 persons (306 unnamed), 20,104 names indexed, **10,790 facts in 16 s** (the index 8 s,
once): 3,110 birth dates, 2,838 death dates, 2,211 birthplaces, 1,180 death places, 1,451 containers.
Volume 13 was person-heavy; the set as a whole is ~370 facts a volume. Sixty hand-read (seed 0, first
pass): **57 true as stated**, one place cut off at the opening's end (`Castle Altenb`), two malformed
names (`Benito Perez Galdos- Perez Galdos`: a running head with a space after its hyphen; `Ibr'ing
Loring`: the OCR's capital `I` for `l` made a pronunciation look like a given name — an apostrophe inside
a name precedes a capital, inside a pronunciation a lowercase letter). Three more rules, pinned: a match
that ends in the opening's last 40 characters is cut off and not read; `A- A` is `A`; the apostrophe
rule. The Wikidata cross-check over all 3,600 persons is a learning-phase job of a few thousand cached
calls and is not run tonight; volume 13's rates are the estimate.

## What it is for

Two uses, both dataset-phase. (1) **Facts into the world store with provenance**: ~10,800 over the 29
volumes, ~3,600 persons; each walked and verified like any other fact when a question reaches it,
and each with the volume, the headword and the sentence on record. Against Wikidata the encyclopedia adds
little *coverage* (famous people, 2005) but adds a second, independent attestation — a fact both sources
state is doubly attested, and a fact they state differently is a recorded disagreement, which is what the
cross-check produces. (2) **Q/A for gen 3**: a question per fact from a template per relation (`When was
Johann Wolfgang von Goethe born?` → `1749-08-28`), the fact in the store, the chain certified by the VM
before the pair enters the set — the reverse-built dataset agreed on 2026-09-12, with an offline source
behind it. What the frames do not say, the source does not say: the encyclopedia's other relations
(occupation, nationality, works) wait for frames of their own, or for the LLM-proposed-and-certified path.

**As the Source behind search-and-learn** (`exp_r11 --source encyclopedia`, the emitter, same 600
SimpleQA): 530 plans, 79 entities fetched, **0 facts** — none of the 79 seeds the walk stalled on has an
entry; 0 verified, 0 wrong, 0 API calls, 0 VM calls, 408 s (the emitter's). The expected null: SimpleQA
asks about the obscure, a 2005 general encyclopedia holds the famous. Its value is the two uses above,
not this benchmark.

## The Wikidata cross-check over all 29 volumes (`_all_wd`, run 2026-09-13)

The learning-phase job, run overnight and again in the afternoon (the first run died after 38 minutes
with nothing written; 1,729 live calls in all, the rest cached; 719 s the second time): of the 3,601 persons, **1,474
resolve to exactly one Wikidata item**, 2,694 share their name with two or more items and are checked
against neither, 555 resolve to none. On the 3,815 facts the resolved persons state: **agree 3,356, near
66, disagree 393 — 0.880 on the checked part**, and it splits by kind exactly as volume 13 said it would:
**dates 0.952 (birth) and 0.986 (death)**, places 0.754 (birth) and 0.778 (death), the container relation
0.558. The 459 near/disagreements, read: the places are the volume-13 story at scale — the same place at
another granularity (`Brooklyn` / `New York City`, `London` / `Stepney`, `Markham` / `Fauquier County`,
`Corsica` / `Ajaccio`) or under another spelling (`Calcutta` / `Kolkata`, `Kiev` / `Kyiv`, `Cintra` /
`Sintra`, and OCR's `Cower` for `Gower`, `Chilian` for `Chillán`); the containers are the same at one
level up (`Teramo Province` / `Province of Teramo`, `Johore State` / `Muar District`); the dates are
where the two sources genuinely differ, and either may be right — Robert S. Abbott 1868 vs 1870, Fra
Bartolommeo 1475 vs 1472, Basedow 1723 vs 1724 (the encyclopedia's older scholarship), Sanford Bates
died 1972 vs Wikidata's 1982 (Wikidata's the doubtful one), Thomas Bardolf 1368–1408 vs 1145–1190 (a
namesake: the one-item rule cannot see a second item filed under a title). One headword miss surfaced:
`ALEXANDER CITY, al-ig-zan'dar, a trading and industrial city ...` was read under the previous entry
(`Alexander Archipelago`) — a run-in headword whose first word is also a common word; the sentence's
`Tallapoosa county` then disagreed with Alaska. Each `facts` row of `_all_wd.json` now carries its
`check` verdict, so the gen-3 builder can leave the 393 `disagree` facts out of the training records
(3.6% of the 10,790) and keep the recorded disagreement as what it is.

Next: the headword pattern's reach, measured against a hand count on one volume; the builder over
`_all_wd.json` with the disagreements excluded.

`exp_r16_encyclopedia_{v13,all,all_wd}.{log,json}`, `exp_r11_search_learn_wikidata_encyclopedia.{log,json}`.
