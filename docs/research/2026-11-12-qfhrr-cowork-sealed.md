# Cowork's own answer to the qFHRR brief, sealed before the panel runs (2026-11-12)

Written and committed before any panel answer is read, so the synthesis can say what the panel added and what it
only repeated. The brief is `2026-11-12-qfhrr-brief.md`; section numbers below are its.

## 1. Diagnosis
- **Entity names: structural for a table distilled from text.** A name is a tail word. It either misses the
  vocabulary and gets a hash code (identity, no meaning), or its teacher vector is noise, because a word encoder
  learns little about a rare proper noun. What a name means lives in the graph (what the item is, where, with whom),
  not in text co-occurrence. A better text teacher does not close this: v4 gained 18% and kept the 2.3-times gap
  (4.1). The table should not be asked to find names; the exact index already does that (4.1).
- **Namesakes: the role, not the codes.** A ranker always ranks. A margin over a bag-of-words cosine cannot tell "no
  evidence" from "weak evidence", and words that are not cues (the template's relation words) move the score (4.2).
  B1q works because it checks one cue against one candidate's facts and refuses when nothing holds. The only place a
  similarity can serve here is inside that check: a soft match between one cue and the label of one value one
  candidate actually has.
- **Relations: crowding, then French, then absence.** Among 12,684 properties a bag of words is too coarse (0.417);
  among the item's own it is the strongest arm in English (0.887). What remains is not the codes: French words are
  hash codes to an English table (0.536), and a ranker cannot say "none of these", so a missing fact becomes a near
  property, which the VM then certifies as a checked fact (36 of 38). The second is the dangerous one.

## 2. Fixes, in the order I would run them
1. **Absence first (relations).** Score the question's relation wording against the whole property codebook and
   against the item's own properties. If the best property overall is one the item does not carry, and it beats the
   item's best by a margin set on dev, refuse: "I don't have that fact." This is the preprint's own open-set rule
   (route or spawn) applied to relations. Line: missing-fact handovers at or below the lexicon's (6 of 38) with recall
   on held rows down by at most 2 points. Kill: more than 16 handovers. Cost: none worth counting (one more cosine
   pass over 12,684 codes).
2. **Two rule changes already written down (relations and namesakes).** The qualifier read along the "located in"
   chain (it makes B1q's 3 wrong picks undecided), and incoming properties kept for "who" asked from the value's side.
   Both are rules, read on test once. Cost: the walk's queries.
3. **French as a table, not a model.** Map French words into the English table's space with one orthogonal map
   fitted on bilingual pairs we already hold: Wikidata's EN and FR property labels and aliases, and the items' EN and
   FR labels, give tens of thousands of pairs. The English table is untouched, so routing does not move. Line: QALD FR
   recall@5 at least 0.80 among the item's own properties with a wrong rate at or below the lexicon's, on test, once.
   Kill: below 0.70 on dev. Cost: CPU minutes; the table grows by the French words (about the same size again).
4. **qFHRR inside B1q, never beside it (namesakes).** Only when the exact check leaves no candidate standing: compare
   each cue with the labels of each candidate's statement values (not a prototype of the whole item), with a floor set
   on Wikidata alias pairs (same item, two names) against different-item pairs. One candidate standing is a pick.
   Line on seed 1: 0 bare single picks, 0 wrong qualified, resolved at least 5 points over B1q. Kill: any bare single
   pick.
5. **The 450M bounded the way qFHRR was.** The Wikidata-label adapter names a property almost always (0.90 to 0.98)
   but picks a wrong one twice as often as the lexicon (4.3). Constrain its relation wording at decode time to the
   labels of the item's own properties (an allowed list, as in order 0068's route 1). The bound that took qFHRR from
   0.417 to 0.887 is free for the 450M too. Line: R3o's gate, and missing-fact handovers no worse than the lexicon's.
   Cost: one decode-time mask; no training.
6. **Entity codes from the graph (research, the longest).** Give each item a code built from its statements: the
   bundle of property bound to value. Learn the relations as phase rotations, which is what qFHRR binding already is:
   RotatE scores a fact as head rotated by relation near tail, with unit phasors; ComplEx's score is an FHRR
   similarity. Quantize the learnt phases to the 128 levels. This is the preprint's open problem 2 (training inside
   the block codes). Its uses are proposals only, never facts: soft evidence for namesakes, the direction of a
   relation. The risk is popularity: such embeddings pull toward well-connected items, which the namesake rule
   forbids, so it is read on the fame set against a degree-matched control before anything else. Line: on seed 1,
   B1q with graph codes beats B1q's resolved share with 0 bare picks and 0 wrong. Cost: my estimate, not measured:
   GPU hours on an A100 for 4.8M items at a few hundred phases; 4.8M items at 80 bytes is 384 MB in compact form.

## 3. The trade-offs, in one line each
- **Finding a name:** the exact index and the typo index. Not qFHRR.
- **Choosing a namesake:** the symbolic check (B1q); qFHRR only as a soft cue-to-value match inside it; later, graph
  codes as proposals. Never a ranker.
- **Choosing the relation:** the item's own properties first (symbolic), then the absence test, then qFHRR or the
  constrained 450M within them.
- **French:** an aligned French table for the fast path; the 450M where word order or direction matters.
- **Routing, retrieval over a known corpus, the n-gram memory:** qFHRR, as measured (3). Nothing here argues to move
  them.

## 4. Growing the model
Not the lever for names or namesakes: what decides them is in the store, and a larger model would know more names
only by memorising facts, which it may not speak. It may become the lever for reading the question (relation,
direction, kind, French) once the cheap bounds above are in place. The test that would justify it: the constrained,
Wikidata-trained 450M still below the gate on QALD FR by a wide margin, read against a grown model on the same rows
with the same adapter data. Order 0045's growth path exists for that comparison.

## 5. What I would not do
- Retrain the text table with a bigger teacher to fix names (4.1 says the gap is structural).
- Set any qFHRR margin as a chooser on its own again (4.2).
- Let a similarity, or a graph embedding, ever stand in for a stored fact.

## 6. Loci tokens (added before the run, when Nick asked for them in the brief)
- **Names:** loci keys cover what qFHRR cannot. A key comes from the name's own subword pieces, so no name is out of
  vocabulary. But a key is identity, not meaning, and two namesakes share it. So loci tokens address a name the host
  has already resolved; they do not choose between namesakes. B1q stays the chooser.
- **Relations:** the loci key from the relation's words breaks on a paraphrase by construction, and that is exactly
  where qFHRR is measured to be tolerant (the n-gram memory: 74.3% of paraphrases reach their row against 4.0% of
  different facts). Proposal: the host resolves the asked relation to a property first (the item's own properties,
  the absence test, then qFHRR or the bounded 450M), and the loci key binds that property's fixed code, not the
  question's words. The loci path's "relation never addressed" miss then becomes the relation choice we already
  measure. Read on the free-text toy with held-out wordings first, then on the 450M.
- **Namesakes inside one context:** one address holds one value, so two namesakes in the same scene would collide.
  Once the host has chosen, the key should bind the item's id code, not its name.
- **The division of labour:** qFHRR ranks only among candidates the store bounds (routing, the relation among an
  item's own properties); the symbolic check chooses the entity; loci tokens carry the chosen facts into the trunk's
  context by address; the VM certifies. Neither qFHRR nor loci tokens chooses an entity.
