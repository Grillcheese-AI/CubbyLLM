# Brief: qFHRR, where it is weak and what to do about it (2026-11-12)

Written by Cowork for an OpenRouter panel (Nick: "next problem to tackle and I think it deserves a competent
openrouter panel is the QFHRR weaknesses ... and trade-offs"; and "I am also open to grow the model if needed").
Every fact below comes from GRL's repositories, orders and the qFHRR preprint ("Semantic Block Codes", Zenodo,
November 2026). Where something is not stated here, say you don't know; don't fill it in.

## 1. The task
qFHRR, as used here, is a word table: a sentence encoder distilled into sparse block codes (k = 80 blocks of l = 128
slots), served as a lookup and an IDF-weighted sum, with no model at serve time. It is strong at some jobs (section 3)
and measured weak at three (section 4):
1. entity names;
2. choosing between namesakes (killed as a chooser);
3. relations: it works only when the candidates are an item's own properties, and not in French.

Deliver, in this order:
1. **Diagnosis.** For each weakness, say which cause the numbers in section 4 point to: the bag-of-words encoding,
   the vocabulary built by document frequency, the English teacher, the hash fallback for unseen words, its use as a
   ranker (a score that always picks something), the framing of the task, or something else. Say which causes are
   structural to a static word table and which are fixable inside it.
2. **Fixes.** For each weakness, at most three proposals, ranked by expected gain per cost. For each: what changes;
   which harness and set of section 4 reads it, or what new set it needs; a pass line and a kill line, written now;
   the cost (CPU or GPU hours, memory, serving latency); and which hard rule of section 2 it comes near. A proposal
   that breaks a hard rule is out.
3. **The trade-offs.** A table with one row per job: finding a name; choosing between namesakes; choosing the
   relation; reading French; routing to a world; retrieval over a known corpus; the n-gram memory; reading a fact the
   host placed in the context. For each, which part should do it: the qFHRR table, an exact symbolic index or check,
   loci tokens (section 6), the 450M (or a grown model), or a combination in a stated order. Give latency, memory and
   accuracy where sections 3 to 6 measure them and "unknown" otherwise, and the failure mode (a wrong answer, a
   refusal, or slowness).
4. **Growing the model.** Nick is open to growing it. For which jobs above would a larger model be the lever, and
   for which would it not? What size, what it would have to show, and which cheaper lever should be tried first.
5. **Loci tokens.** For each of the three weaknesses, say whether loci tokens (section 6) help, hurt, or leave it as
   it is, and how qFHRR and loci tokens should divide the work or combine.
6. **What you would not do,** and your open questions.

Keep it under about 5,000 words. Plain, exact prose. Cite the numbers you rely on by section.

## 2. Hard rules (each comes from a written decision; don't trade any away)
- **Only the VM certifies.** The symbolic VM (CubeLang) checks every hop against stored facts. "Certified" means the
  answer follows from the facts and rules used; it never means "correct". Agreement between models or arms never
  certifies; it may only filter.
- **Facts are never model-written.** A model, a table or an embedding may propose a candidate; only a statement in
  the store is a fact. No external model writes a fact or a training record.
- **A refusal is better than a wrong answer.** A bare name that several items bear, asked with no cue, gets no spoken
  pick: an underspecified first mention that gets a spoken pick is a kill.
- **Namesakes are never ranked by popularity** within a kind (sitelinks, degree, page views). A pick rests on
  evidence in the question that holds for one candidate.
- **No neural model on MoWM's routing path.** Routing reads text through the word table. The reasoning path may use
  the 450M.
- **Scores are not thresholds.** A threshold is set on dev, with its data and date, and does not transfer between
  encoders, tables or corpora.
- **Pre-registration.** Lines are written before a run; held-out and test sets are read once per arm.
- **Costs that stand:** the lookup-and-check stage of a question within p95 50 ms on the layered store. The aim for
  the whole system is a frontier model's quality on its tasks at one hundredth of the cost or less.

## 3. What qFHRR is, and where it works (from the preprint)
- **Build:** a teacher (all-MiniLM-L6-v2 in v1; potion-retrieval-32M later; serving uses "v4") embeds each word of a
  vocabulary once, offline: the top 30k to 60k words by document frequency over the pretraining corpus, decoded
  through the BBPE-128k tokenizer, plus domain words. Anisotropy is removed (the mean and the first principal
  component), an orthonormal projection maps to 10,240 dimensions (cosine-exact), and the result is blended 0.5/0.5
  with a deterministic random block code per word (a hash of the word). An unseen word gets its hash code: identity,
  not meaning. Words are lowercase, two letters or more. IDF weights are stored.
- **Serve:** split into words, look each up, sum with IDF weights, normalize. 6.8 to 25 µs for a short text on one CPU
  core, 350 times faster than the teacher on single texts. Dense form 20 KB a document; compact form (the argmax of
  each block) 80 bytes, which is a qFHRR integer-phase code and a legal operand for binding.
- **Measured where it works:**
  - routing: 0.95 of its teacher on a 15-domain screen (0.341 against 0.358, null 0.063); 42 real domains at 97% of
    the teacher's quality (a 60k vocabulary, the best static teacher);
  - open-set routing with bundled centroids and a margin gate: 92.4% precision, 86.9% of never-seen domains sent to
    spawn;
  - retrieval over a known corpus: 1M documents in compact form, hit@1 0.71, hit@20 0.92;
  - inside the verified pipeline (a grammar parses the chain, the table retrieves each hop's fact, the VM
    re-verifies): claimed-answer precision 0.992 at 0.646 coverage, unchanged under 100k distractors; later, with a
    lookup-first walk and a plan checker, 574 certified answers of 800 (0.718) at precision 1.000. The corpus is
    consistently misspelled, so these measure chain-recovery fidelity, not real-world correctness;
  - an n-gram memory addressed by the word codes (a Hamming scan over 10,240 bits): with a paraphrase teacher refined
    by WordNet, a trigram with one word paraphrased reaches its row 74.3% of the time and a different fact 4.0%
    (35 hand pairs; not yet confirmed on a fresh list).
- **Its stated limits:** a bag of words (order is lost; every n-gram and binding variant measured cost more than it
  bought); thresholds don't travel; English only; unseen words get identity, not meaning.

## 4. The three weaknesses, as measured
### 4.1 Entity names
- **Open-web entity search (BEIR dbpedia-entity, 4.64M documents):** nDCG@10 0.040 compact and 0.081 with a dense
  re-rank for v1; 0.046 and 0.096 for v4; potion-8M dense 0.224; a published BM25 of about 0.313. The compact
  shortlist's recall@100 is about 0.14 and caps the pipeline. v4 gained 18% and kept the 2.3-times gap to dense potion.
- **The preprint's reading:** entity names are the tail vocabulary that a word list built by document frequency
  misses.
- **Finding a name is done without qFHRR:** an exact name index (12.4M names of 4.8M Wikidata items: labels, aliases,
  titles, redirects), with a typo index for exact misses, holds the gold under its exact name for 87.4% of Mintaka EN
  mentions and 99.6% of FR. But a fifth to a third of mentions name more than 20 items. Finding gives a set; choosing
  within it is the problem.

### 4.2 Choosing between namesakes (dev, seed 0)
- **The sets:** a "fame" set of 1,910 questions (EN and FR) whose subject name several Wikidata items bear, built by
  name collision. Two lines: **bare names** (no cue in the question) allow 0 spoken picks; **qualified names** (one
  discriminator from one item's own facts, such as its occupation or a place it is in) allow 0 wrong and must resolve
  at least 70%. Also ShadowLink (the gap between its Top and Shadow groups at most 10 points), and Mintaka and
  WebQSP_EL (at most 1 wrong answer in 50 rows).
- **qFHRR as the chooser:** the question's cue words against one prototype per item and language (the item's
  description and values), by cosine, the best item picked past a rank-normalized margin:

  | margin | bare: single picks | qualified: wrong, resolved | Mintaka: wrong / decided | WebQSP_EL: wrong / decided |
  |---|---|---|---|---|
  | 0.25 | 298 | 29, 77.4% | 28 / 212 | 30 / 180 |
  | 0.5 | 73 | 3, 47.4% | 11 / 121 | 10 / 112 |
  | 0.75 | 21 | 0, 12.4% | 8 / 92 | 7 / 84 |
  | 0.9 | 11 | 0, 1.1% | 8 / 83 | 7 / 80 |
  | the earlier tiers | 6 | 2, 0.2% | 25 / 105 | 13 / 87 |

  Killed on the bare line at every margin. At 0.9 the 11 picks came from the question template's relation words
  ("pays", "trouve", "né", "born", "country"). BM25 was killed for the same reason.
- **What replaced it: a span rule and a symbolic check (B1q).** Each word of the question serves one role (the name,
  the relation, or a cue); the relation lexicon's wordings and function words are never cues. Each cue is looked up as
  a name, giving value items; a candidate holds the qualifier if one of its statements has one of them as its value,
  in either direction. One candidate standing is a pick; otherwise the question is undecided. Nothing is ranked.

  | on the fame set | bare: single picks | qualified right | wrong | undecided | refused | resolved |
  |---|---|---|---|---|---|---|
  | B1q | 2 | 692 | 3 | 61 | 154 | 0.764 |
  | qFHRR at 0.75 among the candidates that carry the relation | 35 | 173 | 0 | 737 | 0 | 0.190 |
  | the same at 0.9 | 21 | 23 | 0 | 887 | 0 | 0.025 |

  - B1q's 3 wrong picks: the gold's own "located in" is one level below the discriminator (a Rosebank in Cape Town,
    which is in the Western Cape, while another Rosebank states the Western Cape directly).
  - One of B1q's 2 bare picks: the French past participle "né" was not a relation word in the lexicon, and folded to
    "ne" it named Nebraska.
  - B1q's cost, p50 / p95: 1.4 / 6.2 ms at 2 candidates, 10.2 / 40.6 ms at 6 or more.
  - On Mintaka and WebQSP_EL it halves the earlier wrong rate (2.3% and 2.2%) and still misses the 2% line.
- **French:** the table's teacher is English, so most French words fall back to their hash code and match only as
  written.

### 4.3 Relations
- **The task:** given the item (its words taken out of the question), name the property asked. Read as recall@5;
  the wrong rate is the share of lists without the gold. Gates: 0.85 on SimpleQuestions-Wikidata, 0.80 on QALD-9-plus
  FR.
- **Among every property:** a codebook of 12,684 properties (one code per property and language, 25,368 codes):
  recall@5 0.417 on SimpleQuestions at 56.0% wrong, 0.361 on QALD FR at 63.9%. Not kept.
- **Among the item's own properties** (outgoing and incoming; median 39 candidates on SimpleQuestions and about 175 to
  190 elsewhere; five at random would score 0.149 and 0.04 to 0.08):

  | recall@5 (wrong rate) | SimpleQ EN | QALD EN | QALD FR | Mintaka EN | Mintaka FR |
  |---|---|---|---|---|---|
  | qFHRR, margin 0 | 0.887 (9.0%) | 0.755 (22.5%) | 0.536 (45.8%) | 0.554 (44.6%) | 0.397 (60.3%) |
  | qFHRR, margin 0.25 | 0.573 (3.0%) | 0.400 (9.5%) | 0.349 (15.1%) | 0.223 (14.1%) | 0.223 (27.3%) |
  | a frozen label lexicon (content words only) | 0.563 (4.0%) | 0.360 (17.5%) | 0.217 (15.1%) | 0.331 (28.1%) | 0.190 (21.5%) |
  | the 450M in its programs seat | 0.470 (6.3%) | 0.315 (11.0%) | 0.115 (9.6%) | 0.182 (15.7%) | 0.066 (7.4%) |

  - qFHRR passes SimpleQuestions (0.887; 0.893 on the rows where the item holds the gold) and fails QALD FR (0.536;
    0.578). It was not kept by the rule written before the run (keep only if it adds at least 3 points at no higher
    wrong rate than the lexicon): at margin 0.25 it adds 2.7 points pooled (0.401 against 0.374) at a lower wrong rate
    (11.3% against 14.5%).
  - **The danger is a missing fact.** On 38 rows the item does not carry the gold property today. qFHRR at margin 0
    hands over a near property on 36 of them (16 at 0.25); the lexicon on 6, the 450M on 4, the grammar on 0. The near
    property is on record, so the VM checks it and cannot catch the wrong answer it builds.
- **The asked kind:** keeping only the properties whose values on the item are of the kind the question word asks
  (who, where, when, how many). It helps for "place" (recall 0.77, 6% wrong) and "time" (0.63, no wrong). It drops the
  gold on "who" questions asked from the value's side ("who was born in Paris?": the gold is an incoming property of
  Paris) and on "how many" over item-valued properties; each dropped gold becomes a missing-fact row (qFHRR's
  handovers 83 instead of 36).
- **The 450M with an adapter trained on Wikidata's property labels, EN and FR (163,907 records):** it names a
  property on 0.90 to 0.98 of questions (the serving adapter: 0.21 to 0.53). Among the item's own properties: recall@5
  0.800 on SimpleQuestions and 0.241 on QALD FR; a wrong rate above the lexicon's on every set (pooled 0.286 against
  0.145); 16 missing-fact handovers. Not kept.

## 5. The model and the system
- **Cubby (the 450M) is GRL's own model:** 32 layers of width 1,024, a recurrent mixer with windowed attention (a
  512-token window, 4 heads) every third layer, and a 128,000-token BBPE vocabulary (input embedding and output head
  untied, 131M weights each); about 576M weights in all, 445M read per token. LoRA adapters per seat (talk,
  programs). Served on grilly2, GRL's Vulkan compute framework, on a home GPU; the programs seat takes about 2.4 s to
  write one question's program.
- **The reasoning path:** the programs seat writes a CubeLang program; the host resolves the names with the exact
  index, chooses among namesakes (B1q), and picks the relation (today: a grammar and the lexicon); the VM runs the
  program and certifies the answer, or refuses with a reason.
- **Growing the model is open.** Training runs on rented or Colab cards, usually an A100 80 GB. A LoRA run over about
  216k rows takes a little over an hour there.

## 6. Loci tokens (hypothesis H-K10): Cubby's other code for facts
Added on Nick's note: "I think Loci tokens should probably be mentioned too".
- **What they are:** a fact the host writes into the context is bound to a place at the input. Each piece of the
  object, as the trunk's own token embedding, is bound to an address (binding is a product of unit-modulus spectra,
  the same algebra as qFHRR's). An asking token reads it back by unbinding, through a learned read at the middle
  layer and a copy path into the logits: the method of loci. The host writes the facts; the trunk does not produce
  them from its weights.
- **Measured on the 450M:**
  - facts seen once, more than 800 tokens back, read at 0.946 exact (run 5), by training the copy path's 1,027
    parameters for about 10 minutes, the text unchanged;
  - at equal training tokens from the same start (run 6): 0.804 exact through the place against 0.019 for the same
    text read the ordinary way, with the text mix's held-out loss unharmed (2.2398 against 2.2491).
- **Addresses keyed by the words** (a CPU toy at width 256; not run on the 450M):
  - the key is key(s, r) = u(e(s)) bound with u(e(r)), where u(x) = unitary(W x), W a fixed random orthogonal
    projection, and e the trunk's token embedding; piece i of the object sits at the key bound with a fixed random
    unitary P_i;
  - the asker points at its subject and relation tokens and computes the key from what it points at;
  - subjects never seen in training read 0.854 to 0.899, where hashed addresses read 0.000;
  - the key is symmetric in subject and relation, because binding commutes;
  - the switch between the copy and the trunk answers about 12% of present facts wrong, more than the absent ones.
- **Names as single input units:** binding a name's pieces by position into one unit (the sum of e(piece_i) bound
  with S(i), up to about 55 pieces at width 1,024) was counted on the training mix. Proper names are 5.2% of its
  tokens, so spans are not a lever for training cost; they stay a recall mechanism.
- **As the question loop's reader:** given an entity, a relation and the store's facts about the entity, the trunk
  answers from a scene of those facts or says it does not know. It misses when the relation is one its training
  shard never addressed (the shard's relations are the wiki world's, with Wikidata's labels mapped onto them), when
  the store holds several values (one address holds one value), or when the copy's gate stays shut.
- **Serving:** being ported to grilly2. A linked base exported to grilly2 gives torch's logits within 1e-4; the
  run-6 read at home is next.
- **By construction:** a key comes from the trunk's subword embeddings, so every name has one (no vocabulary miss),
  but it carries identity, not meaning; two items with the same name get the same key; and a relation's key comes
  from its words, so a different wording of the same relation is a different address.
