# Research brief: a holographic memory read that works for a closed set of names and fails for open-vocabulary facts

*Written 2026-10-21 for a wide literature search (Perplexity). Numbers: `CUBBYLLM_HYPOTHESES.md` H-K10 / H-S12, `validation/logs/*run3*`.*

**Setup.** A 450M-parameter language model: 32 layers, d = 1024, mostly MinGRU recurrent layers with
sliding-window attention (window 512) every third layer, a 128k byte-level BPE vocabulary and an
untied output head. We added an associative memory to the pretrained model. A token that carries an
address `p` is *written* by binding its input embedding to a unit-modulus code of the address
(circular convolution with a fractional-power / spatial-semantic-pointer encoding `S(p)`, as in
holographic reduced representations); the memory is the running sum of the bound vectors, reset per
document. At layer 16, a token carrying an address *reads*: `r = memory ⊛ S(p)⁻¹`, which is ≈ the
embedding of the token stored at `p` plus crosstalk from the other items (~24 items in 1024 dims).
`r` goes through a learned gate and a projection initialised to zero, into the residual stream.
Training is next-token loss, plus an auxiliary loss aligning a learned query with `S(p)`.

**What works.**
- A 3.4M model (d = 256) trained from scratch with the memory: facts whose answer is one token out of
  509 names, asked 120+ tokens past the attention window, answered at 0.94 exact.
- The 450M with the memory added after pretraining, on "villages": answers are names from a closed set
  of a few dozen, answered at 0.95 exact.

**What fails.** The same 450M on facts (subject, relation, object) from Wikidata. The object is a
1–5 piece sub-word name from the 128k vocabulary, written at address(subject, relation) with an offset
per piece. About 800 tokens of unrelated text separate the facts from the questions, and each answer
token's predecessor queries the address of the piece it must predict. Exact answers: 0.02. Per answer
token: 0.309 with the memory read vs 0.300 without, so the read contributes almost nothing. Fact
documents are 4.6% of the training mix and answer tokens ~0.1% of all tokens; 0.45B tokens of
continued training.

**Our candidate causes:**
1. Decoding an arbitrary retrieved embedding back into the right token through 16 layers and an untied
   128k head, learned for a closed set but not for an open vocabulary.
2. Too little supervision.
3. Adding the module to an already-trained trunk.
4. Multi-token objects.
5. Crosstalk.

**Please research, with citations (paper, year, link) and the evidence behind each claim:**
1. How memory-augmented LMs get retrieved content into the output distribution. Covers kNN-LM
   interpolation, pointer-generator and copy mechanisms, Memorizing Transformers, RETRO, LongMem,
   Larimar, MemoryLLM, fast weights and others. Include where they inject (mid vs late layers vs logits)
   and how they initialise and gate it.
2. Decoding retrieved embeddings from superposition memories (HRR / FHRR / VSA) into tokens: cleanup
   memories, scoring against the embedding table directly, tied vs untied embeddings, and the
   logit-lens view.
3. Training-signal sparsity for memory modules: auxiliary losses that supervise the read directly
   (e.g. the read vector matching the target embedding), dense synthetic recall tasks (associative
   recall, MQAR), curricula, answer-token weighting, and the data scale needed at 100M–1B parameters.
4. Recalling multi-token entities from one address: per-piece addresses vs one entity vector decoded
   into a sequence.
5. Adding new modules to pretrained models: zero-init vs warm start, freezing the trunk first, learning
   rates for the new parameters, and known failure modes.
6. HRR/FHRR capacity and crosstalk at ~24 items in 1024 dimensions, and results on rare or open
   vocabulary retrieval.

**Wanted output:** a ranked list of concrete interventions, each with:
- the expected effect;
- its cost;
- the evidence and citations;
- a minimal experiment that tests it on a CPU toy (d = 256, a few thousand steps).

Mark which claims are measured in a paper and which are conjecture.
