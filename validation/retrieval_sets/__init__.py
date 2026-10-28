"""validation/retrieval_sets -- the held-out sets order 0046 measures retrieval on (Mintaka, QALD-9-plus, WebQSP_EL,
GraphQ_EL, SimpleQuestions-Wikidata, ShadowLink, AmbER, CronQuestions), in one schema; the frozen relation lexicon;
the fame set. The raw files and the built rows stay off-repo under ``$CUBBY_DATA_ROOT/retrieval``; only ids, counts,
checksums and the fame set are committed.

Wired: STANDALONE (measurement data builders).
"""
__wiring__ = "STANDALONE"
