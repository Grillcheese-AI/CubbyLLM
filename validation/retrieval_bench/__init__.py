"""validation/retrieval_bench -- order 0046's harness: the local name index (exact, normalized, trigram codes, BM25 as a
supplement), the mentions located for the sets that do not mark them, and the experiments R0 to R8, each writing
``validation/logs/retrieval_R<n>_<date>.json`` with ids and counts only.

Wired: STANDALONE (measurement tooling; nothing in ``cubbyllm/`` imports it).
"""
__wiring__ = "STANDALONE"
