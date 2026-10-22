"""cubbyllm/plugins -- what Cubby can be given without retraining: plugins found on disk (``index``), approved and
checked before any code runs (``guard``), woken by a router (``route``) and then by their own claim on the question
(``check``), loaded and swept by the ``load.Loader``; a fiction world for a world plugin (``world``) and the question
encoder the router reads (``question_encoder``). ``python -m cubbyllm.plugins`` lists, reviews and approves them.
Order 0031 moved them here from standin/; the server's part stays with the server.

Wired: WIRED (the server mounts the loader with ``--plugins``).
"""
__wiring__ = "WIRED"
