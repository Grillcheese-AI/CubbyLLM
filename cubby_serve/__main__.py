"""`python -m cubby_serve ...`: the server (the same arguments as `python standin/serve_api.py ...`, order 0031).

Wired: WIRED (the server's entry point).
"""
from cubby_serve.api import main

__wiring__ = "WIRED"

if __name__ == "__main__":
    main()
