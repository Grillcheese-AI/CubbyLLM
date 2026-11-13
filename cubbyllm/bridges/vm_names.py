"""cubbyllm/bridges/vm_names.py -- the VM's 16-bit name keys, checked before a program runs (order 0067, step 2).

The VM build keys every identifier **and every string literal** by the first two bytes of its BLAKE3 hash (cubelang
`compiler.rs` `name_hash`; `engine.rs` `name_table: HashMap<[u8; 2], String>`). Two different strings with the same
key share one entry: they read as the same spelling, so they compare equal, and two variables with the same key share
one register. 65,536 keys: a program with a few hundred distinct strings meets a shared key often (121 of the first
500 programs of 0067's step-2 build carried one). Found by step 2's keep rule: ``"thing" == "government"`` holds in
the VM (both key ``15d0``). For cubelang's fix list (with P6); until it is fixed, a program whose strings and names
share a key is not run: its rule is "asked", never checked.

`collisions(source)` -> [[the strings sharing a key], ...] among the source's string literals and identifiers (keywords
left out); empty when every string has its own key. Fails closed: without the `blake3` module every program reports
a collision (so nothing is checked on a guess).

Wiring: STANDALONE (`cubbyllm/speech/rule_plan.run` and 0067's step-2 build call it; the serve path's other VM programs
do not yet: proposed in 0067).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

__wiring__ = "STANDALONE"

_STR = re.compile(r'"(?:[^"\\]|\\.)*"')
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
KEYWORDS = frozenset(
    "program implements public private function return create assign add sub mul div if else while for let of in "
    "number str bool true false null mention solve ISolve ISolver len break match use pure storage mutable".split())


def key(s: str) -> bytes:
    import blake3
    return blake3.blake3(s.encode("utf-8")).digest()[:2]


def names(source: str) -> set[str]:
    """The string literals (unescaped) and identifiers a CubeLang source carries."""
    out = set()
    for lit in _STR.findall(source):
        try:
            out.add(json.loads(lit))
        except ValueError:
            out.add(lit[1:-1])
    out |= {w for w in _IDENT.findall(_STR.sub(" ", source)) if w not in KEYWORDS}
    return out


def collisions(source: str) -> list[list[str]]:
    try:
        import blake3  # noqa: F401
    except ImportError:
        return [["(the blake3 module is missing: names cannot be checked)"]]
    by = defaultdict(set)
    for s in names(source):
        by[key(s)].add(s)
    return [sorted(v) for v in by.values() if len(v) > 1]


__all__ = ["KEYWORDS", "collisions", "key", "names"]
