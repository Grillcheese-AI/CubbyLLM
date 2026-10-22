"""cubbyllm/speech/passage.py -- say only what the passage says: `grounded(value, passage)`, the rule the readers
(the web's, the books' and the LFM's) apply before a fact they read is kept. A value counts when its words stand in
the passage in order, or, for a date, its year, month and day all do (written as ISO, "7 November 1867" or
"November 7, 1867"). Order 0031 moved it here from standin/lfm_source.py.

Wired: WIRED (the books' reader, the LFM reader).
"""
from __future__ import annotations

import re

from ..reasoning.planner import normalize

__wiring__ = "WIRED"

_MONTHS = {m: i for i, m in enumerate(("january february march april may june july august september october "
                                         "november december").split(), 1)}
_MONTHS.update({m: i for i, m in enumerate(("janvier février mars avril mai juin juillet août septembre octobre "
                                              "novembre décembre").split(), 1)})
_ISO = re.compile(r"^(?P<y>\d{4})(?:-(?P<m>\d{2})(?:-(?P<d>\d{2}))?)?$")


def grounded(value: str, passage: str) -> bool:
    """Does the passage SAY this value? The value's words in order, or -- for an ISO date, which the
    reader writes in the store's form -- the year, the month (by name or number) and the day, all in
    the passage. What the passage does not say, the reader may not write."""
    p = normalize(passage)
    v = normalize(value)
    if v and f" {v} " in f" {p} ":
        return True
    parts = _date_parts(value)
    if parts is None:
        return False
    y, mo, d = parts
    words = set(p.split())
    if y not in words:
        return False
    if mo and not ({k for k, i in _MONTHS.items() if i == mo} & words or str(mo) in words):
        return False
    return not d or str(d) in words or f"{d:02d}" in words


_DMY = re.compile(r"^(?P<d>\d{1,2})(?:er)? (?P<m>[^\W\d_]+) (?P<y>\d{4})$")
_MDY = re.compile(r"^(?P<m>[^\W\d_]+) (?P<d>\d{1,2}),? (?P<y>\d{4})$")


def _date_parts(value: str) -> tuple[str, int | None, int | None] | None:
    """(year, month, day) of a date written as ISO, '7 November 1867' or 'November 7, 1867' -- the
    passage and the reader may spell one date two ways and it is still the date the passage says."""
    v = " ".join((value or "").split())
    m = _ISO.match(v)
    if m:
        return m["y"], int(m["m"]) if m["m"] else None, int(m["d"]) if m["d"] else None
    m = _DMY.match(v) or _MDY.match(v)
    if m and m["m"].lower() in _MONTHS:
        return m["y"], _MONTHS[m["m"].lower()], int(m["d"])
    return None


def split_mood(line: str) -> tuple[str, str]:
    """'(at ease) Pellet 6/12, nice.' -> ('(at ease) ', 'Pellet 6/12, nice.'); no tag -> ('', line)."""
    m = _MOOD_TAG.match(line)
    return (line[:m.end()], line[m.end():]) if m else ("", line)



def _content_words(text: str) -> set[str]:
    """Stems (first five letters) of the words that carry content (>= 5 letters)."""
    return {w[:5] for w in re.findall(r"[a-z\u00e0-\u00ff]{5,}", text.lower())}



_NAME_RE = re.compile(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b")


# the shapes of a COPIED prompt rather than a spoken sentence: the record's
# own header, its field markers, a bullet, a markdown heading
_COPY = re.compile(r"here is what i just perceived|voici ce que je viens de perceptoir|"
                   r"voici ce que je viens de percevoir|^\s*#|\n\s*[-*]\s|"
                   r"\b(about|i am at|how i feel|pellets i can see|goal|tried|hit)\s*:", re.I | re.M)



# "level-1 cell 0-2-0" is 4 tokens and he must be able to say it; beyond that
# he is reciting. Measured on exp_r37's first run, where the copies ran 20+.
MAX_RUN = 6



def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9à-ÿ\-]+", s.lower())



def longest_run(a: str, b: str) -> int:
    """Longest run of consecutive tokens `a` shares with `b`, in order. A
    sentence in his own words reuses the names; it does not reproduce the
    record's word ORDER for long stretches."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0
    best = 0
    prev = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best
