"""validation/fact_store/audit_values.py -- order 0055 step 0: does a Wikidata statement's value equal a gold answer?

A gold is an item (its Q-id, given by the set or found by its exact name) or text. Text is read as a date (ISO, "14
December 1972", "December 14, 1972", "décembre 1972", a year), as numbers, and as a string. A statement's value
matches a gold:
  item      the same Q-id
  time      the same year, month and day as far as the gold states them, the value at least that precise ("exact");
            the value agreeing but less precise than the gold is "coarser" (counted apart, never as found)
  quantity  the amount equals one of the gold's numbers, exactly (decimals compared as decimals, no float)
  string    the same words once folded, or the value's words making up at least half of the gold's (4+ letters)
No similarity threshold, no fame: a value either states the gold or does not.

Wired: STANDALONE (order 0055's audit).
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

__wiring__ = "STANDALONE"

_EN = "january february march april may june july august september october november december".split()
MONTHS = {m: i + 1 for i, m in enumerate(_EN)}
MONTHS.update({m[:3]: i + 1 for i, m in enumerate(_EN)})                         # English abbreviations only
MONTHS.update({m: i + 1 for i, m in enumerate("janvier fevrier mars avril mai juin juillet aout septembre octobre "
                                               "novembre decembre".split())})
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", s))


def dates(text: str) -> list[tuple[int, int | None, int | None]]:
    """The dates a gold states: (year, month or None, day or None); BC years negative."""
    t, out = fold(text), []
    for y, m, d in re.findall(r"\b(\d{3,4}) (\d{1,2}) (\d{1,2})\b", t):            # ISO, folded
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
            out.append((int(y), int(m), int(d)))
    for d, m, y in re.findall(rf"\b(\d{{1,2}}) (?:of )?({_MON})\w* (\d{{3,4}})\b", t):
        out.append((int(y), MONTHS[m], int(d)))
    for m, d, y in re.findall(rf"\b({_MON})\w* (\d{{1,2}}) (\d{{3,4}})\b", t):
        out.append((int(y), MONTHS[m], int(d)))
    for m, y in re.findall(rf"\b({_MON})\w* (\d{{3,4}})\b", t):
        if not any(o[0] == int(y) and o[1] == MONTHS[m] for o in out):
            out.append((int(y), MONTHS[m], None))
    if not out:
        for y, era in re.findall(r"\b(\d{1,4}) (bc|bce|av j c|ad|ce)\b", t):           # a short year needs its era
            out.append((-int(y) if era in ("bc", "bce", "av j c") else int(y), None, None))
        for y in re.findall(r"\b(\d{3,4})\b", t):
            if not any(abs(o[0]) == int(y) for o in out):
                out.append((int(y), None, None))
    return out


def numbers(text: str) -> list[Decimal]:
    out = []
    for n in _NUM.findall(str(text or "")):
        try:
            out.append(Decimal(n.replace(",", "")))
        except InvalidOperation:
            pass
    return out


def _time(v: dict, golds_dates) -> str | None:
    m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", str(v.get("time", "")))
    if not m:
        return None
    sign, y, mo, d = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    y = -y if sign == "-" else y
    prec = int(v.get("precision", 9))
    vparts = [y, mo if prec >= 10 else None, d if prec >= 11 else None]
    best = None
    for g in golds_dates:
        if g[0] != y:
            continue
        need = [p for p in g[1:] if p is not None]
        have = [p for p in vparts[1:] if p is not None]
        if have[:len(need)] == need[:len(have)]:
            if len(have) >= len(need):
                return "exact"
            best = "coarser"
    return best


def _words(a: str, gold: str) -> bool:
    fa, fg = fold(a), fold(gold)
    if not fa or not fg:
        return False
    if fa == fg:
        return True
    ta, tg = fa.split(), fg.split()
    if len(fa) < 4 or len(ta) * 2 < len(tg):
        return False
    return any(tg[i:i + len(ta)] == ta for i in range(len(tg) - len(ta) + 1))


def match(dv: dict, gold: dict) -> str | None:
    """A datavalue ({"type", "value"}) against a gold ({"qids": [...], "text": str}): "exact", "coarser" or None."""
    t, v = (dv or {}).get("type"), (dv or {}).get("value")
    if t == "wikibase-entityid":
        return "exact" if isinstance(v, dict) and v.get("id") in set(gold.get("qids") or ()) else None
    text = gold.get("text") or ""
    if not text:
        return None
    if t == "time":
        return _time(v or {}, dates(text))
    if t == "quantity":
        try:
            amount = Decimal(str((v or {}).get("amount", "")).lstrip("+"))
        except InvalidOperation:
            return None
        return "exact" if any(amount == n for n in numbers(text)) else None
    if t == "monolingualtext":
        return "exact" if _words((v or {}).get("text", ""), text) else None
    if t == "string":
        return "exact" if isinstance(v, str) and _words(v, text) else None
    return None


def snak_values(st: dict):
    """(where, datavalue) for a statement's main value and each qualifier's ("main" or "qual:Pnnn")."""
    ms = st.get("mainsnak") or {}
    if ms.get("snaktype") == "value" and ms.get("datavalue"):
        yield "main", ms["datavalue"]
    for pid, qs in (st.get("qualifiers") or {}).items():
        for q in qs:
            if q.get("snaktype") == "value" and q.get("datavalue"):
                yield f"qual:{pid}", q["datavalue"]


__all__ = ["dates", "fold", "match", "numbers", "snak_values"]
