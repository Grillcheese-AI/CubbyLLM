"""exp_he19_units - H-E19's search read, no model: does UNIT typing make the plan (nearly) unique?

Roy & Roth (2017, unit dependency graphs) composed arithmetic word problems from per-number unit tags with
small models; this asks how much of the composition the units alone decide on our two evals. Per question:
the numbers the host already placed (`spans`, digits and words and world constants) each get a unit from the
words around them (`70 students`, `$5`, `5 miles per hour`, `3 times` = scalar), the asked unit comes from
the question (`how many stickers`, `how much money`), and every expression tree over the numbers (each leaf
used at most once, + - * /) is enumerated by subset dynamic programming under three regimes:

    untyped   any tree, value >= 0                             (what a blind search faces)
    typed     + - only on equal units, * / by unit algebra, and the result's unit must match the asked one
    typed+all typed, and every placed number used             (GSM's usual shape; distractors break it)

Survivors are the distinct values a regime allows; the read is how often the gold value is among them
and how often it is the ONLY one. Tagging errors count against the units (coverage is reported), so
this is a floor on what a learned tagger would give, not a ceiling.

    python validation/exp_he19_units.py standin/data/out/pf_heldout_eval_w_slots.jsonl \
        standin/data/out/emitter_sft_v12e_w_slots.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LEAVES = 6
MAX_SET = 60000          # a mask's (value, unit) set past this is a search no host would run: counted as not unique
SCALAR_WORDS = {"times", "time", "percent", "%", "half", "twice", "thrice", "double", "triple", "dozen", "as", "than", "x"}
SKIP_WORDS = {"additional", "more", "new", "extra", "other", "different", "total", "full", "whole", "large", "small",
              "big", "little", "same", "separate", "remaining", "identical", "equal", "fewer", "less", "of", "the",
              "a", "an", "such", "these", "those", "those", "further", "another", "last", "first", "second", "third"}
STOP_WORDS = {"of", "at", "in", "on", "for", "per", "from", "to", "by", "with", "than", "as", "and", "or", "but",
              "if", "each", "every", "a", "an", "the", "his", "her", "their", "its", "who", "that", "which", "when",
              "while", "after", "before", "is", "are", "was", "were", "did", "does", "do", "has", "have", "had",
              "will", "would", "can", "could", "so", "then", "into", "onto", "about", "over", "under", "between"}
MONEY = {"dollar", "dollars", "$", "money", "buck", "bucks", "usd", "cash", "price", "cost", "earn", "earnings",
         "pay", "paid", "spend", "spent", "profit", "revenue", "income", "salary", "wage", "wages", "fee", "fees",
         "worth"}
RATE_RX = re.compile(r"^\s*(?:per|an?|each|every|apiece)\b\s*(\w+)?")


def stem(w: str) -> str:
    w = w.lower().strip(".,;:!?'\"()")
    if w in MONEY:
        return "dollar"
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def unit_after(text: str, end: int) -> tuple[str | None, str | None, bool]:
    """(numerator, denominator, found) for the number that ends at `end`: the first noun after it that is not a
    qualifier, then `per X` / `an X` / `each` as the denominator ('*' = per anything). Scalars: (None, None, True)."""
    tail = text[end:end + 60]
    words = re.findall(r"[A-Za-z$%]+|[.,;:!?]", tail)
    num = None
    for i, w in enumerate(words[:4]):
        lw = w.lower()
        if lw in SCALAR_WORDS or w == "%":
            return None, None, True
        if lw in ("more", "less", "fewer") and i + 1 < len(words) and words[i + 1].lower() == "than":
            return None, None, True                       # `3 more than` is a scalar offset
        if lw in SKIP_WORDS:
            continue
        if lw in STOP_WORDS or w in ".,;:!?":
            break
        num = stem(w)
        rest = " ".join(words[i + 1:i + 4])
        break
    if num is None:
        return None, None, False
    den = None
    m = RATE_RX.match(rest) if num else None
    if m and m.group(0).strip():
        head = m.group(0).strip().split()[0].lower()
        if head in ("each", "apiece"):
            den = "*"
        elif m.group(1) and m.group(1).lower() not in STOP_WORDS:
            den = stem(m.group(1))
    return num, den, True


def unit_before(text: str, start: int) -> str | None:
    return "dollar" if text[max(0, start - 1):start] == "$" else None


_EACH_RX = re.compile(r"\b(?:each|every|per)\s+(\w+)")


def rate_before(text: str, start: int) -> str | None:
    """`Each student ... receives 79 stickers`: a per-noun stated earlier in the number's sentence."""
    sent = text[:start].rsplit(".", 1)[-1]
    ms = _EACH_RX.findall(sent)
    return stem(ms[-1]) if ms and ms[-1].lower() not in STOP_WORDS else None


def asked_unit(q: str) -> str | None:
    s = q.lower()
    m = re.search(r"how many (?:more |fewer |total |additional |extra )?(\w+)", s)
    if m:
        w = m.group(1)
        if w in ("dollars", "cents"):
            return stem(w)
        return stem(w) if w not in STOP_WORDS and w not in SKIP_WORDS else None
    if re.search(r"how much (?:money|does .* cost|did .* (?:pay|spend|earn|make)|will .* (?:pay|spend|earn|make|cost)|(?:is|was) .* (?:cost|price|profit|worth|paid|earn))", s) or re.search(r"how much .*\$", s):
        return "dollar"
    m = re.search(r"how much (\w+)", s)
    if m and m.group(1) not in STOP_WORDS and m.group(1) not in ("more", "less", "did", "does", "do", "will", "would", "is", "was", "should", "in", "total"):
        return stem(m.group(1))
    if re.search(r"(?:what|how much) (?:is|was|will be) (?:the |his |her |their )?(?:total |final )?(?:cost|price|profit|revenue|amount|bill|change|balance|earnings|income)", s):
        return "dollar"
    return None


_ANY = (("?", 1),)


# ---- unit algebra: a unit is a tuple of sorted (base, exponent); '*' is a wildcard denominator (per anything)
def u_key(d: dict) -> tuple:
    return tuple(sorted((b, e) for b, e in d.items() if e))


def u_mul(a: dict, b: dict, sign: int = 1) -> dict:
    out = dict(a)
    for base, e in b.items():
        out[base] = out.get(base, 0) + sign * e
    # a wildcard denominator cancels one positive base
    if out.get("*", 0) < 0:
        pos = [k for k, v in out.items() if v > 0 and k != "*"]
        if len(pos) == 1:
            out[pos[0]] -= 1
            out["*"] += 1
    return {k: v for k, v in out.items() if v}


def matches(unit: dict, asked: str | None) -> bool:
    if asked is None or u_key(unit) == _ANY:
        return True
    pos = {k for k, v in unit.items() if v > 0}
    return pos == {asked}


def enumerate_values(leaves: list[tuple[Fraction, dict]], asked: str | None, typed: bool, need_all: bool) -> set:
    n = len(leaves)
    reach: list[dict] = [dict() for _ in range(1 << n)]     # mask -> {(value, unit_key): unit}
    for i, (v, u) in enumerate(leaves):
        reach[1 << i][(v, u_key(u))] = u
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        acc = reach[mask]
        if any(len(reach[m]) > MAX_SET for m in range(1, mask)):
            return None
        sub = (mask - 1) & mask
        while sub:
            rest = mask ^ sub
            if sub < rest:                                   # each unordered split once; ops cover both orders
                for (va, ka), ua in reach[sub].items():
                    for (vb, kb), ub in reach[rest].items():
                        if not typed or ka == kb or (typed and (ka == _ANY or kb == _ANY)):
                            kk, uu = (kb, ub) if ka == _ANY else (ka, ua)
                            for v in (va + vb, va - vb, vb - va):
                                if v >= 0:
                                    acc.setdefault((v, kk), uu)
                        if typed:
                            ua_, ub_ = ({} if ka == _ANY else ua), ({} if kb == _ANY else ub)
                            um = u_mul(ua_, ub_)
                            acc.setdefault((va * vb, u_key(um)), um)
                            if vb:
                                ud = u_mul(ua_, ub_, -1)
                                acc.setdefault((va / vb, u_key(ud)), ud)
                            if va:
                                ud = u_mul(ub_, ua_, -1)
                                acc.setdefault((vb / va, u_key(ud)), ud)
                        else:
                            acc.setdefault((va * vb, ()), {})
                            if vb:
                                acc.setdefault((va / vb, ()), {})
                            if va:
                                acc.setdefault((vb / va, ()), {})
            sub = (sub - 1) & mask
    if any(len(x) > MAX_SET for x in reach):
        return None
    masks = [(1 << n) - 1] if need_all else [m for m in range(1, 1 << n) if bin(m).count("1") >= 2]
    out = set()
    for m in masks:
        for (v, k), u in reach[m].items():
            if not typed or matches(u, asked):
                out.add(v)
    return out


def leaves_of(rec: dict, elided: bool = False) -> tuple[list[tuple[Fraction, dict]], int, int]:
    q = rec["question"]
    leaves, found, total = [], 0, 0
    for s in rec["spans"]:
        if s.get("kind") not in ("N", "K"):
            continue
        try:
            v = Fraction(str(s.get("value") or s["text"]).replace(",", "").replace("$", ""))
        except (ValueError, ZeroDivisionError):
            continue
        total += 1
        if s.get("kind") == "K":                          # a world constant: `60 minutes per hour`
            m = re.match(r"\S+\s+(\w+)\s+per\s+(\w+)", s["text"])
            unit = {stem(m.group(1)): 1, stem(m.group(2)): -1} if m else {}
            found += bool(m)
        else:
            num, den, ok = unit_after(q, s["end"])
            found += ok
            before = unit_before(q, s["start"])
            if before and not num:
                num = before
            if num and not den:
                den = rate_before(q, s["start"])
            unit = {}
            if num:
                unit[num] = 1
            if den:
                unit[den] = unit.get(den, 0) - 1
            if not ok and elided:
                unit = {"?": 1}                          # elided: `sells 36` -- a tagger would give it the sentence's unit
        leaves.append((v, unit))
    return leaves, found, total


def read(path: str) -> dict:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = [r for r in recs if r.get("task") == "arithmetic" and r.get("split", "val") == "val"]
    st = Counter()
    survivors = {"untyped": [], "typed": [], "typed+all": [], "typed+elided": [], "typed+elided+all": []}
    for r in recs:
        leaves, found, total = leaves_of(r)
        leaves_e = leaves_of(r, elided=True)[0]
        try:
            gold = Fraction(str(r["gold"]).replace(",", "").replace("$", ""))
        except ValueError:
            st["gold_unparsed"] += 1
            continue
        if not leaves or len(leaves) > MAX_LEAVES:
            st["skipped_size"] += 1
            continue
        asked = asked_unit(r["question"])
        st["n"] += 1
        st["asked_found"] += asked is not None
        st["tags_found"] += found
        st["tags_total"] += total
        for name, typed, need_all in (("untyped", False, False), ("typed", True, False), ("typed+all", True, True),
                                      ("typed+elided", True, False), ("typed+elided+all", True, True)):
            if name == "untyped" and len(leaves) > 5:
                continue
            vals = enumerate_values(leaves_e if "elided" in name else leaves, asked, typed, need_all)
            if vals is None:
                st[f"{name}:n"] += 1
                st[f"{name}:blown"] += 1
                survivors[name].append(MAX_SET)
                continue
            survivors[name].append(len(vals))
            st[f"{name}:n"] += 1
            st[f"{name}:gold_in"] += gold in vals
            st[f"{name}:unique"] += vals == {gold}
            st[f"{name}:le3"] += gold in vals and len(vals) <= 3
            st[f"{name}:le10"] += gold in vals and len(vals) <= 10
    out = {"file": os.path.basename(path), "n": st["n"], "skipped_size": st["skipped_size"],
           "asked_unit_found": st["asked_found"] / max(1, st["n"]),
           "number_tag_found": st["tags_found"] / max(1, st["tags_total"])}
    for name in survivors:
        k = st[f"{name}:n"]
        if k:
            s = sorted(survivors[name])
            out[name] = {"n": k, "median_survivors": s[len(s) // 2], "gold_in": st[f"{name}:gold_in"] / k,
                         "gold_unique": st[f"{name}:unique"] / k, "gold_in_le3": st[f"{name}:le3"] / k,
                         "gold_in_le10": st[f"{name}:le10"] / k, "blown": st[f"{name}:blown"] / k}
    return out


def main(argv=None) -> None:
    paths = (argv or sys.argv[1:])
    res = [read(p) for p in paths]
    for r in res:
        print(f"== {r['file']}: n={r['n']} (skipped {r['skipped_size']} with > {MAX_LEAVES} numbers)  "
              f"asked unit found {r['asked_unit_found']:.2f}  number tagged {r['number_tag_found']:.2f}")
        for name in ("untyped", "typed", "typed+all", "typed+elided", "typed+elided+all"):
            if name in r:
                d = r[name]
                print(f"   {name:17s} n={d['n']:3d} median survivors {d['median_survivors']:5d}  gold in {d['gold_in']:.2f}  "
                      f"unique {d['gold_unique']:.2f}  in & <=3 {d['gold_in_le3']:.2f}  in & <=10 {d['gold_in_le10']:.2f}")
    out = os.path.join(HERE, "logs", "exp_he19_units.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
