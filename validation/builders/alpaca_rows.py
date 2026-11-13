"""Order 0067, the reader's fourth round (Nick, 2026-11-13: "fold it in now"): rows from Nick's annotated instruction
file (``$CUBBY_INSTRUCT_SVC``: Stanford Alpaca's prompts, CC BY-NC 4.0, with Nick's own svc / pos / lemma / dependency
annotations). Two kinds of row:

- **natural length rules:** a whole prompt (its task and its constraint) whose constraint a strict grammar reads
  completely (`GRAMMAR`: "in a maximum of N words" -> word count <= N; "in N words or less" -> <= N; "in one
  sentence" -> sentence count == 1; "in 1 to 2 sentences" -> 1..2; ...). The rule is the host's reading of the
  phrase, never a model's. A prompt with anything else the rule finders see, or a length phrase the grammar does not
  read in full ("in 50 words"), is left out. This departs from the order's "built in reverse" (a rule drawn, then
  worded): it is recorded in the order.
- **no rule:** a prompt with no constraint on the reply: no hit from the grammar or `exp_if_follow.rules_of`, no
  constraint word (`SIGNALS`), and no number in the annotated complement (Nick's ``svc.c``) before a counted noun.
  Its target is the reader's ``none``.

`rows(seed)` -> rows in `build_rule_rows`' shape: splits ``train`` / ``dev`` and the held-out ``natural_test`` (a fifth
of the natural rows) and ``none_test`` (300 rule-free prompts), never trained on.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]
import exp_if_follow as IF  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

NUMS = {"one": 1, "a single": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "twenty": 20, "fifty": 50, "a hundred": 100, "one hundred": 100}
N = r"(\d{1,4}|one|a single|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|a hundred|one hundred)"
LE, LT, GE = r"(?:a maximum of|no more than|at most|up to)", r"(?:less than|fewer than|under)", r"(?:at least|a minimum of|no less than)"
GRAMMAR = [                                              # (pattern, relation, how the numbers become a plan)
    (rf"\bin {LE} {N} words\b", "word count", "le"), (rf"\b(?:in )?{N} words or (?:less|fewer)\b", "word count", "le"),
    (rf"\bin {LT} {N} words\b", "word count", "lt"), (rf"\bin {GE} {N} words\b", "word count", "ge"),
    (rf"\b{N} words or more\b", "word count", "ge"),
    (rf"\bin {N} (?:to|or|-) {N} sentences\b", "sentence count", "range"), (rf"\bin {N}-{N} sentences\b", "sentence count", "range"),
    (rf"\bin {LE} {N} sentences?\b", "sentence count", "le"), (rf"\bin {LT} {N} sentences\b", "sentence count", "lt"),
    (rf"\bin {GE} {N} sentences\b", "sentence count", "ge"), (rf"\bin {N} sentences?\b", "sentence count", "eq"),
]
LENGTHY = re.compile(r"\b\d+\s*(?:-\s*)?(?:words?|sentences?|paragraphs?|lines?|characters?|letters?|items?|points?|bullets?)\b"
                     r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:words?|sentences?|paragraphs?|lines?)\b", re.I)
SIGNALS = re.compile(r"\b(?:words?|sentences?|paragraphs?|letters?|capital|upper ?case|lower ?case|caps|json|list|bullet"
                     r"|english|french|spanish|german|chinese|italian|japanese|language|translat\w*|quot\w*|title|section"
                     r"|highlight|markdown|comma|exactly|at least|at most|less than|fewer|maximum|minimum|only|limit"
                     r"|start with|begin with|end with|format|table|summar\w*|short|brief|concise|long)\b", re.I)


def _n(s: str) -> int:
    return int(s) if s.isdigit() else NUMS[s]


def _ifeval(rel: str, plan_leaves: list) -> list:
    """The rule as IFEval states it (a word or sentence count: "less than" / "at least" only)."""
    out = []
    key = {"word count": ("length_constraints:number_words", "num_words"),
           "sentence count": ("length_constraints:number_sentences", "num_sentences")}[rel]
    for lf in plan_leaves:
        v, op = lf["v"], lf["op"]
        bound = {"lt": ("less than", v), "le": ("less than", v + 1), "ge": ("at least", v)}.get(
            {"<": "lt", "<=": "le", ">=": "ge"}.get(op, ""))
        if bound:
            out.append([key[0], {key[1]: bound[1], "relation": bound[0]}])
    return out if len(out) == len(plan_leaves) else []


def read_length(text: str):
    """The prompt's one length rule as a plan, or None (no phrase, two phrases, or one the grammar does not read)."""
    low = " ".join(text.lower().split())
    hits = []
    for pat, rel, how in GRAMMAR:
        for m in re.finditer(pat, low):
            hits.append((m.span(), rel, how, [_n(g) for g in m.groups()]))
    spans = {h[0] for h in hits}
    if len(spans) != 1 or len(LENGTHY.findall(low)) > 1:          # one phrase, and no other count in the prompt
        return None
    _, rel, how, ns = max(hits, key=lambda h: len(h[3]))          # the fullest reading of that one phrase
    if how == "range":
        lo, hi = ns
        if not 0 < lo < hi:
            return None
        leaves = [{"m": rel, "op": ">=", "v": lo}, {"m": rel, "op": "<=", "v": hi}]
    else:
        leaves = [{"m": rel, "op": {"le": "<=", "lt": "<", "ge": ">=", "eq": "=="}[how], "v": ns[0]}]
    if any(lf["v"] <= 0 or lf["v"] > 400 for lf in leaves):
        return None
    return (leaves[0] if len(leaves) == 1 else {"and": leaves}), _ifeval(rel, leaves)


def rule_free(rec: dict) -> bool:
    t = rec["text"]
    return not (IF.rules_of(t) or SIGNALS.search(t) or LENGTHY.search(t) or re.search(r"\d", (rec.get("svc") or {}).get("c", "")))


def rows(seed: int = 4, max_chars: int = 300, n_none: int = 2000, n_none_test: int = 300) -> list[dict]:
    path = Path(local_paths.value("CUBBY_INSTRUCT_SVC") or "")
    natural, free = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "_p" not in r["id"] or len(r["text"]) > max_chars:
            continue
        got = read_length(r["text"])
        if got and not [x for x in IF.rules_of(r["text"]) if x["kind"] not in ("words", "sentences")]:
            natural.append({"kind": "natural_" + ("words" if "word" in json.dumps(got[0]) else "sentences"), "lang": "en",
                            "instruction": r["text"], "plan": got[0], "template": None, "ifeval": got[1], "source_id": r["id"]})
        elif rule_free(r):
            free.append({"kind": "none", "lang": "en", "instruction": r["text"], "plan": None, "template": None,
                         "ifeval": [], "source_id": r["id"]})
    rng = random.Random(seed)
    rng.shuffle(natural)
    rng.shuffle(free)
    k = len(natural) // 5
    for i, r in enumerate(natural):
        r["split"] = "natural_test" if i < k else ("dev" if rng.random() < 0.1 else "train")
    free = free[:n_none]
    for i, r in enumerate(free):
        r["split"] = "none_test" if i < n_none_test else ("dev" if rng.random() < 0.1 else "train")
    return natural + free


__all__ = ["GRAMMAR", "read_length", "rule_free", "rows"]
