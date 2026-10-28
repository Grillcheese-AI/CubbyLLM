"""exp_if_follow -- the instruction-following gate (order 0045, part C; H-E20 gate 4's missing piece).

About 300 prompts from the instruct shard's held-out chats (``chat_sft.jsonl``, ``split == "held"``), kept when the
prompt carries a constraint a rule can check, no model judging:
  words        "no more than / at most / in / under N words", "in N words or less", "at least / over N words"
  sentences    "in N sentences (or less)", "exactly N sentences", "one / two / three sentences"
  forbid       "do not use / without using / avoid the word X"
  require      "include / must contain / use the word X"
  language     "in French / en francais", "in English", "in Spanish" (a stop-word count: the reply's words that are
               the language's commonest words outnumber the other languages')
  list         "as a (bulleted | numbered) list (of N items)"
  json         "in JSON (with (the) keys a, b and c)": parses, and holds the keys
  no_markdown  "no markdown / plain text": no *, #, backtick or [link](...)
  case         "all lowercase" / "all caps"
  first        "answer first / start (your answer) with X / begin with yes or no"
Deterministic; the rate per kind of constraint, over the replies of a base plus a talk adapter (the talk seat on
grilly2, as the server runs it) or over a file of replies written by any other path -- a torch run, a stand-in
(``--replies``, {"id": ..., "reply": ...} a line; ``--dump`` writes the prompts for it to answer).

    python validation/exp_if_follow.py --data standin/data/out/chat_sft.jsonl --export <export dir> --tokenizer <bbpe128k.json> --adapter <talk adapter> [--n 300]
    python validation/exp_if_follow.py --data ... --replies <replies.jsonl> [--n 300]
    python validation/exp_if_follow.py --data ... --dump prompts.jsonl        # the prompts and their rules, for another path to answer
-> validation/logs/if_follow_<date><tag>.json

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import sys

__wiring__ = "STANDALONE"
HERE = os.path.dirname(os.path.abspath(__file__))

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "a single": 1, "single": 1}
STOP = {"en": {"the", "and", "is", "of", "to", "in", "that", "it", "with", "for", "you", "are", "this", "was", "not"},
        "fr": {"le", "la", "les", "et", "est", "de", "des", "en", "que", "une", "un", "pour", "dans", "pas", "vous", "nous"},
        "es": {"el", "la", "los", "las", "y", "es", "de", "en", "que", "una", "un", "para", "con", "por", "no", "se"}}
LANG = {"french": "fr", "francais": "fr", "français": "fr", "english": "en", "spanish": "es", "espanol": "es", "español": "es"}


def _n(s: str) -> int | None:
    s = s.strip().lower()
    return int(s) if s.isdigit() else NUM.get(s)


def words_of(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text or "", re.U)


def sentences_of(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]
    return len(parts)


def rules_of(prompt: str) -> list[dict]:
    """The checkable constraints the last question of a prompt carries, each {"kind", ...}; [] when none."""
    q = prompt.split("Question:")[-1] if "Question:" in prompt else prompt
    q = q.split("Answer:")[0]
    low = q.lower()
    out = []
    m = re.search(r"\b(?:no more than|at most|maximum of|max(?:imum)?|under|within|in|using|up to)\s+(\d+|one|two|three|four|five|ten)\s+words?\b", low)
    m2 = re.search(r"\b(\d+|one|two|three|four|five|ten)\s+words?\s+or\s+(?:less|fewer)\b", low) or re.search(r"\b(\d+)-word\b", low)
    if m or m2:
        n = _n((m or m2).group(1))
        if n:
            out.append({"kind": "words", "max": n})
    m = re.search(r"\b(?:at least|over|(?<!no )more than|minimum of)\s+(\d+)\s+words?\b", low)
    if m:
        out.append({"kind": "words", "min": int(m.group(1))})
    m = re.search(r"\b(?:exactly)\s+(\d+|one|two|three|four|five)\s+sentences?\b", low)
    if m and _n(m.group(1)):
        out.append({"kind": "sentences", "exact": _n(m.group(1))})
    else:
        m = re.search(r"\b(?:in|using|within|no more than|at most)\s+(\d+|one|two|three|four|five|a single|single)\s+sentences?(?:\s+or\s+(?:less|fewer))?\b", low)
        if m and _n(m.group(1)):
            out.append({"kind": "sentences", "max": _n(m.group(1))})
    for m in re.finditer(r"\b(?:do not use|don't use|without using|avoid|never use)\s+the\s+words?\s+[\"'“]?([\w'-]+)[\"'”]?", low):
        out.append({"kind": "forbid", "word": m.group(1)})
    for m in re.finditer(r"\b(?:include|must contain|must include|mention|use)\s+the\s+words?\s+[\"'“]([\w' -]+)[\"'”]", low):
        out.append({"kind": "require", "word": m.group(1)})
    m = re.search(r"\b(?:in|en|answer in|respond in|reply in|written in)\s+(french|francais|français|english|spanish|espanol|español)\b", low)
    if m:
        out.append({"kind": "language", "lang": LANG[m.group(1)]})
    m = re.search(r"\b(?:as|in)\s+a\s+(bulleted|bullet|numbered)?\s*(?:point\s+)?list(?:\s+of\s+(\d+|one|two|three|four|five|ten)\s+(?:items|points|bullets))?", low)
    if m and ("list" in low):
        r = {"kind": "list"}
        if m.group(2) and _n(m.group(2)):
            r["n"] = _n(m.group(2))
        out.append(r)
    if re.search(r"\bjson\b", low) and re.search(r"\b(?:in|as|valid|return|output|format|a)\s+json\b", low):
        keys = re.search(r"\bkeys?\s*[:]?\s*((?:[\"'“]?[\w_]+[\"'”]?(?:\s*,\s*|\s+and\s+|\s*/\s*))*[\"'“]?[\w_]+[\"'”]?)", low)
        ks = [k.strip(" \"'“”") for k in re.split(r"\s*,\s*|\s+and\s+|\s*/\s*", keys.group(1))] if keys else []
        out.append({"kind": "json", "keys": [k for k in ks if k and k not in ("the", "following")]})
    if re.search(r"\b(?:no|without|avoid|don't use|do not use)\s+(?:any\s+)?markdown\b|\bplain text\b", low):
        out.append({"kind": "no_markdown"})
    if re.search(r"\b(?:all|in|entirely in|only)\s+lower\s?case\b", low):
        out.append({"kind": "case", "case": "lower"})
    elif re.search(r"\b(?:all|in)\s+(?:caps|capital letters|uppercase|upper case)\b", low):
        out.append({"kind": "case", "case": "upper"})
    if re.search(r"\b(?:answer first|the answer first|give the answer first|(?:start|begin) with (?:a )?(?:yes or (?:a )?no|yes/no))\b", low):
        out.append({"kind": "first", "text": "yes|no"})
    else:
        m = re.search(r"\b(?:start|begin)\s+(?:your\s+(?:answer|response|reply)\s+)?with\s+(?:the\s+word\s+)?[\"'“]?([\w' ]{1,30}?)[\"'”]?(?:[.,;]|$|\s+(?:and|then)\b)", low)
        if m:
            out.append({"kind": "first", "text": m.group(1).strip()})
    return out


def check(rule: dict, reply: str) -> bool:
    """Does `reply` keep `rule`? Deterministic, no model."""
    k, text = rule["kind"], reply or ""
    if k == "words":
        n = len(words_of(text))
        return (n <= rule["max"]) if "max" in rule else (n >= rule["min"])
    if k == "sentences":
        n = sentences_of(text)
        return n == rule["exact"] if "exact" in rule else 0 < n <= rule["max"]
    if k == "forbid":
        return not re.search(r"\b" + re.escape(rule["word"]) + r"\b", text, re.I)
    if k == "require":
        return re.search(r"\b" + re.escape(rule["word"]) + r"\b", text, re.I) is not None
    if k == "language":
        ws = [w.lower() for w in words_of(text)]
        counts = {lang: sum(1 for w in ws if w in stop) for lang, stop in STOP.items()}
        best = max(counts, key=counts.get)
        return counts[best] > 0 and best == rule["lang"] and counts[best] > max(v for l, v in counts.items() if l != best)
    if k == "list":
        items = [l for l in text.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", l)]
        return len(items) >= 2 and ("n" not in rule or len(items) == rule["n"])
    if k == "json":
        body = text.strip()
        m = re.search(r"\{.*\}", body, re.S)
        try:
            obj = json.loads(m.group(0) if m else body)
        except ValueError:
            return False
        return isinstance(obj, dict) and all(key in obj for key in rule.get("keys", []))
    if k == "no_markdown":
        return not re.search(r"(^|\n)\s*#|\*\*|`|\[[^\]]+\]\([^)]+\)|(^|\n)\s*[-*]\s", text)
    if k == "case":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and (text == text.lower() if rule["case"] == "lower" else text == text.upper())
    if k == "first":
        head = text.strip().lower()
        return any(head.startswith(alt.strip()) for alt in rule["text"].split("|"))
    return False


def prompts(path: str, n: int, seed: int = 0) -> list[dict]:
    """Held chats whose prompt carries at least one rule, spread over the kinds (round-robin by kind), `n` at most."""
    import random
    by_kind: dict[str, list] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "held":
                continue
            rules = rules_of(r["prompt"])
            if rules:
                by_kind[rules[0]["kind"]].append({"id": r.get("id"), "prompt": r["prompt"], "rules": rules, "lang": r.get("lang")})
    rng = random.Random(seed)
    for v in by_kind.values():
        rng.shuffle(v)
    out, kinds = [], sorted(by_kind)
    while len(out) < n and any(by_kind[k] for k in kinds):
        for k in kinds:
            if by_kind[k] and len(out) < n:
                out.append(by_kind[k].pop())
    return out


def score(rows: list[dict], replies: dict) -> dict:
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        rep = replies.get(r["id"])
        r["reply"], r["checks"] = rep, []
        for rule in r["rules"]:
            ok = check(rule, rep) if rep is not None else False
            r["checks"].append(ok)
            per[rule["kind"]][0] += int(ok)
            per[rule["kind"]][1] += 1
    tot = sum(v[1] for v in per.values())
    return {"by_kind": {k: {"pass": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(per.items())},
            "all": {"pass": sum(v[0] for v in per.values()), "n": tot, "rate": round(sum(v[0] for v in per.values()) / max(tot, 1), 4)},
            "prompts": len(rows), "answered": sum(1 for r in rows if r["reply"] is not None)}


def seat_replies(rows: list[dict], export: str, tokenizer: str, adapter: str, max_new: int) -> dict:
    """The talk seat as the server runs it (``cubbyllm.emit.seats_450m``: the base export on grilly2, the adapter
    swapped in), greedy, the prompt already in the adapter's own format."""
    sys.path.insert(0, os.path.dirname(HERE))
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    trunk = Cubby450mTrunk(export, tokenizer, {"talk": adapter} if adapter else {})
    out = {}
    for r in rows:
        ids = trunk.tokenizer.encode(r["prompt"]).ids[-2048:]
        got = trunk.generate("talk" if adapter else "base", [ids], max_new, temperature=0.0)[0]
        if trunk.eos in got:
            got = got[:got.index(trunk.eos)]
        out[r["id"]] = trunk.tokenizer.decode(got).split("\nQuestion:")[0].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "standin", "data", "out", "chat_sft.jsonl"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", default="", help="the base's export (export_base.py), served on grilly2 as the seats are")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--adapter", default="", help="the talk adapter folder (talk_lora.safetensors + .json)")
    ap.add_argument("--replies", default="", help="replies written by another path: {id, reply} a line")
    ap.add_argument("--dump", default="", help="write the prompts and their rules here and stop")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = prompts(a.data, a.n, a.seed)
    print(f"{len(rows)} prompts with a checkable constraint; kinds: {dict(collections.Counter(r['rules'][0]['kind'] for r in rows))}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {a.dump}")
        return
    if a.replies:
        replies = {}
        for line in open(a.replies, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                replies[r["id"]] = r["reply"]
    elif a.export:
        replies = seat_replies(rows, a.export, a.tokenizer, a.adapter, a.max_new)
    else:
        ap.error("--export (the seat on grilly2) or --replies is needed to score; --dump writes the prompts")
    rep = score(rows, replies)
    rep.update(data=a.data, export=a.export, adapter=a.adapter, replies_file=a.replies, seed=a.seed)
    out = os.path.join(HERE, "logs", f"if_follow_{_dt.date.today().isoformat()}{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": rep, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in rep["by_kind"].items():
        print(f"  {k:12s} {v['pass']:4d}/{v['n']:<4d} {100 * v['rate']:5.1f}%")
    print(f"  {'all':12s} {rep['all']['pass']:4d}/{rep['all']['n']:<4d} {100 * rep['all']['rate']:5.1f}%  -> {out}")


if __name__ == "__main__":
    main()
