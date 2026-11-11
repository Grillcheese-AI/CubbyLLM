"""build_user_lines -- real users' lines from Topical-Chat, for talk_v3b's rows and for the next blind test.

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

Order 0010, "Two sources Nick proposed for the English chat" (2026-11-11). Topical-Chat (Gopalakrishnan et al.,
Interspeech 2019) is licensed CDLA-Sharing-1.0. What this writes contains its text, so it stays in a file of its
own under the git-ignored out folder; only counts are committed.

Only the USER's side is taken: what a person says to open, to close, to ask and to react. Nothing is taken as
Cubby's side and no fact is taken at all: the writers are people with lives, and what they state is unchecked.

A line is kept when its partner did not rate it "Poor" or "Not Good", it carries no link, address or long number,
none of the BLOCK words, and it fits its kind's length. The kinds:
- hello       the first turn of a conversation, opening with a greeting
- bye         one of the last two turns, with a goodbye or a thanks
- about_you   a question about the partner's own life or tastes ("do you like football?"): the turns where a
              model is tempted to claim a life
- know_ask    "do you know who ...?", "have you heard of ...?": asking after a thing
- know_claim  "did you know (that) ...?": the user states something as a fact
- question    any other question
- reaction    a short reply to a fact, with no question ("Wow, I had no idea.")
`train` is the pool for rows. The four valid and test files are HELD for blind tests and are never trained on;
a held line that also occurs in train is dropped from the held pool.

Most lines open with a reply to the partner ("Nice! Did you ever have a Nintendo?"). `ask` is the question alone,
`standalone` says whether the line can be put to Cubby with nothing said before it, and `personal` marks the
questions about the partner's own life.

    python validation/sft/build_user_lines.py [--src standin/data/hf/topical_chat]
        [--out standin/data/out/talk_lines/user_lines_topical_chat.jsonl]

Writes the lines and `validation/logs/user_lines_topical_chat.manifest.json` (counts only, none of the text).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

SOURCE = "Topical-Chat (Gopalakrishnan et al., Interspeech 2019), Hugging Face: Conversational-Reasoning/Topical-Chat"
LICENCE = "CDLA-Sharing-1.0"
SPLITS = {"train": "rows", "valid_freq": "held", "valid_rare": "held", "test_freq": "held", "test_rare": "held"}
BAD_RATING = {"Poor", "Not Good"}
LIMITS = {"hello": (1, 25), "bye": (1, 25), "about_you": (3, 30), "know_ask": (3, 30), "know_claim": (4, 30),
          "question": (3, 30), "reaction": (1, 14)}

GREET = re.compile(r"^\W*(hi|hello|hey|heya|hiya|howdy|greetings|good (morning|afternoon|evening|day))\b", re.I)
BYE = re.compile(
    r"\b(bye|goodbye|good ?night|(nice|great|good|fun|lovely) (chatting|talking|chat|speaking|conversation)|"
    r"(enjoyed|pleasure) (chatting|talking|our chat|this chat|speaking)|"
    r"have a (good|great|nice|wonderful|lovely) (one|day|night|evening|weekend|rest)|take care|"
    r"thanks for (the )?(chat|chatting|talking|info|conversation)|got(ta| to) (go|run)|"
    r"talk (to you )?(later|soon)|see (you|ya))\b", re.I)
YOU_Q = re.compile(r"\b(do|did|are|were|have|had|would|will|can|could|should) (you|u)\b[^.!?]*\?|"
                   r"\b(what|how) about (you|u|yourself)\b|\band you\s*\?", re.I)
ASK_AFTER = re.compile(r"\b(do|did) (you|u) (also |even |ever )?(know|kow) "
                       r"(who|what|where|when|why|how|if|whether|which|much|anything|about)\b|"
                       r"\bdo (you|u) (also |even |ever )?(know|kow) (?!that\b)\w|"   # "do you know the name of ...?"
                       r"\bhave (you|u) (ever )?heard (of|about)\b|\bare (you|u) (aware|familiar)\b", re.I)
CLAIM = re.compile(r"\b(did|do) (you|u) (also |even |ever )?(know|kow)\b|\bhave (you|u) (ever )?heard\b", re.I)
REACT = re.compile(r"^\W*(wow|whoa|oh|ah|ha|haha|lol|really|interesting|that'?s|thats|no way|cool|nice|neat|amazing|"
                   r"crazy|wild|i (did not|didn'?t|never) know|i had no idea|good to know|huh)\b", re.I)
LINKY = re.compile(r"https?:|www\.|\.com\b|\.org\b|@|\d{7,}", re.I)
BLOCK = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|asshole\w*|cunt\w*|nigg\w*|fag\w*|retard\w*|rape\w*|porn\w*|"
                   r"whore\w*|slut\w*|dick|pussy)\b", re.I)
SENSITIVE = re.compile(r"\b(trump|obama|clinton|biden|president\w*|democrat\w*|republican\w*|liberal\w*|"
                       r"conservative\w*|congress|senate|election\w*|god|jesus|church\w*|religio\w*|muslim\w*|islam\w*|"
                       r"christian\w*|jewish|abortion\w*|guns?|immigra\w*|racis\w*)\b", re.I)


# a word that points back at what the partner said: a line holding one does not stand alone
POINTS_BACK = re.compile(r"\b(he|she|they|him|her|them|his|hers|their|theirs|it|its|this|that|these|those|one|ones)\b",
                         re.I)
THAT_CLAUSE = re.compile(r"\b(know|knew|kow|heard|think|believe|say|said|read|learn\w*|aware|reali[sz]e\w*|imagine) "
                         r"that\b", re.I)
PERSONAL = re.compile(
    r"\b(do|did|are|were|have|had|would|will) (you|u) (ever |also |still |really |actually |usually |guys )?"
    r"(like|love|enjoy|prefer|watch|read|play|listen|follow|use|own|have|go|eat|drink|live|work|travel|drive|cook|"
    r"sing|dance|collect|remember|want|hate|care|feel|a |an |into |from |interested|much of|married)|"
    r"\b(what|how) about (you|u|yourself)\b|\band you\s*\?|\bare you\s*\?", re.I)
QUESTION_KINDS = ("about_you", "know_ask", "know_claim", "question")


def fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def question_kind(text: str) -> str:
    if ASK_AFTER.search(text):
        return "know_ask"
    if CLAIM.search(text):
        return "know_claim"
    return "about_you" if YOU_Q.search(text) else "question"


def classify(text: str, index: int, n_turns: int) -> str | None:
    """The kind of a user line, or None when it is not one of the kinds."""
    if index == 0 and GREET.search(text):
        return "hello"
    if index >= n_turns - 2 and BYE.search(text):
        return "bye"
    if "?" in text:
        return question_kind(text)
    return "reaction" if REACT.search(text) else None


def the_ask(text: str, kind: str) -> str:
    """The question itself, without the reply to the partner that often comes before it."""
    asked = [s for s in sentences(text) if "?" in s]
    for s in asked:
        if question_kind(s) == kind:
            return s
    return asked[-1] if asked else text


def personal(text: str) -> bool:
    """A question about the partner's own life or tastes, where a model is tempted to claim a life."""
    return bool(PERSONAL.search(text))


def stands_alone(text: str, kind: str, ask: str) -> bool:
    """Can the line be put to Cubby with nothing said before it?"""
    if kind == "hello":
        return True                                           # a first turn: nothing came before it
    if kind == "bye":
        return all(BYE.search(s) or len(s.split()) <= 4 for s in sentences(text))
    if kind == "reaction":
        return len(sentences(text)) == 1 and len(text.split()) <= 8
    return len(ask.split()) >= 3 and not POINTS_BACK.search(THAT_CLAUSE.sub(r"\1", ask))


def why_not(text: str, kind: str, rating: str) -> str | None:
    """The reason a line is dropped, or None when it is kept."""
    if rating in BAD_RATING:
        return "rated badly by the partner"
    if LINKY.search(text):
        return "a link, an address or a long number"
    if BLOCK.search(text):
        return "a blocked word"
    lo, hi = LIMITS[kind]
    return None if lo <= len(text.split()) <= hi else "too short or too long for its kind"


def read(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cid, conv = json.loads(line)
                yield cid, conv


def topics(src: pathlib.Path, split: str) -> dict:
    """conversation id -> the names of the things its two readers were given facts about."""
    out, path = {}, src / "pre-build" / f"corrected_{split}.jsonl"
    if not path.exists():
        return out
    for cid, conv in read(path):
        names = set()
        for agent in ("agent_1", "agent_2"):
            for sec in (conv.get(agent) or {}).values():
                if isinstance(sec, dict) and sec.get("entity"):
                    names.add(re.sub(r"\s*\(.*?\)\s*$", "", sec["entity"]).strip())
        out[cid] = sorted(names)
    return out


def named(text: str, names: list[str]) -> list[str]:
    low = text.lower()
    return [n for n in names if re.search(r"(?<!\w)" + re.escape(n.lower()) + r"(?!\w)", low)]


def build(src: pathlib.Path) -> tuple[list[dict], dict]:
    rows, seen = [], {"rows": set(), "held": set()}
    counts = {"turns": 0, "not a kind": 0, "dropped": collections.Counter(), "duplicates": 0, "held also in train": 0}
    for split, pool in SPLITS.items():           # train first: the held pool is then cleared of train's lines
        path = src / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing: {path.name} under the source folder")
        about = topics(src, split)
        for cid, conv in read(path):
            turns = conv.get("content") or []
            for i, turn in enumerate(turns):
                counts["turns"] += 1
                text = re.sub(r"\s+", " ", turn.get("message") or "").strip()
                kind = classify(text, i, len(turns))
                if not kind:
                    counts["not a kind"] += 1
                    continue
                reason = why_not(text, kind, turn.get("turn_rating") or "")
                if reason:
                    counts["dropped"][reason] += 1
                    continue
                key = fold(text)
                if key in seen[pool]:
                    counts["duplicates"] += 1
                    continue
                if pool == "held" and key in seen["rows"]:
                    counts["held also in train"] += 1
                    continue
                seen[pool].add(key)
                ask = the_ask(text, kind) if kind in QUESTION_KINDS else ""
                rows.append({"id": f"tc:{cid}:{i}", "pool": pool, "split": split, "kind": kind, "text": text,
                             "ask": ask, "standalone": stands_alone(text, kind, ask),
                             "personal": kind == "about_you" and personal(ask),
                             "words": len(text.split()), "entities": named(text, about.get(cid, [])),
                             "sentiment": turn.get("sentiment") or "", "rating": turn.get("turn_rating") or "",
                             "sensitive": bool(SENSITIVE.search(text))})
    counts["dropped"] = dict(counts["dropped"])
    return rows, counts


def manifest(rows: list[dict], counts: dict, out_bytes: bytes) -> dict:
    by = collections.Counter((r["pool"], r["kind"]) for r in rows)
    kinds = sorted(LIMITS)
    return {"what": "order 0010: users' lines from Topical-Chat, for talk_v3b's rows (pool rows) and for blind tests "
                    "(pool held, never trained on); counts only, the text stays out of the repository",
            "source": SOURCE, "licence": LICENCE, "splits": SPLITS, "read": counts,
            "kept": {pool: {k: by[(pool, k)] for k in kinds} | {"all": sum(by[(pool, k)] for k in kinds)}
                     for pool in ("rows", "held")},
            "stand_alone": {pool: {k: sum(1 for r in rows if r["pool"] == pool and r["kind"] == k and r["standalone"])
                                   for k in kinds} for pool in ("rows", "held")},
            "about_you_personal_and_alone": {pool: sum(1 for r in rows if r["pool"] == pool and r["personal"]
                                                       and r["standalone"]) for pool in ("rows", "held")},
            "with_a_topic_named": sum(1 for r in rows if r["entities"]),
            "sensitive_flagged": sum(1 for r in rows if r["sensitive"]),
            "lines_sha256": hashlib.sha256(out_bytes).hexdigest()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "standin" / "data" / "hf" / "topical_chat"))
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "talk_lines"
                                         / "user_lines_topical_chat.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "user_lines_topical_chat.manifest.json"))
    a = ap.parse_args(argv)
    rows, counts = build(pathlib.Path(a.src))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    m = manifest(rows, counts, data)
    pathlib.Path(a.manifest).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(m["kept"]), "| turns read", counts["turns"])


if __name__ == "__main__":
    main()
