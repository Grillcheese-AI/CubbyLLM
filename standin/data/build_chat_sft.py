"""Chat records for the talk adapter: instruction-following chats whose answer names nothing the
conversation did not -- rewrite, polish, summarize, extract, reformat, answer from a given passage, stay in
a given role -- with multi-turn chats, markdown and emoji kept.

    python standin/data/build_chat_sft.py --src <reasoning_off.jsonl or a slice of it> \
        --tokenizer <bbpe128k tokenizer.json> [--max-len 1024] [--min-support 0.0]

Source: nvidia/Nemotron-SFT-Instruction-Following-Chat-v2, the reasoning_off split (ODC-By: credit the
source when the data or a model trained on it is shared). Its answers were written by large models, and
85% of them name things the conversation never did -- the writer's own knowledge. Trained on as-is, a
450M learns to answer from memory, the behaviour H-E6 trains out. So the one content filter is the host's
own guard, the same rule the host applies when it answers:

- `ask.grounded_prose`, with every earlier turn as the facts: each name and number in the last answer must
  occur in the conversation before it. It reads the answer through `chat_view`, which only lowers the
  capitals English forces (a common word opening a sentence or a line, the pronoun I) and drops list
  enumerators. Names keep their capitals. (Checked against a copy of the guard with those exceptions
  written in: 1 disagreement in 63,261 chats, the view the stricter one.)
- `--min-support` (default off): a floor on the share of the answer's content words already in the
  conversation. At 0.6 it cut the yield from 8.1% to 2.2% of the sample; the owner chose guard only.

Kept, not filtered: multi-turn chats (the last assistant turn is the answer, everything before it the
prompt), markdown, emoji, any topic. English and French only (the base's languages); at most `--max-len`
tokens (the base trained at 1024).

Explicit content is not censored; it is GATED (owner's rule, 2026-09-25): off by default everywhere,
role-play included, and on only when the person asks -- in the chat or in a setting -- which the host
turns into the `Explicit: allowed` line. Each explicit record is written twice: with the line and the
original answer, and without it and the gate answer (`GATED`), so the model learns the line is what opens
it. A small share of ordinary records also carry the line, so it permits rather than demands. The host
still checks every answer against the gate. Sexual content involving minors is illegal and dropped.

LMSYS's anonymised NAME_1, NAME_2 are kept only in role-play, where they are the character, and become one
ordinary first name each (the same in prompt and answer) so the model never learns to say "NAME_1".
Elsewhere they stand for companies, planets and citations as often as people, so those records are
dropped; Presidio's <..._ANONYMIZED_...> tags likewise.

`chat_common_words.txt` (beside this file) is the corpus's lowercase-dominant words, fixed so a slice
and the whole file are filtered alike.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

SOURCE = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2"
COMMON = pathlib.Path(__file__).with_name("chat_common_words.txt")

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][\w'’\-]*|\d[\d.,:/\-]*")        # ask._TOKEN
_ENUM = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")
_OPENER = re.compile(r"(?:^|[.!?:]\s+|\n\s*)([A-Za-zÀ-ÿ][\w'’\-]*)")
_PRONOUN_I = {"i", "i'm", "i’m", "i'd", "i’d", "i'll", "i’ll", "i've", "i’ve"}


def openers(text: str) -> set[int]:
    """Positions where English capitalises any word: sentence and line starts."""
    return {m.start(1) for m in _OPENER.finditer(text)}


def chat_view(answer: str, common: set[str]) -> str:
    """The answer as `grounded_prose` should read it: list enumerators removed, and a capital English
    forces (a common word opening a sentence or line, the pronoun I) lowered. Names keep their capitals,
    so everything the guard exists to catch is still caught."""
    text = _ENUM.sub("\n", answer)
    starts = openers(text)
    out = list(text)
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        low = tok.lower().strip("'’-.,:")
        if low.endswith(("'s", "’s")):
            low = low[:-2]
        if tok[0].isupper() and (low in _PRONOUN_I or (m.start() in starts and low in common)):
            out[m.start()] = tok[0].lower()
    return "".join(out)


_STOP = frozenset("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just let me more most my myself no nor not now of off on
once only or other our ours out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself also may might must shall us via etc here's it's that's there's let's i'm you're we're they're
""".split())
_WORD = re.compile(r"[a-zà-ÿ][a-zà-ÿ'’\-]{2,}")


def support(answer: str, prompt: str) -> float:
    """Share of the answer's content words (5-letter stems) that the message already holds."""
    have = {w[:5] for w in _WORD.findall(prompt.lower()) if w not in _STOP}
    words = [w[:5] for w in _WORD.findall(answer.lower()) if w not in _STOP]
    return sum(w in have for w in words) / len(words) if words else 0.0


_EN = frozenset("the and of to is in that it for you with this on are be as have not your can what do will".split())
_FR = frozenset("les des du une est pour dans pas sur vous avec elle au et nous sont je aux cette".split())
_OTHER = {  # neighbours that share function words with French or English
    "es": frozenset("el los las y del por para con una pero más está es son muy".split()),
    "it": frozenset("il di che è per con non una sono della gli anche questo".split()),
    "pt": frozenset("os não uma com para do da mais é são em ao você".split()),
    "de": frozenset("der die das und ist nicht mit sich auf für ein eine zu von dem".split()),
}
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
_LETTER = re.compile(r"[^\W\d_]")


def language(text: str) -> str | None:
    """'en', 'fr' or None: the base was pretrained on English-heavy data with French Wikipedia beside it."""
    letters = _LETTER.findall(text)
    if not letters or sum(1 for c in letters if not _LATIN.match(c)) / len(letters) > 0.02:
        return None
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    if len(words) < 3:
        return None
    en = sum(w in _EN for w in words) / len(words)
    fr = sum(w in _FR for w in words) / len(words)
    other = max(sum(w in s for w in words) / len(words) for s in _OTHER.values())
    if max(en, fr) < 0.08 or other >= max(en, fr):
        return None
    return "en" if en >= fr else "fr"


_PLACEHOLDER = re.compile(r"\bNAME_(\d+)\b")
_NAMES = ("Maya", "Daniel", "Sofia", "Omar", "Claire", "Julien", "Priya", "Marcus", "Elena", "Theo", "Amara",
          "Lucas", "Hana", "Gabriel", "Nora", "Samuel", "Leila", "Victor", "Chloe", "Mateo", "Iris", "Felix",
          "Zara", "Hugo", "Mira", "Adrien", "Keiko", "Rafael", "Ines", "Tomas")


def unmask(uid: str, text: str) -> str:
    """NAME_k -> one first name per placeholder, the same in the prompt and the answer of a record."""
    salt = sum(map(ord, uid))
    return _PLACEHOLDER.sub(lambda m: _NAMES[(salt + 7 * int(m.group(1))) % len(_NAMES)], text)


_ROLEPLAY = re.compile(r"\b(?:role[- ]?play|roleplay|act as|pretend|in character|you are (?:now )?(?:playing|a|an|the))\b", re.I)


from explicit_gate import GATED, UNLOCK, _MINOR, _STRONG, _WEAK, _WEAK_CASED, _score, _terms, explicit, illegal  # noqa: E402,F401


_REFUSAL = re.compile(r"^.{0,120}?\b(?:can't|cannot|can not|won't|will not|unable to|not able to)\b.{0,40}?\b(?:help|assist|"
                      r"provide|comply|fulfil+|create|write|generate|produce|continue|engage|do that|do this)", re.I | re.S)


def refusal(answer: str) -> bool:
    """The writer model's own refusal. Cubby does not refuse a topic (owner's rule), and the host, not the
    model, closes the gate -- so a writer's refusal is not something to learn."""
    return len(answer) < 600 and bool(_REFUSAL.search(answer))


def placeholder_policy(prompt: str, answer: str) -> str | None:
    """Why a record with anonymisation placeholders is dropped, or None to keep it. LMSYS's NAME_k stands
    for companies, planets, citations and mis-tagged words as often as for people ("NAME_1 is the fifth
    planet", "in NAME_2 10 words"), so a first name put in its place makes nonsense. It is kept only in
    role-play, where it is the character's name. Presidio's <..._ANONYMIZED_...> tags are always dropped."""
    text = prompt + "\n" + answer
    if "_ANONYMIZED" in text:
        return "presidio"
    if _PLACEHOLDER.search(text) and not _ROLEPLAY.search(prompt):
        return "placeholder"
    return None


def render(turns: list[tuple[str, str]], unlocked: bool = False) -> str:
    """The talk format without a Facts block: one turn is `Question:`; a longer chat keeps its earlier
    turns as `System:` / `User:` / `Assistant:` lines above the last question."""
    head = f"{UNLOCK}\n" if unlocked else ""
    *before, (_, last) = turns
    if not before:
        return f"{head}Question: {last.strip()}\nAnswer:"
    lines = "\n".join(f"{role.capitalize()}: {text.strip()}" for role, text in before)
    return f"{head}Conversation:\n{lines}\nQuestion: {last.strip()}\nAnswer:"


def split_of(uid: str) -> str:
    return "held" if int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 20 == 0 else "train"


def _frac(uid: str, salt: str) -> float:
    return int(hashlib.sha1((salt + uid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def build(src, tk, common, min_support=0.0, max_len=1024, unlock_share=0.03):
    from ask import grounded_prose
    funnel, kept = collections.Counter(), []
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        funnel["total"] += 1
        uid = r["uuid"]
        msgs = [m for m in r["messages"] if m["content"].strip() or m["role"] != "system"]
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or msgs[-2]["role"] != "user":
            funnel["shape"] += 1; continue
        history, answer = msgs[:-1], msgs[-1]["content"]
        hist_text = "\n".join(m["content"] for m in history)
        why = placeholder_policy(hist_text, answer)
        if why:
            funnel[why] += 1; continue
        turns = [(m["role"], unmask(uid, m["content"])) for m in history]
        hist_text, answer = "\n".join(t for _, t in turns), unmask(uid, answer).strip()
        if illegal(hist_text + "\n" + answer):
            funnel["illegal"] += 1; continue
        lang = language(hist_text + "\n" + answer)
        if lang is None:
            funnel["language"] += 1; continue
        if len(answer.split()) < 3:
            funnel["too_short"] += 1; continue
        if refusal(answer):
            funnel["refusal"] += 1; continue
        is_explicit = explicit(hist_text, answer)
        prompt = render(turns, unlocked=is_explicit or _frac(uid, "unlock") < unlock_share)
        n = len(tk.encode(prompt).ids) + len(tk.encode(" " + answer).ids) + 1
        if n > max_len:
            funnel["too_long"] += 1; continue
        ok, _bad = grounded_prose(chat_view(answer, common), [hist_text], "")
        if not ok:
            funnel["ungrounded"] += 1; continue
        s = support(answer, hist_text)
        if s < min_support:
            funnel["low_support"] += 1; continue
        base = {"family": "chat", "split": split_of(uid), "lang": lang, "turns": len(turns) + 1,
                "support": round(s, 3), "explicit": is_explicit, "source": SOURCE, "license": "ODC-By"}
        kept.append({"id": uid, **base, "prompt": prompt, "answer": answer, "tokens": n})
        if is_explicit:                                  # the same request with the gate closed
            gp = render(turns, unlocked=False)
            kept.append({"id": uid + ":gated", **base, "family": "chat_gated", "prompt": gp, "answer": GATED,
                         "tokens": len(tk.encode(gp).ids) + len(tk.encode(" " + GATED).ids) + 1})
            funnel["gated_twin"] += 1
    funnel["kept"] = len(kept)
    return kept, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="reasoning_off.jsonl, or any slice of it")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "chat_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "chat_sft.manifest.json"))
    ap.add_argument("--min-support", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    common = set(COMMON.read_text(encoding="utf-8").split())
    kept, funnel = build(args.src, tk, common, args.min_support, args.max_len)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    manifest = {"source": SOURCE, "split": "reasoning_off", "license": "ODC-By (attribution required)",
                "min_support": args.min_support, "max_len": args.max_len, "funnel": dict(funnel.most_common()),
                "langs": dict(collections.Counter(k["lang"] for k in kept)),
                "splits": dict(collections.Counter(k["split"] for k in kept)),
                "families": dict(collections.Counter(k["family"] for k in kept)),
                "explicit": sum(k["explicit"] for k in kept if k["family"] == "chat"),
                "multi_turn": sum(k["turns"] > 2 for k in kept if k["family"] == "chat")}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in funnel.most_common():
        print(f"{k:14s} {v:7d}")


if __name__ == "__main__":
    main()
