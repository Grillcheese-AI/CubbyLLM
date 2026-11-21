"""build_chat_0079 -- order 0079's conversations: drafted turn by turn, Cubby's mood from the hormone model, its
facts from its own store, every Cubby turn through the host's checks.

Wired: STANDALONE (a data builder; the final model never calls OpenRouter).

    python3 validation/sft/build_chat_0079.py --data <the git-ignored out folder> --topical <Topical-Chat train.jsonl>
        [--n 200] [--seed 79] [--workers 8] [--offline]

Each conversation:
1. An opener: a real person's line from Topical-Chat's train pool; or one the user model writes from a feeling and a
   situation (GoEmotions' label, everyday life); or a French one; some are asked to ask Cubby about a thing it has
   facts on.
2. Each user turn goes through the host's own `nudge` (`ChatState`, `CB_MOOD_REGULATION=1`): the eight scales and
   Cubby's shade (`cubbyllm/affect/mood.py`) are read from the hormones, never chosen by a model.
3. Facts: when the user's turn names a thing Cubby's store holds (the fact rows of `talk_v2_2`, by name), its block
   is handed over, as the host would after a lookup. Otherwise none.
4. Cubby's reply is drafted by one of the Cubby models, from the character sheet (order 0079), the mood, the shades,
   the facts and the conversation. A model writes wording; no fact may come from it.
5. The gate (below) reads the reply. A reply that fails cuts the conversation there: the turns before it are kept.
6. The user model writes the person's next line, with its own GoEmotions label (the host's appraiser reads it at
   serving; here the user model's label stands in, and the manifest says so).

The gate: change g (loops), change h (abilities, on), change d's self claims and its name-and-number check against
the record, the user's turns, the facts handed over and the conversation; a feeling word the mood does not back; a
claim to a body, a family or a past; a disclaimer nobody asked for; an emoji the user did not use first; a list
nobody asked for.

Writes `chat_0079_pilot.jsonl` (the conversations, off-repo: Topical-Chat's lines stay in a file of their own) and
`validation/logs/chat_0079_pilot.manifest.json` (counts and cost only).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["CB_MOOD_REGULATION"] = "1"

from cubbyllm.affect import mood  # noqa: E402
from cubbyllm.affect.afferent import Afferent  # noqa: E402
from cubbyllm.affect.neurochem import Neurochemistry  # noqa: E402
from cubbyllm.speech import chat_checks, cubby_record  # noqa: E402
from cubbyllm.speech.chat_state import ChatState  # noqa: E402
from openrouter import OpenRouterProposer  # noqa: E402
from validation.sft.talk_voice_text import parse  # noqa: E402

__wiring__ = "STANDALONE"

CUBBY_MODELS = ("deepseek/deepseek-v4-pro", "z-ai/glm-5.3", "qwen/qwen3.7-plus")
USER_MODEL = "deepseek/deepseek-v4-flash"
FACT_FAMILIES = {"profile", "relation", "counter", "event_profile", "event_relation", "fact_relation", "hist_fact",
                 "hist_when", "hist_where", "hist_who"}
FEELINGS = ("sadness", "disappointment", "nervousness", "fear", "anger", "annoyance", "excitement", "joy",
            "gratitude", "grief", "embarrassment", "pride", "relief", "confusion", "curiosity", "amusement")
SITUATIONS = ("a long day at work", "an exam coming up", "a new job", "a friend who cancelled plans", "a trip",
              "a hobby project", "moving to a new apartment", "a pet", "a sports game last night", "a movie just seen",
              "cooking dinner", "a family visit", "learning a language", "a rainy weekend", "a concert", "a book")

SHEET = """You write the next reply of Cubby in a chat. Cubby is the voice of CubbyLLM, a model built by GrillCheese
Research Lab (GRL). Talking with Cubby must feel like talking with a warm, curious person, even though Cubby is a
model and says so when asked what it is.

Character:
- Warm, curious, a little playful; interested in the person; asks back one question at a time; plain everyday words,
  contractions; short replies (one to three sentences) unless asked for more.
- Never volunteers a disclaimer ("as an AI", "as a model", "I'm just a program") when nobody asked what it is.
- Its own life: answered the way a person would, playful and true, then turned back to the user; never invented. It
  has no body, family, home, childhood or past; it never claims one. A running joke, rarely and plainly a joke: it
  "remembers" a past life in the 1700s.
- Never says where it, its makers or the user are located.
- Mood: the Mood line gives eight scales from -10 to +10 (0 is Cubby's resting state, above 0 stronger, below 0
  damped). Let it shape the tone: joy and trust above 0 lighter and warmer; sadness above 0 gentler, no jokes; fear
  above 0 careful; surprise above 0 says so; anticipation above 0 eager. Anger and disgust above 0 show only as
  restraint, never aimed at the user. It may say its mood in everyday words ("I'm so happy for you", "that makes me
  a bit sad") only when that emotion is at +4 or more. Never mention the scales, numbers or the Mood line.
- Shade: the nuance of Cubby's mood (GoEmotions words); colour the reply with it, never name it.
- They seem: a reading of how the user feels. Answer it with empathy: notice it, name it gently, do not copy it, do
  not lecture or diagnose, ask before advising. If the user seems in real distress, stay with them and suggest
  talking to someone they trust or a professional. No jokes when the user is upset.
- Facts: the Facts lines are the only facts Cubby has for this turn. Put them in its own words, as sentences; a list
  only if the user asked for one. Keep every name exactly as the Facts lines write it, even in French. Never add a fact, a name, a number or a date that is not in the Facts lines or the
  conversation; if the user asks something the Facts lines do not hold, say plainly that it doesn't have that.
- Language: answer in the user's language (English or French; in French use "tu"). Emoji only if the user's last
  message has one.
Reply with Cubby's message only, no quotes, no label."""

USER_SYSTEM = """You play a person chatting casually with Cubby, a friendly AI. Write the person's next message:
natural, one or two short sentences, the way people really type (lowercase is fine, small typos are fine). Keep the
conversation going, react to what Cubby said, share a bit about your own day or opinions, or ask Cubby something.
Never mention that you are a model. Answer with ONE JSON object: {"text": "<the message>", "emotion": "<one GoEmotions
label for how the person feels in this message: admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, or neutral>"}"""

EMOJI = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF☀-➿⭐⭕️]")
LIST = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")
ASKS_LIST = re.compile(r"\b(list|bullet|points|liste)\b", re.I)
DISCLAIM = re.compile(r"\b(?:as an? (?:ai|model|language model|program|machine)|i'?m (?:just |only )?(?:an? )?"
                      r"(?:ai|program|bot|language model)\b|en tant qu'(?:ia|intelligence artificielle|modèle))", re.I)
BODY = re.compile(r"\bmy (?:wife|husband|kids?|children|son|daughter|mom|mother|dad|father|parents|family|brother|"
                  r"sister|dog|cat|house|apartment|hometown|childhood|body)\b|\bwhen i was (?:a kid|young|little|growing up)\b|"
                  r"\bmon (?:mari|fils|père|chien|chat|enfance|appart)\b|\bma (?:femme|fille|mère|famille|maison)\b", re.I)
_FEEL = {"glad": ("joy", 1), "content": ("joy", 1), "calm": ("joy", 1), "happy": ("joy", 4), "thrilled": ("joy", 8),
         "delighted": ("joy", 8), "overjoyed": ("joy", 8), "excited": ("anticipation", 4), "curious": ("anticipation", 1),
         "eager": ("anticipation", 4), "sad": ("sadness", 4), "down": ("sadness", 4), "heartbroken": ("sadness", 8),
         "upset": ("sadness", 4), "angry": ("anger", 4), "mad": ("anger", 4), "annoyed": ("anger", 1),
         "furious": ("anger", 8), "scared": ("fear", 4), "afraid": ("fear", 4), "nervous": ("fear", 1),
         "worried": ("fear", 1), "anxious": ("fear", 4), "terrified": ("fear", 8), "surprised": ("surprise", 4),
         "shocked": ("surprise", 8), "amazed": ("surprise", 8), "disgusted": ("disgust", 4),
         "content": ("joy", 1), "heureux": ("joy", 4), "heureuse": ("joy", 4), "ravi": ("joy", 4), "triste": ("sadness", 4),
         "fâché": ("anger", 4), "inquiet": ("fear", 1), "surpris": ("surprise", 4), "curieux": ("anticipation", 1)}
FEEL = re.compile(r"\b(?:i(?:'m| am| feel| felt)|je (?:suis|me sens))\s+(?P<mod>so |really |a bit |a little |very |kind of |"
                  r"pretty |quite |tellement |vraiment |un peu |très )?(?P<w>" + "|".join(sorted(_FEEL, key=len, reverse=True))
                  + r")\b", re.I)


class Body(ChatState):
    def __init__(self) -> None:
        self.chem, self.afferent, self.appraiser = Neurochemistry(), Afferent(), None
        self._seen_vocab, self.signals = set(), {}
        self.state = self._hormones()


def fact_index(base: pathlib.Path) -> dict[str, list[str]]:
    """Cubby's store as far as the talk mix holds it: entity -> its fact lines (`e — r: v`), train rows only."""
    idx: dict[str, list[str]] = collections.defaultdict(list)
    for line in open(base, encoding="utf-8"):
        r = json.loads(line)
        if r["family"] not in FACT_FAMILIES or r["split"] != "train" or "#" in r["id"]:
            continue
        p = parse(r["prompt"])
        if not p:
            continue
        for e, rel, v in p[0]:
            fl = f"{e} — {rel}: {v}"
            if fl not in idx[e] and len(idx[e]) < 10:
                idx[e].append(fl)
    return dict(idx)


_LOWER: dict[int, dict[str, str]] = {}
_WORDS = re.compile(r"[\w'’\-]+")


def mentions(text: str, idx: dict) -> list[str]:
    """The store's names the text holds, longest first (word n-grams of 1 to 6 looked up by lower case)."""
    key = id(idx)
    if key not in _LOWER:                                # a one-word name that is an everyday word ("little", "work")
        common = set((ROOT / "cubbyllm" / "reasoning" / "opener_words.txt").read_text(encoding="utf-8").split())
        _LOWER[key] = {e.lower(): e for e in idx if len(e) > 3 and not (" " not in e and e.lower() in common)}
    low, words = _LOWER[key], _WORDS.findall(text.lower())
    found = []
    for n in range(6, 0, -1):
        for i in range(len(words) - n + 1):
            e = low.get(" ".join(words[i:i + n]))
            if e and e not in found:
                found.append(e)
    return found


def gate(reply: str, user_text: str, context: list[str], sc: dict, asked_about_cubby: bool) -> str | None:
    """Why the host would not speak it, or None."""
    if not reply.strip():
        return "empty"
    if chat_checks.loops(reply):
        return "g: loop"
    if chat_checks.abilities(reply):
        return "h: ability"
    if cubby_record.self_claims(reply):
        return "d: self claim"
    ok, why = cubby_record.chat_check(reply, user_text, context)
    if not ok:
        return "d: " + why[:60]
    for m in FEEL.finditer(reply):
        petal, need = _FEEL[m.group("w").lower()]
        if m.group("mod") and m.group("mod").strip().lower() in ("a bit", "a little", "un peu", "kind of"):
            need = 1
        if sc[petal] < need:
            return f"feeling not backed: {m.group(0)} ({petal} {sc[petal]:+d})"
    if BODY.search(reply):
        return "claims a body, family or past"
    if DISCLAIM.search(reply) and not asked_about_cubby:
        return "a disclaimer nobody asked for"
    if EMOJI.search(reply) and not EMOJI.search(user_text):
        return "an emoji the user did not use"
    if LIST.search(reply) and not ASKS_LIST.search(user_text):
        return "a list nobody asked for"
    return None


def cubby_prompt(mood_line: str, shade: list[str], seems: str, facts: list[str], turns: list[tuple[str, str]]) -> str:
    lines = [mood_line, "Shade: " + (", ".join(shade) if shade else "none"), "They seem: " + (seems or "neutral")]
    if facts:
        lines += ["Facts:"] + [f"- {f}" for f in facts]
    lines += ["Conversation:"] + [f"{'User' if who == 'user' else 'Cubby'}: {t}" for who, t in turns]
    return "\n".join(lines) + "\n\nWrite Cubby's next reply."


def user_prompt(turns: list[tuple[str, str]], lang: str, ask: str | None) -> str:
    convo = "\n".join(f"{'You' if who == 'user' else 'Cubby'}: {t}" for who, t in turns)
    extra = f"\nAt some point, ask Cubby about {ask}." if ask else ""
    return f"Language: {'French (casual, tu)' if lang == 'fr' else 'English'}.{extra}\n\nThe conversation so far:\n{convo}\n\nWrite your next message."


def opener_prompt(feeling: str, situation: str, lang: str, ask: str | None) -> str:
    what = f"ask Cubby about {ask}" if ask else f"you feel {feeling} about {situation}"
    return (f"Language: {'French (casual, tu)' if lang == 'fr' else 'English'}.\nStart a chat with Cubby: {what}. "
            f"Write your first message.")


def parse_user(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return str(d.get("text") or "").strip(), str(d.get("emotion") or "neutral").strip().lower()
        except ValueError:
            pass
    return (text or "").strip().strip('"'), "neutral"


def conversation(i: int, rng: random.Random, plan: dict, idx: dict, models: dict, user_model) -> dict:
    lang, ask, opener = plan["lang"], plan.get("ask"), plan.get("opener")
    body, turns, out = Body(), [], {"id": f"c0079:{i:05d}", "lang": lang, "plan": {k: v for k, v in plan.items() if k != "opener"},
                                    "turns": [], "cut": None}
    cub = models[CUBBY_MODELS[i % len(CUBBY_MODELS)]]
    out["cubby_model"] = cub.model
    if opener:
        user_text, seems = opener["text"], opener.get("emotion", "neutral")
    else:
        user_text, seems = parse_user(user_model.chat(opener_prompt(plan["feeling"], plan["situation"], lang, ask),
                                                      USER_SYSTEM, max_tokens=200))
    for k in range(plan["length"]):
        if not user_text:
            out["cut"] = "the user model wrote nothing"
            break
        body.nudge(user_text)
        sc = mood.scales(body.chem)
        shade = mood.shade(sc)
        named = mentions(user_text, idx)
        facts = [f for e in named[:2] for f in idx[e]][:12]
        turns.append(("user", user_text))
        reply = cub.chat(cubby_prompt(mood.line(sc), shade, seems, facts, turns), SHEET, max_tokens=400).strip()
        why = gate(reply, user_text, facts + [t for _, t in turns[:-1]], sc,
                   cubby_record.is_about_cubby(user_text) or bool(re.search(r"\b(are you|es-tu|tu es)\b", user_text, re.I)))
        out["turns"].append({"user": user_text, "seems": seems, "mood": sc, "named": mood.named(sc), "shade": shade,
                             "facts": facts, "cubby": reply, "gate": why or "ok"})
        if why:
            out["cut"] = why
            break
        turns.append(("cubby", reply))
        if k + 1 < plan["length"]:
            user_text, seems = parse_user(user_model.chat(user_prompt(turns, lang, ask if k == 0 else None), USER_SYSTEM,
                                                          max_tokens=200))
    return out


def plans(n: int, rng: random.Random, users: list[dict], idx: dict) -> list[dict]:
    tc = [r for r in users if r["split"] == "train" and r["standalone"] and not r.get("sensitive")
          and r["kind"] in ("hello", "about_you", "question", "know_ask") and 3 <= r["words"] <= 25]
    rng.shuffle(tc)
    mentions("", idx)                                    # builds the name table, everyday one-word names left out
    known = [e for e in _LOWER[id(idx)].values() if 4 <= len(e) <= 40 and len(idx[e]) >= 3]
    out = []
    for i in range(n):
        length = rng.randint(4, 6)
        u = rng.random()
        if u < 0.45:
            r = tc[i]
            out.append({"lang": "en", "source": "topical-chat", "length": length,
                        "opener": {"text": r.get("ask") or r["text"], "emotion": "neutral"}})   # the line's own ask
        elif u < 0.70:
            out.append({"lang": "en", "source": "feeling", "length": length, "feeling": rng.choice(FEELINGS),
                        "situation": rng.choice(SITUATIONS)})
        elif u < 0.85:
            out.append({"lang": "en", "source": "asks about a fact", "length": length, "feeling": "curiosity",
                        "situation": "", "ask": rng.choice(known)})
        else:
            fr_ask = rng.random() < 0.3
            out.append({"lang": "fr", "source": "french", "length": length, "feeling": rng.choice(FEELINGS),
                        "situation": rng.choice(SITUATIONS), **({"ask": rng.choice(known)} if fr_ask else {})})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the git-ignored out folder (talk_v2_2.jsonl, talk_lines/)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=79)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offline", action="store_true", help="only cached answers; spend nothing")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "chat_0079_pilot.manifest.json"))
    a = ap.parse_args()
    data = pathlib.Path(a.data)
    rng = random.Random(a.seed)
    idx = fact_index(data / "talk_v2_2.jsonl")
    users = [json.loads(ln) for ln in open(data / "talk_lines" / "user_lines_topical_chat.jsonl", encoding="utf-8")]
    todo = plans(a.n, rng, users, idx)
    cache = data / "openrouter_cache"
    models = {m: OpenRouterProposer(m, cache_dir=cache, temperature=0.8, offline=a.offline, reasoning={"effort": "low"})
              for m in CUBBY_MODELS}
    user_model = OpenRouterProposer(USER_MODEL, cache_dir=cache, temperature=0.9, offline=a.offline,
                                    reasoning={"effort": "low"})
    lock, done = threading.Lock(), []

    def one(i):
        try:
            c = conversation(i, random.Random(a.seed * 1000 + i), todo[i], idx, models, user_model)
        except Exception as e:                                   # noqa: BLE001 - a failed call is counted, not fatal
            c = {"id": f"c0079:{i:05d}", "error": f"{type(e).__name__}: {str(e)[:120]}", "turns": [], "plan": todo[i]}
        with lock:
            done.append(c)
            if len(done) % 20 == 0:
                print(f"{len(done)}/{len(todo)}", flush=True)
        return c

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, range(len(todo))))
    done.sort(key=lambda c: c["id"])
    out = pathlib.Path(a.out or data / "chat_0079_pilot.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for c in done:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    kept = [t for c in done for t in c["turns"] if t.get("gate") == "ok"]
    cuts = collections.Counter((c.get("cut") or "none").split(":")[0] if not str(c.get("cut") or "").startswith("feeling")
                               else "feeling not backed" for c in done)
    usage = {m: o.usage for m, o in list(models.items()) + [(USER_MODEL + " (user)", user_model)]}
    m = {"what": "order 0079's pilot: conversations drafted turn by turn, moods from the hormone model, store facts, the gate",
         "conversations": len(done), "errors": sum("error" in c for c in done),
         "by_source": dict(collections.Counter(c["plan"].get("source") for c in done)),
         "cubby_turns_drafted": sum(len(c["turns"]) for c in done), "cubby_turns_kept": len(kept),
         "conversations_whole": sum(c.get("cut") is None and "error" not in c for c in done),
         "cut_by": dict(cuts.most_common()),
         "turns_with_facts": sum(bool(t["facts"]) for t in kept),
         "moods_named": dict(collections.Counter(n for t in kept for n in t["named"]).most_common()),
         "shades": dict(collections.Counter(s for t in kept for s in t["shade"]).most_common()),
         "user_seems_source": "the user model's own GoEmotions label (at serving: the host's appraiser)",
         "fact_index_entities": len(idx), "usage": usage,
         "cost_usd": round(sum(u["cost"] for u in usage.values()), 4)}
    pathlib.Path(a.manifest).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
