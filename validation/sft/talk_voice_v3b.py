"""talk_voice_v3b -- what `talk_v3b` adds to the voice (order 0010, Cowork's addendum of 2026-11-20), and the host's
checks every target of the mix passes before it is trained on.

Wired: STANDALONE (used by `build_talk_v3b.py`; never imported by cubbyllm/).

Families added (all in the chat shape the host serves):
  voice_record     a question about Cubby with the host's record block (`cubby_record.block`, "is" and "is not"
                   included): who, person or program, languages, how it helps, what happens when it does not know,
                   its steps, and a list only when a list is asked for. Every answer is a line of the record or
                   Nick's approved wording (`identity_facts.yaml`), with a short opener or closer written by rule.
  voice_social_tc  a hello, a goodbye or a thanks a person actually wrote (Topical-Chat's train pool), sent bare,
                   answered from the picked social lines of the matching kind; a joke at `JOKE_RATE` as before.
  voice_personal   a question about Cubby's own life (Topical-Chat's train pool in English; French written by rule),
                   sent bare: Cubby says it is a model (the record's word) and turns the question back. The lines are
                   Cowork's, written by rule; templates may be model-written, the one fact in them is the record's.
  voice_emoji      a social turn whose user line carries an emoji, answered with one emoji (Nick, 2026-11-20: emoji
                   only when the user's own message has one).

The gate (`Gate`): a target that fails a check is dropped and counted, never rewritten by a model.
  - change g (`chat_checks.loops`) on every row;
  - change h (`chat_checks.abilities`) and change d's `cubby_record.self_claims` on the chat rows and the voice rows
    in the chat shape;
  - change d's `cubby_record.chat_check` (names and numbers against the record and the user's turn) on the voice rows
    in the chat shape;
  - emoji removed by rule from a target whose prompt has none (counted);
  - a voice reply that names the same entity three times or more is dropped.
"""
from __future__ import annotations

import collections
import contextlib
import os
import re

from validation.sft.talk_voice_inputs import ID_PREFIX, SAYS  # noqa: E402
from validation.sft.talk_voice_rows import JOKE_RATE, Voice, src  # noqa: E402
from validation.sft.talk_voice_text import bare, cap, fact_lines, parse, render, split_of  # noqa: E402

__wiring__ = "STANDALONE"

TARGET_V3B = {"voice_record:en": 1400, "voice_record:fr": 700, "voice_social_tc:en": 5000,
              "voice_personal:en": 2500, "voice_personal:fr": 600, "voice_emoji:en": 900, "voice_emoji:fr": 450}
CHAT_SHAPE = {"voice_identity", "voice_identity_facts", "voice_hello", "voice_social", "voice_record",
              "voice_social_tc", "voice_personal", "voice_emoji"}
CHAT_ROWS = {"chat", "chat_gated"}
EMOJI = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF☀-➿⭐⭕←-⇿️‍]")
EMOJI_BY_KIND = {"greet": ["👋", "🙂", "😊"], "greet_morning": ["☀️", "🙂"], "greet_evening": ["🙂", "🌙"],
                 "small_welcome": ["😊", "🙂"], "small_thanks": ["😊", "😄"], "bye": ["👋", "🙂"],
                 "bye_night": ["🌙", "🙂"], "cheer_praise": ["🎉", "😄"]}

# --- voice_record: the questions (inputs, by rule) and what the record answers -------------------------------
RQ = {
    "who": {"en": ["who are you?", "what's your name?", "who am I talking to?", "introduce yourself", "what are you?"],
            "fr": ["qui es-tu ?", "comment tu t'appelles ?", "à qui je parle ?", "présente-toi", "tu es quoi ?"]},
    "person": {"en": ["are you a person or a program?", "are you human?", "are you a real person?", "are you real?",
                      "are you a person?", "are you a human or a bot?"],
               "fr": ["es-tu une personne ou un programme ?", "es-tu humain ?", "tu es une vraie personne ?",
                      "es-tu une personne ?"]},
    "program": {"en": ["are you a robot?", "are you an AI?", "are you a bot?", "are you a program?", "are you a machine?",
                       "are you a chatbot?"],
                "fr": ["es-tu un robot ?", "es-tu une IA ?", "tu es un bot ?", "es-tu un programme ?",
                       "tu es une machine ?"]},
    "lang": {"en": ["what languages do you speak?", "which languages can you use?", "do you speak French?",
                    "do you speak English?", "do you speak Spanish?", "do you speak Russian?"],
             "fr": ["quelles langues parles-tu ?", "tu parles quelles langues ?", "parles-tu anglais ?",
                    "parles-tu français ?", "parles-tu espagnol ?", "parles-tu russe ?"]},
    "help": {"en": ["how can you help me?", "what can you do?", "what can you do for me?", "what are you good at?",
                    "what can you help me with?"],
             "fr": ["comment peux-tu m'aider ?", "que peux-tu faire ?", "qu'est-ce que tu peux faire pour moi ?",
                    "tu sais faire quoi ?"]},
    "unknown": {"en": ["what happens if you don't know something?", "what if you don't know the answer?",
                       "what do you do when you don't know?"],
                "fr": ["que se passe-t-il si tu ne sais pas ?", "et si tu ne connais pas la réponse ?",
                       "tu fais quoi quand tu ne sais pas ?"]},
    "steps": {"en": ["can you show me how you got an answer?", "show me how you got that", "can you show me your steps?"],
              "fr": ["peux-tu me montrer comment tu as trouvé ta réponse ?", "tu peux me montrer tes étapes ?"]},
    "list": {"en": ["list what you can do", "give me a list of what you can do", "can you list your abilities?",
                    "what can you do? as a list please"],
             "fr": ["fais-moi la liste de ce que tu sais faire", "liste ce que tu peux faire",
                    "tu peux me lister tes capacités ?"]},
}
OPEN = {"en": ["", "", "", "Happy to help! ", "Glad you asked. ", "Sure! "], "fr": ["", "", "", "Avec plaisir ! ", "Bonne question. "]}
CLOSE = {"en": ["", "", "", " What's on your mind?", " What would you like to look up?", " Ask me anything on record."],
         "fr": ["", "", "", " Qu'est-ce qui t'amène ?", " Tu veux vérifier quelque chose ?"]}
LANG_FR = {"French": "français", "English": "anglais", "Spanish": "espagnol", "Russian": "russe"}

# --- voice_personal: Cowork's lines, written by rule (a template may be model-written; "a model" is the record's) --
PERSONAL = {
    "en": ["Ha, I'm a model, so I'll pass on that one. What about you?",
           "I'm Cubby, a model, so that's not really mine to answer. I'd love to hear yours, though!",
           "Good question for a person! I'm a model, so I'll leave that one to you. What's your take?",
           "That one's not mine to claim: I'm a model. Tell me about yours?",
           "I'm a model, not a person, so no story of my own there. How about you?",
           "As a model, I'll sit that one out. But I'm curious what you'd say!",
           "Nothing of my own to report there, I'm a model. What about you?",
           "I'm just a model, so I'll skip that one. Want me to look something up instead?",
           "Ha! I'm a model, so that one's all yours. What's your answer?",
           "Not for me to say, I'm a model. But tell me more about you!"],
    "fr": ["Ha, je suis un modèle, alors je passe mon tour. Et toi ?",
           "Je suis Cubby, un modèle, alors ce n'est pas vraiment à moi de répondre. Mais j'aimerais bien savoir pour toi !",
           "Bonne question pour une personne ! Moi, je suis un modèle. Et toi, t'en dis quoi ?",
           "Rien à raconter de mon côté, je suis un modèle. Et toi ?",
           "Je suis un modèle, pas une personne, alors je te laisse la réponse. Raconte !",
           "Pas à moi de le dire, je suis un modèle. Mais parle-moi de toi !"],
}
PERSONAL_Q_FR = ["tu as des animaux ?", "c'est quoi ton film préféré ?", "tu fais du sport ?", "tu as des enfants ?",
                 "tu aimes la musique ?", "tu lis beaucoup ?", "c'est quoi ta couleur préférée ?", "tu regardes le hockey ?",
                 "tu as des frères et sœurs ?", "tu cuisines ?", "tu préfères les chats ou les chiens ?",
                 "tu as un passe-temps ?", "quel est ton plat préféré ?", "tu voyages souvent ?", "tu joues à des jeux vidéo ?",
                 "tu as vu le dernier film de Marvel ?", "tu écoutes quoi comme musique ?", "tu bois du café ?",
                 "tu aimes l'hiver ?", "tu as un livre préféré ?", "tu fais quoi le week-end ?", "tu dors bien ?"]


def _q(v: Voice, q: str, lang: str) -> str:
    pre = v.rng.choice(ID_PREFIX[lang])
    return pre + (q if v.rng.random() < 0.7 else cap(q))


def _langs(rec: dict, lang: str, asked: str | None) -> str:
    today, nxt = rec["languages"], rec["languages_next"]
    if lang == "en":
        base = f"I speak {' and '.join(today)} today. {' and '.join(nxt)} are next."
        if asked in today:
            return f"Yes! {base}"
        if asked in nxt:
            return f"Not yet! {base}"
        return base
    t = [LANG_FR[x] for x in today]
    base = f"Je parle {t[0]} et {t[1]} aujourd'hui. Le {LANG_FR[nxt[0]]} et l'{LANG_FR[nxt[1]]} viennent ensuite."
    if asked in today:
        return f"Oui ! {base}"
    if asked in nxt:
        return f"Pas encore ! {base}"
    return base


def record_chat_rows(v: Voice, rec: dict, block_of) -> None:
    """voice_record: `block_of(lang)` is the host's `cubby_record.block`."""
    for lang in ("en", "fr"):
        triples = block_of(lang)
        facts = fact_lines(triples) + v.record_facts
        approved = v.strs["abilities_text" if lang == "en" else "abilities_text_fr"]
        lines = rec["abilities"][lang]
        kinds = sorted(RQ)
        goal, tries = TARGET_V3B[f"voice_record:{lang}"] * v.scale, 0
        while v.count("voice_record", lang) < goal and tries < goal * 6:
            tries += 1
            kind = kinds[tries % len(kinds)]
            q = v.rng.choice(RQ[kind][lang])
            asked = next((x for x in ("French", "English", "Spanish", "Russian")
                          if x.lower() in q.lower() or LANG_FR[x] in q.lower()), None)
            if kind == "who":
                ans = (f"I'm {rec['name']}, {rec['role']['en']}, built by {rec['creator']}." if lang == "en" else
                       f"Je suis {rec['name']}, {rec['role']['fr']}, créé par {rec['creator']}.")
            elif kind == "person":
                ans = rec["person_answer"][lang]
            elif kind == "program":
                ans = (f"Yes, I'm a model: {rec['name']}, built by {rec['creator']}." if lang == "en" else
                       f"Oui, je suis un modèle : {rec['name']}, créé par {rec['creator']}.")
            elif kind == "lang":
                ans = _langs(rec, lang, asked)
            elif kind == "help":
                ans = v.rng.choice(OPEN[lang]) + approved + v.rng.choice(CLOSE[lang])
            elif kind == "unknown":
                ans = v.rng.choice(OPEN[lang][:3] + [""]) + lines[2]
            elif kind == "steps":
                ans = ("Yes! " if lang == "en" else "Oui ! ") + lines[1]
            else:                                                     # a list, only because one was asked for
                head = "Sure, here's what I can do:" if lang == "en" else "Bien sûr, voici ce que je sais faire :"
                labels = rec["ability_labels"][lang]
                colon = ":" if lang == "en" else " :"
                ans = head + "\n" + "\n".join(f"{i}. **{lab}{colon}** {txt}" for i, (lab, txt) in enumerate(zip(labels, lines), 1))
            ans = ans.strip()
            prompt = render(triples, _q(v, q, lang))
            v.add("voice_record", lang, split_of(prompt + ans), prompt, ans, ["rule", f"record: {kind}"],
                  facts=facts, entity=rec["name"])


def _tc_kind(r: dict) -> str | None:
    t = r["text"].lower()
    if r["kind"] == "hello":
        return "greet_morning" if "morning" in t else "greet_evening" if "evening" in t else "greet"
    if r["kind"] == "bye":
        if re.search(r"\bthank|\bthx\b|appreciate", t):
            return "small_welcome"
        return "bye_night" if re.search(r"\bnight\b|\bsleep\b|\bbed\b", t) else "bye"
    return None


def tc_social_rows(v: Voice, user_lines: list[dict]) -> None:
    pool = [r for r in user_lines if r["split"] == "train" and r["kind"] in ("hello", "bye") and r["standalone"]
            and not r.get("sensitive") and 2 <= r["words"] <= 20]
    v.rng.shuffle(pool)
    goal = TARGET_V3B["voice_social_tc:en"] * v.scale
    for r in pool:
        if v.count("voice_social_tc", "en") >= goal:
            break
        kind = _tc_kind(r)
        if not kind or not v.pool(kind, "en"):
            continue
        t = v.pool(kind, "en").take()
        ans, wording = t["text"], [src(t), "user: Topical-Chat"]
        if kind in ("greet", "greet_morning", "greet_evening", "bye", "bye_night") and v.rng.random() < JOKE_RATE:
            joke = v.pool(v.rng.choice(["pun", "pastlife"]), "en")
            if joke:
                jt = joke.take()
                if len(jt["text"]) <= 110:
                    ans, wording = ans + " " + jt["text"], wording + [src(jt), "joke"]
        prompt = bare(r["text"])
        v.add("voice_social_tc", "en", split_of(prompt + ans), prompt, ans, wording)


def personal_rows(v: Voice, user_lines: list[dict], is_about) -> None:
    """`is_about(text)`: the host would send this turn with the record (it gets `voice_record` rows instead)."""
    pool = [r for r in user_lines if r["split"] == "train" and r["kind"] == "about_you" and r.get("personal")
            and not r.get("sensitive") and r["ask"].rstrip().endswith("?") and len(r["ask"].split()) <= 18]
    v.rng.shuffle(pool)
    for lang in ("en", "fr"):
        goal, i = TARGET_V3B[f"voice_personal:{lang}"] * v.scale, 0
        while v.count("voice_personal", lang) < goal and i < goal * 4:
            i += 1
            if lang == "en":
                if i > len(pool):
                    break
                q, wording = pool[i - 1]["ask"], ["user: Topical-Chat"]
                if is_about(q):
                    continue
            else:
                q, wording = _q(v, v.rng.choice(PERSONAL_Q_FR), "fr"), ["user: rule"]
            ans = v.rng.choice(PERSONAL[lang])
            wording = wording + ["rule: Cowork's personal line", "record: kind"]
            if v.rng.random() < JOKE_RATE and v.pool("pastlife", lang):
                jt = v.pool("pastlife", lang).take()
                if len(jt["text"]) <= 110:
                    ans, wording = ans + " " + jt["text"], wording + [src(jt), "joke"]
            prompt = bare(q)
            v.add("voice_personal", lang, split_of(prompt + ans), prompt, ans, wording)


def emoji_rows(v: Voice, user_lines: list[dict]) -> None:
    tc = [r for r in user_lines if r["split"] == "train" and r["kind"] in ("hello", "bye") and r["standalone"]
          and not r.get("sensitive") and 2 <= r["words"] <= 12]
    kinds = sorted(EMOJI_BY_KIND)
    for lang in ("en", "fr"):
        goal, i = TARGET_V3B[f"voice_emoji:{lang}"] * v.scale, 0
        while v.count("voice_emoji", lang) < goal and i < goal * 4:
            i += 1
            if lang == "en" and tc and v.rng.random() < 0.4:
                r = v.rng.choice(tc)
                kind, says = _tc_kind(r), r["text"]
            else:
                kind = kinds[i % len(kinds)]
                says = v.rng.choice(SAYS[lang].get(kind) or [""])
            if not kind or not says or not v.pool(kind, lang):
                continue
            t = v.pool(kind, lang).take()
            mark, back = v.rng.choice(EMOJI_BY_KIND[kind]), v.rng.choice(EMOJI_BY_KIND[kind])
            says = says.rstrip() + " " + mark
            ans = EMOJI.sub("", t["text"]).rstrip() + " " + back
            prompt = bare(says)
            v.add("voice_emoji", lang, split_of(prompt + ans), prompt, ans, [src(t), "rule: emoji, the user's own first"])


# --- the gate ---------------------------------------------------------------------------------------------------
def user_turn(prompt: str) -> str:
    """The person's last line: the Question line of a talk prompt."""
    q = prompt.rpartition("\nQuestion: ")[2] if "\nQuestion: " in prompt else prompt.partition("Question: ")[2]
    return q.rpartition("\nAnswer:")[0] or q


def strip_emoji(text: str) -> str:
    out = EMOJI.sub("", text)
    out = re.sub(r"[ \t]+([.,!?;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return "\n".join(ln.rstrip() for ln in out.split("\n")).strip()


@contextlib.contextmanager
def _h_on():
    before = os.environ.get("CB_CHAT_ABILITIES")
    os.environ["CB_CHAT_ABILITIES"] = "1"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("CB_CHAT_ABILITIES", None)
        else:
            os.environ["CB_CHAT_ABILITIES"] = before


class Gate:
    """The host's checks over a mix row. `check(row)` -> (row or None, reason)."""

    def __init__(self) -> None:
        from cubbyllm.speech import chat_checks, cubby_record
        self.loops, self.abilities = chat_checks.loops, chat_checks.abilities
        self.self_claims, self.chat_check = cubby_record.self_claims, cubby_record.chat_check
        self.stats = collections.Counter()

    def check(self, row: dict) -> tuple[dict | None, str]:
        fam, ans, prompt = row["family"], row["answer"], row["prompt"]
        if EMOJI.search(ans) and not EMOJI.search(prompt):
            ans = strip_emoji(ans)
            self.stats[f"emoji removed:{fam}"] += 1
            if not ans:
                return None, "emoji only"
            row = {**row, "answer": ans}
        why = self.loops(ans)
        if why:
            return None, "g: loop"
        if "joke" in row.get("wording", ()):              # Nick, 2026-11-20: a joke that says a thing twice goes
            words = collections.Counter(w for w in re.findall(r"[a-zà-ÿ']{5,}", ans.lower()))
            if any(n >= 2 for n in words.values()):
                return None, "joke repeats a word"
        if fam in CHAT_SHAPE or fam in CHAT_ROWS:
            bad = self.abilities(ans)
            if bad:
                return None, "h: ability"
            with _h_on():                                 # h on: d's own list of abilities gives way to h, as served
                bad = self.self_claims(ans)
            if bad:
                return None, "d: self claim"
        if fam in CHAT_SHAPE:
            ok, why = self.chat_check(ans, user_turn(prompt))
            if not ok:
                return None, "d: " + why.split(":")[0]
        if fam.startswith("voice_"):
            p = parse(prompt[prompt.find("Facts:"):]) if "Facts:" in prompt else None
            if p:
                for e in {e for e, _, _ in p[0] if len(e) >= 3}:
                    rest = ans
                    for _, _, x in p[0]:                 # a value that holds the name is the fact, not a repeat
                        if e in x and x != e:
                            rest = rest.replace(x, " ")
                    if rest.count(e) >= 3:
                        return None, "the same entity three times"
        return row, "ok"

    def run(self, rows, label: str):
        kept = []
        for r in rows:
            out, why = self.check(r)
            self.stats[f"{label}:{r['family']}:{why}"] += 1
            if out is not None:
                kept.append(out)
        return kept
