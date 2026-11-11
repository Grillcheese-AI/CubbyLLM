"""talk_voice_self -- the voice rows about Cubby and the social turns: who it is, hello by name, thanks, goodbye.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

  voice_identity        a question about Cubby -> Nick's approved answer (`identity_facts.yaml`), asked many ways
  voice_identity_facts  Cubby's record handed over as a facts block -> the answer says what the block says; some
                        blocks differ from today's record (other languages, fewer abilities), so the record is
                        followed, not recited, and can change without a new adapter
  voice_hello           the person gives a name -> a greeting by name, and what Cubby can do when asked
  voice_social          hello, thanks, sorry, encouragement, goodbye; a pun or the past-life joke when asked for
                        one, and after a hello, a cheer or a goodbye at `JOKE_RATE`
"""
from __future__ import annotations

from validation.sft.talk_voice_inputs import (GREETED, HELLO, ID_PREFIX, LANG_NAME, LANG_ORDER, NAMES, RECORD_LABEL,
                                              RECORD_Q, SAYS)
from validation.sft.talk_voice_rows import JOKE_RATE, TARGET, Voice, src
from validation.sft.talk_voice_text import VOWEL, after, bare, cap, fact_lines, fill_list, join, render, split_of

__wiring__ = "STANDALONE"


def identity_rows(v: Voice, qa: list[dict]) -> None:
    per_group = max(4, int(TARGET["voice_identity"] * v.scale / max(1, len(qa))))
    for g in qa:
        lang = g.get("lang", "en")
        variants = set()
        for q in g["q"]:
            for pre in ID_PREFIX[lang]:
                for form in (q, cap(q), q.rstrip(" ?").rstrip("?")):
                    variants.add((pre + form) if pre else form)
        for q in v.rng.sample(sorted(variants), min(per_group, len(variants))):
            ans, wording = g["a"].format(**v.strs), ["identity: Nick's approved answer"]
            if "abilities_text" in g["a"] and v.pool("about", lang) and v.rng.random() < 0.5:
                t = v.pool("about", lang).take()
                ans = fill_list(t["text"], "abilities", v.abilities(lang), True, lang)
                wording = [src(t), "identity: the record's abilities"]
            turns = [("User", GREETED[lang][0]), ("Assistant", GREETED[lang][1])]
            prompt = after(turns, q) if v.rng.random() < 0.25 else bare(q)
            v.add("voice_identity", lang, split_of(prompt + ans), prompt, ans, wording)


def _languages(today: list[str], nxt: list[str], lang: str) -> str:
    t, n = [LANG_NAME[lang][x] for x in today], [LANG_NAME[lang][x] for x in nxt]
    if lang == "en":
        return f"{join(t, lang)} today." + (f" {join(n, lang)} {'is' if len(n) == 1 else 'are'} next." if n else "")

    def art(name):
        return ("l'" if name[0] in VOWEL else "le ") + name
    out = cap(join([art(x) for x in t], lang)) + " aujourd'hui."
    if n:
        out += f" {cap(join([art(x) for x in n], lang))} {'vient' if len(n) == 1 else 'viennent'} ensuite."
    return out


def record_rows(v: Voice) -> None:
    ident, name = v.ident, v.ident["name"]
    who = {"en": "I was built by {creator}.", "fr": "J'ai été créé par {creator}."}
    for i in range(int(TARGET["voice_identity_facts"] * v.scale * 1.05)):
        lang = "en" if i % 2 == 0 else "fr"
        L = RECORD_LABEL[lang]
        real = v.rng.random() < 0.6                       # today's record, or one that differs from it
        k = 2 if real else v.rng.randint(1, 4)
        today = LANG_ORDER[lang][:k]
        nxt = LANG_ORDER[lang][k:] if real or v.rng.random() < 0.7 else []
        abil = ident["abilities" if lang == "en" else "abilities_fr"]
        abil = abil if real else v.rng.sample(abil, v.rng.randint(2, 4))
        block = [(name, L["creator"], ident["creator"]), (name, L["role"], ident["role" if lang == "en" else "role_fr"])]
        block += [(name, L["lang"], LANG_NAME[lang][x]) for x in today] + [(name, L["next"], LANG_NAME[lang][x]) for x in nxt]
        block += [(name, L["can"], x["text"]) for x in abil]
        v.rng.shuffle(block)
        kind = ("lang", "can", "who")[i % 3]
        if kind == "lang":
            ans, wording = _languages(today, nxt, lang), ["rule", "identity: the record's languages"]
        elif kind == "can":
            if not v.pool("about", lang):
                continue
            t = v.pool("about", lang).take()
            ans = fill_list(t["text"], "abilities", v.abilities(lang, abil, labels=False), True, lang)
            wording = [src(t), "identity: the record's abilities"]
        else:
            ans, wording = who[lang].format(creator=ident["creator"]), ["rule", "identity: the record"]
        prompt = render(block, v.rng.choice(RECORD_Q[kind][lang]))
        v.add("voice_identity_facts", lang, split_of(prompt + ans), prompt, ans, wording,
              facts=fact_lines(block) + v.record_facts, entity=name)


def hello_rows(v: Voice) -> None:
    for i in range(int(TARGET["voice_hello"] * v.scale * 1.05)):
        lang = "en" if i % 2 == 0 else "fr"
        if not v.pool("hello_name", lang):
            continue
        says, asks = v.rng.choice(HELLO[lang])
        raw = v.rng.choice(NAMES)
        shown = "-".join(cap(p) for p in raw.split("-"))
        t = v.pool("hello_name", lang).take()
        for _ in range(8):                             # before the list of abilities, a hello that asks nothing
            if not asks or not t["text"].rstrip().endswith("?"):
                break
            t = v.pool("hello_name", lang).take()
        ans, wording = t["text"].replace("{user}", shown), [src(t)]
        if asks and v.pool("about", lang):
            t2 = v.pool("about", lang).take()
            ans += "\n\n" + fill_list(t2["text"], "abilities", v.abilities(lang), True, lang)
            wording += [src(t2), "identity: the record's abilities"]
        prompt = bare(says.replace("{u}", raw if v.rng.random() < 0.5 else shown))
        v.add("voice_hello", lang, split_of(prompt + ans), prompt, ans, wording)


def social_rows(v: Voice, recent: list) -> None:
    """`recent`: (row, triples, question) of English fact rows, so a thanks can follow an answer."""
    kinds = sorted({k for (k, _) in v.pools if k in SAYS["en"]})
    # a kind is drawn in proportion to how many picked lines it has, so no line is leaned on; the jokes are capped
    weights = {lg: [min(len(v.pool(k, lg)), 40 if k in ("pun", "pastlife") else 10 ** 6) for k in kinds]
               for lg in ("en", "fr")}
    goal, tries = int(TARGET["voice_social"] * v.scale * 1.04), 0
    while sum(v.count("voice_social", lg, s) for lg in ("en", "fr") for s in ("train", "held")) < goal \
            and tries < goal * 4:                      # a pair already written is not written again
        lang = "en" if tries % 2 == 0 else "fr"
        tries += 1
        kind = v.rng.choices(kinds, weights[lang])[0]
        if not v.pool(kind, lang):
            continue
        t = v.pool(kind, lang).take()
        says = v.rng.choice(SAYS[lang][kind])
        says = v.rng.choice([says, cap(says), says + ("!" if says[-1].isalnum() and v.rng.random() < 0.3 else "")])
        ans, wording = t["text"], [src(t)]
        if kind in ("greet", "bye", "cheer_praise") and v.rng.random() < JOKE_RATE:
            joke = v.pool(v.rng.choice(["pun", "pastlife"]), lang)
            if joke:
                jt = joke.take()
                if len(jt["text"]) <= 110:
                    ans, wording = ans + " " + jt["text"], wording + [src(jt), "joke"]
        prompt = bare(says)
        if lang == "en" and kind in ("small_welcome", "small_thanks", "small_sorry") and recent and v.rng.random() < 0.5:
            pr, _, pq = v.rng.choice(recent)
            prompt = after([("User", pq), ("Assistant", pr["answer"])], says)
        v.add("voice_social", lang, split_of(prompt + ans), prompt, ans, wording)
