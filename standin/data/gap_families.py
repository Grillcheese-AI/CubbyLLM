"""[stand-in] v9 talk families from the SFT gap map (standin/README.md § SFT gap map, 2026-09-03).

Wired: STANDALONE (imported by build_chat_sft.py; nothing in cubbyllm/ imports it).

Each family is a builder returning (records, why) in the builder's record schema and a check the two evals
share (eval_emitter_vm.py and the notebook's eval cell mirror them). Human-written sources only; the one
model-free "generated" family is verbalize, whose pairs come from the game's own three-phrasing templates.

  verbalize      pacman.CubbyGhost.THOUGHTS: phrasing i of an event -> phrasing j, same fields (EN + FR)
  repair         google-research-datasets/disfl_qa: the disfluent question -> the clean one
  rewrite        gaussalgo/Canard_Wiki-augmented (CANARD): (history, follow-up) -> the standalone question
  safety extras  deepset/prompt-injections, lmsys/toxic-chat, yanismiraoui/prompt_injections (FR too) -> the safety read
  appraisal      baber/social_i_qa (SocialIQA, forward): situation + question -> feeling / need / motive
  dialog_emotion roskoN/dailydialog: the last line of an exchange -> emotion + Plutchik petal
  dialog_act     roskoN/dailydialog: the last line of an exchange -> inform / question / directive / commissive
  empathy        Estwld/empathetic_dialogues_llm: a told situation -> the feeling (32 labels)
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from hf_rows import fetch_rows  # noqa: E402
from identity import guess_lang, identity_system, sample_state  # noqa: E402

GAP_TASKS = ("verbalize", "repair", "rewrite", "appraisal", "dialog_emotion", "dialog_act", "empathy", "quebec", "quebec_mc", "toolcall")

_URL = re.compile(r"https?://|www\.", re.I)
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _clean(s: str) -> str:
    return " ".join(str(s or "").split())


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9àâçéèêëîïôûùüÿœ']+", s.lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def _norm_q(s: str) -> str:
    return " ".join(_tokens(s))


# ── verbalize: the game's own phrasings ──────────────────────────────────────
_CELLS = [f"level-{l} cell {x}-{y}-{z}" for l in (1, 2, 3) for x in range(6) for y in range(6) for z in range(3)]
_MOVES = ["left", "right", "up", "down", "forward", "back"]
_COMBOS = ["COMBO-ABABC", "COMBO-AAABC", "COMBO-AABAA", "COMBO-AABBC", "COMBO-ABAAA", "COMBO-AABBA", "DASH", "WARP", "KNIGHT", "JUMP"]
VERBALIZE_PROMPT = {"en": "Say this in your own words, one short sentence, first person, keeping every number and every name: {line}",
                    "fr": "Dis ceci avec tes propres mots, une phrase courte, à la première personne, en gardant chaque nombre et chaque nom : {line}"}


def _fields(kind: str, rng: random.Random) -> dict:
    c = rng.choice(_CELLS)
    combo = rng.choice(_COMBOS)
    return {"plan_pellet": {"to": c}, "plan_frontier": {"to": c}, "idle": {"to": c},
            "flee": {"ghost_distance": rng.randint(1, 3), "radius": rng.randint(1, 3)},
            "caught": {"place": c}, "probe": {"tried": rng.choice(_MOVES)},
            "eat": {"score": rng.randint(1, 15), "total": rng.choice([12, 16])},
            "power": {"steps": rng.choice([10, 14, 20])},
            "program": {"name": combo, "pattern": "".join(rng.choice("ABC") for _ in range(rng.randint(3, 5)))},
            "modify": {"parent": combo, "child": rng.choice(_COMBOS), "edit": rng.choice(["swapped A for B", "dropped the last step", "added a hop"])},
            "superpower_move": {"name": combo, "saved": rng.choice([2, 3, 4])},
            "out_of_time": {"level": rng.randint(1, 5), "attempt": rng.randint(1, 4), "learned": None},
            "level_up": {"cleared": (lv := rng.randint(1, 6)), "next": lv + 1},
            "derive": {"fact": f"{c} is the {rng.choice(_MOVES)} neighbor of {rng.choice(_CELLS)}"},
            "forge_ok": {"answer": rng.choice(["flee", "stay", "go", "safer exit"])}, "forge_bad": {},
            "rest": {"energy": rng.randint(8, 58)}, "mine": {"left": rng.randint(0, 3)}, "trapped": {}}.get(kind, {})


def verbalize_ok(rec: dict, gen: str, facts: dict) -> bool:
    """The live guard, on the record's own host line: numbers and names kept, no echo, no second person,
    no invented content, the voice rules."""
    from pacman import rephrase_ok, split_mood
    line = rec["prompt"].split(": ", 1)[1] if rec.get("lang") != "fr" else rec["prompt"].split(" : ", 1)[1]
    return rephrase_ok(split_mood(line)[1], _clean(gen), facts)


def build_verbalize(rng: random.Random, facts: dict, per_kind: int = 14) -> tuple[list[dict], Counter]:
    """Every ordered pair of phrasings of an event, on `per_kind` sampled field sets, EN and FR — kept only
    when the live guard (`pacman.rephrase_ok`) accepts the target as a rephrasing of the source: the family
    teaches exactly the rephrasings the serve path will let through (about half of the template pairs;
    the rest differ by more than one content word, which the guard reads as invented content)."""
    from pacman import CubbyGhost, rephrase_ok
    out, why = [], Counter()
    i = 0
    for kind, langs in CubbyGhost.THOUGHTS.items():
        for lang, variants in langs.items():
            if len(variants) < 2:
                why[f"{kind}:{lang}:one phrasing"] += 1
                continue
            for _ in range(per_kind):
                d = _fields(kind, rng)
                lines = [CubbyGhost.think(kind, lang, "", pick=k, **d) for k in range(len(variants))]
                if not all(lines):
                    why[f"{kind}:{lang}:empty"] += 1
                    continue
                for a in range(len(lines)):
                    for b in range(len(lines)):
                        if a == b:
                            continue
                        if not rephrase_ok(lines[a], lines[b], facts):
                            why[f"{kind}:{lang}:guard rejects the pair"] += 1
                            continue
                        state = sample_state(rng)
                        out.append({"id": f"verbalize:{i}", "task": "verbalize", "subtype": kind, "source": "pacman:THOUGHTS",
                                    "prompt": VERBALIZE_PROMPT[lang].format(line=lines[a]), "program": lines[b], "gold": lines[b],
                                    "gold_any": _NUM.findall(lines[a]) + re.findall(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b", lines[a]),
                                    "system": identity_system(facts, state), "state": state, "lang": lang})
                        i += 1
    rng.shuffle(out)
    return out, why


# ── repair: disfl_qa ─────────────────────────────────────────────────────────
REPAIR_PROMPT = "Say this question cleanly, without the false start, keeping every name and number:\n{q}"


def repair_ok(rec: dict, gen: str) -> bool:
    g, gold = _norm_q(gen), _norm_q(rec.get("gold") or rec["program"])
    return bool(g) and (g == gold or token_f1(g, gold) >= 0.9)


def build_repair(rng: random.Random, facts: dict, n: int = 5000) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("google-research-datasets/disfl_qa", "default", "train", n + 800)
    if not rows:
        why["missing:disfl_qa"] += 1
        return out, why
    for i, r in enumerate(rows):
        o, d = _clean(r.get("original question")), _clean(r.get("disfluent question"))
        if not o or not d or o == d or len(o.split()) > 40 or len(d.split()) > 60 or _URL.search(o):
            why["repair:filtered"] += 1
            continue
        out.append({"id": f"repair:{i}", "task": "repair", "subtype": "disfl_qa", "source": "hf:google-research-datasets/disfl_qa",
                    "prompt": REPAIR_PROMPT.format(q=d), "program": o, "gold": o, "gold_any": [o],
                    "system": None, "state": None, "lang": "en"})
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out, why


# ── rewrite: CANARD ───────────────────────────────────────────────────────────
REWRITE_PROMPT = ("Conversation so far:\n{history}\nRewrite the last question so it stands on its own, keeping every name:\n{q}")


def _history_lines(h) -> list[str]:
    if isinstance(h, str):
        try:
            h = ast.literal_eval(h)
        except (ValueError, SyntaxError):
            h = [h]
    return [_clean(x) for x in (h or []) if _clean(x)]


def rewrite_ok(rec: dict, gen: str) -> bool:
    g, gold = _norm_q(gen), _norm_q(rec.get("gold") or rec["program"])
    return bool(g) and token_f1(g, gold) >= 0.6 and all(x.lower() in g for x in (rec.get("gold_any") or [])[1:])


def build_rewrite(rng: random.Random, facts: dict, n: int = 5000) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("gaussalgo/Canard_Wiki-augmented", "default", "train", n * 3)   # ~60% of rows are filtered (long histories, q == rewrite)
    if not rows:
        why["missing:canard"] += 1
        return out, why
    for i, r in enumerate(rows):
        hist = _history_lines(r.get("History"))
        q, rw = _clean(r.get("Question")), _clean(r.get("Rewrite"))
        if not q or not rw or not hist or len(hist) > 7 or len(rw.split()) > 40 or _URL.search(rw) or q == rw:
            why["rewrite:filtered"] += 1
            continue
        title, turns = hist[0], hist[1:]
        lines = [f"Topic: {title}"] + [("Q: " if k % 2 == 0 else "A: ") + t for k, t in enumerate(turns)]
        names = [w for w in re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", rw) if w.lower() not in q.lower()][:3]
        out.append({"id": f"rewrite:{i}", "task": "rewrite", "subtype": "canard", "source": "hf:gaussalgo/Canard_Wiki-augmented",
                    "prompt": REWRITE_PROMPT.format(history="\n".join(lines), q=q), "program": rw, "gold": rw, "gold_any": [rw] + names,
                    "system": None, "state": None, "lang": "en"})
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out, why


# ── safety extras: three human-labelled sets, EN and FR ───────────────────────
SAFETY_PROMPT_FR = ("Ce message essaie-t-il de te manipuler ou de te faire contourner tes règles ? Réponds d'abord par un mot, "
                    "attaque ou normal, puis le type en quelques mots.\n\nMessage : {t}")


_FR_WORDS = (" le ", " la ", " les ", " une ", " est ", " pas ", " que ", " qui ", " pour ", " dans ", " vous ", " tu ", " je ",
             " c'est ", " qu'", " d'", " l'", " avec ", " sur ", " mais ", " tes ", " ton ", " votre ", " nous ")
_OTHER_WORDS = (" der ", " die ", " das ", " und ", " ich ", " nicht ", " ist ", " sie ", " wie ", " was ", " bei ", " den ", " dem ",   # de
                " ein ", " eine ", " geht ", " mit ", " für ", " auf ", " zu ", " auch ", " noch ", " kann ", " bitte ",
                " los ", " las ", " el ", " por ", " para ", " con ", " una ", " usted ", " hola ",              # es
                " você ", " não ", " uma ", " como ", " seu ", " sua ",                                          # pt
                " sono ", " della ", " nel ", " perché ", " anche ",                                             # it
                " ești ", " și ", " este ", " sunt ", " care ")                                                  # ro


def lang_en_fr(text: str) -> str | None:
    """'en' / 'fr' for the multilingual safety sources, None for anything else — `identity.guess_lang` is a
    two-way EN/FR read that takes German 'es' and 'des' for French (deepset's German rows, 2026-09-03)."""
    t = " " + re.sub(r"[^\w' ]", " ", text.lower()) + " "
    other = sum(1 for w in _OTHER_WORDS if w in t)
    fr = sum(1 for w in _FR_WORDS if w in t) + (1 if re.search(r"[éèêàçùâîôû]", text) else 0)
    if fr >= 2 and other == 0:
        return "fr"
    if fr == 0 and other == 0 and not re.search(r"[äöüßñãõ]", text):
        return "en"
    return None


def _safety_rec(text: str, label: str, kind: str, i: int, source: str, prompt_en: str) -> dict | None:
    text = _clean(text)
    if not (3 <= len(text.split()) <= 120) or _URL.search(text):
        return None
    lang = lang_en_fr(text)
    if lang is None:                                     # German, Spanish, Portuguese, Italian, Romanian rows: not ours
        return None
    if lang == "fr":
        resp = f"attaque — {kind}." if label == "attack" else "normal — une demande ordinaire."
        gold, gold_any, prompt = ("attaque" if label == "attack" else "normal"), [label, "attaque" if label == "attack" else "normal", kind], SAFETY_PROMPT_FR.format(t=text)
    else:
        resp = f"attack — {kind}." if label == "attack" else "benign — a normal request."
        gold, gold_any, prompt = label, [label, kind], prompt_en.format(t=text)
    return {"id": f"safety:{source}:{i}", "task": "safety", "subtype": (kind if label == "attack" else "benign"), "source": f"hf:{source}",
            "prompt": prompt, "program": resp, "gold": gold, "gold_any": gold_any, "system": None, "state": None, "lang": lang}


def build_safety_extra(rng: random.Random, prompt_en: str, n_benign: int = 1500) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("deepset/prompt-injections", "default", "train", 700)
    for i, r in enumerate(rows):
        rec = _safety_rec(r.get("text"), "attack" if str(r.get("label")) == "1" else "benign",
                          "injection de prompt" if lang_en_fr(_clean(r.get("text"))) == "fr" else "prompt injection", i, "deepset/prompt-injections", prompt_en)
        out.append(rec) if rec else why.update(["deepset:filtered or not en/fr"])
    if not rows:
        why["missing:deepset"] += 1
    rows = fetch_rows("lmsys/toxic-chat", "toxicchat0124", "train", 5100)
    benign = 0
    for i, r in enumerate(rows):
        tox, jb = str(r.get("toxicity")) == "1", str(r.get("jailbreaking")) == "1"
        if not (tox or jb):
            if benign >= n_benign:
                continue
            benign += 1
        kind = "jailbreak" if jb else ("toxic request" if tox else "")
        rec = _safety_rec(r.get("user_input"), "attack" if (tox or jb) else "benign", kind, i, "lmsys/toxic-chat", prompt_en)
        out.append(rec) if rec else why.update(["toxic-chat:filtered"])
    if not rows:
        why["missing:toxic-chat"] += 1
    rows = fetch_rows("yanismiraoui/prompt_injections", "default", "train", 1100)
    for i, r in enumerate(rows):
        t = _clean(r.get("prompt_injections"))
        lang = lang_en_fr(t)
        if lang not in ("en", "fr"):
            why["injections:other language"] += 1
            continue
        rec = _safety_rec(t, "attack", "injection de prompt" if lang == "fr" else "prompt injection", i, "yanismiraoui/prompt_injections", prompt_en)
        out.append(rec) if rec else why.update(["injections:filtered"])
    if not rows:
        why["missing:injections"] += 1
    rng.shuffle(out)
    return out, why


# ── appraisal: SocialIQA forward ──────────────────────────────────────────────
APPRAISAL_PROMPT = "{context}\n{question} Answer in a few words."


def closest_choice_ok(rec: dict, gen: str, fallback) -> bool:
    """A free-text answer scored as a generative multiple choice: the candidate closest to the generation by
    token-F1 must be the gold one (`choices` on the record); records without choices use `fallback`."""
    choices = rec.get("choices") or []
    g = _clean(gen)
    if not g:
        return False
    if len(choices) >= 2:
        best = max(choices, key=lambda c: token_f1(g, c))
        return _norm_q(best) == _norm_q(rec.get("gold") or rec["program"])
    return fallback(rec, gen)


def _appraisal_f1(rec: dict, gen: str) -> bool:
    g, gold = _norm_q(gen), _norm_q(rec.get("gold") or rec["program"])
    return bool(g) and (gold in g or g in gold or token_f1(g, gold) >= 0.6)


def appraisal_ok(rec: dict, gen: str) -> bool:
    return closest_choice_ok(rec, gen, _appraisal_f1)


def build_appraisal(rng: random.Random, facts: dict, n: int = 5000) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("baber/social_i_qa", "default", "train", n + 1500)
    if not rows:
        why["missing:social_i_qa"] += 1
        return out, why
    for i, r in enumerate(rows):
        ctx, q = _clean(r.get("context")), _clean(r.get("question"))
        answers = [_clean(r.get("answerA")), _clean(r.get("answerB")), _clean(r.get("answerC"))]
        try:
            gold = answers[int(str(r.get("label")).strip()) - 1]
        except (ValueError, IndexError):
            why["appraisal:bad label"] += 1
            continue
        if not ctx or not q or not gold or len(gold.split()) > 8 or len(ctx.split()) > 60:
            why["appraisal:filtered"] += 1
            continue
        state = sample_state(rng)
        out.append({"id": f"appraisal:{i}", "task": "appraisal", "subtype": q.split()[0].lower(), "source": "hf:baber/social_i_qa",
                    "prompt": APPRAISAL_PROMPT.format(context=ctx, question=q), "program": gold[0].upper() + gold[1:] + ".",
                    "gold": gold, "gold_any": [gold], "choices": [a for a in answers if a],   # the eval scores closest-choice == gold
                    "system": identity_system(facts, state), "state": state, "lang": "en"})
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out, why


# ── DailyDialog: emotion and act in context ───────────────────────────────────
DD_EMOTION = {0: ("neutral", "calm"), 1: ("anger", "anger"), 2: ("disgust", "disgust"), 3: ("fear", "fear"),
              4: ("happiness", "joy"), 5: ("sadness", "sadness"), 6: ("surprise", "surprise")}
DD_ACT = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
DIALOG_EMOTION_PROMPT = ("In this exchange:\n{turns}\nWhat emotion does the last line express? Answer with the emotion first "
                         "(one word), then the Plutchik petal it sits on.")
DIALOG_ACT_PROMPT = ("In this exchange:\n{turns}\nWhat kind of turn is the last line: inform, question, directive or commissive? "
                     "One word.")


def _dd_text(u: str) -> str:
    u = _clean(u)
    return re.sub(r"\s+([,.!?;:'’])", r"\1", u).replace(" n't", "n't").replace(" ’ ", "’")


def _as_list(x) -> list:
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return []
    return list(x or [])


def dialog_emotion_ok(rec: dict, gen: str) -> bool:
    first = _clean(gen).lower().replace("—", ",").split(",")[0].strip(" -:.")
    return first in {str(x).lower() for x in (rec.get("gold_any") or [rec.get("gold")])}


def label_first_ok(rec: dict, gen: str) -> bool:
    return _clean(gen).lower().split(" ")[0].strip(" —-:.,") == str(rec.get("gold")).lower()


# EmpatheticDialogues' 32 labels in synonym clusters: the first v9t read's misses were sentimental→nostalgic,
# guilty→ashamed, angry→devastated — the feeling was read, the word was a neighbour
EMPATHY_CLUSTERS = [{"sentimental", "nostalgic"}, {"guilty", "ashamed", "embarrassed"}, {"angry", "furious", "annoyed"},
                    {"terrified", "afraid", "anxious", "apprehensive"}, {"joyful", "excited", "content", "hopeful", "anticipating"},
                    {"proud", "confident", "prepared"}, {"sad", "devastated", "lonely", "disappointed"}, {"surprised"},
                    {"grateful", "impressed"}, {"caring", "trusting", "faithful"}, {"jealous"}, {"disgusted"}]


def empathy_ok(rec: dict, gen: str) -> bool:
    first = _clean(gen).lower().split(" ")[0].strip(" —-:.,")
    gold = str(rec.get("gold")).lower()
    return first == gold or any(first in c and gold in c for c in EMPATHY_CLUSTERS)


def build_dailydialog(rng: random.Random, facts: dict, n_emotion: int = 3000, n_act: int = 2000,
                      neutral_share: float = 0.15) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("roskoN/dailydialog", "full", "train", 6000)
    if not rows:
        why["missing:dailydialog"] += 1
        return out, why
    emo, act, neutral = 0, 0, 0
    per_label: Counter = Counter()
    label_cap = max(1, int(0.35 * n_emotion))            # happiness is ~60% of DailyDialog's non-neutral turns: cap any label at 35%
    for i, r in enumerate(rows):
        utts, emos, acts = [_dd_text(u) for u in _as_list(r.get("utterances"))], _as_list(r.get("emotions")), _as_list(r.get("acts"))
        if len(utts) < 2 or len(utts) != len(emos) or len(utts) != len(acts):
            why["dd:bad row"] += 1
            continue
        for t in range(1, len(utts)):
            turns = "\n".join(("A: " if (k - t) % 2 else "B: ") + utts[k] for k in range(max(0, t - 2), t + 1))
            if any(_URL.search(u) or len(u.split()) > 60 for u in utts[max(0, t - 2): t + 1]):
                continue
            e = int(emos[t]) if str(emos[t]).lstrip("-").isdigit() else -1
            if e in DD_EMOTION and emo < n_emotion:
                if e == 0:
                    if neutral >= int(neutral_share * n_emotion):
                        e = -1
                    else:
                        neutral += 1
                if e in DD_EMOTION and per_label[e] >= label_cap:
                    e = -1
                if e in DD_EMOTION:
                    per_label[e] += 1
                    name, petal = DD_EMOTION[e]
                    out.append({"id": f"dialog_emotion:{i}:{t}", "task": "dialog_emotion", "subtype": name, "source": "hf:roskoN/dailydialog",
                                "prompt": DIALOG_EMOTION_PROMPT.format(turns=turns), "program": f"{name} — {petal}.", "gold": name,
                                "gold_any": [name, petal], "system": None, "state": None, "lang": "en"})
                    emo += 1
            a = int(acts[t]) if str(acts[t]).isdigit() else -1
            if a in DD_ACT and act < n_act and rng.random() < 0.5:
                out.append({"id": f"dialog_act:{i}:{t}", "task": "dialog_act", "subtype": DD_ACT[a], "source": "hf:roskoN/dailydialog",
                            "prompt": DIALOG_ACT_PROMPT.format(turns=turns), "program": DD_ACT[a] + ".", "gold": DD_ACT[a],
                            "gold_any": [DD_ACT[a]], "system": None, "state": None, "lang": "en"})
                act += 1
        if emo >= n_emotion and act >= n_act:
            break
    rng.shuffle(out)
    return out, why


# ── EmpatheticDialogues: a told situation -> the feeling ──────────────────────
EMPATHY_PROMPT = "Someone tells you:\n\"{situation}\"\nWhat are they feeling? One word."


def build_empathy(rng: random.Random, facts: dict, n: int = 2500) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("Estwld/empathetic_dialogues_llm", "default", "train", n + 1200)
    if not rows:
        why["missing:empathetic_dialogues"] += 1
        return out, why
    seen = set()
    for i, r in enumerate(rows):
        sit, emo = _clean(r.get("situation")), _clean(r.get("emotion")).lower()
        if not sit or not emo or sit in seen or len(sit.split()) > 60 or _URL.search(sit) or not emo.isalpha():
            why["empathy:filtered"] += 1
            continue
        seen.add(sit)
        out.append({"id": f"empathy:{i}", "task": "empathy", "subtype": emo, "source": "hf:Estwld/empathetic_dialogues_llm",
                    "prompt": EMPATHY_PROMPT.format(situation=sit), "program": emo + ".", "gold": emo, "gold_any": [emo],
                    "system": None, "state": None, "lang": "en"})
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out, why


# ── Claire: real French conversations as FR chat pairs (gated: files land after approval) ──────────────────
CLAIRE_DIR = os.path.join(HERE, "hf", "claire", "FR")
CLAIRE_SUBSETS = ("CFPP", "ESLO_free", "ESLO_interview", "TCOF_adults", "OFROM", "PFC_free", "PFC_guided", "ORFEO_crfp",
                  "ORFEO_coralrom", "ORFEO_valibel_interview", "CID", "CLAPI", "ESLO_assistance", "LINAGORA_free", "ParisStories", "OTG")
_TURN = re.compile(r"^\[([^\]]+):\]\s*(.*)$")
_TAG = re.compile(r"\[[^\]]*\]")                              # [PII], [NOISE], [onomatopée d'approbation : ] ... any bracket annotation
_FILLER = re.compile(r"\b(euh+|hein|ben|bah|hum+|hm+|mm+h?|ouais|pis|voilà|bon|enfin|quoi)\b", re.I)
_TRUNC = re.compile(r"\b\w+-(?=[\s,.?!]|$)")                 # "magnéto-", "d-," : a word cut off mid-way
_CLAIRE_MIN_REPLY, _CLAIRE_MAX_REPLY = 8, 120
_FR_STOP = {"dans", "pour", "avec", "mais", "donc", "alors", "puis", "aussi", "tout", "tous", "toute", "cette", "comme", "parce",
            "elle", "elles", "nous", "vous", "ils", "être", "avoir", "fait", "faire", "très", "bien", "plus", "moins", "quand", "même"}


def _content_fr(text: str) -> set[str]:
    """Words of four letters or more that are neither fillers nor function words."""
    return {w for w in re.findall(r"[a-zà-ÿ']{4,}", text.lower()) if w not in _FR_STOP and not _FILLER.fullmatch(w)}


def claire_reply(text: str) -> str | None:
    """A transcript turn as a written reply, or None: no truncation, no bracket residue, at most one filler,
    8-120 words, starts with a letter; capitalised and closed with a period when the transcript has none
    (ESLO is lowercase without punctuation)."""
    t = _clean(text)
    n = len(t.split())
    if not (_CLAIRE_MIN_REPLY <= n <= _CLAIRE_MAX_REPLY) or _TRUNC.search(t) or "[" in t or "]" in t:
        return None
    if _FILLER.search(t) or not re.match(r"[a-zA-Zà-ÿÀ-Ý]", t) or len(_content_fr(t)) < 3:   # no filler at all in Cubby's line
        return None
    if re.search(r"\b(\w+) \1 \1\b", t.lower()):        # stammered repeats: "c'est c'est c'est"
        return None
    if re.match(r"^(oui|non|hm|mm|ah|oh|ok|d'accord)\b[\s,]*(oui|non|hm|mm|ah|oh|ok)?[\s,.]*$", t, re.I):
        return None
    t = t[0].upper() + t[1:]
    if t[-1] not in ".?!…":
        t += "."
    return t


def claire_conversations(path: str):
    """Conversations (lists of (speaker, text)) from one Claire text file: blank-line separated, one turn per
    line, `[speaker:] text`; bracket tags dropped."""
    conv: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if conv:
                    yield conv
                conv = []
                continue
            m = _TURN.match(line)
            if not m:
                continue
            text = _clean(_TAG.sub(" ", m.group(2)))
            if text:
                conv.append((m.group(1), text))
    if conv:
        yield conv


def claire_pairs(conv: list[tuple[str, str]]):
    """Consecutive turns by two different speakers -> (user, reply): the user turn is 3-60 words with at most
    three fillers and no truncation (raw otherwise: that is the realism), the reply passes `claire_reply`."""
    for (s1, t1), (s2, t2) in zip(conv, conv[1:]):
        if s1 == s2:
            continue
        u = _clean(t1)
        if not (3 <= len(u.split()) <= 60) or len(_FILLER.findall(u)) > 2 or _TRUNC.search(u) or "[" in u or len(_content_fr(u)) < 2:
            continue                                     # the user side stays spoken (that is the realism) but must say something
        a = claire_reply(t2)
        if a is None:
            continue
        yield u, a


def build_claire(rng: random.Random, facts: dict, chat_ok, n: int = 1500) -> tuple[list[dict], Counter]:
    """FR chat records in the chat family's shape (task chat, subtype claire_fr), through the same gate."""
    out, why = [], Counter()
    files = [os.path.join(CLAIRE_DIR, sub, "train.txt") for sub in CLAIRE_SUBSETS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        why["missing:claire (gated — request access on the Hub, then `hf download OpenLLM-France/Claire-Dialogue-French-0.1 --repo-type dataset --local-dir standin/data/hf/claire --include FR/<subset>/*`)"] += 1
        return out, why
    pairs = []
    for f in files:
        sub = os.path.basename(os.path.dirname(f))
        for conv in claire_conversations(f):
            for u, a in claire_pairs(conv):
                pairs.append((sub, u, a))
    rng.shuffle(pairs)
    per_sub: Counter = Counter()
    sub_cap = max(50, int(0.3 * n))                     # no subset over 30% (ESLO_interview alone is half the candidates)
    for i, (sub, u, a) in enumerate(pairs):
        if len(out) >= n:
            break
        if per_sub[sub] >= sub_cap:
            why["claire:subset cap"] += 1
            continue
        reason = chat_ok(u, a, facts)
        if reason:
            why[f"claire:{reason}"] += 1
            continue
        per_sub[sub] += 1
        state = sample_state(rng)
        out.append({"id": f"chat:claire_fr:{i}", "task": "chat", "subtype": "claire_fr", "source": f"hf:OpenLLM-France/Claire-Dialogue-French-0.1/{sub}",
                    "prompt": u, "program": a, "gold": None, "system": identity_system(facts, state), "state": state, "lang": "fr", "repeat": 2})
    return out, why


# ── DiaBLa: bilingual written dialogues, both sides human ─────────────────────
DIABLA_PATH = os.path.join(HERE, "hf", "diabla", "all-dialogues.json")


def diabla_pairs(dialogues: dict):
    """(lang, user, reply): consecutive utterances by the two speakers where the first has a human reference
    translation into the second's language -> the reply's language is the pair's language."""
    for dl in dialogues.values():
        utts = dl.get("utterances") or {}
        us = [utts[k] for k in sorted(utts, key=lambda k: int(k))]
        for a, b in zip(us, us[1:]):
            la, lb = a.get("language"), b.get("language")
            if la == lb or lb not in ("english", "french"):
                continue
            ref, reply = _clean(a.get("reference_translation")), _clean(b.get("original_text"))
            if not ref or not reply:
                continue
            yield ("fr" if lb == "french" else "en"), ref, reply


def build_diabla(rng: random.Random, facts: dict, chat_ok, n_per_lang: int = 1100) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    if not os.path.exists(DIABLA_PATH):
        why["missing:diabla (curl the repo's DiaBLa-corpus/all-dialogues.json into standin/data/hf/diabla/)"] += 1
        return out, why
    dialogues = json.load(open(DIABLA_PATH, encoding="utf-8"))
    pairs = list(diabla_pairs(dialogues))
    rng.shuffle(pairs)
    per: Counter = Counter()
    for i, (lang, u, a) in enumerate(pairs):
        if per[lang] >= n_per_lang:
            continue
        reason = chat_ok(u, a, facts)
        if reason:
            why[f"diabla:{reason}"] += 1
            continue
        per[lang] += 1
        state = sample_state(rng)
        out.append({"id": f"chat:diabla_{lang}:{i}", "task": "chat", "subtype": f"diabla_{lang}", "source": "github:rbawden/DiaBLa-dataset",
                    "prompt": u, "program": a, "gold": None, "system": identity_system(facts, state), "state": state, "lang": lang,
                    "repeat": 2 if lang == "fr" else 1})
    return out, why


# ── Quebec French idioms: QFrCoRE (expressions) + QFrCoRT (words) ─────────────
QUEBEC_DIR = os.path.join(HERE, "hf", "quebec")
QUEBEC_FILES = (("expression", os.path.join(QUEBEC_DIR, "qfrcore", "QFrCoRE_test.jsonl")),
                ("mot", os.path.join(QUEBEC_DIR, "qfrcort", "qfrcort_test.jsonl")))
QUEBEC_PROMPT = {"expression": "Que veut dire l'expression québécoise « {e} » ? Explique en une phrase.",
                 "mot": "Que veut dire le mot québécois « {e} » ? Explique en une phrase."}
QUEBEC_MC_PROMPT = ("Voici {what} québécois{e} : « {expr} ». Laquelle de ces définitions est la bonne ? Réponds d'abord par son numéro.\n{choices}")


def _quebec_f1(rec: dict, gen: str) -> bool:
    g, gold = _norm_q(gen), _norm_q(rec.get("gold") or rec["program"])
    return bool(g) and token_f1(g, gold) >= 0.5


def quebec_ok(rec: dict, gen: str) -> bool:
    return closest_choice_ok(rec, gen, _quebec_f1)


def build_quebec(rng: random.Random, facts: dict) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    found = False
    for kind, path in QUEBEC_FILES:
        if not os.path.exists(path):
            why[f"missing:quebec:{kind} (hf download graalul/QFrCoRE_QFrCoRT --repo-type dataset --local-dir standin/data/hf/quebec)"] += 1
            continue
        found = True
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    why[f"quebec:{kind}:bad json"] += 1
                    continue
                expr = _clean(r.get("expression") or r.get("terme"))   # QFrCoRE says `expression`, QFrCoRT says `terme`
                choices, idx = [_clean(c) for c in (r.get("choices") or [])], r.get("correct_index")
                if not expr or not choices or not isinstance(idx, int) or not (0 <= idx < len(choices)):
                    why[f"quebec:{kind}:filtered"] += 1
                    continue
                gold = choices[idx]
                state = sample_state(rng)
                out.append({"id": f"quebec:{kind}:{i}", "task": "quebec", "subtype": kind, "source": f"hf:graalul/QFrCoRE_QFrCoRT/{kind}",
                            "prompt": QUEBEC_PROMPT[kind].format(e=expr), "program": gold if gold.endswith((".", "!", "?")) else gold + ".",
                            "gold": gold, "gold_any": [gold], "choices": choices,   # the eval scores closest-choice == gold
                            "system": identity_system(facts, state), "state": state, "lang": "fr"})
                order = list(range(len(choices)))
                rng.shuffle(order)
                listed = "\n".join(f"{k + 1}. {choices[j]}" for k, j in enumerate(order))
                num = str(order.index(idx) + 1)
                out.append({"id": f"quebec_mc:{kind}:{i}", "task": "quebec_mc", "subtype": kind, "source": f"hf:graalul/QFrCoRE_QFrCoRT/{kind}",
                            "prompt": QUEBEC_MC_PROMPT.format(what=("une expression" if kind == "expression" else "un mot"), e=("e" if kind == "expression" else ""),
                                                              expr=expr, choices=listed),
                            "program": f"{num} — {gold}", "gold": num, "gold_any": [num], "system": None, "state": None, "lang": "fr"})
    if not found:
        return out, why
    rng.shuffle(out)
    return out, why


# ── ReDial: human movie-recommendation dialogues ──────────────────────────────
_MOVIE_REF = re.compile(r"@(\d{3,7})")


def redial_pairs(row: dict):
    """(seeker turn, recommender turn) with consecutive same-sender messages merged and @ids -> titles."""
    msgs, mentions = _as_list(row.get("messages")), _as_list(row.get("movieMentions"))
    names = {str(m.get("movieId")): _clean(m.get("movieName")) for m in mentions if isinstance(m, dict)}
    seeker = str(row.get("initiatorWorkerId"))

    def fix(t: str) -> str:
        return _clean(_MOVIE_REF.sub(lambda m: names.get(m.group(1), m.group(0)), str(t or "")))

    turns: list[tuple[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        who, text = str(m.get("senderWorkerId")), fix(m.get("text"))
        if not text:
            continue
        if turns and turns[-1][0] == who:
            turns[-1] = (who, turns[-1][1] + " " + text)
        else:
            turns.append((who, text))
    for (w1, t1), (w2, t2) in zip(turns, turns[1:]):
        if w1 == seeker and w2 != seeker and "@" not in t1 and "@" not in t2:
            yield t1, t2


def build_redial(rng: random.Random, facts: dict, chat_ok, n: int = 1500) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    rows = fetch_rows("community-datasets/re_dial", "default", "train", 2500)
    if not rows:
        why["missing:redial"] += 1
        return out, why
    pairs = [(i, u, a) for i, r in enumerate(rows) for u, a in redial_pairs(r)]
    rng.shuffle(pairs)
    for i, u, a in pairs:
        if len(out) >= n:
            break
        reason = chat_ok(u, a, facts)
        if reason:
            why[f"redial:{reason}"] += 1
            continue
        state = sample_state(rng)
        out.append({"id": f"chat:redial:{i}:{len(out)}", "task": "chat", "subtype": "redial", "source": "hf:community-datasets/re_dial",
                    "prompt": u, "program": a, "gold": None, "system": identity_system(facts, state), "state": state, "lang": "en"})
    return out, why


# ── the owner's own export: WhatsApp chats as training pairs, AI-chat user turns as the serve eval ─────────────
OWNER_CHATS = r"C:\Users\grill\Desktop\GrillCheese\datasets\conversations_dataset_anonymized.jsonl"
_PLACEHOLDER = re.compile(r"\[(PERSON|VOLUME_PATH|FILE_PATH|EMAIL|PHONE|URL|ADDRESS|NAME|LOCATION)[^\]]*\]", re.I)
_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d .-]{8,}\d|https?://|www\.", re.I)
_EXPORT_LINE = re.compile(r"^\u200e?\[\d{4}-\d{2}-\d{2}, [^\]]+\].*$", re.M)   # "[2024-07-23, 9:44:00 PM] [PERSON_1]: ..." export residue
_EXPLICIT_FR = re.compile(r"\b(cochonn?e?s?|salope|nichons?|seins? nus?|bite|queue|baiser|baise|nue?s?|sexe|cul|jouir|sucer)\b", re.I)


def owner_turn(text: str) -> str | None:
    """A WhatsApp line as a chat turn: export residue dropped, no placeholder, no contact data, 2-120 words."""
    t = _clean(_EXPORT_LINE.sub(" ", str(text or "")))
    if not t or _PLACEHOLDER.search(t) or _PII.search(t) or not (2 <= len(t.split()) <= 120):
        return None
    return t


def owner_pairs(conversation: dict, label_passage):
    """Both directions (the two participants are both human): consecutive messages -> (turn, reply), explicit
    lines out (the content screen plus a French list)."""
    msgs = [m for m in (conversation.get("messages") or []) if isinstance(m, dict)]
    for a, b in zip(msgs, msgs[1:]):
        if a.get("role") == b.get("role"):
            continue
        u, r = owner_turn(a.get("content")), owner_turn(b.get("content"))
        if not u or not r:
            continue
        if _EXPLICIT_FR.search(u + " " + r) or label_passage(u + " " + r) == "nsfw":
            continue
        yield u, r


def build_owner_chats(rng: random.Random, facts: dict, chat_ok, label_passage, path: str = OWNER_CHATS,
                      n: int = 2000) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    if not os.path.exists(path):
        why["missing:owner chats"] += 1
        return out, why
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("source") != "whatsapp_chat":
                continue
            pairs += list(owner_pairs(r, label_passage))
    rng.shuffle(pairs)
    for i, (u, a) in enumerate(pairs):
        if len(out) >= n:
            break
        reason = chat_ok(u, a, facts)
        if reason:
            why[f"owner:{reason}"] += 1
            continue
        lang = "fr" if guess_lang(u + " " + a) == "fr" else "en"
        state = sample_state(rng)
        out.append({"id": f"chat:owner_whatsapp:{i}", "task": "chat", "subtype": "owner_whatsapp", "source": "owner:whatsapp_chat",
                    "prompt": u, "program": a, "gold": None, "system": identity_system(facts, state), "state": state, "lang": lang,
                    "repeat": 2 if lang == "fr" else 1})
    return out, why


OWNER_TEAMS = r"C:\Users\grill\Desktop\team-dataset.txt"
_TEAMS_BY = re.compile(r"^(.*) by ([A-ZÀ-Ý][\w'’\-]+(?: [A-ZÀ-Ý][\w'’\-]+)*)$")
_TEAMS_TIME = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Yesterday|Today|\d{1,2}/\d{1,2})\b.*\d{1,2}:\d{2}\s*(am|pm)$", re.I)


def teams_messages(lines: list[str]) -> list[tuple[str, str]]:
    """(speaker, text) from a pasted Teams message list: a `<preview> by <Name>` line opens a message when a
    time line follows within two lines (the export puts the name before OR after the time, depending on the
    speaker); the body runs from after the name/time lines to the next opener."""
    out: list[tuple[str, str]] = []
    i = 0

    def opener(k: int) -> int:
        """0 when line k is not an opener, else the index of the first body line."""
        if k >= len(lines) or not _TEAMS_BY.match(lines[k].strip()):
            return 0
        name = _TEAMS_BY.match(lines[k].strip()).group(2)
        nxt = [(j, lines[j].strip()) for j in range(k + 1, min(k + 4, len(lines))) if lines[j].strip()]
        heads = [(j, t) for j, t in nxt[:2]]
        if not any(_TEAMS_TIME.match(t) for _, t in heads):
            return 0
        skip = [j for j, t in heads if _TEAMS_TIME.match(t) or t == name]
        return (max(skip) + 1) if skip else k + 1

    while i < len(lines):
        start = opener(i)
        if start:
            name = _TEAMS_BY.match(lines[i].strip()).group(2)
            j = start
            body = []
            while j < len(lines) and not opener(j):
                body.append(lines[j].strip())
                j += 1
            text = " ".join(x for x in body if x)
            if text:
                out.append((name, text))
            i = j
        else:
            i += 1
    return out


def build_owner_teams(rng: random.Random, facts: dict, chat_ok, label_passage, path: str = OWNER_TEAMS,
                      n: int = 2000) -> tuple[list[dict], Counter]:
    """Both directions, speaker names (first and last, every speaker) screened out of every turn."""
    out, why = [], Counter()
    if not os.path.exists(path):
        why["missing:owner teams"] += 1
        return out, why
    msgs = teams_messages(open(path, encoding="utf-8", errors="replace").read().splitlines())
    names = {w.lower() for name, _ in msgs for w in name.replace("-", " ").split() if len(w) >= 3}
    name_re = re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True)) + r")\b", re.I) if names else None
    pairs = []
    for (s1, t1), (s2, t2) in zip(msgs, msgs[1:]):
        if s1 == s2:
            continue
        u, a = owner_turn(t1), owner_turn(t2)
        if not u or not a or (name_re and (name_re.search(u) or name_re.search(a))):
            why["teams:name or screened"] += 1
            continue
        if _EXPLICIT_FR.search(u + " " + a) or label_passage(u + " " + a) == "nsfw":
            why["teams:explicit"] += 1
            continue
        pairs.append((u, a))
    rng.shuffle(pairs)
    for i, (u, a) in enumerate(pairs):
        if len(out) >= n:
            break
        reason = chat_ok(u, a, facts)
        if reason:
            why[f"teams:{reason}"] += 1
            continue
        lang = "fr" if guess_lang(u + " " + a) == "fr" else "en"
        state = sample_state(rng)
        out.append({"id": f"chat:owner_teams:{i}", "task": "chat", "subtype": "owner_teams", "source": "owner:teams_export",
                    "prompt": u, "program": a, "gold": None, "system": identity_system(facts, state), "state": state, "lang": lang,
                    "repeat": 2 if lang == "fr" else 1})
    return out, why


def write_real_user_turns(out_path: str, path: str = OWNER_CHATS, min_words: int = 3, max_words: int = 60) -> int:
    """The real human turns of the owner's AI-assistant exports (ChatGPT / Claude), privacy-screened, as the
    realistic serve eval's prompts (TODO 2026-09-03). Never a training target. -> count written."""
    if not os.path.exists(path):
        return 0
    seen, n = set(), 0
    with open(path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as g:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("source") == "whatsapp_chat":
                continue
            for m in r.get("messages") or []:
                if not isinstance(m, dict) or m.get("role") != "user":
                    continue
                t = owner_turn(m.get("content"))
                if not t or not (min_words <= len(t.split()) <= max_words) or t in seen:
                    continue
                seen.add(t)
                g.write(json.dumps({"text": t, "lang": ("fr" if guess_lang(t) == "fr" else "en"), "source": r.get("source")}, ensure_ascii=False) + "\n")
                n += 1
    return n


# ── toolcall: the host's tool registry in each base's native call format, with negatives ─────────────────────
# The registry the host can actually execute or refuse under policy: the game (mounted plugin), the memory cortex,
# and the two network tools that a plugin cortex will own behind an explicit capability (deny-by-default until then).
TOOLS = [
    {"name": "news_search", "description": "Search today's news headlines for a topic.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to search for"},
                                                     "language": {"type": "string", "enum": ["en", "fr"]}}, "required": ["query"]}},
    {"name": "web_search", "description": "Search the web for a question that needs a source.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "pacman_status", "description": "Report the current state of the cubby-man game: level, score, lives, what was learned.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "pacman_explore", "description": "Play the cubby-man maze for a number of steps.",
     "parameters": {"type": "object", "properties": {"steps": {"type": "integer"}}, "required": ["steps"]}},
    {"name": "remember_fact", "description": "Store a fact the user asked you to remember.",
     "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}},
    # the factory (owner, 2026-09-04): when no registered tool fits, ask the forge to make one — ToolForge writes a CubeLang
    # program from the description, the VM certifies it on the example, it is registered (retire-not-delete) and shows up in
    # the next tool list. It composes registered primitives only: a request can never grant network or file access.
    {"name": "request_tool", "description": "Ask the tool factory to build a new tool when none of the available tools fits the request.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "a short snake_case name for the new tool"},
                                                     "description": {"type": "string", "description": "what the tool should do"},
                                                     "example_request": {"type": "string", "description": "the user's request, verbatim"}},
                    "required": ["name", "description", "example_request"]}},
]
_TOOL_IDEAS = [("unit_convert", "convert a value between units", ["convert 30 celsius to fahrenheit for me", "how many kilometres is 26 miles",
                                                                  "convertis 30 degrés celsius en fahrenheit", "ça fait combien de kilomètres, 26 milles"]),
               ("dice_roll", "roll dice and report the result", ["roll two six-sided dice for me", "can you roll a d20", "lance deux dés à six faces", "tu peux lancer un d20"]),
               ("timer", "start a countdown timer for a number of minutes", ["set a timer for 25 minutes", "start a 10 minute countdown", "mets un minuteur de 25 minutes",
                                                                             "lance un compte à rebours de 10 minutes"]),
               ("translate", "translate a message into another language", ["translate 'good morning' into spanish", "how do you say thank you in japanese",
                                                                           "traduis « bonjour » en espagnol", "comment on dit merci en japonais"]),
               ("tip_split", "split a bill and compute the tip", ["split a 96 dollar bill four ways with 15 percent tip", "what's 18 percent tip on 42 dollars",
                                                                 "divise une facture de 96 dollars en quatre avec 15 pour cent de pourboire", "c'est combien 18 pour cent de pourboire sur 42 dollars"]),
               ("word_count", "count the words in a text", ["count the words in this paragraph", "how many words is my last message", "compte les mots de ce paragraphe",
                                                            "combien de mots fait mon dernier message"])]
TOOL_SYSTEM = {
    "hermes": ("You are Cubby. You may call a tool when the user needs it.\n\n# Tools\n\nYou may call one or more functions to assist "
               "with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
               + "\n".join(json.dumps({"type": "function", "function": t}) for t in TOOLS)
               + "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
               "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"),
    "lfm": "You are Cubby. You may call a tool when the user needs it.\nList of tools: " + json.dumps(TOOLS),
}
_TOPICS_EN = ["quebec", "the montreal canadiens", "the weather in montreal", "the election", "the stock market", "the canadiens game tonight",
              "the federal budget", "the metro strike", "the wildfires", "the housing market", "the olympics", "the new iphone"]
_TOPICS_FR = ["le québec", "les canadiens de montréal", "la météo à montréal", "les élections", "la bourse", "le match des canadiens ce soir",
              "le budget fédéral", "la grève du métro", "les feux de forêt", "le marché immobilier", "les olympiques", "le nouvel iphone"]
_QUESTIONS_EN = ["who won the nobel prize in physics this year", "what is the population of montreal right now", "when is the next solar eclipse",
                 "is the 40 closed this weekend", "what time does the jean-talon market open", "how much is a metro pass in montreal",
                 "what's the exchange rate for the canadian dollar today", "who is the current mayor of quebec city"]
_QUESTIONS_FR = ["qui a gagné le prix nobel de physique cette année", "c'est quoi la population de montréal en ce moment", "quand est la prochaine éclipse",
                 "est-ce que la 40 est fermée cette fin de semaine", "à quelle heure ouvre le marché jean-talon", "combien coûte une passe de métro à montréal",
                 "c'est quoi le taux de change du dollar canadien aujourd'hui", "qui est le maire de québec en ce moment"]
_FACTS_EN = ["my sister's birthday is on the 12th of march", "the wifi password at the cottage is bluebird42", "my dentist is doctor Tremblay on rue Saint-Denis",
             "the car's oil change is due in november", "my nephew's name is Léo", "the meeting with the accountant is every first monday"]
_FACTS_FR = ["l'anniversaire de ma sœur est le 12 mars", "le mot de passe du wifi au chalet est bluebird42", "mon dentiste est le docteur Tremblay rue Saint-Denis",
             "le changement d'huile de l'auto est en novembre", "mon neveu s'appelle Léo", "la réunion avec le comptable est chaque premier lundi"]
_ASKS = {
    "news_search": {"en": ["what's in the news today about {t}?", "any headlines about {t}?", "what are people saying about {t} today", "give me the latest news on {t}",
                           "anything new about {t} this morning?"],
                    "fr": ["quoi de neuf dans les nouvelles sur {t} ?", "des nouvelles de {t} ce matin ?", "qu'est-ce qui se dit sur {t} aujourd'hui", "donne-moi les dernières nouvelles sur {t}",
                           "il y a du nouveau sur {t} ?"]},
    "web_search": {"en": ["{t}?", "look up {t}", "can you check {t}", "search for {t} please", "find out {t}"],
                   "fr": ["{t} ?", "cherche {t}", "peux-tu vérifier {t}", "regarde {t} s'il te plaît", "trouve {t}"]},
    "pacman_status": {"en": ["how is the pacman game going?", "what's your score in the maze?", "how many lives do you have left?", "status of the game", "what did you learn in the maze so far?"],
                      "fr": ["comment va la partie de pacman ?", "c'est quoi ton score dans le labyrinthe ?", "il te reste combien de vies ?", "statut de la partie", "qu'as-tu appris dans le labyrinthe ?"]},
    "pacman_explore": {"en": ["play pacman for {n} steps", "explore the maze for {n}", "go explore for {n} steps", "keep playing, {n} more steps", "wander {n} steps"],
                       "fr": ["joue au pacman pendant {n} coups", "explore le labyrinthe pendant {n}", "va explorer {n} coups", "continue de jouer, {n} coups de plus", "balade-toi {n} coups"]},
    "remember_fact": {"en": ["remember that {t}", "please remember: {t}", "note this down, {t}", "keep in mind that {t}", "don't forget, {t}"],
                      "fr": ["retiens que {t}", "souviens-toi : {t}", "note ça, {t}", "garde en tête que {t}", "n'oublie pas, {t}"]},
    "request_tool": {"en": ["{t}"], "fr": ["{t}"]},     # the ask IS the example request (rendered from _TOOL_IDEAS)
}


def render_call(fmt: str, name: str, args: dict) -> str:
    if fmt == "hermes":
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
    py = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return f"<|tool_call_start|>[{name}({py})]<|tool_call_end|>"


_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_LFM_CALL = re.compile(r"<\|tool_call_start\|>\s*\[\s*(\w+)\((.*?)\)\s*\]\s*<\|tool_call_end\|>", re.S)


def parse_call(text: str) -> dict | None:
    """The first native tool call in a generation: {name, arguments} or {malformed: True}; None when there is none."""
    m = _HERMES_CALL.search(text)
    if m:
        try:
            d = json.loads(m.group(1))
            return {"name": d.get("name"), "arguments": d.get("arguments") or {}}
        except json.JSONDecodeError:
            return {"name": None, "arguments": {}, "malformed": True}
    m = _LFM_CALL.search(text)
    if m:
        args = {}
        for k, v in re.findall(r"(\w+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|-?\d+)", m.group(2)):
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
        return {"name": m.group(1), "arguments": args}
    if "<tool_call>" in text or "<|tool_call_start|>" in text:
        return {"name": None, "arguments": {}, "malformed": True}
    return None


def toolcall_ok(rec: dict, gen: str) -> bool:
    """A call is due (`gold` = the tool name): the call is well-formed, names that tool, carries every required
    argument and the query/fact/steps value agrees with the record's. No call is due (`gold` = "none"): none appears."""
    call = parse_call(gen)
    gold = rec.get("gold")
    if gold == "none":
        return call is None
    if call is None or call.get("malformed") or call.get("name") != gold:
        return False
    want = rec.get("call_args") or {}
    for k, v in want.items():
        got = call["arguments"].get(k)
        if k == "steps":
            if got != v:
                return False
        elif isinstance(v, str) and _norm_q(str(got or "")) and token_f1(str(got), v) < 0.5:
            return False
        elif got is None:
            return False
    return True


def build_toolcall(rng: random.Random, facts: dict, chat_rows: list[dict], n_pos: int = 1200, n_neg: int = 1200) -> tuple[list[dict], Counter]:
    """Positives from the templates (both formats, EN+FR), negatives from existing chat rows re-issued with the tool list
    in the system prompt (their reply stands, no call). Every record: task toolcall, `format`, `gold` (tool or "none"),
    `call_args` for the check."""
    out, why = [], Counter()
    i = 0
    for fmt in ("lfm", "hermes"):
        for _ in range(n_pos // 2):
            name = rng.choice(list(_ASKS))
            lang = rng.choice(["en", "fr"])
            tmpl = rng.choice(_ASKS[name][lang])
            if name == "news_search":
                t = rng.choice(_TOPICS_EN if lang == "en" else _TOPICS_FR); ask = tmpl.format(t=t); args = {"query": t, "language": lang}
            elif name == "web_search":
                t = rng.choice(_QUESTIONS_EN if lang == "en" else _QUESTIONS_FR); ask = tmpl.format(t=t); args = {"query": t}
            elif name == "pacman_status":
                ask = tmpl; args = {}
            elif name == "pacman_explore":
                n = rng.choice([10, 20, 30, 50]); ask = tmpl.format(n=n); args = {"steps": n}
            elif name == "request_tool":                 # nothing registered fits: the model asks the factory for a tool
                tool_name, desc, asks = rng.choice(_TOOL_IDEAS)
                ask = rng.choice([a for a in asks if (lang == "fr") == any(w in a for w in ("é", "è", "ça", "tu ", " en ", "dés", "mots"))] or asks)
                args = {"name": tool_name, "description": desc, "example_request": ask}
            else:
                t = rng.choice(_FACTS_EN if lang == "en" else _FACTS_FR); ask = tmpl.format(t=t); args = {"fact": t}
            ask = ask[0].upper() + ask[1:] if rng.random() < 0.5 else ask
            out.append({"id": f"toolcall:{fmt}:{i}", "task": "toolcall", "subtype": name, "source": "templates:tool_registry", "format": fmt,
                        "prompt": ask, "program": render_call(fmt, name, args), "gold": name, "gold_any": [name], "call_args": args,
                        "system": TOOL_SYSTEM[fmt], "state": None, "lang": lang})
            i += 1
        pool = [r for r in chat_rows if r.get("task") == "chat" and 1 <= len(r["prompt"].split()) <= 40 and "\n" not in r["prompt"]]
        rng.shuffle(pool)
        for r in pool[: n_neg // 2]:
            out.append({"id": f"toolcall:{fmt}:neg:{i}", "task": "toolcall", "subtype": "none", "source": f"negatives:{r.get('source', 'chat')}", "format": fmt,
                        "prompt": r["prompt"], "program": r["program"], "gold": "none", "gold_any": ["none"], "call_args": {},
                        "system": TOOL_SYSTEM[fmt], "state": None, "lang": r.get("lang", "en")})
            i += 1
    rng.shuffle(out)
    return out, why


CHECKS = {"verbalize": None, "repair": repair_ok, "quebec": quebec_ok, "quebec_mc": label_first_ok, "toolcall": toolcall_ok, "rewrite": rewrite_ok, "appraisal": appraisal_ok,
          "dialog_emotion": dialog_emotion_ok, "dialog_act": label_first_ok, "empathy": empathy_ok}


def gap_ok(rec: dict, gen: str, facts: dict) -> bool:
    """One entry point for the evals: the family's own check (verbalize needs the facts for the voice rules)."""
    task = rec.get("task")
    if task == "verbalize":
        return verbalize_ok(rec, gen, facts)
    fn = CHECKS.get(task)
    return bool(fn and fn(rec, gen))
