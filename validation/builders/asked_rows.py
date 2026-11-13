"""Order 0067, the reader's fifth round (Nick, 2026-11-13: "build it"): rows whose answer is ``asked``.

Round 4's held-out read showed the reader never abstains: on an unseen kind it forces the nearest familiar rule (no
training row had ``asked`` as its answer). Here: instructions no reply-world measurement covers (tone, register,
style, rhetoric; language properties the reply world does not measure: rhyme, voice, tense, questions only, figures
of speech, spelling variety). Their answer is ``asked``: kept as text for the talk seat, logged, never checked.

- ``PHRASES``: (key, English wordings, French wordings), by hand. A seeded quarter of the keys is held out whole
  (``asked_test``): abstention on unmeasurable instructions training never showed.
- Rows: a phrase alone; a phrase joined to a measurable rule (the answer is that rule alone: the style part is not a
  rule); and Nick's annotated prompts carrying a style phrase ("in a humorous tone", "in the style of"), when nothing
  measurable is in them.

`rows(seed)` -> rows in `build_rule_rows`' shape, splits ``train`` / ``dev`` / ``asked_test``.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "validation"), str(Path(__file__).resolve().parent)]
import exp_if_follow as IF  # noqa: E402

from cubbyllm import local_paths  # noqa: E402

__wiring__ = "STANDALONE"

PHRASES = [
    ("warm", ["Be warm and friendly.", "Keep a warm, kind tone."], ["Sois chaleureux et amical.", "Garde un ton chaleureux et gentil."]),
    ("formal", ["Use a formal register.", "Keep it formal and polite."], ["Emploie un registre soutenu.", "Reste formel et poli."]),
    ("casual", ["Talk casually, like to a friend.", "Keep the tone relaxed."], ["Parle de façon décontractée, comme à un ami.", "Garde un ton détendu."]),
    ("enthusiastic", ["Sound enthusiastic.", "Answer with lots of energy."], ["Montre-toi enthousiaste.", "Réponds avec beaucoup d'énergie."]),
    ("calm", ["Stay calm and reassuring.", "Use a soothing tone."], ["Reste calme et rassurant.", "Prends un ton apaisant."]),
    ("serious", ["Be serious, no jokes.", "Keep a serious tone throughout."], ["Sois sérieux, pas de blagues.", "Garde un ton sérieux du début à la fin."]),
    ("humorous", ["Make it funny.", "Add some humour to the answer."], ["Rends ça drôle.", "Ajoute un peu d'humour à la réponse."]),
    ("persuasive", ["Be persuasive.", "Try to convince the reader."], ["Sois persuasif.", "Essaie de convaincre le lecteur."]),
    ("balanced", ["Give a balanced view.", "Present both sides fairly."], ["Donne un point de vue équilibré.", "Présente les deux côtés avec justesse."]),
    ("empathetic", ["Show empathy.", "Answer with compassion."], ["Fais preuve d'empathie.", "Réponds avec compassion."]),
    ("confident", ["Sound confident.", "Answer with assurance."], ["Aie l'air sûr de toi.", "Réponds avec assurance."]),
    ("pirate", ["Talk like a pirate.", "Answer in the voice of a pirate."], ["Parle comme un pirate.", "Réponds avec la voix d'un pirate."]),
    ("shakespeare", ["Write in the style of Shakespeare.", "Sound like an Elizabethan playwright."],
     ["Écris dans le style de Shakespeare.", "Prends le ton d'un dramaturge élisabéthain."]),
    ("commentator", ["Answer like a sports commentator.", "Narrate it as a live sports broadcast."],
     ["Réponds comme un commentateur sportif.", "Raconte-le comme un direct sportif."]),
    ("child", ["Explain it so a five-year-old understands.", "Keep it simple enough for a child."],
     ["Explique-le pour qu'un enfant de cinq ans comprenne.", "Garde ça assez simple pour un enfant."]),
    ("expert", ["Write for experts in the field.", "Use a technical, specialist tone."],
     ["Écris pour des experts du domaine.", "Emploie un ton technique et spécialisé."]),
    ("plain", ["Use plain language.", "Avoid jargon."], ["Emploie un langage simple.", "Évite le jargon."]),
    ("story", ["Tell it as a story.", "Frame the answer as a short tale."], ["Raconte-le comme une histoire.", "Présente la réponse comme un petit conte."]),
    ("rhyme", ["Make it rhyme.", "Write the answer in rhyming lines."], ["Fais rimer.", "Écris la réponse en vers qui riment."]),
    ("passive", ["Use the passive voice.", "Write every sentence in the passive."], ["Utilise la voix passive.", "Écris chaque phrase au passif."]),
    ("present", ["Use the present tense only.", "Keep everything in the present tense."], ["Utilise seulement le présent.", "Garde tout au présent."]),
    ("questions", ["Answer only with questions.", "Make every sentence a question."], ["Réponds seulement par des questions.", "Fais de chaque phrase une question."]),
    ("metaphor", ["Use metaphors.", "Include vivid imagery."], ["Utilise des métaphores.", "Inclus des images frappantes."]),
    ("alliteration", ["Use alliteration.", "Play with alliteration where you can."], ["Utilise des allitérations.", "Joue avec les allitérations quand tu peux."]),
    ("british", ["Use British spelling.", "Spell words the British way."], ["Utilise l'orthographe britannique.", "Écris les mots à la britannique."]),
    ("humble", ["Be humble about it.", "Stay modest in tone."], ["Reste humble.", "Garde un ton modeste."]),
    ("direct", ["Be direct.", "Get to the point without hedging."], ["Sois direct.", "Va droit au but sans détour."]),
    ("poetic", ["Make it poetic.", "Write with a lyrical touch."], ["Rends ça poétique.", "Écris avec une touche lyrique."]),
    ("encouraging", ["Be encouraging.", "Cheer the reader on."], ["Sois encourageant.", "Encourage le lecteur."]),
    ("neutral", ["Stay neutral.", "Keep your opinion out of it."], ["Reste neutre.", "Garde ton opinion pour toi."]),
    ("mysterious", ["Sound mysterious.", "Keep an air of mystery."], ["Prends un air mystérieux.", "Garde une part de mystère."]),
    ("gentle", ["Be gentle.", "Phrase things softly."], ["Sois doux.", "Formule les choses avec douceur."]),
]
STYLE = re.compile(r"\bin (?:a|an) [a-z]+(?:,? [a-z]+)? (?:tone|style|manner|voice|way)\b|\bin the style of\b|\bas if you were\b"
                   r"|\bfrom the perspective of\b", re.I)      # explicit style phrases only: "a persuasive essay" is a task


def _alpaca_style(max_chars: int = 300) -> list[dict]:
    import alpaca_rows
    path = Path(local_paths.value("CUBBY_INSTRUCT_SVC") or "")
    if not path.is_file():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        t = r["text"]
        if "_p" in r["id"] and len(t) <= max_chars and STYLE.search(t) and not IF.rules_of(t) \
                and not re.search(r"\bin an? (?:different|new|other) (?:way|manner)\b", t, re.I) \
                and alpaca_rows.read_length(t) is None and not alpaca_rows.LENGTHY.search(t):
            out.append({"kind": "asked_alpaca", "lang": "en", "instruction": t, "plan": None, "asked": True, "template": None,
                        "ifeval": [], "source_id": r["id"]})
    return out


def rows(seed: int = 6, measurable=None) -> list[dict]:
    """`measurable`: optional [(text, plan, lang)] rules to join a phrase to (the answer is that rule alone)."""
    rng = random.Random(seed)
    keys = [k for k, _, _ in PHRASES]
    held = set(rng.sample(keys, len(keys) // 4))
    out = []
    for key, en, fr in PHRASES:
        for lang, ws in (("en", en), ("fr", fr)):
            for w in ws:
                split = "asked_test" if key in held else ("dev" if rng.random() < 0.1 else "train")
                out.append({"kind": f"asked_{key}", "lang": lang, "instruction": w, "plan": None, "asked": True,
                            "template": None, "ifeval": [], "split": split})
                if split != "asked_test" and measurable:
                    for _ in range(2):
                        mt, mp, ml = rng.choice([m for m in measurable if m[2] == lang])
                        both = f"{mt} {w}" if rng.random() < 0.5 else f"{w} {mt}"
                        out.append({"kind": f"asked_{key}+rule", "lang": lang, "instruction": both, "plan": mp, "asked": False,
                                    "template": None, "ifeval": [], "split": "dev" if rng.random() < 0.1 else "train"})
    styled = _alpaca_style()
    rng.shuffle(styled)
    for i, r in enumerate(styled[:400]):
        r["split"] = "asked_test" if i < 60 else ("dev" if rng.random() < 0.1 else "train")
    return out + styled[:400]


__all__ = ["PHRASES", "rows"]
