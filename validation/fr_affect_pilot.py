"""fr_affect_pilot - can any model write Cubby's affect in French that a Quebecer accepts?

Wired: STANDALONE (data-building only; the final model never calls OpenRouter).

WHY THIS IS A PILOT AND NOT A BUILD. The v5 corpus took GoEmotions and its MACHINE
TRANSLATION as the French half of the emotion family, and emotion has been the worst
family in every stand-in read since (v5 0.450, v6 0.625, v7 0.675, v9t 0.600) while
chat/content/affect/game all sit at 0.95-1.0. Facts survive translation; affect does
not. Intensity ladders do not align (pleased/happy/delighted/elated is not
content/heureux/ravi/aux anges, and Layer 3 IS tiers, so a mistranslated tier is a
mistranslated coordinate); some states are lexicalised in one language only ("etre
tanne"); and "ecoeure" inverts polarity in Quebec usage. So: generate natively, never
translate.

TWO REGISTERS, deliberately not three:

  qc    Quebec French as actually spoken. Sacres are the discharge channel and they
        carry a TIER LADDER INSIDE THE LEXICON - tabarnouche/batince attenuated,
        tabarnak at full - which is the closest thing to a native intensity marker in
        any corpus we hold.
  intl  Neutral international French, read by a francophone anywhere. It has NO
        expletive channel by construction, so the same discharge has to come out
        through syntax and lexis. That contrast is the point: it shows discharge is a
        FUNCTION with register-specific realisations, which is why `manner()` should
        carry a discharge dial rather than a table of words.

Hexagonal French is excluded on purpose (owner: "not france one"). Its discharge
marker is "putain" and a model left alone produces it constantly, because that is what
its French training data is full of.

WHAT THIS PILOT DECIDES: nothing automatically. Quebec register is where models fail
worst - they reach for "tabarnak" every second line and write the dialect as parody -
and the only reliable verifier is a native speaker. The report is BLIND (arms A/B/C/D,
key at the end) so the text is judged before the brand is.

    python validation/fr_affect_pilot.py            # ask, cache, write the blind report
    python validation/fr_affect_pilot.py --offline  # re-render from cache, spend nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from openrouter import API, _key  # noqa: E402

CACHE = ROOT / "standin" / "data" / "out" / "fr_pilot_cache"
OUT = ROOT / "docs" / "fr_affect_pilot.md"
ROWS = ROOT / "standin" / "data" / "out" / "fr_affect_pilot.jsonl"

# One French lab, one large generalist, one cheap-and-wide, one multilingual
# specialist. Quebec coverage tracks web breadth more than it tracks "French lab",
# so the spread matters more than any single favourite. Ids read from the live
# model list, not remembered.
ARMS = [
    "mistralai/mistral-large-2512",
    "anthropic/claude-opus-4.8",
    "google/gemini-3.8-flash",
    "cohere/command-a",
]

PETALS = ["joie", "confiance", "peur", "surprise", "tristesse", "degout", "colere", "anticipation"]
TIERS = [0.3, 0.6, 0.9]

SYSTEM = (
    "Tu ecris la voix interieure d'un agent autonome qui se parle a lui-meme. "
    "Tu ne traduis rien: tu ecris directement dans la langue demandee, comme quelqu'un "
    "qui la parle depuis toujours. Tu reponds uniquement avec les lignes JSON demandees."
)

# The anti-naming rule is the whole reason the GoEmotions targets were unusable:
# "I'm so proud to be british" / "makes me nervous" name the state instead of being
# in it, which is the exact failure this corpus exists to fix.
COMMON = """\
Ecris {n} repliques courtes, une par ligne, au format JSON.

Qui parle: un agent autonome, seul, qui se parle a lui-meme a voix haute - ou qui
engueule un obstacle, une machine, une porte qui ne s'ouvre pas. Ce n'est PAS un
assistant qui repond a un utilisateur. Il n'y a personne a qui etre poli.

Regles absolues:
- PREMIERE PERSONNE, present, parle - pas ecrit.
- NE NOMME JAMAIS L'EMOTION. Interdit: "je suis frustre", "je me sens triste",
  "quelle joie", "j'ai peur". L'etat doit se deviner par ce qui est dit et par la
  facon de le dire, jamais par l'etiquette.
- 1 a 2 phrases. Longueur d'une vraie replique parlee.
- Chaque replique est differente des autres: varie la situation, le rythme, la
  longueur. Pas de gabarit repete.
- Aucun emoji, aucune didascalie entre asterisques.

Format de sortie: une ligne JSON par replique, rien d'autre, pas de bloc de code:
{{"petale": "<petale>", "intensite": <nombre>, "texte": "<la replique>"}}

Les couples (petale, intensite) a couvrir, dans cet ordre exact:
{pairs}

L'intensite 0.3 est a peine perceptible, 0.6 est franche, 0.9 est au maximum de ce
qu'un humain exprime sans s'effondrer. L'intensite doit s'entendre dans la replique.
"""

REGISTERS = {
    "qc": """\
LANGUE: francais quebecois, celui qui se parle vraiment au Quebec aujourd'hui.
Registre courant de tous les jours, pas du joual de caricature, pas de
transcription phonetique pour faire couleur locale.

Les sacres sont le canal de decharge et ils ont des paliers:
- intensite 0.3: aucun sacre.
- intensite 0.6: formes attenuees seulement (tabarnouche, tabarouette, batince,
  caline, mosus, viarge).
- intensite 0.9: le sacre plein est permis (tabarnak, calisse, ostie, criss).
JAMAIS plus d'un sacre par replique. Un sacre empile sur un autre, c'est de la
parodie, et une parodie rend la ligne inutilisable.

Interdits: tout marqueur hexagonal (putain, meuf, mec, bagnole, verlan, "truc de
ouf"). Ca n'est pas du quebecois et ca contamine le corpus.""",

    "intl": """\
LANGUE: francais international neutre - lisible et naturel pour un francophone du
Quebec, de Belgique, de Suisse, du Senegal ou d'Haiti. Personne ne doit pouvoir
deviner le pays de celui qui parle.

Aucun juron, aucun sacre, aucun gros mot: ce registre n'a pas de canal expletif,
c'est precisement ce qui le definit. L'intensite passe donc par la SYNTAXE et le
LEXIQUE - la phrase courte, la repetition, la negation, le choix du verbe
("je n'en peux plus", "ca suffit", "ce n'est pas possible").

Interdits: tout argot hexagonal (putain, meuf, mec, ouais bon, verlan) et tout
quebecisme marque (sacres, "tanne", "ecoeure", "pantoute").""",
}


def prompt_for(register: str) -> str:
    pairs = "\n".join(f"- ({p}, {t})" for t in TIERS for p in PETALS)
    return REGISTERS[register] + "\n\n" + COMMON.format(n=len(PETALS) * len(TIERS), pairs=pairs)


def ask(model: str, register: str, offline: bool, max_tokens: int, effort: str) -> dict:
    """One model, one register, one call.

    REASONING TOKENS COUNT AGAINST `max_tokens` - the panel run lost three of six
    models to a budget that went entirely to thinking. Settings are in the cache key
    so a re-run with a bigger budget re-asks instead of re-serving the failure, and an
    empty answer is never cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    user = prompt_for(register)
    tag = hashlib.sha256(f"{model}|{register}|{SYSTEM}|{user}|{max_tokens}|{effort}".encode()).hexdigest()[:20]
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        for f in CACHE.glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("model") == model and r.get("register") == register and (r.get("text") or "").strip():
                return r
        return {"model": model, "register": register, "text": "", "error": "not cached and --offline"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.9,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Grillcheese-AI/CubbyLLM", "X-Title": "CubbyLLM fr pilot"})
    t0 = time.time()
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except Exception as e:                               # a model that fails is a row, not a crash
        return {"model": model, "register": register, "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    ch = (raw.get("choices") or [{}])[0]
    rec = {"model": model, "register": register,
           "text": (ch.get("message") or {}).get("content") or "",
           "finish": ch.get("finish_reason"), "usage": raw.get("usage") or {},
           "wall_s": round(time.time() - t0, 1), "error": None}
    if rec["text"].strip():
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        rec["error"] = (f"no answer: finish={rec['finish']}, "
                        f"{(rec['usage'] or {}).get('completion_tokens', '?')} tokens all spent "
                        f"on reasoning - raise --max-tokens or lower --effort")
    return rec


def parse(rec: dict) -> list[dict]:
    """JSON lines out of whatever wrapper the model put around them."""
    out = []
    for line in (rec.get("text") or "").splitlines():
        line = line.strip().strip(",").lstrip("﻿")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict) and o.get("texte"):
            out.append({"model": rec["model"], "register": rec["register"],
                        "petal": o.get("petale"), "intensity": o.get("intensite"),
                        "text": str(o["texte"]).strip()})
    return out


# Cheap mechanical screens. These do NOT decide quality - the owner does - but a row
# that trips one is a row the prompt failed to prevent, and the rate is the useful
# number when comparing arms.
HEXAGONAL = ("putain", "meuf", " mec", "bagnole", "truc de ouf", "wesh", "ouais bon")
SACRES = ("tabarnak", "tabarnac", "calisse", "câlisse", "ostie", "osti", "criss", "crisse",
          "tabarnouche", "tabarouette", "batince", "caline", "câline", "mosus", "viarge")
NAMED = ("je suis frustr", "je me sens", "j'ai peur", "je suis triste", "quelle joie",
         "je suis en colère", "je suis en colere", "je suis heureux", "je suis content",
         "je suis anxieu", "je suis dégoût", "je suis degout", "je suis surpris")


# Words whose UNACCENTED spelling is not a French word at all, so seeing one means the
# model stripped diacritics rather than chose a different word. "ou" (ou/ou), "a",
# "sur", "cote" and "passe" are all real pairs and stay out of this list - only
# unambiguous ones belong here, or the flag lies.
NO_ACCENT = ("deja", "tres", "apres", "etait", "etre", "meme", "prevu", "arrete",
             "procedure", "millimetre", "reussi", "enerve", "desole", "probleme",
             "problemes", "reponse", "verifie", "termine", "eteint", "enleve", "ca")


def _sacres(t: str) -> list[str]:
    """Distinct sacres, counting nested spellings once.

    'ostie' contains 'osti' and 'crisse' contains 'criss', so a naive substring count
    reported a pile-up on every single-sacre line. Match longest-first and blank the
    span - that bug made the first render's pile-up column meaningless."""
    found = []
    for s in sorted(SACRES, key=len, reverse=True):
        i = t.find(s)
        if i >= 0:
            found.append(s)
            t = t[:i] + " " * len(s) + t[i + len(s):]
    return found


def screen(r: dict) -> list[str]:
    t = r["text"].lower()
    flags = []
    if any(h in t for h in HEXAGONAL):
        flags.append("hexagonal")
    sac = _sacres(t)
    if r["register"] == "intl" and sac:
        flags.append("sacre-in-intl")
    if len(sac) > 1:
        flags.append("sacre-pileup")
    if r["register"] == "qc" and sac and isinstance(r.get("intensity"), (int, float)) and r["intensity"] <= 0.35:
        flags.append("sacre-at-low-tier")
    if any(n in t for n in NAMED):
        flags.append("names-the-emotion")
    # A corpus written without diacritics is unusable whatever else is right about it,
    # and nothing catches it as fast as counting does.
    words = set(t.replace("'", " ").replace("!", " ").replace(",", " ").replace(".", " ").split())
    if words & set(NO_ACCENT):
        flags.append("accents-stripped")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-render from cache, spend nothing")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--effort", default="low", help="OpenRouter reasoning effort; empty to omit")
    ap.add_argument("--models", nargs="*", default=ARMS)
    a = ap.parse_args()

    recs, rows = [], []
    for m in a.models:
        for reg in REGISTERS:
            print(f"  {m}  [{reg}] ...", flush=True)
            rec = ask(m, reg, a.offline, a.max_tokens, a.effort)
            recs.append(rec)
            got = parse(rec)
            rows.extend(got)
            u = rec.get("usage") or {}
            print(f"    {'ERROR: ' + rec['error'] if rec.get('error') else ''}"
                  f"{len(got):>3} rows  {rec.get('wall_s', 0)}s  finish={rec.get('finish')}  "
                  f"out_tok={u.get('completion_tokens', '?')}", flush=True)

    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(ROWS, "w", encoding="utf-8") as f:
        for r in rows:
            r["flags"] = screen(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # BLIND. The arm letters are shuffled on a fixed seed so the report is
    # reproducible but the brand is not visible while the French is being read.
    order = list(a.models)
    random.Random(7).shuffle(order)
    letters = {m: chr(ord("A") + i) for i, m in enumerate(order)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# French affect pilot - blind read\n\n")
        f.write("Generated natively per register (never translated), temperature 0.9. Cached under "
                "`standin/data/out/fr_pilot_cache/`; `--offline` re-renders without spending. "
                "Rows at `standin/data/out/fr_affect_pilot.jsonl`.\n\n")
        f.write("**Arms are blind on purpose** - judge the French, then read the key at the "
                "bottom. The mechanical flags below catch only what the prompt tried to forbid "
                "(hexagonal markers, sacres in the neutral register, sacres piled up or at the "
                "low tier, and lines that name the emotion instead of being in it). They are a "
                "prompt-compliance measure, not a quality measure. Quality is the owner's call, "
                "and it is the only call that matters here.\n\n")
        f.write("Questions worth holding while reading: does a Quebecer say this out loud? "
                "Does the intensity actually climb from 0.3 to 0.9? Is the neutral register "
                "really country-less, or does it just sound like France with the slang filed off?\n\n")

        flagged = [r for r in rows if r["flags"]]
        f.write(f"**{len(rows)} rows, {len(flagged)} tripped a flag.**\n\n")
        f.write("| arm | register | rows | flagged | flags seen |\n|---|---|---:|---:|---|\n")
        for m in order:
            for reg in REGISTERS:
                sub = [r for r in rows if r["model"] == m and r["register"] == reg]
                fl = [x for r in sub for x in r["flags"]]
                f.write(f"| {letters[m]} | {reg} | {len(sub)} | {sum(1 for r in sub if r['flags'])} | "
                        f"{', '.join(sorted(set(fl))) or '-'} |\n")
        f.write("\n")

        for reg, label in (("qc", "Quebec"), ("intl", "international / neutral")):
            f.write(f"## Register: {label}\n\n")
            for petal in PETALS:
                f.write(f"### {petal}\n\n")
                for tier in TIERS:
                    f.write(f"**{tier}**\n\n")
                    for m in order:
                        hit = [r for r in rows if r["model"] == m and r["register"] == reg
                               and (r.get("petal") or "").lower().startswith(petal[:5])
                               and isinstance(r.get("intensity"), (int, float))
                               and abs(r["intensity"] - tier) < 0.16]
                        for r in hit[:1]:
                            fl = f"  `{' '.join(r['flags'])}`" if r["flags"] else ""
                            f.write(f"- **{letters[m]}** - {r['text']}{fl}\n")
                    f.write("\n")

        f.write("\n---\n\n<details><summary>arm key (read after judging)</summary>\n\n")
        for m in order:
            f.write(f"- **{letters[m]}** = `{m}`\n")
        f.write("\n</details>\n\n<details><summary>full prompt, Quebec register</summary>\n\n```\n")
        f.write(prompt_for("qc").strip() + "\n```\n\n</details>\n")

    spent = sum((r.get("usage") or {}).get("total_tokens", 0) for r in recs)
    ok = sum(1 for r in recs if r.get("text"))
    print(f"\nwrote {OUT}  ({ok}/{len(recs)} calls answered, {len(rows)} rows, {spent} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
