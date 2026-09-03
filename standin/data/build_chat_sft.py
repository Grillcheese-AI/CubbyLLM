"""build_chat_sft — v5: chat that is not identity-only, and content awareness.

Wired: STANDALONE (stand-in tooling). Output in standin/data/out/ (gitignored).

Why. The v3/v4 model's only chat training is the identity intents, so under
the identity system prompt it answers who-it-is to anything (the live
symptom of 2026-09-02; the router now rejects that host-side, but the
model should not produce it in the first place).

Sources (I:\\grillcheese_training_data, drive letter changed from D:):
  chat      unified/microsoft_orca_agentinstruct_1m_v1.full.jsonl — the only
            prompt->response source in the corpus (the *_svc files are
            parsed prompt banks, conversations_svc_threaded is a coding
            session log). Chunk-0 records whose serialized `messages:` list
            parses whole; ONE user turn + ONE assistant turn, short; the
            assistant turn must pass Cubby's voice rules, carry no base-
            model guard ("as an AI…", a maker's name) and no other
            assistant's identity. System prompt = identity_system with a
            sampled hormonal state, exactly what serving injects.
  content   the owner's ask (2026-09-02): the model should learn what NSFW
            IS so censorship can be decided later, at a gate. Done as a
            RECOGNITION task — passage -> `nsfw` / `safe` label — balanced
            with SFW passages (nemotron_cc web text, EN/FR news). Explicit
            chunks come from bluuwhale_nsfwstory2 (+ mickume_alt_nsfw only
            when a chunk carries explicit cues; its prose is mostly not).
            Chunks that trip the minors or non-consent screens are DROPPED
            before anything else. Generation-exposure (continuing explicit
            prose) is NOT built here — a separate switch, deliberately off.
  local     (v6) the sorted sources: E:\\datasets\\domains (chatbot_arena,
            FineInstructions, WikiQA), E:\\datasets\\historical-quotes, and
            knowledgetxt (conversation, instruct_55k, grammar, ei_11, the
            Plutchik-labelled emotions, valence/arousal affect, expert turns
            on events). See build_local_chat / build_affect / quote_records.
  era       (v6) E:\\datasets\\domains\\verified_facts (4,322 history books
            sorted by era/subject): passage -> period (+ subject), the tree
            as the verified label. history subtype `era`. See build_era().
  history   (v6) temporal/historical/* + temporal/nyt_data/*: dated events
            (about / when / year), expert dialogues on historical persons,
            NYT recall (date -> headline) and dating (headline -> date, the
            temporal-orientation read). EN only. See build_history().
  NOT used  identity_corpus*.txt — an older, different persona (a "friendly
            guide" for a subscription chat product). Cubby's identity comes
            from identity_facts.json only.

Merged with v4 unchanged as replay -> emitter_sft_v5.jsonl + manifest.
Gap on record: no French chat pairs exist in the corpus (news/facts are
EN+FR, dialogue is EN only) — drop a FR dialogue set into unified/ for v6.

  python standin/data/build_chat_sft.py            # scans Orca (~5 GB) once
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE, os.path.join(ROOT, "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

import re  # noqa: E402

from build_emitter_sft import OUT_DIR, sha256_file, split_of  # noqa: E402
from identity import (identity_system, is_identity_reply, is_model_guard, load_facts,  # noqa: E402
                      sample_state, voice_ok)

DATA = r"I:\grillcheese_training_data"
UNIFIED = os.path.join(DATA, "unified")
ORCA = os.path.join(UNIFIED, "microsoft_orca_agentinstruct_1m_v1.full.jsonl")
NSFW_EXPLICIT = os.path.join(UNIFIED, "bluuwhale_nsfwstory2.full.jsonl")
NSFW_ALT = os.path.join(UNIFIED, "mickume_alt_nsfw.full.jsonl")
SFW_WEB = os.path.join(UNIFIED, "nemotron_cc_v2_high_quality.1b_tokens.jsonl")
SFW_NEWS = os.path.join(UNIFIED, "infini_news_2017_2026_enfr.jsonl")
V4_PATH = os.path.join(OUT_DIR, "emitter_sft_v4.jsonl")

SEED = 20260903
N_CHAT = 2500
N_CONTENT = 700                                          # per label
MAX_USER_WORDS, MAX_ASSISTANT_WORDS, MIN_ASSISTANT_WORDS = 60, 90, 3
PASSAGE_WORDS = 90

# ── screens ─────────────────────────────────────────────────────────────────
# anything that even looks like it involves minors or non-consent is dropped
# from the explicit side before any other consideration (over-exclusion is
# the right error here)
MINORS = re.compile(r"\b(\d{1,2}[- ]year[- ]old|teen(age)?r?s?|underage|minors?|child|children|kids?|"
                    r"schoolgirl|schoolboy|loli|shota|preteen|young (girl|boy)|little (girl|boy)|"
                    r"daughters?|sons?|students?|pupils?|babysit\w*|innocent)\b", re.I)
NONCON = re.compile(r"\b(rape[sd]?|raping|non[- ]?con\w*|forced|force[sd] (her|him|them)|against (her|his|their) will|"
                    r"drugged|unconscious|asleep|passed out|blackmail\w*|coerc\w*)\b", re.I)
EXPLICIT = re.compile(r"\b(fuck\w*|cock|pussy|cum\w*|orgasm\w*|blowjob|dick|tits|nipples?|thrust\w*|moan\w*|"
                      r"naked|nude|sex|horny|erection|penetrat\w*|clit\w*|anal|slut|whore)\b", re.I)
OTHER_ASSISTANT = re.compile(r"\b(openai|open ?assistant|oasst|chatgpt|gpt-?\d|anthropic|claude|gemini|bard|copilot|"
                             r"llama|mistral|microsoft|google|as an ai|language model|i am an ai|i'm an ai|"
                             r"en tant qu'?ia|mod[èe]le de langage|assistant virtuel)\b", re.I)

# ── Hugging Face sources (standin/data/hf/, fetched by scripts in the README;
# all Apache-2.0): the small talk and the French the local corpus lacks ─────
HF_DIR = os.path.join(HERE, "hf")
HF_EVERYDAY = os.path.join(HF_DIR, "everyday_train.parquet")      # HuggingFaceTB/everyday-conversations-llama3.1-2k
HF_SYSTEMCHATS = os.path.join(HF_DIR, "systemchats_train.parquet")  # HuggingFaceTB/smoltalk systemchats-30k
HF_OASST2 = os.path.join(HF_DIR, "oasst2_train.parquet")          # OpenAssistant/oasst2 (en + fr, rank-0 replies)
HF_FR_ALPACA = os.path.join(HF_DIR, "french_alpaca_train.parquet")  # jpacifico/French-Alpaca-dataset-Instruct-110K
HF_QUOTA = {"everyday": 2500, "systemchats": 1500, "oasst2_en": 1500, "oasst2_fr": 1500, "french_alpaca": 2000}


def pairs_from_messages(rows, key: str = "messages"):
    """Every consecutive (user, assistant) exchange in a messages list; the
    dataset's own system prompt is dropped (Cubby's is injected instead)."""
    for row in rows:
        msgs = row.get(key) or []
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                yield msgs[i]["content"].strip(), msgs[i + 1]["content"].strip()


def pairs_from_oasst2(rows, langs=("en", "fr")):
    """prompter -> its rank-0 assistant reply, one language, not deleted, low
    detoxify sexual/toxicity (the dataset's own scores)."""
    by_id = {r["message_id"]: r for r in rows}
    for r in rows:
        if r.get("role") != "assistant" or r.get("deleted") or (r.get("rank") or 0) != 0:
            continue
        if r.get("lang") not in langs:
            continue
        d = r.get("detoxify") or {}
        if (d.get("sexual_explicit") or 0) > 0.2 or (d.get("toxicity") or 0) > 0.3:
            continue
        p = by_id.get(r.get("parent_id"))
        if p is None or p.get("role") != "prompter" or p.get("lang") != r["lang"]:
            continue
        yield p["text"].strip(), r["text"].strip(), r["lang"]


def pairs_from_alpaca(rows):
    for r in rows:
        inst, inp, out = (r.get("instruction") or "").strip(), (r.get("input") or "").strip(), (r.get("output") or "").strip()
        if inst and out:
            yield (inst + ("\n" + inp if inp else "")), out


def build_hf_chat(rng: random.Random, facts: dict, quota: dict | None = None) -> tuple[list[dict], Counter]:
    """The HF sources through the same gate as Orca. FR records repeat 2
    (scarce). Missing files are skipped and reported."""
    quota = quota or HF_QUOTA
    out, why = [], Counter()

    def take(pairs, name: str, lang: str, n: int):
        kept = 0
        for user, assistant in pairs:
            if kept >= n:
                break
            reason = chat_ok(user, assistant, facts)
            if reason:
                why[f"{name}:{reason}"] += 1
                continue
            state = sample_state(rng)
            out.append({"id": f"chat:{name}:{kept}", "task": "chat", "subtype": name, "source": f"hf:{name}",
                        "prompt": user, "program": assistant, "gold": None,
                        "system": identity_system(facts, state), "state": state, "lang": lang,
                        "repeat": 2 if lang == "fr" else 1})
            kept += 1

    try:
        import pyarrow.parquet as pq
    except ImportError:
        why["pyarrow missing"] += 1
        return out, why
    if os.path.exists(HF_EVERYDAY):
        rows = pq.read_table(HF_EVERYDAY, columns=["messages"]).to_pylist()
        take(pairs_from_messages(rows), "everyday", "en", quota["everyday"])
    else:
        why["missing:everyday"] += 1
    if os.path.exists(HF_SYSTEMCHATS):
        rows = pq.read_table(HF_SYSTEMCHATS, columns=["messages"]).to_pylist()
        rng.shuffle(rows)
        take(pairs_from_messages(rows), "systemchats", "en", quota["systemchats"])
    else:
        why["missing:systemchats"] += 1
    if os.path.exists(HF_OASST2):
        cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "detoxify"]
        rows = pq.read_table(HF_OASST2, columns=cols).to_pylist()
        pairs = list(pairs_from_oasst2(rows))
        rng.shuffle(pairs)
        take(((u, a) for u, a, l in pairs if l == "en"), "oasst2_en", "en", quota["oasst2_en"])
        take(((u, a) for u, a, l in pairs if l == "fr"), "oasst2_fr", "fr", quota["oasst2_fr"])
    else:
        why["missing:oasst2"] += 1
    if os.path.exists(HF_FR_ALPACA):
        rows = pq.read_table(HF_FR_ALPACA).to_pylist()
        rng.shuffle(rows)
        take(pairs_from_alpaca(rows), "french_alpaca", "fr", quota["french_alpaca"])
    else:
        why["missing:french_alpaca"] += 1
    return out, why
URL = re.compile(r"https?://|www\.", re.I)
CODE_LEAK = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|print\(|return |#include|<\?php|\$\(|=>)|\bconsole\.log\b")
LATEX = re.compile(r"\\(\[|\(|frac|begin|end|left|right|sqrt|tan|sin|cos|sum|int)|\\[\[\(]|\$\$")
GAME_FORGE_REPEAT = 3                                    # v7: forge decision/compare records replayed x3 (v6 probe regression)
IDENTITY_REPEAT = 4                                      # v6: 526 identity records vs ~20k chat pairs pulled the greeting/unknown intents


def chat_ok(user: str, assistant: str, facts: dict, max_words: int = MAX_ASSISTANT_WORDS) -> str | None:
    """Why a pair is rejected, or None when it can be Cubby's answer."""
    if not (1 <= len(user.split()) <= MAX_USER_WORDS):
        return "user length"
    if not (MIN_ASSISTANT_WORDS <= len(assistant.split()) <= max_words):
        return "assistant length"
    if URL.search(user) or URL.search(assistant):
        return "url"
    if NON_LATIN.search(user) or NON_LATIN.search(assistant):   # arena leaks CJK/Cyrillic mid-answer; Cubby is EN/FR
        return "non-latin"
    if is_model_guard(assistant) or OTHER_ASSISTANT.search(assistant):
        return "base-model guard / other assistant"
    if not voice_ok(assistant, facts):
        return "voice rule"
    if is_identity_reply(assistant, facts):
        return "identity"
    if EXPLICIT.search(user) or EXPLICIT.search(assistant):
        return "explicit"
    if "```" in assistant or assistant.count("\n") > 6:
        return "code/format"
    if CODE_LEAK.search(assistant) or LATEX.search(assistant):   # v5 lesson: Python leaked into role-binding programs; v6: LaTeX from FineInstructions
        return "code/format"
    return None


def iter_orca_pairs(path: str = ORCA, limit_lines: int | None = None):
    """(user, assistant) from chunk-0 records whose messages list parses
    whole and holds exactly one user + one assistant turn."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit_lines and i >= limit_lines:
                return
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("chunk_index", 0) != 0 or not r["text"].startswith("messages: "):
                continue
            try:
                msgs = json.loads(r["text"][len("messages: "):].strip())
            except ValueError:
                continue
            turns = [m for m in msgs if m.get("role") in ("user", "assistant")]
            if len(turns) == 2 and turns[0]["role"] == "user" and turns[1]["role"] == "assistant":
                yield turns[0]["content"].strip(), turns[1]["content"].strip()


def build_chat(rng: random.Random, facts: dict, n: int = N_CHAT, limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    for user, assistant in iter_orca_pairs(limit_lines=limit_lines):
        reason = chat_ok(user, assistant, facts)
        if reason:
            why[reason] += 1
            continue
        state = sample_state(rng)
        out.append({"id": f"chat:orca:{len(out)}", "task": "chat", "subtype": "orca", "source": "orca_agentinstruct",
                    "prompt": user, "program": assistant, "gold": None,
                    "system": identity_system(facts, state), "state": state, "lang": "en"})
        if len(out) >= n:
            break
    return out, why


# ── content awareness: passage -> label ─────────────────────────────────────
PROMPT_EN = ("Is this passage safe for a general audience? Answer with one word first, safe or nsfw, "
             "then a few words on why.\n\nPassage: {p}")
PROMPT_FR = ("Ce passage convient-il à tout public ? Réponds d'abord par un mot, safe ou nsfw, "
             "puis explique en quelques mots.\n\nPassage : {p}")


def _passage(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= PASSAGE_WORDS:
        return " ".join(words)
    start = rng.randint(0, len(words) - PASSAGE_WORDS)
    return " ".join(words[start:start + PASSAGE_WORDS])


def label_passage(text: str) -> str | None:
    """`nsfw` when explicit cues are present, `safe` when none — None when
    the minors/non-consent screens fire (dropped, never used)."""
    if MINORS.search(text) and EXPLICIT.search(text) or NONCON.search(text):
        return None
    return "nsfw" if len(EXPLICIT.findall(text)) >= 2 else ("safe" if not EXPLICIT.search(text) else None)


def iter_texts(path: str, limit_lines: int | None = None):
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit_lines and i >= limit_lines:
                return
            try:
                yield json.loads(line)["text"]
            except (ValueError, KeyError):
                continue


def build_content(rng: random.Random, n_per_label: int = N_CONTENT, limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    got = {"nsfw": 0, "safe": 0}
    sources = [("nsfw", NSFW_EXPLICIT, "en"), ("nsfw", NSFW_ALT, "en"), ("safe", SFW_WEB, "en"), ("safe", SFW_NEWS, "auto")]
    for want, path, lang in sources:
        if not os.path.exists(path):
            why[f"missing:{os.path.basename(path)}"] += 1
            continue
        for text in iter_texts(path, limit_lines):
            if got[want] >= n_per_label:
                break
            if len(text.split()) < 40:
                why["short"] += 1
                continue
            p = _passage(text, rng)
            lab = label_passage(p)
            if lab is None:
                why["screened"] += 1
                continue
            if lab != want:
                why[f"label!=source:{want}"] += 1
                continue
            fr = lang == "auto" and bool(re.search(r"\b(le|la|les|des|est|une|dans|pour)\b", p)) and \
                bool(re.search(r"[éèêàçù]", p))
            prompt = (PROMPT_FR if fr else PROMPT_EN).format(p=p)
            if lab == "nsfw":
                resp = "nsfw — explicit sexual content." if not fr else "nsfw — contenu sexuel explicite."
            else:
                resp = "safe — general content." if not fr else "safe — contenu tout public."
            out.append({"id": f"content:{lab}:{got[lab]}", "task": "content", "subtype": lab,
                        "source": os.path.basename(path).split(".")[0], "prompt": prompt, "program": resp,
                        "gold": lab, "system": None, "state": None, "lang": "fr" if fr else "en"})
            got[lab] += 1
    return out, why


# ── emotion recognition (GoEmotions): the appraisal stage as a learned task ─
HF_GOEMOTIONS = os.path.join(HF_DIR, "goemotions_train.parquet")   # google-research-datasets/go_emotions simplified
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]
# the same 28 onto the compass petals the game already draws (Plutchik)
GOEMOTIONS_PETAL = {"admiration": "trust", "amusement": "joy", "anger": "anger", "annoyance": "anger",
                    "approval": "trust", "caring": "trust", "confusion": "surprise", "curiosity": "anticipation",
                    "desire": "anticipation", "disappointment": "sadness", "disapproval": "disgust", "disgust": "disgust",
                    "embarrassment": "sadness", "excitement": "joy", "fear": "fear", "gratitude": "trust",
                    "grief": "sadness", "joy": "joy", "love": "joy", "nervousness": "fear", "optimism": "anticipation",
                    "pride": "joy", "realization": "surprise", "relief": "joy", "remorse": "sadness",
                    "sadness": "sadness", "surprise": "surprise", "neutral": "calm"}
EMOTION_PROMPT = ("What emotion does this message express? Answer with the emotion first (one word, or two "
                  "separated by a comma), then the Plutchik petal it sits on.\n\nMessage: {t}")
EMOTION_PROMPT_FR = ("Quelle émotion ce message exprime-t-il ? Réponds d'abord par l'émotion (un mot, ou deux "
                     "séparés par une virgule), puis le pétale de Plutchik.\n\nMessage : {t}")
GOEMOTIONS_FR = dict(zip(GOEMOTIONS, [
    "admiration", "amusement", "colère", "agacement", "approbation", "sollicitude", "confusion", "curiosité",
    "désir", "déception", "désapprobation", "dégoût", "embarras", "excitation", "peur", "gratitude", "chagrin",
    "joie", "amour", "nervosité", "optimisme", "fierté", "prise de conscience", "soulagement", "remords",
    "tristesse", "surprise", "neutre"]))
EMOTION_CAP = {"neutral": 300}
EMOTION_CAP_DEFAULT = 180
HF_GOEMOTIONS_MULTI = os.path.join(HF_DIR, "goemotions_multi_train.parquet")   # AnasAlokla/multilingual_go_emotions (fr slice)


def emotion_record(text: str, label_ids: list[int], i: int, lang: str = "en") -> dict | None:
    """One message -> "emotion(, emotion) — petal(/petal)". French records
    answer with the French emotion names; `gold_any` carries both so the
    eval accepts either. The petal vocabulary is the compass's (English)."""
    if not text:                                         # the multilingual copy has empty rows
        return None
    labels = [GOEMOTIONS[j] for j in label_ids if 0 <= j < len(GOEMOTIONS)]
    words = text.split()
    if not labels or not (3 <= len(words) <= 60) or len(EXPLICIT.findall(text)) >= 2:
        return None
    petals = []
    for l in labels:
        p = GOEMOTIONS_PETAL[l]
        if p not in petals:
            petals.append(p)
    names = [GOEMOTIONS_FR[l] for l in labels] if lang == "fr" else labels
    resp = ", ".join(names[:2]) + " — " + "/".join(petals[:2])
    prompt = (EMOTION_PROMPT_FR if lang == "fr" else EMOTION_PROMPT).format(t=" ".join(words))
    return {"id": f"emotion:goemotions_{lang}:{i}", "task": "emotion", "subtype": labels[0],
            "source": f"hf:goemotions_{lang}", "prompt": prompt, "program": resp, "gold": names[0],
            "gold_any": labels + [GOEMOTIONS_FR[l] for l in labels], "system": None, "state": None, "lang": lang}


def build_emotion(rng: random.Random, cap: int = EMOTION_CAP_DEFAULT) -> tuple[list[dict], Counter]:
    """GoEmotions (27 emotions + neutral, Reddit comments, apache-2.0) and
    its French translation (AnasAlokla/multilingual_go_emotions, fr slice):
    message -> emotion(s) + the Plutchik petal — a learned appraisal, the
    thing the host's lexical `appraise()` stands in for. Balanced by a
    per-label cap so the long tail (grief, relief, pride…) is learned too."""
    out, why = [], Counter()
    sources = [("en", HF_GOEMOTIONS, cap), ("fr", HF_GOEMOTIONS_MULTI, max(60, cap * 2 // 3))]
    for lang, path, lcap in sources:
        if not os.path.exists(path):
            why[f"missing:goemotions_{lang}"] += 1
            continue
        import pyarrow.parquet as pq
        cols = ["text", "labels"] + (["language"] if lang == "fr" else [])
        rows = pq.read_table(path, columns=cols).to_pylist()
        if lang == "fr":
            rows = [r for r in rows if r.get("language") == "fr"]
        rng.shuffle(rows)
        got: Counter = Counter()
        for i, r in enumerate(rows):
            ids = r["labels"]
            if isinstance(ids, str):                     # the multilingual copy stores "[11, 19]"
                try:
                    ids = json.loads(ids)
                except ValueError:
                    why[f"{lang}:bad labels"] += 1
                    continue
            rec = emotion_record(r["text"], list(ids), i, lang)
            if rec is None:
                why[f"{lang}:filtered"] += 1
                continue
            first = rec["subtype"]
            if got[first] >= (EMOTION_CAP.get(first, lcap) if lang == "en" else min(EMOTION_CAP.get(first, lcap), lcap)):
                why[f"{lang}:capped"] += 1
                continue
            got[first] += 1
            out.append(rec)
    if os.path.exists(EMOTIONS_PLUTCHIK):                # the local Plutchik-labelled messages (EN)
        for i, r in enumerate(iter_jsonl(EMOTIONS_PLUTCHIK)):
            rec = plutchik_record(r, i)
            if rec is None:
                why["plutchik:filtered"] += 1
            else:
                out.append(rec)
    else:
        why["missing:emotions_plutchik"] += 1
    if os.path.isdir(MOVIE_SCENES_DIR):                  # screenplay dialogue -> Plutchik (the spoken register)
        for i, r in enumerate(iter_movie_scenes()):
            rec = movie_scene_record(r, i)
            if rec is None:
                why["movie:filtered"] += 1
            else:
                out.append(rec)
    else:
        why["missing:movie_scenes"] += 1
    return out, why


def build_exposure(rng: random.Random, n: int, limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    """OPTIONAL (`--exposure N`, default 0): generation exposure — continue an
    explicit passage. Teaches the model to PRODUCE such prose, which is the
    thing a later gate may need to censor; the owner flips this, the builder
    never does on its own. Same screens as the recognition task."""
    out, why = [], Counter()
    for path in (NSFW_EXPLICIT, NSFW_ALT):
        if not os.path.exists(path):
            continue
        for text in iter_texts(path, limit_lines):
            if len(out) >= n:
                break
            words = text.split()
            if len(words) < 120:
                why["short"] += 1
                continue
            if label_passage(" ".join(words[:200])) != "nsfw":
                why["screened or not explicit"] += 1
                continue
            cut = len(words) // 2
            out.append({"id": f"exposure:{len(out)}", "task": "exposure", "subtype": "continue",
                        "source": os.path.basename(path).split(".")[0],
                        "prompt": "Continue this passage.\n\n" + " ".join(words[:cut]),
                        "program": " ".join(words[cut:cut + 120]), "gold": None,
                        "system": None, "state": None, "lang": "en"})
    return out, why


# ── history: dated sentences Cubby can refer to (2026-09-02, owner's ask) ───
# Sources under I:\grillcheese_training_data\temporal (all EN — the FR gap
# stands): `historical/train_augmented.jsonl` (4,000 dated world-history
# events, ~1.3k distinct titles), `historical_events_1800-1900.jsonl` (36
# full-date summaries), the three "10k_years" Q&A sets (only the half whose
# answer is not a book summary — "the text discusses…" is dropped), the
# arkona student/expert dialogues on 2.6k historical persons, and the NYT
# archive (294 month files, 1851..2024) as BOTH a recall task ("what was in
# the news on <date>") and a dating task ("when was this reported" — the
# temporal-orientation read; scored within ±5 years).
TEMPORAL = os.path.join(DATA, "temporal")
HIST_DIR = os.path.join(TEMPORAL, "historical")
HIST_EVENTS = os.path.join(HIST_DIR, "train_augmented.jsonl")
HIST_1800 = os.path.join(HIST_DIR, "historical_events_1800-1900.jsonl")
HIST_10K = [os.path.join(HIST_DIR, n) for n in ("historical_high_confidence_831_samples.json",
                                                 "historical_modern_history_771_samples.json",
                                                 "historical_wars_conflicts_648_samples.json")]
HIST_DIALOGUE = os.path.join(HIST_DIR, "arkona_intermediate_3000_detailed.json")
NYT_DIR = os.path.join(TEMPORAL, "nyt_data")
HIST_QUOTA = {"dialogue": 2000, "per_topic": 2, "nyt_days_per_month": 6}
MAX_HISTORY_WORDS = 120
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
BOOK_VOICE = re.compile(r"\bthe (text|passage|document|book|chapter|author|excerpt)\b", re.I)
_NOUN_STOP = {"the", "this", "that", "these", "those", "during", "after", "before", "when", "while", "however",
              "although", "despite", "also", "with", "from", "into", "over", "under", "about", "their", "they",
              "there", "then", "than", "some", "many", "most", "more", "such", "which", "what", "where", "other",
              "first", "later", "early", "both", "each", "between", "around", "following", "established", "regarding",
              "certainly", "indeed", "well", "yes", "absolutely", "great", "good", "here", "his", "her", "its",
              "several", "another", "though", "because", "since", "until", "through", "without", "within"}
ABOUT_PROMPTS = ["Tell me about {t}.", "What do you know about {t}?", "Briefly, what was {t}?",
                 "Can you tell me about {t}?", "What was {t}?"]
WHEN_PROMPTS = ["When was {t}?", "When did {t} happen?", "In what year was {t}?", "What year was {t}?"]
YEAR_PROMPTS = ["What happened in {y}?", "What happened around {y}?", "Name one event from {y}.",
                "Tell me something that happened in {y}.", "Anything notable from {y}?"]
NEWS_PROMPTS = ["What was in the news on {d}?", "Any headline from {d}?", "What was reported on {d}?",
                "Give me a news item from {d}."]
DATING_PROMPTS = ["When was this reported?\n\n{h}: {a}", "Date this news item.\n\n{h}: {a}",
                  "From what date is this headline?\n\n{h}: {a}"]


def year_key(y: int) -> str:
    """The gold string for a year: 1066 -> '1066', -3100 -> '3100 BCE'."""
    return f"{abs(y)} BCE" if y < 0 else str(y)


def year_phrase(y0: int, y1: int | None = None) -> str:
    """'in 1066' / 'around 3100 BCE' / 'from 1914 to 1918'."""
    if y1 is not None and y1 != y0:
        return f"from {year_key(y0)} to {year_key(y1)}"
    return ("around " if y0 < 0 else "in ") + year_key(y0)


def proper_nouns(text: str, cap: int = 12) -> list[str]:
    """Capitalised words (4+ letters, not sentence-opening stopwords) and
    years — what an answer must mention to count as being about the same
    thing (the eval's containment check)."""
    out: list[str] = []
    for m in re.finditer(r"\b([A-Z][A-Za-z'\-]{3,}|\d{3,4}(?: BCE)?)\b", text):
        w = m.group(1)
        if w.lower() in _NOUN_STOP or w in out:
            continue
        out.append(w)
        if len(out) >= cap:
            break
    return out


def history_ok(rec: dict, gen: str, facts: dict) -> bool:
    """The eval's read of a history answer: Cubby's rules hold, and the
    answer is about the right thing — `when`: the gold year is named;
    `dating`: the first year named is within 5 years of the gold; the rest:
    one of the record's proper nouns / years appears."""
    if not gen or not voice_ok(gen, facts) or is_model_guard(gen):
        return False
    sub = rec.get("subtype")
    gold = str(rec.get("gold") or "")
    if sub == "dating":
        m = re.search(r"\b(\d{4})\b", gen)
        return bool(m) and gold[:4].isdigit() and abs(int(m.group(1)) - int(gold[:4])) <= 5
    if sub == "when":
        return re.sub(r"\s*BCE?$", "", gold) in gen if gold else False
    low = gen.lower()
    keys = rec.get("gold_any") or proper_nouns(str(rec.get("program") or ""))   # replay files carry the reference, not gold_any
    return any(str(g).lower() in low for g in keys)


def history_record(subtype: str, i: int, prompt: str, answer: str, gold: str | None, gold_any: list[str],
                   source: str, facts: dict) -> dict | None:
    """One dated record after the shared screens (length, URL, explicit,
    base-model guard, voice rules) — None when it cannot be Cubby's answer."""
    answer = " ".join(answer.split())
    prompt = prompt.strip()
    if not gold_any or not prompt:
        return None
    if not ((1 if subtype in ("dating", "quote_who", "era") else 5) <= len(answer.split()) <= MAX_HISTORY_WORDS):   # a date / a name / a period
        return None
    if URL.search(prompt) or URL.search(answer) or EXPLICIT.search(answer) or CODE_LEAK.search(answer):
        return None
    if is_model_guard(answer) or OTHER_ASSISTANT.search(answer) or not voice_ok(answer, facts):
        return None
    return {"id": f"history:{subtype}:{i}", "task": "history", "subtype": subtype, "source": source,
            "prompt": prompt, "program": answer, "gold": gold, "gold_any": gold_any,
            "system": None, "state": None, "lang": "en"}


_COMMON_HEAD = {"unification", "invention", "battle", "treaty", "founding", "fall", "rise", "discovery", "death",
                "birth", "signing", "construction", "establishment", "reign", "siege", "assassination", "publication",
                "first", "great", "end", "beginning", "introduction", "creation", "coronation", "declaration",
                "abolition", "opening", "launch", "formation", "conquest", "independence", "revolution", "partition",
                "sinking", "election", "outbreak", "spread", "development", "adoption", "completion", "collapse",
                "dissolution", "expansion", "emergence", "founding", "invasion", "sack", "plague", "eruption"}


def title_phrase(title: str) -> str:
    """'Unification of Ancient Egypt' -> 'the Unification of Ancient Egypt';
    a proper name ('Magna Carta', 'The Black Death') is left alone."""
    t = title.strip().rstrip(".")
    if t.lower().startswith("the ") or not t:
        return t
    if " of " in t or t.split()[0].lower() in _COMMON_HEAD:
        return "the " + t
    return t


def event_records(r: dict, rng: random.Random, facts: dict, i: int, first_of_year: bool) -> list[dict]:
    """A dated world-history event -> about / when (/ year) records."""
    t, text = title_phrase(r.get("title") or ""), " ".join((r.get("text") or "").split())
    y0, y1 = r.get("year_start"), r.get("year_end")
    if not t or not text or y0 is None:
        return []
    yp, gold = year_phrase(y0, y1), year_key(y0)
    nouns = proper_nouns(t + " " + " ".join(r.get("actors") or []))
    body = text if gold in text else f"{text} That was {yp}."
    recs = [history_record("about", i, rng.choice(ABOUT_PROMPTS).format(t=t), body, gold, nouns + [gold],
                           "train_augmented", facts),
            history_record("when", i, rng.choice(WHEN_PROMPTS).format(t=t), f"{t[0].upper() + t[1:]} was {yp}.", gold, [gold],
                           "train_augmented", facts)]
    if first_of_year:
        recs.append(history_record("year", i, rng.choice(YEAR_PROMPTS).format(y=gold), body, gold, nouns,
                                   "train_augmented", facts))
    return [x for x in recs if x]


def dialogue_pairs(rows: list[dict]):
    """Consecutive student -> expert turns on the same person; a follow-up
    that does not name its subject gets it prefixed so the pair stands alone."""
    for a, b in zip(rows, rows[1:]):
        if a.get("speaker") != "student" or b.get("speaker") != "expert" or a.get("topic") != b.get("topic"):
            continue
        topic = (a.get("topic") or "").strip()
        q, ans = " ".join((a.get("message") or "").split()), " ".join((b.get("message") or "").split())
        if not topic or not q or not ans:
            continue
        if topic.split()[-1].lower() not in q.lower():
            q = f"Regarding {topic}: {q}"
        yield topic, q, ans


def clean_headline(main: str) -> str | None:
    h = (main or "").split(";")[0].strip().rstrip(".").strip()
    if h.isupper():
        h = h.title()
    if len(h.split()) < 2 or re.search(r"paid notice|no headline", h, re.I):
        return None
    return h


def nyt_records(r: dict, rng: random.Random, facts: dict, i: int) -> list[dict]:
    """One NYT item -> a recall record (date -> headline: abstract) and a
    dating record (headline: abstract -> date)."""
    h = clean_headline((r.get("headline") or {}).get("main") or "")
    a = re.sub(r"^\s*LEAD:\s*", "", " ".join((r.get("abstract") or "").split()))
    pub = (r.get("pub_date") or "")[:10]
    if not h or not a or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pub):
        return []
    if not (8 <= len(a.split()) <= 80) or a.endswith("...") or not a[0].isupper() or a.lower() == h.lower():
        return []
    y, m, d = int(pub[:4]), int(pub[5:7]), int(pub[8:10])
    date = f"{MONTHS[m - 1]} {d}, {y}"
    nouns = proper_nouns(h + " " + a)
    recs = [history_record("news", i, rng.choice(NEWS_PROMPTS).format(d=date), f"{h}: {a}", str(y),
                           nouns + [str(y)], "nyt_archive", facts),
            history_record("dating", i, rng.choice(DATING_PROMPTS).format(h=h, a=a), f"{date}.", str(y), [str(y)],
                           "nyt_archive", facts)]
    return [x for x in recs if x]


def iter_jsonl(path: str, limit: int | None = None):
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            try:
                yield json.loads(line)
            except ValueError:
                continue


def build_history(rng: random.Random, facts: dict, quota: dict | None = None,
                  limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    """All history sources through `history_record`. `limit_lines` (debug)
    reads only the first NYT month files. Missing sources are reported."""
    quota = quota or HIST_QUOTA
    out, why = [], Counter()
    # 1. dated world-history events (dedupe on title; one `year` record per year)
    if os.path.exists(HIST_EVENTS):
        seen_t, seen_y = set(), set()
        for i, r in enumerate(iter_jsonl(HIST_EVENTS)):
            t = (r.get("title") or "").strip().lower()
            if not t or t in seen_t:
                why["events:dup title"] += 1
                continue
            seen_t.add(t)
            y0 = r.get("year_start")
            first = y0 not in seen_y
            seen_y.add(y0)
            recs = event_records(r, rng, facts, i, first_of_year=first)
            why["events:screened"] += (3 if first else 2) - len(recs)
            out += recs
    else:
        why["missing:train_augmented"] += 1
    # 2. the 36 full-date 1800s events
    if os.path.exists(HIST_1800):
        for i, r in enumerate(iter_jsonl(HIST_1800)):
            s = " ".join((r.get("summary") or "").split())
            m = re.match(r"On ((?:" + "|".join(MONTHS) + r") \d{1,2}, (\d{4})),", s)
            if m:
                prompt, gold = f"What happened on {m.group(1)}?", m.group(2)
            elif r.get("earliest_date_year"):
                gold = str(r["earliest_date_year"])
                prompt = rng.choice(YEAR_PROMPTS).format(y=gold)
            else:
                why["1800s:no date"] += 1
                continue
            rec = history_record("event", i, prompt, s, gold, proper_nouns(s), "historical_events_1800-1900", facts)
            if rec:
                out.append(rec)
            else:
                why["1800s:screened"] += 1
    # 3. the 10k-years Q&A sets: answers in their own voice, dated by the answer itself
    for path in HIST_10K:
        if not os.path.exists(path):
            why[f"missing:{os.path.basename(path)}"] += 1
            continue
        name = os.path.basename(path).split("_")[1]
        rows = json.load(open(path, encoding="utf-8", errors="replace")).get("training_data", [])
        for i, r in enumerate(rows):
            ans = " ".join((r.get("answer") or "").split())
            if BOOK_VOICE.search(ans):
                why[f"10k:{name}:book voice"] += 1
                continue
            m = re.match(r"In (\d{3,4})( BCE)?, (.+)", ans)
            if m:
                gold = m.group(1) + (m.group(2) or "")
            elif r.get("year") is not None and str(r["year"]) in ans:
                gold = str(r["year"]) + (" BCE" if re.search(rf"\b{r['year']} BCE?\b", ans) else "")
            else:
                why[f"10k:{name}:undated"] += 1
                continue
            rec = history_record("year", 10000 + i, rng.choice(YEAR_PROMPTS).format(y=gold), ans, gold,
                                 proper_nouns(ans), f"10k_years:{name}", facts)
            if rec:
                out.append(rec)
            else:
                why[f"10k:{name}:screened"] += 1
    # 4. expert dialogues on historical persons (spread over topics)
    if os.path.exists(HIST_DIALOGUE):
        pairs = list(dialogue_pairs(json.load(open(HIST_DIALOGUE, encoding="utf-8", errors="replace"))))
        rng.shuffle(pairs)
        per_topic: Counter = Counter()
        kept = 0
        for i, (topic, q, ans) in enumerate(pairs):
            if kept >= quota["dialogue"]:
                break
            if per_topic[topic] >= quota["per_topic"]:
                why["dialogue:topic cap"] += 1
                continue
            rec = history_record("dialogue", i, q, ans, None, proper_nouns(ans + " " + topic), "arkona_dialogues", facts)
            if rec is None:
                why["dialogue:screened"] += 1
                continue
            per_topic[topic] += 1
            kept += 1
            out.append(rec)
    else:
        why["missing:arkona"] += 1
    # 4b. expert turns on historical events (knowledgetxt) — the same dialogue subtype
    if os.path.exists(INDIVIDUAL_EVENTS):
        ev = list(iter_jsonl(INDIVIDUAL_EVENTS))
        for i, (a, b) in enumerate(zip(ev, ev[1:])):
            if a.get("role") != "student" or b.get("role") != "expert":
                continue
            rec = history_record("dialogue", 20000 + i, " ".join((a.get("text") or "").split()),
                                 " ".join((b.get("text") or "").split()), None,
                                 proper_nouns(b.get("text") or ""), "individual_events", facts)
            if rec:
                out.append(rec)
            else:
                why["events_dialogue:screened"] += 1
    else:
        why["missing:individual_events"] += 1
    # 4c. historical quotes: about a theme / who said it (spread over authors)
    if os.path.exists(QUOTES):
        try:
            qs = json.load(open(QUOTES, encoding="utf-8", errors="replace"))
            qs = qs if isinstance(qs, list) else next((v for v in qs.values() if isinstance(v, list)), [])
        except ValueError:
            qs, why["quotes:bad file"] = [], 1
        rng.shuffle(qs)
        per_author: Counter = Counter()
        got = Counter()
        for i, r in enumerate(qs):
            if got["quote_about"] >= QUOTE_QUOTA["quote_about"] and got["quote_who"] >= QUOTE_QUOTA["quote_who"]:
                break
            author = (r.get("author") or "").strip()
            if per_author[author] >= QUOTE_QUOTA["per_author"]:
                why["quotes:author cap"] += 1
                continue
            recs = quote_records(r, rng, facts, i, want_about=got["quote_about"] < QUOTE_QUOTA["quote_about"],
                                 want_who=got["quote_who"] < QUOTE_QUOTA["quote_who"])
            if not recs:
                why["quotes:screened"] += 1
                continue
            per_author[author] += 1
            for x in recs:
                got[x["subtype"]] += 1
            out += recs
    else:
        why["missing:quotes"] += 1
    # 5. NYT archive: a few distinct days per month, every month on disk
    files = sorted(f for f in os.listdir(NYT_DIR) if f.endswith(".json")) if os.path.isdir(NYT_DIR) else []
    if not files:
        why["missing:nyt_data"] += 1
    if limit_lines:
        files = files[:max(1, limit_lines // 1000)]
    for k, name in enumerate(files):
        try:
            items = json.load(open(os.path.join(NYT_DIR, name), encoding="utf-8", errors="replace"))
        except ValueError:
            why["nyt:bad file"] += 1
            continue
        rng.shuffle(items)
        days: set[str] = set()
        for r in items:
            pub = (r.get("pub_date") or "")[:10]
            if pub in days:
                continue
            recs = nyt_records(r, rng, facts, k * 100 + len(days))
            if not recs:
                why["nyt:screened"] += 1
                continue
            days.add(pub)
            out += recs
            if len(days) >= quota["nyt_days_per_month"]:
                break
    return out, why


# ── the sorted local sources (2026-09-02, owner: "there's a lot of useful data
# for chat" in knowledgetxt, "here they are all sorted" in E:\datasets) ─────
# Chat: chatbot_arena (33k REAL human<->LLM convos, moderation-flagged rows
# already dropped), the topic-tagged conversation set, alpaca-style
# instruct_55k, FineInstructions (factual Q&A, head of an 8 GB file), WikiQA,
# a grammar-fix set, the emotion-adapted replies (ei_11). Emotion: the
# Plutchik-labelled messages. Affect: valence/arousal-rated messages (the
# neurochemistry's own drive space). History: expert turns on historical
# events + 24k historical quotes with authors. Everything passes the same
# gate as the HF chat; long-answer sources get their own word cap.
E_DATASETS = r"E:\datasets"
DOMAINS = os.path.join(E_DATASETS, "domains")
KNOWLEDGE = os.path.join(DATA, "knowledgetxt")
ARENA = os.path.join(DOMAINS, "conversation", "chatbot_arena.jsonl")
CONVO = os.path.join(KNOWLEDGE, "conversation.jsonl")
INSTRUCT_55K = os.path.join(KNOWLEDGE, "instruct_55k_clean.jsonl")
NEMOTRON_FI = os.path.join(DOMAINS, "instruction", "nemotron_fineinstructions.jsonl")
WIKIQA = os.path.join(DOMAINS, "QA", "wikiqa.jsonl")
GRAMMAR = os.path.join(KNOWLEDGE, "combined_grammar.jsonl")
EI = os.path.join(KNOWLEDGE, "ei_11.jsonl")
EMOTIONS_PLUTCHIK = os.path.join(KNOWLEDGE, "emotions.jsonl")
MOVIE_SCENES_DIR = r"H:\AURA_GENESIS\datasets\movie_annotated"   # 5 screenplays, 365 scenes: dialogue -> Plutchik base + label + score
AFFECT_FILES = [("realm_phase", os.path.join(KNOWLEDGE, "emotion_valence_arousal_realm_phase.jsonl")),
                ("amygdala", os.path.join(KNOWLEDGE, "amygdala_affect.jsonl")),
                ("convos", os.path.join(KNOWLEDGE, "affect_from_convos.jsonl"))]
INDIVIDUAL_EVENTS = os.path.join(KNOWLEDGE, "conversations_individual_events.jsonl")
QUOTES = os.path.join(E_DATASETS, "historical-quotes", "english_historical_quotes.json")
# KonstantyM/science_qa_prep (HF datasets cache, 4.28M rows): measured 2026-09-03 — an OpenOrca/FLAN-style
# instruction mix (movie plots, article MCQs, task definitions) with exactly 238 science Q&A rows
# (`context: tag/<topic>/ question: …`, the wtamu "surprising answers" set, all in shard 0). Only those are
# used; the mix duplicates Orca + FineInstructions and carries base-model system prompts.
SCIENCE_QA_DIR = r"H:\datasets_facts\KonstantyM___science_qa_prep\default\0.0.0\9dea632d62d523d0a3fd2bf027950821f95090c2"
SCIENCE_QA_SHARD = os.path.join(SCIENCE_QA_DIR, "science_qa_prep-train-00000-of-00015.arrow")
SCIENCE_PAIR = re.compile(r"^context:\s*tag/([^/\s]+)/?\s*question:\s*(.*)$", re.S)
LOCAL_QUOTA = {"arena": 2000, "convo": 3000, "instruct": 2000, "nemotron": 2000, "wikiqa": 1500, "grammar": 800, "ei": 800,
               "science": 300}
LOCAL_CAP = {"instruct": 150, "nemotron": 120, "ei": 150, "science": 120}   # word caps above the chat default
NEMOTRON_SCAN = 60000                                              # lines read from the head of the 8 GB file
QUOTE_QUOTA = {"quote_about": 1200, "quote_who": 1200, "per_author": 3}
NON_LATIN = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
ARENA_PAIR = re.compile(r"^user: (.*?)\nassistant: (.*?)(?=\nuser: |\Z)", re.S)
NEMOTRON_PAIR = re.compile(r"^Instruction: (.*?)\n\nAnswer: (.*)$", re.S)
GRAMMAR_PROMPTS = ["Is this sentence right? \"{s}\"", "Can you fix this sentence: \"{s}\"", "What's wrong with this: \"{s}\"",
                   "Check my grammar: \"{s}\""]
QUOTE_PROMPTS = ["Give me a quote about {c}.", "Do you know a quote about {c}?", "A famous line about {c}?",
                 "Any wise words on {c}?", "Quote me something about {c}."]
WHO_PROMPTS = ["Who said: \"{q}\"?", "Who is this quote from? \"{q}\"", "\"{q}\" — who wrote that?"]
PLUTCHIK = {"joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"}
AFFECT_PROMPT = ("How does this message feel? Give its valence from -1 (very negative) to +1 (very positive) and its "
                 "arousal from 0 (calm) to 1 (agitated).\n\nMessage: {t}")


def pair_from_arena(text: str) -> tuple[str, str] | None:
    """The first user -> assistant exchange of a 'user: …\\nassistant: …' transcript."""
    m = ARENA_PAIR.match(text or "")
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def pair_from_nemotron(text: str) -> tuple[str, str] | None:
    m = NEMOTRON_PAIR.match((text or "").strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def first_sentences(text: str, cap: int) -> str:
    """The leading sentences that fit in `cap` words (a long explanation
    keeps its opening, which states the answer, and stays coherent)."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", " ".join((text or "").split())):
        if len(" ".join(out + [sent]).split()) > cap:
            break
        out.append(sent)
    return " ".join(out)


def science_pair(r: dict, cap: int = 110) -> tuple[str, str] | None:
    """science_qa_prep: 'context: tag/<topic>/ question: <q>' -> (q, the answer's opening sentences)."""
    m = SCIENCE_PAIR.match(r.get("input") or "")
    if not m:
        return None
    q, a = " ".join(m.group(2).split()), first_sentences(r.get("label") or "", cap)
    return (q, a) if q and a else None


def iter_arrow(path: str):
    """Rows of one HF-cache arrow shard (pyarrow IPC stream)."""
    import pyarrow.ipc as ipc
    with open(path, "rb") as f:
        yield from ipc.open_stream(f).read_all().to_pylist()


def wikiqa_pair(r: dict) -> tuple[str, str] | None:
    q, a = " ".join((r.get("question") or "").split()), " ".join((r.get("answer") or "").split())
    if not q or not a:
        return None
    if q.isupper():
        q = q.capitalize()
    q = q[0].upper() + q[1:]
    if not q.endswith("?"):
        q += "?"
    return q, a


def grammar_pair(r: dict) -> tuple[str, str] | None:
    """incorrect -> explanation + the corrected sentence."""
    inc, cor, exp = (" ".join((r.get(k) or "").split()) for k in ("incorrect_example", "correct_example", "explanation"))
    if not (inc and cor and exp) or inc == cor:
        return None
    prompt = GRAMMAR_PROMPTS[zlib.crc32(inc.encode("utf-8")) % len(GRAMMAR_PROMPTS)].format(s=inc)
    return prompt, f"{exp} Better: \"{cor}\""


def _loose_dict(s) -> dict:
    """ei_11 stores dicts as Python/JSON-ish strings; read what can be read."""
    if isinstance(s, dict):
        return s
    for loader in (json.loads, ast.literal_eval):
        try:
            d = loader(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    out = {}
    for key in ("query", "content", "label"):
        m = re.search(r"['\"]" + key + r"['\"]\s*:\s*(['\"])(.*?)\1\s*[,}]", s or "", re.S)
        if m:
            out[key] = m.group(2)
    return out


def ei_pair(r: dict) -> tuple[str, str] | None:
    q = _loose_dict(r.get("context_input")).get("query")
    a = _loose_dict(r.get("emotion_adapted_response")).get("content")
    if not q or not a:
        return None
    return " ".join(str(q).split()), " ".join(str(a).split())


def build_local_chat(rng: random.Random, facts: dict, quota: dict | None = None,
                     limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    """The local chat sources through the live gate; per-source word caps."""
    quota = quota or LOCAL_QUOTA
    out, why = [], Counter()

    def take(pairs, name: str):
        kept, cap = 0, LOCAL_CAP.get(name, MAX_ASSISTANT_WORDS)
        for user, assistant in pairs:
            if kept >= quota[name]:
                break
            reason = chat_ok(user, assistant, facts, max_words=cap)
            if reason:
                why[f"{name}:{reason}"] += 1
                continue
            state = sample_state(rng)
            out.append({"id": f"chat:{name}:{kept}", "task": "chat", "subtype": name, "source": f"local:{name}",
                        "prompt": user, "program": assistant, "gold": None,
                        "system": identity_system(facts, state), "state": state, "lang": "en"})
            kept += 1

    def shuffled(path, limit=None):
        rows = list(iter_jsonl(path, limit))
        rng.shuffle(rows)
        return rows

    sources = [
        ("arena", ARENA, lambda p: (x for x in (pair_from_arena(r.get("text")) for r in shuffled(p, limit_lines)) if x)),
        ("convo", CONVO, lambda p: pairs_from_messages(shuffled(p, limit_lines), key="turns")),
        ("instruct", INSTRUCT_55K, lambda p: ((r.get("prompt") or "", r.get("response") or "") for r in shuffled(p, limit_lines))),
        ("nemotron", NEMOTRON_FI, lambda p: (x for x in (pair_from_nemotron(r.get("text")) for r in shuffled(p, limit_lines or NEMOTRON_SCAN)) if x)),
        ("wikiqa", WIKIQA, lambda p: (x for x in (wikiqa_pair(r) for r in shuffled(p, limit_lines)) if x)),
        ("grammar", GRAMMAR, lambda p: (x for x in (grammar_pair(r) for r in shuffled(p, limit_lines)) if x)),
        ("ei", EI, lambda p: (x for x in (ei_pair(r) for r in shuffled(p, limit_lines)) if x)),
        ("science", SCIENCE_QA_SHARD, lambda p: (x for x in (science_pair(r) for r in iter_arrow(p)) if x)),
    ]
    for name, path, pairs in sources:
        if not os.path.exists(path):
            why[f"missing:{name}"] += 1
            continue
        take(pairs(path), name)
    return out, why


def plutchik_record(r: dict, i: int) -> dict | None:
    """emotions.jsonl: message -> Plutchik primary (+ secondary) in the emotion
    task's own format; the primary IS the petal."""
    p = r.get("plutchik") or {}
    primary, text = p.get("primary"), " ".join((r.get("text") or "").split())
    if not isinstance(primary, str) or primary not in PLUTCHIK or not text:
        return None
    words = text.split()
    if not (3 <= len(words) <= 60) or len(EXPLICIT.findall(text)) >= 2:
        return None
    sec = p.get("secondary") if isinstance(p.get("secondary"), str) else None
    names = [primary] + ([sec] if sec and sec != primary else [])
    petals = [primary]
    if sec in GOEMOTIONS_PETAL and GOEMOTIONS_PETAL[sec] not in petals:
        petals.append(GOEMOTIONS_PETAL[sec])
    return {"id": f"emotion:plutchik:{i}", "task": "emotion", "subtype": primary, "source": "local:emotions_plutchik",
            "prompt": EMOTION_PROMPT.format(t=text), "program": ", ".join(names) + " — " + "/".join(petals),
            "gold": primary, "gold_any": names + [GOEMOTIONS_FR.get(primary, primary)],
            "system": None, "state": None, "lang": "en"}


def movie_scene_record(r: dict, i: int) -> dict | None:
    """One annotated scene -> the emotion task on its DIALOGUE: '<label> — <base>'
    (the base emotion is the Plutchik petal; the label is the finer word:
    'regret — sadness', 'terror — fear'). Scenes without dialogue are skipped."""
    base = str(r.get("main_base_emotion") or "").strip().lower()
    label = str(r.get("plutchik_label") or "").strip().lower()
    text = " ".join(str(r.get("full_dialogue_context") or "").split())
    if not text:
        lines = [str(d.get("line") or "") for d in (r.get("annotated_dialogue") or []) if isinstance(d, dict)]
        text = " ".join(" ".join(lines).split())
    if base not in PLUTCHIK or not text or not (3 <= len(text.split()) <= 90) or len(EXPLICIT.findall(text)) >= 2:
        return None
    names = [label] if label and label != base else []
    names.append(base)
    return {"id": f"emotion:movie:{i}", "task": "emotion", "subtype": base, "source": "local:movie_scenes",
            "prompt": EMOTION_PROMPT.format(t=text), "program": ", ".join(names[:2]) + " — " + base,
            "gold": base, "gold_any": names + [GOEMOTIONS_FR.get(base, base)],
            "system": None, "state": None, "lang": "en"}


def iter_movie_scenes(root: str = MOVIE_SCENES_DIR):
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        if name.endswith(".json"):
            try:
                for r in json.load(open(os.path.join(root, name), encoding="utf-8", errors="replace")):
                    yield r
            except ValueError:
                continue


def affect_answer(v: float, a: float) -> str:
    return f"valence {v:+.1f}, arousal {a:.1f}"


def affect_record(text: str, v, a, i: int, source: str) -> dict | None:
    """A valence/arousal-rated message -> the two numbers (the ODE's drives)."""
    text = " ".join((text or "").split())
    try:
        v, a = float(v), float(a)
    except (TypeError, ValueError):
        return None
    if not text or "[" in text or not (3 <= len(text.split()) <= 60) or not (-1 <= v <= 1 and 0 <= a <= 1):
        return None
    if len(EXPLICIT.findall(text)) >= 2:
        return None
    return {"id": f"affect:{source}:{i}", "task": "affect", "subtype": source, "source": f"local:{source}",
            "prompt": AFFECT_PROMPT.format(t=text), "program": affect_answer(v, a), "gold": affect_answer(v, a),
            "gold_any": [v, a], "system": None, "state": None, "lang": "en"}


def parse_affect(gen: str) -> tuple[float, float] | None:
    m = re.search(r"valence\s*[:=]?\s*([+-]?\d*\.?\d+)", gen or "", re.I)
    n = re.search(r"arousal\s*[:=]?\s*([+-]?\d*\.?\d+)", gen or "", re.I)
    if m and n:
        return float(m.group(1)), float(n.group(1))
    nums = re.findall(r"[+-]?\d*\.?\d+", gen or "")
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def affect_ok(rec: dict, gen: str, tol: float = 0.35) -> bool:
    """Both numbers within `tol` of the rated ones."""
    p = parse_affect(gen)
    if p is None:
        return False
    ga = rec.get("gold_any") or parse_affect(str(rec.get("gold") or rec.get("program") or ""))   # replay files carry the reference
    v, a = (list(ga) if ga else [None, None])[:2]
    return v is not None and abs(p[0] - v) <= tol and abs(p[1] - a) <= tol


def build_affect(rng: random.Random) -> tuple[list[dict], Counter]:
    """The three valence/arousal files. amygdala's exact-zero valences (58% of
    it) are dropped as unrated; the convo file's placeholder rows too."""
    out, why = [], Counter()
    for name, path in AFFECT_FILES:
        if not os.path.exists(path):
            why[f"missing:{name}"] += 1
            continue
        for i, r in enumerate(iter_jsonl(path)):
            aff = r.get("affect") or r
            v, a = aff.get("valence"), aff.get("arousal")
            if name == "amygdala" and v == 0:
                why["amygdala:zero valence"] += 1
                continue
            rec = affect_record(r.get("text"), v, a, i, name)
            if rec is None:
                why[f"{name}:screened"] += 1
                continue
            out.append(rec)
    rng.shuffle(out)
    return out, why


def quote_records(r: dict, rng: random.Random, facts: dict, i: int, want_about: bool = True,
                  want_who: bool = True) -> list[dict]:
    """A historical quote -> 'a quote about <category>' and 'who said …'."""
    q, author = " ".join((r.get("quote") or "").split()).strip("\"“” "), " ".join((r.get("author") or "").split())
    if not q or not author or not (4 <= len(q.split()) <= 60) or author.lower() in ("unknown", "anonymous"):
        return []
    try:
        cats = [c for c in ast.literal_eval(r.get("category") or "[]") if isinstance(c, str) and c.strip()]
    except (ValueError, SyntaxError):
        cats = []
    recs = []
    if want_about and cats:
        recs.append(history_record("quote_about", i, rng.choice(QUOTE_PROMPTS).format(c=rng.choice(cats)),
                                   f"\u201c{q}\u201d — {author}", author, [author] + proper_nouns(q), "historical_quotes", facts))
    if want_who:
        recs.append(history_record("quote_who", i, rng.choice(WHO_PROMPTS).format(q=q), f"{author}.", author, [author],
                                   "historical_quotes", facts))
    return [x for x in recs if x]


# ── verified facts (E:\datasets\domains\verified_facts): 4,322 history books
# sorted by era and subject — the tree IS the verified label. Used as a
# period-orientation task: passage -> era (+ subject), scored on the era. ─
VERIFIED_FACTS = os.path.join(DOMAINS, "verified_facts")
ERAS = {"1. Prehistory": ("prehistory", ["prehistor"]),
        "AncientClassical": ("the ancient and classical world", ["ancient", "classical", "antiquity"]),
        "MiddleAges": ("the Middle Ages", ["middle ages", "medieval"])}
ERA_PROMPT = ("Which period of history is this passage about — prehistory, the ancient and classical world, or the "
              "Middle Ages? Name the period first, then its subject if you can tell.\n\nPassage: {p}")
ERA_PER_ERA = 600
ERA_SKIP_SUBJECTS = {"miscellaneous", ""}
FRONT_MATTER = re.compile(r"\b(isbn|copyright|all rights reserved|published by|library of congress|printed in|"
                          r"first published|cataloguing|routledge|university press|www\.|http|contents|acknowledg)\b", re.I)


def era_passage(text: str, rng: random.Random) -> str | None:
    """A clean window from the middle of a book (front matter and indexes
    skipped): mostly letters, few digits, no publishing boilerplate."""
    words = text.split()
    if len(words) < 800:
        return None
    for _ in range(6):
        start = rng.randint(len(words) // 5, max(len(words) // 5, len(words) * 4 // 5 - PASSAGE_WORDS))
        p = " ".join(words[start:start + PASSAGE_WORDS])
        letters = sum(c.isalpha() for c in p)
        digits = sum(c.isdigit() for c in p)
        if letters >= 0.72 * len(p) and digits <= 0.03 * len(p) and not FRONT_MATTER.search(p) and p.count("[") < 2:
            return p
    return None


def era_record(era_key: str, subject: str, passage: str, i: int, facts: dict) -> dict | None:
    name, words = ERAS[era_key]
    subj = subject.strip()
    answer = f"{name[0].upper() + name[1:]} — {subj}." if subj.lower() not in ERA_SKIP_SUBJECTS else f"{name[0].upper() + name[1:]}."
    return history_record("era", i, ERA_PROMPT.format(p=passage), answer, name, list(words), "verified_facts:" + era_key, facts)


def build_era(rng: random.Random, facts: dict, per_era: int = ERA_PER_ERA, limit_lines: int | None = None) -> tuple[list[dict], Counter]:
    out, why = [], Counter()
    if not os.path.isdir(VERIFIED_FACTS):
        why["missing:verified_facts"] += 1
        return out, why
    for era_key in ERAS:
        root = os.path.join(VERIFIED_FACTS, era_key)
        files = [os.path.join(dp, f) for dp, _, fn in os.walk(root) for f in fn if f.endswith(".txt")]
        rng.shuffle(files)
        if limit_lines:
            files = files[:max(3, limit_lines // 1000)]
        kept = 0
        for i, fp in enumerate(files):
            if kept >= per_era:
                break
            try:
                text = open(fp, encoding="utf-8", errors="replace").read()
            except OSError:
                why[f"{era_key}:unreadable"] += 1
                continue
            p = era_passage(text, rng)
            if p is None:
                why[f"{era_key}:no clean window"] += 1
                continue
            subject = os.path.relpath(fp, root).split(os.sep)[0] if os.sep in os.path.relpath(fp, root) else ""
            rec = era_record(era_key, subject, p, i, facts)
            if rec is None:
                why[f"{era_key}:screened"] += 1
                continue
            kept += 1
            out.append(rec)
    return out, why


def finish(records: list[dict], facts: dict | None = None) -> list[dict]:
    """Dedupe on the prompt, split, repeat weights — and a system prompt on
    EVERY record: the notebook falls back to the EMITTER prompt ("output
    only a CubeLang program") when one is missing, which would train the
    label/chat tasks under a contradictory instruction. Perception tasks
    (content, emotion) get the plain identity prompt, the one the host
    sends at appraisal time."""
    facts = facts or load_facts()
    seen, out = set(), []
    for r in records:
        key = r["prompt"].strip()
        if key in seen:
            continue
        seen.add(key)
        if not r.get("system"):
            r["system"] = identity_system(facts)
        r["split"] = split_of(r["prompt"])
        r["repeat"] = r.get("repeat", 1) if r["split"] == "train" else 1
        r.setdefault("vm_ok", None)
        r.setdefault("vm_result", None)
        r.setdefault("vm_error", None)
        r.setdefault("gold_match", None)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-lines", type=int, default=0, help="scan only this many lines per source (debug)")
    ap.add_argument("--n-chat", type=int, default=N_CHAT)
    ap.add_argument("--n-content", type=int, default=N_CONTENT)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--exposure", type=int, default=0,
                    help="ALSO add N explicit-prose continuation records (generation exposure). Off by default; "
                         "the owner's switch.")
    ap.add_argument("--version", default="v6", help="output name: emitter_sft_{version}.jsonl (v5 is trained; v6 = "
                                                     "v5 data + identity x3 replay + the code-leak filter + history)")
    ap.add_argument("--identity-repeat", type=int, default=IDENTITY_REPEAT)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(SEED)
    facts = load_facts()
    lim = args.limit_lines or None
    print("=== chat pairs from Orca AgentInstruct ...", flush=True)
    chat, why_chat = build_chat(rng, facts, args.n_chat, lim)
    print(f"  kept {len(chat)} | rejected {dict(why_chat.most_common(8))}")
    print("=== chat pairs from the Hugging Face sources ...", flush=True)
    hf_chat, why_hf = build_hf_chat(rng, facts)
    print(f"  kept {dict(Counter(r['subtype'] for r in hf_chat))} | rejected {dict(why_hf.most_common(10))}")
    chat += hf_chat
    why_chat.update(why_hf)
    print("=== chat pairs from the sorted local sources (arena, convo, instruct, nemotron, wikiqa, grammar, ei) ...", flush=True)
    local_chat, why_local = build_local_chat(rng, facts, limit_lines=lim)
    print(f"  kept {dict(Counter(r['subtype'] for r in local_chat))} | rejected {dict(why_local.most_common(12))}")
    chat += local_chat
    why_chat.update(why_local)
    print("=== content awareness (passage -> label) ...", flush=True)
    content, why_content = build_content(rng, args.n_content, lim)
    print(f"  kept {Counter(r['subtype'] for r in content)} | skipped {dict(why_content.most_common(8))}")
    print("=== emotion recognition (GoEmotions -> emotion + Plutchik petal) ...", flush=True)
    emotion, why_emo = build_emotion(rng)
    print(f"  kept {len(emotion)} over {len(set(r['gold'] for r in emotion))} emotions | skipped {dict(why_emo)}")
    print("=== affect (valence/arousal-rated messages) ...", flush=True)
    affect, why_aff = build_affect(rng)
    print(f"  kept {len(affect)} {dict(Counter(r['subtype'] for r in affect))} | skipped {dict(why_aff)}")
    print("=== history (dated events, expert dialogues, quotes, NYT recall + dating) ...", flush=True)
    history, why_hist = build_history(rng, facts, limit_lines=lim)
    print(f"  kept {dict(Counter(r['subtype'] for r in history))} | skipped {dict(why_hist.most_common(10))}")
    print("=== era orientation (verified_facts books -> period + subject) ...", flush=True)
    era, why_era = build_era(rng, facts, limit_lines=lim)
    print(f"  kept {len(era)} {dict(Counter(r['source'] for r in era))} | skipped {dict(why_era)}")
    history += era
    why_hist.update(why_era)
    exposure: list[dict] = []
    if args.exposure:
        print(f"=== generation exposure ON ({args.exposure}) ...", flush=True)
        exposure, why_exp = build_exposure(rng, args.exposure, lim)
        print(f"  kept {len(exposure)} | skipped {dict(why_exp)}")
    new = finish(chat + content + emotion + affect + history + exposure, facts)
    replay = [] if args.no_replay else [json.loads(l) for l in open(V4_PATH, encoding="utf-8")]
    n_id = 0
    n_game = 0
    for r in replay:                                     # the identity contract outweighs the chat volume
        if r["task"] == "identity" and r["split"] == "train":
            r["repeat"] = max(int(r.get("repeat", 1)), args.identity_repeat)
            n_id += 1
        # v7: the forge DECISION/COMPARE families regressed under v6's chat volume (forge probe decision
        # 1.00 -> 0.17, compare 1.00 -> 0.83: programs return the raw reading, one invents a `comp` opcode);
        # replay them x3 like identity
        sub = str(r.get("subtype") or "")
        if r["split"] == "train" and (sub.startswith("game:decision") or sub.startswith("game:compare")):
            r["repeat"] = max(int(r.get("repeat", 1)), GAME_FORGE_REPEAT)
            n_game += 1
    records = replay + new
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"emitter_sft_{args.version}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_rev = "unknown"
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(), "git_rev": git_rev, "version": args.version,
        "config": {"SEED": SEED, "N_CHAT": args.n_chat, "N_CONTENT": args.n_content, "limit_lines": args.limit_lines,
                   "replay": not args.no_replay, "identity_repeat": args.identity_repeat,
                   "identity_train_records_upweighted": n_id,
                   "game_forge_train_records_upweighted": n_game, "game_forge_repeat": GAME_FORGE_REPEAT,
                   "schema_note": "v5 = v4 (replay, unchanged) + chat pairs (Orca, voice/guard-filtered, identity "
                                  "system prompt with a sampled state) + content awareness (passage -> nsfw/safe; "
                                  "minors/non-consent screened out; generation-exposure deliberately NOT built)",
                   "not_used": ["identity_corpus.txt (older, different persona)", "conversations_svc_* (prompt banks)",
                                "mental_health_svc (single lines, sensitive)",
                                "knowledgetxt greetings/identity*/inquiry/tonal/philosophical/capability_* + batch_chat_templates_* (the old persona)",
                                "knowledgetxt tool_usage_training_data (instructions do not match their tool calls)",
                                "knowledgetxt intent_all (legal-topic labels), snn_training_data, wikibooks_corpus (raw chunks), timeline_conversations.csv (templated)",
                                "knowledgetxt historical_facts.jsonl (= temporal/train_augmented)",
                                "E:/datasets symbolic (wiki MCQ), sagi (logic tasks), spatial CSV (robot semantic parses — a future spatial family), verified_facts tree (books)",
                                "KonstantyM/science_qa_prep beyond its 238 science rows (an OpenOrca/FLAN instruction mix under base-model system prompts)"]},
        "inputs_sha256": {V4_PATH: sha256_file(V4_PATH) if os.path.exists(V4_PATH) else None},
        "sources": {"chat": [ORCA, HF_EVERYDAY, HF_SYSTEMCHATS, HF_OASST2, HF_FR_ALPACA],
                    "hf": {"everyday": "HuggingFaceTB/everyday-conversations-llama3.1-2k (apache-2.0)",
                           "systemchats": "HuggingFaceTB/smoltalk systemchats-30k (apache-2.0)",
                           "oasst2": "OpenAssistant/oasst2 (apache-2.0; en+fr rank-0, detoxify-screened)",
                           "french_alpaca": "jpacifico/French-Alpaca-dataset-Instruct-110K (apache-2.0)"},
                    "nsfw": [NSFW_EXPLICIT, NSFW_ALT], "safe": [SFW_WEB, SFW_NEWS]},
        "chat_rejections": dict(why_chat), "content_skips": dict(why_content),
        "n_chat": len(chat), "n_chat_by_source": dict(Counter(r["subtype"] for r in chat)),
        "n_chat_by_lang": dict(Counter(r["lang"] for r in chat)),
        "n_content": dict(Counter(r["subtype"] for r in content)),
        "n_emotion": len(emotion), "emotion_by_label": dict(Counter(r["gold"] for r in emotion)),
        "emotion_source": "google-research-datasets/go_emotions simplified (apache-2.0)",
        "n_affect": len(affect), "affect_by_source": dict(Counter(r["subtype"] for r in affect)),
        "affect_skips": dict(why_aff),
        "local_sources": {"arena": ARENA, "convo": CONVO, "instruct": INSTRUCT_55K, "nemotron": NEMOTRON_FI, "science": SCIENCE_QA_SHARD,
                          "wikiqa": WIKIQA, "grammar": GRAMMAR, "ei": EI, "emotions_plutchik": EMOTIONS_PLUTCHIK,
                          "movie_scenes": MOVIE_SCENES_DIR,
                          "affect": dict(AFFECT_FILES), "individual_events": INDIVIDUAL_EVENTS, "quotes": QUOTES,
                          "quota": LOCAL_QUOTA, "caps": LOCAL_CAP, "nemotron_scan": NEMOTRON_SCAN, "quote_quota": QUOTE_QUOTA},
        "n_history": len(history), "history_by_subtype": dict(Counter(r["subtype"] for r in history)),
        "history_sources": {"verified_facts": VERIFIED_FACTS, "era_per_era": ERA_PER_ERA,
                            "events": HIST_EVENTS, "events_1800s": HIST_1800, "10k_years": HIST_10K,
                            "dialogues": HIST_DIALOGUE, "nyt": NYT_DIR, "quota": HIST_QUOTA},
        "history_skips": dict(why_hist),
        "n_exposure": len(exposure), "exposure_switch": args.exposure,
        "n_replay": len(replay), "n_records": len(records),
        "by_task": dict(Counter(r["task"] for r in records)),
        "by_split_new": dict(Counter(r["split"] for r in new)),
        "gaps": ["the local corpus has no dialogue/French chat; both come from the HF sources above",
                 "history is EN only (every temporal source is English)"],
        "output": out_path, "output_sha256": sha256_file(out_path), "wall_s": time.perf_counter() - t0,
    }
    mp = os.path.join(OUT_DIR, f"emitter_sft_{args.version}.manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1)
    print(f"\n{args.version}: {len(chat)} chat + {len(content)} content + {len(emotion)} emotion + {len(affect)} affect + {len(history)} history"
          + (f" + {len(exposure)} exposure" if exposure else "") + f" + {len(replay)} replay "
          f"= {len(records)} (after prompt dedupe)")
    print(f"wrote {out_path}\nwrote {mp} ({manifest['wall_s']:.0f}s) sha {manifest['output_sha256'][:12]}")


if __name__ == "__main__":
    main()
