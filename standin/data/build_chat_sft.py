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
import hashlib
import json
import os
import random
import subprocess
import sys
import time
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


def chat_ok(user: str, assistant: str, facts: dict) -> str | None:
    """Why a pair is rejected, or None when it can be Cubby's answer."""
    if not (1 <= len(user.split()) <= MAX_USER_WORDS):
        return "user length"
    if not (MIN_ASSISTANT_WORDS <= len(assistant.split()) <= MAX_ASSISTANT_WORDS):
        return "assistant length"
    if URL.search(user) or URL.search(assistant):
        return "url"
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


def finish(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = r["prompt"].strip()
        if key in seen:
            continue
        seen.add(key)
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
    print("=== content awareness (passage -> label) ...", flush=True)
    content, why_content = build_content(rng, args.n_content, lim)
    print(f"  kept {Counter(r['subtype'] for r in content)} | skipped {dict(why_content.most_common(8))}")
    print("=== emotion recognition (GoEmotions -> emotion + Plutchik petal) ...", flush=True)
    emotion, why_emo = build_emotion(rng)
    print(f"  kept {len(emotion)} over {len(set(r['gold'] for r in emotion))} emotions | skipped {dict(why_emo)}")
    exposure: list[dict] = []
    if args.exposure:
        print(f"=== generation exposure ON ({args.exposure}) ...", flush=True)
        exposure, why_exp = build_exposure(rng, args.exposure, lim)
        print(f"  kept {len(exposure)} | skipped {dict(why_exp)}")
    new = finish(chat + content + emotion + exposure)
    replay = [] if args.no_replay else [json.loads(l) for l in open(V4_PATH, encoding="utf-8")]
    records = replay + new
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "emitter_sft_v5.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_rev = "unknown"
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(), "git_rev": git_rev, "version": "v5",
        "config": {"SEED": SEED, "N_CHAT": args.n_chat, "N_CONTENT": args.n_content, "limit_lines": args.limit_lines,
                   "replay": not args.no_replay,
                   "schema_note": "v5 = v4 (replay, unchanged) + chat pairs (Orca, voice/guard-filtered, identity "
                                  "system prompt with a sampled state) + content awareness (passage -> nsfw/safe; "
                                  "minors/non-consent screened out; generation-exposure deliberately NOT built)",
                   "not_used": ["identity_corpus.txt (older, different persona)", "conversations_svc_* (prompt banks)",
                                "mental_health_svc (single lines, sensitive)"]},
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
        "n_exposure": len(exposure), "exposure_switch": args.exposure,
        "n_replay": len(replay), "n_records": len(records),
        "by_task": dict(Counter(r["task"] for r in records)),
        "by_split_new": dict(Counter(r["split"] for r in new)),
        "gaps": ["the local corpus has no dialogue/French chat; both come from the HF sources above"],
        "output": out_path, "output_sha256": sha256_file(out_path), "wall_s": time.perf_counter() - t0,
    }
    mp = os.path.join(OUT_DIR, "emitter_sft_v5.manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=1)
    print(f"\nv5: {len(chat)} chat + {len(content)} content + {len(emotion)} emotion"
          + (f" + {len(exposure)} exposure" if exposure else "") + f" + {len(replay)} replay "
          f"= {len(records)} (after prompt dedupe)")
    print(f"wrote {out_path}\nwrote {mp} ({manifest['wall_s']:.0f}s) sha {manifest['output_sha256'][:12]}")


if __name__ == "__main__":
    main()
