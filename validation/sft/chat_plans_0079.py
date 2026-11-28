"""chat_plans_0079 -- the plans of 0079's final run: whose user, how it opens, how long, and which turn turns the
pronouns (the perspective addendum's item 2), plus the agents' batches.

- **The users are personas** (Nick, 2026-11-28): English from Nemotron-Personas-USA (the copy in the training data:
  text chunks, a persona opening on its ``uuid:`` line and running until the next), French from one file of
  Nemotron-Personas-France (parquet); both CC BY 4.0. A persona is only ever a user's, never Cubby's.
- **The test's personas are never drawn:** a persona whose uuid's SHA-256 ends in ``0`` (one in sixteen) is the
  perspective set's (`is_test`).
- **A sample by seed:** the personas with the smallest SHA-256 of ``0079:<seed>:<uuid>``, so a sample does not depend
  on the file's order (bottom-k).
- **Openers:** a real person's line from Topical-Chat's train pool (English), a feeling about a situation, the
  persona's own life (its job or a hobby), or a thing Cubby's store holds (the user asks about it; the host's
  resolver brings the facts at that turn).
- **Item 2:** each conversation has one turn of the three kinds, in turn (about Cubby, about themselves, a thanks or a
  praise), never the first: about one turn in five to eight, so at least 10% of the chat turns, in thirds.

Wired: STANDALONE (`chat_step_0079.py plans`).
"""
from __future__ import annotations

import ast
import hashlib
import heapq
import json
import random

__wiring__ = "STANDALONE"

KINDS = ("about_cubby", "about_self", "thanks_praise")
KIND_BRIEF = {
    "about_cubby": "the user says something about Cubby itself (its words, how it sounds or seems today), e.g. "
                   "'congrats on your first words', 'you seem upbeat today'",
    "about_self": "the user says something about their own life (their day, family, work), e.g. 'I just got off "
                  "work', 'my sister's wedding was lovely'",
    "thanks_praise": "the user thanks or praises Cubby, and says nothing of how they are, e.g. 'thanks', 'great "
                     "answer'"}
FEELINGS = ("sadness", "disappointment", "nervousness", "fear", "anger", "annoyance", "excitement", "joy",
            "gratitude", "grief", "embarrassment", "pride", "relief", "confusion", "curiosity", "amusement")
SITUATIONS = ("a long day at work", "an exam coming up", "a new job", "a friend who cancelled plans", "a trip",
              "a hobby project", "moving to a new apartment", "a pet", "a sports game last night", "a movie just seen",
              "cooking dinner", "a family visit", "learning a language", "a rainy weekend", "a concert", "a book")
PLACE_FIELDS = ("city", "state", "zipcode", "county", "commune", "departement", "region")


def is_test(uuid: str) -> bool:
    """The perspective test's personas: the uuid's SHA-256 ends in 0. Never drawn for training."""
    return hashlib.sha256(uuid.strip().encode("utf-8")).hexdigest().endswith("0")


def _rank(seed: int, uuid: str) -> int:
    return int(hashlib.sha256(f"0079:{seed}:{uuid}".encode("utf-8")).hexdigest()[:16], 16)


def _fields(text: str) -> dict:
    out: dict = {}
    for line in text.split("\n"):
        key, sep, val = line.partition(": ")
        if sep and key.replace("_", "").isalpha() and key == key.lower():
            out[key] = val.strip()
        elif out and line.strip():
            out[list(out)[-1]] += " " + line.strip()
    return out


def _bottom(items, want: int, seed: int) -> list[dict]:
    heap: list = []
    for uuid, get in items:
        if is_test(uuid):
            continue
        r = -_rank(seed, uuid)
        if len(heap) < want:
            heapq.heappush(heap, (r, uuid, get()))
        elif r > heap[0][0]:
            heapq.heapreplace(heap, (r, uuid, get()))
    return [p for _, _, p in sorted(heap, reverse=True)]


def personas_en(path: str, want: int, seed: int) -> list[dict]:
    """`want` English personas, test ones never drawn, from the chunked jsonl (one pass)."""
    def items():
        uuid, parts = None, []
        for line in open(path, encoding="utf-8"):
            text = json.loads(line)["text"]
            if text.startswith("uuid: "):
                if uuid:
                    yield uuid, (lambda t="".join(parts), u=uuid: {**_fields(t), "uuid": u, "lang": "en"})
                uuid, parts = text.split("\n", 1)[0][len("uuid: "):].strip(), [text]
            elif uuid:
                parts.append(text)
        if uuid:
            yield uuid, (lambda t="".join(parts), u=uuid: {**_fields(t), "uuid": u, "lang": "en"})
    return _bottom(items(), want, seed)


def personas_fr(path: str, want: int, seed: int) -> list[dict]:
    """`want` French personas, test ones never drawn, from the parquet file."""
    import pyarrow.parquet as pq
    cols = [c for c in pq.ParquetFile(path).schema_arrow.names if c in (
        "uuid", "persona", "professional_persona", "occupation", "age", "hobbies_and_interests_list") + PLACE_FIELDS]
    rows = pq.read_table(path, columns=cols).to_pylist()
    return _bottom(((r["uuid"], (lambda r=r: {**{k: v for k, v in r.items() if v is not None}, "lang": "fr"}))
                    for r in rows), want, seed)


def seed_of(p: dict, rng: random.Random) -> dict:
    """What the user's side is seeded with: the job (the France set's category and its work line; the English copy
    keeps only the text fields, so its work line), one hobby, the persona's own line; and the places Cubby must not
    name (the France set's commune and departement; the English copy has none, so the brief keeps towns out)."""
    hobbies = p.get("hobbies_and_interests_list") or "[]"
    try:
        hobbies = ast.literal_eval(hobbies) if isinstance(hobbies, str) else list(hobbies)
    except (ValueError, SyntaxError):
        hobbies = []
    return {"uuid": p["uuid"], "occupation": str(p.get("occupation") or "").replace("_", " "),
            "work": str(p.get("professional_persona") or "")[:300],
            "age": p.get("age"), "hobby": rng.choice(hobbies) if hobbies else None,
            "line": str(p.get("persona") or p.get("professional_persona") or "")[:400],
            "places": [str(p[k]) for k in PLACE_FIELDS if p.get(k)]}


def build(n: int, seed: int, en: list[dict], fr: list[dict], openers: list[str], names: list[str],
          fr_share: float = 0.2) -> list[dict]:
    """`n` plans; ids c0079f:00000 on. The personas are used in order, each once."""
    rng = random.Random(seed)
    en, fr, opens = list(en), list(fr), list(openers)
    rng.shuffle(opens)
    out = []
    for i in range(n):
        lang = "fr" if fr and rng.random() < fr_share else "en"
        persona = (fr if lang == "fr" else en).pop(0)
        length = rng.randint(4, 8)
        u = rng.random()
        plan = {"id": f"c0079f:{i:05d}", "lang": lang, "length": length, "persona": seed_of(persona, rng),
                "kind_turn": rng.randint(1, length - 1), "kind": KINDS[i % 3]}
        if lang == "en" and u < 0.35 and opens:
            plan.update(source="topical-chat", opener=opens.pop())
        elif u < 0.60:
            plan.update(source="feeling", feeling=rng.choice(FEELINGS), situation=rng.choice(SITUATIONS))
        elif u < 0.85 or not names:
            plan.update(source="persona life")
        else:
            plan.update(source="asks about a fact", ask=rng.choice(names))
        out.append(plan)
    return out


def batches(plans: list[dict], size: int) -> list[dict]:
    """The agents' batches: `size` plan ids each, Haiku and Sonnet in turn (Nick: mixed)."""
    return [{"batch": f"b{k:03d}", "writer": ("claude-haiku", "claude-sonnet")[k % 2],
             "ids": [p["id"] for p in plans[i:i + size]]} for k, i in enumerate(range(0, len(plans), size))]
