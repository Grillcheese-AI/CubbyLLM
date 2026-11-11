"""talk_voice_rows -- the voice rows being built, and the checks every row passes before it is kept.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

- A fact row must pass the host's name-and-number guard (`ask_words.grounded_prose`) read through the opener view
  (`build_chat_sft.chat_view`: a common word that opens a sentence is not a name), and the value check when the
  row knows what the VM returned. The raw guard is run too and counted: today's host reads the raw reply.
- No slot is left unfilled, and no row is written twice.
- Every row records where its wording came from (`wording`): a template with its model's id, a rule, Cubby's
  record, the fact row it was built on.
"""
from __future__ import annotations

import collections

__wiring__ = "STANDALONE"

# how many train rows each family aims for (held rows come on top)
TARGET = {"voice_identity": 1500, "voice_identity_facts": 800, "voice_hello": 3000, "voice_social": 7000,
          "voice_fact:en": 11000, "voice_fact:fr": 3000, "voice_list:en": 4000, "voice_list:fr": 2500,
          "voice_correct:en": 2000, "voice_correct:fr": 1200, "voice_unconfirmed:en": 1500,
          "voice_unconfirmed:fr": 1000}
JOKE_RATE = 0.08        # a pun or a past-life line after a hello, a cheer or a goodbye (Nick sets it; "low")
CLOSER_RATE = 0.2       # a closing invitation after a framed fact


class Pool:
    """Templates of one kind and language, dealt in turn so each is used about as often as the others."""

    def __init__(self, rows, rng) -> None:
        self.rows, self.rng, self.i = list(rows), rng, 0
        rng.shuffle(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def take(self) -> dict:
        if self.i >= len(self.rows):
            self.rng.shuffle(self.rows)
            self.i = 0
        self.i += 1
        return self.rows[self.i - 1]


def src(template: dict) -> str:
    return f"template:{template['model']}"


class Voice:
    """`checks` carries the host's own functions: grounded_prose, value_check, has_value, chat_view."""

    def __init__(self, rng, picked: list[dict], ident: dict, checks, common: set[str], scale: float = 1.0) -> None:
        self.rng, self.ident, self.checks, self.common, self.scale = rng, ident, checks, common, scale
        pools: dict = {}
        for r in picked:
            pools.setdefault((r["kind"], r["lang"]), []).append(r)
        self.pools = {k: Pool(v, rng) for k, v in pools.items()}
        self.strs = {k: v for k, v in ident.items() if isinstance(v, str)}
        self.record_facts = [str(v) for v in self.strs.values()] \
            + [x[k] for key in ("abilities", "abilities_fr") for x in ident[key] for k in ("label", "text")] \
            + ident["languages"] + ident["languages_next"] + [ident["character"]["running_joke"]]
        self.rows: list[dict] = []
        self.stats, self.guard, self.bad_tokens = collections.Counter(), collections.Counter(), collections.Counter()
        self._n, self._seen = collections.Counter(), set()

    def pool(self, kind: str, lang: str) -> Pool:
        return self.pools.get((kind, lang)) or Pool([], self.rng)

    def count(self, family: str, lang: str, split: str = "train") -> int:
        return self.stats[f"{family}:{lang}:{split}"]

    def want(self, key: str, split: str) -> bool:
        """Does `family:lang` still need a train row (a held row is always welcome, up to the caller's cap)."""
        family, lang = key.split(":")
        return split == "held" or self.count(family, lang) < TARGET[key] * self.scale

    def abilities(self, lang: str, items=None, labels: bool = True) -> str:
        items = items or self.ident["abilities" if lang == "en" else "abilities_fr"]
        colon = ":" if lang == "en" else " :"
        return "\n".join(f"- **{x['label']}{colon}** {x['text']}" if labels else f"- {x['text']}" for x in items)

    def add(self, family: str, lang: str, split: str, prompt: str, answer: str, wording: list[str],
            facts: list[str] | None = None, entity: str = "", returned: list[str] | None = None,
            others: list[str] | None = None) -> bool:
        """One row after its checks. `facts` given = a fact row: the guard, and the value check when `returned` is."""
        key = f"{family}:{lang}"
        if "{" in answer or "}" in answer:
            self.stats[f"dropped:{key}:unfilled slot"] += 1
            return False
        if facts is not None:
            raw_ok, _ = self.checks.grounded_prose(answer, facts, entity)
            ok, bad = self.checks.grounded_prose(self.checks.chat_view(answer, self.common), facts, entity)
            self.guard[f"{key}:rows"] += 1
            self.guard[f"{key}:raw guard passes"] += int(raw_ok)
            if not ok:
                self.stats[f"dropped:{key}:guard (opener view)"] += 1
                for b in bad[:3]:
                    self.bad_tokens[b] += 1
                return False
            if returned is not None and not self.checks.value_check(answer, returned, others or [], entity)[0]:
                self.stats[f"dropped:{key}:value check"] += 1
                return False
        if (prompt, answer) in self._seen:
            self.stats[f"dropped:{key}:the same row twice"] += 1
            return False
        self._seen.add((prompt, answer))
        self._n[family] += 1
        self.rows.append({"id": f"voice:{family}:{self._n[family]:06d}", "family": family, "split": split,
                          "lang": lang, "prompt": prompt, "answer": answer, "wording": wording})
        self.stats[f"{key}:{split}"] += 1
        return True
