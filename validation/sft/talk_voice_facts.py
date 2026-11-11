"""talk_voice_facts -- the voice rows over a facts block: a framed fact, a list, a wrong assumption answered.

Wired: STANDALONE (used by `build_talk_voice.py`; never imported by cubbyllm/).

Every row is built on a fact row of the current mix (its block, its question, its checked answer), keeps that
row's split (so a held entity stays held), and passes the guard through the opener view (`Voice.add`).

  voice_fact         English: the row's checked sentence behind a frame. French: the mix holds no French
                     sentence and none is invented, so the one fact is shown as its label and value behind a
                     French frame, over a block with French labels and a French question written by rule.
  voice_list         all the block holds about one thing: a lead-in, a numbered list, a closing line
  voice_correct      the host's check says the question's premise is contradicted (`Premise: contradicted`):
                     Cubby says so kindly and gives the checked value. The false value is never repeated.
  voice_unconfirmed  the records are silent on the premise (`Premise: not on record`): Cubby says it cannot
                     confirm it, and gives what it has.
The verdict is the host's (order 0017), never the adapter's: the adapter words the verdict it is handed.
"""
from __future__ import annotations

import collections
import json

from validation.sft.talk_voice_inputs import EN_FALSE_Q, EN_NOUN, FR_FALSE_Q, FR_LABEL, FR_LIST_Q, FR_NOUN, FR_REL_Q
from validation.sft.talk_voice_rows import CLOSER_RATE, Voice, src
from validation.sft.talk_voice_text import (ABSENT, article, cap, de, fact_lines, fill_frame, fill_list, frame_one,
                                            grouped, parse, render, render_items, to_french)

__wiring__ = "STANDALONE"

FRAME_SRC = {"relation", "bind", "counter", "fact_relation", "fact_bind", "fact_counter", "event_relation",
             "event_bind", "hist_when", "hist_where", "hist_who", "hist_fact"}
LIST_SRC = {"profile", "event_profile"}
REL_SRC = {"relation", "counter", "fact_relation", "fact_counter", "hist_fact", "event_relation"}
HELD_CAP = {"voice_fact:en": 500, "voice_fact:fr": 200, "voice_list:en": 200, "voice_list:fr": 200}


def load(base: str, rng):
    """The current mix's fact rows -> (framed-fact sources, list sources, one-relation sources, values by
    (split, label) for a false premise). A repeated row (`#2`) and a refusal are left out."""
    by_fam: dict[str, list] = collections.defaultdict(list)
    for line in open(base, encoding="utf-8"):
        r = json.loads(line)
        if r["family"] in FRAME_SRC | LIST_SRC and "#" not in r["id"] and r["answer"] != ABSENT:
            p = parse(r["prompt"])
            if p and "\n" not in r["answer"] and len(r["answer"]) <= 220:
                by_fam[r["family"]].append((r, p[0], p[1]))
    values = collections.defaultdict(list)
    for fam in by_fam:
        for r, triples, _ in by_fam[fam]:
            for _, rel, val in triples:
                values[(r["split"], rel)].append(val)
    out = [[x for f in sorted(fams) for x in by_fam[f]] for fams in (FRAME_SRC, LIST_SRC, REL_SRC)]
    for s in out:
        rng.shuffle(s)
    return out[0], out[1], out[2], values


def _full(v: Voice, key: str, split: str) -> bool:
    family, lang = key.split(":")
    return not v.want(key, split) or (split == "held" and v.count(family, lang, "held") >= HELD_CAP.get(key, 120))


def asked(v: Voice, triples, q: str, answer: str):
    """(entity, relation, values) a row is about: the entity its question names, and the one relation of it
    whose values its answer states. None when that is not clear."""
    ents = sorted({e for e, _, _ in triples}, key=len, reverse=True)
    e = next((e for e in ents if e.lower() in q.lower()), None)
    if e is None:
        return None
    hit = [(rel, vs) for rel, vs in grouped(triples, e) if any(v.checks.has_value(answer, x) for x in vs)]
    if len(hit) != 1:
        return None
    rel, vs = hit[0]
    return e, rel, [x for x in vs if v.checks.has_value(answer, x)]


def fact_rows_en(v: Voice, sources) -> None:
    for r, triples, _ in sources:
        if _full(v, "voice_fact:en", r["split"]):
            continue
        if not v.pool("frame", "en"):
            return
        t = v.pool("frame", "en").take()
        ans, wording = fill_frame(t["text"], r["answer"]), [src(t), f"fact: {r['family']} row"]
        if v.pool("invite", "en") and v.rng.random() < CLOSER_RATE:
            c = v.pool("invite", "en").take()
            ans, wording = ans + " " + c["text"], wording + [src(c)]
        v.add("voice_fact", "en", r["split"], r["prompt"], ans, wording, facts=fact_lines(triples))


def fact_rows_fr(v: Voice, sources) -> None:
    for r, triples, q in sources:
        if _full(v, "voice_fact:fr", r["split"]):
            continue
        got, fr = asked(v, triples, q, r["answer"]), to_french(triples)
        if not got or not fr or not v.pool("frame", "fr"):
            continue
        e, rel, vs = got
        label, gender = FR_LABEL[rel]
        if gender is None:
            continue
        art = article(label, gender)
        fq = v.rng.choice(FR_REL_Q).format(Quel="Quel" if gender == "m" else "Quelle", art=art, Art=cap(art),
                                           r=label, de_e=de(e))
        t = v.pool("frame", "fr").take()
        ans = frame_one(t["text"], label, vs, "fr")
        if ans is None:
            continue
        others = [x for ee, _, vv in fr for x in (ee, vv) if x != e and x not in vs]
        v.add("voice_fact", "fr", r["split"], render(fr, fq), ans, [src(t), "rule", f"fact: {r['family']} row"],
              facts=fact_lines(fr), entity=e, returned=vs, others=others)


def list_rows(v: Voice, sources) -> None:
    for lang in ("en", "fr"):
        for r, triples, q in sources:
            if _full(v, f"voice_list:{lang}", r["split"]):
                continue
            ents = {e for e, _, _ in triples}
            block = triples if lang == "en" else to_french(triples)
            if len(ents) != 1 or not block:
                continue
            e = next(iter(ents))
            question = q if lang == "en" else v.rng.choice(FR_LIST_Q).format(e=e, de_e=de(e))
            items = grouped(block, e)
            kind = "frame" if len(items) == 1 else "answer_list"     # one fact is not a list
            if not v.pool(kind, lang):
                continue
            t = v.pool(kind, lang).take()
            if len(items) == 1:
                ans = frame_one(t["text"], items[0][0], items[0][1], lang)
                if ans is None:
                    continue
            else:
                text, is_list = render_items(items, lang)
                ans = fill_list(t["text"], "facts", text, is_list, lang)
            v.add("voice_list", lang, r["split"], render(block, question), ans,
                  [src(t), "rule", f"fact: {r['family']} row"], facts=fact_lines(block), entity=e,
                  returned=[x for _, vs in items for x in vs], others=[])


def premise_rows(v: Voice, sources, values) -> None:
    for lang in ("en", "fr"):
        usable = EN_NOUN if lang == "en" else FR_NOUN
        for verdict, fam, kind in (("contradicted", "voice_correct", "correct"), ("absent", "voice_unconfirmed", "unconfirmed")):
            for r, triples, _ in sources:
                if _full(v, f"{fam}:{lang}", r["split"]):
                    continue
                if not v.pool(kind, lang):
                    break
                e = collections.Counter(x for x, _, _ in triples).most_common(1)[0][0]
                block = triples if lang == "en" else to_french(triples)
                if not block:
                    continue
                mine = grouped(triples, e)
                if verdict == "contradicted":            # the block holds the relation; the question claims another value
                    opts = [(rel, vs) for rel, vs in mine if rel in usable and len(vs) <= 4]
                    if not opts:
                        continue
                    rel, vs = v.rng.choice(opts)
                    shown = [(rel, vs)]
                else:                                    # the block holds nothing for the relation asked about
                    have = {rel for rel, _ in mine}
                    opts = sorted(rel for rel in usable if rel not in have and values.get((r["split"], rel)))
                    if not opts or len(mine) < 2:
                        continue
                    rel, vs = v.rng.choice(opts), []
                    block = [x for x in block if x[0] == e]
                    shown = mine[:4]
                returned = [x for _, v2 in shown for x in v2]
                cands = [x for x in values[(r["split"], rel)] if x not in vs and x != e]
                if not cands:
                    continue
                false_value = v.rng.choice(cands)
                if lang == "en":
                    question, items = v.rng.choice(EN_FALSE_Q).format(r=rel, e=e, x=false_value), shown
                else:
                    label, gender = FR_LABEL[rel]
                    art = article(label, gender)
                    question = v.rng.choice(FR_FALSE_Q).format(art=art, Art=cap(art), r=label, de_e=de(e), x=false_value)
                    items = [(FR_LABEL[rl][0], v2) for rl, v2 in shown]
                t = v.pool(kind, lang).take()
                text, is_list = render_items(items, lang)
                ans = fill_list(t["text"].replace("{subject}", e), "facts", text, is_list, lang)
                others = [y for ee, _, vv in block for y in (ee, vv) if y != e and y not in returned]
                v.add(fam, lang, r["split"], render(block, question, verdict), ans,
                      [src(t), "rule", f"fact: {r['family']} row"], facts=fact_lines(block), entity=e,
                      returned=returned, others=others)
