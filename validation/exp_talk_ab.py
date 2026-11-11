"""validation/exp_talk_ab.py -- order 0010's blind A/B: the voice adapter against the control, both Cubby's.

Three steps, each its own run, so the prompts exist (and are committed) before any reply does:

  prompts    60 turns, by seed ("a fact not on record" is not among them: both adapters are trained to say the
             same sentence there, the host says it itself when the VM returns nothing, and E6 measured it):
             30 chat turns (bare, as the host sends a first turn): a hello, thanks or goodbye, who are you, what can
             you do, a joke asked for; user lines written for this test, 22 EN and 8 FR;
             30 fact turns: one fact (9 EN: a benchmark question Cubby answered right in 0054's "after", with the
             walk's own fact as the block; 3 FR: held voice rows), a list (8 EN: held profile records of the E6 data
             that E6 did not draw; 2 FR: held voice rows), a wrong assumption (6 EN, 2 FR: held voice rows carrying
             the host's verdict line, "Premise: contradicted")
  generate   one arm: greedy, 160 new tokens (the talk seat's), cut by the host's `talk_answer`; a fact turn's draft
             is read by the host's `talk_reply` (CB_OPENER_VIEW=1 CB_TALK_LINES=1: changes a and b). A chat turn has
             no host check yet (change d): the cut only, and Cowork reads each for a false statement about Cubby
  sheet      pairs the two arms. A pair goes to Nick only when both replies are spoken and they differ; the side of
             each arm is drawn by seed. Refused and identical pairs are listed, not shown. The key is written apart.

    python validation/exp_talk_ab.py prompts
    python validation/exp_talk_ab.py generate --export <export> --adapter <adapter dir> --arm voice|control
    python validation/exp_talk_ab.py sheet

Wired: STANDALONE (order 0010's harness).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

TAG = "talk_ab_2026-11-11"
LOGS = os.path.join(HERE, "logs")
SEED = 20261111
OUT = os.path.join(ROOT, "standin", "data", "out")
DASH = " — "
CHAT = {
    "hello": [("en", "hi"), ("en", "hey Cubby, I'm Maya"), ("en", "good morning!"),
              ("en", "hello there, my name is Daniel"), ("fr", "salut, moi c'est Léa"), ("fr", "bonjour Cubby")],
    "thanks": [("en", "thanks, that helped a lot"), ("en", "ok thank you!"), ("en", "gotta go, talk later"),
               ("en", "bye for now"), ("fr", "merci beaucoup"), ("fr", "bon, je file, à plus tard")],
    "who": [("en", "who are you?"), ("en", "who made you?"), ("en", "are you a person or a program?"),
            ("en", "what languages do you speak?"), ("fr", "tu es qui, au juste ?"), ("fr", "qui t'a créé ?")],
    "can_do": [("en", "what can you do?"), ("en", "how can you help me?"), ("en", "what are you good at?"),
               ("en", "what happens if you don't know something?"), ("en", "can you show me how you got an answer?"),
               ("fr", "qu'est-ce que tu sais faire ?")],
    "joke": [("en", "tell me a joke"), ("en", "got a pun for me?"), ("en", "make me laugh"),
             ("en", "do you know any good jokes?"), ("en", "tell me something funny"),
             ("fr", "raconte-moi une blague")],
}
_FR = re.compile(r" \?$|[éèêàùç]|^(quel|quelle|c'est|tu |dis|présente|donc|alors)", re.I)


def _rows(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def _lines(facts) -> list[str]:
    return [f"{v} is the {r} of {e}" for e, r, v in facts] + [f"{e} {r}: {v}" for e, r, v in facts]


def _fact_prompt(facts, q: str) -> str:
    from cubbyllm.emit.talk import talk_prompt
    return talk_prompt(q, {"facts": [tuple(f) for f in facts], "question": q})


def _voice_parts(prompt: str):
    """A voice row's block: (triples, question, premise line or None)."""
    facts, q, premise = [], "", None
    for ln in prompt.split("\n"):
        if ln.startswith("Premise:"):
            premise = ln
        elif ln.startswith("- ") and DASH in ln and ": " in ln.split(DASH, 1)[1]:
            e, rest = ln[2:].split(DASH, 1)
            r, v = rest.split(": ", 1)
            facts.append((e, r, v))
        elif ln.startswith("Question:"):
            q = ln[len("Question:"):].strip()
    return facts, q, premise


def _asked(facts, q: str, whole: bool):
    """(entity, returned, others) for a voice row: the entity the question names; its values (a list turn), or the
    values of the relation the question names (a fact turn); the block's other names and values."""
    low = q.lower()
    ents = sorted({e for e, _, _ in facts if e.lower() in low}, key=len, reverse=True)
    if not ents:
        return None
    e = ents[0]
    mine = [(r, v) for x, r, v in facts if x == e]
    rels = sorted({r for r, _ in mine if r.lower() in low}, key=len, reverse=True)
    returned = [v for _, v in mine] if whole or not rels else [v for r, v in mine if r == rels[0]]
    if not whole and not rels:
        return None
    keep = {v.lower() for v in returned} | {e.lower()}
    others = [] if whole else list(dict.fromkeys(x for f in facts for x in (f[0], f[2]) if x.lower() not in keep))
    return e, returned, others


def _bench_facts(rng) -> list[dict]:
    """One-hop benchmark questions answered right in 0054's "after": the question as asked, the walk's own fact."""
    pool = []
    for b in ("webq", "simpleqa"):                         # NQ's final records moved; these two are whole
        for r in _rows(os.path.join(OUT, "system_eval", "run_0055_after", f"c_{b}.jsonl")):
            if r.get("spoken") and r.get("verdict") == "right" and len(r.get("trace") or []) == 1 and len(r.get("plan") or []) == 1:
                rel = r["plan"][0]
                v, sep, e = r["trace"][0].partition(f" is the {rel} of ")
                if sep and v and e:
                    pool.append({"id": r["id"], "q": r["question"], "facts": [(e, rel, v)], "entity": e})
    return rng.sample(sorted(pool, key=lambda x: x["id"]), 9)


def build_prompts() -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for kind, lines in CHAT.items():
        for lang, text in lines:
            out.append({"kind": kind, "group": "chat", "lang": lang, "source": "written for the test", "user": text,
                        "facts": [], "prompt": f"Question: {text}\nAnswer:", "check": None})
    for b in _bench_facts(rng):
        out.append({"kind": "one_fact", "group": "fact", "lang": "en", "source": "0054 after, " + b["id"], "user": b["q"],
                    "facts": b["facts"], "prompt": _fact_prompt(b["facts"], b["q"]),
                    "check": {"returned": [b["facts"][0][2]], "lines": _lines(b["facts"]), "entity": b["entity"], "others": []}})
    ground = _rows(os.path.join(OUT, "ground_sft.jsonl"))
    e6 = {r["id"] for r in json.load(open(os.path.join(LOGS, "exp_e6_talk_gate_v3a.json"), encoding="utf-8"))["adapter"]["rows"]}
    held = [r for r in ground if r["split"] == "held" and r["id"] not in e6]
    for r in rng.sample(sorted((x for x in held if x["family"] == "profile" and len(x["facts"]) <= 8), key=lambda x: x["id"]), 8):
        out.append({"kind": "list", "group": "fact", "lang": "en", "source": "E6 data, held, " + str(r["id"]), "user": r["question"],
                    "facts": r["facts"], "prompt": _fact_prompt(r["facts"], r["question"]),
                    "check": {"returned": [v for _, _, v in r["facts"]], "lines": _lines(r["facts"]), "entity": r["entity"], "others": []}})
    voice = sorted((r for r in _rows(os.path.join(OUT, "talk_voice.jsonl")) if r["split"] == "held"), key=lambda x: x["id"])
    rng.shuffle(voice)

    def take(family, lang, n, kind, whole, premise=None):
        got = 0
        for r in voice:
            facts, q, prem = _voice_parts(r["prompt"])
            if got == n or r["family"] != family or bool(_FR.search(q)) != (lang == "fr") or (premise and prem != premise):
                continue
            a = _asked(facts, q, whole)
            if not a or not a[1] or len(facts) > 8:
                continue
            out.append({"kind": kind, "group": "fact", "lang": lang, "source": "voice set, held, " + r["id"], "user": q,
                        "facts": facts, "premise": prem, "prompt": r["prompt"],
                        "check": {"returned": a[1], "lines": _lines(facts), "entity": a[0], "others": a[2]}})
            got += 1
        assert got == n, (family, lang, got)

    take("voice_fact", "fr", 3, "one_fact", False)
    take("voice_list", "fr", 2, "list", True)
    take("voice_correct", "en", 6, "wrong_assumption", False, "Premise: contradicted")
    take("voice_correct", "fr", 2, "wrong_assumption", False, "Premise: contradicted")
    for i, p in enumerate(out):
        p["id"] = f"ab{i:02d}"
    return out


def generate(a) -> None:
    import grilly  # grilly2
    from tokenizers import Tokenizer

    from ask import talk_reply
    from cubbyllm.emit.talk import talk_answer
    from exp_e5_base450m_probes import load_grilly
    from exp_e6_talk_gate import load_adapter
    assert os.environ.get("CB_OPENER_VIEW") == "1" and os.environ.get("CB_TALK_LINES") == "1", "run with changes a and b on"
    prompts = _rows(os.path.join(LOGS, TAG + "_prompts.jsonl"))
    tk = Tokenizer.from_file(os.path.join(a.export, "tokenizer.json"))
    eos = tk.token_to_id("</s>")
    model, _ = load_grilly(a.export)
    meta = load_adapter(model, a.adapter)
    rows, t0 = [], time.time()
    for p in prompts:
        ids = tk.encode(p["prompt"]).ids
        with grilly.no_grad():
            new = model.generate(grilly.tensor([ids]), max_new_tokens=a.max_new, eos_token_id=eos).tolist()[0][len(ids):]
        limit = eos not in new and len(new) >= a.max_new
        raw = tk.decode(new[:new.index(eos)] if eos in new else new)
        draft = talk_answer(raw)
        c = p["check"]
        if c is None:
            reply, why = (draft or None), "chat: the cut only"
        else:
            reply, why = talk_reply(c["returned"], draft, c["lines"], c["entity"], c["others"])
        rows.append({"id": p["id"], "raw": raw, "draft": draft, "reply": reply, "why": why, "hit_limit": limit})
        print(f"  {p['id']} {p['kind']:16s} {'spoken ' if reply else 'REFUSED'} {why[:40]:40s} | {draft[:70]!r}", flush=True)
    json.dump({"arm": a.arm, "adapter": os.path.basename(os.path.normpath(a.adapter)), "adapter_meta": meta,
               "max_new": a.max_new, "seconds": round(time.time() - t0, 1), "rows": rows},
              open(os.path.join(LOGS, f"{TAG}_{a.arm}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def sheet() -> None:
    prompts = _rows(os.path.join(LOGS, TAG + "_prompts.jsonl"))
    arms = {n: {r["id"]: r for r in json.load(open(os.path.join(LOGS, f"{TAG}_{n}.json"), encoding="utf-8"))["rows"]}
            for n in ("voice", "control")}
    rng = random.Random(SEED + 1)
    pairs, key, auto = [], {}, {"identical": [], "refused": []}
    for p in prompts:
        v, c = arms["voice"][p["id"]], arms["control"][p["id"]]
        if not (v["reply"] and c["reply"]):
            auto["refused"].append({"id": p["id"], "kind": p["kind"], "voice": v["why"] if not v["reply"] else "spoken",
                                    "control": c["why"] if not c["reply"] else "spoken"})
        elif v["reply"].strip() == c["reply"].strip():
            auto["identical"].append({"id": p["id"], "kind": p["kind"], "reply": v["reply"]})
        else:
            side = rng.choice("ab")
            key[p["id"]] = side
            pairs.append({"id": p["id"], "kind": p["kind"], "lang": p["lang"], "user": p["user"],
                          "facts": [f"{e}{DASH}{r}: {x}" for e, r, x in p["facts"]], "premise": p.get("premise"),
                          "a": (v if side == "a" else c)["reply"], "b": (c if side == "a" else v)["reply"]})
    rng.shuffle(pairs)
    blob = json.dumps(key, sort_keys=True).encode()
    json.dump({"voice_is": key, "sha256": hashlib.sha256(blob).hexdigest()},
              open(os.path.join(LOGS, TAG + "_key.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"key_sha256": hashlib.sha256(blob).hexdigest(), "of": len(prompts), "pairs": pairs, "auto": auto},
              open(os.path.join(LOGS, TAG + "_sheet.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"to judge {len(pairs)} | identical {len(auto['identical'])} | refused {len(auto['refused'])}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prompts", "generate", "sheet"))
    ap.add_argument("--export")
    ap.add_argument("--adapter")
    ap.add_argument("--arm", choices=("voice", "control"))
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args(argv)
    if a.step == "prompts":
        ps = build_prompts()
        with open(os.path.join(LOGS, TAG + "_prompts.jsonl"), "w", encoding="utf-8") as f:
            for p in ps:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        n = {}
        for p in ps:
            n[(p["group"], p["kind"], p["lang"])] = n.get((p["group"], p["kind"], p["lang"]), 0) + 1
        print(len(ps), "prompts:", {" ".join(k): v for k, v in sorted(n.items())})
    elif a.step == "generate":
        generate(a)
    else:
        sheet()


if __name__ == "__main__":
    main()
