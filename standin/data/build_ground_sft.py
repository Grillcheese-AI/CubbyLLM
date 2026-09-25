"""build_ground_sft -- the talk adapter's first family on GRL's own base: say what the VM returned.

Wired: STANDALONE (a data builder; never imported by cubbyllm/).

Why this family first. exp_e5 (2026-09-24, CUBBYLLM_HYPOTHESES.md H-E5) measured the 450M base on
facts it was handed: it copies a value from the context 157/160 times but binds it to the person
asked about only 83/160 (chance), and when the context contradicts what it memorised it answers
from memory 22/24 times. A CubbyLLM answer is what the VM returned, so the talk adapter's first job
is exactly those two things -- the right entity's value, and the facts over the weights -- plus
saying so when the facts hold no answer.

Five families over a facts block (lines `entity — relation: value`, the display form `ask.say` gives):
  profile   one entity, 3-7 facts, "Who is X?" -> one or two sentences from the facts
  relation  one entity, 3-7 facts, a question about one relation -> the sentence that states it
  bind      2-3 entities sharing a relation with different values, plus other facts; a question
            about one of them -> its value, and NONE of the others'
  counter   one entity whose fact for the asked relation carries ANOTHER entity's value (the real
            one removed) -> the value the facts give, never the real one
  absent    a finished relation/bind question with the asked fact taken out -> "The facts don't say."

Facts come from the wikikg graph (standin/data/wikikg.py), subject-centric, relations on a
whitelist (no timeline events, variants or 'related entity'). A frontier model over OpenRouter
writes the questions and answers -- dataset building only, never serving (Nick, 2026-09-12) --
and nothing it writes is kept unless the host's checks pass: `ask.grounded_prose` (every name and
number in the answer occurs in the facts), the gold value present, every forbidden value absent,
the question naming its entity and not containing its answer. The absent family is built from
kept records by the host, with a fixed line, and no model.

Split by the asked entity's hash (held = 10%); a bind block never mixes splits. Cached by (model,
prompt) under standin/data/out/openrouter_cache/; the key is in the gitignored validation/.env.

  python standin/data/build_ground_sft.py --items 400 --tag _pilot          # a pilot: cost + keep rate
  python standin/data/build_ground_sft.py --items 26000                     # the set
"""
from __future__ import annotations

import argparse, collections, concurrent.futures as cf, hashlib, json, pathlib, random, re, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

ABSENT = "The facts don't say."
DASH = " — "

# subject-centric relations worth asking about (wikikg's nouns, forward or inverse readings)
ASKABLE = frozenset("""birth date|death date|birthplace|place of death|burial place|spouse|parent|child|sibling|
relative|occupation|field of work|alma mater|employer|employee|author|work|creator|creation|director|film|
founder|founding|capital|location|genre|publisher|nationality|award|recipient|composer|composition|team|
leader|headquarters|predecessor|successor|member|group|performer|instance|type|inventor|invention|
discoverer|discovery|language|religion|position|holder|residence|teacher|student|mentor|producer|
designer|developer|writer|cast member|actor|role|owner|mouth|coach|illustrator|translator|industry|
namesake|subsidiary|parent company|album|track|episode|series|origin|basis|adaptation|inspiration|
influence|cause|effect|use|genre|label|host|period""".replace("\n", "").split("|"))

# relations one question can reach through either word: an absent record drops the whole group
NEAR = [frozenset(g.split("|")) for g in ("teacher|mentor", "student|mentor", "work|creation|film|album|composition",
                                          "creator|author|writer", "type|instance", "group|member",
                                          "award|recipient", "leader|position|holder", "parent|parent company",
                                          "child|subsidiary", "influence|inspiration|basis|origin")]

PROFILE_Q = {"who": ["Who is {X}?", "Who was {X}?", "Tell me about {X}.", "What do you know about {X}?"],
             "what": ["What is {X}?", "Tell me about {X}.", "What do you know about {X}?", "Describe {X}."],
             "where": ["Where is {X}?", "Tell me about {X}.", "What do you know about {X}?"]}
PERSON = frozenset(["birth date", "death date", "birthplace", "occupation", "spouse", "alma mater", "place of death"])

SYSTEM = (
    "You write training data for a small assistant that answers ONLY from a list of facts it is handed. "
    "Each item gives FACTS, lines 'entity — relation: value', and a TASK. Rules for every answer: use only "
    "the facts; write every name, date and number exactly as the facts write it (an ISO date may be written "
    "out, e.g. 1895-10-31 as October 31, 1895); add NOTHING a fact does not state -- no kind, title, "
    "nationality, adjective, era or context; the facts are always right, even where they differ from what "
    "you know; plain natural English, one or two sentences, no lists. TASK 'profile': the question is given; "
    "answer it with one or two sentences built from the facts (you need not use every fact). TASK 'ask': "
    "first write a natural question a person would ask for the given RELATION of the given ENTITY -- name "
    "the entity exactly as written, vary the wording, never include the answer -- then the answer, one "
    "sentence stating the value(s) the facts give for that entity and relation. Reply with ONLY a JSON "
    "array, one object per item, in order: {\"id\": ..., \"question\": ..., \"answer\": ...}.")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def split_of(entity: str) -> str:
    return "held" if int(hashlib.sha256(norm(entity).encode()).hexdigest(), 16) % 10 == 0 else "train"


def render(facts: list, question: str) -> str:
    """The talk prompt the base is trained and served on: the facts block, the question, 'Answer:'."""
    return "Facts:\n" + "\n".join(f"- {e}{DASH}{r}: {v}" for e, r, v in facts) + f"\nQuestion: {question}\nAnswer:"


def load_entities(min_facts: int):
    """entity -> {relation: [values]} over the whitelist, from wikikg's template facts (subject-centric:
    'V is the R of E' reads as E's R is V, forward and inverse readings alike)."""
    import wikikg as wk
    wk.ensure_data(("triplets",))
    by: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, r, o in wk.load_triples():
        for fact in wk.facts_for(s, r, o):
            head, _, tail = fact.partition(" is the ")
            rel, _, ent = tail.rpartition(" of ")
            if rel in ASKABLE and head and ent and head != ent and len(head) <= 60 and len(ent) <= 60:
                vals = by[ent][rel]
                if head not in vals and len(vals) < 4:
                    vals.append(head)
    out = {}
    for e, rels in by.items():
        if sum(len(v) for v in rels.values()) >= min_facts and not e[:1].isdigit() and len(e.split()) <= 6:
            out[e] = {r: list(v) for r, v in rels.items()}
    return out


def kind_of(rels) -> str:
    if set(rels) & PERSON:
        return "who"
    return "where" if "location" in rels and len(rels) <= 3 else "what"


def facts_of(rng, ent, rels, n_lo=3, n_hi=7, must=None):
    """Up to n facts about ent, one relation each (all values of it), `must` first."""
    order = [r for r in rels if r != must]
    rng.shuffle(order)
    pick = ([must] if must else []) + order
    out, n = [], rng.randint(n_lo, n_hi)
    for r in pick:
        for v in rels[r]:
            out.append([ent, r, v])
        if len(out) >= n:
            break
    return out


def make_items(ents: dict, n: int, rng: random.Random, mix: dict) -> list[dict]:
    names = list(ents)
    by_rel: dict[str, list[str]] = collections.defaultdict(list)
    for e, rels in ents.items():
        for r, vals in rels.items():
            if len(vals) == 1:
                by_rel[r].append(e)
    shared = [r for r, es in by_rel.items() if len(es) >= 20]
    items = []
    fams = [f for f, w in mix.items() for _ in range(int(w * 100))]
    tries = 0
    while len(items) < n and tries < n * 20:
        tries += 1
        fam = rng.choice(fams)
        if fam == "profile":
            e = rng.choice(names)
            f = facts_of(rng, e, ents[e])
            q = rng.choice(PROFILE_Q[kind_of(ents[e])]).replace("{X}", e)
            items.append({"family": fam, "task": "profile", "entity": e, "facts": f, "question": q,
                          "gold": None, "forbid": [], "split": split_of(e)})
        elif fam == "relation":
            e = rng.choice(names)
            r = rng.choice(list(ents[e]))
            f = facts_of(rng, e, ents[e], must=r)
            items.append({"family": fam, "task": "ask", "entity": e, "relation": r, "facts": f,
                          "gold": ents[e][r], "forbid": [], "split": split_of(e)})
        elif fam == "bind":
            r = rng.choice(shared)
            k = rng.choice([2, 2, 3])
            es = rng.sample(by_rel[r], k)
            vals = [ents[x][r][0] for x in es]
            if len({norm(v) for v in vals}) < k or len({split_of(x) for x in es}) > 1:
                continue
            f = []
            for x in es:
                f += facts_of(rng, x, ents[x], n_lo=1, n_hi=3, must=r)
            rng.shuffle(f)
            ask = rng.randrange(k)
            items.append({"family": fam, "task": "ask", "entity": es[ask], "relation": r, "facts": f,
                          "gold": [vals[ask]], "forbid": [v for i, v in enumerate(vals) if i != ask] +
                          [x for i, x in enumerate(es) if i != ask], "split": split_of(es[ask])})
        elif fam == "counter":
            r = rng.choice(shared)
            e, donor = rng.sample(by_rel[r], 2)
            real, fake = ents[e][r][0], ents[donor][r][0]
            if norm(real) == norm(fake) or norm(fake) in norm(e) or norm(real) in norm(fake):
                continue
            f = facts_of(rng, e, ents[e], must=r)
            f = [[a, b, fake] if b == r else [a, b, c] for a, b, c in f]
            items.append({"family": fam, "task": "ask", "entity": e, "relation": r, "facts": f,
                          "gold": [fake], "forbid": [real], "split": split_of(e)})
    for i, it in enumerate(items):
        rng.shuffle(it["facts"]) if it["family"] != "bind" else None
        it["id"] = f"g{i:06d}"
    return items


def prompt_for(batch: list[dict]) -> str:
    rows = []
    for it in batch:
        row = {"id": it["id"], "task": it["task"],
               "facts": [f"{e}{DASH}{r}: {v}" for e, r, v in it["facts"]]}
        if it["task"] == "profile":
            row["question"] = it["question"]
        else:
            row["entity"], row["relation"] = it["entity"], it["relation"]
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=0)


def parse_reply(text: str) -> list[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
    try:
        data = json.loads(t[t.find("["): t.rfind("]") + 1])
        return [d for d in data if isinstance(d, dict)]
    except (ValueError, TypeError):
        return []


def has_value(answer: str, value: str, date_words=None) -> bool:
    """`ask.has_value` -- the host's one definition (moved there 2026-09-25 with the serving value check;
    `date_words` is kept in the signature for the callers that pass it)."""
    from ask import has_value as _has_value
    return _has_value(answer, value)


def check(it: dict, q: str, a: str) -> str | None:
    """None when the record is kept, else the reason."""
    from ask import date_words, grounded_prose
    if not a or len(a.split()) > 70 or "\n" in a.strip() or a.strip().startswith(("-", "*", "{")):
        return "answer_shape"
    if not q or norm(it["entity"]) not in norm(q):
        return "question_lacks_entity"
    lines = [f"{v} is the {r} of {e}" for e, r, v in it["facts"]] + [f"{e} {r}: {v}" for e, r, v in it["facts"]]
    ok, bad = grounded_prose(a, lines, it["entity"])
    if not ok:
        return "ungrounded"
    if it["task"] == "ask":
        if any(f" {norm(g)} " in f" {norm(q)} " for g in it["gold"]):
            return "question_gives_answer"
        if not any(has_value(a, g, date_words) for g in it["gold"]):
            return "gold_missing"
    for v in it["forbid"]:
        if norm(v) and f" {norm(v)} " in f" {norm(a)} ":
            return "forbidden_value"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=400, help="items sent to the model (before checks)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-facts", type=int, default=3)
    ap.add_argument("--absent-frac", type=float, default=0.12, help="absent records, as a share of kept ask records")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "ground_sft.jsonl"))
    a = ap.parse_args()
    t0 = time.perf_counter()
    from openrouter import OpenRouterProposer
    rng = random.Random(a.seed)
    ents = load_entities(a.min_facts)
    print(f"{len(ents):,} entities with >= {a.min_facts} askable facts ({time.perf_counter() - t0:.0f}s)", flush=True)
    mix = {"profile": 0.28, "relation": 0.24, "bind": 0.30, "counter": 0.18}
    items = make_items(ents, a.items, rng, mix)
    batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]

    local = threading.local()
    lock = threading.Lock()
    usage = collections.Counter()

    stop = threading.Event()          # set on a refusal that every later call would hit too (402: no credit)

    def run(batch):
        if not hasattr(local, "llm"):
            local.llm = OpenRouterProposer(a.model, system=SYSTEM, offline=a.offline or stop.is_set(),
                                           max_tokens=3000, reasoning={"effort": "low"})
        local.llm.offline = a.offline or stop.is_set()          # cached replies still count after a stop
        before = dict(local.llm.usage)
        try:
            text = local.llm.chat(prompt_for(batch))
        except Exception as e:                                  # noqa: BLE001 -- counted, never fatal
            code = getattr(e, "code", None)
            with lock:
                usage[f"error_{code or type(e).__name__}"] += 1
            if code in (401, 402, 403):
                stop.set()
            return batch, []
        with lock:
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost"):
                usage[k] += local.llm.usage[k] - before[k]
        return batch, parse_reply(text)

    kept, why = [], collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    done = 0
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for batch, replies in pool.map(run, batches):
            done += 1
            got = {str(r.get("id")): r for r in replies}
            for it in batch:
                r = got.get(it["id"])
                if r is None:
                    why["no_reply"] += 1
                    continue
                q = it.get("question") if it["task"] == "profile" else str(r.get("question") or "").strip()
                ans = " ".join(str(r.get("answer") or "").split())
                reason = check(it, q, ans)
                if reason:
                    why[reason] += 1
                    if len(examples[reason]) < 3:
                        examples[reason].append({"family": it["family"], "q": q, "a": ans,
                                                 "gold": it.get("gold"), "forbid": it["forbid"][:3]})
                    continue
                why[f"kept:{it['family']}"] += 1
                kept.append(dict(it, question=q, answer=ans))
            if done % 50 == 0:
                print(f"  {done}/{len(batches)} calls, kept {len(kept)}, ${usage['cost']:.3f}, "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

    # absent: a kept ask record with its answer taken out of the facts -- the host's line, no model.
    # Every line carrying an answer value goes, not only the asked relation's: wikikg reads one triple
    # both ways, so 'creation: X' and 'work: X' can hold the same value (pilot, 2026-09-24: "What did
    # William Bradford create?" kept 'work: Of Plymouth Plantation' and was labelled don't-know).
    asks = [r for r in kept if r["task"] == "ask"]
    absent = []
    for r in rng.sample(asks, min(len(asks), int(a.absent_frac * len(asks)))):
        gold = {norm(g) for g in r["gold"]}
        near = next((g for g in NEAR if r["relation"] in g), {r["relation"]})   # a question can be answered by a near relation
        f = [x for x in r["facts"] if not (x[0] == r["entity"] and x[1] in near) and norm(x[2]) not in gold]
        if f and not any(g and f" {g} " in f" {norm(x[2])} " for x in f for g in gold):
            absent.append(dict(r, family="absent", facts=f, answer=ABSENT, gold=None,
                               forbid=list(r["gold"]) + r["forbid"], id=r["id"] + "a"))
    records = kept + absent
    for r in records:
        r["prompt"] = render(r["facts"], r["question"])
        r["source"], r["model"] = "wikikg", a.model
    out = pathlib.Path(a.out.replace(".jsonl", f"{a.tag}.jsonl") if a.tag else a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = collections.Counter(f"{r['family']}:{r['split']}" for r in records)
    sent = collections.Counter(it["family"] for it in items)
    manifest = {"version": "ground_sft_v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": a.model,
                "items_sent": len(items), "sent_by_family": dict(sent), "calls": len(batches),
                "verdicts": dict(why), "records": len(records), "counts": dict(counts), "usage": dict(usage),
                "absent_line": ABSENT, "prompt_format": render([["E", "relation", "value"]], "question?"),
                "examples_rejected": examples, "wall_s": round(time.perf_counter() - t0, 1), "output": str(out)}
    mpath = ROOT / "validation" / "logs" / f"ground_sft{a.tag}.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsent {len(items)} items in {len(batches)} calls | kept {len(kept)} + absent {len(absent)} = "
          f"{len(records)} | ${usage['cost']:.3f} | {time.perf_counter() - t0:.0f}s")
    if stop.is_set():
        print("STOPPED EARLY: the API refused (see usage error_*); only cached replies after that point. "
              "Rerun when fixed -- every reply so far is cached and costs nothing again.")
    print("verdicts: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print(f"wrote {out}\nwrote {mpath}")


if __name__ == "__main__":
    main()
