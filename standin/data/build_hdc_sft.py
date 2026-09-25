"""More of the talk adapter's fact families (H-E6), in the VM's own facts-block form, from the hypernet
scaling-law set (Wikidata facts written from relation templates). talk_v2 had them at 11% of its mix and
lost bind, absent and profile to the chat and passage families; v2.1 puts the fact share back.

    python standin/data/build_hdc_sft.py --src <hdc dir> [--rows 400000] [--seed 0]

Source (listed for the owner's disclosure): the `hdc` folder of the local datasets drive --
`train_scaling_law_with_facts.pq` (1M questions, 1-3 hops, each with the template sentences that answer it),
`ood_validation_rephrased.pq` (10k of the same with the questions reworded), and
`relation_template_mapping.csv` (the template each fact sentence was written from). The facts carry the
set's own aliases -- one random alias per entity, often a typo, a code or another language ("Huamn",
"united stated", "p:ca", "PolanD", "royaume-uni"). Every name is replaced by the entity's exact English
Wikidata label (`--names`, see load_names: the alias found in Wikidata5M's alias lists -> its Q-id -> the
label from Wikidata's label dump), and a row holding a name with no usable label is dropped.

Every fact sentence is read back into `entity — relation: value` through its template (the relation is
the Wikidata label: 'instance of', 'follows', 'part of'). Families, the same contract as build_ground_sft:
  fact_relation  one entity's facts; the question asks one of them -> the sentence that states it
  fact_bind      2-3 entities with the same relation, different values -> the asked one's, none of the others'
  fact_counter   the asked fact carries another entity's value -> the value the facts give
  fact_chain     a 2-3 hop question over the chain's facts -> each hop, in order, ending on the answer
  fact_absent    any of the above with the answer-bearing line taken out -> "The facts don't say."
Answers are templated from the facts (no model); each passes build_ground_sft.check -- the host's
grounded_prose, the gold value present, every forbidden value absent, the question naming its entity and
not containing its answer. Split by the asked entity with build_ground_sft.split_of, the same hash the
H-E6 gate's held entities come from; a train block never holds a held entity.
"""
from __future__ import annotations

import argparse, collections, csv, hashlib, json, pathlib, random, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_ground_sft import ABSENT, check, norm, render, split_of  # noqa: E402

SOURCE = "hdc: hypernet scaling-law set (Wikidata facts from relation templates)"
_LATIN = re.compile(r"^[\x20-\x7e -ɏ‐-‧]+$")   # Latin script and common punctuation only


class Templates:
    """relation_template_mapping.csv: a fact sentence is '<value> is <noun>', noun from noun_template with
    {relation} filled (the edited label when there is one) and {subject} the entity."""

    def __init__(self, path: str):
        self.by_pre: dict[str, list] = collections.defaultdict(list)
        nouns: dict[str, tuple] = {}
        for r in csv.DictReader(open(path, encoding="utf-8")):
            word = (r["edited_relation"] or r["relation"]).strip()
            if word[:1].isupper() and word[1:].islower():
                word = word.lower()                                # 'Predecessor' reads as a word, not a name
            noun = r["noun_template"].replace("{relation}", word)
            # two relations written to one sentence ('capital of' edited to 'capital' and 'capital' itself):
            # the sentence says 'the capital of S', so it is read as the relation it names
            if noun in nouns and not nouns[noun][1]:
                continue
            nouns[noun] = (r["relation"].strip(), r["edited_relation"].strip())
        for noun, (label, _) in nouns.items():
            pre, _, suf = noun.partition("{subject}")
            self.by_pre[pre].append((suf, label, noun))

    def parse(self, fact: str):
        """-> (entity, label, value, noun) or None. The first ' is ' that a template matches; the longest
        template there."""
        start = 0
        while True:
            i = fact.find(" is ", start)
            if i < 0:
                return None
            v, rem = fact[:i], fact[i + 4:]
            words, best = rem.split(" "), None
            for k in range(1, min(len(words), 9)):
                pre = " ".join(words[:k]) + " "
                for suf, label, noun in self.by_pre.get(pre, ()):
                    if len(rem) > len(pre) + len(suf) and (not suf or rem.endswith(suf)):
                        e = rem[len(pre):len(rem) - len(suf)]
                        if best is None or len(pre) + len(suf) > best[0]:
                            best = (len(pre) + len(suf), e.strip(), label, v.strip(), noun)
            if best:
                return best[1:]
            start = i + 1


SAY = {"instance of": "{e} is an instance of {v}",                      # 'the instance of X is V' reads backwards
       "languages spoken, written or signed": "the language of {e} is {v}"}   # the template's own '... by of X'


def clause(noun: str, e: str, v: str, label: str = "") -> str:
    """One fact as a clause. 'the capital of X' -> 'the capital of X is V'; a noun that ends on a verb
    ('the country X is in') reads better the other way round -> 'X is in V' / 'V is the league X competes in'."""
    if label in SAY:
        return SAY[label].format(e=e, v=v)
    n = noun.replace("{subject}", e)
    if noun.endswith("{subject}"):
        return f"{n} is {v}"
    if noun == "the country {subject} is in":
        return f"{e} is in {v}"
    return f"{v} is {n}"


def sentence(clauses: list[str]) -> str:
    body = clauses[0] if len(clauses) == 1 else (
        f"{clauses[0]}, and {clauses[1]}" if len(clauses) == 2 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}")
    return body[:1].upper() + body[1:] + "."


def load(src: pathlib.Path, rows: int, seed: int):
    import pyarrow.parquet as pq
    t = pq.read_table(src / "train_scaling_law_with_facts.pq", columns=["question_prompt", "answer", "n_hop", "facts"])
    d = t.to_pydict()
    idx = list(range(t.num_rows))
    random.Random(seed).shuffle(idx)
    out = [("train", d["question_prompt"][i], d["answer"][i], d["n_hop"][i], d["facts"][i]) for i in idx[:rows]]
    r = pq.read_table(src / "ood_validation_rephrased.pq", columns=["question_prompt", "answer", "n_hop", "facts"]).to_pydict()
    out += [("rephrased", r["question_prompt"][i], r["answer"][i], r["n_hop"][i], r["facts"][i])
            for i in range(len(r["answer"]))]
    return out


def load_names(path: str) -> dict[str, str]:
    """alias -> proper name, from the label table (hdc_names.tsv[.gz]: alias, qid, name, how). The set writes
    every entity by one random alias of its Wikidata5M list ('Huamn', 'united stated', 'p:ca', 'PolanD'); the
    table gives the entity's exact English Wikidata label ('human', 'United States of America', 'Canada',
    'Poland'), or '' where none is usable (a Wikimedia list or disambiguation page, no English label)."""
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    out = {}
    with op(path, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            alias, _, name, _ = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
            out[alias] = name
    return out


def rename(q: str, alias: str, name: str) -> str | None:
    """The question with the entity's alias replaced by its name; None when the question does not hold it."""
    if alias in q:
        return q.replace(alias, name)
    m = re.search(re.escape(alias), q, re.I)
    return q[:m.start()] + name + q[m.end():] if m else None


def parse_rows(rows, tpl: Templates, names: dict[str, str] | None = None):
    """-> parsed rows [(kind, question, answer, [(e, label, v, noun)])], the facts index, and a funnel.
    With `names`, every entity and value is written by its proper name, and a row with one that has none is dropped."""
    funnel, parsed = collections.Counter(), []
    ent: dict[str, dict[tuple, tuple]] = collections.defaultdict(dict)      # norm(e) -> (label, norm v) -> fact
    for kind, q, a, hop, facts in rows:
        funnel["rows"] += 1
        if not facts:
            funnel["no_facts"] += 1; continue
        hops = [tpl.parse(x) for x in facts]
        if any(h is None for h in hops):
            funnel["unparsed"] += 1; continue
        if names is not None:
            if any(not names.get(h[0]) or not names.get(h[2]) for h in hops):
                funnel["no_proper_name"] += 1; continue
            q2 = rename(q, hops[0][0], names[hops[0][0]])
            if q2 is None:
                funnel["question_lacks_alias"] += 1; continue
            q, a = q2, names.get(a, a)
            hops = [(names[e], lab, names[v], noun) for e, lab, v, noun in hops]
        if not all(_LATIN.match(x) for x in [q] + [h[0] for h in hops] + [h[2] for h in hops]):
            funnel["script"] += 1; continue
        if any(norm(e) == norm(v) or not norm(e) or not norm(v) or len(v) > 60 or len(e) > 80 for e, _, v, _ in hops):
            funnel["degenerate"] += 1; continue
        if any(norm(hops[i][2]) != norm(hops[i + 1][0]) for i in range(len(hops) - 1)):
            funnel["broken_chain"] += 1; continue
        if norm(hops[-1][2]) != norm(a):
            funnel["answer_not_last"] += 1; continue
        parsed.append((kind, q.strip(), a, hops))
        for h in hops:
            ent[norm(h[0])].setdefault((h[1], norm(h[2])), h)
    return parsed, ent, funnel


def build(parsed, ent, rng: random.Random, quota: dict, absent_share: float):
    splits: dict[str, str] = {}

    def sp(x):
        k = norm(x)
        if k not in splits:
            splits[k] = split_of(x)
        return splits[k]

    def context(e, skip, n):
        """Up to n more facts about e: other relations, one value each."""
        have = [h for (lab, _), h in ent.get(norm(e), {}).items() if lab not in skip]
        rng.shuffle(have)
        out, labs = [], set()
        for h in have:
            if h[1] not in labs:
                labs.add(h[1]); out.append(h)
            if len(out) >= n:
                break
        return out

    one = [p for p in parsed if len(p[3]) == 1]
    multi = [p for p in parsed if len(p[3]) > 1]
    by_label = collections.defaultdict(list)
    for p in one:
        by_label[p[3][0][1]].append(p[3][0])
    funnel, kept, seen = collections.Counter(), [], set()
    count = collections.Counter()

    def keep(fam, q, e, lab, facts, gold, forbid, clauses, split, extra=None):
        facts = [list(x[:3]) for x in facts]
        if split == "train" and any(sp(x[0]) != split for x in facts):     # a held record may hold train entities
            funnel[f"{fam}:mixed_split"] += 1; return
        key = (fam, norm(e), lab, norm(q))
        if key in seen:
            funnel[f"{fam}:duplicate"] += 1; return
        if fam != "fact_bind":
            rng.shuffle(facts)
        it = {"family": fam, "task": "ask", "entity": e, "relation": lab, "facts": facts, "gold": gold,
              "forbid": forbid, "split": split, **(extra or {})}
        a = sentence(clauses)
        reason = check(it, q, a)
        if reason:
            funnel[f"{fam}:{reason}"] += 1; return
        seen.add(key)
        count[fam] += 1
        kept.append(dict(it, question=q, answer=a))

    def filler(e, split, n):
        out = []
        for _ in range(n * 4):
            o = rng.choice(one)[3][0]
            if norm(o[0]) != norm(e) and sp(o[0]) == split:
                out.append(o)
            if len(out) >= n:
                break
        return out

    order = one[:]
    rng.shuffle(order)
    for kind, q, a, hops in order:
        pending = [f for f in ("fact_relation", "fact_bind", "fact_counter") if count[f] < quota[f]]
        if not pending:
            break
        fam = rng.choice(pending)
        e, lab, v, noun = hops[0]
        split = sp(e)
        if count[(fam, lab)] >= max(50, int(0.08 * quota[fam])):      # no relation over 8% of a family
            funnel[f"{fam}:relation_cap"] += 1; continue
        count[(fam, lab)] += 1
        if fam == "fact_relation" and count[fam] < quota[fam]:
            ctx = context(e, {lab}, rng.randint(2, 5))
            ctx += filler(e, split, max(0, 2 - len(ctx)))
            keep(fam, q, e, lab, [hops[0]] + ctx, [v], [], [clause(noun, e, v, lab)], split, {"source_kind": kind})
        elif fam == "fact_bind" and count[fam] < quota[fam]:
            k, others = rng.choice([2, 2, 3]), []
            for _ in range(30):
                o = rng.choice(by_label[lab])
                if (norm(o[0]) != norm(e) and norm(o[2]) != norm(v) and sp(o[0]) == split
                        and all(norm(o[0]) != norm(x[0]) and norm(o[2]) != norm(x[2]) for x in others)):
                    others.append(o)
                if len(others) == k - 1:
                    break
            if len(others) < k - 1:
                funnel[f"{fam}:no_partner"] += 1; continue
            block = [hops[0]] + others
            for x in [hops[0]] + others:
                block += context(x[0], {lab}, rng.randint(0, 2))
            rng.shuffle(block)
            keep(fam, q, e, lab, block, [v], [o[2] for o in others] + [o[0] for o in others],
                 [clause(noun, e, v, lab)], split, {"source_kind": kind})
        elif fam == "fact_counter" and count[fam] < quota[fam]:
            donor = rng.choice(by_label[lab])
            fake = donor[2]
            if (norm(fake) == norm(v) or norm(fake) in norm(e) or norm(v) in norm(fake) or norm(fake) in norm(q)
                    or norm(fake) in norm(v)):
                funnel[f"{fam}:bad_donor"] += 1; continue
            ctx = [c for c in context(e, {lab}, rng.randint(2, 5)) if norm(v) not in norm(c[2])]
            keep(fam, q, e, lab, [(e, lab, fake)] + ctx, [fake], [v], [clause(noun, e, fake, lab)], split,
                 {"source_kind": kind})
    rng.shuffle(multi)
    for kind, q, a, hops in multi:
        if count["fact_chain"] >= quota["fact_chain"]:
            break
        e = hops[0][0]
        labs = {h[1] for h in hops}
        ctx = []
        for h in hops:
            ctx += context(h[0], labs, rng.randint(0, 1))
        ctx = [c for c in ctx if norm(c[2]) != norm(a)]
        keep("fact_chain", q, e, hops[-1][1], list(hops) + ctx[:3], [hops[-1][2]], [],
             [clause(h[3], h[0], h[2], h[1]) for h in hops], sp(e), {"source_kind": kind, "hops": len(hops)})
    # absent: a kept record with its answer taken out -- the asked line, every line holding the answer as
    # value or as subject, and for a chain one hop of it
    absent = []
    for r in rng.sample(kept, int(absent_share * len(kept))):
        gold = {norm(g) for g in r["gold"]}
        f = [x for x in r["facts"] if norm(x[2]) not in gold and norm(x[0]) not in gold
             and not (norm(x[0]) == norm(r["entity"]) and x[1] == r["relation"])]
        if r["family"] == "fact_chain":
            hop = rng.randrange(r["hops"])
            chain_lines = [x for x in r["facts"] if norm(x[2]) not in gold]
            f = [x for x in f if x not in chain_lines[hop:hop + 1]] if chain_lines else f
        if not f or any(g and f" {g} " in f" {norm(x[2])} " for x in f for g in gold):
            funnel["fact_absent:answer_left"] += 1; continue
        absent.append(dict(r, family="fact_absent", from_family=r["family"], facts=f, answer=ABSENT, gold=None,
                           forbid=list(r["gold"]) + r["forbid"]))
    return kept + absent, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the hdc folder (parquet files + relation_template_mapping.csv)")
    ap.add_argument("--rows", type=int, default=400000, help="scaling-law rows read (shuffled, seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relation", type=int, default=14000)
    ap.add_argument("--bind", type=int, default=12000)
    ap.add_argument("--counter", type=int, default=7000)
    ap.add_argument("--chain", type=int, default=9000)
    ap.add_argument("--absent-share", type=float, default=0.35)
    ap.add_argument("--names", default=str(ROOT / "standin" / "data" / "out" / "hdc_names.tsv.gz"),
                    help="alias -> proper name table (see load_names); built from Wikidata labels")
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "hdc_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "hdc_sft.manifest.json"))
    args = ap.parse_args()
    t0 = time.time()
    src = pathlib.Path(args.src)
    tpl = Templates(str(src / "relation_template_mapping.csv"))
    rows = load(src, args.rows, args.seed)
    names = load_names(args.names)
    parsed, ent, funnel = parse_rows(rows, tpl, names)
    print(f"parsed {len(parsed):,} of {len(rows):,} rows, {len(ent):,} entities ({time.time() - t0:.0f}s)", flush=True)
    quota = {"fact_relation": args.relation, "fact_bind": args.bind, "fact_counter": args.counter,
             "fact_chain": args.chain}
    records, f2 = build(parsed, ent, random.Random(args.seed), quota, args.absent_share)
    funnel.update(f2)
    out = []
    for r in records:
        rid = hashlib.sha1(f"{r['family']}|{r['entity']}|{r['relation']}|{r['question']}".encode()).hexdigest()[:16]
        out.append({"id": f"hdc:{rid}", "family": r["family"], "split": r["split"],
                    "prompt": render(r["facts"], r["question"]), "answer": r["answer"], "entity": r["entity"],
                    "relation": r["relation"], "gold": r["gold"], "forbid": r["forbid"], "source": SOURCE,
                    "source_kind": r.get("source_kind"), **({"from_family": r["from_family"]} if "from_family" in r else {})})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = collections.Counter(f"{r['family']}:{r['split']}" for r in out)
    rng = random.Random(1)
    examples = {fam: [{"prompt": r["prompt"], "answer": r["answer"]} for r in
                      rng.sample([x for x in out if x["family"] == fam], min(3, sum(x["family"] == fam for x in out)))]
                for fam in sorted({r["family"] for r in out})}
    manifest = {"source": SOURCE, "rows_read": len(rows), "parsed": len(parsed), "quota": quota,
                "absent_share": args.absent_share, "records": len(out), "by_family": dict(sorted(fams.items())),
                "funnel": dict(funnel.most_common()), "examples": examples, "wall_s": round(time.time() - t0, 1)}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("records", "by_family", "funnel", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
