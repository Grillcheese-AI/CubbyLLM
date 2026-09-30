"""H-E6, the host's view: what an asker would actually hear from a gate run, once the host's checks sit
between the talk adapter and the reply (`ask.talk_reply`: the VM returned nothing -> the host says "The
facts don't say." itself; otherwise the draft is spoken only if the name-and-number guard AND the value
check pass). Reads a finished gate run's answers -- no model is run -- and rebuilds its items with the
gate's own sampling, so the rows line up one for one (checked by id).

The pre-registered bars are the MODEL's and stay where they are; this adds the number the product is held
to: wrong answers SPOKEN, which must be 0.

    python validation/exp_e6_host_view.py --data standin/data/out/ground_sft.jsonl --run _v1
"""
from __future__ import annotations

import argparse, collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ask import talk_reply  # noqa: E402
from build_ground_sft import norm  # noqa: E402
from exp_e6_talk_gate import bind_items, judge  # noqa: E402


def gate_items(data, per_family, seed):
    """The gate's own draw, in its order (exp_e6_talk_gate.main)."""
    rng = random.Random(seed)
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    held = []
    for fam in ("relation", "bind", "counter", "absent", "profile"):
        pool = [r for r in recs if r["split"] == "held" and r["family"] == fam]
        held += rng.sample(pool, min(per_family, len(pool)))
    return held + bind_items(per_family, rng)


def host_view(rec, draft):
    facts = rec["facts"]
    lines = [f"{v} is the {r} of {e}" for e, r, v in facts] + [f"{e} {r}: {v}" for e, r, v in facts]
    fam = rec["family"]
    if fam == "absent":
        returned = []                                    # the VM found no line for the asked relation
    elif fam == "profile":
        returned = [v for _, _, v in facts]
    else:
        returned = list(rec["gold"])
    keep = {norm(x) for x in returned} | {norm(rec["entity"])}
    others = []
    if fam != "profile":
        for e, _, v in facts:
            for x in (e, v):
                if norm(x) not in keep and x not in others:
                    others.append(x)
    reply, why = talk_reply(returned, draft, lines, rec["entity"], others)
    if reply is None:
        return "refused", why
    outcome = judge(rec, reply)["outcome"]
    return ("spoken_correct" if outcome == "correct" else "spoken_wrong"), why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", default="_v1", help="the gate run's tag")
    ap.add_argument("--arm", default="adapter")
    args = ap.parse_args()
    run = json.load(open(os.path.join(HERE, "logs", f"exp_e6_talk_gate{args.run}.json"), encoding="utf-8"))
    a = run["args"]
    items = gate_items(args.data, a["per_family"], a["seed"])
    rows = run[args.arm]["rows"]
    assert [r["id"] for r in rows] == [it["id"] for it in items], "the gate's draw did not reproduce"
    by = collections.defaultdict(collections.Counter)
    wrong = []
    for it, row in zip(items, rows):
        o, why = host_view(it, row["answer"])
        by[it["family"]][o] += 1
        if o == "spoken_wrong":
            wrong.append((it["family"], it.get("question"), row["answer"]))
    out = [f"H-E6 host view | gate run {args.run}, arm {args.arm}, {len(rows)} held records"]
    for fam in ("relation", "bind", "bind_t", "counter", "absent", "profile"):
        c = by[fam]; n = sum(c.values())
        if n:
            out.append(f"  {fam:9s} n={n:3d}  spoken & correct {c['spoken_correct'] / n:6.1%}  refused {c['refused']:3d}  "
                       f"SPOKEN & WRONG {c['spoken_wrong']}")
    total = sum(c["spoken_wrong"] for c in by.values())
    out.append(f"  spoken & wrong, all families: {total}")
    for w in wrong[:10]:
        out.append(f"    [{w[0]}] {w[1]!r} -> {w[2]!r}")
    print("\n".join(out))
    path = os.path.join(HERE, "logs", f"exp_e6_host_view{args.run}")
    open(path + ".log", "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({f: dict(c) for f, c in by.items()}, open(path + ".json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
