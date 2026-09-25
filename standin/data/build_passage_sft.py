"""Passage records for the talk adapter: a real passage as the facts block, a question about it, and the
answer copied from it -- or, with the answer's sentences removed, "The facts don't say."

    python standin/data/build_passage_sft.py --chatqa <dir with squad2.0/ quoref/ tatqa/ narrativeqa/> \
        --tokenizer <bbpe128k tokenizer.json> [--max-len 512]

Source: nvidia/ChatQA-Training-Data, the subsets whose own licences allow it (SQuAD 2.0 CC BY-SA 4.0,
Quoref CC BY 4.0, TAT-QA CC BY 4.0, NarrativeQA Apache-2.0 -- per each original dataset's card; check
before release). Left out on purpose: `synthetic_convqa` (non-commercial, OpenAI terms), NewsQA (not in the
release), ROPES (the release dropped the situation paragraph: 61% of its answers are nowhere in the
document or the question), and DROP / TAT-QA arithmetic (the number is computed, which is the VM's job,
not the talk adapter's).

Why this family: talk_v1 missed H-E6's absent bar (73.4%) by speaking when the facts were silent. ChatQA's
SQuAD 2.0 holds only the answerable half, so the silent cases are made here, the way build_ground_sft makes
its absent family: remove the sentences that hold the answer, and keep the record only if nothing that
could answer survives -- neither the answer span nor any of its content words anywhere in what is left.

Every record passes `ask.grounded_prose` against its own facts block. The passage is rendered one sentence
per `- ` line, the same block the VM's facts arrive in. Answers are the datasets' own spans, unrewritten.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

SOURCE = "nvidia/ChatQA-Training-Data"
LICENSES = {"squad2.0": "CC BY-SA 4.0", "quoref": "CC BY 4.0", "tatqa": "CC BY 4.0", "narrativeqa": "Apache-2.0"}
ABSENT = "The facts don't say."

_BOUND = re.compile(r"([.!?][\"')\]]?)\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][\w'’\-.,]*")
_STOP = frozenset("the a an and or of in on at to for by with from as is was were are be been it its this that "
                  "he she they his her their who which what when where how why not no".split())


_ABBR = frozenset("dr mr mrs ms st jr sr v vs no gen col lt capt prof inc co corp ltd mt ft ca approx e.g i.e etc fig "
                  "jan feb mar apr jun jul aug sep sept oct nov dec rev gov sen rep".split())
_INITIALS = re.compile(r"(?:[A-Z]\.)+[A-Z]?$|[A-Z]$")


def sentences(doc: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of each sentence; start/end are offsets into `doc`. A period after an
    abbreviation or an initial ("U.S.", "Dr.", "State v.") does not end a sentence."""
    out, start = [], 0
    for m in _BOUND.finditer(doc):
        if m.group(1) and m.group(1)[0] == ".":
            before = doc[max(start, m.start() - 12):m.start()].split()
            last = before[-1] if before else ""
            if last.lower().rstrip(".") in _ABBR or _INITIALS.search(last):
                continue
        end = m.start() + len(m.group(1) or "")
        if doc[start:end].strip():
            out.append((start, end, doc[start:end].strip()))
        start = m.end()
    if doc[start:].strip():
        out.append((start, len(doc), doc[start:].strip()))
    return out


def render(lines: list[str], question: str) -> str:
    body = "\n".join(f"- {l}" for l in lines)
    return f"Facts:\n{body}\nQuestion: {question.strip()}\nAnswer:"


def split_of(key: str) -> str:
    return "held" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


def content_words(text: str) -> set[str]:
    return {w.lower().strip("'’-.,") for w in _WORD.findall(text)} - _STOP - {""}


def answer_text(subset: str, x) -> tuple[str, int | None]:
    a = x["answers"][0]
    if subset == "squad2.0":
        return a["text"], a["answer_start"]
    if subset == "narrativeqa":
        return a[0].strip(), None                       # the first of two human-written references
    if subset == "tatqa":
        items = [s.strip() for s in re.findall(r'"([^"]+)"', a)] or [a.strip()]
        return (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]), None
    return a.strip(), None


def build(chatqa: pathlib.Path, tk, max_len: int):
    from ask import grounded_prose
    funnel, out = collections.Counter(), []
    for subset in ("squad2.0", "quoref", "tatqa", "narrativeqa"):
        path = chatqa / subset / ("train_others.json" if subset == "tatqa" else "train.json")
        for i, x in enumerate(json.load(open(path, encoding="utf-8"))):
            funnel[f"{subset}:total"] += 1
            doc, q = x["document"].strip(), x["messages"][-1]["content"]
            ans, start = answer_text(subset, x)
            sents = sentences(doc)
            lines = [s for _, _, s in sents]
            prompt = render(lines, q)
            n = len(tk.encode(prompt).ids) + len(tk.encode(" " + ans).ids) + 1
            if n > max_len:
                funnel[f"{subset}:too_long"] += 1
                continue
            ok, _ = grounded_prose(ans, lines, "")
            if not ok:                                  # NarrativeQA's free answers are checked like any other
                funnel[f"{subset}:ungrounded"] += 1
                continue
            key = f"{subset}:{hashlib.sha1(doc.encode()).hexdigest()[:16]}"
            rid = f"{subset}:{i}"
            out.append({"id": rid, "family": "passage", "split": split_of(key), "prompt": prompt, "answer": ans,
                        "source": f"{SOURCE}/{subset}", "license": LICENSES[subset]})
            funnel[f"{subset}:answer"] += 1
            if subset != "squad2.0":
                continue                                # SQuAD only: Quoref asks about people the rest of its passage keeps naming, and hand checks found it still answerable
            if start is None:
                start = doc.find(ans)
            hold = [(a, b) for a, b, _ in sents if a < start + len(ans) and start < b]
            rest = [s for a, b, s in sents if (a, b) not in hold]
            left = " ".join(rest).lower()
            if not hold or len(rest) < 2 or ans.lower() in left or content_words(ans) & content_words(left):
                funnel[f"{subset}:absent_unsafe"] += 1  # the answer, or a word of it, survives elsewhere
                continue
            out.append({"id": rid + ":absent", "family": "passage_absent", "split": split_of(key),
                        "prompt": render(rest, q), "answer": ABSENT,
                        "source": f"{SOURCE}/{subset}", "license": LICENSES[subset]})
            funnel[f"{subset}:absent"] += 1
    return out, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chatqa", required=True, help="directory holding the ChatQA subset folders")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "passage_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "passage_sft.manifest.json"))
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    out, funnel = build(pathlib.Path(args.chatqa), tk, args.max_len)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"source": SOURCE, "subsets": LICENSES, "max_len": args.max_len, "funnel": dict(sorted(funnel.items())),
                "families": dict(collections.Counter(r["family"] for r in out)),
                "splits": dict(collections.Counter(r["split"] for r in out))}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in sorted(funnel.items()):
        print(f"{k:28s} {v:7d}")


if __name__ == "__main__":
    main()
