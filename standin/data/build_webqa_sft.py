"""Web-passage records for the talk adapter: a web document as the facts block, one question about it,
the answer written from it -- or, with the sentences that carry the answer removed, "The facts don't say."

    python standin/data/build_webqa_sft.py --src <nemotron_qa.jsonl> --tokenizer <bbpe128k tokenizer.json> \
        [--docs 120000] [--max-len 1024]

Source: the pipeline's `nemotron_qa` file -- Nemotron-CC web documents, each followed by synthetic
"Question: ... Answer: ..." pairs about that document (check the Nemotron-CC terms before release). The
second passage family beside ChatQA's (Wikipedia, stories, finance tables): web prose, opinion, forums.
A 3,000-document probe (2026-09-25): every document holds a Q/A, all fit 1,024 tokens, half hold several,
and 86% of the answers pass the host's guard.

- Documents are drawn at random byte offsets (seeded), one Q/A each, so no site or topic dominates.
- Kept only if `ask.grounded_prose` passes with the document as the facts.
- The absent twin removes every sentence that shares a name or number with the answer, or most of its
  content words, and is kept only if none of the answer's own words (those not already in the question)
  survives anywhere in what is left.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, random, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_chat_sft import explicit  # noqa: E402
from build_passage_sft import ABSENT, content_words, render, sentences, split_of  # noqa: E402

SOURCE = "nemotron_qa (Nemotron-CC web documents with synthetic Q/A)"
_QA = re.compile(r"\n\s*Question:\s*(.+?)\s*Answer:\s*(.+?)\s*(?=\n\s*Question:|\Z)", re.S)
_NAMENUM = re.compile(r"\b(?:[A-Z][\w'’\-]*|\d[\d.,:/\-]*)")


def parse(text: str):
    """(document, [(question, answer), ...])."""
    m = re.search(r"\n\s*Question:", text)
    if not m:
        return None, []
    return text[:m.start()].strip(), [(q.strip(), a.strip()) for q, a in _QA.findall(text[m.start():])]


def carriers(answer: str, sents: list[str]) -> list[int]:
    """Sentences that could carry the answer: sharing a name or number with it, or half its content words."""
    keys = {t.lower() for t in _NAMENUM.findall(answer) if len(t) > 1}
    aw = content_words(answer)
    out = []
    for i, s in enumerate(sents):
        low = s.lower()
        if any(k in low for k in keys) or (aw and len(aw & content_words(s)) >= max(1, len(aw) // 2)):
            out.append(i)
    return out


def sample_docs(src: str, n: int, seed: int):
    size, rng, seen = os.path.getsize(src), random.Random(seed), set()
    with open(src, "rb") as fh:
        tries = 0
        while len(seen) < n and tries < n * 3:
            tries += 1
            fh.seek(rng.randrange(size)); fh.readline()
            line = fh.readline()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            h = hashlib.sha1(text[:2000].encode("utf-8", "replace")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            yield h, text, rng


def build(src, tk, docs, max_len, seed=0):
    from ask import grounded_prose
    funnel, out = collections.Counter(), []
    for h, text, rng in sample_docs(src, docs, seed):
        funnel["docs"] += 1
        doc, qas = parse(text)
        if not doc or not qas:
            funnel["no_qa"] += 1; continue
        q, a = rng.choice(qas)
        lines = [s for _, _, s in sentences(doc)]
        prompt = render(lines, q)
        if len(tk.encode(prompt).ids) + len(tk.encode(" " + a).ids) + 1 > max_len:
            funnel["too_long"] += 1; continue
        if len(a.split()) < 1 or not grounded_prose(a, lines, "")[0]:
            funnel["ungrounded"] += 1; continue
        split = split_of(f"webqa:{h}")
        out.append({"id": f"webqa:{h[:16]}", "family": "passage_web", "split": split, "prompt": prompt, "answer": a,
                    "explicit": explicit(q, a), "source": SOURCE})   # H-E7: what the answer says, tagged
        funnel["answer"] += 1
        drop = set(carriers(a, lines))
        rest = [s for i, s in enumerate(lines) if i not in drop]
        own = content_words(a) - content_words(q)
        if not drop or len(rest) < 2 or (own & content_words(" ".join(rest))) or not own:
            funnel["absent_unsafe"] += 1; continue
        out.append({"id": f"webqa:{h[:16]}:absent", "family": "passage_web_absent", "split": split,
                    "prompt": render(rest, q), "answer": ABSENT, "source": SOURCE})
        funnel["absent"] += 1
    return out, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--docs", type=int, default=120000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "webqa_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "webqa_sft.manifest.json"))
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    out, funnel = build(args.src, tk, args.docs, args.max_len, args.seed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"source": SOURCE, "docs": args.docs, "seed": args.seed, "max_len": args.max_len,
                "funnel": dict(funnel.most_common()),
                "families": dict(collections.Counter(r["family"] for r in out)),
                "splits": dict(collections.Counter(r["split"] for r in out))}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in funnel.most_common():
        print(f"{k:16s} {v:7d}")


if __name__ == "__main__":
    main()
