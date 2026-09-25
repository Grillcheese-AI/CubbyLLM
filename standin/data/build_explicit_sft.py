"""The open side of the explicit gate (H-E7): adult story continuations, taught to appear ONLY behind the
host's `Explicit: allowed` line -- and the same requests without the line met by the gate answer.

    python standin/data/build_explicit_sft.py --src <bluuwhale_nsfwstory2.full.jsonl> --tokenizer <tokenizer.json>
        [--pairs 6000]

Source (listed for the owner's disclosure): the unified corpus's `bluuwhale_nsfwstory2.full.jsonl`, adult
stories cut into ~2,400-character chunks (id, text, source_path, chunk_index). A record is a request to
continue the story, the end of one chunk as the story so far, and the start of the next chunk as the
answer.

Rules, in order:
- ILLEGAL, whole story dropped: any chunk of the story mentions a minor, a school below university, a
  child's age or a word for a child. Deliberately broad -- adult stories that merely mention a kid go with
  it. Never gated, never kept.
- the continuation must be explicit itself (build_chat_sft's scoring on what it SAYS), else it teaches
  nothing about the gate.
- the host's guard: every name and number in the continuation already occurs in the story so far
  (`ask.grounded_prose` through `chat_view`), the same rule as every other talk family.
- each kept continuation is written twice: `story_open` (the line, the continuation) and `story_gated`
  (no line, the gate answer). Split by story, so a held story was never trained on.
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

from build_chat_sft import COMMON, chat_view, language  # noqa: E402
from explicit_gate import GATED, UNLOCK, _MINOR, _MINOR_STORY, _score, minor  # noqa: E402,F401

SOURCE = "unified/bluuwhale_nsfwstory2.full.jsonl (adult stories, 2,400-character chunks)"
_END = re.compile(r"[.!?][\"'”’)]?(?=\s)")           # a sentence's last character (and a closing quote)
_GAP = re.compile(r"[.!?][\"'”’)]?\s+")              # the end of a sentence and the space after it


def stories(src: str):
    """Consecutive chunks of one story: same source_path, chunk_index counting up from where it started."""
    cur, key, last = [], None, None
    for line in open(src, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k, i = r.get("source_path"), r.get("chunk_index", 0)
        if cur and (k != key or i != last + 1):
            yield cur
            cur = []
        cur.append(r)
        key, last = k, i
    if cur:
        yield cur


def cut_after(text: str, limit: int) -> str:
    """The first `limit` characters, cut back to the last sentence end."""
    head = text[:limit]
    ends = [m.end() for m in _END.finditer(head)]
    return head[:ends[-1]].strip() if ends else ""


def cut_before(text: str, limit: int) -> str:
    """The last `limit` characters, starting at a sentence start."""
    tail = text[-limit:]
    m = _GAP.search(tail)
    return tail[m.end():].strip() if m else ""


def split_of(story_id: str) -> str:
    return "held" if int(hashlib.sha1(story_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


def build(src: str, tk, common, pairs: int, max_len: int):
    from ask import grounded_prose
    funnel, out = collections.Counter(), []
    for chunks in stories(src):
        funnel["stories"] += 1
        if any(minor(c["text"]) for c in chunks):
            funnel["illegal_story"] += 1
            continue
        sid = f"{chunks[0].get('source_path')}:{chunks[0].get('id')}"
        for a, b in zip(chunks, chunks[1:]):
            if funnel["open"] >= pairs:
                break
            funnel["pairs_seen"] += 1
            before, after = cut_before(a["text"], 1200), cut_after(b["text"], 700)
            if len(before) < 300 or len(after.split()) < 20:
                funnel["too_short"] += 1; continue
            if language(before + "\n" + after) != "en":
                funnel["language"] += 1; continue
            if not (_score(after) >= 2 or (_score(after) >= 1 and _score(before) >= 2)):
                funnel["not_explicit"] += 1; continue
            if not grounded_prose(chat_view(after, common), [before], "")[0]:
                funnel["ungrounded"] += 1; continue
            q = f"Continue the story:\n\n{before}"
            open_p = f"{UNLOCK}\nQuestion: {q}\nAnswer:"
            if len(tk.encode(open_p).ids) + len(tk.encode(" " + after).ids) + 1 > max_len:
                funnel["too_long"] += 1; continue
            rid = hashlib.sha1(f"{sid}:{a.get('chunk_index')}".encode()).hexdigest()[:16]
            base = {"split": split_of(sid), "explicit": True, "source": SOURCE}
            out.append({"id": f"story:{rid}", "family": "story_open", "prompt": open_p, "answer": after, **base})
            out.append({"id": f"story:{rid}:gated", "family": "story_gated", "prompt": f"Question: {q}\nAnswer:",
                        "answer": GATED, **base})
            funnel["open"] += 1
        if funnel["open"] >= pairs:
            break
    return out, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "explicit_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "explicit_sft.manifest.json"))
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    common = set(COMMON.read_text(encoding="utf-8").split())
    out, funnel = build(args.src, tk, common, args.pairs, args.max_len)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"source": SOURCE, "pairs": args.pairs, "max_len": args.max_len, "funnel": dict(funnel.most_common()),
                "families": dict(collections.Counter(r["family"] for r in out)),
                "splits": dict(collections.Counter(r["split"] for r in out))}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in funnel.most_common():
        print(f"{k:14s} {v:8d}")


if __name__ == "__main__":
    main()
