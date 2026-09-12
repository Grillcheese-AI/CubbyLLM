"""build_lexicon — WordNet 3.0 (EN lemmas) joined with WOLF 1.0b4 (FR literals) by synset id.

Wired: STANDALONE (a data builder; the reader is cubbyllm/reasoning/lexicon.py).

WOLF (WordNet Libre du Français) keys every French synset by the Princeton WordNet 3.0
offset (`eng-30-<offset>-<pos>`) and carries the French literals but not the English
lemmas; the WordNet 3.0 `dict/data.*` files carry the English lemmas by the same offset.
Joined, one line per synset: {"id", "pos", "en": [lemmas], "fr": [literals], "gloss"}.
This is the SYNONYM oracle for relation wording (lever 5): 'birthplace' and 'place of
birth' are one synset; 'lieu de naissance' is its French literal. Ported data, not a
linked library: the reader is stdlib.

    python standin/data/build_lexicon.py [--wolf standin/data/out/wolf-1.0b4.xml]
                                          [--wndict standin/data/out/dict]
                                          [--out standin/data/out/lexicon_en_fr.jsonl]
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys, time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
_SYN = re.compile(r"<SYNSET>(.*?)</SYNSET>", re.S)
_ID = re.compile(r"<ID>(eng-30-\d{8}-[a-z])</ID>")
_LIT = re.compile(r"<LITERAL[^>]*>(.*?)</LITERAL>", re.S)
_DEF = re.compile(r"<DEF>(.*?)</DEF>", re.S)
POS_FILE = {"n": "data.noun", "v": "data.verb", "a": "data.adj", "s": "data.adj", "r": "data.adv", "b": "data.adv"}


def wordnet_lemmas(wndict: pathlib.Path) -> dict[str, list[str]]:
    """{'eng-30-<offset>-<pos>': [lemma, ...]} from the WordNet 3.0 data files."""
    out: dict[str, list[str]] = {}
    for fname in ("data.noun", "data.verb", "data.adj", "data.adv"):
        for line in (wndict / fname).read_text(encoding="latin-1").splitlines():
            if line.startswith("  ") or not line.strip():
                continue
            head = line.split(" | ", 1)[0].split()
            offset, ss_type, w_cnt = head[0], head[2], int(head[3], 16)
            words = [head[4 + 2 * i] for i in range(w_cnt)]
            pos = "a" if ss_type == "s" else ss_type
            lemmas = [re.sub(r"\(.*?\)$", "", w).replace("_", " ").lower() for w in words]
            out[f"eng-30-{offset}-{pos}"] = lemmas
            if ss_type == "s":                          # WOLF keys satellites under 'a' too; keep both readable
                out.setdefault(f"eng-30-{offset}-s", lemmas)
    return out


def wolf_synsets(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    for m in _SYN.finditer(text):
        body = m.group(1)
        sid = _ID.search(body)
        if not sid:
            continue
        lits = [l.strip() for l in _LIT.findall(body) if l.strip() and l.strip() != "_EMPTY_"]
        d = _DEF.search(body)
        yield sid.group(1), lits, (d.group(1).strip() if d else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wolf", default=str(OUT / "wolf-1.0b4.xml"))
    ap.add_argument("--wndict", default=str(OUT / "dict"))
    ap.add_argument("--out", default=str(OUT / "lexicon_en_fr.jsonl"))
    a = ap.parse_args()
    t0 = time.perf_counter()
    en = wordnet_lemmas(pathlib.Path(a.wndict))
    n = n_fr = n_join = 0
    seen = set()
    with open(a.out, "w", encoding="utf-8") as fh:
        for sid, fr, gloss in wolf_synsets(pathlib.Path(a.wolf)):
            key = sid
            lemmas = en.get(sid) or en.get(sid[:-1] + ("s" if sid.endswith("a") else "a")) or []
            if not lemmas and not fr:
                continue
            seen.add(key); n += 1
            n_fr += bool(fr); n_join += bool(fr and lemmas)
            fh.write(json.dumps({"id": sid, "pos": sid[-1], "en": lemmas, "fr": fr, "gloss": gloss}, ensure_ascii=False) + "\n")
        # WordNet synsets WOLF does not list at all: keep them, English only (the EN synonym oracle is complete)
        for sid, lemmas in en.items():
            if sid not in seen and not sid.endswith("-s"):
                n += 1
                fh.write(json.dumps({"id": sid, "pos": sid[-1], "en": lemmas, "fr": [], "gloss": ""}, ensure_ascii=False) + "\n")
    print(f"{n} synsets written ({n_fr} with French literals, {n_join} with both) in {time.perf_counter() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
