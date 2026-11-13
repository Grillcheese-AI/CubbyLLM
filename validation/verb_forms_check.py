"""Order 0072 (Nick, 2026-11-13): the French verb forms made by rule (`cubbyllm/speech/verb_kinds.py`) and the French
cues of `cubbyllm/speech/task_verbs.py`, checked against UniMorph French (CC BY-SA 3.0, from Wiktionary).

UniMorph is kept off-repo (``$CUBBY_DATASETS/unimorph_fra/fra``: lemma, form, features) and read only here: the log
carries counts and this repo's own forms that disagree, never a UniMorph row, and nothing reads UniMorph at run time.

For each verb: the infinitive is a UniMorph lemma, and the imperative tu / nous / vous made by rule are UniMorph's
forms with features IMP 2 SG / 1 PL / 2 PL (a multi-word verb, "mettre à jour", is checked on its first word). For
each one-word French task cue: it is some UniMorph form (a verb form, or another word: "résumé" is a noun too).

  python validation/verb_forms_check.py [--out validation/logs/verb_forms_check_<date>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import task_verbs, verb_kinds  # noqa: E402

__wiring__ = "STANDALONE"

PERSON = {"tu": ("2", "SG"), "nous": ("1", "PL"), "vous": ("2", "PL")}


def load(path: Path):
    imp, lemmas, forms = defaultdict(set), set(), set()
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        lemma, form, feats = parts[0], parts[1], set(parts[2].split(";"))
        lemmas.add(lemma)
        forms.add(form.lower())
        if "IMP" in feats:
            for who, (p, n) in PERSON.items():
                if p in feats and n in feats and "NEG" not in feats:
                    imp[(lemma, who)].add(form.lower())
    return imp, lemmas, forms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = Path(local_paths.value("CUBBY_DATASETS") or ".") / "unimorph_fra" / "fra"
    imp, lemmas, forms = load(path)
    rows, agree, checked, missing_lemma = [], 0, 0, []
    for name, v in verb_kinds.VERBS.items():
        inf = v["fr"]["infinitive"]
        head = inf.split(" ")[0]
        if head not in lemmas:
            missing_lemma.append(inf)
            continue
        for who in PERSON:
            ours = v["fr"][who].split(" ")[0]
            known = imp.get((head, who), set())
            if not known:
                continue
            checked += 1
            ok = ours in known
            agree += ok
            if not ok:
                rows.append({"verb": name, "infinitive": inf, "person": who, "ours": ours})
    cues = sorted({c for _, fr, _, _ in task_verbs.TASKS.values() for c in fr if " " not in c and "'" not in c})
    folded = {f.translate(str.maketrans("éèêàâîôûçë", "eeeaaiouce")) for f in forms}
    cue_found = [c for c in cues if c in forms or c in folded]  # the cues are written without accents (folded)
    out = {"order": "0072", "what": "French verb forms by rule and task cues, checked against UniMorph French (off-repo)",
           "verbs": len(verb_kinds.VERBS), "infinitives_known_to_unimorph": len(verb_kinds.VERBS) - len(missing_lemma),
           "infinitives_unknown": missing_lemma, "imperatives_checked": checked, "imperatives_agreeing": agree,
           "our_forms_disagreeing": rows, "task_cues_one_word": len(cues), "task_cues_found": len(cue_found),
           "task_cues_not_found": [c for c in cues if c not in cue_found],
           "unimorph": "CC BY-SA 3.0; read here only, no row of it in this log or the repo",
           "notes": {"extraire, tu": "UniMorph's form for the 2sg imperative is the subjunctive's; the imperative is "
                                    "'extrais' (ours, kept)",
                     "infinitives_unknown": "not UniMorph lemmas (lire, créer, mettre among them): not checked here",
                     "task_cues_not_found": "a noun and an adverb, not verb forms"}}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
