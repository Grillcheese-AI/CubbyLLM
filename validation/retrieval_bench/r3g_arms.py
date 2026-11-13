"""r3g_arms -- order 0070: R3o's relation arms, each able to score any set of properties (the row's candidates, the
whole codebook for G1, the candidates and the expected-but-absent ones for G2), on this interpreter's CPU.

  lexicon_content     the frozen lexicon's wordings found in the words left once the names are out (R3o's)
  450m, 450m_wd       the saved program's ``HOP1`` wording: the serving property table's properties for the whole
                      wording (``r3o.WHOLE``), and the lexicon's wordings found inside it (R3o's); programs read back
  qfhrr@0.0, @0.25    the qFHRR codebook (``r3_qfhrr.codebook``) against the context's code, the cosine; the same codes
                      ``r3_qfhrr_score`` builds on grilly2, scored here in numpy (the scorer's own check) and rounded to 6
                      decimals as that scorer hands them over (a tie there is a tie here) -- on every
                      one of R3o's 908 scored rows the lists and margins are R3o's (``Arms.check`` in the run)

Wired: STANDALONE (order 0070's bench).
"""
from __future__ import annotations

import json
import os

import numpy as np

__wiring__ = "STANDALONE"

TAUS = (0.0, 0.25)
QF = tuple(f"qfhrr@{t}" for t in TAUS)
MODELS = ("450m", "450m_wd")
ARMS = ("lexicon_content",) + MODELS + QF
PROGRAMS = {"450m": "r3_programs_dev_n300.jsonl", "450m_wd": "r3_programs_dev_n300_cubby450m_wd_cont.jsonl"}


def _unit(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class Arms:
    """What every arm needs, loaded once: the lexicon, the grammar, the programs, the codebook's codes."""

    def __init__(self, lx: dict, gram) -> None:
        from cubbyllm.worlds.fact_pool import V4_TABLE
        from cubbyllm.worlds.fast_words import FastWordEncoder
        from retrieval_bench import r3_qfhrr as Q
        from retrieval_sets.common import data_root
        self.lx, self.gram = lx, gram
        self.enc = FastWordEncoder.from_npz(str(V4_TABLE))
        book = Q.codebook(lx)
        self.codes = {}
        for lang in ("en", "fr"):
            keys = sorted(k for k in book if k.endswith("|" + lang))
            names = [k.split("|")[0] for k in keys]
            self.codes[lang] = (names, np.stack([_unit(self.enc.encode(book[k])) for k in keys]), {n: i for i, n in enumerate(names)})
        self.programs = {}
        for arm, f in PROGRAMS.items():
            got = {}
            for line in open(os.path.join(data_root(), f), encoding="utf-8"):
                d = json.loads(line)
                if "id" in d:
                    got[d["id"]] = d.get("program")
            self.programs[arm] = got

    def context(self, r: dict) -> dict:
        """The row's readings, which no property set changes: the wordings found, the programs' wordings, the grammar's
        properties, the context's code."""
        from retrieval_bench import r3_450m as M
        from retrieval_bench import r3_arms as A
        from retrieval_bench import r3_qfhrr as Q
        en = self.lx["by_wording"]["en"]
        c = {"lex": A.matches(A.leftover(r["question"], r["spans"]), self.lx["by_wording"].get(r["lang"], {}), True),
             "gram": list(self.gram(r["question"]).get("top") or ()), "lang": r["lang"]}
        for arm in MODELS:
            hop = M.read(self.programs[arm].get(r["id"]))["hop1"]
            if hop:
                wording = hop.replace("$E1", " ").replace("$E2", " ")
                c[arm] = {"hop1": hop, "found": A.matches(wording, en, True), "whole": list(self.gram.aliases.pids(wording))}
            else:
                c[arm] = None
        ctx = Q.words(A.leftover(r["question"], r.get("spans")))
        c["q"] = _unit(self.enc.encode(" ".join(ctx))) if ctx else None
        return c

    def qscores(self, c: dict, allowed) -> dict:
        """{property: cosine} over `allowed` (None: the whole codebook), in the row's language."""
        if c["q"] is None:
            return {}
        names, M, at = self.codes[c["lang"]]
        if allowed is None:
            s = M @ c["q"]
            return {n: round(v, 6) for n, v in zip(names, s.tolist())}
        idx = np.array([at[p] for p in sorted(allowed, key=lambda p: int(p[1:])) if p in at], dtype=np.int64)
        if not len(idx):
            return {}
        return {names[i]: round(float(v), 6) for i, v in zip(idx, M[idx] @ c["q"])}

    def qbests(self, c: dict, carried: set) -> tuple[float, float]:
        """G1's numbers for qFHRR, the whole codebook in one product: (best not carried, best carried)."""
        if c["q"] is None:
            return 0.0, 0.0
        names, M, at = self.codes[c["lang"]]
        s = M @ c["q"]
        mask = np.zeros(len(names), dtype=bool)
        mask[[at[p] for p in carried if p in at]] = True
        own = round(float(s[mask].max()), 6) if mask.any() else 0.0
        absent = round(float(s[~mask].max()), 6) if (~mask).any() else 0.0
        return absent, own

    def scores(self, arm: str, c: dict, allowed) -> dict:
        """The arm's scores over `allowed` (None: every property it can name)."""
        from retrieval_bench.r3o import WHOLE
        if arm in QF:
            return self.qscores(c, allowed)
        if arm == "lexicon_content":
            found, extra = c["lex"], {}
        elif c.get(arm):
            found, extra = c[arm]["found"], {p: WHOLE for p in c[arm]["whole"]}
        else:
            return {}
        out = {p: s for p, s in extra.items() if allowed is None or p in allowed}
        for _w, n, ps in found:
            for p in ps:
                if allowed is None or p in allowed:
                    out[p] = max(out.get(p, 0), n)
        return out

    def out(self, arm: str, c: dict, cands: set) -> dict:
        """The arm's list among `cands`, exactly as R3o hands it over."""
        from retrieval_bench import r3_qfhrr as Q
        from retrieval_bench.r3_arms import top
        if arm in QF:
            sc = self.qscores(c, cands)
            order = sorted(sc, key=lambda p: int(p[1:]))
            vals = np.array([sc[p] for p in order], dtype=np.float32)
            best = np.argsort(-vals, kind="stable")[:20]
            return Q.decide([[order[j], float(vals[j])] for j in best], float(arm.split("@")[1]))
        if arm in MODELS and not c.get(arm):
            return {"read": False, "top": [], "undecided": False}
        sc = self.scores(arm, c, cands)
        return {"read": bool(sc), **top(sc)}

    def named(self, c: dict, cands: set) -> set:
        """G3's support: what the lexicon (any wording found) or the grammar names among the candidates."""
        lex = {p for _w, _n, ps in c["lex"] for p in ps}
        return (lex | set(c["gram"])) & set(cands)


def check(arms: Arms, rows: list[dict], cands: dict, ctxs: dict, logs: dict) -> dict:
    """Every arm's list and judgement on R3o's scored rows against R3o's logs ({arm: log rows by id}): the rows that
    differ (the list, or the judgement, or a qFHRR margin by more than 2e-4)."""
    from retrieval_bench.r3_arms import judge
    diff = {}
    for arm, log in logs.items():
        bad = []
        for r in rows:
            lg = log.get(r["id"])
            if lg is None:
                continue
            o, lo = arms.out(arm, ctxs[r["id"]], cands[r["id"]]), lg["out"].get(arm) or {}
            if (o.get("top", []) != lo.get("top", []) or judge(o, r["golds"]) != lg["judge"][arm]
                    or abs((o.get("margin") or 0) - (lo.get("margin") or 0)) > 2e-4):
                bad.append(r["id"])
        diff[arm] = {"rows": sum(1 for r in rows if r["id"] in log), "differ": len(bad), "first": bad[:5]}
    return diff


__all__ = ["ARMS", "Arms", "MODELS", "QF", "TAUS", "check"]
