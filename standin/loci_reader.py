"""standin/loci_reader.py -- the 450M's loci path as the ask loop's reader (order 0007): given an (entity,
relation) and what the store holds about the entity, the trunk itself answers from a scene, or says it does not
know.

The scene is the layout the 450M was trained on (`make_fact_shard --lead`, the far-fact shard run 5 read 0.946 on):
`Facts:` then each fact the store holds about the entity as `<subject> <relation> <object>.`, the object's pieces
WRITING at the fact's address (`make_fact_shard.address`, piece i at +6 i z), then `Question: What is the
<relation> of <subject>? here` (ASKING for piece 0) and `Answer: ` (its last token asking for piece 0 too). The
answer is decoded greedily, each piece in turn asking for the next; the newline ends it. A **miss**: the relation
is one the shard never addressed, nothing is written about the entity, the copy's read-aware gate stays shut at the
first piece (lam below 0.5), or the first piece is the newline. The relations are the shard's own (the wiki
world's, `RELATION_OF` maps Wikidata's labels and the world's nouns onto them).

The checkpoint and the tokenizer come from the environment (`CUBBY_LOCI_CKPT`, `CUBBY_SPM`), never a committed
path. CPU is fine: a scene is ~100-300 tokens.

Wired: WIRED (`ask.AskLoop(loci=...)`, `serve_api --loci`).
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "validation", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

__wiring__ = "WIRED"

# Wikidata's labels -> the shard's relations (the wiki world's UPPER_SNAKE names); the world's own nouns join below
WIKIDATA = {"date of birth": "BORN_ON", "place of birth": "BORN_IN", "date of death": "DIED_ON", "place of death": "DIED_IN",
            "place of burial": "BURIED_AT", "founded by": "FOUNDED_BY", "occupation": "OCCUPATION", "educated at": "EDUCATED_AT",
            "spouse": "SPOUSE_OF", "country of citizenship": "NATIONALITY", "employer": "EMPLOYED_BY", "member of": "MEMBER_OF",
            "part of": "PART_OF", "location": "LOCATED_IN", "located in the administrative territorial entity": "LOCATED_IN",
            "author": "AUTHORED_BY", "director": "DIRECTED_BY", "genre": "GENRE", "residence": "RESIDED_IN",
            "field of work": "FIELD_OF_WORK", "position held": "POSITION_HELD", "notable work": "CREATOR_OF"}
STEP = 6


def relation_of() -> dict[str, str]:
    """A relation as the store words it (Wikidata's label, the world's noun) -> the shard's relation name."""
    import wikikg as wk
    from cubbyllm.reasoning.planner import normalize
    out = {normalize(k): v for k, v in WIKIDATA.items()}
    for upper, (noun, _inv, _form) in wk.RELATIONS.items():
        out.setdefault(normalize(noun), upper)
    return out


class LociReader:
    name = "loci"

    def __init__(self, ckpt: str | None = None, tokenizer: str | None = None, device: str = "cpu", max_pieces: int = 16):
        import torch
        from tokenizers import Tokenizer
        from train_base import build_model
        import wikikg as wk
        self.torch, self.max_pieces = torch, int(max_pieces)
        ck = torch.load(ckpt or os.environ["CUBBY_LOCI_CKPT"], map_location="cpu", weights_only=False)
        self.model = build_model(dict(ck["meta"]), torch.device(device))
        with torch.no_grad():
            for p, s in zip(self.model.parameters(), ck["params"]):
                p.copy_(s.to(p.device))
        del ck
        tok = tokenizer or os.environ.get("CUBBY_SPM") or os.path.join(
            os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS"), "base450m", "grilly", "tokenizer.json")
        self.tk = Tokenizer.from_file(tok)
        self.eos, self.nl = self.tk.token_to_id("</s>"), self.tk.encode("\n", add_special_tokens=False).ids[-1]
        self.rel_index = {r: i for i, r in enumerate(sorted({r for _s, r, _o in wk.load_triples()}))}
        self.rel_of = relation_of()
        self.calls = 0

    def shard_relation(self, relation: str) -> str | None:
        from cubbyllm.reasoning.planner import normalize
        r = self.rel_of.get(normalize(relation))
        return r if r in self.rel_index else None

    def scene(self, entity: str, relation: str, facts) -> tuple[list, list, list]:
        """(ids, places, roles) of the scene up to `Answer: `, for `facts` [(subject, object, shard relation)]."""
        from make_fact_shard import address, words
        ids, pos, roles = [], [], []

        def put(text, at=None, role=0):
            t = self.tk.encode(text, add_special_tokens=False).ids
            ids.extend(t); pos.extend([at] * len(t)); roles.extend([role if at else 0] * len(t))

        put("Facts:\n")
        for s, o, R in facts:
            put(f"{words(s)} {words(R)} ")
            a0 = address(s, self.rel_index[R])
            for i, tok in enumerate(self.tk.encode(words(o), add_special_tokens=False).ids):
                ids.append(tok); pos.append((a0[0], a0[1], a0[2] + STEP * i)); roles.append(1)
            put(".\n")
        R = self.shard_relation(relation)
        at = address(entity, self.rel_index[R])
        put(f"Question: What is the {words(R)} of {words(entity)}?")
        put(" ")
        put("here", at, 2)
        put("\nAnswer: ")
        pos[-1], roles[-1] = at, 2                                    # --lead: the token before the answer asks
        return ids, pos, roles

    def read(self, entity: str, relation: str, facts) -> dict:
        """`facts`: [(subject, relation as the store words it, object)] about `entity`. -> {answer, miss, why, lam}."""
        from cubbyllm.model.spatial import NO_PLACE
        from make_fact_shard import address
        torch = self.torch
        R = self.shard_relation(relation)
        if R is None:
            return {"answer": None, "miss": True, "why": "a relation the shard never addressed"}
        rows = [(s, o, self.shard_relation(r)) for s, r, o in facts if self.shard_relation(r)]
        if not rows:
            return {"answer": None, "miss": True, "why": "nothing written about it"}
        ids, pos, roles = self.scene(entity, relation, rows)
        at = address(entity, self.rel_index[R])
        cp = self.model.spatial.copy_path
        lams, out = [], []
        for j in range(self.max_pieces):
            seen = []
            hook = cp.register_forward_hook(lambda _m, inp, o: seen.append((inp[4], o[2][:, 0])))
            try:
                with torch.no_grad():
                    x = torch.tensor([ids])
                    P = torch.tensor([[p if p else (NO_PLACE, NO_PLACE, NO_PLACE) for p in pos]])
                    placed = P[..., 0] != NO_PLACE
                    positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
                    Rl = torch.tensor([roles])
                    logits = self.model.forward(x, positions=positions, placed=placed,
                                                writes=placed & ((Rl & 1) != 0), asks=placed & ((Rl & 2) != 0))
            finally:
                hook.remove()
            self.calls += 1
            last = len(ids) - 1
            lam = next((float(v) for ix, lv in seen for i, v in zip(ix.tolist(), lv.tolist()) if i == last), None)
            lams.append(lam)
            tok = int(logits[0, -1].argmax())
            if j == 0 and (lam is None or lam < 0.5):
                return {"answer": None, "miss": True, "why": "the gate stayed shut", "lam": lams}
            if tok in (self.nl, self.eos):
                break
            out.append(tok)
            ids.append(tok); pos.append((at[0], at[1], at[2] + STEP * (j + 1))); roles.append(2)
        answer = self.tk.decode(out).strip()
        return {"answer": answer or None, "miss": not answer, "why": None if answer else "nothing read", "lam": lams}


def through_loci(loop, rec: dict, rels: list, seed: str, learn) -> dict | None:
    """The ask loop's one-hop question through the reader (order 0007); `learn()` runs `learn_and_answer` as today.
    The trunk reads first: a reading the host can check against the store's value is spoken when equal and refused
    when not; a miss (or a reading with no stored value to check) goes to the lookup, and once the gate admits
    the fact the trunk reads again under the same check. -> `rec` filled in (the path that answered, the source
    calls), or None for a question the reader does not take (several hops)."""
    from cubbyllm.reasoning.planner import normalize
    if len(rels) != 1:
        return None
    reader, rel = loop.loci, rels[0]
    R = reader.shard_relation(rel)

    def about():
        return [(t.subj, t.rel, t.obj) for _f, t in loop.world.index._by_subj.get(normalize(seed), [])]

    def stored():
        return {normalize(o): o for _s, r, o in about() if reader.shard_relation(r) == R}

    first, vals = reader.read(seed, rel, about()), stored()
    rec.update(path="loci", source_calls=0, loci=[{k: v for k, v in first.items() if k != "lam"}])
    if not first["miss"] and vals:
        ok = normalize(first["answer"]) in vals
        rec.update(answer=vals[normalize(first["answer"])] if ok else None, verified=ok,
                   reason=None if ok else "loci_mismatch", refused=None if ok else {"read": first["answer"], "stored": list(vals.values())})
        return rec
    lr = learn()
    second, vals = reader.read(seed, rel, about()), stored()
    ok = bool(vals) and not second["miss"] and normalize(second["answer"]) in vals
    rec.update(path="loci_after_lookup" if vals else "lookup", source_calls=len(lr.entities), learned=[
        {"fact": p.fact, "status": p.status, "source": p.source, "entity": p.entity} for p in lr.learned],
        entities=list(lr.entities), loci=rec["loci"] + [{k: v for k, v in second.items() if k != "lam"}],
        answer=vals[normalize(second["answer"])] if ok else None, verified=ok,
        reason=None if ok else ("loci_mismatch" if vals and not second["miss"] else lr.result.reason or "loci_miss"),
        refused=None if ok else ({"read": second["answer"], "stored": list(vals.values())} if vals else lr.result.refused))
    return rec
