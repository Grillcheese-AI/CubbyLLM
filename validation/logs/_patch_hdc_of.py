import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\hdc_store.py"
s = open(p, encoding="utf-8").read()
old = '''    def hop(self, plan, hop: int, entity: str | None) -> list:
        """A hop as `TripleIndex.hop` serves it. A tail that names its entity by an alias or a Q-id ("rurtania") is
        accepted as the tail with the entity's label: the set writes every entity by one of its aliases."""
        if hop == 0:
            cands, seen = [], set()
            for exact in (True, False):
                for rel, ent in tail_splits(plan.tail):
                    lab = self.label(ent)
                    as_label = plan if not lab or normalize(lab) == ent else type(plan)(
                        relations=plan.relations, tail=f"{rel} of {lab}", n_hop=plan.n_hop, answer_class=plan.answer_class)
                    for f, t in self._by_subj.get(ent):
                        same = normalize(t.rel) == normalize(rel)
                        if f not in seen and (same if exact else self._rel_n.get(normalize(t.rel), 0) >= 2) \\
                                and accepts(as_label, hop, entity, t):
                            cands.append((f, t)); seen.add(f)
            return cands'''
new = '''    def hop(self, plan, hop: int, entity: str | None) -> list:
        """A hop as `TripleIndex.hop` serves it. A tail that names its entity by an alias or a Q-id ("rurtania") is
        accepted as the tail with the entity's label: the set writes every entity by one of its aliases. A label that
        ends in 'of' ('instance of', 'part of') is asked without it -- "What is the instance of X?" cannot hold the
        label's own 'of' -- so it serves the plan's 'instance' when the subject has no plain 'instance' fact (where it
        has one, 'capital' and 'capital of' read opposite ways, and only the plain one is asked; 2026-10-22: 141 of
        the hdc bench's 200 one-hop questions were refused for this)."""
        if hop == 0:
            cands, seen = [], set()
            for exact in (True, False):
                for rel, ent in tail_splits(plan.tail):
                    lab = self.label(ent)
                    as_label = plan if not lab or normalize(lab) == ent else type(plan)(
                        relations=plan.relations, tail=f"{rel} of {lab}", n_hop=plan.n_hop, answer_class=plan.answer_class)
                    facts = self._by_subj.get(ent)
                    plain = any(normalize(t.rel) == normalize(rel) for _f, t in facts)
                    for f, t in facts:
                        of_label = not plain and normalize(t.rel) == f"{normalize(rel)} of"
                        asked = as_label if not of_label else type(plan)(
                            relations=plan.relations, tail=f"{t.rel} of {t.subj}", n_hop=plan.n_hop, answer_class=plan.answer_class)
                        same = normalize(t.rel) == normalize(rel) or of_label
                        if f not in seen and (same if exact else self._rel_n.get(normalize(t.rel), 0) >= 2) \\
                                and accepts(asked, hop, entity, t):
                            cands.append((f, t)); seen.add(f)
            return cands'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\n"))
