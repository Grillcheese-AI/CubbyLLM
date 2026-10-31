import ast
R = r"C:\Users\grill\Documents\GitHub\CubbyLLM"
p = R + r"\standin\ask.py"
s = open(p, encoding="utf-8").read()
old = '''            def learn():
                return learn_and_answer(question, lambda q, k: [], self._run_fn, store=self.world, known=self.known,
                                        source=source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                        plan=plan, resolvers=self.resolvers, choose=parse_choose(choose) or None,
                                        fallbacks=self.fallbacks, before=[self.books] if self.books is not None else None)'''
new = '''            def learn(src=None):                         # `src`: another source for this walk (the reader's: none)
                return learn_and_answer(question, lambda q, k: [], self._run_fn, store=self.world, known=self.known,
                                        source=src or source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                        plan=plan, resolvers=self.resolvers, choose=parse_choose(choose) or None,
                                        fallbacks=self.fallbacks if src is None else [],
                                        before=[self.books] if self.books is not None and src is None else None)'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = R + r"\standin\loci_reader.py"
s = open(p, encoding="utf-8").read()
old = '''    lr = learn()'''
new = '''    # the store already holds several values (the reader defers): the walk decides on the store alone, no fetch --
    # the fact was learned once, and asking again is the source call the next time must not make
    lr = learn(NoSource()) if vals and first["miss"] and str(first.get("why", "")).startswith("several") else learn()'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''def through_loci('''
new = '''class NoSource:
    """A source that states nothing: the walk on what the store holds."""
    name = "none"

    def facts(self, entity, **_kw):
        return []


def through_loci('''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
