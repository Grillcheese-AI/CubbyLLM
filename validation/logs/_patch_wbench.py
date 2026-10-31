import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\exp_worlds_bench.py"
s = open(p, encoding="utf-8").read()
reps = [
('''    from question_forms import Forms, TemplateEmitter
    from test_ask import FakeEmitter  # noqa: F401  (the loop's test emitter; TemplateEmitter is the one used)
    from sources import WikidataSource
    fw = store.fiction
    real = store.reality()
    names = set(getattr(getattr(real, "primary", real), "index")._by_subj)''',
 '''    from question_forms import Forms, TemplateEmitter
    from sources import WikidataSource
    fw = store.fiction
    real = store.reality()
    names = set(getattr(real, "primary", real).index._by_subj)
    hdc = (getattr(real, "others", None) or [None])[0]
    known = lambda x: normalize(x) in names or (hdc is not None and hdc.label(x) is not None)   # noqa: E731'''),
('''    from cubbyllm.reasoning.planner import normalize
    picks, seen = [], set()
    for w, s, r, o in rows:
        if len(picks) >= n or (w, s, r) in seen or normalize(s) in names or normalize(o) in names or fw.work_named(f"In {w},") != w:''',
 '''    picks, seen = [], set()
    for w, s, r, o in rows:
        if len(picks) >= n or (w, s, r) in seen or known(s) or known(o) or fw.work_named(f"In {w},") != w:'''),
('''    forms = Forms()
    loop_for = lambda world: AskLoop(TemplateEmitter(forms), world=world, source=WikidataSource(offline=True),  # noqa: E731
                                     lexicon=False)''',
 '''    forms, loops = Forms(), {}

    def loop_for(world):                                  # one loop a world: each starts the CubeLang VM
        if id(world) not in loops:
            loops[id(world)] = AskLoop(TemplateEmitter(forms), world=world, source=WikidataSource(offline=True), lexicon=False)
        return loops[id(world)]'''),
('''        a = store.ask(q, loop_for)''', '''        a = fw.ask(q, loop_for(real), loop_for)'''),
('''        b = store.ask(f"In {w}, {q[0].lower()}{q[1:]}", loop_for)''', '''        b = fw.ask(f"In {w}, {q[0].lower()}{q[1:]}", loop_for(real), loop_for)'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
s = s.replace('''def fiction(store, n, rng) -> dict:
    """In-fiction kin facts whose subject and object no reality world names, asked both ways."""
    from ask import AskLoop''', '''def fiction(store, n, rng) -> dict:
    """In-fiction kin facts whose subject and object no reality world names, asked both ways."""
    from ask import AskLoop
    from cubbyllm.reasoning.planner import normalize''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\n"))
