import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\exp_k10_switch.py"
s = open(p, encoding="utf-8").read()
a = '''    rng, lang = np.random.default_rng(a.seed + 7), kh.Language(np.random.default_rng(a.seed + 8))'''
assert s.count(a) == 1
s = s.replace(a, '''    rng, lang = stream(a)''')
a = '''def copy_only('''
assert s.count(a) == 1
s = s.replace(a, '''def stream(a):
    """The switch-alone phase's documents: new draws (seed + 7) in the joint run's own filler language -- the language's
    grammar is drawn from its seed (seed + 1, as the joint run and the held-out set), and another seed is a language
    the trunk never learned (the first runs used seed + 8: the training loss sat at ~10.5 nats, 2026-10-22)."""
    return np.random.default_rng(a.seed + 7), kh.Language(np.random.default_rng(a.seed + 1))


def copy_only(''', 1)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
t = r"C:\Users\grill\Documents\GitHub\CubbyLLM\tests\test_k10_switch.py"
u = open(t, encoding="utf-8").read()
u += '''

def test_the_switch_alone_trains_in_the_joint_runs_language(joint):
    """The filler language's grammar comes from its seed: the switch's documents must be in the language the trunk
    learned (seed + 1), only new draws of it."""
    a, _rep = joint
    rng, lang = sw.stream(a)
    assert np.array_equal(lang.next, kh.Language(np.random.default_rng(a.seed + 1)).next)
    assert not np.array_equal(lang.next, kh.Language(np.random.default_rng(a.seed + 8)).next)
'''
open(t, "w", encoding="utf-8", newline="\n").write(u)
print("ok")
