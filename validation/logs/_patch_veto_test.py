p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\tests\test_books_bulk.py"
s = open(p, encoding="utf-8").read()
i = s.index("def test_what_the_bulk_pass_holds_is_never_said_even_by_a_live_read(tmp_path):")
s = s[:i] + '''def test_what_the_bulk_pass_holds_is_never_said_even_from_the_store(tmp_path):
    """A held (subject, relation) -- two values of a relation that has one, in different books -- vetoes an answer
    whoever states it: here a store fact said flat a moment before. The real bulk output loads Napoleon's birth date."""
    import json as _json
    from ask import AskLoop
    from test_ask import FakeEmitter
    _index(tmp_path)
    bulk.run(tmp_path / "books.sqlite", tmp_path / "bulk", workers=0, wiki={}, log=lambda m: None)
    plans = {"Where was Napoleon Bonaparte born?": ("Napoleon Bonaparte", ["place of birth"])}
    world = TimedStore(STORE + ["Ajaccio is the place of birth of Napoleon Bonaparte"])
    world.index.declare_relation("place of birth")
    loop = AskLoop(FakeEmitter(plans), world=world, source=DictSource({}), lexicon=False, run_fn=exact_vm())
    assert loop.ask("Where was Napoleon Bonaparte born?")["verified"]                  # flat, before any veto
    loop.load_held(tmp_path / "bulk" / "worlds" / "_held.jsonl")
    assert ("napoleon bonaparte", "date of birth") in loop.contested
    held = tmp_path / "held.jsonl"
    held.write_text(_json.dumps({"fact": "Corte is the place of birth of Napoleon Bonaparte", "rel": "place of birth",
                                 "subject": "Napoleon Bonaparte", "tier": "held"}) + "\\n", encoding="utf-8")
    assert loop.load_held(held) == 1
    rec = loop.ask("Where was Napoleon Bonaparte born?")
    assert not rec["verified"] and rec["reason"] == "contested" and rec["refused"]["answer"] == "Ajaccio"
'''
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
