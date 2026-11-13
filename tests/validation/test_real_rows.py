"""Order 0067, the reader's sixth round: real prompts carry a rule or a phrase without losing a word of either; the
held-out phrases never reach a training row; IFBench joins the overlap check only from round 6, so earlier rounds'
rows rebuild to their recorded hashes; the structure phrases are not IFBench's or IFEval's held-out kinds."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "validation" / "builders"), str(ROOT / "validation")]
import bench_overlap as BO  # noqa: E402
import real_rows as RR  # noqa: E402
import struct_phrases as SP  # noqa: E402


def test_a_piece_put_inside_a_prompt_keeps_both_whole():
    rng = random.Random(0)
    prompt = "How do tides work? I live near the sea and I keep wondering.\n\nThanks!"
    for _ in range(200):
        out = RR.embed(prompt, "Answer in fewer than 50 words.", rng)
        assert "Answer in fewer than 50 words." in out
        assert all(w in out for w in prompt.split())


def test_held_out_phrases_never_train():
    train, held = RR._phrases(6)
    for lang in ("en", "fr"):
        assert held[lang] and train[lang]
        assert not {k for k, _ in train[lang]} & {k for k, _ in held[lang]}
        assert not {w for _, w in train[lang]} & {w for _, w in held[lang]}


def test_every_structure_phrase_has_two_wordings_a_language():
    keys = [k for k, _, _ in SP.STRUCT]
    assert len(keys) == len(set(keys))
    assert all(len(en) == 2 and len(fr) == 2 for _, en, fr in SP.STRUCT)
    assert len(SP.IFBENCH_KINDS) == 58


def test_ifbench_joins_the_overlap_check_only_when_asked():
    assert BO.WITH_IFBENCH is False                      # rounds 1-5 rebuild to their recorded hashes
    if not (Path(BO.local_paths.value("CUBBY_DATASETS") or ".") / "ifbench" / "ifbench_test.parquet").exists():
        return
    assert "IFBench" not in BO.benchmark_grams()
    assert "IFBench" in BO.benchmark_grams(ifbench=True)


def test_screened_prompts_carry_no_constraint_words():
    assert RR._screen("Why was Lima founded where it is, and who chose the place?", "en")
    assert not RR._screen("What is the capital of Peru?", "en")    # "capital" can name a rule: left out
    assert not RR._screen("Answer in at most 20 words: what is rain?", "en")
    assert not RR._screen("Réponds en trois phrases : pourquoi le ciel est bleu ?", "fr")
    assert RR._screen("Pourquoi les forgerons utilisent-ils du borax pour souder ?", "fr")
