"""The decoder on CubbyModel.step: the guards do what sample_text's did, the loop
is broken, and the returned state is resumable."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.core.decoding import DecodeConfig, NGramBlock, choose, generate, process  # noqa: E402

V = 12


class Looper:
    """A rigged model that always prefers 1 > 2 > 3 > ... , whatever it sees:
    greedy decoding of it is the tightest possible loop ('1 1 1 1 ...')."""

    def __init__(self):
        self.consumed = []

    def step(self, token, state=None):
        self.consumed.append(int(token[0]))
        lg = torch.tensor([[float(V - i) for i in range(V)]])
        lg[0, 0] = -1.0
        return lg, (0 if state is None else state) + 1


def _ngrams(ids, n):
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def test_unguarded_greedy_loops_on_the_rigged_model():
    out, _ = generate(Looper(), [0], DecodeConfig(max_new_tokens=10, repetition_penalty=1.0,
                                                   no_repeat_ngram=0))
    assert out == [1] * 10


def test_ngram_block_never_repeats_an_ngram_of_prompt_plus_output():
    prompt = [5, 1, 1, 2]
    out, _ = generate(Looper(), prompt, DecodeConfig(max_new_tokens=40, repetition_penalty=1.0,
                                                     no_repeat_ngram=3))
    grams = _ngrams(prompt + out, 3)
    assert len(grams) == len(set(grams)), "a 3-gram repeated"


def test_repetition_penalty_is_sample_texts_frequency_aware_form():
    lg = np.array([2.0, -2.0, 1.0, 0.5], dtype=np.float32)
    got = process(lg, generated=[0, 0, 1], banned=set(), penalty=1.5)
    # token 0 seen twice (positive: divided by 1.5**2), token 1 once (negative: *1.5)
    np.testing.assert_allclose(got, [2.0 / 2.25, -3.0, 1.0, 0.5], rtol=1e-6)
    assert lg[0] == 2.0, "process must not modify its input"


def test_guards_never_ban_every_candidate():
    lg = np.full(V, -np.inf, dtype=np.float32)
    lg[3] = 1.0                                   # a top-K head left one candidate
    got = process(lg, generated=[], banned={3}, penalty=1.0)
    assert np.isfinite(got[3])


def test_eos_stops_and_is_not_returned():
    out, _ = generate(Looper(), [0], DecodeConfig(max_new_tokens=10, eos_id=1))
    assert out == []


def test_top_k_1_sampling_is_greedy_and_seeds_are_deterministic():
    lg = np.random.default_rng(0).standard_normal(V).astype(np.float32)
    rng = np.random.default_rng(0)
    assert choose(lg, DecodeConfig(temperature=0.7, top_k=1), rng) == int(np.argmax(lg))
    a = [choose(lg, DecodeConfig(temperature=1.0, top_k=5, top_p=0.9), np.random.default_rng(7))
         for _ in range(3)]
    b = [choose(lg, DecodeConfig(temperature=1.0, top_k=5, top_p=0.9), np.random.default_rng(7))
         for _ in range(3)]
    assert a == b


def test_returned_state_has_consumed_prompt_and_every_output_token():
    m = Looper()
    prompt = [5, 6, 7]
    out, state = generate(m, prompt, DecodeConfig(max_new_tokens=6))
    assert m.consumed == prompt + out and state == len(prompt) + len(out)
    m2 = Looper()
    out2, state2 = generate(m2, prompt, DecodeConfig(max_new_tokens=6, eos_id=out[2]))
    assert m2.consumed == prompt + out2 and state2 == len(prompt) + len(out2)


def test_ngram_index_matches_a_rescan():
    rng = np.random.default_rng(1)
    ids = [int(x) for x in rng.integers(0, 4, 60)]
    blk = NGramBlock(3, ids[:10])
    for t in range(10, 60):
        want = {ids[i + 2] for i in range(t - 2) if ids[i:i + 2] == ids[t - 2:t]}
        assert blk.banned(ids[:t]) == want
        blk.push(ids[:t + 1])


def _tiny_model():
    """test_assembly's small CubbyModel (its head keeps only top-8 of 40)."""
    from cubbyllm.core.config import CubbyConfig
    from cubbyllm.core.context import FrozenSlotRouter
    from cubbyllm.core.generation import HyperGenerator, SnapshotHardener
    from cubbyllm.model.assembly import CubbyModel
    from cubbyllm.model.backbone import MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    D, Vm, CTX = 16, 40, 8
    return CubbyModel(
        config=CubbyConfig(d_model=D, n_layers=2, ctx_dim=CTX, vocab_core=Vm),
        context_source=FrozenSlotRouter(input_dim=D, n_slots=4, ctx_dim=CTX).freeze(),
        backbone=MinGRUBackbone(D, n_layers=2),
        memory=MemoryLayer(generator=HyperGenerator(ctx_dim=CTX, n_out=D * D),
                           hardener=SnapshotHardener(), d_model=D),
        binding=BindingHead(), embedding=HybridEmbedding(vocab_core=Vm, d_model=D),
        head=TopKRetrievalHead(torch.randn(Vm, D)), retrieval_k=8)


def test_cubbymodel_generate_runs_on_the_real_step_path():
    torch.manual_seed(0)
    m = _tiny_model()
    out, state = m.generate([1, 2, 3], max_new_tokens=8)
    assert len(out) == 8 and all(0 <= t < m.config.vocab_core for t in out)
    assert state["ctx_n"] == 3 + 8
    more, state = m.generate([4], state=state, max_new_tokens=2)
    assert len(more) == 2 and state["ctx_n"] == 3 + 8 + 1 + 2


# ── the guards, on the device ────────────────────────────────────────────
#
# The numpy forms above stay the reference. These assert the device path
# makes the same DECISION -- exactly, because that is the thing the
# decoder promises and the thing a backend swap must not change.
#
# The logits themselves are compared to **one ULP**, and the reason is
# specific rather than a shrug at float32: the penalty divides, and a
# device divide is not required to be correctly rounded. Vulkan allows
# ``OpFDiv`` 2.5 ULP; grilly2 measures 2 against numpy over 200000 pairs
# (grilly2's ``docs/kernels.md``, "What is exact, and what is two ULP"),
# while CUDA's divide is correctly rounded. On real torch this comparison
# is therefore exact and the bound is slack; on grilly2 it is the bound.
# Everything that does NOT divide -- the n-gram ban, the never-ban-
# everything rule, the choice -- is still asserted at zero.
#
# One thing this bound deliberately does not cover: ``penalty ** count``
# comes from a table numpy built, not a device ``pow``, precisely so the
# difference stays at the divide. A device pow measured 3e-8 relative,
# which is ~500 ULP at these magnitudes and is visible to an argmax.


def _within_one_ulp(a, b):
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    return int(np.abs(ia - ib).max())


def _rigged(vocab=64, seed=0):
    """Logits with ties, negatives and an -inf tail — the three cases the
    penalty and the ban treat differently."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(vocab).astype(np.float32) * 3.0
    v[vocab // 2:] = -np.inf                  # a top-K-masked head's tail
    return torch.from_numpy(v)


@pytest.mark.parametrize("penalty", [1.0, 1.3, 2.0])
@pytest.mark.parametrize("banned", [set(), {1, 2, 3}, set(range(0, 32))])
def test_the_device_guards_agree_with_the_numpy_reference(penalty, banned):
    from cubbyllm.core.decoding import DeviceGuards

    cfg = DecodeConfig(temperature=0.0, repetition_penalty=penalty, no_repeat_ngram=3)
    logits = _rigged()
    generated = [0, 5, 5, 9, 9, 9]
    guards = DeviceGuards(logits.shape[0], cfg, logits.device)
    for t in generated:
        guards.observe(t)
    rng = np.random.default_rng(0)

    host = process(logits.numpy(), generated, banned, penalty)
    device = guards.guard(guards.apply(logits), banned).numpy()
    both_inf = np.isinf(host) & np.isinf(device)
    assert (np.sign(host[both_inf]) == np.sign(device[both_inf])).all()
    bound = 0 if penalty == 1.0 else 1        # no divide happens at 1.0
    drift = _within_one_ulp(host[~both_inf], device[~both_inf])
    assert drift <= bound, f"the guards drifted to {drift} ULP, allowed {bound}"
    # the decision itself is exact
    assert choose(host, cfg, rng) == guards.choose(
        guards.guard(guards.apply(logits), banned), np.random.default_rng(0))


def test_the_device_path_also_refuses_to_ban_every_candidate():
    """The same rule as the host path, decided on the device so the logits
    still do not come back."""
    from cubbyllm.core.decoding import DeviceGuards

    cfg = DecodeConfig(temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=3)
    logits = _rigged(vocab=8)
    everything = set(range(8))
    guards = DeviceGuards(8, cfg, logits.device)
    kept = guards.guard(guards.apply(logits), everything)
    assert torch.isfinite(kept).any(), "the device path banned every candidate"
    assert torch.equal(kept, logits)


def test_a_ban_list_longer_than_the_padding_is_truncated_not_dropped():
    """``max_banned`` bounds the per-token work. Overflow bans the first
    ``max_banned`` rather than failing or banning none."""
    from cubbyllm.core.decoding import DeviceGuards

    cfg = DecodeConfig(repetition_penalty=1.0, no_repeat_ngram=3, max_banned=4)
    logits = torch.zeros(16)
    guards = DeviceGuards(16, cfg, logits.device)
    out = guards.guard(logits, set(range(10)))
    assert int(torch.isinf(out).sum()) == 4


def test_the_two_guard_paths_generate_the_same_tokens():
    """End to end on the rigged model the file already uses, so the
    comparison includes the n-gram bookkeeping and the penalty counts."""
    a, _ = generate(Looper(), [0], DecodeConfig(max_new_tokens=24, device_guards=True))
    b, _ = generate(Looper(), [0], DecodeConfig(max_new_tokens=24, device_guards=False))
    assert a == b

    cfg = dict(max_new_tokens=24, temperature=0.8, top_k=8, top_p=0.9, seed=3)
    a, _ = generate(Looper(), [0], DecodeConfig(device_guards=True, **cfg))
    b, _ = generate(Looper(), [0], DecodeConfig(device_guards=False, **cfg))
    assert a == b, "sampling must draw from the same candidate set on both paths"
