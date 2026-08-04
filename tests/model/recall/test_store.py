import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.recall import EpisodicStore

DK, D = 16, 32


def test_write_grows_and_retrieves_the_nearest_by_cosine():
    torch.manual_seed(0)
    s = EpisodicStore(DK, D)
    keys = torch.randn(10, DK); vals = torch.randn(10, D)
    s.write(keys, vals)
    assert len(s) == 10
    # query = one of the stored keys -> its own value comes back on top
    q = keys[4]
    k_top, v_top = s.retrieve_cosine(q, topk=3)
    assert k_top.shape == (3, DK) and v_top.shape == (3, D)
    assert torch.allclose(v_top[0], vals[4], atol=1e-5)


def test_state_dict_round_trip_preserves_the_store():
    s = EpisodicStore(DK, D)
    s.write(torch.randn(5, DK), torch.randn(5, D))
    sd = s.state_dict()
    s2 = EpisodicStore(DK, D); s2.load_state_dict(sd)
    assert len(s2) == 5
    assert torch.equal(s2.keys, s.keys) and torch.equal(s2.values, s.values)
