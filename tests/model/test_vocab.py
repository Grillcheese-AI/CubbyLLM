import cubbyllm.model.vocab.embedding as emb_mod
import cubbyllm.model.vocab.output_head as head_mod
from cubbyllm.core.protocols import Wiring
from cubbyllm.model.vocab import (
    EmbeddingSource,
    HybridEmbedding,
    RetrievalHead,
    TopKRetrievalHead,
)


def test_hybrid_embedding_core_lookup():
    import torch

    emb = HybridEmbedding(vocab_core=100, d_model=16)
    assert isinstance(emb, EmbeddingSource)
    ids = torch.tensor([1, 2, 3])
    out = emb.embed(ids)
    assert out.shape == (3, 16)


def test_retrieval_head_keeps_only_top_k():
    import torch

    torch.manual_seed(0)
    codebook = torch.randn(50, 16)
    head = TopKRetrievalHead(codebook)
    assert isinstance(head, RetrievalHead)
    query = codebook[7].unsqueeze(0)          # query points at row 7
    lg = head.logits(query, k=5)
    assert lg.shape == (1, 50)
    finite = torch.isfinite(lg[0])
    assert int(finite.sum()) == 5             # exactly K survive; rest are -inf
    assert int(lg[0].argmax()) == 7           # the true match is retrieved


def test_retrieval_local_softmax_is_over_candidates():
    import torch
    import torch.nn.functional as F

    codebook = torch.randn(30, 8)
    head = TopKRetrievalHead(codebook)
    lg = head.logits(codebook[3].unsqueeze(0), k=4)
    probs = F.softmax(lg, dim=-1)[0]
    assert abs(float(probs.sum()) - 1.0) < 1e-5   # normalized over the K survivors
    assert int((probs > 0).sum()) == 4


def test_fixed_head_has_no_params_learnable_head_does():
    import torch

    fixed = TopKRetrievalHead(torch.randn(20, 8))
    assert list(fixed.parameters()) == []            # fixed VSA codebook: no grads
    learn = TopKRetrievalHead(torch.randn(20, 8), learnable=True)
    params = list(learn.parameters())
    assert len(params) == 1 and params[0].requires_grad
    assert params[0].shape == (20, 8)


def test_full_softmax_when_k_ge_vocab():
    import torch

    head = TopKRetrievalHead(torch.randn(12, 8), learnable=True)
    lg = head.logits(torch.randn(1, 8), k=12)        # k == vocab -> no masking
    assert bool(torch.isfinite(lg).all())


def test_wiring_declared():
    assert emb_mod.__wiring__ is Wiring.WIRED
    assert head_mod.__wiring__ is Wiring.WIRED
