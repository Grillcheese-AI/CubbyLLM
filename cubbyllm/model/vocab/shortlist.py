"""cubbyllm/model/vocab/shortlist.py -- a shortlist index over a trained output head's rows (order 0050, stage 1).

The 128k head scores every row at every token. A shortlist scores only K rows, exactly, with the trained rows, and
softmaxes over those (``CB_HEAD_SHORTLIST=1`` with an index attached to the head; off by default, and off the head
is unchanged bit for bit). Built once, offline, deterministically from a seed:
  (a) clusters   k-means over the head's rows; a query probes the clusters whose centroids score best
  (b) router     a linear map from the trunk's final state to cluster scores, trained (``validation/
                 head_shortlist_eval.py``) on the frozen trunk's held-in states against the full head's top-k
  (c) + succ     (b), plus the previous token's most frequent successors (counted on held-in text)
The shortlist: whole clusters in score order until at least K rows (a cluster is never cut, so the rows scored are
K plus at most one cluster; ``sizes`` reports them), with (c)'s successors added. K >= the vocabulary: every row.

numpy only (``cubbyllm`` stays torch-free); ``masked_logits`` takes torch tensors when the torch head calls it, and
the grilly2 harness calls ``shortlist`` with numpy arrays.

Wired: WIRED (``TopKRetrievalHead.logits`` reads it when ``CB_HEAD_SHORTLIST=1`` and an index is attached).
"""
from __future__ import annotations

import os

import numpy as np

__wiring__ = "WIRED"

ARMS = ("clusters", "router", "router+succ")


def on() -> bool:
    """``CB_HEAD_SHORTLIST=1``: the head scores a shortlist when it has an index. Off by default."""
    return os.environ.get("CB_HEAD_SHORTLIST") == "1"


def kmeans(rows: np.ndarray, n: int, seed: int = 0, iters: int = 10, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """(centroids (n, d), assignment (V,)): Lloyd's k-means from n rows drawn by `seed`; deterministic. An emptied
    cluster takes the row farthest from its centroid, so every cluster keeps at least one row."""
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    rng = np.random.default_rng(seed)
    cent = rows[np.sort(rng.choice(len(rows), n, replace=False))].copy()
    assign = np.zeros(len(rows), dtype=np.int64)
    for _ in range(iters):
        c2 = (cent * cent).sum(1)
        best = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), chunk):
            s = rows[i:i + chunk] @ cent.T * 2.0 - c2          # argmax of 2 x.c - |c|^2 = argmin of |x - c|^2
            assign[i:i + chunk] = s.argmax(1)
            best[i:i + chunk] = s.max(1)
        sums = np.zeros_like(cent)
        np.add.at(sums, assign, rows)
        counts = np.bincount(assign, minlength=n)
        far = np.argsort(best, kind="stable")                   # worst-fit rows first, for emptied clusters
        k = 0
        for c in np.flatnonzero(counts == 0):
            r = far[k]
            k += 1
            counts[assign[r]] -= 1
            sums[assign[r]] -= rows[r]
            assign[r], sums[c], counts[c] = c, rows[r], 1
        cent = sums / counts[:, None]
    return cent.astype(np.float32), assign


class ShortlistIndex:
    """Clusters of a head's rows, an optional router and successor table; ``shortlist`` picks the rows to score."""

    def __init__(self, centroids: np.ndarray, assign: np.ndarray, router: tuple | None = None,
                 succ: np.ndarray | None = None) -> None:
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.assign = np.asarray(assign, dtype=np.int64)
        self.vocab, self.n = len(self.assign), len(self.centroids)
        order = np.argsort(self.assign, kind="stable")
        self.members = order                                    # token ids grouped by cluster, ascending within one
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(self.assign, minlength=self.n))])
        self.router = router                                    # (W (n, d), b (n,)) or None
        self.succ = succ                                        # (V, s) successor ids, -1 padded, or None

    @classmethod
    def build(cls, rows: np.ndarray, n_clusters: int, seed: int = 0, iters: int = 10) -> "ShortlistIndex":
        cent, assign = kmeans(rows, n_clusters, seed, iters)
        return cls(cent, assign)

    def cluster_scores(self, h: np.ndarray, arm: str) -> np.ndarray:
        h = np.asarray(h, dtype=np.float32).reshape(-1, self.centroids.shape[1])
        if arm == "clusters" or self.router is None:
            if arm != "clusters":
                raise ValueError(f"arm {arm!r} needs a router")
            return h @ self.centroids.T
        w, b = self.router
        return h @ w.T + b

    def shortlist(self, h: np.ndarray, k: int, arm: str = "clusters", prev: np.ndarray | None = None) -> list:
        """[token ids (sorted)] per query: whole clusters in score order until >= k rows; (c) adds successors."""
        if k >= self.vocab:
            return [np.arange(self.vocab)] * len(np.asarray(h).reshape(-1, self.centroids.shape[1]))
        scores = self.cluster_scores(h, arm)
        sizes = np.diff(self.offsets)
        out = []
        for i, row in enumerate(scores):
            rank = np.argsort(-row, kind="stable")
            upto = int(np.searchsorted(np.cumsum(sizes[rank]), k)) + 1
            ids = np.concatenate([self.members[self.offsets[c]:self.offsets[c + 1]] for c in rank[:upto]])
            if arm == "router+succ" and self.succ is not None and prev is not None:
                s = self.succ[int(np.asarray(prev).reshape(-1)[i])]
                ids = np.concatenate([ids, s[s >= 0]])
            out.append(np.unique(ids))
        return out

    def masked_logits(self, q, codes, temperature: float = 1.0, k: int = 1024, arm: str = "clusters", prev=None):
        """torch: (..., V) logits with only the shortlist's rows scored (exactly, with `codes`), the rest -inf."""
        import torch
        flat = q.reshape(-1, q.shape[-1])
        lists = self.shortlist(flat.detach().float().cpu().numpy(), k, arm,
                               None if prev is None else prev.reshape(-1).cpu().numpy())
        out = torch.full((flat.shape[0], self.vocab), float("-inf"), dtype=flat.dtype, device=flat.device)
        for i, ids in enumerate(lists):
            t = torch.as_tensor(ids, device=flat.device)
            out[i, t] = (flat[i:i + 1] @ codes.index_select(0, t).t())[0] * temperature
        return out.reshape(*q.shape[:-1], self.vocab)

    def save(self, path: str) -> None:
        extra = {}
        if self.router is not None:
            extra["router_w"], extra["router_b"] = self.router
        if self.succ is not None:
            extra["succ"] = self.succ
        np.savez(path, centroids=self.centroids, assign=self.assign, **extra)

    @classmethod
    def load(cls, path: str) -> "ShortlistIndex":
        d = np.load(path)
        router = (d["router_w"], d["router_b"]) if "router_w" in d else None
        return cls(d["centroids"], d["assign"], router, d["succ"] if "succ" in d else None)


def successors(tokens: np.ndarray, vocab: int, top: int = 32) -> np.ndarray:
    """(V, top) each token's most frequent next tokens in `tokens` (ties by id), -1 where fewer were seen."""
    t = np.asarray(tokens, dtype=np.int64)
    pair, cnt = np.unique(t[:-1] * vocab + t[1:], return_counts=True)
    a, b = pair // vocab, pair % vocab
    order = np.lexsort((b, -cnt, a))                            # by token, then count descending, then id
    a, b = a[order], b[order]
    out = np.full((vocab, top), -1, dtype=np.int32)
    starts = np.searchsorted(a, np.arange(vocab))
    ends = np.searchsorted(a, np.arange(vocab), side="right")
    for v in np.flatnonzero(ends > starts):
        s = b[starts[v]:min(ends[v], starts[v] + top)]
        out[v, :len(s)] = s
    return out


__all__ = ["ARMS", "ShortlistIndex", "kmeans", "on", "successors"]
