"""cubbyllm/model/vocab/shortlist.py -- a shortlist index over a trained output head's rows (order 0050, stage 1).

The 128k head scores every row at every token. A shortlist scores only K rows, exactly, with the trained rows, and
softmaxes over those (``CB_HEAD_SHORTLIST=1`` with an index attached to the head; off by default, and off the head
is unchanged bit for bit). Built once, offline, deterministically from a seed:
  (a) clusters   k-means over the head's rows; a query probes the clusters whose centroids score best
  (b) router     a linear map from the trunk's final state to cluster scores, trained (``validation/
                 head_shortlist_eval.py``) on the frozen trunk's held-in states against the full head's top-k
  (c) + succ     (b), plus the previous token's most frequent successors (counted on held-in text)
  (d) screen     added by Claude Code (recorded in 0050's report), after "learning to screen" (Chen et al., 2018):
                 k-means over the trunk's held-in states; each context cluster keeps the tokens the full head ranks in
                 its top-k for that cluster's states, by their summed probability. A query takes its nearest context
                 cluster's list (then the next nearest) up to K. "screen+succ" adds (c)'s successors.
The clusters are spherical (rows centred on their mean, compared by direction): in 1,024 dimensions plain Euclidean
k-means put half the rows in clusters of one (``head_shortlist_<date>.json``, the first read).
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

ARMS = ("clusters", "router", "router+succ", "screen", "screen+succ")


def on() -> bool:
    """``CB_HEAD_SHORTLIST=1``: the head scores a shortlist when it has an index. Off by default."""
    return os.environ.get("CB_HEAD_SHORTLIST") == "1"


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def kmeans(rows: np.ndarray, n: int, seed: int = 0, iters: int = 10, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """(centroids (n, d), assignment (V,)): spherical k-means (rows centred, compared by direction) from n rows drawn
    by `seed`; deterministic. The centroids returned are the members' raw means (what a query is scored against). An
    emptied cluster takes the row worst fitted to its own, so every cluster keeps at least one row."""
    raw = np.ascontiguousarray(rows, dtype=np.float32)
    rows = _unit(raw - raw.mean(0))
    rng = np.random.default_rng(seed)
    cent = rows[np.sort(rng.choice(len(rows), n, replace=False))].copy()
    assign = np.zeros(len(rows), dtype=np.int64)
    for _ in range(iters):
        best = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), chunk):
            s = rows[i:i + chunk] @ cent.T
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
        cent = _unit(sums / counts[:, None])
    raw_sums = np.zeros((n, raw.shape[1]), dtype=np.float64)
    np.add.at(raw_sums, assign, raw)
    return (raw_sums / np.bincount(assign, minlength=n)[:, None]).astype(np.float32), assign


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
        if arm.startswith("screen"):
            return self.screen.shortlist(h, k, self.succ if arm.endswith("succ") else None, prev)
        return self._clusters(h, k, arm, prev)

    screen = None             # a ScreenIndex (arm d), attached by the builder

    def _clusters(self, h: np.ndarray, k: int, arm: str, prev: np.ndarray | None) -> list:
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
        extra = {} if self.screen is None else {"screen_c": self.screen.centroids, "screen_ids": self.screen.ids,
                                                "screen_off": self.screen.offsets}
        if self.router is not None:
            extra["router_w"], extra["router_b"] = self.router
        if self.succ is not None:
            extra["succ"] = self.succ
        np.savez(path, centroids=self.centroids, assign=self.assign, **extra)

    @classmethod
    def load(cls, path: str) -> "ShortlistIndex":
        d = np.load(path)
        router = (d["router_w"], d["router_b"]) if "router_w" in d else None
        idx = cls(d["centroids"], d["assign"], router, d["succ"] if "succ" in d else None)
        if "screen_c" in d:
            idx.screen = ScreenIndex(d["screen_c"], d["screen_ids"], d["screen_off"], len(idx.assign))
        return idx


class ScreenIndex:
    """Arm (d): context clusters of the trunk's states, each with the tokens the full head ranks for it (CSR)."""

    def __init__(self, centroids: np.ndarray, ids: np.ndarray, offsets: np.ndarray, vocab: int) -> None:
        self.centroids, self.ids, self.offsets, self.vocab = centroids, ids, offsets, vocab

    @classmethod
    def build(cls, hidden: np.ndarray, top: np.ndarray, weight: np.ndarray, vocab: int, n: int, seed: int = 0,
              iters: int = 10, keep: int = 8192) -> "ScreenIndex":
        """`hidden` (N, d) held-in states; `top` (N, t) the full head's top ids there; `weight` (N, t) their
        probabilities. Each cluster's list: its tokens by summed probability (ties by id), at most `keep`."""
        cent, assign = kmeans(hidden, n, seed, iters)
        lists = []
        for c in range(n):
            sel = assign == c
            tok, w = top[sel].reshape(-1), weight[sel].reshape(-1)
            sums = np.bincount(tok, weights=w, minlength=vocab)
            nz = np.flatnonzero(sums)
            lists.append(nz[np.lexsort((nz, -sums[nz]))][:keep])
        off = np.concatenate([[0], np.cumsum([len(x) for x in lists])])
        return cls(cent, np.concatenate(lists).astype(np.int64), off, vocab)

    def shortlist(self, h: np.ndarray, k: int, succ=None, prev=None) -> list:
        h = np.asarray(h, dtype=np.float32).reshape(-1, self.centroids.shape[1])
        rank = np.argsort(-(h @ self.centroids.T), axis=1, kind="stable")
        out = []
        for i, row in enumerate(rank):
            got, seen = [], set()
            for c in row:                                   # nearest clusters first, until k distinct tokens
                lst = [t for t in self.ids[self.offsets[c]:self.offsets[c + 1]].tolist() if t not in seen]
                got.extend(lst[:k - len(got)])
                seen.update(lst)
                if len(got) >= k:
                    break
            ids = np.asarray(got, dtype=np.int64)
            if succ is not None and prev is not None:
                s = succ[int(np.asarray(prev).reshape(-1)[i])]
                ids = np.concatenate([ids, s[s >= 0]])
            out.append(np.unique(ids))
        return out


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


__all__ = ["ARMS", "ScreenIndex", "ShortlistIndex", "kmeans", "on", "successors"]
