"""Bradley-Terry ranking on frozen embeddings: the hills model, baselines,
evaluation, and pair selection.

A pick "a beat b" says P(a > b) = sigmoid(s(a) - s(b)).

The ranker is the hills model. Beauty is a landscape with several hills, one
per look: each hill has a peak (the ideal face of that look) and a height,
and a face scores by how close it sits to a peak, so going past the peak
(towards caricature) scores lower again:

    z      = P e                                 project to a small "taste space"
    t_k(e) = h_k - |z - q_k|^2 / 2               height on hill k
    s(e)   = log sum_k exp t_k(e)                ~ height on the hill it's on

Peaks q_k start at the k-means look centres and are then learned from picks,
so a peak can move away from its look's average face (attractive faces are
not always average). By default every peak is equally tall (h_k = 0): the
ideal face of any look ranks as high as the ideal face of any other.
Three simpler heads are fitted on the same picks only as comparisons in the
held-out report:

    A  linear:          s(e) = w . e
    B  look-centred:    s(e) = w . (e - centroid[look(e)])
    C  mlp:             s(e) = v . tanh(W e + b)

Each face's score std (for choosing informative pairs) comes from a small
bootstrap ensemble of hills models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * x))


# --------------------------------------------------------------------------- looks


@dataclass
class Looks:
    centroids: np.ndarray  # k x d
    labels: np.ndarray  # n

    @classmethod
    def fit(cls, emb: np.ndarray, k: int = 12, iters: int = 50, seed: int = 0) -> "Looks":
        """Plain k-means (k-means++ init) over all applicants, labelled or not."""
        rng = np.random.default_rng(seed)
        n = len(emb)
        k = min(k, n)
        centroids = [emb[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((emb[:, None, :] - np.asarray(centroids)[None]) ** 2).sum(-1), axis=1)
            centroids.append(emb[rng.choice(n, p=d2 / d2.sum()) if d2.sum() > 0 else rng.integers(n)])
        c = np.asarray(centroids, np.float64)
        labels = np.zeros(n, np.int64)
        for _ in range(iters):
            d = (emb**2).sum(1)[:, None] - 2 * emb @ c.T + (c**2).sum(1)[None]
            new = d.argmin(1)
            for j in range(k):
                if (new == j).any():
                    c[j] = emb[new == j].mean(0)
            if (new == labels).all():
                break
            labels = new
        return cls(c, labels)


# --------------------------------------------------------------------------- heads


class LinearHead:
    """Head A (looks=None) or head B (looks given): L2-regularised BT, Newton fit."""

    def __init__(self, l2: float = 1.0, looks: Looks | None = None):
        self.l2 = l2
        self.looks = looks
        self.name = "B_look_centred" if looks is not None else "A_linear"

    def features(self, emb: np.ndarray) -> np.ndarray:
        if self.looks is None:
            return emb
        return emb - self.looks.centroids[self.looks.labels]

    def fit(self, emb: np.ndarray, winners: np.ndarray, losers: np.ndarray, iters: int = 25):
        f = self.features(emb)
        x = f[winners] - f[losers]
        d = x.shape[1]
        w = np.zeros(d)
        for _ in range(iters):
            p = sigmoid(x @ w)
            grad = -x.T @ (1 - p) + self.l2 * w
            hess = (x * (p * (1 - p))[:, None]).T @ x + self.l2 * np.eye(d)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        p = sigmoid(x @ w)
        hess = (x * (p * (1 - p))[:, None]).T @ x + self.l2 * np.eye(d)
        self.w = w
        self.cov = np.linalg.inv(hess)
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        return self.features(emb) @ self.w

    def score_std(self, emb: np.ndarray) -> np.ndarray:
        f = self.features(emb)
        return np.sqrt(np.maximum(np.einsum("nd,de,ne->n", f, self.cov, f), 0))


class MLPHead:
    """Head C: one hidden tanh layer, pairwise logistic loss, full-batch Adam."""

    name = "C_mlp"

    def __init__(self, hidden: int = 64, l2: float = 1e-3, steps: int = 400, lr: float = 1e-2, seed: int = 0):
        self.hidden, self.l2, self.steps, self.lr, self.seed = hidden, l2, steps, lr, seed

    def _forward(self, e):
        h = np.tanh(e @ self.W.T + self.b)
        return h @ self.v, h

    def fit(self, emb: np.ndarray, winners: np.ndarray, losers: np.ndarray):
        rng = np.random.default_rng(self.seed)
        d = emb.shape[1]
        self.W = rng.normal(0, 1 / np.sqrt(d), (self.hidden, d))
        self.b = np.zeros(self.hidden)
        self.v = rng.normal(0, 1 / np.sqrt(self.hidden), self.hidden)
        params = [self.W, self.b, self.v]
        m = [np.zeros_like(p) for p in params]
        s = [np.zeros_like(p) for p in params]
        ea, eb = emb[winners], emb[losers]
        n = len(winners)
        for t in range(1, self.steps + 1):
            fa, ha = self._forward(ea)
            fb, hb = self._forward(eb)
            g = -(1 - sigmoid(fa - fb)) / n  # dL/d(fa - fb)
            ga = (g[:, None] * self.v) * (1 - ha**2)  # dL/d(pre-activation a)
            gb = (g[:, None] * self.v) * (1 - hb**2)
            grads = [
                ga.T @ ea - gb.T @ eb + self.l2 * self.W,
                ga.sum(0) - gb.sum(0),
                g @ (ha - hb) + self.l2 * self.v,
            ]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                s[i] = 0.999 * s[i] + 0.001 * gr**2
                p -= self.lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(s[i] / (1 - 0.999**t)) + 1e-8)
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        return self._forward(emb)[0]


class HillsHead:
    """The ranker: one learned peak per look, score = height on the hills."""

    name = "D_hills"

    def __init__(
        self, looks: Looks, rank: int = 16, l2: float = 1e-3, steps: int = 500, lr: float = 2e-2, seed: int = 0,
        equal_peaks: bool = True,
    ):
        self.looks, self.rank, self.l2, self.steps, self.lr, self.seed = looks, rank, l2, steps, lr, seed
        # equal_peaks: every look's ideal face counts as equally beautiful, so
        # the best face of any look can reach the top. False lets picks make
        # some hills taller than others (and can copy a bias in the picks).
        self.equal_peaks = equal_peaks

    def _init(self, emb: np.ndarray):
        # Taste space starts as the top principal directions, scaled to unit
        # spread; peaks start at the look centres. Picks then move both.
        mu = emb.mean(0)
        _, sv, vt = np.linalg.svd(emb - mu, full_matrices=False)
        r = min(self.rank, int((sv > 1e-6 * sv[0]).sum()))  # skip empty directions
        self.mu = mu
        self.P = vt[:r] / (sv[:r, None] / np.sqrt(len(emb)) + 1e-8)
        self.q = (self.looks.centroids - mu) @ self.P.T
        self.h = np.zeros(len(self.q))

    def _terms(self, e: np.ndarray):
        z = (e - self.mu) @ self.P.T
        diff = z[:, None, :] - self.q[None]  # n x K x r
        t = self.h[None] - 0.5 * (diff**2).sum(-1)
        tmax = t.max(1, keepdims=True)
        w = np.exp(t - tmax)
        s = np.log(w.sum(1)) + tmax[:, 0]
        return s, w / w.sum(1, keepdims=True), diff

    def fit(self, emb: np.ndarray, winners: np.ndarray, losers: np.ndarray):
        self._init(emb)
        params = [self.P, self.q, self.h]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        ea, eb = emb[winners] - self.mu, emb[losers] - self.mu
        n = max(len(winners), 1)
        for t in range(1, self.steps + 1):
            sa, pa, da = self._terms(emb[winners])
            sb, pb, db = self._terms(emb[losers])
            g = -(1 - sigmoid(sa - sb)) / n  # dL/d(sa - sb)
            # ds/dh_k = pi_k ; ds/dq_k = pi_k (z - q_k) ; ds/dz = -sum_k pi_k (z - q_k)
            ga, gb = g[:, None], -g[:, None]
            dz_a = -(pa[..., None] * da).sum(1) * ga
            dz_b = -(pb[..., None] * db).sum(1) * gb
            grads = [
                dz_a.T @ ea + dz_b.T @ eb + self.l2 * self.P,
                (pa[..., None] * da * ga[..., None]).sum(0) + (pb[..., None] * db * gb[..., None]).sum(0),
                (pa * ga).sum(0) + (pb * gb).sum(0) + self.l2 * self.h,
            ]
            if self.equal_peaks:
                grads[2] = np.zeros_like(self.h)
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr**2
                p -= self.lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        return self._terms(emb)[0]

    def hill(self, emb: np.ndarray) -> np.ndarray:
        """Which hill each face is standing on."""
        return self._terms(emb)[1].argmax(1)


def hills_with_std(emb, winners, losers, looks: Looks, members: int = 5, seed: int = 0, equal_peaks: bool = True):
    """Fit the hills model on every pick, plus a bootstrap ensemble whose
    spread is each face's score std (high = the model can't place it yet)."""
    main = HillsHead(looks, seed=seed, equal_peaks=equal_peaks).fit(emb, winners, losers)
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(members):
        idx = rng.integers(len(winners), size=len(winners))
        s = HillsHead(looks, seed=seed + 1 + i, equal_peaks=equal_peaks).fit(emb, winners[idx], losers[idx]).score(emb)
        runs.append(s - s.mean())  # scores are only defined up to a shift
    return main, np.std(runs, axis=0)


# --------------------------------------------------------------------------- evaluation


def split_by_applicant(winners, losers, n_items: int, test_frac: float = 0.2, seed: int = 0):
    """Hold out applicants, not pairs: test = pairs touching a held-out face,
    train = pairs touching none. This is the real job (scoring unseen faces)."""
    rng = np.random.default_rng(seed)
    held = rng.random(n_items) < test_frac
    touch = held[winners] | held[losers]
    return ~touch, touch


def agreement(scores: np.ndarray, winners: np.ndarray, losers: np.ndarray) -> float:
    if len(winners) == 0:
        return float("nan")
    return float(np.mean(scores[winners] > scores[losers]))


def per_look_agreement(scores, winners, losers, labels) -> dict[str, tuple[float, int]]:
    """Agreement within each look (both faces in it) plus one cross-look bucket."""
    out = {}
    lw, ll = labels[winners], labels[losers]
    for k in np.unique(labels):
        sel = (lw == k) & (ll == k)
        if sel.any():
            out[f"look_{k}"] = (agreement(scores, winners[sel], losers[sel]), int(sel.sum()))
    cross = lw != ll
    if cross.any():
        out["cross_look"] = (agreement(scores, winners[cross], losers[cross]), int(cross.sum()))
    return out


def rater_ceiling(winners: np.ndarray, losers: np.ndarray) -> tuple[float, int]:
    """How often repeated showings of the same pair got the majority answer.
    No model can beat the team's own consistency; this is that ceiling."""
    votes: dict[tuple[int, int], list[int]] = {}
    for a, b in zip(winners.tolist(), losers.tolist()):
        key = (min(a, b), max(a, b))
        votes.setdefault(key, []).append(1 if a == key[0] else 0)
    agree = total = 0
    for v in votes.values():
        if len(v) > 1:
            ones = sum(v)
            agree += max(ones, len(v) - ones)
            total += len(v)
    return (agree / total if total else float("nan")), total


def compare_heads(emb, winners, losers, looks: Looks, test_frac=0.2, seed=0, l2=1.0):
    """Fit the hills model and the A/B/C baselines on the train split; report
    held-out and per-look agreement for each."""
    train, test = split_by_applicant(winners, losers, len(emb), test_frac, seed)
    heads = [HillsHead(looks, seed=seed), LinearHead(l2), LinearHead(l2, looks), MLPHead(seed=seed)]
    report = {}
    for h in heads:
        h.fit(emb, winners[train], losers[train])
        s = h.score(emb)
        report[h.name] = {
            "train": agreement(s, winners[train], losers[train]),
            "heldout": agreement(s, winners[test], losers[test]),
            "per_look": per_look_agreement(s, winners[test], losers[test], looks.labels),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        }
    return report


# --------------------------------------------------------------------------- pair selection


def next_pairs(scores, std, n: int, mix=(0.7, 0.2, 0.1), top_k: int = 500, near: int = 5, seed=None, hill=None):
    """Choose n pairs: mix[0] uncertain, mix[1] among the top_k, mix[2] random.

    Uncertain: draw a face weighted by its score std, pair it with one of the
    `near` faces closest in score, so the outcome is a coin flip the model
    can't yet call. Given `hill`, every other uncertain pair stays on the
    face's own hill, to pin down where that look's peak sits. Top: both faces from the current top_k, to sharpen the
    order where the shortlist is drawn. Random: keeps the whole range honest.
    """
    rng = np.random.default_rng(seed)
    m = len(scores)
    order = np.argsort(scores)
    rank = np.empty(m, np.int64)
    rank[order] = np.arange(m)
    n_unc = round(n * mix[0])
    n_top = round(n * mix[1])
    n_rand = n - n_unc - n_top
    pairs, kinds = [], []

    p = std / std.sum() if std.sum() > 0 else None
    for i, a in enumerate(rng.choice(m, n_unc, p=p)):
        if hill is not None and i % 2 == 0:
            mine = order[hill[order] == hill[a]]
            r = int(np.flatnonzero(mine == a)[0])
            cand = [x for x in mine[max(0, r - near) : r + near + 1] if x != a]
            if cand:
                pairs.append((int(a), int(rng.choice(cand))))
                kinds.append("same_hill")
                continue
        r = rank[a]
        lo, hi = max(0, r - near), min(m, r + near + 1)
        cand = [order[i] for i in range(lo, hi) if order[i] != a]
        pairs.append((int(a), int(rng.choice(cand))))
        kinds.append("uncertain")

    top = order[-min(top_k, m) :]
    for _ in range(n_top):
        a, b = rng.choice(top, 2, replace=False)
        pairs.append((int(a), int(b)))
        kinds.append("top")

    for _ in range(n_rand):
        a, b = rng.choice(m, 2, replace=False)
        pairs.append((int(a), int(b)))
        kinds.append("random")

    # Randomise left/right so position bias doesn't leak into the labels.
    flip = rng.random(len(pairs)) < 0.5
    pairs = [(b, a) if f else (a, b) for (a, b), f in zip(pairs, flip)]
    return pairs, kinds


def band_check(scores: np.ndarray, winners: np.ndarray, losers: np.ndarray, bands: int = 10):
    """Bin the faces that were judged into equal-count bands by straight-arrow
    (head A) score and return the team's win rate per band, lowest band first.

    If the arrow were the whole story, win rate would climb band by band. A
    peak followed by a drop says "more along the arrow" stops being better,
    which is the case for the curved head C.
    """
    seen = np.unique(np.concatenate([winners, losers]))
    edges = np.quantile(scores[seen], np.linspace(0, 1, bands + 1))
    band_of = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bands - 1)
    wins = np.bincount(band_of[winners], minlength=bands)
    games = wins + np.bincount(band_of[losers], minlength=bands)
    rows = []
    for k in range(bands):
        rate = wins[k] / games[k] if games[k] else float("nan")
        rows.append({"band": k + 1, "lo": float(edges[k]), "hi": float(edges[k + 1]), "win_rate": float(rate), "games": int(games[k])})
    rates = np.array([r["win_rate"] for r in rows])
    peak = int(np.nanargmax(rates)) + 1 if np.isfinite(rates).any() else None
    return rows, peak
