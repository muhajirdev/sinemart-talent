import numpy as np

from sinemart.bt import (
    LinearHead,
    Looks,
    MLPHead,
    agreement,
    compare_heads,
    next_pairs,
    rater_ceiling,
    sigmoid,
)


def synthetic(n=600, d=24, n_pairs=4000, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    w_true = rng.normal(size=d) * 3
    true = emb @ w_true
    a, b = rng.integers(n, size=(2, n_pairs))
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = rng.random(len(a)) < sigmoid(true[a] - true[b])
    winners = np.where(a_wins, a, b)
    losers = np.where(a_wins, b, a)
    return emb, winners, losers, true


def test_linear_head_recovers_true_order():
    emb, w, l, true = synthetic()
    head = LinearHead(l2=1.0).fit(emb, w, l)
    s = head.score(emb)
    assert np.corrcoef(s, true)[0, 1] > 0.9
    # On noiseless "true" comparisons the fitted score agrees almost always.
    rng = np.random.default_rng(1)
    a, b = rng.integers(len(emb), size=(2, 2000))
    hi, lo = np.where(true[a] > true[b], a, b), np.where(true[a] > true[b], b, a)
    assert agreement(s, hi, lo) > 0.85
    std = head.score_std(emb)
    assert std.shape == (len(emb),) and (std > 0).all()


def test_all_three_heads_beat_chance_on_unseen_faces():
    emb, w, l, _ = synthetic()
    looks = Looks.fit(emb, k=6)
    report, best = compare_heads(emb, w, l, looks)
    assert set(report) == {"A_linear", "B_look_centred", "C_mlp"}
    for r in report.values():
        assert r["heldout"] > 0.65, r
        assert r["n_test"] > 0 and r["n_train"] > 0
        assert "cross_look" in r["per_look"]
    assert best in report


def test_mlp_head_learns_nonlinear_taste():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 8))
    true = -np.abs(emb[:, 0]) * 4  # "closer to the middle is better": no straight arrow
    a, b = rng.integers(500, size=(2, 5000))
    keep = a != b
    a, b = a[keep], b[keep]
    w = np.where(true[a] > true[b], a, b)
    l = np.where(true[a] > true[b], b, a)
    lin = LinearHead().fit(emb, w, l).score(emb)
    mlp = MLPHead(steps=600).fit(emb, w, l).score(emb)
    assert agreement(mlp, w, l) > agreement(lin, w, l) + 0.1


def test_look_centring_removes_cluster_offset():
    rng = np.random.default_rng(0)
    emb = np.concatenate([rng.normal(0, 0.1, (50, 4)) + 3, rng.normal(0, 0.1, (50, 4)) - 3])
    looks = Looks.fit(emb, k=2)
    assert len(set(looks.labels[:50])) == 1 and looks.labels[0] != looks.labels[60]
    head = LinearHead(looks=looks)
    f = head.features(emb)
    for k in range(2):
        assert np.allclose(f[looks.labels == k].mean(0), 0, atol=1e-9)


def test_pair_mix_and_shapes():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=2000)
    std = rng.random(2000)
    pairs, kinds = next_pairs(scores, std, 1000, top_k=100, seed=0)
    assert len(pairs) == 1000
    assert kinds.count("uncertain") == 700 and kinds.count("top") == 200 and kinds.count("random") == 100
    assert all(a != b for a, b in pairs)
    top = set(np.argsort(scores)[-100:].tolist())
    for (a, b), k in zip(pairs, kinds):
        if k == "top":
            assert a in top and b in top
        if k == "uncertain":
            assert abs(np.argsort(np.argsort(scores))[a] - np.argsort(np.argsort(scores))[b]) <= 5


def test_rater_ceiling_counts_repeats_only():
    w = np.array([1, 1, 2, 3])
    l = np.array([2, 2, 1, 4])  # pair (1,2) asked 3 times: 2 vs 1
    acc, n = rater_ceiling(w, l)
    assert n == 3 and abs(acc - 2 / 3) < 1e-9


def test_band_check_shows_peak_for_inverted_u():
    from sinemart.bt import band_check

    rng = np.random.default_rng(0)
    arrow = rng.normal(size=3000)
    true = -((arrow - 1.0) ** 2)  # best around +1, worse beyond it
    a, b = rng.integers(3000, size=(2, 20000))
    keep = a != b
    a, b = a[keep], b[keep]
    w = np.where(true[a] > true[b], a, b)
    l = np.where(true[a] > true[b], b, a)
    rows, peak = band_check(arrow, w, l)
    assert len(rows) == 10 and sum(r["games"] for r in rows) == 2 * len(w)
    assert 6 <= peak <= 9  # peak before the top band
    assert rows[9]["win_rate"] < rows[peak - 1]["win_rate"] - 0.05


def test_head_choice_is_best_heldout_with_ties_to_simpler():
    from sinemart.bt import choose_head

    r = lambda x: {"heldout": x}  # noqa: E731
    assert choose_head({"A_linear": r(0.70), "B_look_centred": r(0.70), "C_mlp": r(0.71)}) == "C_mlp"
    assert choose_head({"A_linear": r(0.70), "B_look_centred": r(0.70), "C_mlp": r(0.70)}) == "A_linear"


def test_train_cli_end_to_end(tmp_path):
    import csv
    import json

    from sinemart.train import run

    emb, w, l, _ = synthetic(n=300, n_pairs=1500)
    ids = np.arange(1000, 1300)
    np.savez(tmp_path / "e.npz", ids=ids, emb=emb.astype(np.float32), model="test")
    with open(tmp_path / "picks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["winner_id", "loser_id"])
        wr.writerows(zip(ids[w], ids[l]))
    report = run(tmp_path / "e.npz", tmp_path / "picks.csv", tmp_path / "out", n_pairs=100, looks_k=4)
    assert report["best"] in ("A_linear", "B_look_centred", "C_mlp")
    assert len(report["bands"]) == 10
    assert json.loads((tmp_path / "out/report.json").read_text())["n_picks"] == len(w)
    sql = (tmp_path / "out/update.sql").read_text()
    assert sql.count("INSERT INTO pair_queue") == 100 and sql.count("UPDATE applicants") == 300
    # Cold start: no picks at all still produces a queue.
    (tmp_path / "none.csv").write_text("winner_id,loser_id\n")
    assert run(tmp_path / "e.npz", tmp_path / "none.csv", tmp_path / "cold", n_pairs=10)["best"] == "A_linear"
