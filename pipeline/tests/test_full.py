import json

import numpy as np

from sinemart.full import TEST_IDS, norm_of, sql_rows, vectors


def test_test_batches_never_move_the_space():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 12))
    ids = np.arange(60)
    ids[-10:] += TEST_IDS
    norm = norm_of(ids, emb)
    base = norm_of(ids[:50], emb[:50])
    assert np.allclose(norm["mean"], base["mean"]) and np.isclose(norm["scale"], base["scale"])
    x = vectors(emb[:50], norm)
    assert np.allclose(x.mean(0), 0)
    # Spread along the main direction is 1 after scaling.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    assert np.isclose((x @ vt[0]).std(), 1)


def test_sql_rows_round_trip():
    x = np.array([[0.123456789, -1.0], [2.0, 0.5]])
    rows = sql_rows("m6", np.array([7, 260001]), x)
    assert rows[0].startswith("INSERT OR REPLACE INTO full_z (key, applicant_id, z) VALUES ('m6', 7, ")
    z = json.loads(rows[0].split(", '", 1)[1].rsplit("');", 1)[0])
    assert z == [0.12346, -1.0]
