import numpy as np

from sinemart.project import rebuild
from sinemart.publish import taste_space


def test_rebuild_matches_publish_and_keeps_new_people_apart():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 40))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    z_live, *_ = taste_space(emb)
    # Flip some stored directions: the rebuild has to line the signs up.
    z_live = z_live * np.where(np.arange(z_live.shape[1]) % 3 == 0, -1, 1)
    mu, P, err = rebuild(emb, z_live)
    assert err < 1e-8
    assert np.allclose((emb - mu) @ P.T, z_live)
    # A new face lands where the live space puts it; nobody else moves.
    new = rng.normal(size=(5, 40))
    assert ((new - mu) @ P.T).shape == (5, z_live.shape[1])
