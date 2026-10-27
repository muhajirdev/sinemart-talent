"""Put new people into the live taste space without moving anyone already in it.

    python -m sinemart.project LIVE.npz NEW.npz LIVE_Z.json OUT.json

For testing the model on a batch that should not join the picks (e.g. the
"bc" showcase photos, 2026-10-27). Re-publishing would refit the taste space
and clear every score; this keeps the live one and only adds coordinates.

LIVE.npz   embeddings of the people in D1, re-made from their R2 crops with
           sinemart.embed (same model as the taste space being matched)
NEW.npz    embeddings of the new people, same model
LIVE_Z.json  [{"id": ..., "z": "[...]"}] as stored in D1 (applicants.z, or
           alt_z for a comparison model), e.g. from
           SELECT id, z FROM applicants   /   SELECT applicant_id AS id, z FROM alt_z WHERE key = 'm6'
OUT.json   {"max_abs_err": ..., "z": {id: [16 numbers]}}

The rebuild must match what is stored (max_abs_err ~0), or the new people
land in a slightly different space. A mismatch usually means the crops
embedded aren't the ones the live load used (e.g. old plain ``<id>.jpg``
crops next to ``<id>-closeup.jpg``).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .bt import HillsHead, Looks


def rebuild(emb: np.ndarray, z_live: np.ndarray, rank: int = 16):
    """(mu, P, max_abs_err): the taste space publish.py built from emb, with
    each direction's sign lined up with the stored z (SVD signs are arbitrary)."""
    head = HillsHead(Looks(np.zeros((1, emb.shape[1])), np.zeros(len(emb), int)), rank=rank)
    head._init(emb)
    z = (emb - head.mu) @ head.P.T
    sign = np.where((z * z_live).sum(0) < 0, -1.0, 1.0)
    P = head.P * sign[:, None]
    return head.mu, P, float(np.abs(z * sign - z_live).max())


def main(argv: list[str]) -> None:
    if len(argv) != 5:
        sys.exit("usage: python -m sinemart.project LIVE.npz NEW.npz LIVE_Z.json OUT.json")
    live, new = np.load(argv[1]), np.load(argv[2])
    stored = {int(r["id"]): json.loads(r["z"]) for r in json.load(open(argv[3]))}
    pos = {int(a): i for i, a in enumerate(live["ids"])}
    # Same people, same order as publish.py / alt.py used (sorted ids).
    ids = sorted(stored)
    missing = [a for a in ids if a not in pos]
    if missing:
        sys.exit(f"{len(missing)} people in D1 have no embedding (e.g. {missing[:5]}): embed their crops first")
    emb = live["emb"].astype(np.float64)[[pos[a] for a in ids]]
    mu, P, err = rebuild(emb, np.array([stored[a] for a in ids]))
    z = (new["emb"].astype(np.float64) - mu) @ P.T
    rnd = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    with open(argv[4], "w") as f:
        json.dump({"max_abs_err": err, "z": {int(a): rnd(v) for a, v in zip(new["ids"], z)}}, f)
    print(f"taste space rebuilt from {len(ids)} people (max |dz| {err:.5f}); {len(z)} new people projected")
    if err > 1e-3:
        print("warning: the rebuild doesn't match D1; check which crops were embedded")


if __name__ == "__main__":
    main(sys.argv)
