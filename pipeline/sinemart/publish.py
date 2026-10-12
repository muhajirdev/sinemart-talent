"""Prepare crops + embeddings for Cloudflare.

    python -m sinemart.publish out/siglip2.npz crops/quality.csv out/

Writes:
    out/seed.sql          applicants rows for passed crops, each with its
                          taste-space coordinates and look (D1)
    out/vectors.ndjson    full embeddings for `wrangler vectorize insert`

The Worker trains the hills model live on the team's picks. It doesn't need
the 768-dim embeddings for that: it works in a small "taste space" (the top
principal directions, the same start the Python HillsHead uses) and learns a
map inside it plus one peak per look. seed.sql also stores that starting
point (look centres) in the `model` table.

Upload the crops themselves with rclone to R2 (see README); the Worker serves
them from ``crops/<applicant_id>.jpg``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from .bt import HillsHead, Looks


def taste_space(emb: np.ndarray, looks_k: int = 12, rank: int = 16, seed: int = 0):
    """(z, look, peaks): coordinates in taste space, look per face, look centres."""
    k = max(1, min(looks_k, len(emb) // 3))
    looks = Looks.fit(emb, k=k, seed=seed)
    head = HillsHead(looks, rank=rank)
    head._init(emb)
    z = (emb - head.mu) @ head.P.T
    return z, looks.labels, head.q


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        sys.exit("usage: python -m sinemart.publish EMB.npz quality.csv OUT_DIR")
    data = np.load(argv[1])
    with open(argv[2]) as f:
        quality = {int(r["applicant_id"]): r for r in csv.DictReader(f)}
    keep = np.array([quality.get(int(a), {}).get("ok") == "1" for a in data["ids"]])
    ids, emb = data["ids"][keep], data["emb"][keep].astype(np.float64)
    z, look, peaks = taste_space(emb)
    out = Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rnd = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    with open(out / "seed.sql", "w") as sql, open(out / "vectors.ndjson", "w") as vec:
        for i, aid in enumerate(ids.tolist()):
            stage = quality[aid]["stage"].replace("'", "''")
            sql.write(
                "INSERT OR REPLACE INTO applicants (id, image_key, stage, look, z) VALUES "
                f"({aid}, 'crops/{aid}.jpg', '{stage}', {int(look[i])}, '{json.dumps(rnd(z[i]))}');\n"
            )
            vec.write(json.dumps({"id": str(aid), "values": rnd(emb[i])}) + "\n")
        init = json.dumps({"peaks": [rnd(p) for p in peaks], "model": str(data["model"])})
        sql.write(f"INSERT OR REPLACE INTO model (id, init) VALUES (1, '{init}');\n")
    print(f"{len(ids)} applicants ready ({(~keep).sum()} skipped by the quality gate), {len(peaks)} looks")


if __name__ == "__main__":
    main(sys.argv)
