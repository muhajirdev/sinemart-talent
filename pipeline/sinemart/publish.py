"""Prepare crops + embeddings for Cloudflare.

    python -m sinemart.publish out/siglip2.npz crops/quality.csv out/

Writes:
    out/seed.sql          one applicants row per passed crop (D1)
    out/vectors.ndjson    embeddings for `wrangler vectorize insert`

Upload the crops themselves with rclone to R2 (see README); the Worker serves
them from ``crops/<applicant_id>.jpg``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        sys.exit("usage: python -m sinemart.publish EMB.npz quality.csv OUT_DIR")
    data = np.load(argv[1])
    ids, emb = data["ids"], data["emb"]
    with open(argv[2]) as f:
        quality = {int(r["applicant_id"]): r for r in csv.DictReader(f)}
    out = Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "seed.sql", "w") as sql, open(out / "vectors.ndjson", "w") as vec:
        for aid, e in zip(ids.tolist(), emb):
            q = quality.get(aid)
            if not q or q["ok"] != "1":
                continue
            stage = q["stage"].replace("'", "''")
            sql.write(
                f"INSERT OR REPLACE INTO applicants (id, image_key, stage) VALUES ({aid}, 'crops/{aid}.jpg', '{stage}');\n"
            )
            vec.write(json.dumps({"id": str(aid), "values": [round(float(v), 6) for v in e]}) + "\n")
            n += 1
    print(f"{n} applicants ready ({len(ids) - n} skipped by the quality gate)")


if __name__ == "__main__":
    main(sys.argv)
