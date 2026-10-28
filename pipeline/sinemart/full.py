"""Full-size vectors for the ranker (D1 table ``full_z``).

Since 2026-10-28 the ranker's line is fitted on all of a model's numbers (768
for Model 6), not the 16-number taste space: offline that was 72.4% vs 69.7%
held-out, and 74.4% with the per-face correction (worker/src/full.ts).

Each person's vector (embed.py output, one L2-normalised mean per person) is
centred on the applicants' average and divided by the spread along the main
direction, so the Worker's fixed l2 means the same thing after a reload. Test
batches (or any later addition) reuse the same centre and scale (``--norm``),
so they land in the same space without moving anyone else.

    python -m sinemart.full EMB.npz m6 OUT_DIR              # all applicants
    python -m sinemart.full BATCH.npz m6 OUT_DIR --norm OUT_DIR/full_norm_m6.json

Writes ``OUT_DIR/full_<key>.sql`` (one INSERT per line, ready for the
~60-line ``--command`` chunks) and ``OUT_DIR/full_norm_<key>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Applicant ids from here up are test batches (e.g. "bc" 900001+): they never
# set the centre or scale.
TEST_IDS = 900_000


def norm_of(ids: np.ndarray, emb: np.ndarray) -> dict:
    keep = emb[ids < TEST_IDS]
    mean = keep.mean(0)
    _, _, vt = np.linalg.svd(keep - mean, full_matrices=False)
    scale = float(((keep - mean) @ vt[0]).std())
    return {"mean": mean.tolist(), "scale": scale}


def vectors(emb: np.ndarray, norm: dict) -> np.ndarray:
    return (emb - np.asarray(norm["mean"])) / norm["scale"]


def sql_rows(key: str, ids: np.ndarray, x: np.ndarray, digits: int = 5) -> list[str]:
    return [
        f"INSERT OR REPLACE INTO full_z (key, applicant_id, z) VALUES ('{key}', {int(i)}, "
        f"'{json.dumps([round(float(v), digits) for v in row], separators=(',', ':'))}');"
        for i, row in zip(ids, x)
    ]


def main(argv: list[str]) -> None:
    if len(argv) not in (4, 6) or (len(argv) == 6 and argv[4] != "--norm"):
        sys.exit("usage: python -m sinemart.full EMB.npz KEY OUT_DIR [--norm NORM.json]")
    d = np.load(argv[1])
    ids, emb, key, out = d["ids"], d["emb"].astype(np.float64), argv[2], Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    norm = json.loads(Path(argv[5]).read_text()) if len(argv) == 6 else norm_of(ids, emb)
    if len(argv) == 4:
        (out / f"full_norm_{key}.json").write_text(json.dumps(norm))
    rows = sql_rows(key, ids, vectors(emb, norm))
    (out / f"full_{key}.sql").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {out / f'full_{key}.sql'} (scale {norm['scale']:.4f})")


if __name__ == "__main__":
    main(sys.argv)
