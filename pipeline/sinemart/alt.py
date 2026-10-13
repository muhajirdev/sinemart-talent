"""Load a second embedding as a comparison model ("Model 2") next to the main one.

    python -m sinemart.alt KEY EMB.npz gender.csv OUT.sql [FIXED.csv COLUMN]

KEY names the model in the app (e.g. "m2"). The live app trains it on the same
picks as the main model and the scores page can switch to it; battles keep
using the main model. Applicants must already be loaded by sinemart.publish.

People the embedding has no vector for (e.g. no frontal photo for the face
geometry) get the average of their group, so they sit mid-table. FIXED.csv,
if given, is a score that is never trained (e.g. geometry_score from
sinemart.geometry); the app reports how often it agrees with the same
held-out picks.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

from .publish import taste_space


def main(argv: list[str]) -> None:
    if len(argv) not in (5, 7):
        sys.exit("usage: python -m sinemart.alt KEY EMB.npz gender.csv OUT.sql [FIXED.csv COLUMN]")
    key, npz, gender_csv, out = argv[1:5]
    data = np.load(npz)
    with open(gender_csv) as f:
        gender = {int(r["applicant_id"]): r["gender"] for r in csv.DictReader(f)}
    ids, emb, filled = fill_missing(data["ids"].astype(np.int64), data["emb"].astype(np.float64), gender)
    groups = np.array([gender.get(int(a), "") for a in ids], dtype=object)
    fixed = None
    if len(argv) == 7:
        with open(argv[5]) as f:
            fixed = {int(r["applicant_id"]): round(float(r[argv[6]]), 5) for r in csv.DictReader(f) if r[argv[6]]}
    z, _, peaks, peak_group = taste_space(emb, groups=groups)
    rnd = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    with open(out, "w") as sql:
        sql.write(f"DELETE FROM alt_z WHERE key = '{key}';\n")
        for i, aid in enumerate(ids.tolist()):
            sql.write(f"INSERT INTO alt_z (key, applicant_id, z) VALUES ('{key}', {aid}, '{json.dumps(rnd(z[i]))}');\n")
        init = json.dumps({"peaks": [rnd(p) for p in peaks], "peak_gender": peak_group})
        fx = "NULL" if fixed is None else f"'{json.dumps(fixed)}'"
        sql.write(
            f"INSERT OR REPLACE INTO alt_models (key, init, fixed, n_picks) VALUES ('{key}', '{init}', {fx}, 0);\n"
        )
    print(f"{len(ids)} applicants in {key} ({filled} filled with their group average), {len(peaks)} looks")


def fill_missing(ids: np.ndarray, emb: np.ndarray, gender: dict[int, str]):
    """Give everyone in gender.csv a vector: missing people get their group's mean."""
    have = {int(a) for a in ids}
    missing = sorted(a for a in gender if a not in have)
    if not missing:
        return ids, emb, 0
    g = np.array([gender.get(int(a), "") for a in ids], dtype=object)
    extra = [emb[g == gender[a]].mean(axis=0) if (g == gender[a]).any() else emb.mean(axis=0) for a in missing]
    return np.concatenate([ids, np.array(missing, np.int64)]), np.vstack([emb, extra]), len(missing)


if __name__ == "__main__":
    main(sys.argv)
