"""Load a second embedding as a comparison model ("Model 2") next to the main one.

    python -m sinemart.alt KEY EMB.npz gender.csv OUT.sql

KEY names the model in the app (e.g. "m2"). The live app trains it on the same
picks as the main model and the scores page can switch to it; battles keep
using the main model. Applicants must already be loaded by sinemart.publish.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

from .publish import taste_space


def main(argv: list[str]) -> None:
    if len(argv) != 5:
        sys.exit("usage: python -m sinemart.alt KEY EMB.npz gender.csv OUT.sql")
    key, npz, gender_csv, out = argv[1:]
    data = np.load(npz)
    ids, emb = data["ids"], data["emb"].astype(np.float64)
    with open(gender_csv) as f:
        gender = {int(r["applicant_id"]): r["gender"] for r in csv.DictReader(f)}
    groups = np.array([gender.get(int(a), "") for a in ids], dtype=object)
    z, _, peaks, peak_group = taste_space(emb, groups=groups)
    rnd = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    with open(out, "w") as sql:
        sql.write(f"DELETE FROM alt_z WHERE key = '{key}';\n")
        for i, aid in enumerate(ids.tolist()):
            sql.write(f"INSERT INTO alt_z (key, applicant_id, z) VALUES ('{key}', {aid}, '{json.dumps(rnd(z[i]))}');\n")
        init = json.dumps({"peaks": [rnd(p) for p in peaks], "peak_gender": peak_group})
        sql.write(
            f"INSERT OR REPLACE INTO alt_models (key, init, n_picks) VALUES ('{key}', '{init}', 0);\n"
        )
    print(f"{len(ids)} applicants in {key}, {len(peaks)} looks")


if __name__ == "__main__":
    main(sys.argv)
