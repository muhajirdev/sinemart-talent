"""Prepare crops + embeddings for Cloudflare.

    python -m sinemart.publish out/siglip2.npz crops/quality.csv out/

Writes:
    out/seed.sql          applicants rows (one per person with a usable face),
                          each with its taste-space coordinates and look,
                          plus a photos row per photo the raters can flip
                          through (D1)
    out/vectors.ndjson    full embeddings for `wrangler vectorize insert`

The Worker trains the hills model live on the team's picks. It doesn't need
the 768-dim embeddings for that: it works in a small "taste space" (the top
principal directions, the same start the Python HillsHead uses) and learns a
map inside it plus one peak per look. seed.sql also stores that starting
point (look centres) in the `model` table.

Upload CROP_DIR/*.jpg to R2 ``crops/`` and CROP_DIR/photos/*.jpg to R2
``photos/`` (see README). Battles show a person's main face crop first
(closeup preferred); the other photos are a swipe away.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from .bt import HillsHead, Looks


def taste_space(emb: np.ndarray, looks_k: int = 12, rank: int = 16, seed: int = 0, groups: np.ndarray | None = None):
    """(z, look, peaks, peak_group): taste-space coordinates, look per face,
    look centres, and which group (cewek/cowok) each look belongs to.

    With groups, looks are found within each group separately, so a hill
    never mixes cewek and cowok. The taste space itself is shared."""
    if groups is None:
        groups = np.full(len(emb), "", dtype=object)
    centroids, labels, peak_group = [], np.zeros(len(emb), np.int64), []
    for g in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == g)
        k = max(1, min(looks_k, len(idx) // 3))
        looks = Looks.fit(emb[idx], k=k, seed=seed)
        labels[idx] = looks.labels + len(centroids)
        centroids.extend(looks.centroids)
        peak_group.extend([g] * len(looks.centroids))
    head = HillsHead(Looks(np.asarray(centroids), labels), rank=rank)
    head._init(emb)
    z = (emb - head.mu) @ head.P.T
    return z, labels, head.q, peak_group


def source(aid: int, stage: str) -> str:
    """Where a person came from, for the Lab's Sources checklist: 2021 ids are
    Sinemart's own numbers (stage = the Drive folder); later loads put the
    source in stage ("2026")."""
    return "2021" if aid < 200000 else stage


def main(argv: list[str]) -> None:
    if len(argv) not in (4, 5):
        sys.exit("usage: python -m sinemart.publish EMB.npz quality.csv OUT_DIR [gender.csv]")
    from .manifest import PhotoRow, photo_rank

    data = np.load(argv[1])
    with open(argv[2]) as f:
        rows = list(csv.DictReader(f))
    per: dict[int, list[dict]] = {}
    for r in rows:
        r.setdefault("file", f"{r['applicant_id']}.jpg")  # older one-photo quality.csv
        per.setdefault(int(r["applicant_id"]), []).append(r)
    order = lambda r: photo_rank(PhotoRow(0, r["kind"], r["file"], 0, ""))  # noqa: E731
    for photos in per.values():
        photos.sort(key=order)
    # Main face = the first usable crop, closeup preferred.
    main_crop = {a: next((r["file"] for r in p if r["ok"] == "1"), None) for a, p in per.items()}
    quality = {a: p[0] for a, p in per.items()}
    keep = np.array([main_crop.get(int(a)) is not None for a in data["ids"]])
    ids, emb = data["ids"][keep], data["emb"][keep].astype(np.float64)
    gender = {}
    if len(argv) == 5:
        with open(argv[4]) as f:
            gender = {int(r["applicant_id"]): r["gender"] for r in csv.DictReader(f)}
    groups = np.array([gender.get(int(a), "") for a in ids], dtype=object) if gender else None
    z, look, peaks, peak_group = taste_space(emb, groups=groups)
    out = Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rnd = lambda v: [round(float(x), 5) for x in v]  # noqa: E731
    q = lambda v: "'" + str(v).replace("'", "''") + "'"  # noqa: E731
    with open(out / "seed.sql", "w") as sql, open(out / "vectors.ndjson", "w") as vec:
        sql.write("DELETE FROM photos;\n")
        # People from an earlier load that aren't in this one would carry
        # coordinates from an old taste space: drop them (their picks stay).
        # Test batches (test = 1, e.g. "bc") are kept; re-project them
        # afterwards with sinemart.project.
        sql.write(f"DELETE FROM applicants WHERE test = 0 AND id NOT IN ({', '.join(str(a) for a in ids.tolist())});\n")
        for i, aid in enumerate(ids.tolist()):
            sql.write(
                "INSERT INTO applicants (id, image_key, stage, source, look, z, gender, gender_src) VALUES "
                f"({aid}, {q('crops/' + main_crop[aid])}, {q(quality[aid]['stage'])}, {q(source(aid, quality[aid]['stage']))}, {int(look[i])}, "
                f"'{json.dumps(rnd(z[i]))}', {q(gender[aid]) if aid in gender else 'NULL'}, "
                f"{q('auto') if aid in gender else 'NULL'}) "
                # A gender someone fixed by hand on the scores page survives a reload.
                "ON CONFLICT (id) DO UPDATE SET image_key = excluded.image_key, stage = excluded.stage, source = excluded.source, "
                "look = excluded.look, z = excluded.z, score = NULL, score_std = NULL, model = NULL, "
                "gender = CASE WHEN applicants.gender_src = 'manual' THEN applicants.gender ELSE excluded.gender END, "
                "gender_src = CASE WHEN applicants.gender_src = 'manual' THEN 'manual' ELSE excluded.gender_src END;\n"
            )
            # Photos raters can flip through: the face crop first, then each whole photo.
            n = 0
            for r in per[aid]:
                if r.get("view") == "1":
                    sql.write(
                        "INSERT INTO photos (applicant_id, n, kind, image_key, face_ok) VALUES "
                        f"({aid}, {n}, {q(r['kind'])}, {q('photos/' + r['file'])}, {int(r['ok'] == '1')});\n"
                    )
                    n += 1
            vec.write(json.dumps({"id": str(aid), "values": rnd(emb[i])}) + "\n")
        init = json.dumps({"peaks": [rnd(p) for p in peaks], "peak_gender": peak_group, "model": str(data["model"])})
        sql.write(f"INSERT OR REPLACE INTO model (id, init) VALUES (1, '{init}');\n")
        sql.write("DELETE FROM rater_models;\n")  # a new taste space; personal models refit on demand
    n_photos = data["n_photos"][keep] if "n_photos" in data else np.ones(len(ids), int)
    print(f"{len(ids)} applicants ready ({(~keep).sum()} skipped by the quality gate), {len(peaks)} looks, "
          f"{int(n_photos.sum())} face crops averaged")


if __name__ == "__main__":
    main(sys.argv)
