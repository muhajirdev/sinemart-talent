"""Fixed "objective" face-geometry scores, as a baseline for the learned taste.

    python -m sinemart.geometry CROP_DIR gender.csv OUT_DIR --model face_landmarker.task [--picks picks.csv]

Runs MediaPipe Face Landmarker (478 points) on every aligned face crop
(``<applicant_id>-<kind>.jpg``, as written by sinemart.crop) and computes four
fixed scores per face. Nothing here is trained on the team's picks:

- ``phi_dev``: how far five popular "golden ratio" proportions are from 1.618
  (mean |log(ratio / phi)|).
- ``pallett_dev``: distance from the two proportions Pallett, Link & Lee (2010)
  found most attractive: eye-to-mouth = 36% of face length and pupil distance
  = 46% of face width. MediaPipe has no hairline point, so face length runs
  from the top of the forehead mesh to the chin; read it as an approximation.
- ``asymmetry``: mean distance between each landmark and its mirror partner
  after reflecting the face, in pupil-distance units.
- ``atypicality``: Procrustes distance from the average face shape of the same
  gender (averageness, reversed).

Lower is "better" for all four. ``geometry_score`` is the mean of their
negated z-scores, so higher = more ideal by these rules. Each applicant's
values are averaged over their usable crops.

Writes OUT_DIR/geometry.csv (one row per applicant) and OUT_DIR/geometry.npz
(``ids``, ``emb``: standardised ratio + shape features) so the same features
can also be loaded as a learned comparison model via sinemart.alt. With
``--picks`` (columns winner_id, loser_id, tie optional) it also prints how
often each fixed score agrees with the team's picks.

Model file: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

PHI = (1 + 5**0.5) / 2

# MediaPipe face mesh indices (subject's right = image left).
TOP, CHIN = 10, 152
CHEEK_R, CHEEK_L = 234, 454
EYE_R_OUT, EYE_R_IN, EYE_L_IN, EYE_L_OUT = 33, 133, 362, 263
PUPIL_R, PUPIL_L = 468, 473
NOSE_TIP, SUBNASALE = 1, 2
ALA_R, ALA_L = 64, 294
MOUTH_R, MOUTH_L = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
N_MESH = 468  # iris points (468-477) are left out of the shape comparisons

MAX_POSE_DEG = 20.0  # skip crops turned or tilted more than this
SMILE = ("mouthSmileLeft", "mouthSmileRight")


def _d(p, a, b) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def ratios(p: np.ndarray) -> dict[str, float]:
    """Named proportions from 2D landmarks (pixels)."""
    stomion = (p[LIP_TOP] + p[LIP_BOTTOM]) / 2
    pupils = (p[PUPIL_R] + p[PUPIL_L]) / 2
    length = _d(p, TOP, CHIN)
    width = _d(p, CHEEK_R, CHEEK_L)
    mouth = _d(p, MOUTH_R, MOUTH_L)
    nose_w = _d(p, ALA_R, ALA_L)
    ipd = _d(p, PUPIL_R, PUPIL_L)
    lips_chin = float(np.linalg.norm(stomion - p[CHIN]))
    return {
        # Golden-ratio claims (each "should" be 1.618).
        "phi_face": length / width,
        "phi_mouth_nose": mouth / nose_w,
        "phi_nose_chin": float(np.linalg.norm(p[NOSE_TIP] - p[CHIN])) / lips_chin,
        "phi_eyespan_mouth": _d(p, EYE_R_OUT, EYE_L_OUT) / mouth,
        "phi_pupils_chin": float(np.linalg.norm(pupils - p[CHIN])) / float(np.linalg.norm(pupils - p[TOP])),
        # Pallett, Link & Lee (2010).
        "pallett_v": float(np.linalg.norm(pupils - stomion)) / length,
        "pallett_h": ipd / width,
        # Extra descriptors for the learned variant.
        "eye_width": (_d(p, EYE_R_OUT, EYE_R_IN) + _d(p, EYE_L_IN, EYE_L_OUT)) / 2 / ipd,
        "nose_width": nose_w / ipd,
        "mouth_width": mouth / ipd,
        "lower_face": lips_chin / length,
        "midface": float(np.linalg.norm(pupils - p[SUBNASALE])) / length,
    }


def phi_dev(r: dict[str, float]) -> float:
    return float(np.mean([abs(math.log(v / PHI)) for k, v in r.items() if k.startswith("phi_")]))


def pallett_dev(r: dict[str, float]) -> float:
    return abs(math.log(r["pallett_v"] / 0.36)) + abs(math.log(r["pallett_h"] / 0.46))


def procrustes(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align shape x onto ref (translation, scale, rotation; no reflection)."""
    x = x - x.mean(0)
    x = x / np.linalg.norm(x)
    r = ref - ref.mean(0)
    r = r / np.linalg.norm(r)
    u, _, vt = np.linalg.svd(x.T @ r)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return x @ (u @ vt)


def mean_shape(shapes: list[np.ndarray], iters: int = 5) -> np.ndarray:
    ref = shapes[0] - shapes[0].mean(0)
    ref /= np.linalg.norm(ref)
    for _ in range(iters):
        ref = np.mean([procrustes(s, ref) for s in shapes], axis=0)
        ref /= np.linalg.norm(ref)
    return ref


def mirror_pairs(ref: np.ndarray) -> np.ndarray:
    """For each landmark, the index of its mirror partner on a frontal mean shape."""
    c = ref - ref.mean(0)
    flipped = c * np.array([-1.0, 1.0])
    d = np.linalg.norm(c[:, None] - flipped[None], axis=2)
    return d.argmin(1)


def asymmetry(p: np.ndarray, pairs: np.ndarray) -> float:
    mesh = p[:N_MESH]
    mirrored = (mesh * np.array([-1.0, 1.0]))[pairs]
    a = procrustes(mirrored, mesh)
    b = procrustes(mesh, mesh)
    ipd = _d(p, PUPIL_R, PUPIL_L)
    scale = np.linalg.norm(mesh - mesh.mean(0))  # undo procrustes' unit scaling
    return float(np.linalg.norm(a - b, axis=1).mean() * scale / ipd)


class Landmarker:
    def __init__(self, model_path: str):
        import mediapipe as mp

        self.mp = mp
        v = mp.tasks.vision
        opts = v.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = v.FaceLandmarker.create_from_options(opts)

    def __call__(self, path: Path):
        """(landmarks Nx2 in pixels, yaw deg, pitch deg, smile 0-1) or None."""
        img = self.mp.Image.create_from_file(str(path))
        res = self.lm.detect(img)
        if not res.face_landmarks:
            return None
        w, h = img.width, img.height
        p = np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]], np.float64)
        m = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        yaw = math.degrees(math.asin(max(-1.0, min(1.0, -m[2, 0]))))
        pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
        smile = float(np.mean([bs.get(k, 0.0) for k in SMILE]))
        return p, yaw, pitch, smile

    def close(self):
        self.lm.close()


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() or 1.0)


def agreement(score: dict[int, float], picks: list[tuple[int, int]]) -> tuple[float, int, float]:
    """(share of picks where the winner scores higher, picks used, bootstrap 95% half-width)."""
    hits = np.array([score[w] > score[l] for w, l in picks if w in score and l in score], float)
    if len(hits) == 0:
        return float("nan"), 0, float("nan")
    rng = np.random.default_rng(0)
    boot = rng.choice(hits, (2000, len(hits))).mean(1)
    return float(hits.mean()), len(hits), float((np.quantile(boot, 0.975) - np.quantile(boot, 0.025)) / 2)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("crop_dir")
    ap.add_argument("gender_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    args = ap.parse_args(argv)

    with open(args.gender_csv) as f:
        gender = {int(r["applicant_id"]): r["gender"] for r in csv.DictReader(f)}

    lm = Landmarker(args.model)
    faces: list[dict] = []
    skipped = {"no_landmarks": 0, "pose": 0}
    for path in sorted(Path(args.crop_dir).glob("*.jpg")):
        m = re.match(r"(\d+)", path.name)
        if not m:
            continue
        out = lm(path)
        if out is None:
            skipped["no_landmarks"] += 1
            continue
        p, yaw, pitch, smile = out
        if abs(yaw) > MAX_POSE_DEG or abs(pitch) > MAX_POSE_DEG:
            skipped["pose"] += 1
            continue
        faces.append({"id": int(m.group(1)), "p": p, "yaw": yaw, "pitch": pitch, "smile": smile})
    lm.close()
    if not faces:
        sys.exit("no usable faces")

    ref = mean_shape([f["p"][:N_MESH] for f in faces])
    pairs = mirror_pairs(ref)
    by_gender = {}
    for g in sorted({gender.get(f["id"], "") for f in faces}):
        by_gender[g] = mean_shape([f["p"][:N_MESH] for f in faces if gender.get(f["id"], "") == g])

    for f in faces:
        r = ratios(f["p"])
        f["ratios"] = r
        f["phi_dev"] = phi_dev(r)
        f["pallett_dev"] = pallett_dev(r)
        f["asymmetry"] = asymmetry(f["p"], pairs)
        g_ref = by_gender[gender.get(f["id"], "")]
        f["atypicality"] = float(np.linalg.norm(procrustes(f["p"][:N_MESH], g_ref) - g_ref))
        f["shape"] = procrustes(f["p"][:N_MESH], ref).ravel()

    # One row per applicant: average over their usable crops.
    ids = sorted({f["id"] for f in faces})
    keys = ["phi_dev", "pallett_dev", "asymmetry", "atypicality", "smile", "yaw", "pitch"]
    ratio_keys = list(faces[0]["ratios"])
    rows, shapes, feats = [], [], []
    for a in ids:
        fs = [f for f in faces if f["id"] == a]
        row = {"applicant_id": a, "gender": gender.get(a, ""), "n_photos": len(fs)}
        row.update({k: float(np.mean([f[k] for f in fs])) for k in keys})
        rows.append(row)
        feats.append([np.mean([f["ratios"][k] for f in fs]) for k in ratio_keys])
        shapes.append(np.mean([f["shape"] for f in fs], 0))

    cols = {k: np.array([r[k] for r in rows]) for k in ["phi_dev", "pallett_dev", "asymmetry", "atypicality"]}
    combo = -np.mean([zscore(v) for v in cols.values()], axis=0)
    for r, s in zip(rows, combo):
        r["geometry_score"] = float(s)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["applicant_id", "gender", "n_photos", "geometry_score", *keys]
    with open(out / "geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Feature vector for a learned geometry model: standardised ratios plus the
    # top principal components of the aligned shape.
    x = np.asarray(feats)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    s = np.asarray(shapes)
    s = s - s.mean(0)
    u, sv, _ = np.linalg.svd(s, full_matrices=False)
    k = min(16, len(ids) - 1)
    pcs = u[:, :k] * sv[:k]
    pcs = (pcs - pcs.mean(0)) / (pcs.std(0) + 1e-9)
    emb = np.hstack([x, pcs])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(out / "geometry.npz", ids=np.array(ids, np.int64), emb=emb.astype(np.float32),
             n_photos=np.array([r["n_photos"] for r in rows]), model="geometry")

    n_crops = len(faces) + sum(skipped.values())
    print(f"{len(faces)}/{n_crops} crops measured ({skipped}), {len(ids)} applicants")

    if args.picks:
        with open(args.picks) as f:
            picks = [(int(r["winner_id"]), int(r["loser_id"])) for r in csv.DictReader(f)
                     if r.get("winner_id") and r.get("tie", "0") in ("", "0")]
        print(f"\nAgreement with {len(picks)} team picks (50% = chance):")
        names = {"geometry_score": "combined", "phi_dev": "golden ratio", "pallett_dev": "Pallett ratios",
                 "asymmetry": "symmetry", "atypicality": "averageness"}
        for key, name in names.items():
            sign = 1 if key == "geometry_score" else -1
            sc = {r["applicant_id"]: sign * r[key] for r in rows}
            acc, n, ci = agreement(sc, picks)
            print(f"  {name:15s} {acc:6.1%} ± {ci:.1%}  (n={n})")


if __name__ == "__main__":
    main()
