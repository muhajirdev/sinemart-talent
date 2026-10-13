"""Face-geometry scores on straightened faces ("Golden ratio" in the app).

    python -m sinemart.facegeom CROP_DIR face_landmarker.task gender.csv OUT_DIR

MediaPipe Face Landmarker gives 478 3D points per face crop. A face turned
left/right or tipped up/down looks narrower or shorter in the photo, which
skews every ratio and above all symmetry. Two steps take the head angle out:

1. Straighten: the 3D points are rotated onto MediaPipe's own frontal
   reference face (stored in the model file), so the face looks at the camera.
   Depth from a single photo is only roughly right (DEPTH_SCALE was tuned on
   our crops so turned and frontal faces come out alike).
2. Correct: whatever the head angle still explains in each measure, across
   all crops, is regressed out (yaw, pitch, their squares and product).

Then each face is measured:

- ``phi_dev``: how far five "golden ratio" proportions are from 1.618
  (mean |log(ratio / phi)|).
- ``pallett_dev``: distance from Pallett, Link & Lee (2010): eyes-to-mouth =
  36% of face length, pupil distance = 46% of face width. There is no hairline
  point, so face length runs from the top of the forehead to the chin.
- ``asymmetry``: mean gap between each point and its mirror partner, in
  pupil-distance units.
- ``atypicality``: Procrustes distance from the average face of the same
  gender (averageness, reversed).

Lower is closer to the ideal for all four; ``score`` is the mean of their
negated z-scores (higher = more ideal). Faces turned more than MAX_POSE_DEG
are left out, since the hidden side of the face is guessed, not seen.

For comparison the same four measures are also taken on the points as they
sit in the photo (no straightening, no correction): fixed model ``geo_raw``.

Writes OUT_DIR/geometry.sql (tables ``geometry`` and ``face_points`` and the
fixed models ``geo`` and ``geo_raw``) and OUT_DIR/geometry.csv.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from .embed import CROP_RE

PHI = (1 + 5**0.5) / 2
MAX_POSE_DEG = 35.0
DEPTH_SCALE = 1.07  # landmark z is roughly, not exactly, in x units
N_MESH = 468  # 468-477 are iris points

# MediaPipe face mesh indices (subject's right = image left).
KEY = {
    "top": 10, "chin": 152, "cheek_r": 234, "cheek_l": 454,
    "eye_r_out": 33, "eye_l_out": 263, "pupil_r": 468, "pupil_l": 473,
    "nose_tip": 1, "ala_r": 64, "ala_l": 294,
    "mouth_r": 61, "mouth_l": 291, "lip_top": 13, "lip_bottom": 14,
}


def pose(mat: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees from MediaPipe's face transformation matrix."""
    r = np.asarray(mat)[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def canonical_face(model_path: str) -> np.ndarray:
    """MediaPipe's frontal reference face (468x3; x right, y up, z to the viewer)."""
    return canonical_mesh(model_path)[0]


def canonical_mesh(model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe's frontal reference face and its triangles (468x3, Tx3).

    It sits in the .task bundle as geometry_pipeline_metadata_landmarks.binarypb:
    field 1 (canonical mesh) holds the vertex buffer as repeated float32 field 3,
    five floats per vertex (x, y, z, u, v), and the triangle indices as field 4.
    """
    import struct
    import zipfile

    blob = zipfile.ZipFile(model_path).read("geometry_pipeline_metadata_landmarks.binarypb")

    def varint(b, i):
        r = s = 0
        while True:
            x = b[i]
            i += 1
            r |= (x & 0x7F) << s
            s += 7
            if x < 0x80:
                return r, i

    def fields(b):
        i = 0
        while i < len(b):
            key, i = varint(b, i)
            f, t = key >> 3, key & 7
            if t == 0:
                v, i = varint(b, i)
            elif t == 5:
                v, i = b[i : i + 4], i + 4
            elif t == 1:
                v, i = b[i : i + 8], i + 8
            elif t == 2:
                n, i = varint(b, i)
                v, i = b[i : i + n], i + n
            else:
                raise ValueError(f"unexpected wire type {t}")
            yield f, t, v

    mesh = next(v for f, _, v in fields(blob) if f == 1)
    floats = [struct.unpack("<f", v)[0] for f, t, v in fields(mesh) if f == 3 and t == 5]
    idx = []
    for f, t, v in fields(mesh):
        if f == 4 and t == 0:
            idx.append(v)
        elif f == 4 and t == 2:  # packed
            i = 0
            while i < len(v):
                x, i = varint(v, i)
                idx.append(x)
    return np.array(floats).reshape(N_MESH, 5)[:, :3], np.array(idx).reshape(-1, 3)


def straighten(mesh: np.ndarray, ref: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Rotate 3D landmarks (normalised x, y down, z depth) onto the frontal reference; Nx2 pixels."""
    w, h = size
    p = np.stack([mesh[:, 0] * w, -mesh[:, 1] * h, -mesh[:, 2] * w * DEPTH_SCALE], axis=1)
    c = p[:N_MESH].mean(0)
    u, _, vt = np.linalg.svd((p[:N_MESH] - c).T @ (ref - ref.mean(0)))
    r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    f = (p - c) @ r + c
    return np.stack([f[:, 0], -f[:, 1]], axis=1)


def pose_corrected(values: np.ndarray, yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Remove what head angle explains (least squares on yaw, pitch, squares, product)."""
    x = np.column_stack([np.ones_like(yaw), yaw, pitch, yaw**2, pitch**2, yaw * pitch])
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x[:, 1:] @ beta[1:]


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def ratios(p: np.ndarray) -> dict[str, float]:
    """The five golden-ratio proportions and Pallett's two, from 2D points."""
    k = {n: p[i] for n, i in KEY.items()}
    st = (k["lip_top"] + k["lip_bottom"]) / 2
    pu = (k["pupil_r"] + k["pupil_l"]) / 2
    length, width = _d(k["top"], k["chin"]), _d(k["cheek_r"], k["cheek_l"])
    mouth = _d(k["mouth_r"], k["mouth_l"])
    return {
        "phi_face": length / width,
        "phi_mouth_nose": mouth / _d(k["ala_r"], k["ala_l"]),
        "phi_nose_chin": _d(k["nose_tip"], k["chin"]) / _d(st, k["chin"]),
        "phi_eyes_mouth": _d(k["eye_r_out"], k["eye_l_out"]) / mouth,
        "phi_pupils_chin": _d(pu, k["chin"]) / _d(pu, k["top"]),
        "pallett_v": _d(pu, st) / length,
        "pallett_h": _d(k["pupil_r"], k["pupil_l"]) / width,
    }


def phi_dev(r: dict[str, float]) -> float:
    return float(np.mean([abs(math.log(v / PHI)) for n, v in r.items() if n.startswith("phi_")]))


def pallett_dev(r: dict[str, float]) -> float:
    return abs(math.log(r["pallett_v"] / 0.36)) + abs(math.log(r["pallett_h"] / 0.46))


def align(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Procrustes: x centred, unit size, rotated onto ref (no reflection)."""
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
    ref = ref / np.linalg.norm(ref)
    for _ in range(iters):
        ref = np.mean([align(s, ref) for s in shapes], axis=0)
        ref = ref / np.linalg.norm(ref)
    return ref


def mirror_pairs(ref: np.ndarray) -> np.ndarray:
    """Each mesh point's mirror partner, found on a symmetric mean face."""
    c = ref - ref.mean(0)
    flipped = c * np.array([-1.0, 1.0])
    return np.linalg.norm(c[:, None] - flipped[None], axis=2).argmin(1)


def asymmetry(p: np.ndarray, pairs: np.ndarray) -> float:
    """Mean gap between the face and its mirror image, in pupil distances."""
    mesh = p[:N_MESH]
    mirrored = (mesh * np.array([-1.0, 1.0]))[pairs]
    scale = np.linalg.norm(mesh - mesh.mean(0))
    gap = np.linalg.norm(align(mirrored, mesh) - align(mesh, mesh), axis=1).mean() * scale
    return float(gap / _d(p[KEY["pupil_r"]], p[KEY["pupil_l"]]))


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() or 1.0)


def detect(crop_dir: Path, model_path: str):
    """(file name, 478x3 landmarks, 4x4 pose matrix, (w, h)) for every crop with a face."""
    import mediapipe as mp

    v = mp.tasks.vision
    lm = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        num_faces=1, output_facial_transformation_matrixes=True))
    out = []
    try:
        for p in sorted(x for x in crop_dir.glob("*.jpg") if CROP_RE.match(x.stem)):
            img = mp.Image.create_from_file(str(p))
            res = lm.detect(img)
            if res.face_landmarks:
                mesh = np.array([[q.x, q.y, q.z] for q in res.face_landmarks[0]])
                out.append((p.name, mesh, np.asarray(res.facial_transformation_matrixes[0]), (img.width, img.height)))
    finally:
        lm.close()
    return out


def measure(faces, gender: dict[int, str], ref: np.ndarray):
    """Per-crop records: raw and straightened points, plus both sets of measures.

    ``r["front"]`` / ``r["m_front"]``: straightened points, pose-corrected measures
    (the main score). ``r["raw"]`` / ``r["m_raw"]``: the points as they sit in the
    photo, measured as is (the "original" score, for comparison).
    """
    recs = []
    for name, mesh, mat, size in faces:
        yaw, pitch = pose(mat)
        aid = int(CROP_RE.match(Path(name).stem)[1])
        rec = {"name": name, "id": aid, "yaw": yaw, "pitch": pitch, "size": size,
               "raw": mesh[:, :2] * np.array(size), "front": straighten(mesh, ref, size)}
        rec["ok"] = abs(yaw) <= MAX_POSE_DEG and abs(pitch) <= MAX_POSE_DEG
        recs.append(rec)
    ok = [r for r in recs if r["ok"]]
    _measure(ok, gender, "front", correct=True)
    _measure(ok, gender, "raw", correct=False)
    return recs


def _measure(ok, gender: dict[int, str], pts: str, correct: bool) -> None:
    pairs = mirror_pairs(mean_shape([r["front"][:N_MESH] for r in ok]))
    refs = {g: mean_shape([r[pts][:N_MESH] for r in ok if gender.get(r["id"], "") == g])
            for g in {gender.get(r["id"], "") for r in ok}}
    for r in ok:
        p = r[pts]
        rt = ratios(p)
        g_ref = refs[gender.get(r["id"], "")]
        r["m_" + pts] = {
            "phi_dev": phi_dev(rt), "pallett_dev": pallett_dev(rt), "asymmetry": asymmetry(p, pairs),
            "atypicality": float(np.linalg.norm(align(p[:N_MESH], g_ref) - g_ref)),
        }
    if correct:
        yaw, pitch = np.array([r["yaw"] for r in ok]), np.array([r["pitch"] for r in ok])
        for m in MEASURES:
            for r, v in zip(ok, pose_corrected(np.array([r["m_" + pts][m] for r in ok]), yaw, pitch)):
                r["m_" + pts][m] = float(v)


MEASURES = ["phi_dev", "pallett_dev", "asymmetry", "atypicality"]


def per_person(recs, gender: dict[int, str], pts: str = "front") -> list[dict]:
    ok = [r for r in recs if r["ok"]]
    rows = []
    for aid in sorted({r["id"] for r in ok}):
        mine = [r for r in ok if r["id"] == aid]
        row = {"applicant_id": aid, "gender": gender.get(aid, ""), "n_photos": len(mine)}
        row.update({m: float(np.mean([r["m_" + pts][m] for r in mine])) for m in MEASURES})
        rows.append(row)
    combo = -np.mean([zscore(np.array([r[m] for r in rows])) for m in MEASURES], axis=0)
    for row, s in zip(rows, combo):
        row["score"] = float(s)
    return rows


def write(recs, rows, raw_rows, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["applicant_id", "gender", "n_photos", "score", *MEASURES])
        w.writeheader()
        w.writerows(rows)
    keypts = lambda p, size: {n: [round(float(p[i][0] / size[0]), 4), round(float(p[i][1] / size[1]), 4)]  # noqa: E731
                              for n, i in KEY.items()}
    with open(out / "geometry.sql", "w") as sql:
        sql.write("DELETE FROM geometry;\nDELETE FROM face_points;\n")
        for r in rows:
            sql.write(
                "INSERT INTO geometry (applicant_id, n_photos, score, phi_dev, pallett_dev, asymmetry, atypicality) "
                f"VALUES ({r['applicant_id']}, {r['n_photos']}, {r['score']:.5f}, "
                + ", ".join(f"{r[m]:.5f}" for m in MEASURES) + ");\n"
            )
        for r in recs:
            # Points as drawn on the photo, plus the straightened ones the numbers come from.
            sql.write(
                "INSERT INTO face_points (image_key, points, front, yaw, pitch) VALUES "
                f"('crops/{r['name']}', '{json.dumps(keypts(r['raw'], r['size']))}', "
                f"'{json.dumps(keypts(r['front'], r['size']))}', {r['yaw']:.1f}, {r['pitch']:.1f});\n"
            )
        # Two fixed models: straightened (the main one) and the photo as is.
        for key, rs in (("geo", rows), ("geo_raw", raw_rows)):
            fixed = {r["applicant_id"]: round(r["score"], 5) for r in rs}
            sql.write(f"INSERT OR REPLACE INTO alt_models (key, init, fixed, n_picks) VALUES ('{key}', '{{}}', '{json.dumps(fixed)}', 0);\n")


def main(argv: list[str]) -> None:
    if len(argv) != 5:
        sys.exit("usage: python -m sinemart.facegeom CROP_DIR face_landmarker.task gender.csv OUT_DIR")
    with open(argv[3]) as f:
        gender = {int(r["applicant_id"]): r["gender"] for r in csv.DictReader(f)}
    recs = measure(detect(Path(argv[1]), argv[2]), gender, canonical_face(argv[2]))
    rows = per_person(recs, gender)
    write(recs, rows, per_person(recs, gender, "raw"), Path(argv[4]))
    n_ok = sum(r["ok"] for r in recs)
    print(f"{n_ok}/{len(recs)} crops measured (straightened), {len(rows)} applicants")


if __name__ == "__main__":
    main(sys.argv)
