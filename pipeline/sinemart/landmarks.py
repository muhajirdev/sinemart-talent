"""Key face points per crop, so the app can draw the golden-ratio lines.

    python -m sinemart.landmarks CROP_DIR face_landmarker.task OUT.sql

Runs MediaPipe Face Landmarker on every aligned crop (``<id>-<kind>.jpg``) and
keeps the 15 points the ratio lines need, as fractions of the crop (0-1), plus
head yaw/pitch in degrees. Writes rows for the ``face_points`` table, keyed by
the crop's R2 key (``crops/<file>``). Model file:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from .embed import CROP_RE

# MediaPipe face mesh indices (subject's right = image left).
POINTS = {
    "top": 10, "chin": 152, "cheek_r": 234, "cheek_l": 454,
    "eye_r_out": 33, "eye_l_out": 263, "pupil_r": 468, "pupil_l": 473,
    "nose_tip": 1, "ala_r": 64, "ala_l": 294,
    "mouth_r": 61, "mouth_l": 291, "lip_top": 13, "lip_bottom": 14,
}


def head_pose(matrix: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees from the face transformation matrix."""
    m = np.asarray(matrix)[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -m[2, 0]))))
    pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
    return yaw, pitch


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        sys.exit("usage: python -m sinemart.landmarks CROP_DIR face_landmarker.task OUT.sql")
    import mediapipe as mp

    v = mp.tasks.vision
    opts = v.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=argv[2]),
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    lm = v.FaceLandmarker.create_from_options(opts)
    paths = sorted(p for p in Path(argv[1]).glob("*.jpg") if CROP_RE.match(p.stem))
    found = 0
    with open(argv[3], "w") as sql:
        sql.write("DELETE FROM face_points;\n")
        for p in paths:
            res = lm.detect(mp.Image.create_from_file(str(p)))
            if not res.face_landmarks:
                continue
            pts = res.face_landmarks[0]
            yaw, pitch = head_pose(res.facial_transformation_matrixes[0])
            points = {k: [round(pts[i].x, 4), round(pts[i].y, 4)] for k, i in POINTS.items()}
            sql.write(
                "INSERT INTO face_points (image_key, points, yaw, pitch) VALUES "
                f"('crops/{p.name}', '{json.dumps(points)}', {yaw:.1f}, {pitch:.1f});\n"
            )
            found += 1
    lm.close()
    print(f"{found}/{len(paths)} crops with face points")


if __name__ == "__main__":
    main(sys.argv)
