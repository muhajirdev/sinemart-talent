"""Face detect -> align -> crop -> grey background -> quality gate.

Every applicant photo becomes a 512x512 headshot with the eyes level, at the
same height and the same distance apart, on a flat grey background. That way
the embedding sees the face, not the bedroom wall or the phone's lens.

Detector: OpenCV YuNet (``cv2.FaceDetectorYN``), which gives a box, 5
landmarks and a score. Download the model once:
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Background: MediaPipe selfie segmenter when installed (``pip install mediapipe``
plus ``selfie_segmenter.tflite``); without it the crop keeps its background and
the report says ``bg="kept"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

OUT = 512
GREY = 128
# Crop side = SIDE_IOD x inter-ocular distance; eyes sit EYE_Y down from the top.
SIDE_IOD = 5.0
EYE_Y = 0.40


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h
    right_eye: tuple[float, float]  # subject's right eye (image left)
    left_eye: tuple[float, float]
    nose: tuple[float, float]
    score: float

    @property
    def iod(self) -> float:
        return math.dist(self.right_eye, self.left_eye)

    @classmethod
    def from_yunet(cls, row: np.ndarray) -> "Face":
        r = [float(v) for v in row]
        return cls((r[0], r[1], r[2], r[3]), (r[4], r[5]), (r[6], r[7]), (r[8], r[9]), r[14])


@dataclass
class Quality:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    face_score: float = 0.0
    iod_px: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    yaw: float = 0.0
    bg: str = "kept"


# Gate thresholds. Tune on a few hundred real crops, not in the abstract.
MIN_SCORE = 0.8
MIN_IOD_PX = 30.0  # face roughly 75px wide in the original
MIN_SHARPNESS = 15.0  # Laplacian variance on the 512px crop's face region (a sharp portrait scores ~20+)
BRIGHTNESS = (50.0, 210.0)
MAX_YAW = 0.30  # nose offset from the eye midpoint, in inter-ocular units
SECOND_FACE_RATIO = 0.3  # a second face this big (by area) means a group photo


class Detector:
    def __init__(self, model_path: str, score_threshold: float = 0.6):
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)

    def __call__(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        # YuNet is happiest around 640px; detect on a downscale, map back.
        s = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.det.detect(small)
        if rows is None:
            return []
        rows = rows.copy()
        rows[:, :14] /= s
        faces = [Face.from_yunet(r) for r in rows]
        return sorted(faces, key=lambda f: f.box[2] * f.box[3], reverse=True)


def align_matrix(face: Face, out: int = OUT) -> np.ndarray:
    """Affine map from the original photo to the aligned out x out crop."""
    (rx, ry), (lx, ly) = face.right_eye, face.left_eye
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    cx, cy = (rx + lx) / 2, (ry + ly) / 2
    scale = out / (SIDE_IOD * face.iod)
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    m[0, 2] += out / 2 - cx
    m[1, 2] += EYE_Y * out - cy
    return m


def yaw_proxy(face: Face) -> float:
    """Nose offset along the eye axis, in inter-ocular units (0 = frontal)."""
    (rx, ry), (lx, ly) = face.right_eye, face.left_eye
    ex, ey = (lx - rx) / face.iod, (ly - ry) / face.iod
    nx, ny = face.nose[0] - (rx + lx) / 2, face.nose[1] - (ry + ly) / 2
    return abs(nx * ex + ny * ey) / face.iod


def face_region(crop: np.ndarray) -> np.ndarray:
    """Central face area of an aligned crop (between brows and chin)."""
    n = crop.shape[0]
    return crop[int(0.30 * n) : int(0.75 * n), int(0.30 * n) : int(0.70 * n)]


def grey_background(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite the person onto flat grey; mask is float in [0, 1], HxW."""
    m = np.clip(mask, 0, 1)[..., None].astype(np.float32)
    out = crop.astype(np.float32) * m + GREY * (1 - m)
    return out.round().astype(np.uint8)


class Segmenter:
    """MediaPipe selfie segmenter; returns a person-probability mask."""

    def __init__(self, model_path: str):
        import mediapipe as mp  # optional dependency

        self.mp = mp
        opts = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_confidence_masks=True,
        )
        self.seg = mp.tasks.vision.ImageSegmenter.create_from_options(opts)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.seg.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        # selfie_segmenter has one mask (person); multiclass has background first.
        masks = res.confidence_masks
        m = masks[0].numpy_view() if len(masks) == 1 else 1 - masks[0].numpy_view()
        return cv2.GaussianBlur(np.asarray(m, np.float32), (5, 5), 0)

    def close(self):
        self.seg.close()


def assess(faces: list[Face], crop: np.ndarray, allow_many: bool = False) -> Quality:
    if not faces:
        return Quality(False, ["no_face"])
    f = faces[0]
    region = cv2.cvtColor(face_region(crop), cv2.COLOR_BGR2GRAY)
    q = Quality(
        ok=True,
        face_score=f.score,
        iod_px=f.iod,
        sharpness=float(cv2.Laplacian(region, cv2.CV_64F).var()),
        brightness=float(region.mean()),
        yaw=yaw_proxy(f),
    )
    if f.score < MIN_SCORE:
        q.reasons.append("low_face_score")
    if f.iod < MIN_IOD_PX:
        q.reasons.append("face_too_small")
    if q.sharpness < MIN_SHARPNESS:
        q.reasons.append("blurry")
    if not BRIGHTNESS[0] <= q.brightness <= BRIGHTNESS[1]:
        q.reasons.append("bad_exposure")
    if q.yaw > MAX_YAW:
        q.reasons.append("not_frontal")
    if len(faces) > 1 and not allow_many:
        a0 = f.box[2] * f.box[3]
        if faces[1].box[2] * faces[1].box[3] > SECOND_FACE_RATIO * a0:
            q.reasons.append("multiple_faces")
    q.ok = not q.reasons
    return q


def process(img: np.ndarray, detector: Detector, segmenter: Segmenter | None = None):
    """Return (crop or None, Quality) for one BGR photo."""
    faces = detector(img)
    if not faces:
        return None, Quality(False, ["no_face"])
    crop = cv2.warpAffine(
        img, align_matrix(faces[0]), (OUT, OUT),
        flags=cv2.INTER_AREA if faces[0].iod * SIDE_IOD > OUT else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(GREY, GREY, GREY),
    )
    q = assess(faces, crop)
    if segmenter is not None:
        crop = grey_background(crop, segmenter(crop))
        q.bg = "grey"
    return crop, q


def grid_faces(faces: list[Face]) -> list[Face]:
    """Every face in the photo about as big as the largest one.

    The 2026 sign-up form asks for a photo GRID: several shots of the same
    person in one image. Each big face becomes its own crop; small faces in
    the background are ignored."""
    if not faces:
        return []
    a0 = faces[0].box[2] * faces[0].box[3]
    return [f for f in faces if f.box[2] * f.box[3] > SECOND_FACE_RATIO * a0]


def process_grid(img: np.ndarray, detector: Detector, segmenter: Segmenter | None = None):
    """Like process(), but one (crop, Quality) per big face, for grid photos."""
    faces = detector(img)
    big = grid_faces(faces)
    if not big:
        return [(None, Quality(False, ["no_face"]))]
    out = []
    for f in big:
        crop = cv2.warpAffine(
            img, align_matrix(f), (OUT, OUT),
            flags=cv2.INTER_AREA if f.iod * SIDE_IOD > OUT else cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(GREY, GREY, GREY),
        )
        q = assess([f], crop, allow_many=True)
        if segmenter is not None:
            crop = grey_background(crop, segmenter(crop))
            q.bg = "grey"
        out.append((crop, q))
    return out


VIEW_MAX = 1280  # long side of the full photo kept for viewing


def main(argv: list[str] | None = None) -> None:
    """Crop every photo of every applicant from a manifest.

    python -m sinemart.crop manifest.csv PHOTO_ROOT CROP_DIR --yunet yunet.onnx [--seg selfie_segmenter.tflite]

    One person can have several photos (closeup, full-body 1 and 2, ...).
    For each photo it writes:
      CROP_DIR/<id>-<kind>.jpg         aligned grey face crop, if it passes the gate
      CROP_DIR/rejected/<id>-<kind>.jpg  the crop if it doesn't
      CROP_DIR/photos/<id>-<kind>.jpg  the whole photo, resized, for raters to look at
    plus CROP_DIR/quality.csv with one row per photo. The model embeds every
    passing face crop of a person and averages them (sinemart.embed).
    """
    import argparse
    import csv
    from pathlib import Path

    from .manifest import PhotoRow, group_photos

    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("photo_root")
    ap.add_argument("crop_dir")
    ap.add_argument("--yunet", required=True)
    ap.add_argument("--seg")
    ap.add_argument("--grid", action="store_true", help="crop every big face of a photo (2026 photo grids)")
    args = ap.parse_args(argv)

    with open(args.manifest) as f:
        rows = [PhotoRow(int(r["applicant_id"]), r["kind"], r["path"], int(r["size"]), r["stage"]) for r in csv.DictReader(f)]
    people = group_photos(rows)
    out = Path(args.crop_dir)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(parents=True, exist_ok=True)
    detector = Detector(args.yunet)
    segmenter = Segmenter(args.seg) if args.seg else None
    fields = ["applicant_id", "kind", "file", "stage", "ok", "view", "reasons", "face_score", "iod_px",
              "sharpness", "brightness", "yaw", "bg"]
    done = passed = people_ok = 0
    with open(out / "quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for aid, photos in sorted(people.items()):
            any_ok = False
            used: set[str] = set()
            for row in photos:
                name = f"{aid}-{row.kind}.jpg"
                for n in range(2, 100):  # several "other" photos
                    if name not in used:
                        break
                    name = f"{aid}-{row.kind}{n}.jpg"
                used.add(name)
                img = cv2.imread(str(Path(args.photo_root) / row.path))
                if img is None:
                    results = [(None, Quality(False, ["unreadable"]))]
                else:
                    results = process_grid(img, detector, segmenter) if args.grid else [process(img, detector, segmenter)]
                    s = VIEW_MAX / max(img.shape[:2])
                    view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
                    cv2.imwrite(str(out / "photos" / name), view, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # The whole photo is viewed once (view=1 on its first row);
                # each extra face of a grid is a crop of its own (view=0).
                for j, (crop, q) in enumerate(results):
                    crop_name = name if j == 0 else name.replace(".jpg", f"f{j + 1}.jpg")
                    if crop is not None:
                        cv2.imwrite(str(out / ("" if q.ok else "rejected") / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    passed += q.ok
                    any_ok |= q.ok
                    w.writerow({
                        "applicant_id": aid, "kind": row.kind, "file": crop_name, "stage": row.stage, "ok": int(q.ok),
                        "view": int(img is not None and j == 0), "reasons": " ".join(q.reasons),
                        "face_score": f"{q.face_score:.3f}", "iod_px": f"{q.iod_px:.1f}", "sharpness": f"{q.sharpness:.1f}",
                        "brightness": f"{q.brightness:.1f}", "yaw": f"{q.yaw:.3f}", "bg": q.bg,
                    })
                done += 1
                print(f"\r{done}/{len(rows)} photos, {passed} usable faces", end="", flush=True)
            people_ok += any_ok
    print(f"\n{people_ok}/{len(people)} applicants have at least one usable face")
    if segmenter is not None:
        segmenter.close()


if __name__ == "__main__":
    main()
