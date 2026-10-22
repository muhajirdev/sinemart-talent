"""Face-part embeddings ("Model 8"): eyes, nose, mouth and jaw, each on its own.

    python -m sinemart.parts CROP_DIR face_landmarker.task OUT.npz [MODEL]

The other models see the whole face at once. Here MediaPipe's landmarks place
four boxes on every aligned crop, each box is embedded on its own (SigLIP 2
base by default, the same as Model 1), the parts are averaged over a person's
photos, and the four vectors are joined into one. sinemart.alt then loads it
like any other comparison model, so the app learns which parts matter from
the same picks.

Crops where MediaPipe finds no face are skipped; a person with no usable crop
gets their group's average in sinemart.alt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .embed import CROP_RE, load, per_person
from .facegeom import KEY, detect

PARTS = ("eyes", "nose", "mouth", "jaw")


def boxes(p: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """(left, top, right, bottom) in pixels for each part, from 478 landmark points in pixels."""
    k = {n: p[i, :2] for n, i in KEY.items()}
    eye_w = float(np.linalg.norm(k["eye_l_out"] - k["eye_r_out"]))
    eye_mid = (k["eye_l_out"] + k["eye_r_out"]) / 2
    pupil_y = float((k["pupil_r"][1] + k["pupil_l"][1]) / 2)
    nose_h = float(k["nose_tip"][1]) - pupil_y
    ala_w = float(np.linalg.norm(k["ala_l"] - k["ala_r"]))
    mouth_w = float(np.linalg.norm(k["mouth_l"] - k["mouth_r"]))
    mouth_mid = (k["mouth_l"] + k["mouth_r"] + k["lip_top"] + k["lip_bottom"]) / 4
    chin_y = float(k["chin"][1])
    return {
        # Both eyes and brows: brows sit roughly half an eye-width above the pupils.
        "eyes": (eye_mid[0] - 0.65 * eye_w, pupil_y - 0.45 * eye_w, eye_mid[0] + 0.65 * eye_w, pupil_y + 0.2 * eye_w),
        "nose": (k["nose_tip"][0] - 0.9 * ala_w, pupil_y + 0.15 * nose_h,
                 k["nose_tip"][0] + 0.9 * ala_w, k["nose_tip"][1] + 0.25 * nose_h),
        "mouth": (mouth_mid[0] - 0.75 * mouth_w, mouth_mid[1] - 0.4 * mouth_w,
                  mouth_mid[0] + 0.75 * mouth_w, mouth_mid[1] + 0.4 * mouth_w),
        # Jawline and chin: cheek to cheek, from the mouth down past the chin.
        "jaw": (k["cheek_r"][0], mouth_mid[1] - 0.15 * mouth_w, k["cheek_l"][0], chin_y + 0.1 * (chin_y - pupil_y)),
    }


def cut(img, box):
    """The box, padded with grey onto a square so the encoder doesn't stretch it."""
    from PIL import Image

    part = img.crop(tuple(round(v) for v in box))
    side = max(part.size)
    sq = Image.new("RGB", (side, side), (128, 128, 128))
    sq.paste(part, ((side - part.width) // 2, (side - part.height) // 2))
    return sq


def main(argv: list[str]) -> None:
    if len(argv) not in (4, 5):
        sys.exit("usage: python -m sinemart.parts CROP_DIR face_landmarker.task OUT.npz [MODEL]")
    from PIL import Image

    crop_dir, name = Path(argv[1]), argv[4] if len(argv) == 5 else "siglip2"
    faces = detect(crop_dir, argv[2])
    encode = load(name)
    ids = [int(CROP_RE.match(Path(f).stem)[1]) for f, *_ in faces]
    per_part = {part: [] for part in PARTS}
    for i in range(0, len(faces), 32):
        group = faces[i : i + 32]
        cuts = {part: [] for part in PARTS}
        for fname, mesh, _, size in group:
            img = Image.open(crop_dir / fname).convert("RGB")
            for part, box in boxes(mesh[:, :2] * np.array(size)).items():
                cuts[part].append(cut(img, box))
        for part in PARTS:
            per_part[part].append(encode(cuts[part]))
        print(f"\rparts: {i + len(group)}/{len(faces)} photos", end="", flush=True)
    print()
    people, joined, n = None, [], None
    for part in PARTS:
        people, emb, n = per_person(ids, np.concatenate(per_part[part]))
        joined.append(emb)
    emb = np.concatenate(joined, axis=1) / np.sqrt(len(PARTS))  # still unit length
    np.savez(argv[3], ids=people, emb=emb.astype(np.float32), n_photos=n, model=f"{name} parts {','.join(PARTS)}")
    print(f"saved {emb.shape} to {argv[3]} ({len(faces)} crops with a face, {len(people)} people)")


if __name__ == "__main__":
    main(sys.argv)
