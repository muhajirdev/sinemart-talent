"""Build an applicant photo manifest from a Google Drive listing.

The Sinemart casting Drive ("Hasil Screening COI 2021") stores one folder per
applicant, e.g. ``01- Lolos SELEKSI TAHAP AWAL/98 - HAYATUN ROSANA/`` holding
``98-Foto-Closeup-hayatun-rosana.jpg``, ``98-Foto-Fullbody1-...`` and a ``.ts``
video. We read a listing produced by::

    rclone lsjson -R --drive-shared-with-me "gdrive:Hasil Screening COI 2021" > listing.json

and turn it into one row per photo. Names stay out of everything downstream:
the app and the model only ever see ``applicant_id``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PHOTO_RE = re.compile(
    r"^(?P<id>\d+)\s*-\s*Foto\s*-\s*(?P<kind>Closeup|Fullbody\d*)\s*-\s*(?P<slug>.+?)\.(?P<ext>jpe?g|png|webp|heic)$",
    re.IGNORECASE,
)
FOLDER_ID_RE = re.compile(r"^(?P<id>\d+)\s*-\s*")


@dataclass
class PhotoRow:
    applicant_id: int
    kind: str  # "closeup" | "fullbody1" | "fullbody2" | ...
    path: str  # path relative to the listing root
    size: int
    stage: str  # top-level selection folder, e.g. "01- Lolos SELEKSI TAHAP AWAL"


def parse_entry(path: str, size: int = 0) -> PhotoRow | None:
    """Return a PhotoRow for a photo path, or None for videos, decks, folders."""
    parts = path.split("/")
    name = parts[-1]
    m = PHOTO_RE.match(name)
    if m:
        applicant_id = int(m["id"])
        kind = m["kind"].lower()
    else:
        # Some uploads are renamed; fall back to the "<id> - NAME" folder.
        if not re.search(r"\.(jpe?g|png|webp|heic)$", name, re.IGNORECASE) or len(parts) < 2:
            return None
        fm = FOLDER_ID_RE.match(parts[-2])
        if not fm:
            return None
        applicant_id = int(fm["id"])
        kind = "closeup" if "close" in name.lower() else "other"
    stage = next((p for p in parts[:-1] if re.match(r"^\d{2}\s*-", p)), "")
    return PhotoRow(applicant_id, kind, path, int(size), stage)


def build_manifest(listing: list[dict]) -> list[PhotoRow]:
    rows = []
    for item in listing:
        if item.get("IsDir"):
            continue
        row = parse_entry(item["Path"], item.get("Size", 0))
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r.applicant_id, r.kind, r.path))
    # The same photo often sits in two stage folders ("01- Lolos" and
    # "02- Video Unik") or is re-uploaded as "...(1).jpg": keep one per kind.
    seen: set[tuple[int, str]] = set()
    out = []
    for r in rows:
        key = (r.applicant_id, r.kind)
        if r.kind != "other" and key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def photo_rank(r: PhotoRow) -> tuple:
    """Closeup first, then full-body 1, 2, ..., then anything else."""
    return (0 if r.kind == "closeup" else 1 if r.kind.startswith("fullbody") else 2, r.kind, r.path)


def group_photos(rows: list[PhotoRow]) -> dict[int, list[PhotoRow]]:
    """Every photo of each applicant, closeup first. One person, many photos."""
    out: dict[int, list[PhotoRow]] = {}
    for r in rows:
        out.setdefault(r.applicant_id, []).append(r)
    for photos in out.values():
        photos.sort(key=photo_rank)
    return out


def pick_headshots(rows: list[PhotoRow]) -> dict[int, PhotoRow]:
    """One headshot per applicant: the closeup, else the first full-body shot."""
    return {aid: photos[0] for aid, photos in group_photos(rows).items()}


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: python -m sinemart.manifest listing.json manifest.csv")
    listing = json.loads(Path(argv[1]).read_text())
    rows = build_manifest(listing)
    with open(argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PhotoRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    people = group_photos(rows)
    closeups = sum(1 for p in people.values() if p[0].kind == "closeup")
    print(f"{len(rows)} photos, {len(people)} applicants, {closeups} with a closeup")


if __name__ == "__main__":
    main(sys.argv)
