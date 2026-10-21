"""Download the 2026 sign-up photos and build a manifest.

The 2026 search ("SINEMART MENCARI BINTANG 2026! [RAW & SORTIR]") is a Google
Form sheet, not a folder per applicant. Sheet RAW has every sign-up; sheet
SORTIR has the people the team shortlisted. Column "Foto profile (GRID)" holds
a Drive link per person (a file or a folder), mostly shared by link, so they
download without a Drive login.

    python -m sinemart.sheet SHEET_ID OUT_DIR

Writes:
    OUT_DIR/photos/<row>/<n>     the photos as downloaded (no extension)
    OUT_DIR/manifest.csv         one row per photo, for sinemart.crop --grid
    OUT_DIR/labels.csv           applicant_id, shortlisted (1 = in SORTIR)

Applicant ids are ID_BASE + the RAW row, so they never collide with the 2021
applicant numbers (up to ~17k). Names, contacts and every other column stay
in the sheet: nothing here writes them out. labels.csv stays local. It's a
check on the model ("do shortlisted people rank high?"), never shown to
raters, so picks aren't nudged by it.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

ID_BASE = 260000
STAGE = "2026"
PHOTO_COL = "Foto profile (GRID)"
MAX_PER_FOLDER = 6
LINK_RE = re.compile(r"(?:/d/|/folders/|id=)([A-Za-z0-9_-]{20,})")


def curl(url: str, out: Path | None = None) -> str:
    args = ["curl", "-sSL", "--max-time", "120", url]
    if out is not None:
        args[1:1] = ["-o", str(out), "-w", "%{http_code} %{content_type}"]
    return subprocess.run(args, capture_output=True, text=True).stdout


def read_sheet(sheet_id: str, name: str | None = None) -> list[dict]:
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    url = f"{base}/gviz/tq?tqx=out:csv&sheet={name}" if name else f"{base}/export?format=csv"
    return list(csv.DictReader(io.StringIO(curl(url))))


def file_ids(cell: str) -> list[str]:
    """Drive file ids behind a cell: the file itself, or the first files of a folder."""
    ids = []
    for fid in LINK_RE.findall(cell or ""):
        if f"/folders/{fid}" in cell:
            page = curl(f"https://drive.google.com/embeddedfolderview?id={fid}")
            ids += re.findall(r"file/d/([A-Za-z0-9_-]+)", page)[:MAX_PER_FOLDER]
        else:
            ids.append(fid)
    return list(dict.fromkeys(ids))


def is_image(status: str) -> bool:
    parts = status.split()
    return len(parts) > 1 and parts[0] == "200" and (parts[1].startswith("image/") or "octet-stream" in parts[1])


def fetch(row: int, cell: str, out: Path) -> list[Path]:
    folder = out / "photos" / f"{row:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    got = []
    for n, fid in enumerate(file_ids(cell)):
        p = folder / str(n)
        if p.exists() and p.stat().st_size:
            got.append(p)
            continue
        if is_image(curl(f"https://drive.google.com/uc?export=download&id={fid}", p)):
            got.append(p)
        else:  # private link, a video, or an error page
            p.unlink(missing_ok=True)
    return got


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: python -m sinemart.sheet SHEET_ID OUT_DIR")
    out = Path(argv[2])
    raw, sortir = read_sheet(argv[1]), read_sheet(argv[1], "SORTIR")
    key = lambda r: (r.get("Timestamp", "").strip(), r.get("Email Address", "").strip().lower())  # noqa: E731
    short_ts = {key(r)[0] for r in sortir}
    short_em = {key(r)[1] for r in sortir}
    with cf.ThreadPoolExecutor(16) as ex:
        photos = list(ex.map(lambda a: fetch(a[0], a[1].get(PHOTO_COL, ""), out), enumerate(raw)))
    with open(out / "manifest.csv", "w", newline="") as f, open(out / "labels.csv", "w", newline="") as lf:
        w = csv.writer(f)
        w.writerow(["applicant_id", "kind", "path", "size", "stage"])
        lw = csv.writer(lf)
        lw.writerow(["applicant_id", "shortlisted"])
        for row, (r, ps) in enumerate(zip(raw, photos)):
            aid = ID_BASE + row
            ts, em = key(r)
            lw.writerow([aid, int(ts in short_ts or em in short_em)])
            for n, p in enumerate(ps):
                w.writerow([aid, f"photo{n}", p.relative_to(out).as_posix(), p.stat().st_size, STAGE])
    people = sum(1 for ps in photos if ps)
    print(f"{len(raw)} sign-ups, {len(sortir)} shortlisted; {sum(map(len, photos))} photos for {people} people")


if __name__ == "__main__":
    main(sys.argv)
