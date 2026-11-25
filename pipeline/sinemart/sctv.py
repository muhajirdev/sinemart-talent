"""Daily top-up of the SCTV Mencari Bintang form into the live app.

    CLOUDFLARE_API_TOKEN=... python -m sinemart.sctv SHEET_ID WORK_DIR --yunet yunet.onnx --seg seg.tflite

(The sheet id is the form's responses sheet; it isn't committed, see AGENTS.md.)

Only sign-ups not in D1 yet are touched, so a daily run takes minutes and
re-running is safe. For each new person: download the close-up and full-body
photos, crop faces (the normal gate first; whoever has no usable face gets a
second, looser pass, stage '<TAG>-relaxed'), embed with the ranker's model,
and load as a test batch: applicants (test = 1, source = TAG), full_z with the
live centre and scale (R2 models/m6/full_norm.json), crops to R2. Whoever
still has no usable face goes on the casting page's "Foto tidak terbaca" list
(R2 casting/unreadable/<TAG>.json).

Ids are 910000 + response index (form sheet row = id - 910000 + 2): the form
only appends, so they stay put. Names, phones and emails never leave the
sheet. 2026-11-24: first load (2,063 of 2,141) was done by hand the same way.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

TAG = "sctv-mencari-bintang-2026"
ID_BASE = 910_000
ACCOUNT = "67f961331110b81774851ee4f54349b9"  # same as worker/wrangler.toml
DB = "aaf71364-974f-4ce8-a59a-c4122b30d26c"
BUCKET = "sinemart-talent"
KEY = "m6"  # the ranker's model (clipl)
GENDER = {"perempuan": "f", "laki - laki": "m", "laki-laki": "m"}
# Looser gate for a second pass (2026-11-24: rescued 168 of 246; a few odd
# shots got through, so these are tagged '<TAG>-relaxed').
RELAXED = {"MIN_SCORE": 0.6, "MIN_IOD_PX": 18.0, "MIN_SHARPNESS": 6.0, "MAX_YAW": 0.45, "SECOND_FACE_RATIO": 0.95}
FILE_ID = re.compile(r"id=([\w-]{20,})")
API = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}"


def _token() -> str:
    return os.environ["CLOUDFLARE_API_TOKEN"]


def curl(args: list[str]) -> str:
    return subprocess.run(["curl", "-sS", *args], capture_output=True, text=True).stdout


def d1(sql: str) -> list[dict]:
    """One D1 query over the REST API, with backoff (it throttles bulk writes)."""
    for t in range(8):
        r = curl(["-H", f"Authorization: Bearer {_token()}", "-H", "Content-Type: application/json",
                  f"{API}/d1/database/{DB}/query", "-d", json.dumps({"sql": sql})])
        try:
            j = json.loads(r)
        except json.JSONDecodeError:
            j = {"success": False}
        if j.get("success"):
            return j["result"][0]["results"]
        time.sleep(5 * (t + 1))
    raise SystemExit(f"D1 failed: {r[:300]}")


def r2_put(key: str, path: Path | None = None, data: bytes | None = None, ctype: str = "image/jpeg") -> bool:
    body = ["--data-binary", f"@{path}"] if path else ["--data-binary", "@-"]
    for t in range(6):
        p = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-X", "PUT",
                            "-H", f"Authorization: Bearer {_token()}", "-H", f"Content-Type: {ctype}", *body,
                            f"{API}/r2/buckets/{BUCKET}/objects/{key}"], input=data, capture_output=True)
        if p.stdout.decode() == "200":
            return True
        time.sleep(3 * (t + 1))
    return False


def r2_get(key: str) -> bytes:
    return subprocess.run(["curl", "-sS", "-H", f"Authorization: Bearer {_token()}",
                           f"{API}/r2/buckets/{BUCKET}/objects/{key}"], capture_output=True).stdout


def read_form(sheet_id: str) -> tuple[list[str], list[list[str]]]:
    text = curl(["-L", f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"])
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def columns(header: list[str]) -> dict[str, list[int]]:
    """Gender, close-up and full-body columns by their question text (the form
    renamed the close-up question once, so there are two close-up columns)."""
    low = [h.lower() for h in header]
    return {
        "gender": [i for i, h in enumerate(low) if h.startswith("jenis kelamin")],
        "close": [i for i, h in enumerate(low) if "close up" in h],
        "body": [i for i, h in enumerate(low) if h.startswith("foto full body")],
    }


def first_id(row: list[str], cols: list[int]) -> str | None:
    for i in cols:
        m = FILE_ID.search(row[i] if i < len(row) else "")
        if m:
            return m[1]
    return None


def download(fid: str, out: Path) -> bool:
    for url in (f"https://drive.google.com/uc?export=download&id={fid}",
                f"https://drive.google.com/thumbnail?id={fid}&sz=w4000"):
        s = curl(["-L", "--max-time", "120", "-o", str(out), "-w", "%{http_code} %{content_type}", url])
        if s.startswith("200 image"):
            return True
    out.unlink(missing_ok=True)
    return False


def crop(manifest: Path, root: Path, out: Path, yunet: str, seg: str | None, loose: bool) -> dict[int, list[str]]:
    """Run sinemart.crop; returns each person's rejection reasons (empty = usable)."""
    from . import crop as c

    saved = {k: getattr(c, k) for k in RELAXED}
    if loose:
        for k, v in RELAXED.items():
            setattr(c, k, v)
    try:
        c.main([str(manifest), str(root), str(out), "--yunet", yunet] + (["--seg", seg] if seg else []))
    finally:
        for k, v in saved.items():
            setattr(c, k, v)
    people: dict[int, list[str]] = {}
    for r in csv.DictReader(open(out / "quality.csv")):
        reasons = people.setdefault(int(r["applicant_id"]), [])
        reasons.append("" if r["ok"] == "1" else (r["reasons"].split() or ["no_face"])[0])
    return people


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_id")
    ap.add_argument("work")
    ap.add_argument("--yunet", required=True)
    ap.add_argument("--seg")
    ap.add_argument("--dry-run", action="store_true", help="crop and score, write nothing to the app")
    a = ap.parse_args(argv)
    work = Path(a.work)
    (work / "photos").mkdir(parents=True, exist_ok=True)

    header, rows = read_form(a.sheet_id)
    col = columns(header)
    have = {r["id"] for r in d1(f"SELECT id FROM applicants WHERE id >= {ID_BASE} AND id < {ID_BASE + 100_000}")}
    old_unread = json.loads(r2_get(f"casting/unreadable/{TAG}.json") or b"[]")
    tried = {u["id"] for u in old_unread}
    new = [i for i in range(len(rows)) if ID_BASE + i not in have and ID_BASE + i not in tried]
    print(f"{len(rows)} sign-ups, {len(have)} in the app, {len(tried)} unreadable before, {len(new)} new")
    if not new:
        return

    gender = {ID_BASE + i: GENDER.get(rows[i][col["gender"][0]].strip().lower(), "") for i in new}
    jobs = [(ID_BASE + i, kind, fid) for i in new for kind in ("close", "body") if (fid := first_id(rows[i], col[kind]))]
    with cf.ThreadPoolExecutor(12) as ex:
        got = list(ex.map(lambda j: (j, download(j[2], work / "photos" / f"{j[0]}-{j[1]}")), jobs))
    with open(work / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["applicant_id", "kind", "path", "size", "stage"])
        for (aid, kind, _), ok in got:
            if ok and gender[aid]:
                p = work / "photos" / f"{aid}-{kind}"
                w.writerow([aid, kind, p.relative_to(work).as_posix(), p.stat().st_size, TAG])

    strict = crop(work / "manifest.csv", work, work / "crops", a.yunet, a.seg, loose=False)
    ok = {aid for aid, rs in strict.items() if "" in rs}
    stage = {aid: TAG for aid in ok}
    retry = [r for r in csv.DictReader(open(work / "manifest.csv")) if int(r["applicant_id"]) not in ok]
    reasons = {aid: rs for aid, rs in strict.items()}
    if retry:
        with open(work / "manifest2.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(retry[0].keys()))
            w.writeheader()
            w.writerows(retry)
        loose = crop(work / "manifest2.csv", work, work / "crops2", a.yunet, a.seg, loose=True)
        for aid, rs in loose.items():
            reasons[aid] = rs
            if "" in rs:
                stage[aid] = f"{TAG}-relaxed"
        for p in (work / "crops2").glob("*.jpg"):  # one folder for embedding
            p.replace(work / "crops" / p.name)

    unread = []
    for aid in sorted(ID_BASE + i for i in new):
        if aid in stage:
            continue
        rs = [r for r in reasons.get(aid, []) if r]
        why = "no_gender" if not gender[aid] else ("no_photo" if aid not in reasons else (rs[-1] if rs else "no_face"))
        unread.append({"id": aid, "row": aid - ID_BASE + 2, "gender": gender[aid] or None, "reason": why})

    # Embed and score with the live ranker (no per-face correction: no battles).
    from .embed import embed_dir
    from .full import sql_rows, vectors

    loaded = 0
    if stage:
        ids, emb, _ = embed_dir(work / "crops", "clipl")
        keep = np.isin(ids, list(stage))
        ids, emb = ids[keep], emb[keep].astype(np.float64)
        norm = json.loads(r2_get(f"models/{KEY}/full_norm.json"))
        x = vectors(emb, norm)
        p = json.loads(d1("SELECT params_ranker FROM model WHERE id = 1")[0]["params_ranker"])
        raw = x @ np.asarray(p["w"])
        print(f"scored {len(ids)}: {sum(stage[int(i)] == TAG for i in ids)} normal, {sum(stage[int(i)] != TAG for i in ids)} relaxed")
        if a.dry_run:
            print(f"dry run: {len(unread)} unreadable; nothing written")
            return
        crops = [c for c in (work / "crops").glob("9*.jpg") if int(c.stem.split("-")[0]) in stage]
        with cf.ThreadPoolExecutor(6) as ex:
            up = sum(ex.map(lambda c: r2_put(f"crops/{c.name}", c), crops))
        print(f"uploaded {up}/{len(crops)} crops")
        vals = []
        for i, s in zip(ids, raw):
            i = int(i)
            key = f"crops/{i}-close.jpg" if (work / "crops" / f"{i}-close.jpg").exists() else f"crops/{i}-body.jpg"
            vals.append(f"({i},'{key}','{stage[i]}','{gender[i]}','form','{TAG}',1,'line',{s:.5f},datetime('now'))")
        for k in range(0, len(vals), 100):
            d1("INSERT OR REPLACE INTO applicants (id,image_key,stage,gender,gender_src,source,test,model,score,updated_at) VALUES "
               + ",".join(vals[k:k + 100]))
        lines = sql_rows(KEY, ids, x)
        for k in range(0, len(lines), 15):
            d1("\n".join(lines[k:k + 15]))
        loaded = len(ids)
    elif a.dry_run:
        print(f"dry run: {len(unread)} unreadable; nothing written")
        return
    if unread:
        r2_put(f"casting/unreadable/{TAG}.json", data=json.dumps(old_unread + unread).encode(), ctype="application/json")
    print(f"done: {loaded} added to the app, {len(unread)} unreadable")


if __name__ == "__main__":
    main()
