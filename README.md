# Sinemart talent screening prototype

The goal is to rank ~50k casting applicants by one "Sinemart taste" score that
the talent team teaches the model with quick A/B picks, so the team only has to
look closely at the top few hundred.

```
Drive photos ──► crop/align/grey/gate ──► SigLIP 2 (or DINOv3) embeddings
                                                │
   team A/B picks (PWA ► Worker ► D1) ──► heads A/B/C ──► scores + next pairs ──► D1
```

- **One universal score**, not one per role (decided 2026-10-12). Per-role tweaks
  can go on top later.
- **Beauty is a landscape of hills.** There is one hill per look. Each hill
  has a peak (that look's ideal face) and a height. A face scores by how close
  it stands to a peak, so pushing past the ideal, towards caricature, scores
  lower again. Picks train this as a Bradley–Terry model,
  P(a beats b) = sigmoid(s(a) − s(b)), with
  `s(e) = log Σ_k exp(h_k − |P e − q_k|² / 2)` (`pipeline/sinemart/bt.py`,
  `HillsHead`). The peaks start at k-means look centres and then move with
  the picks, because the most attractive face of a look isn't its average
  face. Every peak is equally tall by default, so the best face of any look
  can reach the top; `equal_peaks=False` lets the picks make some hills
  taller. The hills model always does the ranking and decides which pairs
  are worth asking.
- **Baselines, for comparison only:** a straight arrow (A), the arrow measured
  from each look's centre (B), and a small MLP (C). They sit next to the hills
  model in the held-out report so you can see whether the hills earn their
  keep.
- **Pairs:** 70% uncertain, half of them between two faces on the same hill (faces the hills model can't place yet, measured
  by the spread of a 5-model bootstrap ensemble), 20% from the
  current top 500, 10% random. Left and right are shuffled.
- **Measured:** held-out agreement on *unseen applicants* (the split holds out
  faces, not pairs), agreement within each look, the team's own repeat-pick
  consistency (the ceiling), and a 10-band check: bin faces by
  straight-arrow score and plot the team's win rate per band. A peak followed
  by a drop means the curved head is earning its keep.
- Elo duels are only for the final ~150. They are not part of this repo yet.

## The Drive data

The shared folder `Hasil Screening COI 2021` (owner: the Sinemart casting data
account, shared read-only) has this layout:

```
Hasil Screening COI 2021/
  Foto _ Video Peserta/
    01- Lolos SELEKSI TAHAP AWAL/      passed first screening
      <id> - <NAME>/
        <id>-Foto-Closeup-<name>.jpg   headshot   (jpg/jpeg/png, ~80 KB – 4 MB)
        <id>-Foto-Fullbody1-<name>.jpg
        <id>-Foto-Fullbody2-<name>.jpg
        *.ts                           short video clip
    02- Video Unik (Pilihan)/
  HASIL SELEKSI AWAL/
    Seleksi Peserta COI 2021 -diva.pptx  23 finalists sent to Jakarta, with IDs
```

Applicant IDs in the deck go up to ~15,500, so the 2021 season had at least that
many sign-ups. How many of them have photos in Drive still needs a full listing
(step 1 below). The folder names and the finalist deck are free labels for
sanity checks: finalists should land near the top. They are not training data.

## Run it

### 1. List and download the photos (read-only)

```bash
rclone config        # add a Google Drive remote "gdrive", scope drive.readonly
mkdir -p data && rclone lsjson -R --drive-shared-with-me "gdrive:Hasil Screening COI 2021" > data/listing.json
cd pipeline && pip install -r requirements.txt
python -m sinemart.manifest ../data/listing.json ../data/manifest.csv
rclone copy --drive-shared-with-me "gdrive:Hasil Screening COI 2021" ../data/photos \
  --include "*Foto-*.{jpg,jpeg,JPG,JPEG,png}"     # every photo; videos skipped
```

### 2. Crop, then embed

```bash
curl -LO https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
python -m sinemart.crop ../data/manifest.csv ../data/photos ../data/crops \
  --yunet face_detection_yunet_2023mar.onnx --seg selfie_segmenter.tflite
python -m sinemart.embed ../data/crops siglip2 ../data/siglip2.npz
python -m sinemart.embed ../data/crops dinov3  ../data/dinov3.npz   # challenger
python -m sinemart.gender ../data/crops ../data/gender.csv           # cewek / cowok guess
```

Cewek and cowok are scored separately: battles never cross the two, and each
has its own hills and leaderboard. The COI sheet has no gender column, so
`sinemart.gender` guesses zero-shot with SigLIP 2 ("a photo of a woman" vs "a
photo of a man", averaged over a person's face crops). Fix wrong guesses on the
scores page; a fix survives reloads. Gender never feeds into the score.

One person can have several photos (closeup, full-body 1 and 2). Every photo
is cropped; each one whose face passes the gate is embedded, and a person's
embedding is the average of theirs, so one odd photo moves their score less.
Battles, scores and leaderboards are per person: raters see the face crop
first and swipe (or tap the counter) to see that person's whole photos.

Each crop is 512×512 with the eyes level and fixed in position, on grey. The
gate rejects photos with no face, a face too small, blur, bad exposure, a
non-frontal pose, or several faces. Rejected crops go to `crops/rejected/` with
reasons in `crops/quality.csv`. Look through a few hundred of these before
trusting the thresholds.

### 3. Cloudflare

```bash
cd ../worker && npm install
npx wrangler d1 create sinemart-talent          # paste the id into wrangler.toml
npx wrangler r2 bucket create sinemart-talent
npx wrangler d1 execute sinemart-talent --remote --file schema.sql
npx wrangler d1 execute sinemart-talent --remote --file migrate.sql   # only for a database made before the gender columns
npx wrangler secret put ADMIN_TOKEN
python -m sinemart.publish ../data/siglip2.npz ../data/crops/quality.csv ../data/out ../data/gender.csv   # from pipeline/
npx wrangler d1 execute sinemart-talent --remote --file ../data/out/seed.sql
rclone copy ../data/crops r2:sinemart-talent/crops --max-depth 1 --include "*.jpg"   # face crops
rclone copy ../data/crops/photos r2:sinemart-talent/photos                          # whole photos
npx wrangler deploy
```

The app is open: anyone with the link types their name (no password, no
code) and starts picking. Search engines are kept out (robots.txt, noindex); the same name on another phone is the same rater, and
every pick is saved under it. Then they tap the more screen-worthy face. On a laptop they can use ← and →, and space to skip.
Until the first training run the app serves random pairs.

The scores page trains on everyone's picks by default (that model also picks
the pairs). Its "Taste of" switch shows one rater's own model instead, and the
agreement table shows how often the team model, and each other rater's
model, gets that rater's picks right.

### 4. Train, then refill the queue (repeat every few hundred picks)

```bash
curl "https://<worker>/api/export.csv?admin=$ADMIN_TOKEN" > ../data/picks.csv
python -m sinemart.train ../data/siglip2.npz ../data/picks.csv ../data/out
npx wrangler d1 execute sinemart-talent --remote --file ../data/out/update.sql
```

`out/report.json` has held-out agreement for the hills model and A, B and C,
how many faces stand on each hill, agreement within each
look, the repeat-pick ceiling and the 10 bands. Run the same picks against
`dinov3.npz` to choose the backbone.

### 5. Model 2 (a second embedding to compare on the scores page)

```bash
python -m sinemart.embed ../data/crops dinov2 ../data/dinov2.npz   # dinov3 once the licence is accepted (HF_TOKEN)
python -m sinemart.alt m2 ../data/dinov2.npz ../data/gender.csv ../data/out/alt_m2.sql
npx wrangler d1 execute sinemart-talent --remote --file ../data/out/alt_m2.sql
```

Every retrain also fits Model 2 on the same picks and split; the scores page
can switch to it. Battles always use Model 1.

Loaded now (key = scores page name, by key order): m2 dinov2, m3 farl,
m4 arcface, m5 siglip2l, m6 clipl, m7 pecore. Add another the same way with
the next key. FaRL and ArcFace (InsightFace buffalo_l) weights are for research
or non-commercial use only; check licences before using them beyond the prototype.

### 6. Golden ratio (fixed face-geometry rule)

```bash
curl -L -o ../data/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python -m sinemart.facegeom ../data/crops ../data/face_landmarker.task ../data/gender.csv ../data/geo
# apply ../data/geo/geometry.sql (tables geometry, face_points and the two golden-ratio models), then retrain
```

Faces are straightened in 3D and the remaining head-angle effect is regressed
out before measuring golden ratio, Pallett, symmetry and averageness
("Golden ratio · straightened", key `geo`). The same measures on the points as
they sit in the photo give "Golden ratio · original" (key `geo_raw`) for
comparison.

## Tests

```bash
cd pipeline && python -m pytest -q     # heads, sampler, band check, crop geometry, manifest
cd worker && npx tsc --noEmit
```
