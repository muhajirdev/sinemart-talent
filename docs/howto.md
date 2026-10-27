# How-tos

Every task you are likely to need, with the gotchas that bit us. Background
and decisions are in [context.md](./context.md); how each step works is in
[pipeline.md](./pipeline.md).

- [Setup](#setup)
- [Deploy the Worker](#deploy-the-worker)
- [Look at the live data (read-only)](#look-at-the-live-data-read-only)
- [Retrain](#retrain)
- [Mark bad photos and duplicates](#mark-bad-photos-and-duplicates)
- [Load new applicants](#load-new-applicants)
- [Test the model on a batch outside the picks](#test-the-model-on-a-batch-outside-the-picks)
- [Add a comparison model](#add-a-comparison-model)
- [Golden ratio and face parts](#golden-ratio-and-face-parts)
- [Change the schema](#change-the-schema)
- [Offline training report](#offline-training-report)
- [Tests](#tests)

## Setup

```bash
cd worker && npm install                     # Worker: TypeScript, wrangler 4
cd ../pipeline && pip install -r requirements.txt   # Python 3.11+: torch, transformers, opencv, mediapipe
```

Cloudflare (all pinned in `worker/wrangler.toml`):

| What | Value |
|---|---|
| Account | "Muhajir", `67f961331110b81774851ee4f54349b9` |
| Worker | `sinemart-talent` → <https://sinemart-talent.qalam.workers.dev> |
| D1 | `sinemart-talent`, id `aaf71364-974f-4ce8-a59a-c4122b30d26c`, binding `DB` |
| R2 | bucket `sinemart-talent`, binding `IMAGES` (`crops/…` face crops, `photos/…` whole photos) |
| Secret | `ADMIN_TOKEN` (guards `/api/export.csv`); set with `wrangler secret put ADMIN_TOKEN`, never commit it |
| Cron | every minute (`maybeRetrain`) |
| Limits | `cpu_ms = 300000` (paid plan; a retrain of 840 people × 8 models takes seconds of CPU) |

Static pages are in `worker/public/` (`index.html` picking, `results.html`
the Lab, `phi.js` golden-ratio overlay), served as Worker assets.

## Deploy the Worker

Deploying changes what the talent team is using: **get Muajir's OK first.**

```bash
cd worker
npx tsc --noEmit          # must pass
npx wrangler deploy
```

There is no CI and no auto-deploy: merging a PR does **not** deploy. PRs merge
into the default branch `claude/project-thread-60dxh0`.

After deploying, open the picking page and the Lab once and check a pair loads
and the scores show.

## Look at the live data (read-only)

The API needs a rater name, so for quick checks query D1 directly:

```bash
npx wrangler d1 execute sinemart-talent --remote --command "SELECT COUNT(*) FROM picks WHERE winner_id IS NOT NULL OR tie = 1 OR neither = 1"
```

Useful queries:

```sql
-- model health: picks trained on, held-out agreement of the ranker
SELECT n_picks, heldout_line, heldout_hills, heldout_one, n_test, trained_at FROM model;
SELECT key, heldout_line, heldout, heldout_one, heldout_fixed FROM alt_models ORDER BY key;
-- decided picks per rater (skips have winner_id NULL and tie = neither = 0)
SELECT rater, COUNT(*) FROM picks WHERE winner_id IS NOT NULL OR tie = 1 OR neither = 1 GROUP BY rater ORDER BY 2 DESC;
-- people never in a decided battle
SELECT gender, COUNT(*) FROM applicants a WHERE noise IS NULL AND NOT EXISTS (
  SELECT 1 FROM picks p WHERE (p.left_id = a.id OR p.right_id = a.id)
    AND (p.winner_id IS NOT NULL OR p.tie = 1 OR p.neither = 1)) GROUP BY gender;
```

Count **decided** picks, never raw `picks` rows: most rows are skips.

Picks as CSV (decided winners only): `GET /api/export.csv?admin=<ADMIN_TOKEN>`.

Endpoints (all need a signed-in rater: the token from `POST /api/login {"name": …}`
sent as the `x-rater-token` header or `?t=<token>`): `/api/pair`, `/api/pick`, `/api/undo`, `/api/me`,
`/api/results?mode=&rater=`, `/api/map`, `/api/battles/<id>`, `/api/points/<id>`,
`/api/geometry`, `/api/gender`, `/api/noise`, `/api/check`, `/api/raters`,
`/api/retrain`, `/api/reset`, `/img/<id>[/<n>]`.

> `/api/reset` deletes **every pick**. Any signed-in name can call it (the
> Lab's "Reset all scores" button). Never call it without Muajir asking.

## Retrain

Nothing to do normally: the cron retrains when 10+ new usable picks are in.

- Force it now: Lab → "Retrain now", or `POST /api/retrain`.
- Force it on the next cron run (e.g. after a direct D1 change):
  `UPDATE model SET n_picks = 0 WHERE id = 1;`
- Reloading a comparison model (`sinemart.alt`) wipes its fit; `/api/results`
  retrains automatically when it sees an untrained model.

What a retrain does (`worker/src/retrain.ts`): loads taste-space vectors for
all non-noise people and all same-gender, non-repeat picks; fits the line,
hills and one-cluster on 80% of people and scores the held-out 20%; refits
everything on all picks; writes `applicants.score / score_std / look`;
**replaces** the open `pair_queue` with fresh battles; fits every
comparison model the same way.

## Mark bad photos and duplicates

One person, from the app:

- Picking page → "Foto bermasalah" under a face → "Bukan foto wajah / orang
  lain" (noise) or "Bukan cewek/cowok" (move gender).
- Lab → open a face → "wrong? move to …" for gender; "Marked as bad photos"
  section at the bottom to restore.

Many people at once (e.g. duplicate sign-ups), **after Muajir approves the
list**:

```sql
UPDATE applicants SET noise = 'duplicate' WHERE id IN (260834, 260877 /* … the extras, keep one id per person */);
DELETE FROM pair_queue WHERE done_at IS NULL AND (left_id IN (…) OR right_id IN (…));
UPDATE model SET n_picks = 0 WHERE id = 1;   -- retrain on the next cron run
```

Noise people leave battles, training and leaderboards; their picks stay in
`picks` and come back if restored (`UPDATE applicants SET noise = NULL …`).
Finding duplicates: compare face crops with a face-recognition embedding and
look at the close pairs by eye; exact re-uploads also show as identical
embeddings.

## Load new applicants

Loading is a one-off job per source; it touches personal data, so read the
rules in [context.md](./context.md#data-and-personal-data-rules) first.

### 1. Get photos and a manifest

2021-style Drive folders (one folder per applicant):

```bash
rclone config        # Google Drive remote "gdrive", scope drive.readonly
mkdir -p data && rclone lsjson -R --drive-shared-with-me "gdrive:Hasil Screening COI 2021" > data/listing.json
cd pipeline
python -m sinemart.manifest ../data/listing.json ../data/manifest.csv
rclone copy --drive-shared-with-me "gdrive:Hasil Screening COI 2021" ../data/photos \
  --include "*Foto-*.{jpg,jpeg,JPG,JPEG,png}"
```

2026-style Google Form sheet (a Drive link per person; ids = 260000 + row):

```bash
python -m sinemart.sheet <SHEET_ID> ../data/2026     # photos, manifest.csv, labels.csv (local only)
```

A new source needs a new id range that can't collide (2021 ids go up to
~17k; 2026 uses 260000+). Add a new `ID_BASE` rather than reusing one.

### 2. Crop, embed, guess gender

```bash
curl -LO https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
python -m sinemart.crop ../data/manifest.csv ../data/photos ../data/crops \
  --yunet face_detection_yunet_2023mar.onnx --seg selfie_segmenter.tflite   # add --grid for photo grids (2026)
python -m sinemart.embed ../data/crops siglip2 ../data/siglip2.npz
python -m sinemart.gender ../data/crops ../data/gender.csv
```

Crops are 512×512, eyes level and fixed, grey background. The gate rejects no
face, tiny face, blur, bad exposure, non-frontal pose and several faces;
rejects go to `crops/rejected/` with reasons in `crops/quality.csv`. Look
through a few hundred before trusting the thresholds.

**Check whole photos before uploading them.** The 2026 "other photos" held KTP
cards and screenshots and had to be removed. If in doubt, upload face crops
only and skip `photos/`.

### 3. Publish to D1 and R2

```bash
python -m sinemart.publish ../data/siglip2.npz ../data/crops/quality.csv ../data/out ../data/gender.csv
npx wrangler d1 execute sinemart-talent --remote --file ../data/out/seed.sql
rclone copy ../data/crops r2:sinemart-talent/crops --max-depth 1 --include "*.jpg"
rclone copy ../data/crops/photos r2:sinemart-talent/photos      # only after checking them
```

Gotchas:

- `publish` computes the taste space (principal directions + look centres)
  from the vectors it is given and writes it to `model.init`. Adding people to
  a live app means re-publishing **everyone** (old + new) into one shared
  taste space; picks are kept and the next retrain uses them.
- From a Claude cloud session, `wrangler d1 execute --file` is blocked (it
  uploads through an R2 host the proxy refuses). Split the SQL into ~60-line
  chunks and run each with `--command`. R2 uploads go fastest through the
  Cloudflare API (`PUT /accounts/{id}/r2/buckets/sinemart-talent/objects/{key}`)
  in parallel.
- Every comparison model must be re-run on the new people too
  (see below), or they get their group's average.
- Outside labels (like the 2026 shortlist) go into `outside_labels`
  (`applicant_id, source, label`) and show only as aggregates in `/api/check`.
  Never show them to raters.

## Test the model on a batch outside the picks

For a batch that should be scored but never picked (the "bc" showcase
photos, 2026-10-27). It keeps the live taste space and every score as they
are; re-publishing would refit both.

1. Crop the batch with its own id range (bc: `900000 + post number`) and
   `stage`, embed it and guess gender (steps 1–2 above).
2. Re-make the live taste space: download the live crops from R2 `crops/`
   (skip the old plain `<id>.jpg` files; the live load used only
   `<id>-<kind>.jpg`), embed them with the same model, export
   `SELECT id, z FROM applicants` as JSON, then
   `python -m sinemart.project live.npz batch.npz live_z.json batch_z.json`.
   It must report max |dz| ≈ 0; otherwise the crops don't match the load.
3. Score = `w · z` with `w` from `model.params_line` (the Lab shows
   `sigmoid(score − gender mean)` as 0–100).
4. Upload the crops to R2 `crops/` and insert `applicants` rows with
   `test = 1` and `source = stage = '<tag>'`. `test` keeps them out of
   battles, training, the map and the screening check; the Lab's Sources
   checklist offers them, unticked by default. Don't use `noise` for this:
   that is for bad photos and lists them under "Marked as bad photos".
5. For a comparison model, the same with `alt_z WHERE key = 'mN'` and its
   `params_line`; insert `alt_z` rows for the batch (retrain ignores ids
   that aren't in training).

Gotchas: retrains never touch test rows, so the stored `score` is from the
model at load time; the Lab re-scores test batches with the current line on
every load (other models need their `alt_z` rows). `sinemart.publish`
keeps test rows but builds a new taste space, so after a re-publish their
`z` is stale: re-run step 2 onwards.

## Add a comparison model

1. Add the Hugging Face id to `MODELS` in `pipeline/sinemart/embed.py`
   (and its loading code if it isn't a standard vision model).
2. Embed the same crops: `python -m sinemart.embed ../data/crops <name> ../data/<name>.npz`
3. Build the SQL with the **next free key** (m9, m10…; the Lab names models
   "Model N" by key order):
   `python -m sinemart.alt m9 ../data/<name>.npz ../data/gender.csv ../data/out/alt_m9.sql`
4. Apply it to D1 (chunked `--command` from a cloud session).
5. Open the Lab: it retrains the new model on the existing picks and shows its
   held-out agreement next to the others.

Keep model and vendor names out of the UI: the app only ever shows "Model N".
Check the licence of the weights (FaRL and ArcFace are non-commercial).

To make a comparison model the ranker, change what `retrain()` fits as `line`
and what `nextPair` serves. That is a decision for Muajir (rule: beat the line
by >5 points held-out at ~1,000 picks).

## Golden ratio and face parts

```bash
curl -L -o ../data/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Golden ratio (fixed rules geo + geo_raw, tables geometry and face_points)
python -m sinemart.facegeom ../data/crops ../data/face_landmarker.task ../data/gender.csv ../data/geo
#   apply ../data/geo/geometry.sql
# Model 8, face parts
python -m sinemart.parts ../data/crops ../data/face_landmarker.task ../data/parts.npz
python -m sinemart.alt m8 ../data/parts.npz ../data/gender.csv ../data/out/alt_m8.sql
```

MediaPipe needs `libegl1` on headless Linux. Faces turned more than
`MAX_POSE_DEG` are left out of the geometry.

## Change the schema

`worker/schema.sql` is the full schema for a new database. For the live one,
append the `ALTER TABLE` to `worker/migrate.sql` with a dated comment and run
just that statement with `--command`. Update `/api/reset` if the new column
holds trained state.

## Offline training report

`python -m sinemart.train ../data/siglip2.npz ../data/picks.csv ../data/out`
fits the Python heads (hills, A line, B look-centred, C MLP) on exported picks
and writes `out/report.json` (held-out, per-look, repeat ceiling, 10-band
check). It is for analysis only: **don't apply its `update.sql` to the live
app**, it would overwrite the live straight-line scores with hills scores
(the Worker retrains itself).

## Tests

```bash
cd pipeline && python -m pytest -q    # heads, sampler, band check, crop geometry, manifest, parts
cd worker && npx tsc --noEmit
```
