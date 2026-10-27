# Scorer ("Nilai Foto")

Score a photo of someone who isn't in the app, with the live models, without
storing it. Built 2026-10-27 at Muajir's request ("upload a new photo, pick
Model 6 straight line, get a score").

```
browser ─► TanStack Start on Workers (src/)        ─► Container (container/, Python)
           /api/score: checks the rater, sends each    sinemart.crop + sinemart.embed:
           photo to the container, then applies         face crop + 768 numbers per model
           R2 models/<key>/taste.json and the
           model's current weights from D1, and ranks
           the result against the applicants
```

- **Same numbers as the pipeline.** The container runs `pipeline/sinemart`
  with the library versions that embedded the applicants (pinned in
  `container/Dockerfile`) and the same JPEG 90 round trip, so a bc photo gets
  the vector the pipeline gave it (checked to ~1e-7).
- **Always the current model.** Weights are read from D1 on every request
  (`model.params_line`, `alt_models.params_line`), so a retrain shows up at
  once. The taste space comes from R2 `models/<key>/taste.json`, written by
  `sinemart.publish` / `sinemart.alt` (or `sinemart.project --taste`).
- **Read-only, nothing kept.** No writes to D1 or R2; photos live in memory
  only (the container writes none to disk and logs no requests). Sign-in
  uses a name that already exists on the picking page; the scorer never
  creates raters.
- **Models:** Model 6 (the leaderboard's ranker since 2026-10-27, shown
  first) and Model 1 (`MODELS` in `src/lib/score.ts`,
  `SCORER_MODELS` in `container/server.py`). Adding one: put its
  `sinemart.embed` name in both, upload its `taste.json`, rebuild.

## Run and deploy

```bash
cd scorer && npm install
npm run dev        # local: needs Docker; builds the container image
npm run typecheck
npm run deploy     # builds the image (~5 GB, weights baked in) and pushes it
```

The container is a custom instance (2 vCPU, 6 GiB) that sleeps after 30
quiet minutes; the first photo after that waits while ~3 GB of weights load.

Live since 2026-10-27: <https://sinemart-scorer.qalam.workers.dev>, image
`registry.cloudflare.com/67f961331110b81774851ee4f54349b9/sinemart-scorer:v1`.

Behind an HTTPS proxy that re-signs traffic (e.g. a Claude cloud session),
`npm run deploy` can't build the image. Build it yourself, push it, and
deploy a copy of this folder whose `wrangler.jsonc` has `image` set to the
pushed reference (and no `image_build_context`):

```bash
# from the repo root; BASE only if Docker Hub rate-limits (429)
docker build --secret id=ca,src=<proxy-ca.crt> --build-arg BASE=mirror.gcr.io/library/python:3.13-slim \
  -f scorer/container/Dockerfile -t sinemart-scorer:v2 .
cd scorer && npx wrangler containers push sinemart-scorer:v2   # retry if a big layer times out; done layers are skipped
```

Gotchas: the registry times out committing a layer much over ~1.7 GB, so
the weights are one layer per model; right after a deploy the container app
"provisions" for a few minutes before the first photo can be scored; the
first photo after a sleep takes ~50 s, later ones ~4 s.
