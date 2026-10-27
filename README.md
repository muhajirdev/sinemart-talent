# Sinemart talent screening

Sinemart's casting calls get tens of thousands of sign-ups. This app learns the
talent team's taste from quick A/B picks ("Mana yang lebih menarik?") and ranks
every applicant on one score, so the team can drop the clear "no" faces and
look closely only at the top few hundred.

- **Picking page:** <https://sinemart-talent.qalam.workers.dev> (type a name, pick)
- **Lab (results):** <https://sinemart-talent.qalam.workers.dev/results>

## Start here

| Read | For |
|---|---|
| [docs/context.md](docs/context.md) | The goal, how the model works, every decision and why, the data and its rules, current status and open items |
| [docs/pipeline.md](docs/pipeline.md) | The whole pipeline end to end: data in, face crops, embeddings, gender, golden ratio, face parts, picks, the straight-line scorer, battles, comparison models, retraining, the Lab, D1/R2 |
| [docs/howto.md](docs/howto.md) | Deploy, query the live data, retrain, mark bad photos and duplicates, load new applicants, add a model, tests |
| [AGENTS.md](AGENTS.md) | Rules for anyone (human or AI agent) changing this repo |

## In one picture

```
Drive / sign-up sheet photos ──► crop · align · grey · gate ──► frozen image embedding per person
                                                                       │ (pipeline/, Python, run locally)
                                                                       ▼
team A/B picks ──► Worker ──► D1 ──► cron retrains every 10 picks ──► straight-line score + next battles
 (worker/public/index.html)            (worker/src/retrain.ts)          Lab: worker/public/results.html
```

- One universal score (not per role). Cewek and cowok are scored and battled
  separately.
- The ranker is a straight line (Bradley–Terry) on the face embedding; seven
  other embeddings, a hills (clusters) shape and two golden-ratio rules run
  alongside on the same picks as comparisons.
- Bad photos and duplicate sign-ups are marked as noise, never deleted.

## Layout

```
pipeline/            Python: photos → crops → embeddings → SQL for D1
  sinemart/manifest.py   2021 Drive listing → photo manifest
  sinemart/sheet.py      2026 sign-up sheet → photos + manifest (ids 260000+row)
  sinemart/crop.py       face detect, align, grey background, quality gate
  sinemart/embed.py      frozen embeddings, averaged per person (MODELS list)
  sinemart/gender.py     zero-shot cewek/cowok guess
  sinemart/publish.py    taste space + seed.sql for D1
  sinemart/alt.py        load a comparison model (m2, m3, …)
  sinemart/facegeom.py   golden-ratio rules (straightened faces)
  sinemart/parts.py      Model 8: eyes / nose / mouth / jaw embeddings
  sinemart/bt.py, train.py   offline heads and report (analysis only)
worker/              Cloudflare Worker + D1 + R2
  src/index.ts           API routes, pair serving, picks, noise, cron entry
  src/retrain.ts         retrain: fit, held-out check, rescore, refill battles
  src/hills.ts           the line, hills, held-out split, pair selection
  src/map.ts             territory map for the Lab
  public/                index.html (picking), results.html (Lab)
  schema.sql, migrate.sql
```

## Quick commands

```bash
cd pipeline && python -m pytest -q        # tests
cd worker && npx tsc --noEmit             # typecheck
cd worker && npx wrangler deploy          # deploy (ask Muajir first; merging does not deploy)
```
