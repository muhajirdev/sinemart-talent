# Rules for agents (and humans) working in this repo

Read [docs/context.md](docs/context.md) first; how-tos are in
[docs/howto.md](docs/howto.md); how every pipeline step works is in [docs/pipeline.md](docs/pipeline.md). The owner is Muajir (GitHub `muhajirdev`); he
often writes Indonesian, so reply in the language he uses.

## Never


- **Never commit secrets** (`ADMIN_TOKEN`, API tokens, `.dev.vars`) or the
  client's sheet ids and Drive links.
- **Never name the technology in the app**: no Google, SigLIP, DINO, CLIP,
  FaRL, ArcFace, Bradley–Terry, embeddings, Cloudflare… in UI text, page
  titles, meta tags or client-visible API responses. Say "model", "skor",
  "Model N".
- Never show outside labels (e.g. the 2026 shortlist) to raters, per person.
  Aggregates only.
- Never re-add 2026 whole photos without filtering out ID cards and
  screenshots.
- Never call `/api/reset` or delete picks / applicants. Bad data is marked
  `applicants.noise`, which is reversible.

## Ask Muajir first

- Deploying (`wrangler deploy`): the team uses the live app. Merging a PR does
  not deploy.
- Bulk D1 writes on live data (marking duplicates, moving picks, reloading a
  model, schema changes).
- Changing what ranks or picks battles (the straight line, Model 1). The rule:
  switch only if another model beats the line by >5 points held-out at
  ~1,000 picks.
- New pages, buttons, steps or flows. Propose them before building; confirm
  your reading of his idea against his actual words.

## How to work

- Branch from the default branch `claude/project-thread-60dxh0`, open a PR
  there, merge when tests pass.
- Before pushing: `cd pipeline && python -m pytest -q` and
  `cd worker && npx tsc --noEmit`.
- Write code like the surrounding code: short plain-English comments that say
  *why*, dated when they record a decision ("2026-10-22: …").
- Keep `worker/schema.sql` (fresh database) and `worker/migrate.sql` (dated
  `ALTER`s for the live one) in step.
- When you make or learn a decision, add it to the decisions table in
  `docs/context.md`, and update "Where things stand" when numbers change.
- Quote **decided** picks (winner, tie or neither), never raw `picks` rows;
  most rows are skips. Held-out differences under ~5 points are noise at the
  current size.
- UI copy is Indonesian and casual ("Pilih yang ini", "Sama saja",
  "Keduanya kurang menarik", "Lewati", "Foto bermasalah").
