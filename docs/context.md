# Context: why this exists and what has been decided

Read this before changing anything. It is the memory of the project: the goal,
how the model works, every decision Muajir (the owner) has made and why, where
the data comes from, and where things stand. How-tos are in
[howto.md](./howto.md).
Every step of the pipeline in detail (crop rules, taste space, scoring maths,
battle choice, held-out check, tables) is in [pipeline.md](./pipeline.md).

## The goal

Sinemart (an Indonesian TV and film production house) runs open casting calls
with tens of thousands of sign-ups (~50k). The talent team cannot look at all
of them. This app learns the team's own taste from quick A/B picks and ranks
every applicant on one "screen-worthy" score, so the team can **drop the clear
"no" faces** and spend their time on the top few hundred.

Out of scope: legal review, contracts, contacting people, per-role casting.

## How it works

```
photos (Drive 2021, 2026 sign-up sheet)
  └► crop: find the face, level the eyes, 512×512 on grey, quality gate
      └► frozen image embedding per face crop, averaged per person
          └► small "taste space" (top principal directions) stored in D1

team A/B picks (picking page ► Worker ► D1 picks)
  └► Worker cron retrains every 10 new picks
      └► straight-line score per person + next battles (pair_queue)
          └► Lab page: leaderboard, top vs bottom, map, model comparison
```

- **Picks.** A rater sees two faces of the same gender and taps "Pilih yang
  ini" under the more attractive one. Other answers: "Sama saja" (tie, trained
  as a half win), "Keduanya kurang menarik" (neither: both are a clear no),
  "Lewati" (skip, not trained) and "↶ Batalkan" (undo the last answer within
  30 minutes).
- **The ranker is a straight line** (Bradley–Terry) **on Model 6** (since
  2026-10-27; `RANKER` in `worker/src/retrain.ts`):
  P(a beats b) = sigmoid(w·z(a) − w·z(b)), fitted on every pick from every
  rater (`worker/src/hills.ts`, class `Arrow`) in Model 6's taste space. "Neither" picks teach a
  per-gender "no" line `tau`: both faces should score below it. The spread of
  3 bootstrap fits gives each person's `score_std`.
- **Battles** are chosen after every retrain (`groupedPairs`, `nextPairs`):
  70% uncertain pairs (half within the same cluster), 20% among the current
  top, 10% random; sides shuffled; never across genders. About 1 in 20 battles
  re-asks a pair the rater already answered (kind `repeat`); repeats measure
  how consistent people are and stay out of training. When the queue is empty
  the app serves random same-gender pairs.
- **Held-out check.** Each retrain holds out 20% of *people* (not picks) and
  reports how often each model gets the held-out picks right. That number is
  the main health metric.
- **Comparison models.** Same picks, other embeddings or other shapes. They
  never drive battles; they exist to see whether something beats the ranker.
  Model 1's own line is the Lab's "Model 1 · Straight line" (`?model=1:line`).
  - Shapes on Model 1: "Many clusters" (hills: one peak per look, score falls
    off with distance from the peak) and "One cluster" (one hill per gender).
  - Models (UI shows only "Model N", by key order): Model 1 = SigLIP 2 base
    (the ranker until 2026-10-27; its hills still name the clusters), m2 DINOv2 base, m3 FaRL, m4 ArcFace (InsightFace buffalo_l),
    m5 SigLIP 2 so400m, m6 OpenAI CLIP ViT-L/14, m7 PE-Core-L-14-336,
    m8 face parts (eyes, nose, mouth, jaw each embedded with SigLIP 2 base,
    `pipeline/sinemart/parts.py`).
  - Fixed rules, never trained: "Golden ratio · straightened" (key `geo`) and
    "Golden ratio · original" (key `geo_raw`) from
    `pipeline/sinemart/facegeom.py` (golden ratio, Pallett, symmetry,
    averageness, after straightening the head in 3D).
  - FaRL and ArcFace weights are research / non-commercial only. Check
    licences before anything beyond the prototype.
- **Per-rater taste.** The Lab "Taste of" switch fits one rater's own model
  on whichever model and shape is selected (cached in `rater_models` under the
  name for Model 1 many clusters, `name<TAB>mode` otherwise; min 5 picks). The rater agreement table shows how
  well the team model and each other rater predict that rater.

## Decisions (with dates) — don't undo these without asking Muajir

| Date | Decision | Why |
|---|---|---|
| 2026-10-12 | **One universal score**, not one per role. Per-role tweaks can sit on top later. | Simpler pilot; every pick feeds one model. |
| 2026-10-12 | Train only on the team's own picks. MEBeauty / SCUT-FBP are research-only and don't transfer. | Licence and taste. |
| 2026-10-12 | **Cewek and cowok separate**: same-gender battles only, per-gender leaderboards. Gender never feeds the score. | Comparisons across genders mean little. |
| 2026-10-12 | Gender is a zero-shot guess (no gender column in the data), fixable by hand ("wrong? move to …" on the Lab, or "Foto bermasalah" on the picking page). Manual fixes survive reloads (`gender_src = 'manual'`). | |
| 2026-10-12 | **Applicant-centric**: several photos per person, every usable face crop embedded and averaged. Raters see the face crop first and swipe for whole photos. | One odd photo moves a score less. |
| 2026-10-12 | **Name-only login**, no password or team code. Same name (any case) on any phone = same rater. | Lowest friction for the team. |
| 2026-10-12 | The app is public but kept out of search engines (robots.txt, noindex meta, `X-Robots-Tag` on everything incl. images). | |
| 2026-10-12 | **Never name the technology in the app** (no Google, SigLIP, DINO, CLIP, Bradley–Terry, embeddings, Cloudflare…) in UI text, titles, meta or client-visible API responses. Say "model" / "skor" / "Model N". | Owner's rule. |
| 2026-10-13 | Golden ratio kept as fixed comparison rules (straightened and original). The warped/straightened *face photos* were tried and **removed** ("looks bad"); don't re-add. Picking-page golden-ratio numbers and lines are toggles, default off, so raters aren't steered. | |
| 2026-10-21 | 2026 sign-ups loaded next to 2021 in one shared taste space. The 2026 SORTIR shortlist is **never shown to raters**; only aggregate checks (`/api/check`, table `outside_labels`). | Don't bias the pickers. |
| 2026-10-22 | **All 2026 whole photos removed** from `photos`: some were KTP (ID cards with NIK and address) and Instagram screenshots. 2026 people are judged by face crop only. Don't re-add 2026 whole photos without filtering documents out. (R2 objects still exist.) | Personal data exposure. |
| 2026-10-22 | **The straight line ranks and picks battles**; hills / one cluster became comparisons. Revisit at ~1,000 picks: switch only if another model beats the line by more than 5 points held-out. | The line won held-out; ±4 points is noise at this size. |
| 2026-10-22 | Retrain runs from a **1-minute cron**, only when 10+ new picks are in (or `model.n_picks = 0`). | Retraining inside a request was cut off at ~30 s and scores froze. |
| 2026-10-22 | Shortlist page and Panggil/Skip buttons removed. The Lab is the results page. | |
| 2026-10-22 | Question copy: "Mana yang lebih menarik?" with a separate "Pilih yang ini" button and undo. | |
| 2026-10-22 | Bad data is marked as **noise**, not deleted: `applicants.noise` = who marked it (or `'duplicate'`). Noise people drop out of battles, training and leaderboards; their picks stay stored; restorable from the Lab. 15 exact duplicate 2026 sign-ups marked `duplicate`. | Reversible. |
| 2026-10-25 | "Keduanya kurang menarik" (neither) added. Trained as "both below the gender's no line", not as a tie. | The goal is spotting clear no's. |
| 2026-10-27 | **Model 6's straight line ranks and picks battles** (was Model 1's). Model 1's line, hills and one cluster stay as comparisons; Model 1's hills still name the clusters (`look`). | Muajir: "can we try model 6". Offline, 30 held-out splits by applicant: 74.8% vs 64.0% at 552 comparisons, and 71.9% vs 60.6% with the "neither" burst, ahead in 30 of 30 both times. Dropping aa / ivan / Test didn't help (+0.8, noise); combining models, other regularisation, fewer dimensions or a line per gender were all worse. |
| 2026-10-27 | Lab leaderboard "No battles yet (N)" filter (`?battles=none`). | Find people never shown. |
| 2026-10-27 | **"bc" test batch**: 201 faces from an Instagram showcase account (R2 `talent-applicant/bc/`, expandra account) loaded as a model test only: ids 900001+, `source = 'bc'`, `test = 1`, so they are never in battles or training. Projected into the live taste space (`sinemart.project`), not re-published. | Muajir: "test our model against this new bucket"; "no need to be in listing for pickings". |
| 2026-10-27 | **Scorer app** (`scorer/`, "Nilai Foto"): score photos of someone not in the app with Model 1 and Model 6, read-only, photos never stored. TanStack Start on Workers for the page and maths; a Cloudflare Container runs the Python face pipeline (pinned versions, so numbers match the pipeline). Taste spaces now saved in R2 `models/<key>/taste.json`. Live at <https://sinemart-scorer.qalam.workers.dev> (separate Worker `sinemart-scorer`; the picking app is untouched). | Muajir: "upload a new photo … pick model 6 with straight line … give score"; "run on tanstack server", Cloudflare Containers. |
| 2026-10-27 | The Lab shows **comparisons** (winner or tie) as the headline count, with "neither" and skips beside it, and the rater table shows every rater's winners / ties / neither / skips and median seconds per answer (under 1.5 s highlighted). | "Decided" mixed in a 1,363-answer "neither" burst and read as 2,348 picks. |
| 2026-10-27 | Lab leaderboard **Sources** checklist (2021, 2026, bc). From `applicants.source`; test batches (`applicants.test = 1`) are off by default; ticked, they rank among applicants, scored live by Model 1's line, and the 0–100 scale stays relative to applicants only. Test batches are a column of their own, not `noise` (first loaded as `noise = 'bc'`, which put all 201 under "Marked as bad photos"). | Muajir: "show applicant sources, like checklist"; "we can choose to include this bc sources". |
| 2026-10-28 | Lab: tapping a person opens a **right-side panel** (photos, score, battle history; full width on phones) instead of a full-screen view; tapping a rater's name opens their **pick history** there (`/api/rater-picks`). **Rater agreement** is collapsed by default. **Raters checklist** on the ranker view: untick raters to preview the ranking and held-out accuracy without their answers (`/api/results?without=`); a preview only, never saved, battles untouched. | Muajir: "on click talent there, open on right sidebar", "option to exclude a rater, I wanna see if anyone introduced lots of noise", "it should be like a checklist instead". |
| 2026-10-28 | "Taste of" works with every model and shape (default straight line on Model 1, like the team view). The agreement table still uses each rater's many-clusters model. | Muajir asked to compare raters on any model. |
| 2026-10-28 | Side panel slides in/out and the page glides aside (no jump), and its width is drag-resizable (remembered per device). **Elo** per person from wins/losses only (start 1500, K 32, "sama saja" = draw, "neither"/skips/repeats ignored; respects the Raters checklist; in "Taste of" only that rater's picks); leaderboard can sort by Elo (`?sort=elo`). | Muajir: "its patah2 hard to see … animate it", "right sidebar should be resizable", "i also want an elo please (not model prediction), just based on their lost / win". |
| 2026-10-28 | Lab **Taste match** card (collapsed, loads when opened, `/api/taste-match`): each rater with 5+ picks gets their own straight line on the ranker's model (fresh fit on their picks and "neither" only, not cached); every pair scored by how often one's line picks the face the other picked (both ways, pooled; 50% = coin flip, below = clash), plus "same order" (score correlation per group). Matrix ordered so similar raters sit together, "Most alike" / "Most clashing" lists (raters with 30+ picks), diagonal = own 5-fold consistency (flag under 60% as noise). Read-only. | Muajir: "who's taste are bertabrakan and more similar to each other", "the analysis should be shown in the web". |
| 2026-10-28 | "Taste of" a rater: the side panel's battle history shows that rater's battles first (switch to everyone's), and the rater's scores are rescaled to the team ranking's spread per group (same order), so the 0–100 scale reads like the team view instead of the top hitting 100. | Muajir: "on click the history should be siska's history, even though we can see full other history as well"; "why the #1 score becomes 100 when choosing siskas taste" (one person's few picks fit perfectly, so their line was very steep). |


## Data and personal data rules

- **2021: "Hasil Screening COI 2021"** (Google Drive, shared read-only with
  Muajir by Sinemart's casting account). Folder per applicant under
  `Foto _ Video Peserta/01- Lolos SELEKSI TAHAP AWAL/<id> - <NAME>/` with
  Closeup, Fullbody1/2 photos and a video. Only ~170 round-1 passers are there,
  not the full ~17k sign-ups. A finalist deck (`HASIL SELEKSI AWAL/…pptx`) and
  `Hasil Screening COI Season 1.xlsx` are sanity labels only. Loaded from a zip
  Muajir downloaded: 189 people with a usable face. Ids = the Sinemart
  applicant number.
- **2026: "SINEMART MENCARI BINTANG 2026! [RAW & SORTIR]"** (Google Form
  sheet, owned by the talent management team). RAW = every sign-up (~1,050),
  SORTIR = the team's shortlist (~123). Column "Foto profile (GRID)" holds a
  Drive link per person. 651 people with a usable face loaded; ids =
  `260000 + RAW row` (`pipeline/sinemart/sheet.py`, `ID_BASE`). Ask Muajir for
  the sheet id; it is deliberately not in this repo.
- Other sources mentioned by the client (not loaded): "Yang Mendaftar Talent
  Search LNZ" sheet (mostly social links, no photos), a 12-photo reject folder
  from talent management, a Canva board.

Rules:

- **Never copy names, phone numbers, emails, addresses, NIK or social handles**
  into D1, R2, this repo, logs or chat. Code reads only the photo column and
  the row number; everything downstream sees `applicant_id` only.
- Applicants include minors (14+). Keep everything private to the team.
- Don't share, move or modify the client's Drive files.
- Local working data (`data/`, crops, labels) is git-ignored. Keep it that way.

## Where things stand (2026-10-27)

- Live: <https://sinemart-talent.qalam.workers.dev> (picking) and `/results`
  (the Lab).
- 840 people (189 from 2021, 651 from 2026); 17 marked noise. Plus 201 "bc"
  test faces (`test = 1`, ids 900001+), scored on the Lab only.
- 846 comparisons (703 with a winner, 143 ties) plus ~1,600 "neither" and
  ~1,400 skips (late 2026-10-27). Count **comparisons** for milestones:
  "decided" includes "neither", and 1,363 of those came from one burst
  (open items).
- Held-out agreement at 520 picks (174 held-out picks; ±4 points is noise):
  Model 1 line 65% (the ranker then), hills 57%; m6 CLIP L/14 line 76%,
  m3 FaRL 73%, m7 67%, m4 65%, m5 65%, m2 60%, m8 parts 56%;
  golden ratio straightened 58%, original 54%.

Open items:

- **"Neither" burst** (2026-10-27): "OKI Hartanto (2)" answered "Keduanya
  kurang menarik" 1,363 times in 34 minutes (15:39–16:13 UTC, median ~0.8 s),
  covering 775 of 823 people, and picked a winner only 61 times. It trains
  the "no" line but says little about who is better. Ask OKI; if it wasn't
  real, leave that burst out of training (keep the rows). Probably the same
  person as "Oki Hartanto" (the "(2)" makes a separate name).
- **Recheck the ranker at ~1,000 comparisons** (not "decided" answers) (weekly check from 2026-10-29).
  Model 6 ranks since 2026-10-27; switch again only if another model beats it
  by >5 points held-out. Remember FaRL (m3) is non-commercial.
- **Near-duplicate scan** (2026-10-27): 6 groups / 15 extra sign-ups found by
  face match (one man signed up 11 times). Awaiting Muajir's OK to mark them
  `noise = 'duplicate'` (and possibly move their picks to the kept id).
- **Reminder**: tell the client's "Sinemart AI Projects" group that the team
  hasn't reached 1,000 picks yet (Muajir sends it himself).
- More consented applicants (see decision 2026-10-27). No decision yet.
- Elo duels for the final ~150 are not built.
- Research notes from the start of the project live in
  `muhajirdev/sinemart-dev`, branch `claude/cinema-talent-filtering-ai-s8u4rk`,
  folder `reports/`.
