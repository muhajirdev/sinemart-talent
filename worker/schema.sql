-- D1 schema for the A/B picking app.
-- wrangler d1 execute sinemart-talent --remote --file schema.sql

CREATE TABLE IF NOT EXISTS applicants (
  id         INTEGER PRIMARY KEY,          -- Sinemart applicant number
  image_key  TEXT NOT NULL,                -- R2 key of the aligned grey crop
  stage      TEXT,                         -- Drive selection folder, if any
  z          TEXT,                         -- JSON taste-space coordinates (from the pipeline)
  look       INTEGER,                      -- hill the face stands on (k-means look before training)
  score      REAL,                         -- hills-model score
  score_std  REAL,                         -- bootstrap spread of the score
  model      TEXT,                         -- which model wrote the score
  gender     TEXT,                         -- 'f' cewek | 'm' cowok: groups battles and hills, never the score
  gender_src TEXT,                         -- 'auto' (zero-shot guess) | 'manual' (fixed on the scores page)
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS applicants_score ON applicants (score DESC);

-- Every photo of a person (closeup, full-body 1, 2, ...), in the order raters
-- flip through them. The person's score comes from the average of all their
-- usable face crops; applicants.image_key is the main face shown first.
CREATE TABLE IF NOT EXISTS photos (
  applicant_id INTEGER NOT NULL,
  n            INTEGER NOT NULL,           -- 0 = first (closeup when there is one)
  kind         TEXT NOT NULL,              -- closeup | fullbody1 | fullbody2 | other
  image_key    TEXT NOT NULL,              -- R2 key of the whole photo, resized
  face_ok      INTEGER NOT NULL,           -- 1 = its face crop counts toward the score
  PRIMARY KEY (applicant_id, n)
);

-- Raters sign in with the team code and their name (POST /api/login), which
-- hands back a token kept on the phone. Older per-person links (?t=<token>)
-- still work. picks.rater holds the name, so every pick traces to a person.
CREATE TABLE IF NOT EXISTS raters (
  token TEXT PRIMARY KEY,
  name  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS raters_name ON raters (lower(name));

-- Each rater's personal hills model, cached until they make new picks.
CREATE TABLE IF NOT EXISTS rater_models (
  rater      TEXT PRIMARY KEY,
  n_picks    INTEGER NOT NULL,
  params     TEXT NOT NULL,
  trained_at TEXT
);

-- Pairs chosen by the trainer (70% uncertain, 20% top, 10% random).
CREATE TABLE IF NOT EXISTS pair_queue (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  left_id   INTEGER NOT NULL,
  right_id  INTEGER NOT NULL,
  kind      TEXT NOT NULL,                 -- uncertain | top | random
  batch     TEXT NOT NULL,
  served_to TEXT,
  served_at TEXT,
  done_at   TEXT
);
CREATE INDEX IF NOT EXISTS pair_queue_open ON pair_queue (done_at, served_at);

-- Every answer. winner_id NULL and tie 0 = skipped ("can't judge": bad photo,
-- someone they know).
CREATE TABLE IF NOT EXISTS picks (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  rater      TEXT NOT NULL,
  left_id    INTEGER NOT NULL,
  right_id   INTEGER NOT NULL,
  winner_id  INTEGER,
  tie        INTEGER NOT NULL DEFAULT 0,   -- 1 = "Sama saja": both about equal (winner_id NULL)
  kind       TEXT NOT NULL,                -- queue kind, or 'fallback'
  ms         INTEGER,                      -- time to answer
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS picks_rater ON picks (rater);

-- The live hills model: one row. `init` (look peaks in taste space) comes
-- from the pipeline; everything else is written by each retrain.
CREATE TABLE IF NOT EXISTS model (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  init          TEXT,
  params        TEXT,
  active        TEXT,
  heldout_hills REAL,                      -- held-out agreement, hills model
  heldout_line  REAL,                      -- held-out agreement, straight arrow
  params_line   TEXT,                      -- the straight arrow: the live ranker since 2026-10-22
  params_one    TEXT,                      -- comparison model: one hill per group (cewek, cowok)
  heldout_one   REAL,                      -- held-out agreement, one hill per group
  n_test        INTEGER,
  split_by      TEXT,                      -- 'applicant' or 'pick'
  n_picks       INTEGER NOT NULL DEFAULT 0,  -- picks the current scores learned from
  trained_at    TEXT
);

-- Comparison models on another embedding ("Model 2" on the scores page).
-- Same people and picks as the main model, each in its own taste space
-- (alt_z, from pipeline sinemart.alt). Battles always use the main model.
CREATE TABLE IF NOT EXISTS alt_models (
  key        TEXT PRIMARY KEY,
  init       TEXT NOT NULL,
  params     TEXT,
  heldout    REAL,
  params_one  TEXT,                        -- the same embedding with one hill per group
  heldout_one REAL,
  params_line  TEXT,                       -- the same embedding with a straight arrow
  heldout_line REAL,
  fixed       TEXT,                        -- optional never-trained score per applicant (JSON {id: score})
  heldout_fixed REAL,                      -- its agreement with the same held-out picks
  n_picks    INTEGER NOT NULL DEFAULT 0,
  trained_at TEXT
);
CREATE TABLE IF NOT EXISTS alt_z (
  key          TEXT NOT NULL,
  applicant_id INTEGER NOT NULL,
  z            TEXT NOT NULL,
  PRIMARY KEY (key, applicant_id)
);

-- Fixed face-geometry measurements (pipeline sinemart.geometry), one row per
-- applicant with a frontal enough photo. Lower = closer to the rule's ideal;
-- score = the combined "Golden ratio" ranking (higher = more ideal).
CREATE TABLE IF NOT EXISTS geometry (
  applicant_id INTEGER PRIMARY KEY,
  n_photos     INTEGER,
  score        REAL,
  phi_dev      REAL,                         -- distance from the golden ratio
  pallett_dev  REAL,                         -- distance from Pallett's 36% / 46%
  asymmetry    REAL,
  atypicality  REAL                          -- distance from the average face (averageness, reversed)
);

-- Key face points on each face crop (pipeline sinemart.landmarks), as 0-1
-- fractions of the crop, so the pages can draw the golden-ratio lines.
CREATE TABLE IF NOT EXISTS face_points (
  image_key TEXT PRIMARY KEY,              -- R2 key of the crop, as in applicants.image_key
  points    TEXT NOT NULL,                 -- JSON {name: [x, y]}, as seen in the photo (lines are drawn here)
  front     TEXT,                          -- the same points with the head straightened (numbers come from these)
  yaw       REAL,                          -- head turn, degrees
  pitch     REAL
);

-- Outside verdicts used only to check the model (e.g. 'sortir2026' = the
-- 2026 human shortlist). Only aggregates leave the API, never per person,
-- so raters can't be nudged by them.
CREATE TABLE IF NOT EXISTS outside_labels (
  applicant_id INTEGER NOT NULL,
  source       TEXT NOT NULL,
  label        INTEGER NOT NULL,           -- 1 = passed that screening
  PRIMARY KEY (source, applicant_id)
);
