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

-- Every answer. winner_id NULL = skipped ("can't tell" / bad photo).
CREATE TABLE IF NOT EXISTS picks (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  rater      TEXT NOT NULL,
  left_id    INTEGER NOT NULL,
  right_id   INTEGER NOT NULL,
  winner_id  INTEGER,
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
  params_one    TEXT,                      -- comparison model: one hill per group (cewek, cowok)
  heldout_one   REAL,                      -- held-out agreement, one hill per group
  n_test        INTEGER,
  split_by      TEXT,                      -- 'applicant' or 'pick'
  n_picks       INTEGER NOT NULL DEFAULT 0,  -- picks the current scores learned from
  trained_at    TEXT
);
