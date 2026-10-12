-- D1 schema for the A/B picking app.
-- wrangler d1 execute sinemart-talent --remote --file schema.sql

CREATE TABLE IF NOT EXISTS applicants (
  id         INTEGER PRIMARY KEY,          -- Sinemart applicant number
  image_key  TEXT NOT NULL,                -- R2 key of the aligned grey crop
  stage      TEXT,                         -- Drive selection folder, if any
  look       INTEGER,                      -- k-means cluster from training
  score      REAL,                         -- current head's score
  score_std  REAL,                         -- linear-head posterior std
  model      TEXT,                         -- which head wrote the score
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS applicants_score ON applicants (score DESC);

-- Raters sign in with a per-person link (?t=<token>).
CREATE TABLE IF NOT EXISTS raters (
  token TEXT PRIMARY KEY,
  name  TEXT NOT NULL
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
