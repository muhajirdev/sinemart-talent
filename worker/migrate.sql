-- One-off changes for a database created from an older schema.sql.
ALTER TABLE applicants ADD COLUMN gender TEXT;
ALTER TABLE applicants ADD COLUMN gender_src TEXT;
ALTER TABLE model ADD COLUMN params_one TEXT;
ALTER TABLE model ADD COLUMN heldout_one REAL;
ALTER TABLE picks ADD COLUMN tie INTEGER NOT NULL DEFAULT 0;
ALTER TABLE alt_models ADD COLUMN params_one TEXT;
ALTER TABLE alt_models ADD COLUMN heldout_one REAL;
ALTER TABLE alt_models ADD COLUMN fixed TEXT;
ALTER TABLE alt_models ADD COLUMN heldout_fixed REAL;

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
  points    TEXT NOT NULL,                 -- JSON {name: [x, y]}
  yaw       REAL,                          -- head turn, degrees
  pitch     REAL
);
ALTER TABLE face_points ADD COLUMN front TEXT;

-- 2026-10-22: the straight arrow ranks and picks battles; clusters are comparisons.
ALTER TABLE model ADD COLUMN params_line TEXT;
ALTER TABLE alt_models ADD COLUMN params_line TEXT;
ALTER TABLE alt_models ADD COLUMN heldout_line REAL;

-- 2026-10-22: raters can mark a person's photos as bad data ("Foto bermasalah").
ALTER TABLE applicants ADD COLUMN noise TEXT;

-- 2026-10-22: "Keduanya tidak" (neither): both faces a clear no.
ALTER TABLE picks ADD COLUMN neither INTEGER NOT NULL DEFAULT 0;

-- 2026-10-27: where each person came from, and test batches (e.g. "bc"
-- showcase photos) that are scored but never picked or trained on.
ALTER TABLE applicants ADD COLUMN source TEXT;
ALTER TABLE applicants ADD COLUMN test INTEGER NOT NULL DEFAULT 0;
UPDATE applicants SET source = CASE WHEN id < 200000 THEN '2021' WHEN stage = 'bc' THEN 'bc' ELSE '2026' END;
UPDATE applicants SET test = 1, noise = NULL WHERE stage = 'bc' AND noise = 'bc';
