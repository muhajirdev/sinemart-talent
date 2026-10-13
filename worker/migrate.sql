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
