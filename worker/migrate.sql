-- One-off changes for a database created from an older schema.sql.
ALTER TABLE applicants ADD COLUMN gender TEXT;
ALTER TABLE applicants ADD COLUMN gender_src TEXT;
ALTER TABLE model ADD COLUMN params_one TEXT;
ALTER TABLE model ADD COLUMN heldout_one REAL;
ALTER TABLE picks ADD COLUMN tie INTEGER NOT NULL DEFAULT 0;
ALTER TABLE alt_models ADD COLUMN params_one TEXT;
ALTER TABLE alt_models ADD COLUMN heldout_one REAL;
