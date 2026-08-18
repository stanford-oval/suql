-- Example schema for the CI liveness pipeline. Replace with your own
-- DDL for any dataset that has at least one free-text column SUQL can
-- run answer() over.
--
-- Two things to preserve when swapping schemas:
--   1. The select_user / creator_role pair below. SUQL's compiler needs
--      a distinct creator role for its temp-table dance.
--   2. The column-list ordering of the CSV pushed to the fixtures repo
--      must match the CREATE TABLE column order, since the workflow
--      uses \COPY ... WITH HEADER.

CREATE TABLE IF NOT EXISTS events (
    event_id_cnty   TEXT PRIMARY KEY,
    event_date      DATE        NOT NULL,
    year            INTEGER,
    event_type      TEXT,
    sub_event_type  TEXT,
    country         TEXT,
    admin1          TEXT,
    admin2          TEXT,
    admin3          TEXT,
    location        TEXT,
    fatalities      INTEGER,
    notes           TEXT
);

-- SUQL's compiler needs roles distinct from the connecting user for its
-- temp-table dance (creator_role creates tables, select_user reads). We
-- create both here so the workflow can SET ROLE into them without GRANT
-- ceremony per query.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'select_user') THEN
        CREATE ROLE select_user LOGIN PASSWORD 'select_user';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'creator_role') THEN
        CREATE ROLE creator_role LOGIN PASSWORD 'creator_role';
    END IF;
END$$;

GRANT SELECT  ON events TO select_user;
GRANT ALL    ON SCHEMA public TO creator_role;
GRANT CREATE ON SCHEMA public TO creator_role;

CREATE INDEX IF NOT EXISTS events_country_date_idx ON events (country, event_date);
