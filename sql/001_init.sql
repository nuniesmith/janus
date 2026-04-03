-- =============================================================================
-- janus/001_init.sql — janus_db setup
-- =============================================================================
-- Configures the janus_db database that is created automatically by the
-- PostgreSQL Docker image via the POSTGRES_DB env var.
--
-- Runs on first container start (empty volume).
-- Schema tables are created by Janus sqlx migrations at service startup:
--   src/janus/services/backward/migrations/
--
-- Run order: first (postgres image creates DB + user before initdb scripts run)
-- =============================================================================

\set janus_db  `echo "${POSTGRES_DB:-janus_db}"`
\set fks_user  `echo "${POSTGRES_USER:-fks_user}"`

\connect :janus_db

-- Required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Ensure the app user has full privileges
GRANT ALL PRIVILEGES ON DATABASE :janus_db TO :fks_user;

ALTER SCHEMA public OWNER TO :fks_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO :fks_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES    TO :fks_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON SEQUENCES TO :fks_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON FUNCTIONS TO :fks_user;
