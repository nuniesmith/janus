-- =============================================================================
-- 04_api_keys.sql — API key encrypted storage & audit trail
-- Runs automatically on first postgres container start (or manually).
--
-- Tables created:
--   api_keys       — Fernet-encrypted API key values, one row per key name
--   api_key_audit  — Immutable audit log for every key mutation
-- =============================================================================

\set janus_db  `echo "${POSTGRES_DB:-janus_db}"`
\connect :janus_db

-- ---------------------------------------------------------------------------
-- api_keys
-- ---------------------------------------------------------------------------
-- Stores encrypted API key values managed through the Settings UI.
-- encrypted_value: Fernet-encrypted base64 ciphertext.
--   NULL means "not stored in DB — fall back to the environment variable".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_keys (
    id              SERIAL PRIMARY KEY,
    key_name        TEXT NOT NULL UNIQUE,
    encrypted_value TEXT,                          -- NULL → use env var fallback
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by      TEXT NOT NULL DEFAULT 'system' -- 'web', 'cli', 'system', …
);

-- Partial index: only index rows that actually have an encrypted value
CREATE INDEX IF NOT EXISTS api_keys_name_idx
    ON api_keys (key_name)
    WHERE encrypted_value IS NOT NULL;

-- ---------------------------------------------------------------------------
-- api_key_audit
-- ---------------------------------------------------------------------------
-- Append-only audit trail.  Rows are never updated or deleted.
-- action: one of 'set', 'delete', 'validate'
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_key_audit (
    id           SERIAL PRIMARY KEY,
    key_name     TEXT        NOT NULL,
    action       TEXT        NOT NULL,             -- 'set' | 'delete' | 'validate'
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    performed_by TEXT        NOT NULL DEFAULT 'system',
    ip_address   TEXT        NOT NULL DEFAULT ''
);

-- Index for efficient per-key audit queries and time-ordered listing
CREATE INDEX IF NOT EXISTS api_key_audit_key_time_idx
    ON api_key_audit (key_name, performed_at DESC);

-- Index for listing all recent events regardless of key
CREATE INDEX IF NOT EXISTS api_key_audit_time_idx
    ON api_key_audit (performed_at DESC);
