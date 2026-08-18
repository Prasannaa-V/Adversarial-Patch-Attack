-- InfraEdge-Shield SQLite Schema
-- dashboard/backend/db/schema.sql

CREATE TABLE IF NOT EXISTS attack_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,                  -- ISO 8601 UTC
    frame_id        TEXT    NOT NULL,
    frame_path      TEXT    NOT NULL DEFAULT '',
    confidence      REAL    NOT NULL,
    gate_status     TEXT    NOT NULL DEFAULT 'LOCKED', -- LOCKED / UNLOCKED
    operator_ack    INTEGER NOT NULL DEFAULT 0,        -- 0=pending, 1=acked
    notes           TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_health (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    fps             REAL    NOT NULL DEFAULT 0,
    sdnl_latency_ms REAL    NOT NULL DEFAULT 0,
    esp32_online    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS operators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'operator',  -- operator / admin
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Index for paginated event queries
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON attack_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp DESC);
