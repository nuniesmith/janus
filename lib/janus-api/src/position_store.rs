//! Position event persistence (JFLOW-C, persistence slice).
//!
//! Writes each [`PositionEvent`] received by `POST /api/v1/positions/event`
//! into the Postgres `janus_position_events` ingest log. The Python JanusAI
//! service owns the table schema; this module probes for it at startup and
//! degrades to a logged no-op when it is absent (mirrors the JFLOW-D
//! `bootstrap_affinity_from_postgres` pattern).
//!
//! ## Expected schema
//!
//! The table is created and managed by the JanusAI service. Janus only
//! INSERTs:
//!
//! ```text
//!   id              BIGSERIAL    PRIMARY KEY
//!   received_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
//!   symbol          TEXT         NOT NULL
//!   side            TEXT         NOT NULL    -- 'Buy' / 'Sell'
//!   qty             DOUBLE PRECISION NOT NULL
//!   entry_price     DOUBLE PRECISION NOT NULL
//!   current_price   DOUBLE PRECISION NOT NULL
//!   pnl_unrealized  DOUBLE PRECISION NOT NULL
//!   position_id     TEXT         NULL
//!   session_id      TEXT         NULL
//! ```
//!
//! Compaction of this raw event log into closed-trade rows in
//! `janus_memories` is a JanusAI-side follow-up.

use janus_core::PositionEvent;
#[cfg(feature = "persistence")]
use std::time::Duration;
use tracing::debug;
#[cfg(feature = "persistence")]
use tracing::{info, warn};

/// Append-only writer for position events.
///
/// Construct once at startup with [`connect`] (or [`disabled`] when no
/// `DATABASE_URL` is configured) and share via `Arc`. All `record` calls
/// are best-effort: storage failures are logged and do not propagate, so
/// the HTTP handler can still acknowledge the event.
#[derive(Debug, Clone)]
pub struct PositionEventStore {
    inner: Option<ActiveStore>,
}

#[cfg(feature = "persistence")]
#[derive(Debug, Clone)]
struct ActiveStore {
    pool: sqlx::PgPool,
}

#[cfg(not(feature = "persistence"))]
#[derive(Debug, Clone)]
struct ActiveStore;

impl PositionEventStore {
    /// Disabled store — `record` is a debug-logged no-op.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// True when `record` will attempt a database write.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Connect to Postgres and probe for the `janus_position_events` table.
    ///
    /// Never errors. Returns a disabled store with a warning when the URL is
    /// unreachable or the table is missing — the HTTP API must boot even
    /// when persistence isn't available.
    #[cfg(feature = "persistence")]
    pub async fn connect(database_url: &str) -> Self {
        use sqlx::postgres::PgPoolOptions;

        let pool = match PgPoolOptions::new()
            .max_connections(2)
            .acquire_timeout(Duration::from_secs(5))
            .connect(database_url)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "position event store: failed to connect to Postgres, persistence disabled");
                return Self::disabled();
            }
        };

        let exists: Result<bool, _> = sqlx::query_scalar(
            "SELECT EXISTS (
                 SELECT 1 FROM information_schema.tables
                 WHERE table_name = 'janus_position_events'
             )",
        )
        .fetch_one(&pool)
        .await;

        match exists {
            Ok(true) => {
                info!("Position event store connected to Postgres");
                Self {
                    inner: Some(ActiveStore { pool }),
                }
            }
            Ok(false) => {
                warn!(
                    "janus_position_events table not present — position events will not be persisted \
                     (the JanusAI service owns this schema)"
                );
                Self::disabled()
            }
            Err(e) => {
                warn!(error = %e, "position event store: failed to probe schema, persistence disabled");
                Self::disabled()
            }
        }
    }

    /// Persistence feature disabled at compile time — always returns
    /// a disabled store.
    #[cfg(not(feature = "persistence"))]
    pub async fn connect(_database_url: &str) -> Self {
        debug!("Position event store: persistence feature disabled at compile time");
        Self::disabled()
    }

    /// Persist a single position event. Best-effort: logs on failure and
    /// returns `Ok(())` so the HTTP handler can still acknowledge.
    #[cfg(feature = "persistence")]
    pub async fn record(&self, event: &PositionEvent) {
        let Some(store) = &self.inner else {
            debug!("Position event store disabled, skipping record");
            return;
        };

        let side_str = format!("{:?}", event.side);
        let result = sqlx::query(
            "INSERT INTO janus_position_events
             (symbol, side, qty, entry_price, current_price, pnl_unrealized, position_id, session_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(&event.symbol)
        .bind(side_str)
        .bind(event.qty)
        .bind(event.entry_price)
        .bind(event.current_price)
        .bind(event.pnl_unrealized)
        .bind(event.position_id.as_deref())
        .bind(event.session_id.as_deref())
        .execute(&store.pool)
        .await;

        if let Err(e) = result {
            warn!(error = %e, symbol = %event.symbol, "failed to persist position event");
        }
    }

    /// Persistence feature disabled at compile time — no-op.
    #[cfg(not(feature = "persistence"))]
    pub async fn record(&self, _event: &PositionEvent) {
        debug!("Position event store: persistence feature disabled at compile time");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::market::Side;

    fn sample() -> PositionEvent {
        PositionEvent {
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            qty: 0.1,
            entry_price: 60_000.0,
            current_price: 61_000.0,
            pnl_unrealized: 100.0,
            position_id: None,
            session_id: None,
        }
    }

    #[test]
    fn disabled_store_reports_disabled() {
        let store = PositionEventStore::disabled();
        assert!(!store.is_enabled());
    }

    #[tokio::test]
    async fn disabled_record_is_a_no_op() {
        let store = PositionEventStore::disabled();
        // Doesn't panic, doesn't error — just logs.
        store.record(&sample()).await;
    }
}
