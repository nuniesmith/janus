//! Position event persistence (JFLOW-C, persistence slice).
//!
//! Writes the position telemetry received by the `POST /api/v1/positions/*`
//! endpoints into Postgres: open snapshots ([`PositionEvent`]) into the
//! `janus_position_events` ingest log, and closed-trade outcomes
//! ([`PositionOutcome`]) into `janus_position_outcomes`. The Python JanusAI
//! service owns both schemas; this module probes for each at startup and
//! degrades to a logged no-op when a table is absent (mirrors the JFLOW-D
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
//! And `janus_position_outcomes` (closed trades, with the guidance history
//! Janus accumulated while the position was open):
//!
//! ```text
//!   id                     BIGSERIAL        PRIMARY KEY
//!   received_at            TIMESTAMPTZ      NOT NULL DEFAULT NOW()
//!   symbol                 TEXT             NOT NULL
//!   side                   TEXT             NOT NULL    -- 'Buy' / 'Sell'
//!   qty                    DOUBLE PRECISION NOT NULL
//!   entry_price            DOUBLE PRECISION NOT NULL
//!   exit_price             DOUBLE PRECISION NOT NULL
//!   pnl_realized           DOUBLE PRECISION NOT NULL
//!   realized_ratio         DOUBLE PRECISION NOT NULL
//!   result                 TEXT             NOT NULL    -- 'win'/'loss'/'breakeven'
//!   rr_ratio               DOUBLE PRECISION NULL
//!   strategy               TEXT             NULL
//!   peak_pnl_ratio         DOUBLE PRECISION NULL
//!   samples                BIGINT           NOT NULL
//!   last_guidance          TEXT             NULL        -- 'hold'/'reduce'/'exit'
//!   time_in_position_secs  DOUBLE PRECISION NULL
//!   position_id            TEXT             NULL
//!   session_id             TEXT             NULL
//! ```
//!
//! Compaction of these raw logs into closed-trade rows in `janus_memories`
//! is a JanusAI-side follow-up.

use janus_core::{PositionEvent, PositionOutcome};
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
    /// Whether the `janus_position_events` table exists.
    events: bool,
    /// Whether the `janus_position_outcomes` table exists.
    outcomes: bool,
}

#[cfg(not(feature = "persistence"))]
#[derive(Debug, Clone)]
struct ActiveStore;

impl PositionEventStore {
    /// Disabled store — `record` is a debug-logged no-op.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// True when `record` will attempt a database write (the
    /// `janus_position_events` table is present).
    pub fn is_enabled(&self) -> bool {
        match &self.inner {
            #[cfg(feature = "persistence")]
            Some(s) => s.events,
            #[cfg(not(feature = "persistence"))]
            Some(_) => false,
            None => false,
        }
    }

    /// True when `record_outcome` will attempt a database write (the
    /// `janus_position_outcomes` table is present).
    pub fn is_outcomes_enabled(&self) -> bool {
        match &self.inner {
            #[cfg(feature = "persistence")]
            Some(s) => s.outcomes,
            #[cfg(not(feature = "persistence"))]
            Some(_) => false,
            None => false,
        }
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

        let events = table_exists(&pool, "janus_position_events").await;
        let outcomes = table_exists(&pool, "janus_position_outcomes").await;

        if !events && !outcomes {
            warn!(
                "neither janus_position_events nor janus_position_outcomes is present — \
                 position persistence disabled (the JanusAI service owns these schemas)"
            );
            return Self::disabled();
        }

        info!(events, outcomes, "Position store connected to Postgres");
        Self {
            inner: Some(ActiveStore {
                pool,
                events,
                outcomes,
            }),
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
        if !store.events {
            debug!("janus_position_events table absent, skipping record");
            return;
        }

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

    /// Persist a single closed-trade outcome. Best-effort: logs on failure
    /// so the HTTP handler can still acknowledge the close.
    #[cfg(feature = "persistence")]
    pub async fn record_outcome(&self, outcome: &PositionOutcome) {
        let Some(store) = &self.inner else {
            debug!("Position store disabled, skipping outcome");
            return;
        };
        if !store.outcomes {
            debug!("janus_position_outcomes table absent, skipping outcome");
            return;
        }

        let side_str = format!("{:?}", outcome.side);
        let result = sqlx::query(
            "INSERT INTO janus_position_outcomes
             (symbol, side, qty, entry_price, exit_price, pnl_realized, realized_ratio, result,
              rr_ratio, strategy, peak_pnl_ratio, samples, last_guidance, time_in_position_secs,
              position_id, session_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)",
        )
        .bind(&outcome.symbol)
        .bind(side_str)
        .bind(outcome.qty)
        .bind(outcome.entry_price)
        .bind(outcome.exit_price)
        .bind(outcome.pnl_realized)
        .bind(outcome.realized_ratio)
        .bind(outcome.result.as_str())
        .bind(outcome.rr_ratio)
        .bind(outcome.strategy.as_deref())
        .bind(outcome.peak_pnl_ratio)
        .bind(outcome.samples as i64)
        .bind(outcome.last_guidance.map(|g| g.as_str()))
        .bind(outcome.time_in_position_secs)
        .bind(outcome.position_id.as_deref())
        .bind(outcome.session_id.as_deref())
        .execute(&store.pool)
        .await;

        if let Err(e) = result {
            warn!(error = %e, symbol = %outcome.symbol, "failed to persist position outcome");
        }
    }

    /// Persistence feature disabled at compile time — no-op.
    #[cfg(not(feature = "persistence"))]
    pub async fn record_outcome(&self, _outcome: &PositionOutcome) {
        debug!("Position store: persistence feature disabled at compile time");
    }
}

/// Probe `information_schema` for a table the JanusAI service owns.
#[cfg(feature = "persistence")]
async fn table_exists(pool: &sqlx::PgPool, table: &str) -> bool {
    let exists: Result<bool, _> = sqlx::query_scalar(
        "SELECT EXISTS (
             SELECT 1 FROM information_schema.tables WHERE table_name = $1
         )",
    )
    .bind(table)
    .fetch_one(pool)
    .await;
    match exists {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, table, "position store: failed to probe table");
            false
        }
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
            atr_pct: None,
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
