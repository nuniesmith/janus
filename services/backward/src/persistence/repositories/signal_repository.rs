//! # Signal Repository
//!
//! Database repository for trading signals providing CRUD operations
//! and querying capabilities.

use crate::persistence::DatabaseError;
use crate::persistence::models::{NewSignal, SignalRecord, SignalStats};
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

/// Repository for signal database operations
#[derive(Clone)]
pub struct SignalRepository {
    pool: PgPool,
}

impl SignalRepository {
    /// Create a new signal repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert a new signal
    pub async fn create(&self, signal: NewSignal) -> Result<SignalRecord, DatabaseError> {
        let record = sqlx::query_as::<_, SignalRecord>(
            r#"
            INSERT INTO signals (
                signal_id, symbol, signal_type, timeframe, confidence, strength,
                timestamp, entry_price, stop_loss, take_profit, position_size,
                risk_amount, risk_reward_ratio, source_type, source_name,
                strategy_name, strategy_score, model_name, model_version,
                model_confidence, indicators, metadata, filtered, is_backtest
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
            RETURNING *
            "#,
        )
        .bind(signal.signal_id)
        .bind(&signal.symbol)
        .bind(&signal.signal_type)
        .bind(&signal.timeframe)
        .bind(signal.confidence)
        .bind(signal.strength)
        .bind(signal.timestamp)
        .bind(signal.entry_price)
        .bind(signal.stop_loss)
        .bind(signal.take_profit)
        .bind(signal.position_size)
        .bind(signal.risk_amount)
        .bind(signal.risk_reward_ratio)
        .bind(&signal.source_type)
        .bind(&signal.source_name)
        .bind(&signal.strategy_name)
        .bind(signal.strategy_score)
        .bind(&signal.model_name)
        .bind(&signal.model_version)
        .bind(signal.model_confidence)
        .bind(&signal.indicators)
        .bind(&signal.metadata)
        .bind(signal.filtered)
        .bind(signal.is_backtest)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Find signal by ID
    pub async fn find_by_id(&self, signal_id: Uuid) -> Result<SignalRecord, DatabaseError> {
        let record =
            sqlx::query_as::<_, SignalRecord>("SELECT * FROM signals WHERE signal_id = $1")
                .bind(signal_id)
                .fetch_one(&self.pool)
                .await?;

        Ok(record)
    }

    /// Find signals by symbol
    pub async fn find_by_symbol(
        &self,
        symbol: &str,
        limit: i64,
    ) -> Result<Vec<SignalRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, SignalRecord>(
            "SELECT * FROM signals WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2",
        )
        .bind(symbol)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find signals by symbol and timeframe
    pub async fn find_by_symbol_and_timeframe(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: i64,
    ) -> Result<Vec<SignalRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, SignalRecord>(
            r#"
            SELECT * FROM signals
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
            "#,
        )
        .bind(symbol)
        .bind(timeframe)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find signals within time range
    pub async fn find_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<SignalRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, SignalRecord>(
            r#"
            SELECT * FROM signals
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp DESC
            LIMIT $3
            "#,
        )
        .bind(start)
        .bind(end)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find recent signals
    pub async fn find_recent(&self, limit: i64) -> Result<Vec<SignalRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, SignalRecord>(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find actionable signals (not filtered, high confidence)
    pub async fn find_actionable(
        &self,
        min_confidence: f64,
        limit: i64,
    ) -> Result<Vec<SignalRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, SignalRecord>(
            r#"
            SELECT * FROM signals
            WHERE filtered = false
            AND confidence >= $1
            AND status = 'generated'
            ORDER BY confidence DESC, strength DESC
            LIMIT $2
            "#,
        )
        .bind(min_confidence)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Update signal status
    pub async fn update_status(
        &self,
        signal_id: Uuid,
        status: &str,
    ) -> Result<SignalRecord, DatabaseError> {
        let record = sqlx::query_as::<_, SignalRecord>(
            "UPDATE signals SET status = $1 WHERE signal_id = $2 RETURNING *",
        )
        .bind(status)
        .bind(signal_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Mark signal as executed
    pub async fn mark_executed(
        &self,
        signal_id: Uuid,
        execution_price: f64,
    ) -> Result<SignalRecord, DatabaseError> {
        let record = sqlx::query_as::<_, SignalRecord>(
            r#"
            UPDATE signals
            SET status = 'executed',
                executed_at = NOW(),
                execution_price = $1
            WHERE signal_id = $2
            RETURNING *
            "#,
        )
        .bind(execution_price)
        .bind(signal_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Mark signal as closed with PnL
    pub async fn mark_closed(
        &self,
        signal_id: Uuid,
        close_price: f64,
        pnl: f64,
        pnl_percentage: f64,
    ) -> Result<SignalRecord, DatabaseError> {
        let record = sqlx::query_as::<_, SignalRecord>(
            r#"
            UPDATE signals
            SET status = 'closed',
                closed_at = NOW(),
                close_price = $1,
                pnl = $2,
                pnl_percentage = $3
            WHERE signal_id = $4
            RETURNING *
            "#,
        )
        .bind(close_price)
        .bind(pnl)
        .bind(pnl_percentage)
        .bind(signal_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get signal statistics
    pub async fn get_stats(&self) -> Result<SignalStats, DatabaseError> {
        let stats = sqlx::query_as::<_, SignalStats>(
            r#"
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE signal_type = 'Buy') as buy_count,
                COUNT(*) FILTER (WHERE signal_type = 'Sell') as sell_count,
                COUNT(*) FILTER (WHERE signal_type = 'Hold') as hold_count,
                AVG(confidence) as avg_confidence,
                AVG(strength) as avg_strength,
                COUNT(*) FILTER (WHERE filtered = true) as filtered_count,
                CASE
                    WHEN COUNT(*) > 0 THEN
                        COUNT(*) FILTER (WHERE filtered = true)::float / COUNT(*)::float
                    ELSE 0
                END as filter_rate
            FROM signals
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(stats)
    }

    /// Get stats for a specific symbol
    pub async fn get_stats_by_symbol(&self, symbol: &str) -> Result<SignalStats, DatabaseError> {
        let stats = sqlx::query_as::<_, SignalStats>(
            r#"
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE signal_type = 'Buy') as buy_count,
                COUNT(*) FILTER (WHERE signal_type = 'Sell') as sell_count,
                COUNT(*) FILTER (WHERE signal_type = 'Hold') as hold_count,
                AVG(confidence) as avg_confidence,
                AVG(strength) as avg_strength,
                COUNT(*) FILTER (WHERE filtered = true) as filtered_count,
                CASE
                    WHEN COUNT(*) > 0 THEN
                        COUNT(*) FILTER (WHERE filtered = true)::float / COUNT(*)::float
                    ELSE 0
                END as filter_rate
            FROM signals
            WHERE symbol = $1
            "#,
        )
        .bind(symbol)
        .fetch_one(&self.pool)
        .await?;

        Ok(stats)
    }

    /// Delete old signals (cleanup)
    pub async fn delete_older_than(
        &self,
        cutoff_date: DateTime<Utc>,
    ) -> Result<u64, DatabaseError> {
        let result = sqlx::query("DELETE FROM signals WHERE created_at < $1")
            .bind(cutoff_date)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Count signals
    pub async fn count(&self) -> Result<i64, DatabaseError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM signals")
            .fetch_one(&self.pool)
            .await?;

        Ok(count)
    }

    /// Count signals by status
    pub async fn count_by_status(&self, status: &str) -> Result<i64, DatabaseError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM signals WHERE status = $1")
            .bind(status)
            .fetch_one(&self.pool)
            .await?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;

    /// Helper: connect to the test database or skip gracefully.
    async fn test_pool() -> Option<PgPool> {
        let url = std::env::var("DATABASE_URL").ok()?;
        PgPoolOptions::new()
            .max_connections(2)
            .acquire_timeout(Duration::from_secs(5))
            .connect(&url)
            .await
            .ok()
    }

    #[tokio::test]
    async fn test_signal_repository_creation() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = SignalRepository::new(pool);
        // Verify we can query signals (empty result is fine – proves the repo + connection work)
        let result = repo.find_by_symbol("NONEXISTENT_ASSET", 10).await;
        assert!(
            result.is_ok(),
            "find_by_asset should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!(result.unwrap().is_empty());
    }
}
