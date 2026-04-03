//! # Performance Repository
//!
//! Database repository for performance metrics and analytics.

use crate::persistence::DatabaseError;
use crate::persistence::models::{PerformanceStatsRecord, TradeMetricRecord};
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

/// Parameters for creating a trade metric record
#[derive(Debug, Clone)]
pub struct TradeMetricParams {
    pub position_id: Uuid,
    pub signal_id: Option<Uuid>,
    pub portfolio_id: Uuid,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub gross_pnl: f64,
    pub gross_pnl_percentage: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub exit_reason: Option<String>,
}

/// Repository for performance metrics
#[derive(Clone)]
pub struct PerformanceRepository {
    pool: PgPool,
}

impl PerformanceRepository {
    /// Create a new performance repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a trade metric record
    pub async fn create_trade_metric(
        &self,
        params: TradeMetricParams,
    ) -> Result<TradeMetricRecord, DatabaseError> {
        let holding_duration = (params.exit_time - params.entry_time).num_seconds();

        let record = sqlx::query_as::<_, TradeMetricRecord>(
            r#"
            INSERT INTO trade_metrics (
                position_id, signal_id, portfolio_id, symbol, side,
                entry_price, exit_price, quantity, gross_pnl, gross_pnl_percentage,
                entry_time, exit_time, holding_duration_seconds, exit_reason
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING *
            "#,
        )
        .bind(params.position_id)
        .bind(params.signal_id)
        .bind(params.portfolio_id)
        .bind(&params.symbol)
        .bind(&params.side)
        .bind(params.entry_price)
        .bind(params.exit_price)
        .bind(params.quantity)
        .bind(params.gross_pnl)
        .bind(params.gross_pnl_percentage)
        .bind(params.entry_time)
        .bind(params.exit_time)
        .bind(holding_duration)
        .bind(&params.exit_reason)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get trade metrics for a portfolio
    pub async fn get_trade_metrics(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<TradeMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, TradeMetricRecord>(
            r#"
            SELECT * FROM trade_metrics
            WHERE portfolio_id = $1
            ORDER BY exit_time DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Get trade metrics by symbol
    pub async fn get_trade_metrics_by_symbol(
        &self,
        portfolio_id: Uuid,
        symbol: &str,
        limit: i64,
    ) -> Result<Vec<TradeMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, TradeMetricRecord>(
            r#"
            SELECT * FROM trade_metrics
            WHERE portfolio_id = $1 AND symbol = $2
            ORDER BY exit_time DESC
            LIMIT $3
            "#,
        )
        .bind(portfolio_id)
        .bind(symbol)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Get trade metrics within time range
    pub async fn get_trade_metrics_by_time_range(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<TradeMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, TradeMetricRecord>(
            r#"
            SELECT * FROM trade_metrics
            WHERE portfolio_id = $1
            AND exit_time >= $2
            AND exit_time <= $3
            ORDER BY exit_time DESC
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Calculate and store performance statistics
    pub async fn calculate_performance_stats(
        &self,
        portfolio_id: Uuid,
        symbol: Option<String>,
        timeframe: Option<String>,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<PerformanceStatsRecord, DatabaseError> {
        // This is a complex calculation that aggregates trade metrics
        let record = sqlx::query_as::<_, PerformanceStatsRecord>(
            r#"
            INSERT INTO performance_stats (
                portfolio_id, symbol, timeframe, period_start, period_end,
                total_trades, winning_trades, losing_trades, breakeven_trades,
                win_rate, loss_rate, gross_profit, gross_loss, net_profit,
                avg_win, avg_loss, avg_win_pct, avg_loss_pct,
                largest_win, largest_loss, largest_win_pct, largest_loss_pct,
                profit_factor, expectancy, expectancy_pct
            )
            SELECT
                $1 as portfolio_id,
                $2 as symbol,
                $3 as timeframe,
                $4 as period_start,
                $5 as period_end,
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE gross_pnl > 0) as winning_trades,
                COUNT(*) FILTER (WHERE gross_pnl < 0) as losing_trades,
                COUNT(*) FILTER (WHERE gross_pnl = 0) as breakeven_trades,
                CASE
                    WHEN COUNT(*) > 0 THEN
                        COUNT(*) FILTER (WHERE gross_pnl > 0)::float / COUNT(*)::float
                    ELSE 0
                END as win_rate,
                CASE
                    WHEN COUNT(*) > 0 THEN
                        COUNT(*) FILTER (WHERE gross_pnl < 0)::float / COUNT(*)::float
                    ELSE 0
                END as loss_rate,
                COALESCE(SUM(gross_pnl) FILTER (WHERE gross_pnl > 0), 0) as gross_profit,
                COALESCE(ABS(SUM(gross_pnl)) FILTER (WHERE gross_pnl < 0), 0) as gross_loss,
                COALESCE(SUM(gross_pnl), 0) as net_profit,
                AVG(gross_pnl) FILTER (WHERE gross_pnl > 0) as avg_win,
                AVG(gross_pnl) FILTER (WHERE gross_pnl < 0) as avg_loss,
                AVG(gross_pnl_percentage) FILTER (WHERE gross_pnl > 0) as avg_win_pct,
                AVG(gross_pnl_percentage) FILTER (WHERE gross_pnl < 0) as avg_loss_pct,
                MAX(gross_pnl) as largest_win,
                MIN(gross_pnl) as largest_loss,
                MAX(gross_pnl_percentage) as largest_win_pct,
                MIN(gross_pnl_percentage) as largest_loss_pct,
                CASE
                    WHEN COALESCE(ABS(SUM(gross_pnl)) FILTER (WHERE gross_pnl < 0), 0) > 0 THEN
                        COALESCE(SUM(gross_pnl) FILTER (WHERE gross_pnl > 0), 0) /
                        COALESCE(ABS(SUM(gross_pnl)) FILTER (WHERE gross_pnl < 0), 1)
                    ELSE 0
                END as profit_factor,
                AVG(gross_pnl) as expectancy,
                AVG(gross_pnl_percentage) as expectancy_pct
            FROM trade_metrics
            WHERE portfolio_id = $1
            AND ($2::text IS NULL OR symbol = $2)
            AND exit_time >= $4
            AND exit_time <= $5
            HAVING COUNT(*) > 0
            RETURNING *
            "#,
        )
        .bind(portfolio_id)
        .bind(&symbol)
        .bind(&timeframe)
        .bind(period_start)
        .bind(period_end)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get latest performance statistics
    pub async fn get_latest_performance_stats(
        &self,
        portfolio_id: Uuid,
        symbol: Option<String>,
        timeframe: Option<String>,
    ) -> Result<Option<PerformanceStatsRecord>, DatabaseError> {
        let record = sqlx::query_as::<_, PerformanceStatsRecord>(
            r#"
            SELECT * FROM performance_stats
            WHERE portfolio_id = $1
            AND ($2::text IS NULL OR symbol = $2)
            AND ($3::text IS NULL OR timeframe = $3)
            ORDER BY calculated_at DESC
            LIMIT 1
            "#,
        )
        .bind(portfolio_id)
        .bind(&symbol)
        .bind(&timeframe)
        .fetch_optional(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get all performance statistics for a portfolio
    pub async fn get_all_performance_stats(
        &self,
        portfolio_id: Uuid,
    ) -> Result<Vec<PerformanceStatsRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PerformanceStatsRecord>(
            r#"
            SELECT * FROM performance_stats
            WHERE portfolio_id = $1
            ORDER BY period_start DESC, symbol ASC
            "#,
        )
        .bind(portfolio_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Get performance statistics by timeframe
    pub async fn get_performance_stats_by_timeframe(
        &self,
        portfolio_id: Uuid,
        timeframe: &str,
    ) -> Result<Vec<PerformanceStatsRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PerformanceStatsRecord>(
            r#"
            SELECT * FROM performance_stats
            WHERE portfolio_id = $1 AND timeframe = $2
            ORDER BY period_start DESC
            "#,
        )
        .bind(portfolio_id)
        .bind(timeframe)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Calculate win rate for a portfolio
    pub async fn calculate_win_rate(&self, portfolio_id: Uuid) -> Result<f64, DatabaseError> {
        let (win_rate,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT
                CASE
                    WHEN COUNT(*) > 0 THEN
                        COUNT(*) FILTER (WHERE gross_pnl > 0)::float / COUNT(*)::float
                    ELSE 0
                END as win_rate
            FROM trade_metrics
            WHERE portfolio_id = $1
            "#,
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(win_rate.unwrap_or(0.0))
    }

    /// Calculate average holding duration
    pub async fn calculate_avg_holding_duration(
        &self,
        portfolio_id: Uuid,
    ) -> Result<i64, DatabaseError> {
        let (avg_duration,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT AVG(holding_duration_seconds)
            FROM trade_metrics
            WHERE portfolio_id = $1
            "#,
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(avg_duration.unwrap_or(0.0) as i64)
    }

    /// Get best performing symbols
    pub async fn get_best_performing_symbols(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(String, f64, i32)>, DatabaseError> {
        let results: Vec<(String, f64, i32)> = sqlx::query_as(
            r#"
            SELECT
                symbol,
                SUM(gross_pnl) as total_pnl,
                COUNT(*)::int as trade_count
            FROM trade_metrics
            WHERE portfolio_id = $1
            GROUP BY symbol
            ORDER BY total_pnl DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get worst performing symbols
    pub async fn get_worst_performing_symbols(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(String, f64, i32)>, DatabaseError> {
        let results: Vec<(String, f64, i32)> = sqlx::query_as(
            r#"
            SELECT
                symbol,
                SUM(gross_pnl) as total_pnl,
                COUNT(*)::int as trade_count
            FROM trade_metrics
            WHERE portfolio_id = $1
            GROUP BY symbol
            ORDER BY total_pnl ASC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get monthly performance summary
    pub async fn get_monthly_performance(
        &self,
        portfolio_id: Uuid,
        year: i32,
    ) -> Result<Vec<(i32, f64, i32)>, DatabaseError> {
        let results: Vec<(i32, f64, i32)> = sqlx::query_as(
            r#"
            SELECT
                EXTRACT(MONTH FROM exit_time)::int as month,
                SUM(gross_pnl) as monthly_pnl,
                COUNT(*)::int as trade_count
            FROM trade_metrics
            WHERE portfolio_id = $1
            AND EXTRACT(YEAR FROM exit_time) = $2
            GROUP BY EXTRACT(MONTH FROM exit_time)
            ORDER BY month ASC
            "#,
        )
        .bind(portfolio_id)
        .bind(year)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get total PnL for a portfolio
    pub async fn get_total_pnl(&self, portfolio_id: Uuid) -> Result<f64, DatabaseError> {
        let (total_pnl,): (Option<f64>,) = sqlx::query_as(
            "SELECT COALESCE(SUM(gross_pnl), 0) FROM trade_metrics WHERE portfolio_id = $1",
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(total_pnl.unwrap_or(0.0))
    }

    /// Delete old trade metrics (cleanup)
    pub async fn delete_metrics_older_than(
        &self,
        cutoff_date: DateTime<Utc>,
    ) -> Result<u64, DatabaseError> {
        let result = sqlx::query("DELETE FROM trade_metrics WHERE exit_time < $1")
            .bind(cutoff_date)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
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
    async fn test_performance_repository_creation() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = PerformanceRepository::new(pool);
        // Verify we can query performance stats (empty result is fine – proves the repo + connection work)
        let portfolio_id = Uuid::new_v4();
        let result = repo.get_all_performance_stats(portfolio_id).await;
        assert!(
            result.is_ok(),
            "get_all_performance_stats should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_total_pnl_empty_portfolio() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = PerformanceRepository::new(pool);
        let portfolio_id = Uuid::new_v4();
        let result = repo.get_total_pnl(portfolio_id).await;
        assert!(
            result.is_ok(),
            "get_total_pnl should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!((result.unwrap() - 0.0).abs() < f64::EPSILON);
    }
}
