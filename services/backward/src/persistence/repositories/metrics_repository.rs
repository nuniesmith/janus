//! # Metrics Repository
//!
//! Database repository for risk metrics and analytics.

use crate::persistence::DatabaseError;
use crate::persistence::models::RiskMetricRecord;
use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;

/// Type alias for risk summary query result tuple
type RiskSummaryQueryResult = (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

/// Parameters for creating a full risk snapshot
#[derive(Debug, Clone)]
pub struct RiskSnapshotParams {
    pub portfolio_id: Uuid,
    pub portfolio_value: f64,
    pub cash_balance: f64,
    pub total_exposure: Option<f64>,
    pub exposure_percentage: Option<f64>,
    pub gross_exposure: Option<f64>,
    pub net_exposure: Option<f64>,
    pub active_positions: i32,
    pub max_position_size: Option<f64>,
    pub avg_position_size: Option<f64>,
    pub largest_position_pct: Option<f64>,
    pub top_5_concentration_pct: Option<f64>,
    pub avg_correlation: Option<f64>,
    pub max_correlation: Option<f64>,
    pub portfolio_heat: Option<f64>,
    pub total_risk_amount: Option<f64>,
    pub avg_risk_per_position: Option<f64>,
    pub max_risk_per_position: Option<f64>,
    pub var_95: Option<f64>,
    pub var_99: Option<f64>,
    pub cvar_95: Option<f64>,
    pub portfolio_volatility: Option<f64>,
    pub realized_volatility: Option<f64>,
    pub current_drawdown: Option<f64>,
    pub drawdown_from_peak: Option<f64>,
    pub peak_portfolio_value: Option<f64>,
    pub limits_exceeded: Option<JsonValue>,
    pub metadata: Option<JsonValue>,
}

/// Repository for risk metrics
#[derive(Clone)]
pub struct MetricsRepository {
    pool: PgPool,
}

impl MetricsRepository {
    /// Create a new metrics repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Record a risk metrics snapshot
    #[allow(clippy::too_many_arguments)]
    pub async fn create_risk_snapshot(
        &self,
        portfolio_id: Uuid,
        portfolio_value: f64,
        cash_balance: f64,
        total_exposure: Option<f64>,
        exposure_percentage: Option<f64>,
        active_positions: i32,
        portfolio_heat: Option<f64>,
        total_risk_amount: Option<f64>,
        current_drawdown: Option<f64>,
        metadata: Option<JsonValue>,
    ) -> Result<RiskMetricRecord, DatabaseError> {
        let record = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            INSERT INTO risk_metrics (
                portfolio_id, portfolio_value, cash_balance, total_exposure,
                exposure_percentage, active_positions, portfolio_heat,
                total_risk_amount, current_drawdown, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            "#,
        )
        .bind(portfolio_id)
        .bind(portfolio_value)
        .bind(cash_balance)
        .bind(total_exposure)
        .bind(exposure_percentage)
        .bind(active_positions)
        .bind(portfolio_heat)
        .bind(total_risk_amount)
        .bind(current_drawdown)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Record a complete risk metrics snapshot with all fields
    pub async fn create_full_risk_snapshot(
        &self,
        params: RiskSnapshotParams,
    ) -> Result<RiskMetricRecord, DatabaseError> {
        let record = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            INSERT INTO risk_metrics (
                portfolio_id, portfolio_value, cash_balance, total_exposure,
                exposure_percentage, gross_exposure, net_exposure, active_positions,
                max_position_size, avg_position_size, largest_position_pct,
                top_5_concentration_pct, avg_correlation, max_correlation,
                portfolio_heat, total_risk_amount, avg_risk_per_position,
                max_risk_per_position, var_95, var_99, cvar_95,
                portfolio_volatility, realized_volatility, current_drawdown,
                drawdown_from_peak, peak_portfolio_value, limits_exceeded, metadata
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
            )
            RETURNING *
            "#,
        )
        .bind(params.portfolio_id)
        .bind(params.portfolio_value)
        .bind(params.cash_balance)
        .bind(params.total_exposure)
        .bind(params.exposure_percentage)
        .bind(params.gross_exposure)
        .bind(params.net_exposure)
        .bind(params.active_positions)
        .bind(params.max_position_size)
        .bind(params.avg_position_size)
        .bind(params.largest_position_pct)
        .bind(params.top_5_concentration_pct)
        .bind(params.avg_correlation)
        .bind(params.max_correlation)
        .bind(params.portfolio_heat)
        .bind(params.total_risk_amount)
        .bind(params.avg_risk_per_position)
        .bind(params.max_risk_per_position)
        .bind(params.var_95)
        .bind(params.var_99)
        .bind(params.cvar_95)
        .bind(params.portfolio_volatility)
        .bind(params.realized_volatility)
        .bind(params.current_drawdown)
        .bind(params.drawdown_from_peak)
        .bind(params.peak_portfolio_value)
        .bind(params.limits_exceeded)
        .bind(params.metadata)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get latest risk metrics for a portfolio
    pub async fn get_latest_risk_metrics(
        &self,
        portfolio_id: Uuid,
    ) -> Result<Option<RiskMetricRecord>, DatabaseError> {
        let record = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            SELECT * FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT 1
            "#,
        )
        .bind(portfolio_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get risk metrics history for a portfolio
    pub async fn get_risk_metrics_history(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<RiskMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            SELECT * FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Get risk metrics within time range
    pub async fn get_risk_metrics_by_time_range(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<RiskMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            SELECT * FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            ORDER BY measured_at DESC
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Get average portfolio heat over a period
    pub async fn get_avg_portfolio_heat(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<f64, DatabaseError> {
        let (avg_heat,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT AVG(portfolio_heat)
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await?;

        Ok(avg_heat.unwrap_or(0.0))
    }

    /// Get maximum drawdown over a period
    pub async fn get_max_drawdown(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<f64, DatabaseError> {
        let (max_dd,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT MAX(current_drawdown)
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await?;

        Ok(max_dd.unwrap_or(0.0))
    }

    /// Get average exposure percentage
    pub async fn get_avg_exposure(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<f64, DatabaseError> {
        let (avg_exposure,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT AVG(exposure_percentage)
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await?;

        Ok(avg_exposure.unwrap_or(0.0))
    }

    /// Get VaR (Value at Risk) history
    pub async fn get_var_history(
        &self,
        portfolio_id: Uuid,
        confidence_level: u8, // 95 or 99
        limit: i64,
    ) -> Result<Vec<(DateTime<Utc>, f64)>, DatabaseError> {
        let field = match confidence_level {
            95 => "var_95",
            99 => "var_99",
            _ => {
                return Err(DatabaseError::QueryError(
                    "Invalid confidence level".to_string(),
                ));
            }
        };

        let query = format!(
            r#"
            SELECT measured_at, {}
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND {} IS NOT NULL
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
            field, field
        );

        let results: Vec<(DateTime<Utc>, f64)> = sqlx::query_as(&query)
            .bind(portfolio_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(results)
    }

    /// Get volatility history
    pub async fn get_volatility_history(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(DateTime<Utc>, Option<f64>, Option<f64>)>, DatabaseError> {
        let results: Vec<(DateTime<Utc>, Option<f64>, Option<f64>)> = sqlx::query_as(
            r#"
            SELECT measured_at, portfolio_volatility, realized_volatility
            FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get concentration risk over time
    pub async fn get_concentration_history(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(DateTime<Utc>, Option<f64>, Option<f64>)>, DatabaseError> {
        let results: Vec<(DateTime<Utc>, Option<f64>, Option<f64>)> = sqlx::query_as(
            r#"
            SELECT measured_at, largest_position_pct, top_5_concentration_pct
            FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get correlation metrics history
    pub async fn get_correlation_history(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(DateTime<Utc>, Option<f64>, Option<f64>)>, DatabaseError> {
        let results: Vec<(DateTime<Utc>, Option<f64>, Option<f64>)> = sqlx::query_as(
            r#"
            SELECT measured_at, avg_correlation, max_correlation
            FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get times when risk limits were exceeded
    pub async fn get_limit_violations(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<RiskMetricRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, RiskMetricRecord>(
            r#"
            SELECT * FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            AND limits_exceeded IS NOT NULL
            AND jsonb_array_length(limits_exceeded) > 0
            ORDER BY measured_at DESC
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Count limit violations
    pub async fn count_limit_violations(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<i64, DatabaseError> {
        let (count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            AND limits_exceeded IS NOT NULL
            AND jsonb_array_length(limits_exceeded) > 0
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }

    /// Get portfolio value history
    pub async fn get_portfolio_value_history(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<(DateTime<Utc>, f64)>, DatabaseError> {
        let results: Vec<(DateTime<Utc>, f64)> = sqlx::query_as(
            r#"
            SELECT measured_at, portfolio_value
            FROM risk_metrics
            WHERE portfolio_id = $1
            ORDER BY measured_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(results)
    }

    /// Get risk metrics summary statistics
    pub async fn get_risk_summary(
        &self,
        portfolio_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<RiskSummary, DatabaseError> {
        let result: RiskSummaryQueryResult = sqlx::query_as(
            r#"
            SELECT
                AVG(exposure_percentage) as avg_exposure,
                MAX(exposure_percentage) as max_exposure,
                AVG(portfolio_heat) as avg_heat,
                MAX(portfolio_heat) as max_heat,
                AVG(current_drawdown) as avg_drawdown,
                MAX(current_drawdown) as max_drawdown
            FROM risk_metrics
            WHERE portfolio_id = $1
            AND measured_at >= $2
            AND measured_at <= $3
            "#,
        )
        .bind(portfolio_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await?;

        Ok(RiskSummary {
            avg_exposure: result.0.unwrap_or(0.0),
            max_exposure: result.1.unwrap_or(0.0),
            avg_heat: result.2.unwrap_or(0.0),
            max_heat: result.3.unwrap_or(0.0),
            avg_drawdown: result.4.unwrap_or(0.0),
            max_drawdown: result.5.unwrap_or(0.0),
        })
    }

    /// Delete old risk metrics (cleanup)
    pub async fn delete_metrics_older_than(
        &self,
        cutoff_date: DateTime<Utc>,
    ) -> Result<u64, DatabaseError> {
        let result = sqlx::query("DELETE FROM risk_metrics WHERE measured_at < $1")
            .bind(cutoff_date)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Get count of risk snapshots
    pub async fn count_snapshots(&self, portfolio_id: Uuid) -> Result<i64, DatabaseError> {
        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM risk_metrics WHERE portfolio_id = $1")
                .bind(portfolio_id)
                .fetch_one(&self.pool)
                .await?;

        Ok(count)
    }
}

/// Risk summary statistics
#[derive(Debug, Clone)]
pub struct RiskSummary {
    pub avg_exposure: f64,
    pub max_exposure: f64,
    pub avg_heat: f64,
    pub max_heat: f64,
    pub avg_drawdown: f64,
    pub max_drawdown: f64,
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
    async fn test_metrics_repository_creation() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = MetricsRepository::new(pool);
        // Verify we can query risk metrics (empty result is fine – proves the repo + connection work)
        let portfolio_id = Uuid::new_v4();
        let result = repo.get_risk_metrics_history(portfolio_id, 10).await;
        assert!(
            result.is_ok(),
            "get_risk_metrics_history should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_risk_summary_creation() {
        let summary = RiskSummary {
            avg_exposure: 50.0,
            max_exposure: 75.0,
            avg_heat: 0.015,
            max_heat: 0.025,
            avg_drawdown: 0.05,
            max_drawdown: 0.12,
        };

        assert_eq!(summary.avg_exposure, 50.0);
        assert_eq!(summary.max_drawdown, 0.12);
    }
}
