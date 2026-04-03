//! # Position Repository
//!
//! Database repository for position tracking and management.

use crate::persistence::DatabaseError;
use crate::persistence::models::{
    ClosePosition, NewPosition, PositionRecord, PositionUpdate, PositionUpdateRecord,
};

use sqlx::PgPool;
use uuid::Uuid;

/// Repository for position database operations
#[derive(Clone)]
pub struct PositionRepository {
    pool: PgPool,
}

impl PositionRepository {
    /// Create a new position repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new position
    pub async fn create(&self, position: NewPosition) -> Result<PositionRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PositionRecord>(
            r#"
            INSERT INTO positions (
                portfolio_id, signal_id, symbol, side, entry_price, quantity,
                position_value, stop_loss, take_profit, risk_amount,
                risk_percentage, risk_reward_ratio, metadata, notes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING *
            "#,
        )
        .bind(position.portfolio_id)
        .bind(position.signal_id)
        .bind(&position.symbol)
        .bind(&position.side)
        .bind(position.entry_price)
        .bind(position.quantity)
        .bind(position.position_value)
        .bind(position.stop_loss)
        .bind(position.take_profit)
        .bind(position.risk_amount)
        .bind(position.risk_percentage)
        .bind(position.risk_reward_ratio)
        .bind(&position.metadata)
        .bind(&position.notes)
        .fetch_one(&self.pool)
        .await?;

        // Create audit trail entry
        self.create_update_record(
            record.position_id,
            "open",
            record.entry_price,
            record.stop_loss,
            record.take_profit,
            None,
            None,
        )
        .await?;

        Ok(record)
    }

    /// Find position by ID
    pub async fn find_by_id(&self, position_id: Uuid) -> Result<PositionRecord, DatabaseError> {
        let record =
            sqlx::query_as::<_, PositionRecord>("SELECT * FROM positions WHERE position_id = $1")
                .bind(position_id)
                .fetch_one(&self.pool)
                .await?;

        Ok(record)
    }

    /// Find all positions for a portfolio
    pub async fn find_by_portfolio(
        &self,
        portfolio_id: Uuid,
    ) -> Result<Vec<PositionRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionRecord>(
            "SELECT * FROM positions WHERE portfolio_id = $1 ORDER BY opened_at DESC",
        )
        .bind(portfolio_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find open positions for a portfolio
    pub async fn find_open_positions(
        &self,
        portfolio_id: Uuid,
    ) -> Result<Vec<PositionRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionRecord>(
            r#"
            SELECT * FROM positions
            WHERE portfolio_id = $1 AND status = 'open'
            ORDER BY opened_at DESC
            "#,
        )
        .bind(portfolio_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find positions by symbol
    pub async fn find_by_symbol(
        &self,
        symbol: &str,
        limit: i64,
    ) -> Result<Vec<PositionRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionRecord>(
            "SELECT * FROM positions WHERE symbol = $1 ORDER BY opened_at DESC LIMIT $2",
        )
        .bind(symbol)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Find closed positions
    pub async fn find_closed_positions(
        &self,
        portfolio_id: Uuid,
        limit: i64,
    ) -> Result<Vec<PositionRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionRecord>(
            r#"
            SELECT * FROM positions
            WHERE portfolio_id = $1 AND status = 'closed'
            ORDER BY closed_at DESC
            LIMIT $2
            "#,
        )
        .bind(portfolio_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Update position current price and unrealized PnL
    pub async fn update(
        &self,
        position_id: Uuid,
        update: PositionUpdate,
    ) -> Result<PositionRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PositionRecord>(
            r#"
            UPDATE positions
            SET current_price = COALESCE($1, current_price),
                stop_loss = COALESCE($2, stop_loss),
                take_profit = COALESCE($3, take_profit),
                unrealized_pnl = COALESCE($4, unrealized_pnl),
                unrealized_pnl_percentage = COALESCE($5, unrealized_pnl_percentage),
                updated_at = NOW()
            WHERE position_id = $6
            RETURNING *
            "#,
        )
        .bind(update.current_price)
        .bind(update.stop_loss)
        .bind(update.take_profit)
        .bind(update.unrealized_pnl)
        .bind(update.unrealized_pnl_percentage)
        .bind(position_id)
        .fetch_one(&self.pool)
        .await?;

        // Create audit trail entry
        self.create_update_record(
            position_id,
            "update",
            update
                .current_price
                .unwrap_or(record.current_price.unwrap_or(0.0)),
            update.stop_loss,
            update.take_profit,
            update.unrealized_pnl,
            update.unrealized_pnl_percentage,
        )
        .await?;

        Ok(record)
    }

    /// Update stop loss
    pub async fn update_stop_loss(
        &self,
        position_id: Uuid,
        stop_loss: f64,
    ) -> Result<PositionRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PositionRecord>(
            r#"
            UPDATE positions
            SET stop_loss = $1,
                updated_at = NOW()
            WHERE position_id = $2
            RETURNING *
            "#,
        )
        .bind(stop_loss)
        .bind(position_id)
        .fetch_one(&self.pool)
        .await?;

        // Create audit trail entry
        self.create_update_record(
            position_id,
            "stop_update",
            record.current_price.unwrap_or(record.entry_price),
            Some(stop_loss),
            record.take_profit,
            None,
            None,
        )
        .await?;

        Ok(record)
    }

    /// Update take profit
    pub async fn update_take_profit(
        &self,
        position_id: Uuid,
        take_profit: f64,
    ) -> Result<PositionRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PositionRecord>(
            r#"
            UPDATE positions
            SET take_profit = $1,
                updated_at = NOW()
            WHERE position_id = $2
            RETURNING *
            "#,
        )
        .bind(take_profit)
        .bind(position_id)
        .fetch_one(&self.pool)
        .await?;

        // Create audit trail entry
        self.create_update_record(
            position_id,
            "tp_update",
            record.current_price.unwrap_or(record.entry_price),
            record.stop_loss,
            Some(take_profit),
            None,
            None,
        )
        .await?;

        Ok(record)
    }

    /// Close a position
    pub async fn close(
        &self,
        position_id: Uuid,
        close_data: ClosePosition,
    ) -> Result<PositionRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PositionRecord>(
            r#"
            UPDATE positions
            SET exit_price = $1,
                realized_pnl = $2,
                realized_pnl_percentage = $3,
                exit_reason = $4,
                status = 'closed',
                closed_at = NOW(),
                updated_at = NOW()
            WHERE position_id = $5
            RETURNING *
            "#,
        )
        .bind(close_data.exit_price)
        .bind(close_data.realized_pnl)
        .bind(close_data.realized_pnl_percentage)
        .bind(&close_data.exit_reason)
        .bind(position_id)
        .fetch_one(&self.pool)
        .await?;

        // Create audit trail entry
        self.create_update_record(
            position_id,
            "close",
            close_data.exit_price,
            record.stop_loss,
            record.take_profit,
            Some(close_data.realized_pnl),
            Some(close_data.realized_pnl_percentage),
        )
        .await?;

        Ok(record)
    }

    /// Get position update history
    pub async fn get_update_history(
        &self,
        position_id: Uuid,
    ) -> Result<Vec<PositionUpdateRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionUpdateRecord>(
            r#"
            SELECT * FROM position_updates
            WHERE position_id = $1
            ORDER BY updated_at ASC
            "#,
        )
        .bind(position_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Count open positions for a portfolio
    pub async fn count_open(&self, portfolio_id: Uuid) -> Result<i64, DatabaseError> {
        let (count,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM positions WHERE portfolio_id = $1 AND status = 'open'",
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }

    /// Get total exposure for a portfolio
    pub async fn get_total_exposure(&self, portfolio_id: Uuid) -> Result<f64, DatabaseError> {
        let (exposure,): (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT COALESCE(SUM(position_value), 0)
            FROM positions
            WHERE portfolio_id = $1 AND status = 'open'
            "#,
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(exposure.unwrap_or(0.0))
    }

    /// Get positions by status
    pub async fn find_by_status(
        &self,
        portfolio_id: Uuid,
        status: &str,
    ) -> Result<Vec<PositionRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PositionRecord>(
            r#"
            SELECT * FROM positions
            WHERE portfolio_id = $1 AND status = $2
            ORDER BY opened_at DESC
            "#,
        )
        .bind(portfolio_id)
        .bind(status)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Delete position (should rarely be used, prefer closing)
    pub async fn delete(&self, position_id: Uuid) -> Result<u64, DatabaseError> {
        let result = sqlx::query("DELETE FROM positions WHERE position_id = $1")
            .bind(position_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    // Private helper methods

    /// Create an audit trail entry for position updates
    #[allow(clippy::too_many_arguments)]
    async fn create_update_record(
        &self,
        position_id: Uuid,
        update_type: &str,
        price: f64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
        unrealized_pnl: Option<f64>,
        unrealized_pnl_percentage: Option<f64>,
    ) -> Result<(), DatabaseError> {
        sqlx::query(
            r#"
            INSERT INTO position_updates (
                position_id, update_type, price, stop_loss, take_profit,
                unrealized_pnl, unrealized_pnl_percentage
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(position_id)
        .bind(update_type)
        .bind(price)
        .bind(stop_loss)
        .bind(take_profit)
        .bind(unrealized_pnl)
        .bind(unrealized_pnl_percentage)
        .execute(&self.pool)
        .await?;

        Ok(())
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
    async fn test_position_repository_creation() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = PositionRepository::new(pool);
        // Verify we can query positions (empty result is fine – proves the repo + connection work)
        let result = repo.find_by_symbol("NONEXISTENT", 10).await;
        assert!(
            result.is_ok(),
            "find_by_symbol should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!(result.unwrap().is_empty());
    }
}
