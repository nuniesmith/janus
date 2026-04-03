//! # Portfolio Repository
//!
//! Database repository for portfolio management operations.

use crate::persistence::DatabaseError;
use crate::persistence::models::{NewPortfolio, PortfolioRecord, PortfolioSummary};
use sqlx::PgPool;
use uuid::Uuid;

/// Repository for portfolio database operations
#[derive(Clone)]
pub struct PortfolioRepository {
    pool: PgPool,
}

impl PortfolioRepository {
    /// Create a new portfolio repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new portfolio
    pub async fn create(&self, portfolio: NewPortfolio) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            r#"
            INSERT INTO portfolios (name, account_id, initial_balance, current_balance, risk_config)
            VALUES ($1, $2, $3, $3, $4)
            RETURNING *
            "#,
        )
        .bind(&portfolio.name)
        .bind(&portfolio.account_id)
        .bind(portfolio.initial_balance)
        .bind(&portfolio.risk_config)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Find portfolio by ID
    pub async fn find_by_id(&self, portfolio_id: Uuid) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            "SELECT * FROM portfolios WHERE portfolio_id = $1",
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Find portfolio by account ID and name
    pub async fn find_by_account(
        &self,
        account_id: &str,
        name: &str,
    ) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            "SELECT * FROM portfolios WHERE account_id = $1 AND name = $2",
        )
        .bind(account_id)
        .bind(name)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// List all portfolios for an account
    pub async fn list_by_account(
        &self,
        account_id: &str,
    ) -> Result<Vec<PortfolioRecord>, DatabaseError> {
        let records = sqlx::query_as::<_, PortfolioRecord>(
            "SELECT * FROM portfolios WHERE account_id = $1 ORDER BY created_at DESC",
        )
        .bind(account_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(records)
    }

    /// Update portfolio balance
    pub async fn update_balance(
        &self,
        portfolio_id: Uuid,
        new_balance: f64,
    ) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            r#"
            UPDATE portfolios
            SET current_balance = $1,
                updated_at = NOW()
            WHERE portfolio_id = $2
            RETURNING *
            "#,
        )
        .bind(new_balance)
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Update portfolio PnL
    pub async fn update_pnl(
        &self,
        portfolio_id: Uuid,
        total_pnl: f64,
        daily_pnl: f64,
    ) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            r#"
            UPDATE portfolios
            SET total_pnl = $1,
                total_pnl_percentage = CASE
                    WHEN initial_balance > 0 THEN ($1 / initial_balance) * 100
                    ELSE 0
                END,
                daily_pnl = $2,
                daily_pnl_percentage = CASE
                    WHEN current_balance > 0 THEN ($2 / current_balance) * 100
                    ELSE 0
                END,
                updated_at = NOW()
            WHERE portfolio_id = $3
            RETURNING *
            "#,
        )
        .bind(total_pnl)
        .bind(daily_pnl)
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Update portfolio status
    pub async fn update_status(
        &self,
        portfolio_id: Uuid,
        status: &str,
    ) -> Result<PortfolioRecord, DatabaseError> {
        let record = sqlx::query_as::<_, PortfolioRecord>(
            "UPDATE portfolios SET status = $1, updated_at = NOW() WHERE portfolio_id = $2 RETURNING *",
        )
        .bind(status)
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(record)
    }

    /// Get portfolio summary
    pub async fn get_summary(&self, portfolio_id: Uuid) -> Result<PortfolioSummary, DatabaseError> {
        let summary = sqlx::query_as::<_, PortfolioSummary>(
            r#"
            SELECT
                p.portfolio_id,
                p.name,
                p.current_balance,
                p.total_pnl,
                p.total_pnl_percentage,
                p.active_positions,
                p.total_exposure,
                CASE
                    WHEN (p.winning_positions + p.losing_positions) > 0 THEN
                        p.winning_positions::float / (p.winning_positions + p.losing_positions)::float
                    ELSE NULL
                END as win_rate,
                p.sharpe_ratio
            FROM portfolios p
            WHERE p.portfolio_id = $1
            "#,
        )
        .bind(portfolio_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(summary)
    }

    /// Delete portfolio
    pub async fn delete(&self, portfolio_id: Uuid) -> Result<u64, DatabaseError> {
        let result = sqlx::query("DELETE FROM portfolios WHERE portfolio_id = $1")
            .bind(portfolio_id)
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
    async fn test_portfolio_repository_creation() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = PortfolioRepository::new(pool);
        // Verify we can list portfolios (empty result is fine – proves the repo + connection work)
        let result = repo.list_by_account("test-account-nonexistent").await;
        assert!(
            result.is_ok(),
            "list_by_account should succeed on a valid connection: {:?}",
            result.err()
        );
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_portfolio_crud() {
        let Some(pool) = test_pool().await else {
            eprintln!("Skipping test: DATABASE_URL not set or database unavailable");
            return;
        };

        let repo = PortfolioRepository::new(pool);

        // Create
        let new_portfolio = NewPortfolio {
            name: "test-portfolio".to_string(),
            account_id: "test-crud-account".to_string(),
            initial_balance: 10_000.0,
            risk_config: Some(serde_json::json!({"max_drawdown": 0.1})),
        };

        let created = repo.create(new_portfolio).await;
        assert!(
            created.is_ok(),
            "create should succeed: {:?}",
            created.err()
        );
        let record = created.unwrap();
        assert_eq!(record.name, "test-portfolio");
        assert_eq!(record.account_id, "test-crud-account");

        // Read
        let found = repo.find_by_id(record.portfolio_id).await;
        assert!(
            found.is_ok(),
            "find_by_id should succeed: {:?}",
            found.err()
        );
        assert_eq!(found.unwrap().portfolio_id, record.portfolio_id);

        // Update balance
        let updated = repo.update_balance(record.portfolio_id, 12_000.0).await;
        assert!(
            updated.is_ok(),
            "update_balance should succeed: {:?}",
            updated.err()
        );
        assert!((updated.unwrap().current_balance - 12_000.0).abs() < f64::EPSILON);

        // Delete
        let deleted = repo.delete(record.portfolio_id).await;
        assert!(
            deleted.is_ok(),
            "delete should succeed: {:?}",
            deleted.err()
        );
        assert_eq!(deleted.unwrap(), 1);
    }
}
