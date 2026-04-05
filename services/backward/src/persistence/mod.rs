//! # Persistence Layer
//!
//! Database persistence for JANUS service including:
//! - Signal storage and retrieval
//! - Portfolio and position management
//! - Performance metrics tracking
//! - Trade analytics
//!
//! ## Database Support
//!
//! Currently supports PostgreSQL via SQLx with async operations.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_backward::persistence::{Database, DatabaseConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = DatabaseConfig::from_env()?;
//!     let db = Database::connect(config).await?;
//!
//!     // Use repositories
//!     let signal_repo = db.signal_repository();
//!
//!     Ok(())
//! }
//! ```

pub mod experience_store;
pub mod models;
pub mod repositories;

pub use experience_store::{ExperienceStore, ExperienceStoreConfig, PersistenceMetrics};

use anyhow::{Context, Result};
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;
use tracing::info;

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Database URL (postgresql://user:pass@host:port/db)
    pub database_url: String,

    /// Maximum number of connections in the pool
    pub max_connections: u32,

    /// Minimum number of connections in the pool
    pub min_connections: u32,

    /// Connection timeout in seconds
    pub connect_timeout: u64,

    /// Idle connection timeout in seconds
    pub idle_timeout: u64,

    /// Maximum lifetime of a connection in seconds
    pub max_lifetime: u64,

    /// Enable SQL statement logging
    pub enable_logging: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://janus:janus@localhost:5432/janus".to_string(),
            max_connections: 10,
            min_connections: 2,
            connect_timeout: 30,
            idle_timeout: 600,
            max_lifetime: 1800,
            enable_logging: false,
        }
    }
}

impl DatabaseConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://janus:janus@localhost:5432/janus".to_string());

        let max_connections = std::env::var("DB_MAX_CONNECTIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let min_connections = std::env::var("DB_MIN_CONNECTIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let connect_timeout = std::env::var("DB_CONNECT_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let idle_timeout = std::env::var("DB_IDLE_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600);

        let max_lifetime = std::env::var("DB_MAX_LIFETIME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800);

        let enable_logging = std::env::var("DB_ENABLE_LOGGING")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        Ok(Self {
            database_url,
            max_connections,
            min_connections,
            connect_timeout,
            idle_timeout,
            max_lifetime,
            enable_logging,
        })
    }

    /// Create a new configuration with custom database URL
    pub fn new(database_url: String) -> Self {
        Self {
            database_url,
            ..Default::default()
        }
    }

    /// Set maximum connections
    pub fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Set minimum connections
    pub fn with_min_connections(mut self, min: u32) -> Self {
        self.min_connections = min;
        self
    }

    /// Enable SQL logging
    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = enable;
        self
    }
}

/// Database connection pool and repository factory
pub struct Database {
    pool: PgPool,
}

impl Database {
    /// Connect to the database and create connection pool
    pub async fn connect(config: DatabaseConfig) -> Result<Self> {
        info!("Connecting to database...");
        info!("  Max connections: {}", config.max_connections);
        info!("  Min connections: {}", config.min_connections);
        info!("  Connect timeout: {}s", config.connect_timeout);
        info!("  Idle timeout: {}s", config.idle_timeout);
        info!("  Max lifetime: {}s", config.max_lifetime);

        // Log sanitized URL for diagnostics (redact password)
        let sanitized_url = Self::sanitize_url(&config.database_url);
        info!("  Database URL: {}", sanitized_url);

        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(Duration::from_secs(config.connect_timeout))
            .idle_timeout(Duration::from_secs(config.idle_timeout))
            .max_lifetime(Duration::from_secs(config.max_lifetime))
            .connect(&config.database_url)
            .await
            .map_err(|e| {
                tracing::error!("❌ Database connection failed!");
                tracing::error!("  URL (sanitized): {}", sanitized_url);
                tracing::error!("  Error type: {:?}", e);
                tracing::error!("  Error details: {}", e);
                // Check for common failure modes
                let err_str = e.to_string();
                if err_str.contains("password authentication failed") {
                    tracing::error!("  💡 HINT: Password mismatch — the POSTGRES_PASSWORD in .env");
                    tracing::error!("     may not match what postgres was initialized with.");
                    tracing::error!(
                        "     If postgres data volume was created with a different password,"
                    );
                    tracing::error!("     either reset the volume or update .env to match.");
                } else if err_str.contains("Connection refused") {
                    tracing::error!(
                        "  💡 HINT: Postgres is not reachable at the configured host/port."
                    );
                    tracing::error!(
                        "     Check that the postgres container is running and healthy."
                    );
                } else if err_str.contains("does not exist") {
                    tracing::error!("  💡 HINT: The database or role may not exist.");
                    tracing::error!("     Check JANUS_DB and POSTGRES_USER settings.");
                } else if err_str.contains("SSL") || err_str.contains("ssl") {
                    tracing::error!("  💡 HINT: SSL negotiation issue. The DATABASE_URL may need");
                    tracing::error!(
                        "     '?sslmode=disable' or the ssl parameters may be invalid."
                    );
                }
                e
            })
            .context("Failed to connect to database")?;

        info!("✅ Database connection established");
        info!("  Pool size: {}", pool.size());

        Ok(Self { pool })
    }

    /// Sanitize a database URL for safe logging (redact password)
    fn sanitize_url(url: &str) -> String {
        // Match postgresql://user:password@host:port/db?params
        if let Some(at_pos) = url.find('@')
            && let Some(colon_pos) = url[..at_pos].rfind(':')
        {
            // Check there's a :// before the user:pass part
            if let Some(scheme_end) = url.find("://")
                && colon_pos > scheme_end + 3
            {
                // There's a password between the last : before @ and the @
                let mut sanitized = String::with_capacity(url.len());
                sanitized.push_str(&url[..colon_pos + 1]);
                sanitized.push_str("****");
                sanitized.push_str(&url[at_pos..]);
                return sanitized;
            }
        }
        // If we can't parse it, just return a generic message
        format!("(unparseable URL, len={})", url.len())
    }

    /// Get the underlying connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run database migrations
    pub async fn run_migrations(&self) -> Result<()> {
        info!("Running database migrations...");

        // Note: In production, use sqlx-cli or embedded migrations
        // For now, we'll implement a simple migration runner

        self.run_migration_file(include_str!(
            "../../migrations/001_create_signals_table.sql"
        ))
        .await
        .context("Failed to run migration 001")?;

        self.run_migration_file(include_str!(
            "../../migrations/002_create_portfolio_tables.sql"
        ))
        .await
        .context("Failed to run migration 002")?;

        self.run_migration_file(include_str!(
            "../../migrations/003_create_metrics_tables.sql"
        ))
        .await
        .context("Failed to run migration 003")?;

        self.run_migration_file(include_str!(
            "../../migrations/004_fix_decimal_to_float.sql"
        ))
        .await
        .context("Failed to run migration 004")?;

        info!("✅ Database migrations completed");

        Ok(())
    }

    /// Run a single migration file
    async fn run_migration_file(&self, sql: &str) -> Result<()> {
        // Split SQL by semicolons but handle dollar-quoted strings ($$) properly
        // PostgreSQL functions use $$ delimiters which can contain semicolons
        let mut statements = Vec::new();
        let mut current_statement = String::new();
        let mut in_dollar_quote = false;

        for line in sql.lines() {
            let trimmed = line.trim();

            // Skip comment-only lines when not in a dollar-quoted block
            if !in_dollar_quote && (trimmed.is_empty() || trimmed.starts_with("--")) {
                continue;
            }

            // Add line to current statement
            if !current_statement.is_empty() {
                current_statement.push('\n');
            }
            current_statement.push_str(line);

            // Check for dollar-quote delimiters
            let dollar_count = trimmed.matches("$$").count();
            if dollar_count > 0 {
                // Toggle dollar-quote state for each $$ found
                for _ in 0..dollar_count {
                    in_dollar_quote = !in_dollar_quote;
                }
            }

            // If line ends with semicolon and we're not in a dollar-quoted block, it's the end of a statement
            if !in_dollar_quote && trimmed.ends_with(';') {
                // Remove the trailing semicolon
                if let Some(last_semicolon) = current_statement.rfind(';') {
                    current_statement.truncate(last_semicolon);
                }
                let stmt = current_statement.trim().to_string();
                if !stmt.is_empty() {
                    statements.push(stmt);
                }
                current_statement.clear();
            }
        }

        // Add any remaining statement
        let remaining = current_statement.trim();
        if !remaining.is_empty() {
            statements.push(remaining.to_string());
        }

        // Execute each statement
        for statement in statements {
            sqlx::query(&statement)
                .execute(&self.pool)
                .await
                .context(format!("Failed to execute SQL statement: {}", statement))?;
        }
        Ok(())
    }

    /// Check database health
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("Database health check failed")?;
        Ok(())
    }

    /// Get signal repository
    pub fn signal_repository(&self) -> repositories::SignalRepository {
        repositories::SignalRepository::new(self.pool.clone())
    }

    /// Get portfolio repository
    pub fn portfolio_repository(&self) -> repositories::PortfolioRepository {
        repositories::PortfolioRepository::new(self.pool.clone())
    }

    /// Get position repository
    pub fn position_repository(&self) -> repositories::PositionRepository {
        repositories::PositionRepository::new(self.pool.clone())
    }

    /// Get performance repository
    pub fn performance_repository(&self) -> repositories::PerformanceRepository {
        repositories::PerformanceRepository::new(self.pool.clone())
    }

    /// Get metrics repository
    pub fn metrics_repository(&self) -> repositories::MetricsRepository {
        repositories::MetricsRepository::new(self.pool.clone())
    }

    /// Close the database connection pool
    pub async fn close(&self) {
        info!("Closing database connection pool...");
        self.pool.close().await;
        info!("✅ Database connection pool closed");
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

/// Database error types
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("Database connection error: {0}")]
    ConnectionError(String),

    #[error("Query execution error: {0}")]
    QueryError(String),

    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Duplicate record: {0}")]
    DuplicateKey(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Migration error: {0}")]
    MigrationError(String),
}

impl From<sqlx::Error> for DatabaseError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => DatabaseError::NotFound("Record not found".to_string()),
            sqlx::Error::Database(db_err) => {
                if let Some(code) = db_err.code() {
                    if code == "23505" {
                        // PostgreSQL unique violation
                        return DatabaseError::DuplicateKey(db_err.to_string());
                    } else if code.starts_with("23") {
                        // Other constraint violations
                        return DatabaseError::ConstraintViolation(db_err.to_string());
                    }
                }
                DatabaseError::QueryError(db_err.to_string())
            }
            sqlx::Error::PoolTimedOut => {
                DatabaseError::ConnectionError("Connection pool timeout".to_string())
            }
            sqlx::Error::PoolClosed => {
                DatabaseError::ConnectionError("Connection pool closed".to_string())
            }
            _ => DatabaseError::QueryError(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DatabaseConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_connections, 2);
        assert!(!config.enable_logging);
    }

    #[test]
    fn test_config_builder() {
        let config = DatabaseConfig::new("postgresql://localhost/test".to_string())
            .with_max_connections(20)
            .with_min_connections(5)
            .with_logging(true);

        assert_eq!(config.max_connections, 20);
        assert_eq!(config.min_connections, 5);
        assert!(config.enable_logging);
    }

    #[test]
    fn test_config_from_env() {
        // SAFETY: env var mutation is inherently racy in multi-threaded programs;
        // acceptable here because this test does not run concurrently with code
        // that reads these variables.
        unsafe { std::env::set_var("DATABASE_URL", "postgresql://test:test@localhost/testdb") };
        unsafe { std::env::set_var("DB_MAX_CONNECTIONS", "15") };
        unsafe { std::env::set_var("DB_ENABLE_LOGGING", "true") };

        let config = DatabaseConfig::from_env().unwrap();
        assert_eq!(
            config.database_url,
            "postgresql://test:test@localhost/testdb"
        );
        assert_eq!(config.max_connections, 15);
        assert!(config.enable_logging);

        // SAFETY: cleanup mirrors the set_var calls above; same safety reasoning.
        unsafe { std::env::remove_var("DATABASE_URL") };
        unsafe { std::env::remove_var("DB_MAX_CONNECTIONS") };
        unsafe { std::env::remove_var("DB_ENABLE_LOGGING") };
    }
}
