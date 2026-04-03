//! Configuration management for the FKS Execution Service

use crate::error::{ExecutionError, Result};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration for the execution service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Service configuration
    pub service: ServiceConfig,

    /// Execution mode (simulated, paper, live)
    pub execution_mode: ExecutionModeConfig,

    /// Exchange configurations
    pub exchanges: HashMap<String, ExchangeConfig>,

    /// Risk limits
    pub risk: RiskConfig,

    /// QuestDB configuration
    pub questdb: QuestDbConfig,

    /// Redis configuration
    pub redis: RedisConfig,

    /// Simulation configuration (for simulated mode)
    pub simulation: Option<SimulationConfig>,

    /// Kraken configuration (for paper/live modes)
    pub kraken: Option<KrakenConfig>,
}

impl Config {
    /// Load configuration from environment and config files
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let service = ServiceConfig {
            grpc_port: std::env::var("GRPC_PORT")
                .unwrap_or_else(|_| "50052".to_string())
                .parse()
                .map_err(|e| ExecutionError::Config(format!("Invalid GRPC_PORT: {}", e)))?,
            http_port: std::env::var("HTTP_PORT")
                .unwrap_or_else(|_| "8081".to_string())
                .parse()
                .map_err(|e| ExecutionError::Config(format!("Invalid HTTP_PORT: {}", e)))?,
            log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        };

        let execution_mode = ExecutionModeConfig {
            mode: std::env::var("EXECUTION_MODE")
                .unwrap_or_else(|_| "simulated".to_string())
                .parse()
                .map_err(|e: String| ExecutionError::Config(e))?,
        };

        // Load exchange configurations
        let mut exchanges = HashMap::new();

        // Kraken configuration (primary exchange)
        if let Ok(api_key) = std::env::var("KRAKEN_API_KEY") {
            let api_secret = std::env::var("KRAKEN_API_SECRET")
                .map_err(|_| ExecutionError::Config("KRAKEN_API_SECRET required".to_string()))?;
            let testnet = std::env::var("KRAKEN_TESTNET")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false);

            exchanges.insert(
                "kraken".to_string(),
                ExchangeConfig {
                    name: "kraken".to_string(),
                    api_key,
                    api_secret,
                    testnet,
                    enabled: true,
                    rate_limit_per_second: 10,
                    timeout_seconds: 30,
                },
            );
        }

        // Risk configuration
        let risk = RiskConfig {
            max_position_size_usd: std::env::var("MAX_POSITION_SIZE_USD")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()
                .map_err(|e| {
                    ExecutionError::Config(format!("Invalid MAX_POSITION_SIZE_USD: {}", e))
                })?,
            max_portfolio_exposure_usd: std::env::var("MAX_PORTFOLIO_EXPOSURE_USD")
                .unwrap_or_else(|_| "50000".to_string())
                .parse()
                .map_err(|e| {
                    ExecutionError::Config(format!("Invalid MAX_PORTFOLIO_EXPOSURE_USD: {}", e))
                })?,
            max_open_positions: std::env::var("MAX_OPEN_POSITIONS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .map_err(|e| {
                    ExecutionError::Config(format!("Invalid MAX_OPEN_POSITIONS: {}", e))
                })?,
            max_daily_loss_usd: std::env::var("MAX_DAILY_LOSS_USD")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()
                .map_err(|e| {
                    ExecutionError::Config(format!("Invalid MAX_DAILY_LOSS_USD: {}", e))
                })?,
            enable_risk_checks: std::env::var("ENABLE_RISK_CHECKS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        };

        // QuestDB configuration
        let questdb = QuestDbConfig {
            host: std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost".to_string()),
            ilp_port: std::env::var("QUESTDB_ILP_PORT")
                .unwrap_or_else(|_| "9009".to_string())
                .parse()
                .map_err(|e| ExecutionError::Config(format!("Invalid QUESTDB_ILP_PORT: {}", e)))?,
            enabled: std::env::var("QUESTDB_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        };

        // Redis configuration
        let redis = RedisConfig {
            url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            pool_size: std::env::var("REDIS_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
        };

        // Simulation configuration (for simulated mode)
        let simulation = if execution_mode.mode == ExecutionMode::Simulated {
            Some(SimulationConfig {
                initial_balance: std::env::var("SIMULATION_INITIAL_BALANCE")
                    .unwrap_or_else(|_| "100000".to_string())
                    .parse()
                    .map_err(|e| {
                        ExecutionError::Config(format!("Invalid SIMULATION_INITIAL_BALANCE: {}", e))
                    })?,
                slippage_bps: std::env::var("SIMULATION_SLIPPAGE_BPS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .map_err(|e| {
                        ExecutionError::Config(format!("Invalid SIMULATION_SLIPPAGE_BPS: {}", e))
                    })?,
                fee_bps: std::env::var("SIMULATION_FEE_BPS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .map_err(|e| {
                        ExecutionError::Config(format!("Invalid SIMULATION_FEE_BPS: {}", e))
                    })?,
                fill_delay_ms: std::env::var("SIMULATION_FILL_DELAY_MS")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .map_err(|e| {
                        ExecutionError::Config(format!("Invalid SIMULATION_FILL_DELAY_MS: {}", e))
                    })?,
                enable_slippage: std::env::var("SIMULATION_ENABLE_SLIPPAGE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
            })
        } else {
            None
        };

        // Kraken configuration (for paper/live modes)
        let kraken = if execution_mode.mode == ExecutionMode::Paper
            || execution_mode.mode == ExecutionMode::Live
        {
            Some(KrakenConfig {
                dry_run: std::env::var("KRAKEN_DRY_RUN")
                    .unwrap_or_else(|_| {
                        if execution_mode.mode == ExecutionMode::Paper {
                            "true".to_string()
                        } else {
                            "false".to_string()
                        }
                    })
                    .parse()
                    .unwrap_or(true),
                slippage_tolerance_bps: std::env::var("KRAKEN_SLIPPAGE_TOLERANCE_BPS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                fee_rate_bps: std::env::var("KRAKEN_FEE_RATE_BPS")
                    .unwrap_or_else(|_| "26".to_string())
                    .parse()
                    .unwrap_or(26),
                stop_loss_pct: std::env::var("KRAKEN_DEFAULT_STOP_LOSS_PCT")
                    .unwrap_or_else(|_| "2.0".to_string())
                    .parse()
                    .unwrap_or(2.0),
                take_profit_pct: std::env::var("KRAKEN_DEFAULT_TAKE_PROFIT_PCT")
                    .unwrap_or_else(|_| "5.0".to_string())
                    .parse()
                    .unwrap_or(5.0),
            })
        } else {
            None
        };

        Ok(Config {
            service,
            execution_mode,
            exchanges,
            risk,
            questdb,
            redis,
            simulation,
            kraken,
        })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate ports
        if self.service.grpc_port == 0 {
            return Err(ExecutionError::Config("Invalid gRPC port".to_string()));
        }
        if self.service.http_port == 0 {
            return Err(ExecutionError::Config("Invalid HTTP port".to_string()));
        }

        // Validate exchange configs for non-simulated modes
        if self.execution_mode.mode != ExecutionMode::Simulated && self.exchanges.is_empty() {
            return Err(ExecutionError::Config(
                "At least one exchange must be configured for non-simulated mode".to_string(),
            ));
        }

        // Validate risk limits
        if self.risk.max_position_size_usd <= 0.0 {
            return Err(ExecutionError::Config(
                "max_position_size_usd must be positive".to_string(),
            ));
        }
        if self.risk.max_portfolio_exposure_usd <= 0.0 {
            return Err(ExecutionError::Config(
                "max_portfolio_exposure_usd must be positive".to_string(),
            ));
        }

        // Validate simulation config for simulated mode
        if self.execution_mode.mode == ExecutionMode::Simulated && self.simulation.is_none() {
            return Err(ExecutionError::Config(
                "Simulation config required for simulated mode".to_string(),
            ));
        }

        Ok(())
    }
}

/// Service-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub grpc_port: u16,
    pub http_port: u16,
    pub log_level: String,
}

/// Execution mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionModeConfig {
    pub mode: ExecutionMode,
}

/// Execution mode enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    Simulated,
    Paper,
    Live,
}

impl std::str::FromStr for ExecutionMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simulated" => Ok(ExecutionMode::Simulated),
            "paper" => Ok(ExecutionMode::Paper),
            "live" => Ok(ExecutionMode::Live),
            _ => Err(format!("Invalid execution mode: {}", s)),
        }
    }
}

/// Exchange-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
    pub enabled: bool,
    pub rate_limit_per_second: u32,
    pub timeout_seconds: u64,
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position_size_usd: f64,
    pub max_portfolio_exposure_usd: f64,
    pub max_open_positions: usize,
    pub max_daily_loss_usd: f64,
    pub enable_risk_checks: bool,
}

impl RiskConfig {
    /// Check if position size is within limits
    pub fn check_position_size(&self, size_usd: f64) -> Result<()> {
        if !self.enable_risk_checks {
            return Ok(());
        }

        if size_usd > self.max_position_size_usd {
            return Err(ExecutionError::RiskViolation(format!(
                "Position size {} exceeds limit {}",
                size_usd, self.max_position_size_usd
            )));
        }

        Ok(())
    }

    /// Check if portfolio exposure is within limits
    pub fn check_portfolio_exposure(&self, exposure_usd: f64) -> Result<()> {
        if !self.enable_risk_checks {
            return Ok(());
        }

        if exposure_usd > self.max_portfolio_exposure_usd {
            return Err(ExecutionError::RiskViolation(format!(
                "Portfolio exposure {} exceeds limit {}",
                exposure_usd, self.max_portfolio_exposure_usd
            )));
        }

        Ok(())
    }

    /// Check if number of open positions is within limits
    pub fn check_open_positions(&self, count: usize) -> Result<()> {
        if !self.enable_risk_checks {
            return Ok(());
        }

        if count >= self.max_open_positions {
            return Err(ExecutionError::RiskViolation(format!(
                "Open positions {} exceeds limit {}",
                count, self.max_open_positions
            )));
        }

        Ok(())
    }

    /// Check if daily loss is within limits
    pub fn check_daily_loss(&self, loss_usd: f64) -> Result<()> {
        if !self.enable_risk_checks {
            return Ok(());
        }

        if loss_usd.abs() > self.max_daily_loss_usd {
            return Err(ExecutionError::RiskViolation(format!(
                "Daily loss {} exceeds limit {}",
                loss_usd.abs(),
                self.max_daily_loss_usd
            )));
        }

        Ok(())
    }
}

/// QuestDB configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestDbConfig {
    pub host: String,
    pub ilp_port: u16,
    pub enabled: bool,
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
}

/// Simulation-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub initial_balance: f64,
    pub slippage_bps: i32,
    pub fee_bps: i32,
    pub fill_delay_ms: u64,
    pub enable_slippage: bool,
}

impl SimulationConfig {
    /// Calculate slippage for a trade
    pub fn calculate_slippage(&self, price: Decimal, is_buy: bool) -> Decimal {
        if !self.enable_slippage {
            return price;
        }

        let slippage_factor = Decimal::from(self.slippage_bps) / Decimal::from(10000);
        let slippage = price * slippage_factor;

        if is_buy {
            price + slippage
        } else {
            price - slippage
        }
    }

    /// Calculate fee for a trade
    pub fn calculate_fee(&self, trade_value: Decimal) -> Decimal {
        let fee_factor = Decimal::from(self.fee_bps) / Decimal::from(10000);
        trade_value * fee_factor
    }
}

/// Kraken-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrakenConfig {
    /// Enable dry run mode (validate orders but don't execute)
    pub dry_run: bool,
    /// Default slippage tolerance in basis points
    pub slippage_tolerance_bps: u32,
    /// Fee rate in basis points (for P&L calculation)
    pub fee_rate_bps: u32,
    /// Default stop-loss percentage
    pub stop_loss_pct: f64,
    /// Default take-profit percentage
    pub take_profit_pct: f64,
}

impl Default for KrakenConfig {
    fn default() -> Self {
        Self {
            dry_run: true,
            slippage_tolerance_bps: 10,
            fee_rate_bps: 26,
            stop_loss_pct: 2.0,
            take_profit_pct: 5.0,
        }
    }
}

impl KrakenConfig {
    /// Calculate stop-loss price for a long position
    pub fn calculate_stop_loss(&self, entry_price: Decimal, is_long: bool) -> Decimal {
        let sl_factor = Decimal::new((self.stop_loss_pct * 100.0) as i64, 4); // Convert to basis points
        if is_long {
            entry_price * (Decimal::ONE - sl_factor)
        } else {
            entry_price * (Decimal::ONE + sl_factor)
        }
    }

    /// Calculate take-profit price for a long position
    pub fn calculate_take_profit(&self, entry_price: Decimal, is_long: bool) -> Decimal {
        let tp_factor = Decimal::new((self.take_profit_pct * 100.0) as i64, 4); // Convert to basis points
        if is_long {
            entry_price * (Decimal::ONE + tp_factor)
        } else {
            entry_price * (Decimal::ONE - tp_factor)
        }
    }

    /// Calculate fee for a trade
    pub fn calculate_fee(&self, trade_value: Decimal) -> Decimal {
        let fee_factor = Decimal::from(self.fee_rate_bps) / Decimal::from(10000);
        trade_value * fee_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_mode_from_str() {
        assert_eq!(
            "simulated".parse::<ExecutionMode>().unwrap(),
            ExecutionMode::Simulated
        );
        assert_eq!(
            "paper".parse::<ExecutionMode>().unwrap(),
            ExecutionMode::Paper
        );
        assert_eq!(
            "live".parse::<ExecutionMode>().unwrap(),
            ExecutionMode::Live
        );
        assert!("invalid".parse::<ExecutionMode>().is_err());
    }

    #[test]
    fn test_risk_checks() {
        let risk = RiskConfig {
            max_position_size_usd: 10000.0,
            max_portfolio_exposure_usd: 50000.0,
            max_open_positions: 10,
            max_daily_loss_usd: 1000.0,
            enable_risk_checks: true,
        };

        assert!(risk.check_position_size(5000.0).is_ok());
        assert!(risk.check_position_size(15000.0).is_err());

        assert!(risk.check_portfolio_exposure(30000.0).is_ok());
        assert!(risk.check_portfolio_exposure(60000.0).is_err());

        assert!(risk.check_open_positions(5).is_ok());
        assert!(risk.check_open_positions(10).is_err());

        assert!(risk.check_daily_loss(-500.0).is_ok());
        assert!(risk.check_daily_loss(-1500.0).is_err());
    }

    #[test]
    fn test_simulation_slippage() {
        let sim = SimulationConfig {
            initial_balance: 100000.0,
            slippage_bps: 5,
            fee_bps: 10,
            fill_delay_ms: 100,
            enable_slippage: true,
        };

        let price = Decimal::from(50000);
        let buy_price = sim.calculate_slippage(price, true);
        let sell_price = sim.calculate_slippage(price, false);

        assert!(buy_price > price);
        assert!(sell_price < price);
    }

    #[test]
    fn test_simulation_fee() {
        let sim = SimulationConfig {
            initial_balance: 100000.0,
            slippage_bps: 5,
            fee_bps: 10,
            fill_delay_ms: 100,
            enable_slippage: true,
        };

        let trade_value = Decimal::from(10000);
        let fee = sim.calculate_fee(trade_value);

        assert_eq!(fee, Decimal::from(10)); // 10 bps = 0.1% = 10 on 10000
    }
}
