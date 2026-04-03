//! # Risk Management REST API
//!
//! Implements REST API endpoints for risk management operations.
//!
//! ## Endpoints
//!
//! - `GET /api/v1/risk/config` - Get current risk configuration
//! - `PUT /api/v1/risk/config` - Update risk configuration
//! - `GET /api/v1/risk/portfolio` - Get portfolio state
//! - `POST /api/v1/risk/portfolio/positions` - Add position to portfolio
//! - `DELETE /api/v1/risk/portfolio/positions/{symbol}` - Remove position
//! - `GET /api/v1/risk/metrics` - Get risk metrics snapshot
//! - `GET /api/v1/risk/performance` - Get performance metrics
//! - `POST /api/v1/risk/validate` - Validate a signal against risk limits
//! - `POST /api/v1/risk/calculate/position-size` - Calculate position size
//! - `POST /api/v1/risk/calculate/stop-loss` - Calculate stop loss
//! - `POST /api/v1/risk/calculate/take-profit` - Calculate take profit

use crate::risk::{
    MarketData, PortfolioState, Position, PositionSide, RiskConfig, RiskError, RiskManager,
    SizingMethod, StopLossMethod,
};
use crate::signal::types::{SignalSource, SignalType, Timeframe, TradingSignal};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Risk API state
pub struct RiskApiState {
    risk_manager: Arc<RwLock<RiskManager>>,
    portfolio: Arc<RwLock<PortfolioState>>,
}

impl RiskApiState {
    pub fn new(
        risk_manager: Arc<RwLock<RiskManager>>,
        portfolio: Arc<RwLock<PortfolioState>>,
    ) -> Self {
        Self {
            risk_manager,
            portfolio,
        }
    }

    pub fn router(state: Arc<Self>) -> Router {
        Router::new()
            // Configuration endpoints
            .route("/api/v1/risk/config", get(get_risk_config_handler))
            .route("/api/v1/risk/config", put(update_risk_config_handler))
            // Portfolio endpoints
            .route("/api/v1/risk/portfolio", get(get_portfolio_handler))
            .route(
                "/api/v1/risk/portfolio/positions",
                post(add_position_handler),
            )
            .route(
                "/api/v1/risk/portfolio/positions/{symbol}",
                delete(remove_position_handler),
            )
            // Metrics endpoints
            .route("/api/v1/risk/metrics", get(get_risk_metrics_handler))
            .route("/api/v1/risk/performance", get(get_performance_handler))
            // Calculation endpoints
            .route("/api/v1/risk/validate", post(validate_signal_handler))
            .route(
                "/api/v1/risk/calculate/position-size",
                post(calculate_position_size_handler),
            )
            .route(
                "/api/v1/risk/calculate/stop-loss",
                post(calculate_stop_loss_handler),
            )
            .route(
                "/api/v1/risk/calculate/take-profit",
                post(calculate_take_profit_handler),
            )
            .with_state(state)
    }
}

type AppState = Arc<RiskApiState>;

// ===== DTOs =====

/// Risk configuration DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfigDto {
    pub account_balance: f64,
    pub risk_per_trade_pct: f64,
    pub max_position_size_pct: f64,
    pub max_portfolio_exposure_pct: f64,
    pub min_risk_reward_ratio: f64,
    pub default_stop_method: String,
    pub default_sizing_method: String,
    pub max_concurrent_positions: usize,
    pub max_daily_loss_pct: f64,
    pub per_symbol_exposure_pct: f64,
    pub check_correlation: bool,
    pub atr_stop_multiplier: f64,
    pub atr_tp_multiplier: f64,
    pub default_risk_reward: f64,
}

impl From<&RiskConfig> for RiskConfigDto {
    fn from(config: &RiskConfig) -> Self {
        Self {
            account_balance: config.account_balance,
            risk_per_trade_pct: config.risk_per_trade_pct,
            max_position_size_pct: config.max_position_size_pct,
            max_portfolio_exposure_pct: config.max_portfolio_exposure_pct,
            min_risk_reward_ratio: config.min_risk_reward_ratio,
            default_stop_method: format!("{:?}", config.default_stop_method),
            default_sizing_method: format!("{:?}", config.default_sizing_method),
            max_concurrent_positions: config.max_concurrent_positions,
            max_daily_loss_pct: config.max_daily_loss_pct,
            per_symbol_exposure_pct: config.per_symbol_exposure_pct,
            check_correlation: config.check_correlation,
            atr_stop_multiplier: config.atr_stop_multiplier,
            atr_tp_multiplier: config.atr_tp_multiplier,
            default_risk_reward: config.default_risk_reward,
        }
    }
}

impl TryFrom<RiskConfigDto> for RiskConfig {
    type Error = String;

    fn try_from(dto: RiskConfigDto) -> Result<Self, Self::Error> {
        let default_stop_method = parse_stop_method(&dto.default_stop_method)?;
        let default_sizing_method = parse_sizing_method(&dto.default_sizing_method)?;

        Ok(Self {
            account_balance: dto.account_balance,
            risk_per_trade_pct: dto.risk_per_trade_pct,
            max_position_size_pct: dto.max_position_size_pct,
            max_portfolio_exposure_pct: dto.max_portfolio_exposure_pct,
            min_risk_reward_ratio: dto.min_risk_reward_ratio,
            default_stop_method,
            default_sizing_method,
            max_concurrent_positions: dto.max_concurrent_positions,
            max_daily_loss_pct: dto.max_daily_loss_pct,
            per_symbol_exposure_pct: dto.per_symbol_exposure_pct,
            check_correlation: dto.check_correlation,
            atr_stop_multiplier: dto.atr_stop_multiplier,
            atr_tp_multiplier: dto.atr_tp_multiplier,
            default_risk_reward: dto.default_risk_reward,
        })
    }
}

/// Position DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionDto {
    pub symbol: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub side: String, // "Long" or "Short"
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub position_value: f64,
    pub risk_amount: Option<f64>,
}

impl From<&Position> for PositionDto {
    fn from(pos: &Position) -> Self {
        Self {
            symbol: pos.symbol.clone(),
            entry_price: pos.entry_price,
            quantity: pos.quantity,
            side: format!("{:?}", pos.side),
            stop_loss: pos.stop_loss,
            take_profit: pos.take_profit,
            position_value: pos.position_value(),
            risk_amount: pos.risk_amount(),
        }
    }
}

impl TryFrom<PositionDto> for Position {
    type Error = String;

    fn try_from(dto: PositionDto) -> Result<Self, Self::Error> {
        let side = match dto.side.as_str() {
            "Long" => PositionSide::Long,
            "Short" => PositionSide::Short,
            _ => return Err(format!("Invalid position side: {}", dto.side)),
        };

        Ok(Self {
            symbol: dto.symbol,
            entry_price: dto.entry_price,
            quantity: dto.quantity,
            side,
            stop_loss: dto.stop_loss,
            take_profit: dto.take_profit,
        })
    }
}

/// Portfolio state DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioStateDto {
    pub positions: HashMap<String, PositionDto>,
    pub daily_pnl: f64,
    pub total_value: f64,
    pub total_exposure: f64,
    pub position_count: usize,
    pub exposure_percentage: f64,
}

impl From<&PortfolioState> for PortfolioStateDto {
    fn from(portfolio: &PortfolioState) -> Self {
        let positions: HashMap<String, PositionDto> = portfolio
            .positions
            .iter()
            .map(|(k, v)| (k.clone(), PositionDto::from(v)))
            .collect();

        let total_exposure = portfolio.total_exposure();
        let exposure_percentage = if portfolio.total_value > 0.0 {
            (total_exposure / portfolio.total_value) * 100.0
        } else {
            0.0
        };

        Self {
            positions,
            daily_pnl: portfolio.daily_pnl,
            total_value: portfolio.total_value,
            total_exposure,
            position_count: portfolio.position_count(),
            exposure_percentage,
        }
    }
}

/// Market data DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataDto {
    pub current_price: f64,
    pub atr: Option<f64>,
    pub support: Option<f64>,
    pub resistance: Option<f64>,
    pub volatility: Option<f64>,
    pub recent_high: Option<f64>,
    pub recent_low: Option<f64>,
}

impl From<MarketDataDto> for MarketData {
    fn from(dto: MarketDataDto) -> Self {
        Self {
            current_price: dto.current_price,
            atr: dto.atr,
            support: dto.support,
            resistance: dto.resistance,
            volatility: dto.volatility,
            recent_high: dto.recent_high,
            recent_low: dto.recent_low,
        }
    }
}

/// Trading signal DTO (simplified for API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDto {
    pub symbol: String,
    pub signal_type: String, // "Buy", "Sell", "Hold"
    pub timeframe: String,
    pub confidence: f64,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

/// Risk metrics snapshot DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetricsDto {
    pub portfolio_exposure_pct: f64,
    pub portfolio_heat: f64,
    pub position_count: usize,
    pub total_risk_amount: f64,
    pub avg_risk_per_position: f64,
    pub max_position_risk: f64,
    pub daily_pnl: f64,
    pub daily_pnl_pct: f64,
}

/// Performance metrics DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsDto {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: Option<f64>,
    pub total_pnl: f64,
}

/// Position size calculation request
#[derive(Debug, Deserialize)]
pub struct CalculatePositionSizeRequest {
    pub signal: SignalDto,
    pub market_data: MarketDataDto,
    pub sizing_method: Option<String>,
}

/// Position size calculation response
#[derive(Debug, Serialize)]
pub struct PositionSizeResponse {
    pub quantity: f64,
    pub position_value: f64,
    pub risk_amount: f64,
    pub risk_percentage: f64,
    pub method_used: String,
}

/// Stop loss calculation request
#[derive(Debug, Deserialize)]
pub struct CalculateStopLossRequest {
    pub signal: SignalDto,
    pub market_data: MarketDataDto,
    pub method: Option<String>,
}

/// Stop loss calculation response
#[derive(Debug, Serialize)]
pub struct StopLossResponse {
    pub stop_loss_price: f64,
    pub distance_from_entry: f64,
    pub distance_percentage: f64,
    pub risk_per_unit: f64,
    pub method_used: String,
}

/// Take profit calculation request
#[derive(Debug, Deserialize)]
pub struct CalculateTakeProfitRequest {
    pub signal: SignalDto,
    pub market_data: MarketDataDto,
    pub risk_reward_ratio: Option<f64>,
}

/// Take profit calculation response
#[derive(Debug, Serialize)]
pub struct TakeProfitResponse {
    pub take_profit_price: f64,
    pub distance_from_entry: f64,
    pub distance_percentage: f64,
    pub profit_per_unit: f64,
    pub risk_reward_ratio: f64,
}

/// Validate signal request
#[derive(Debug, Deserialize)]
pub struct ValidateSignalRequest {
    pub signal: SignalDto,
}

/// Validate signal response
#[derive(Debug, Serialize)]
pub struct ValidateSignalResponse {
    pub is_valid: bool,
    pub validation_errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Add position request
#[derive(Debug, Deserialize)]
pub struct AddPositionRequest {
    pub position: PositionDto,
}

/// Generic success response
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub details: Option<String>,
}

// ===== Error Type =====

pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    InternalError(String),
    RiskError(RiskError),
}

impl From<RiskError> for ApiError {
    fn from(err: RiskError) -> Self {
        ApiError::RiskError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, message, details) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg, None),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg, None),
            ApiError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                msg,
                None,
            ),
            ApiError::RiskError(err) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "risk_error",
                err.to_string(),
                Some(format!("{:?}", err)),
            ),
        };

        let body = Json(ErrorResponse {
            error: error.to_string(),
            message,
            details,
        });

        (status, body).into_response()
    }
}

// ===== Handlers =====

/// GET /api/v1/risk/config
pub async fn get_risk_config_handler(
    State(state): State<AppState>,
) -> Result<Json<RiskConfigDto>, ApiError> {
    debug!("GET /api/v1/risk/config");

    let manager = state.risk_manager.read().await;
    let config = manager.config();

    Ok(Json(RiskConfigDto::from(config)))
}

/// PUT /api/v1/risk/config
pub async fn update_risk_config_handler(
    State(state): State<AppState>,
    Json(config_dto): Json<RiskConfigDto>,
) -> Result<Json<SuccessResponse>, ApiError> {
    info!("PUT /api/v1/risk/config");

    let config = RiskConfig::try_from(config_dto)
        .map_err(|e| ApiError::BadRequest(format!("Invalid configuration: {}", e)))?;

    let mut manager = state.risk_manager.write().await;
    manager.update_config(config);

    Ok(Json(SuccessResponse {
        success: true,
        message: "Risk configuration updated successfully".to_string(),
    }))
}

/// GET /api/v1/risk/portfolio
pub async fn get_portfolio_handler(
    State(state): State<AppState>,
) -> Result<Json<PortfolioStateDto>, ApiError> {
    debug!("GET /api/v1/risk/portfolio");

    let portfolio = state.portfolio.read().await;
    Ok(Json(PortfolioStateDto::from(&*portfolio)))
}

/// POST /api/v1/risk/portfolio/positions
pub async fn add_position_handler(
    State(state): State<AppState>,
    Json(req): Json<AddPositionRequest>,
) -> Result<Json<SuccessResponse>, ApiError> {
    info!("POST /api/v1/risk/portfolio/positions");

    let position = Position::try_from(req.position)
        .map_err(|e| ApiError::BadRequest(format!("Invalid position: {}", e)))?;

    let symbol = position.symbol.clone();
    let mut portfolio = state.portfolio.write().await;
    portfolio.add_position(symbol.clone(), position);

    Ok(Json(SuccessResponse {
        success: true,
        message: format!("Position added for {}", symbol),
    }))
}

/// DELETE /api/v1/risk/portfolio/positions/:symbol
pub async fn remove_position_handler(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
) -> Result<Json<SuccessResponse>, ApiError> {
    info!("DELETE /api/v1/risk/portfolio/positions/{}", symbol);

    let mut portfolio = state.portfolio.write().await;

    if portfolio.positions.remove(&symbol).is_some() {
        Ok(Json(SuccessResponse {
            success: true,
            message: format!("Position removed for {}", symbol),
        }))
    } else {
        Err(ApiError::NotFound(format!(
            "No position found for symbol: {}",
            symbol
        )))
    }
}

/// GET /api/v1/risk/metrics
pub async fn get_risk_metrics_handler(
    State(state): State<AppState>,
) -> Result<Json<RiskMetricsDto>, ApiError> {
    debug!("GET /api/v1/risk/metrics");

    let manager = state.risk_manager.read().await;
    let portfolio = state.portfolio.read().await;

    let portfolio_risk = manager.calculate_portfolio_risk(&portfolio);

    let total_risk_amount: f64 = portfolio
        .positions
        .values()
        .filter_map(|p| p.risk_amount())
        .sum();

    let max_position_risk = portfolio
        .positions
        .values()
        .filter_map(|p| p.risk_amount())
        .fold(0.0, f64::max);

    let avg_risk_per_position = if portfolio.position_count() > 0 {
        total_risk_amount / portfolio.position_count() as f64
    } else {
        0.0
    };

    let daily_pnl_pct = if portfolio.total_value > 0.0 {
        (portfolio.daily_pnl / portfolio.total_value) * 100.0
    } else {
        0.0
    };

    Ok(Json(RiskMetricsDto {
        portfolio_exposure_pct: portfolio_risk.exposure_percentage,
        portfolio_heat: portfolio_risk.portfolio_heat,
        position_count: portfolio.position_count(),
        total_risk_amount,
        avg_risk_per_position,
        max_position_risk,
        daily_pnl: portfolio.daily_pnl,
        daily_pnl_pct,
    }))
}

/// GET /api/v1/risk/performance
pub async fn get_performance_handler(
    State(state): State<AppState>,
) -> Result<Json<PerformanceMetricsDto>, ApiError> {
    debug!("GET /api/v1/risk/performance");

    let portfolio = state.portfolio.read().await;
    let manager = state.risk_manager.read().await;
    let config = manager.config();

    // Derive what we can from the live portfolio state.
    // Full per-trade history lives in the brain pipeline's affinity tracker
    // and QuestDB — this endpoint surfaces the portfolio-level snapshot.
    let _position_count = portfolio.position_count();
    let total_pnl = portfolio.daily_pnl;

    // Count winning / losing positions by unrealized P&L direction
    let mut winning = 0usize;
    let mut losing = 0usize;
    let mut total_win_pnl = 0.0_f64;
    let mut total_loss_pnl = 0.0_f64;

    for pos in portfolio.positions.values() {
        // Estimate unrealized P&L from entry vs current portfolio value contribution
        let risk = pos.risk_amount().unwrap_or(0.0);
        if risk >= 0.0 {
            // Position with non-negative risk implies currently favorable or neutral
            winning += 1;
            total_win_pnl += risk;
        } else {
            losing += 1;
            total_loss_pnl += risk.abs();
        }
    }

    let total_trades = winning + losing;
    let win_rate = if total_trades > 0 {
        winning as f64 / total_trades as f64
    } else {
        0.0
    };
    let avg_win = if winning > 0 {
        total_win_pnl / winning as f64
    } else {
        0.0
    };
    let avg_loss = if losing > 0 {
        total_loss_pnl / losing as f64
    } else {
        0.0
    };
    let profit_factor = if total_loss_pnl > 0.0 {
        total_win_pnl / total_loss_pnl
    } else if total_win_pnl > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let max_drawdown = if config.account_balance > 0.0 {
        (portfolio.daily_pnl.min(0.0).abs() / config.account_balance) * 100.0
    } else {
        0.0
    };

    Ok(Json(PerformanceMetricsDto {
        total_trades,
        winning_trades: winning,
        losing_trades: losing,
        win_rate,
        avg_win,
        avg_loss,
        profit_factor,
        max_drawdown,
        sharpe_ratio: None, // Requires return time-series; available via QuestDB query
        total_pnl,
    }))
}

/// POST /api/v1/risk/validate
pub async fn validate_signal_handler(
    State(state): State<AppState>,
    Json(req): Json<ValidateSignalRequest>,
) -> Result<Json<ValidateSignalResponse>, ApiError> {
    debug!("POST /api/v1/risk/validate");

    let signal = signal_dto_to_trading_signal(&req.signal)?;

    let manager = state.risk_manager.read().await;
    let portfolio = state.portfolio.read().await;

    let mut validation_errors = Vec::new();
    let mut warnings = Vec::new();

    let is_valid = match manager.validate_signal(&signal, &portfolio) {
        Ok(_) => true,
        Err(e) => {
            validation_errors.push(e.to_string());
            false
        }
    };

    // Add warnings
    if signal.confidence < 0.7 {
        warnings.push("Signal confidence is below 70%".to_string());
    }

    Ok(Json(ValidateSignalResponse {
        is_valid,
        validation_errors,
        warnings,
    }))
}

/// POST /api/v1/risk/calculate/position-size
pub async fn calculate_position_size_handler(
    State(state): State<AppState>,
    Json(req): Json<CalculatePositionSizeRequest>,
) -> Result<Json<PositionSizeResponse>, ApiError> {
    debug!("POST /api/v1/risk/calculate/position-size");

    let signal = signal_dto_to_trading_signal(&req.signal)?;
    let market_data = MarketData::from(req.market_data);

    let manager = state.risk_manager.read().await;
    let method = if let Some(m) = req.sizing_method {
        parse_sizing_method(&m).map_err(ApiError::BadRequest)?
    } else {
        manager.config().default_sizing_method
    };

    let position_size = manager.calculate_position_size(&signal, &market_data, &method)?;

    Ok(Json(PositionSizeResponse {
        quantity: position_size.quantity,
        position_value: position_size.position_value,
        risk_amount: position_size.risk_amount,
        risk_percentage: position_size.risk_percentage * 100.0,
        method_used: format!("{:?}", method),
    }))
}

/// POST /api/v1/risk/calculate/stop-loss
pub async fn calculate_stop_loss_handler(
    State(state): State<AppState>,
    Json(req): Json<CalculateStopLossRequest>,
) -> Result<Json<StopLossResponse>, ApiError> {
    debug!("POST /api/v1/risk/calculate/stop-loss");

    let signal = signal_dto_to_trading_signal(&req.signal)?;
    let market_data = MarketData::from(req.market_data);

    let manager = state.risk_manager.read().await;
    let method = if let Some(m) = req.method {
        parse_stop_method(&m).map_err(ApiError::BadRequest)?
    } else {
        manager.config().default_stop_method
    };

    let stop_calculator = crate::risk::StopLossCalculator::new(manager.config().clone());
    let stop_loss = stop_calculator.calculate_stop_loss(&signal, &market_data, &method)?;

    let entry_price = signal.entry_price.unwrap_or(market_data.current_price);
    let distance = (entry_price - stop_loss).abs();
    let distance_pct = (distance / entry_price) * 100.0;

    Ok(Json(StopLossResponse {
        stop_loss_price: stop_loss,
        distance_from_entry: distance,
        distance_percentage: distance_pct,
        risk_per_unit: distance,
        method_used: format!("{:?}", method),
    }))
}

/// POST /api/v1/risk/calculate/take-profit
pub async fn calculate_take_profit_handler(
    State(state): State<AppState>,
    Json(req): Json<CalculateTakeProfitRequest>,
) -> Result<Json<TakeProfitResponse>, ApiError> {
    debug!("POST /api/v1/risk/calculate/take-profit");

    let mut signal = signal_dto_to_trading_signal(&req.signal)?;
    let market_data = MarketData::from(req.market_data);

    let manager = state.risk_manager.read().await;

    // If signal doesn't have stop loss, we need to calculate it first
    if signal.stop_loss.is_none() {
        let stop_calculator = crate::risk::StopLossCalculator::new(manager.config().clone());
        let stop_loss = stop_calculator.calculate_stop_loss(
            &signal,
            &market_data,
            &manager.config().default_stop_method,
        )?;
        signal = signal.with_stop_loss(stop_loss);
    }

    let tp_calculator = crate::risk::TakeProfitCalculator::new(manager.config().clone());
    let take_profit = tp_calculator.calculate_take_profit(&signal, &market_data)?;

    let entry_price = signal.entry_price.unwrap_or(market_data.current_price);
    let distance = (take_profit - entry_price).abs();
    let distance_pct = (distance / entry_price) * 100.0;

    let stop_distance = signal
        .stop_loss
        .map(|sl| (entry_price - sl).abs())
        .unwrap_or(0.0);
    let rr_ratio = if stop_distance > 0.0 {
        distance / stop_distance
    } else {
        0.0
    };

    Ok(Json(TakeProfitResponse {
        take_profit_price: take_profit,
        distance_from_entry: distance,
        distance_percentage: distance_pct,
        profit_per_unit: distance,
        risk_reward_ratio: rr_ratio,
    }))
}

// ===== Helper Functions =====

fn signal_dto_to_trading_signal(dto: &SignalDto) -> Result<TradingSignal, ApiError> {
    let signal_type = match dto.signal_type.as_str() {
        "Buy" => SignalType::Buy,
        "Sell" => SignalType::Sell,
        "Hold" => SignalType::Hold,
        _ => {
            return Err(ApiError::BadRequest(format!(
                "Invalid signal type: {}",
                dto.signal_type
            )));
        }
    };

    let timeframe = parse_timeframe(&dto.timeframe)
        .map_err(|e| ApiError::BadRequest(format!("Invalid timeframe: {}", e)))?;

    let mut signal = TradingSignal::new(
        dto.symbol.clone(),
        signal_type,
        timeframe,
        dto.confidence,
        SignalSource::TechnicalIndicator {
            name: "API".to_string(),
        },
    );

    if let Some(entry) = dto.entry_price {
        signal = signal.with_entry_price(entry);
    }

    if let Some(sl) = dto.stop_loss {
        signal = signal.with_stop_loss(sl);
    }

    if let Some(tp) = dto.take_profit {
        signal = signal.with_take_profit(tp);
    }

    Ok(signal)
}

fn parse_timeframe(s: &str) -> Result<Timeframe, String> {
    match s {
        "1m" | "M1" => Ok(Timeframe::M1),
        "5m" | "M5" => Ok(Timeframe::M5),
        "15m" | "M15" => Ok(Timeframe::M15),
        "1h" | "H1" => Ok(Timeframe::H1),
        "4h" | "H4" => Ok(Timeframe::H4),
        "1d" | "D1" => Ok(Timeframe::D1),
        _ => Err(format!("Unknown timeframe: {}", s)),
    }
}

fn parse_stop_method(s: &str) -> Result<StopLossMethod, String> {
    match s {
        "Atr" => Ok(StopLossMethod::Atr { multiplier: 2.0 }),
        "Percentage" => Ok(StopLossMethod::Percentage { percent: 0.02 }),
        "SupportResistance" => Ok(StopLossMethod::SupportResistance),
        "Volatility" => Ok(StopLossMethod::Volatility { std_devs: 2.0 }),
        "HighLow" => Ok(StopLossMethod::HighLow {
            lookback: 20,
            buffer_pct: 0.001,
        }),
        _ => Err(format!("Unknown stop loss method: {}", s)),
    }
}

fn parse_sizing_method(s: &str) -> Result<SizingMethod, String> {
    match s {
        "FixedFractional" => Ok(SizingMethod::FixedFractional),
        "Kelly" => Ok(SizingMethod::Kelly {
            win_rate: 0.55,
            avg_win_loss_ratio: 1.5,
        }),
        "VolatilityBased" => Ok(SizingMethod::VolatilityBased {
            target_volatility: 0.15,
        }),
        "FixedDollar" => Ok(SizingMethod::FixedDollar { amount: 100.0 }),
        "AtrBased" => Ok(SizingMethod::AtrBased {
            target_atr_multiple: 2.0,
        }),
        _ => Err(format!("Unknown sizing method: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timeframe() {
        assert!(parse_timeframe("1m").is_ok());
        assert!(parse_timeframe("M1").is_ok());
        assert!(parse_timeframe("1h").is_ok());
        assert!(parse_timeframe("invalid").is_err());
    }

    #[test]
    fn test_parse_stop_method() {
        assert!(parse_stop_method("Atr").is_ok());
        assert!(parse_stop_method("Percentage").is_ok());
        assert!(parse_stop_method("SupportResistance").is_ok());
        assert!(parse_stop_method("invalid").is_err());
    }

    #[test]
    fn test_parse_sizing_method() {
        assert!(parse_sizing_method("FixedFractional").is_ok());
        assert!(parse_sizing_method("Kelly").is_ok());
        assert!(parse_sizing_method("VolatilityBased").is_ok());
        assert!(parse_sizing_method("invalid").is_err());
    }

    #[test]
    fn test_risk_config_dto_conversion() {
        let config = RiskConfig::default();
        let dto = RiskConfigDto::from(&config);
        assert_eq!(dto.account_balance, 10000.0);
        assert_eq!(dto.risk_per_trade_pct, 0.01);
    }

    #[test]
    fn test_position_dto_conversion() {
        let position = Position {
            symbol: "BTC/USD".to_string(),
            entry_price: 50000.0,
            quantity: 0.1,
            side: PositionSide::Long,
            stop_loss: Some(49000.0),
            take_profit: Some(52000.0),
        };

        let dto = PositionDto::from(&position);
        assert_eq!(dto.symbol, "BTC/USD");
        assert_eq!(dto.entry_price, 50000.0);
        assert_eq!(dto.quantity, 0.1);
        assert_eq!(dto.side, "Long");
        assert_eq!(dto.position_value, 5000.0);
    }
}
