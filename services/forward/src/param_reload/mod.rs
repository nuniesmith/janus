//! # Parameter Hot-Reload Module
//!
//! This module handles hot-reloading of optimized trading parameters from Redis.
//! It subscribes to the `fks:{instance}:param_updates` channel and applies
//! new parameters to strategies at runtime without service restart.
//!
//! ## Components
//!
//! - `ParamReloadManager` - Manages Redis subscription and param distribution
//! - `ParamApplier` - Trait for components that can receive param updates
//! - `IndicatorParamApplier` - Applies params to indicator analyzers
//! - `RiskParamApplier` - Applies params to risk management
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::param_reload::{ParamReloadManager, ParamReloadConfig};
//!
//! // Create manager
//! let config = ParamReloadConfig::from_env();
//! let manager = Arc::new(ParamReloadManager::new(config));
//!
//! // Register appliers
//! manager.register_applier(indicator_applier).await;
//! manager.register_applier(risk_applier).await;
//!
//! // Start background reload task
//! let handle = manager.clone().start_background_task().await?;
//! ```

pub mod appliers;
mod manager;
pub mod model_reload;

// Re-export main types
pub use appliers::{
    AssetRiskConfig, IndicatorParamApplier, LoggingApplier, NoOpApplier, OptimizedStrategyConfig,
    RiskParamApplier, StrategyParamApplier,
};
pub use manager::{ParamApplier, ParamReloadConfig, ParamReloadManager, ReloadStats};
pub use model_reload::{
    CallbackReloadHandler, LoggingReloadHandler, ModelReloadConfig, ModelReloadHandler,
    ModelReloadListener, ModelReloadStats,
};
