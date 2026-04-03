//! # Persistence Module
//!
//! Provides durable persistence for brain-inspired trading system state,
//! ensuring critical data (strategy affinity records, correlation state,
//! kill switch coordination) survives service restarts.
//!
//! ## Backends
//!
//! - **Redis** — Low-latency persistence via `AffinityRedisStore`
//! - **Redis Kill Switch** — Multi-instance kill switch coordination
//!   via `RedisKillSwitch`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::persistence::{AffinityRedisStore, AffinityRedisConfig};
//!
//! // Connect with environment-based config
//! let store = AffinityRedisStore::from_env().await?;
//!
//! // Save pipeline state
//! save_pipeline_affinity(&pipeline, &store).await?;
//!
//! // Load on next boot
//! let restored = load_pipeline_affinity(&pipeline, &store, 10).await?;
//! ```
//!
//! ### Multi-Instance Kill Switch
//!
//! ```rust,ignore
//! use janus_forward::persistence::kill_switch_redis::{
//!     RedisKillSwitch, RedisKillSwitchConfig,
//! };
//!
//! let ks = RedisKillSwitch::new(RedisKillSwitchConfig::from_env()).await?;
//! ks.activate("operator", "emergency halt").await?;
//! ```

pub mod affinity_redis;
pub mod kill_switch_redis;

pub use affinity_redis::{
    AffinityRedisConfig, AffinityRedisStore, load_pipeline_affinity, save_pipeline_affinity,
    spawn_affinity_autosave,
};

pub use kill_switch_redis::{
    KillSwitchEvent, KillSwitchState, RedisKillSwitch, RedisKillSwitchConfig,
    wire_and_spawn_redis_kill_switch, wire_redis_kill_switch,
};
