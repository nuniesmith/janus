//! Message Bus — Inter-region communication
//!
//! This module re-exports the full [`MessageBus`] implementation from
//! [`super::state::message_bus`], which provides topic-based pub/sub
//! messaging with configurable channels, buffer management, EMA
//! smoothing, windowed statistics, and overflow handling.
//!
//! # Overview
//!
//! The `MessageBus` is the primary communication backbone for the
//! neuromorphic integration layer. Brain regions publish messages to
//! named topics and subscribe via channels with topic-prefix filters.
//!
//! # Example
//!
//! ```rust,no_run
//! use janus_neuromorphic::integration::message_bus::{MessageBus, MessageBusConfig};
//!
//! let mut bus = MessageBus::new();
//! bus.create_channel("risk_alerts", "risk.");
//! bus.create_channel("market_events", "market.");
//!
//! // Publish a risk alert — routed to the "risk_alerts" channel
//! bus.emit("risk.circuit_breaker", "amygdala", 0.95);
//!
//! // Drain messages from a channel
//! let alerts = bus.drain("risk_alerts");
//! ```

pub use super::state::message_bus::*;
