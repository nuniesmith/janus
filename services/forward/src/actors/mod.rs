//! Actor definitions for the Forward service.
//!
//! Uses the Actor Model pattern with tokio tasks and mpsc channels.

pub mod market;
pub mod strategy;

pub use strategy::StrategyActor;
