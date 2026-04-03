//! State Component
//!
//! Part of Integration region

pub mod event_dispatcher;
pub mod global_state;
pub mod message_bus;
pub mod state_sync;

// Re-exports
pub use event_dispatcher::EventDispatcher;
pub use global_state::GlobalState;
pub use message_bus::MessageBus;
pub use state_sync::StateSync;
