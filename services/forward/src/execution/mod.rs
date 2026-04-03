//! Execution Service Integration
//!
//! This module provides integration with the FKS Execution Service,
//! allowing the forward service to submit trading signals for execution.

pub mod brain_gated;
pub mod client;

pub use brain_gated::{
    BrainGatedConfig, BrainGatedExecutionClient, GatedExecutionStats, GatedSubmissionResult,
};
pub use client::{ExecutionClient, ExecutionClientConfig, SubmitSignalResponse};
