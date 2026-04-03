//! Worker job type definitions for the Backward Service.
//!
//! These types define the jobs that the tokio-based worker pool can process.
//! Workers receive jobs through channels or polling and dispatch them to the
//! appropriate task handler in [`crate::tasks`].
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────┐
//! │  Job Scheduler   │
//! │  (periodic tick)  │
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │  Worker Pool     │──► handle_ingest(IngestJob)
//! │  (tokio tasks)   │──► handle_training(TrainJob)
//! └──────────────────┘
//! ```
//!
//! The worker pool is managed by [`crate::BackwardService`] using
//! `tokio::spawn` and coordinated via `tokio_util::sync::CancellationToken`
//! for graceful shutdown.

use serde::{Deserialize, Serialize};

/// Ingest job - reads experience data from shared memory and persists to storage.
///
/// The shared memory region contains Arrow IPC formatted record batches
/// written by the Forward service during live trading.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngestJob {
    /// Unique identifier for this ingest batch
    pub batch_id: String,
    /// Path to the shared memory region or Arrow IPC file
    pub shm_path: String,
}

/// Training job - samples experiences and updates model weights.
///
/// Coordinates with the Prioritized Experience Replay (PER) buffer to sample
/// batches, compute TD errors, and update model parameters.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainJob {
    /// Number of experiences to sample per training step
    pub batch_size: usize,
}

impl Default for TrainJob {
    fn default() -> Self {
        Self { batch_size: 32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_job_serialization() {
        let job = IngestJob {
            batch_id: "test-001".to_string(),
            shm_path: "/dev/shm/janus_batch_001.ipc".to_string(),
        };

        let json = serde_json::to_string(&job).unwrap();
        let deserialized: IngestJob = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.batch_id, "test-001");
        assert_eq!(deserialized.shm_path, "/dev/shm/janus_batch_001.ipc");
    }

    #[test]
    fn test_train_job_default() {
        let job = TrainJob::default();
        assert_eq!(job.batch_size, 32);
    }

    #[test]
    fn test_train_job_serialization() {
        let job = TrainJob { batch_size: 64 };

        let json = serde_json::to_string(&job).unwrap();
        let deserialized: TrainJob = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.batch_size, 64);
    }
}
