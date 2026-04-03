//! Shared memory buffer for zero-copy communication with Backward service.
//!
//! Uses Apache Arrow IPC format for efficient data transfer.
//!
//! # Status
//!
//! This module is a **placeholder** for a planned feature. The backward service
//! communication currently uses Redis streams and gRPC. When throughput demands
//! require zero-copy IPC, this module will be implemented using:
//!
//! - [`arrow::ipc::writer::FileWriter`] for serializing `RecordBatch` data
//! - Memory-mapped files (`memmap2`) for cross-process shared memory
//! - A ring-buffer protocol so the backward service can consume experiences
//!   without blocking the forward service's hot path
//!
//! Until then, calls to [`SharedMemoryBuffer::write`] and [`SharedMemoryBuffer::flush`]
//! are no-ops that log the intent for observability.

use anyhow::Result;
use common::Experience;
use tracing::info;

/// Shared memory buffer writer
///
/// Planned implementation will memory-map a file at `path` and write
/// Arrow IPC record batches into it for the backward service to consume.
pub struct SharedMemoryBuffer {
    path: String,
}

impl SharedMemoryBuffer {
    /// Create a new shared memory buffer targeting the given file path.
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    /// Write an experience to the buffer.
    ///
    /// Currently a no-op — will serialize `experience` into an Arrow
    /// RecordBatch and append it to the memory-mapped ring buffer.
    pub fn write(&mut self, _experience: Experience) -> Result<()> {
        info!("Writing experience to shared memory: {}", self.path);
        Ok(())
    }

    /// Flush any buffered data to the underlying shared memory region.
    ///
    /// Currently a no-op — will call `msync` on the memory-mapped file
    /// to ensure the backward service sees the latest writes.
    pub fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
