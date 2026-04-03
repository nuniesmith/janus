//! JanusService trait — the contract every supervised service must implement.
//!
//! This trait is the cornerstone of the Janus Supervisor Model. Every service
//! managed by the [`JanusSupervisor`](super::JanusSupervisor) must implement
//! this trait so the supervisor can:
//!
//! - Start the service in a tracked task
//! - Monitor its lifecycle (detect unexpected exits)
//! - Propagate shutdown signals via [`CancellationToken`]
//! - Restart the service according to the configured backoff strategy
//!
//! # Design Decisions
//!
//! - **`async_trait`**: We use `async_trait` to allow dynamic dispatch
//!   (`Box<dyn JanusService>`). The one-time heap allocation for the boxed
//!   future is negligible for long-running service loops that start once.
//!
//! - **`CancellationToken`**: Passed explicitly into `run()` so the service
//!   *must* listen for cancellation. This is superior to relying on `Drop`
//!   semantics, which are unpredictable in async contexts.
//!
//! - **`anyhow::Result`**: Allows services to return diverse error types that
//!   the supervisor can uniformly log and use to decide on restart strategies.
//!
//! - **`Send + Sync + 'static`**: Required because services are spawned onto
//!   the Tokio runtime via `TaskTracker::spawn`.
//!
//! - **`&self` (not `&mut self`) on `run()`**: The trait takes `&self` so
//!   that services can be wrapped in `Arc`, shared across boundaries, or
//!   composed without requiring exclusive access. Services that need
//!   mutable state across restarts should use interior mutability
//!   (`AtomicU64`, `Mutex`, etc.), which is already required by the
//!   `Send + Sync` bounds anyway.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

/// Restart policy for a service managed by the supervisor.
///
/// Controls how the supervisor responds when a service exits unexpectedly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RestartPolicy {
    /// Always restart the service on failure (transient or permanent).
    /// The supervisor will apply exponential backoff between attempts.
    Always,

    /// Only restart on error returns. If the service exits with `Ok(())`,
    /// it is considered complete and will not be restarted.
    #[default]
    OnFailure,

    /// Never restart. The service runs once and the supervisor only logs
    /// its exit status. Useful for one-shot initialization tasks.
    Never,
}

impl std::fmt::Display for RestartPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => write!(f, "always"),
            Self::OnFailure => write!(f, "on_failure"),
            Self::Never => write!(f, "never"),
        }
    }
}

/// The core trait every supervised service must implement.
///
/// # Example
///
/// ```rust,ignore
/// use janus_core::supervisor::{JanusService, RestartPolicy};
/// use tokio_util::sync::CancellationToken;
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// pub struct MarketDataFeed {
///     exchange: String,
///     polls: AtomicU64,
/// }
///
/// #[async_trait::async_trait]
/// impl JanusService for MarketDataFeed {
///     fn name(&self) -> &str {
///         "market-data-feed"
///     }
///
///     fn restart_policy(&self) -> RestartPolicy {
///         RestartPolicy::OnFailure
///     }
///
///     async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
///         loop {
///             tokio::select! {
///                 _ = cancel.cancelled() => {
///                     tracing::info!("Market data feed shutting down");
///                     break;
///                 }
///                 _ = self.poll_exchange() => {
///                     self.polls.fetch_add(1, Ordering::Relaxed);
///                 }
///             }
///         }
///         Ok(())
///     }
/// }
///
/// impl MarketDataFeed {
///     async fn poll_exchange(&self) {
///         tokio::time::sleep(std::time::Duration::from_millis(100)).await;
///     }
/// }
/// ```
#[async_trait]
pub trait JanusService: Send + Sync + 'static {
    /// Returns the unique name of the service for logging, metrics, and
    /// supervisor identification.
    ///
    /// This name is used in:
    /// - `tracing` spans (`service = name`)
    /// - Prometheus metric labels (`janus_supervisor_restarts_total{service="..."}`)
    /// - The supervisor's internal service registry
    fn name(&self) -> &str;

    /// The restart policy for this service.
    ///
    /// Defaults to [`RestartPolicy::OnFailure`], meaning the supervisor will
    /// restart the service if `run()` returns an `Err`, but will treat a
    /// clean `Ok(())` exit as intentional completion.
    fn restart_policy(&self) -> RestartPolicy {
        RestartPolicy::OnFailure
    }

    /// The main execution loop of the service.
    ///
    /// This method should run until either:
    /// 1. The service completes its work naturally (returns `Ok(())`)
    /// 2. The `cancel` token is cancelled (graceful shutdown)
    /// 3. An unrecoverable error occurs (returns `Err(...)`)
    ///
    /// # Cancellation Contract
    ///
    /// Implementations **must** select on `cancel.cancelled()` in their main
    /// loop. Failure to do so will cause the service to hang during shutdown,
    /// ultimately hitting the supervisor's shutdown timeout.
    ///
    /// ```rust,ignore
    /// async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
    ///     loop {
    ///         tokio::select! {
    ///             _ = cancel.cancelled() => break,
    ///             result = self.do_work() => {
    ///                 result?;
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Interior Mutability
    ///
    /// Because `run()` takes `&self`, services that need to track mutable
    /// state (e.g., restart counters, connection handles) should use
    /// interior mutability primitives like [`AtomicU64`](std::sync::atomic::AtomicU64),
    /// [`Mutex`](tokio::sync::Mutex), or [`RwLock`](tokio::sync::RwLock).
    /// This is consistent with the `Send + Sync` requirements and allows
    /// services to be wrapped in `Arc` or composed without requiring
    /// exclusive ownership.
    ///
    /// # Errors
    ///
    /// Returning an error signals the supervisor that the service has failed.
    /// The supervisor will then apply the service's [`RestartPolicy`] and
    /// backoff strategy to decide whether and when to restart.
    async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A minimal test service to verify trait implementation compiles
    /// and the default restart policy is correct.
    ///
    /// Uses [`AtomicU32`] for the run counter since `run()` takes `&self`.
    struct DummyService {
        name: String,
        run_count: AtomicU32,
    }

    impl DummyService {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                run_count: AtomicU32::new(0),
            }
        }

        fn run_count(&self) -> u32 {
            self.run_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl JanusService for DummyService {
        fn name(&self) -> &str {
            &self.name
        }

        async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
            self.run_count.fetch_add(1, Ordering::SeqCst);
            cancel.cancelled().await;
            Ok(())
        }
    }

    #[test]
    fn test_default_restart_policy() {
        let svc = DummyService::new("test");
        assert_eq!(svc.restart_policy(), RestartPolicy::OnFailure);
    }

    #[test]
    fn test_restart_policy_display() {
        assert_eq!(RestartPolicy::Always.to_string(), "always");
        assert_eq!(RestartPolicy::OnFailure.to_string(), "on_failure");
        assert_eq!(RestartPolicy::Never.to_string(), "never");
    }

    #[tokio::test]
    async fn test_dummy_service_runs_and_cancels() {
        let svc = DummyService::new("test-svc");
        let token = CancellationToken::new();

        let token_clone = token.clone();
        let handle = tokio::spawn(async move {
            // Cancel after a short delay
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            token_clone.cancel();
        });

        let result = svc.run(token).await;
        assert!(result.is_ok());
        assert_eq!(svc.run_count(), 1);

        handle.await.unwrap();
    }

    #[test]
    fn test_service_name() {
        let svc = DummyService::new("market-data");
        assert_eq!(svc.name(), "market-data");
    }
}
