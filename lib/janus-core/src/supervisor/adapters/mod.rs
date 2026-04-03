//! Service adapters for integrating existing Janus modules with the
//! [`JanusSupervisor`](super::JanusSupervisor).
//!
//! The existing Janus services (Forward, Backward, CNS, Data) expose a
//! `start_module(state: Arc<JanusState>) -> Result<()>` entry point that
//! internally polls `state.is_shutdown_requested()` for lifecycle control.
//!
//! The supervisor, however, manages services through the [`JanusService`]
//! trait which passes a [`CancellationToken`] for shutdown signalling.
//!
//! This module provides [`ModuleAdapter`] — a bridge that:
//!
//! 1. Wraps any `start_module`-style async function into a [`JanusService`]
//! 2. Bridges the supervisor's `CancellationToken` → `JanusState::request_shutdown()`
//! 3. Propagates errors back to the supervisor for restart decisions
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_core::supervisor::adapters::ModuleAdapter;
//! use janus_core::supervisor::RestartPolicy;
//!
//! let data_adapter = ModuleAdapter::new(
//!     "data",
//!     state.clone(),
//!     |s| Box::pin(janus_data::start_module(s)),
//!     RestartPolicy::OnFailure,
//! );
//!
//! supervisor.spawn_service(Box::new(data_adapter));
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::service::{JanusService, RestartPolicy};
use crate::state::JanusState;

// ---------------------------------------------------------------------------
// ModuleStartFn — type alias for the start_module function signature
// ---------------------------------------------------------------------------

/// A boxed async function that takes `Arc<JanusState>` and returns a `Result`.
///
/// This matches the signature of every existing Janus service's
/// `start_module()` function.
pub type ModuleStartFn = Box<
    dyn Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// ModuleAdapter
// ---------------------------------------------------------------------------

/// Adapts an existing `start_module(Arc<JanusState>) -> Result<()>` function
/// into a [`JanusService`] that the supervisor can manage.
///
/// ## Shutdown Bridging
///
/// When the supervisor cancels this service's [`CancellationToken`], the
/// adapter calls `state.request_shutdown()` to signal the inner module
/// through its existing shutdown pathway. This avoids having to rewrite
/// every service to accept a `CancellationToken` directly — they continue
/// polling `state.is_shutdown_requested()` as before.
///
/// ### Shared-State Shutdown Invariant
///
/// Because `JanusState` is shared across all modules, `request_shutdown()`
/// sets a **global** `AtomicBool`.  This is safe under the current
/// supervisor model because:
///
/// 1. The supervisor propagates cancellation to **all** services at once
///    (child tokens of the same root), so if one bridge fires they all do.
/// 2. When a service fails on its own (returns `Err`), the bridge task is
///    aborted *before* it can call `request_shutdown()`, keeping the flag
///    `false` for the restart.
/// 3. The supervisor checks `cancel.is_cancelled()` before each restart
///    attempt, so a stale `true` flag never leads to a restart loop.
///
/// If per-module independent restart semantics are ever needed (e.g.,
/// restarting one module while others keep running), this must be
/// refactored to use a per-adapter shutdown signal instead of the global
/// `JanusState` flag.
///
/// ## Error Propagation
///
/// Errors returned by the inner `start_module` are converted to
/// `anyhow::Error` and bubbled up to the supervisor, which then applies
/// the configured [`RestartPolicy`].
pub struct ModuleAdapter {
    /// Human-readable name for logging and metrics.
    name: String,

    /// Shared application state passed to the module on each (re)start.
    state: Arc<JanusState>,

    /// The `start_module`-style function to call.
    start_fn: ModuleStartFn,

    /// Restart policy for this module.
    policy: RestartPolicy,
}

impl ModuleAdapter {
    /// Create a new adapter.
    ///
    /// # Arguments
    ///
    /// * `name` — Service name for the supervisor (e.g., `"forward"`,
    ///   `"data"`, `"cns"`).
    /// * `state` — Shared `JanusState` that the module uses for
    ///   configuration, health reporting, and shutdown signalling.
    /// * `start_fn` — A closure or function pointer matching the
    ///   `start_module` signature. Must be `Send + Sync + 'static`.
    /// * `policy` — How the supervisor should handle failures.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let adapter = ModuleAdapter::new(
    ///     "forward",
    ///     state.clone(),
    ///     |s| Box::pin(janus_forward::start_module(s)),
    ///     RestartPolicy::OnFailure,
    /// );
    /// ```
    pub fn new<F>(name: &str, state: Arc<JanusState>, start_fn: F, policy: RestartPolicy) -> Self
    where
        F: Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.to_string(),
            state,
            start_fn: Box::new(start_fn),
            policy,
        }
    }

    /// Create an adapter with the default [`RestartPolicy::OnFailure`].
    pub fn on_failure<F>(name: &str, state: Arc<JanusState>, start_fn: F) -> Self
    where
        F: Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self::new(name, state, start_fn, RestartPolicy::OnFailure)
    }

    /// Create an adapter that never restarts (one-shot module).
    pub fn one_shot<F>(name: &str, state: Arc<JanusState>, start_fn: F) -> Self
    where
        F: Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self::new(name, state, start_fn, RestartPolicy::Never)
    }

    /// Create an adapter that always restarts (even on clean exit).
    pub fn always_restart<F>(name: &str, state: Arc<JanusState>, start_fn: F) -> Self
    where
        F: Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self::new(name, state, start_fn, RestartPolicy::Always)
    }
}

#[async_trait]
impl JanusService for ModuleAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn restart_policy(&self) -> RestartPolicy {
        self.policy
    }

    #[tracing::instrument(skip(self, cancel), fields(module = %self.name, policy = %self.policy))]
    async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
        // Defensive check: if `shutdown_requested` is already `true` on
        // the shared JanusState, the module will likely exit immediately.
        // This should never happen during a normal restart cycle (the
        // supervisor checks `cancel.is_cancelled()` first, and the bridge
        // task is aborted on natural module exit), but we log a warning
        // so the condition is diagnosable if the invariant is ever broken.
        if self.state.is_shutdown_requested() {
            tracing::warn!(
                module = %self.name,
                "JanusState.shutdown_requested is already true at module start — \
                 the module may exit immediately. This indicates a stale shutdown \
                 flag from a previous lifecycle or a concurrent shutdown in progress."
            );
        }

        // Spawn a bridge task that watches the CancellationToken and
        // translates it into a JanusState shutdown request.
        let bridge_state = self.state.clone();
        let bridge_cancel = cancel.clone();
        let bridge_handle = tokio::spawn(async move {
            bridge_cancel.cancelled().await;
            tracing::info!(
                "ModuleAdapter shutdown bridge: cancellation received, requesting state shutdown"
            );
            bridge_state.request_shutdown();
        });

        // Register health before starting
        self.state
            .register_module_health(&self.name, true, Some("starting".to_string()))
            .await;

        // Run the module
        let state_clone = self.state.clone();
        let module_result = (self.start_fn)(state_clone).await;

        // Abort the bridge task if the module exited on its own
        // (e.g., due to an internal error, not a cancellation).
        bridge_handle.abort();

        // Update health based on result
        match &module_result {
            Ok(()) => {
                self.state
                    .register_module_health(&self.name, true, Some("stopped".to_string()))
                    .await;
            }
            Err(e) => {
                self.state
                    .register_module_health(&self.name, false, Some(format!("error: {e}")))
                    .await;
            }
        }

        module_result.map_err(|e| anyhow::anyhow!("{}: {}", self.name, e))
    }
}

// We can't derive Debug because ModuleStartFn contains a trait object,
// but we can implement it manually for diagnostics.
impl std::fmt::Debug for ModuleAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleAdapter")
            .field("name", &self.name)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// ApiModuleAdapter — specialised for the always-on API module
// ---------------------------------------------------------------------------

/// A convenience wrapper for the API module which is always started
/// immediately (it doesn't wait for `start_services()`) and should
/// always be restarted on failure.
///
/// This is functionally identical to `ModuleAdapter::always_restart()`
/// but has a dedicated type for clarity in the supervisor setup code.
pub struct ApiModuleAdapter {
    inner: ModuleAdapter,
}

impl ApiModuleAdapter {
    /// Create a new API module adapter.
    pub fn new<F>(state: Arc<JanusState>, start_fn: F) -> Self
    where
        F: Fn(Arc<JanusState>) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            inner: ModuleAdapter::new("api", state, start_fn, RestartPolicy::Always),
        }
    }
}

#[async_trait]
impl JanusService for ApiModuleAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn restart_policy(&self) -> RestartPolicy {
        RestartPolicy::Always
    }

    async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
        self.inner.run(cancel).await
    }
}

impl std::fmt::Debug for ApiModuleAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiModuleAdapter")
            .field("inner", &self.inner)
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Helper: create a JanusState for testing.
    async fn test_state() -> Arc<JanusState> {
        let config = crate::Config::default();
        Arc::new(JanusState::new(config).await.unwrap())
    }

    #[tokio::test]
    async fn test_module_adapter_clean_exit() {
        let state = test_state().await;

        let ran = Arc::new(AtomicU64::new(0));
        let ran_clone = ran.clone();

        let adapter = ModuleAdapter::on_failure("test-clean", state.clone(), move |_s| {
            let ran = ran_clone.clone();
            Box::pin(async move {
                ran.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        let cancel = CancellationToken::new();
        let svc: Box<dyn JanusService> = Box::new(adapter);
        let result = svc.run(cancel).await;

        assert!(result.is_ok());
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_module_adapter_error_propagation() {
        let state = test_state().await;

        let adapter = ModuleAdapter::on_failure("test-fail", state.clone(), |_s| {
            Box::pin(async move { Err(crate::Error::Config("boom".into())) })
        });

        let cancel = CancellationToken::new();
        let svc: Box<dyn JanusService> = Box::new(adapter);
        let result = svc.run(cancel).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("test-fail"),
            "error should contain service name: {err_msg}"
        );
        assert!(
            err_msg.contains("boom"),
            "error should contain cause: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_module_adapter_cancellation_bridge() {
        let state = test_state().await;

        let adapter = ModuleAdapter::on_failure("test-cancel", state.clone(), |s| {
            Box::pin(async move {
                // Simulate a module that polls shutdown
                loop {
                    if s.is_shutdown_requested() {
                        return Ok(());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
        });

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        // Cancel after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let svc: Box<dyn JanusService> = Box::new(adapter);
        let result = svc.run(cancel).await;

        assert!(result.is_ok());
        assert!(state.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_module_adapter_health_registration() {
        let state = test_state().await;

        let adapter = ModuleAdapter::on_failure("health-test", state.clone(), |_s| {
            Box::pin(async move { Ok(()) })
        });

        let cancel = CancellationToken::new();
        let svc: Box<dyn JanusService> = Box::new(adapter);
        svc.run(cancel).await.unwrap();

        let health = state.get_module_health().await;
        let entry = health.iter().find(|h| h.name == "health-test");
        assert!(entry.is_some(), "module should have registered health");
        assert!(entry.unwrap().healthy);
    }

    #[tokio::test]
    async fn test_module_adapter_health_on_error() {
        let state = test_state().await;

        let adapter = ModuleAdapter::on_failure("err-health", state.clone(), |_s| {
            Box::pin(async move { Err(crate::Error::Config("kaboom".into())) })
        });

        let cancel = CancellationToken::new();
        let svc: Box<dyn JanusService> = Box::new(adapter);
        let _ = svc.run(cancel).await;

        let health = state.get_module_health().await;
        let entry = health.iter().find(|h| h.name == "err-health");
        assert!(entry.is_some());
        assert!(!entry.unwrap().healthy);
        assert!(
            entry
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("kaboom")
        );
    }

    #[test]
    fn test_module_adapter_debug() {
        // Just ensure Debug doesn't panic
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = test_state().await;
            let adapter =
                ModuleAdapter::on_failure("dbg", state, |_s| Box::pin(async move { Ok(()) }));
            let dbg = format!("{:?}", adapter);
            assert!(dbg.contains("ModuleAdapter"));
            assert!(dbg.contains("dbg"));
        });
    }

    #[test]
    fn test_restart_policies() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = test_state().await;

            let a1 = ModuleAdapter::on_failure("a", state.clone(), |_| Box::pin(async { Ok(()) }));
            assert_eq!(a1.restart_policy(), RestartPolicy::OnFailure);

            let a2 = ModuleAdapter::one_shot("b", state.clone(), |_| Box::pin(async { Ok(()) }));
            assert_eq!(a2.restart_policy(), RestartPolicy::Never);

            let a3 =
                ModuleAdapter::always_restart("c", state.clone(), |_| Box::pin(async { Ok(()) }));
            assert_eq!(a3.restart_policy(), RestartPolicy::Always);
        });
    }

    #[tokio::test]
    async fn test_api_module_adapter() {
        let state = test_state().await;
        let ran = Arc::new(AtomicU64::new(0));
        let ran_clone = ran.clone();

        let adapter = ApiModuleAdapter::new(state.clone(), move |_s| {
            let ran = ran_clone.clone();
            Box::pin(async move {
                ran.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });

        assert_eq!(adapter.name(), "api");
        assert_eq!(adapter.restart_policy(), RestartPolicy::Always);

        let cancel = CancellationToken::new();
        let svc: Box<dyn JanusService> = Box::new(adapter);
        svc.run(cancel).await.unwrap();

        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }
}
