//! Lightweight task supervision for raw `tokio::spawn` critical loops.
//!
//! The full [`JanusSupervisor`](super::JanusSupervisor) supervises *modules*
//! whose `start_module` future returns on failure. Many critical tasks,
//! however, are launched as bare `tokio::spawn(...)` handles that are only
//! ever `.abort()`ed at shutdown — a panic or an unexpected early return
//! inside them ends the task **permanently and silently**, with the parent
//! still reporting healthy.
//!
//! [`supervise`] closes that gap without changing a task's happy-path
//! behavior. It runs the task inside an isolated child spawn and, if the
//! child panics or returns before shutdown was requested, re-spawns it with
//! capped exponential backoff. The task factory is re-invoked on every
//! restart, so any per-run setup (a fresh broadcast subscription, a new
//! connection, …) is re-established cleanly — mirroring the resilience model
//! the live candle sink already uses.
//!
//! The circuit breaker is intentionally disabled: these are always-on,
//! money-adjacent ingestion/execution loops for which giving up permanently
//! is never the desired outcome. Backoff caps the restart rate (default 60s)
//! so a hot-looping panic cannot spin the CPU.

use std::future::Future;

use super::backoff::{BackoffAction, BackoffConfig, BackoffState};

/// Supervise a critical task until `is_shutdown` returns `true`.
///
/// `factory` is called to produce the task future for each run; it is
/// re-invoked on every restart so the task re-subscribes / reconnects fresh.
/// The future is driven inside an isolated [`tokio::spawn`], so a panic is
/// caught (as a [`JoinError`](tokio::task::JoinError)) rather than unwinding
/// the supervisor.
///
/// Restart triggers:
/// - the task **panics**, or
/// - the task **returns** (any value) while shutdown has *not* been requested.
///
/// The loop exits (stops restarting) when:
/// - `is_shutdown()` is `true` before a start or after a task exit, or
/// - the child task was **aborted / cancelled** (the shutdown abort path).
///
/// Backoff uses [`BackoffConfig::default`] with the circuit breaker disabled,
/// so restarts continue indefinitely with a capped delay.
pub async fn supervise<F, Fut>(name: &str, is_shutdown: impl Fn() -> bool, mut factory: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    supervise_with_backoff(
        name,
        is_shutdown,
        BackoffConfig::default().without_circuit_breaker(),
        &mut factory,
    )
    .await
}

/// [`supervise`] with an explicit backoff configuration (used by tests and
/// callers that want a tighter restart cadence).
pub async fn supervise_with_backoff<F, Fut>(
    name: &str,
    is_shutdown: impl Fn() -> bool,
    config: BackoffConfig,
    factory: &mut F,
) where
    F: FnMut() -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let mut backoff = BackoffState::new(config);

    loop {
        if is_shutdown() {
            break;
        }

        backoff.record_start();
        let handle = tokio::spawn(factory());

        match handle.await {
            Ok(()) => {
                if is_shutdown() {
                    break;
                }
                tracing::warn!(
                    task = name,
                    "supervised task returned unexpectedly before shutdown; restarting"
                );
            }
            Err(join_err) if join_err.is_panic() => {
                if is_shutdown() {
                    break;
                }
                tracing::error!(
                    task = name,
                    error = %join_err,
                    "supervised task PANICKED; restarting"
                );
            }
            Err(_cancelled) => {
                // The child was aborted — this only happens on the shutdown
                // path, so stop supervising.
                break;
            }
        }

        backoff.maybe_reset_on_cooldown();
        match backoff.next_backoff() {
            BackoffAction::Retry(delay) => tokio::time::sleep(delay).await,
            // Circuit breaker is disabled for critical tasks; this arm is
            // unreachable in practice but degrades safely to the max delay
            // rather than giving up.
            BackoffAction::CircuitOpen { .. } => {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await
            }
        }
    }

    tracing::info!(task = name, "supervised task stopped (shutdown)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    fn fast_backoff() -> BackoffConfig {
        BackoffConfig::new(Duration::from_millis(1), Duration::from_millis(5))
            .without_circuit_breaker()
    }

    #[tokio::test]
    async fn restarts_after_panic_until_shutdown() {
        let runs = Arc::new(AtomicU32::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let runs_f = runs.clone();
        let shutdown_f = shutdown.clone();
        let mut factory = move || {
            let runs = runs_f.clone();
            let shutdown = shutdown_f.clone();
            async move {
                let n = runs.fetch_add(1, Ordering::SeqCst) + 1;
                if n >= 4 {
                    // Enough restarts observed — request shutdown so the
                    // supervisor stops after this run returns.
                    shutdown.store(true, Ordering::SeqCst);
                    return;
                }
                // Simulate a task death by panicking.
                panic!("simulated task death #{n}");
            }
        };

        let shutdown_check = shutdown.clone();
        supervise_with_backoff(
            "test-panic",
            move || shutdown_check.load(Ordering::SeqCst),
            fast_backoff(),
            &mut factory,
        )
        .await;

        // Proves the task was restarted after each panic rather than dying
        // permanently on the first one.
        assert_eq!(runs.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn restarts_after_early_return() {
        let runs = Arc::new(AtomicU32::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let runs_f = runs.clone();
        let shutdown_f = shutdown.clone();
        let mut factory = move || {
            let runs = runs_f.clone();
            let shutdown = shutdown_f.clone();
            async move {
                let n = runs.fetch_add(1, Ordering::SeqCst) + 1;
                if n >= 3 {
                    shutdown.store(true, Ordering::SeqCst);
                }
                // Returns immediately every time (the "gave up and returned"
                // failure mode) — supervisor must re-spawn it.
            }
        };

        let shutdown_check = shutdown.clone();
        supervise_with_backoff(
            "test-early-return",
            move || shutdown_check.load(Ordering::SeqCst),
            fast_backoff(),
            &mut factory,
        )
        .await;

        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_start_when_already_shutdown() {
        let runs = Arc::new(AtomicU32::new(0));
        let runs_f = runs.clone();
        let mut factory = move || {
            let runs = runs_f.clone();
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
            }
        };

        supervise_with_backoff("test-noop", || true, fast_backoff(), &mut factory).await;
        assert_eq!(runs.load(Ordering::SeqCst), 0);
    }
}
