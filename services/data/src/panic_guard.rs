//! Panic isolation for single-consumer actor loops.
//!
//! Actors like [`Router`](crate::actors::Router) and
//! [`IndicatorActor`](crate::actors::IndicatorActor) own a single-consumer
//! `mpsc::UnboundedReceiver`, so their processing loop cannot simply be
//! re-spawned by a supervisor after a panic — the receiver would be lost and
//! producers would wedge. Instead we keep the loop alive by isolating the
//! per-message work: a panic in one unit of work (a bad prometheus label, a
//! math edge case in indicator computation, …) is caught and logged, and the
//! loop continues with the next message rather than dying permanently and
//! silently.
//!
//! This mirrors the supervisor's "restart on death" intent at message
//! granularity for tasks that cannot be re-spawned wholesale.

use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;

/// Drive `fut` to completion, catching any panic so the caller's loop stays
/// alive.
///
/// Returns `Ok(T)` with the future's output on normal completion, or
/// `Err(())` if the future panicked (already logged with `task` context).
///
/// The future is wrapped in [`AssertUnwindSafe`]: state shared across the
/// panic boundary (behind `Arc`/async locks that do not poison) is
/// deliberately treated as recoverable, because dropping the whole ingestion
/// loop on a single bad message is strictly worse than continuing.
pub async fn catch_panic<F, T>(task: &str, fut: F) -> Result<T, ()>
where
    F: Future<Output = T>,
{
    match AssertUnwindSafe(fut).catch_unwind().await {
        Ok(value) => Ok(value),
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            tracing::error!(
                task,
                panic = %detail,
                "task panicked; isolating the failure and keeping the loop alive"
            );
            Err(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_value_on_success() {
        let out = catch_panic("ok", async { 21 * 2 }).await;
        assert_eq!(out, Ok(42));
    }

    #[tokio::test]
    async fn catches_panic_and_returns_err() {
        let out: Result<(), ()> = catch_panic("boom", async {
            panic!("simulated per-message death");
        })
        .await;
        assert_eq!(out, Err(()));
    }

    #[tokio::test]
    async fn loop_survives_repeated_panics() {
        // Prove a loop wrapped with catch_panic keeps running after a task
        // that would otherwise unwind and kill it permanently.
        let mut processed = 0u32;
        let mut survived_panics = 0u32;
        for i in 0..5 {
            let res: Result<(), ()> = catch_panic("loop", async move {
                if i % 2 == 0 {
                    panic!("die on even #{i}");
                }
            })
            .await;
            match res {
                Ok(()) => processed += 1,
                Err(()) => survived_panics += 1,
            }
        }
        assert_eq!(processed, 2); // i = 1, 3
        assert_eq!(survived_panics, 3); // i = 0, 2, 4 — loop never died
    }
}
