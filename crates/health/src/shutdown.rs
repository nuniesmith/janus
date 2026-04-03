//! # Graceful Shutdown Module
//!
//! Handles graceful shutdown of the trading system with safety guarantees.
//!
//! ## Features
//!
//! - Signal handling (SIGINT, SIGTERM)
//! - Dead man's switch (auto-close positions on crash)
//! - Resource cleanup (flush buffers, close connections)
//! - Timeout enforcement
//! - State persistence
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_health::shutdown::{ShutdownCoordinator, ShutdownHook};
//!
//! let coordinator = ShutdownCoordinator::new();
//!
//! // Register cleanup hooks
//! coordinator.register_hook("questdb", Box::new(move || {
//!     Box::pin(async move {
//!         writer.flush().await?;
//!         Ok(())
//!     })
//! }));
//!
//! // Listen for shutdown signals
//! tokio::spawn(async move {
//!     coordinator.wait_for_signal().await;
//! });
//!
//! // Trigger shutdown
//! coordinator.shutdown().await?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;
use tracing::{error, info, warn};

/// Type alias for async shutdown hooks
pub type ShutdownFuture = Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>;

/// Shutdown hook function type
pub type ShutdownHook = Box<dyn FnOnce() -> ShutdownFuture + Send>;

/// Shutdown coordinator
pub struct ShutdownCoordinator {
    inner: Arc<ShutdownInner>,
}

struct ShutdownInner {
    hooks: Mutex<Vec<(String, ShutdownHook)>>,
    shutdown_requested: RwLock<bool>,
    timeout_duration: Duration,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

    /// Create a shutdown coordinator with custom timeout
    pub fn with_timeout(timeout_duration: Duration) -> Self {
        Self {
            inner: Arc::new(ShutdownInner {
                hooks: Mutex::new(Vec::new()),
                shutdown_requested: RwLock::new(false),
                timeout_duration,
            }),
        }
    }

    /// Register a shutdown hook
    ///
    /// Hooks are executed in reverse order of registration (LIFO)
    pub async fn register_hook(&self, name: impl Into<String>, hook: ShutdownHook) {
        let mut hooks = self.inner.hooks.lock().await;
        hooks.push((name.into(), hook));
    }

    /// Check if shutdown has been requested
    pub async fn is_shutdown_requested(&self) -> bool {
        *self.inner.shutdown_requested.read().await
    }

    /// Wait for shutdown signal (SIGINT or SIGTERM)
    pub async fn wait_for_signal(&self) {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("🛑 Received SIGINT (Ctrl+C), initiating graceful shutdown");
            }
            _ = terminate => {
                info!("🛑 Received SIGTERM, initiating graceful shutdown");
            }
        }

        // Mark shutdown as requested
        *self.inner.shutdown_requested.write().await = true;
    }

    /// Trigger graceful shutdown
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        info!("🔄 Starting graceful shutdown sequence");

        // Mark shutdown as requested
        *self.inner.shutdown_requested.write().await = true;

        // Execute all hooks in reverse order
        let mut hooks = self.inner.hooks.lock().await;
        let hook_count = hooks.len();

        info!("📋 Executing {} shutdown hooks", hook_count);

        while let Some((name, hook)) = hooks.pop() {
            info!("⏳ Running shutdown hook: {}", name);

            match timeout(self.inner.timeout_duration, hook()).await {
                Ok(Ok(())) => {
                    info!("✅ Shutdown hook completed: {}", name);
                }
                Ok(Err(e)) => {
                    error!("❌ Shutdown hook failed: {} - {}", name, e);
                }
                Err(_) => {
                    warn!("⏱️ Shutdown hook timed out: {}", name);
                }
            }
        }

        info!("✅ Graceful shutdown complete");
        Ok(())
    }

    /// Get a shutdown signal future
    pub fn get_shutdown_signal(&self) -> ShutdownSignal {
        ShutdownSignal {
            coordinator: self.clone(),
        }
    }
}

impl Clone for ShutdownCoordinator {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Shutdown signal future
pub struct ShutdownSignal {
    coordinator: ShutdownCoordinator,
}

impl ShutdownSignal {
    /// Wait for shutdown signal
    pub async fn wait(&self) {
        loop {
            if self.coordinator.is_shutdown_requested().await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Dead man's switch - emergency position closer
pub struct DeadMansSwitch {
    enabled: bool,
    emergency_close_fn: Option<Box<dyn Fn() -> ShutdownFuture + Send + Sync>>,
}

impl DeadMansSwitch {
    /// Create a new dead man's switch
    pub fn new() -> Self {
        Self {
            enabled: true,
            emergency_close_fn: None,
        }
    }

    /// Set the emergency close function
    pub fn set_emergency_close_fn<F>(&mut self, f: F)
    where
        F: Fn() -> ShutdownFuture + Send + Sync + 'static,
    {
        self.emergency_close_fn = Some(Box::new(f));
    }

    /// Enable the dead man's switch
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the dead man's switch
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Trigger emergency close (called on panic or unexpected exit)
    pub async fn trigger(&self) -> anyhow::Result<()> {
        if !self.enabled {
            info!("⚠️ Dead man's switch disabled, skipping emergency close");
            return Ok(());
        }

        error!("🚨 DEAD MAN'S SWITCH TRIGGERED - Emergency position close");

        if let Some(ref close_fn) = self.emergency_close_fn {
            match timeout(Duration::from_secs(10), close_fn()).await {
                Ok(Ok(())) => {
                    info!("✅ Emergency position close completed");
                    Ok(())
                }
                Ok(Err(e)) => {
                    error!("❌ Emergency position close failed: {}", e);
                    Err(e)
                }
                Err(_) => {
                    error!("⏱️ Emergency position close timed out");
                    Err(anyhow::anyhow!("Emergency close timeout"))
                }
            }
        } else {
            warn!("⚠️ No emergency close function registered");
            Ok(())
        }
    }
}

impl Default for DeadMansSwitch {
    fn default() -> Self {
        Self::new()
    }
}

/// Install panic hook with dead man's switch
pub fn install_panic_hook(switch: Arc<Mutex<DeadMansSwitch>>) {
    let default_panic = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        error!("💥 PANIC DETECTED: {}", panic_info);

        // Trigger dead man's switch
        let switch_clone = Arc::clone(&switch);
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                if let Ok(switch) = switch_clone.try_lock() {
                    let _ = switch.trigger().await;
                }
            });
        });

        // Call the default panic hook
        default_panic(panic_info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn test_shutdown_coordinator_creation() {
        let coordinator = ShutdownCoordinator::new();
        assert!(!coordinator.is_shutdown_requested().await);
    }

    #[tokio::test]
    async fn test_register_and_execute_hooks() {
        let coordinator = ShutdownCoordinator::new();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        coordinator
            .register_hook(
                "test",
                Box::new(move || {
                    let executed = executed_clone.clone();
                    Box::pin(async move {
                        executed.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            )
            .await;

        coordinator.shutdown().await.unwrap();
        assert!(executed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_hook_execution_order() {
        let coordinator = ShutdownCoordinator::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let order1 = order.clone();
        coordinator
            .register_hook(
                "first",
                Box::new(move || {
                    let order = order1.clone();
                    Box::pin(async move {
                        order.lock().await.push(1);
                        Ok(())
                    })
                }),
            )
            .await;

        let order2 = order.clone();
        coordinator
            .register_hook(
                "second",
                Box::new(move || {
                    let order = order2.clone();
                    Box::pin(async move {
                        order.lock().await.push(2);
                        Ok(())
                    })
                }),
            )
            .await;

        coordinator.shutdown().await.unwrap();

        let execution_order = order.lock().await;
        // Hooks should execute in reverse order (LIFO)
        assert_eq!(*execution_order, vec![2, 1]);
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        let coordinator = ShutdownCoordinator::new();
        let signal = coordinator.get_shutdown_signal();

        let coordinator_clone = coordinator.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let mut shutdown = coordinator_clone.inner.shutdown_requested.write().await;
            *shutdown = true;
        });

        // Should wait until shutdown is requested
        signal.wait().await;
        assert!(coordinator.is_shutdown_requested().await);
    }

    #[tokio::test]
    async fn test_dead_mans_switch() {
        let mut switch = DeadMansSwitch::new();
        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        switch.set_emergency_close_fn(move || {
            let triggered = triggered_clone.clone();
            Box::pin(async move {
                triggered.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        switch.trigger().await.unwrap();
        assert!(triggered.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_dead_mans_switch_disabled() {
        let mut switch = DeadMansSwitch::new();
        switch.disable();

        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        switch.set_emergency_close_fn(move || {
            let triggered = triggered_clone.clone();
            Box::pin(async move {
                triggered.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        switch.trigger().await.unwrap();
        // Should not trigger when disabled
        assert!(!triggered.load(Ordering::SeqCst));
    }
}
