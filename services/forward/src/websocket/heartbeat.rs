//! # WebSocket Heartbeat Monitoring
//!
//! Monitors WebSocket client connections and removes stale connections.
//! Sends periodic ping messages and removes clients that don't respond.

use crate::websocket::{ClientManager, WebSocketMessage};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};

/// Heartbeat manager configuration
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval between heartbeat checks
    pub check_interval: Duration,

    /// Client timeout duration
    pub client_timeout: Duration,

    /// Send ping messages
    pub send_pings: bool,

    /// Ping interval
    pub ping_interval: Duration,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            client_timeout: Duration::from_secs(90),
            send_pings: true,
            ping_interval: Duration::from_secs(30),
        }
    }
}

/// Heartbeat manager
pub struct HeartbeatManager {
    client_manager: Arc<ClientManager>,
    config: HeartbeatConfig,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl HeartbeatManager {
    /// Create a new heartbeat manager
    pub fn new(client_manager: Arc<ClientManager>, config: HeartbeatConfig) -> Self {
        Self {
            client_manager,
            config,
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    /// Start heartbeat monitoring
    pub async fn start(&self) -> Result<(), HeartbeatError> {
        let mut running = self.running.write().await;
        if *running {
            return Err(HeartbeatError::AlreadyRunning);
        }
        *running = true;
        drop(running);

        info!(
            "Starting heartbeat manager (check_interval: {:?}, timeout: {:?})",
            self.config.check_interval, self.config.client_timeout
        );

        // Spawn heartbeat check task
        let client_manager = self.client_manager.clone();
        let config = self.config.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut check_timer = interval(config.check_interval);

            loop {
                check_timer.tick().await;

                // Check if we should stop
                {
                    let is_running = running.read().await;
                    if !*is_running {
                        info!("Heartbeat manager stopped");
                        break;
                    }
                }

                // Check for stale clients
                if let Err(e) = Self::check_stale_clients(&client_manager, &config).await {
                    warn!("Error checking stale clients: {}", e);
                }

                // Send pings if enabled
                if config.send_pings
                    && let Err(e) = Self::send_pings(&client_manager).await
                {
                    warn!("Error sending pings: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Stop heartbeat monitoring
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Stopping heartbeat manager");
    }

    /// Check if heartbeat manager is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Check for stale clients and remove them
    async fn check_stale_clients(
        client_manager: &ClientManager,
        config: &HeartbeatConfig,
    ) -> Result<(), HeartbeatError> {
        let stale_clients = client_manager
            .get_stale_clients(config.client_timeout)
            .await;

        if !stale_clients.is_empty() {
            debug!("Found {} stale client(s)", stale_clients.len());

            for client_id in stale_clients {
                info!("Removing stale client: {}", client_id);

                // Send goodbye message before removing
                let goodbye = WebSocketMessage::Goodbye(crate::websocket::GoodbyeMessage {
                    reason: "Connection timeout".to_string(),
                    reconnect: true,
                });

                let _ = client_manager.send_to_client(&client_id, goodbye).await;

                // Give a moment for the message to be sent
                sleep(Duration::from_millis(100)).await;

                // Remove the client
                client_manager.remove_client(&client_id).await;
            }
        }

        Ok(())
    }

    /// Send ping messages to all clients
    async fn send_pings(client_manager: &ClientManager) -> Result<(), HeartbeatError> {
        let ping = WebSocketMessage::Ping;
        let client_count = client_manager.client_count().await;

        if client_count > 0 {
            debug!("Sending ping to {} client(s)", client_count);
            client_manager.broadcast(ping).await;
        }

        Ok(())
    }

    /// Get heartbeat statistics
    pub async fn get_stats(&self) -> HeartbeatStats {
        HeartbeatStats {
            running: self.is_running().await,
            client_count: self.client_manager.client_count().await,
            check_interval: self.config.check_interval,
            client_timeout: self.config.client_timeout,
        }
    }
}

/// Heartbeat statistics
#[derive(Debug, Clone)]
pub struct HeartbeatStats {
    pub running: bool,
    pub client_count: usize,
    pub check_interval: Duration,
    pub client_timeout: Duration,
}

/// Heartbeat errors
#[derive(Debug, thiserror::Error)]
pub enum HeartbeatError {
    #[error("Heartbeat manager is already running")]
    AlreadyRunning,

    #[error("Client check error: {0}")]
    ClientCheckError(String),

    #[error("Ping send error: {0}")]
    PingSendError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_config_default() {
        let config = HeartbeatConfig::default();
        assert_eq!(config.check_interval, Duration::from_secs(30));
        assert_eq!(config.client_timeout, Duration::from_secs(90));
        assert!(config.send_pings);
    }

    #[tokio::test]
    async fn test_heartbeat_manager_creation() {
        let client_manager = Arc::new(ClientManager::new());
        let config = HeartbeatConfig::default();
        let manager = HeartbeatManager::new(client_manager, config);

        assert!(!manager.is_running().await);
    }

    #[tokio::test]
    async fn test_heartbeat_start_stop() {
        let client_manager = Arc::new(ClientManager::new());
        let config = HeartbeatConfig {
            check_interval: Duration::from_millis(100),
            client_timeout: Duration::from_secs(90),
            send_pings: false,
            ping_interval: Duration::from_secs(30),
        };
        let manager = HeartbeatManager::new(client_manager, config);

        manager.start().await.unwrap();
        assert!(manager.is_running().await);

        // Try to start again (should error)
        assert!(manager.start().await.is_err());

        manager.stop().await;
        sleep(Duration::from_millis(200)).await;
        assert!(!manager.is_running().await);
    }

    #[tokio::test]
    async fn test_heartbeat_stats() {
        let client_manager = Arc::new(ClientManager::new());
        let config = HeartbeatConfig::default();
        let manager = HeartbeatManager::new(client_manager, config.clone());

        let stats = manager.get_stats().await;
        assert!(!stats.running);
        assert_eq!(stats.client_count, 0);
        assert_eq!(stats.check_interval, config.check_interval);
    }
}
